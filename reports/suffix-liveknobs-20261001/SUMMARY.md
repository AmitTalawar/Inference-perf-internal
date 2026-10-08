# Large-session CPU sim

## What `14×32K` is

**32K is each session’s starting prompt: 32,768 tokens.** It is not the GPU cache and not the 16K peel (`T` / `S`).

`14×32K` means **14 concurrent sessions**, each beginning at **32,768 tokens**, then **6 turns** that add **512 user + 128 output tokens** per turn. Last-turn working set is about **505,344 tokens** against a **318,816-token** cache (19927 blocks × 16, null block excluded) — **~1.6×**. That is the published synthetic overload case the CPU sim reuses.

`eq32k-s24-*` / `eq32k-s32-*` keep the same 32,768-token start and 6-turn recipe; only the session count changes (working set ~866K ≈ 2.7× cache, and ~1.16M ≈ 3.6×).

## Workload: purely synthetic, not Weka

The simulator does **not** replay `weka traces`. Specs come from `suffix_eviction_workload.py`: random token IDs with **fixed lengths**. Only lengths and return order matter for eviction; the actual token values are dummy.

| | this CPU sim | live Weka 7/21 |
|---|---|---|
| source | generated spec (`cyclic` / `random` / `pareto` / `abandon`) | real weka races (1261 unique sessions; we take 21) |
| session length | every session starts at exactly 32,768 tokens | highly variable; median ~68 turns in the full file |
| turns | exactly 6 | whatever the trace recorded |
| per-turn add | +512 user, +128 output, every turn | real prompt/output sizes |
| think time | **cyclic**: all 30s, staggered starts (oldest idle returns next). **random**: exponential around 30s (return order ≠ idle age). **pareto**: heavy tail, capped at 5× mean (long-idle tends to stay idle). **abandon**: same as cyclic, but a subset gets 600s thinks (they never come back in-window) | real inter-arrival / think / abandon from weka traces |
| prefix sharing | none beyond the growing per-session history | real conversation prefixes |
| concurrency | all N sessions in flight | 7 concurrent of 21 |

So the sim answers “if N equal 32K chats return in this *order*, who wins?” It does **not** claim the same mix as Weka. Session-awareness only shows here when we **force** abandon. Live Weka is where real think times and real drop-off can matter.

## What `cyclic` / `random` / `pareto` / `abandon` mean

A table heading like `eq32k-s24-abandon / abandon` is **spec file / pattern**:

- `eq32k` — every session starts at an **equal** 32,768 tokens
- `s24` — **24** sessions
- `abandon` — the think-time recipe below (repeated after the slash from the spec’s `pattern` field)

These are **who comes back next**, not token-content distributions. Prompt/output sizes are the same in every pattern. Mean think time is 30s unless noted.

| pattern | how think times are drawn | who returns next | what it tests |
|---|---|---|---|
| **cyclic** | Every gap is exactly 30s. Starts are staggered (`session_id × 30 / N`). | The session that went idle **longest ago** (round-robin). | Worst case for “wipe the oldest”: that session is the next arrival. |
| **random** | Exponential gaps, mean 30s (`expovariate`). Random start in `[0, 30]`. | Unrelated to idle age. | Typical mixed arrivals; LRU’s “oldest” is not the next return. |
| **pareto** | Pareto(α=1.5) gaps, mean 30s, **capped at 150s** (5× mean). | A session idle a long time **tends to stay idle** (heavy tail). | Oldest KV is often stale; prefix-pruning / idle-TTL can look good. |
| **abandon** | Same as cyclic for the live sessions, but a **subset never comes back**: their `think_s` is **600s** (far past the 6-turn window). | Live sessions still cycle every 30s; abandoned ones sit in cache until evicted. | Only pattern where **session-aware wipe** should beat prefix-pruning. |

How many are abandoned in the specs we used:

- `eq32k-s14-abandon` — **4 of 14** get 600s thinks
- `eq32k-s24-abandon` — **8 of 24** get 600s thinks

`abandon` is hand-built (not a `--pattern` in `suffix_eviction_workload.py`, which only emits cyclic/random/pareto).

### Why session-aware only wins on abandon

Prefix-pruning and session-aware do the **same 16K front peels** while a session still looks alive. Session-aware’s extra move is: if idle time says the user is **gone**, delete that session’s **remaining unique KV in one wipe**.

On cyclic / random / pareto **every session comes back** (all 6 turns). Think times stay around 30s (Pareto capped at 150s). That is below the 180s dead floor, so the detector never fires. Session-aware ≡ prefix-pruning. Same recomputed counts.

Abandon **fakes walk-aways**: after a turn, 4 (or 8) sessions set the next think to **600s**. The other sessions finish the whole 6-turn run in a few minutes. Those 4/8 never send another request, but their ~32K KV is still sitting in the prefix cache.

- **Prefix-pruning** still treats them as idle-but-alive: peel 16K from the front, **keep the suffix**. That leftover suffix is useless — the user will not return — and it occupies space the live 30s-cycle sessions need.
- **Session-aware** after ~180s idle: “this session is dead” → wipe the rest. Live sessions then hit more of their own suffix. That is the 0.626× → 0.609× gap on `eq32k-s14-abandon`.

“Subset sitting in cache” = KV held for users who will not come back. There is no magic on cyclic/random/pareto because **nobody sits**; everyone returns, so a full wipe would hurt the next turn.

## Should we add even more sessions?

Already ran 24 and 32 (same 32K × 6-turn recipe, T=4096 sweep in `../suffix-session-large-20261001/`):

- **24 × 32K (~2.7× cache)** — prefix-pruning ≈ native (0.996–0.997×). Cyclic is the wipe regime: overflow is larger than one peel can cover, so almost every return is a full miss. Session-aware only helps on **abandon** (0.903× vs native, vs pruning 0.921×).
- **32 × 32K (~3.6× cache)** — prefix-pruning ≈ native again; Pareto was **1.006×** (slightly worse than native). No session win without abandon.

More equal always-returning 32K sessions (48, 64, …) will stay in that wipe regime. It will not rank the arms differently. A useful extra sim is **Weka-shaped**: many sessions, **variable** lengths, **real-ish abandon**, only ~7 in flight — not “more identical 32K chats.”

## What the colon numbers mean

Policy strings after the arm name are colon-separated knobs. The **3rd field is not the same thing on every arm**.

### `vllm-prefix-pruning:T:S:K:idle` → `16384:16384:16:60`

| field | env | this run | meaning |
|---|---|---:|---|
| **T** | `VLLM_SUFFIX_EVICTION_TIER_TOKENS` | 16384 | First peel: drop this many tokens from the **front** of each finished request before touching anyone else's suffix. |
| **S** | `VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS` | 16384 | If more KV is still needed, grow that evicted prefix by S tokens on the next pass (usually S = T). |
| **K** | `VLLM_SUFFIX_EVICTION_MAX_TIERS` | 16 | Cap how many peels a request can take. After `K × S` tokens the leftover suffix is ordinary LRU. `16 × 16384 = 262144` covers a 256K window. `0` = uncapped. |
| **idle** | `VLLM_SUFFIX_EVICTION_IDLE_TTL_S` | 60 | Seconds a finished request may sit unused before its remaining blocks are demoted to tier 0 (evict first). |

So `16384:16384:16:60` means: peel 16K from the prefix, grow in 16K steps, at most 16 peels, and if a session has been idle 60s treat it as reclaimable.

### `vllm-prefix-pruning-on-sessions:T:S:dead:k:gaps` → `16384:16384:180:3:2`

Same **T / S** ladder, then:

| field | env | this run | meaning |
|---|---|---:|---|
| **dead** | `VLLM_SUFFIX_SESSION_DEAD_TTL_S` | 180 | Floor: if a session has been idle this many seconds, wipe its remaining unique KV instead of another prefix peel. |
| **k** | `VLLM_SUFFIX_SESSION_DEAD_K` | 3 | EWA detector: threshold is `max(dead, ewa + k·std)` of **that session's** think times. |
| **gaps** | `VLLM_SUFFIX_SESSION_MIN_GAPS` | 2 | How many think-time samples before EWA may raise the threshold above `dead`. |

### `vllm-prefix-pruning-on-sessions-global:T:S:dead:q` → `16384:16384:180:0.95`

Same **T / S / dead**, then **q** (`VLLM_SUFFIX_SESSION_QUANTILE` = 0.95) is a **server-wide** think-time quantile: wipe if idle longer than 95% of gaps seen on this server (`max(dead, q-quantile)`).

`vllm-native-lru` has no extra knobs (APC + LRU, prefix-pruning flag off).

## What the table columns mean

These are **CPU-sim** totals over **returning** turns only (turn 0 is the first prompt; it is excluded). Same eviction volume can still change these numbers: what remains in cache is a front gap vs a tail miss.

| column | meaning |
|---|---|
| **policy** | Eviction arm and knobs (`T:S:K:idle` or session `T:S:dead:…`). |
| **recomputed** | Sum of `prompt_tokens − cached_tokens` on returning turns. Tokens that had to be prefilled again. **Lower is better.** Not “blocks evicted.” |
| **vs native** | `recomputed / native recomputed`. `0.79×` = 21% less recompute than native LRU. |
| **prefill s** | Cost-model time to do that recompute (`b·tokens + c·tokens²` per scheduled range). Front gaps are cheaper than the same-sized tail hole on a long sequence. |
| **TTFT p50 / p99** | Simulated time-to-first-token (seconds) for returning turns. p50 = typical; p99 = tail. |
| **gaps** | How many returning turns had a **front hole** `[0, g)` and still reused a cached suffix after `g`. Prefix-pruning’s intended hit shape. Native is usually `0`. |
| **tail miss** | How many returning turns lost unique tokens at the **end** (the conversation tail). Native’s usual miss. `0` means every return kept its suffix. |

A good prefix-pruning row looks like: **recomputed down**, **gaps ≈ number of returns**, **tail miss ≈ 0**. Native is the opposite: **gaps 0**, **tail miss high**.

**Why recomputed can fall if “tokens evicted” look the same.**  
At one allocation, both arms free about the same number of GPU blocks (whatever the new request needs). `recomputed` is not that. It is how much of **this returning prompt** missed cache.

If you take **G tokens from the session that is about to return**, front vs tail is almost the same *count*: you recompute `G + new` either way (only **prefill s** / TTFT change, because a front hole is cheaper).

The count drops because **G is not the same for that session**:

- Native APC wipes **unique suffixes**. Under cyclic overflow it often clears most of one chat’s unique KV. That chat is the next arrival → recompute ~the whole 32K every turn (`tail miss`).
- Tiers take **16K from everyone’s front** before taking more from anyone. The chat that returns only lost 16K and **still has its suffix** → recompute ~16K + new (`gap`), not 32K.

Same free-space demand, different *victim*. Prefix tokens *are* recomputed (that is `gaps`). You just stop recomputing the much larger tail.

Toy 32K session, 5 returns, 16K overflow each time:

- native: miss ~32K each return → ~160K recomputed  
- pruning: miss 16K front, hit the rest → ~80K recomputed  

That is the 2.43M → 1.91M cyclic gap (14 sessions), not “we evicted fewer blocks.”

## eq32k-s14-abandon / abandon

| policy | recomputed | vs native | prefill s | TTFT p50 | TTFT p99 | gaps | tail miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-native-lru` | 1,614,352 | 1.000× | 100.05 | 2.435 | 17.393 | 0 | 54 |
| `vllm-prefix-pruning-on-sessions-global:16384:16384:180:0.95` | 982,336 | 0.609× | 49.07 | 1.116 | 4.524 | 54 | 0 |
| `vllm-prefix-pruning-on-sessions:16384:16384:180:3:2` | 982,336 | 0.609× | 49.07 | 1.116 | 4.524 | 54 | 0 |
| `vllm-prefix-pruning:16384:16384:16:60` | 1,011,184 | 0.626× | 50.95 | 1.116 | 5.627 | 57 | 1 |

## eq32k-s14-cyclic / cyclic

| policy | recomputed | vs native | prefill s | TTFT p50 | TTFT p99 | gaps | tail miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-native-lru` | 2,428,160 | 1.000× | 144.36 | 3.992 | 17.907 | 0 | 70 |
| `vllm-prefix-pruning-on-sessions-global:16384:16384:180:0.95` | 1,911,168 | 0.787× | 107.17 | 5.044 | 14.072 | 70 | 0 |
| `vllm-prefix-pruning-on-sessions:16384:16384:180:3:2` | 1,911,168 | 0.787× | 107.17 | 5.044 | 14.072 | 70 | 0 |
| `vllm-prefix-pruning:16384:16384:16:60` | 1,911,168 | 0.787× | 107.17 | 5.044 | 14.072 | 70 | 0 |

## eq32k-s14-pareto / pareto

| policy | recomputed | vs native | prefill s | TTFT p50 | TTFT p99 | gaps | tail miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-native-lru` | 1,207,088 | 1.000× | 72.90 | 2.193 | 7.151 | 0 | 37 |
| `vllm-prefix-pruning-on-sessions-global:16384:16384:180:0.95` | 1,138,160 | 0.943× | 57.83 | 0.991 | 3.541 | 63 | 0 |
| `vllm-prefix-pruning-on-sessions:16384:16384:180:3:2` | 1,138,160 | 0.943× | 57.83 | 0.991 | 3.541 | 63 | 0 |
| `vllm-prefix-pruning:16384:16384:16:60` | 1,138,048 | 0.943× | 57.83 | 0.991 | 3.541 | 62 | 1 |

## eq32k-s14-random / random

| policy | recomputed | vs native | prefill s | TTFT p50 | TTFT p99 | gaps | tail miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-native-lru` | 1,011,552 | 1.000× | 61.36 | 1.592 | 3.834 | 0 | 31 |
| `vllm-prefix-pruning-on-sessions-global:16384:16384:180:0.95` | 944,624 | 0.934× | 47.05 | 0.928 | 2.493 | 56 | 0 |
| `vllm-prefix-pruning-on-sessions:16384:16384:180:3:2` | 944,624 | 0.934× | 47.05 | 0.928 | 2.493 | 56 | 0 |
| `vllm-prefix-pruning:16384:16384:16:60` | 955,408 | 0.944× | 47.54 | 0.928 | 2.493 | 56 | 1 |

## eq32k-s24-abandon / abandon

| policy | recomputed | vs native | prefill s | TTFT p50 | TTFT p99 | gaps | tail miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vllm-native-lru` | 3,059,200 | 1.000× | 182.03 | 2.549 | 18.345 | 0 | 88 |
| `vllm-prefix-pruning-on-sessions-global:16384:16384:180:0.95` | 2,964,736 | 0.969× | 174.41 | 2.348 | 17.377 | 71 | 17 |
| `vllm-prefix-pruning-on-sessions:16384:16384:180:3:2` | 2,964,736 | 0.969× | 174.41 | 2.348 | 17.377 | 71 | 17 |
| `vllm-prefix-pruning:16384:16384:16:60` | 2,982,400 | 0.975× | 175.08 | 2.348 | 17.377 | 83 | 17 |


No real win here (~3% vs native, session only 0.6% ahead of pruning). After 8 of 24 walk away, **16 live 32K sessions remain**. 16 × 32,768 = 524K at turn 0 already, vs 319K cache (~1.6×) — same overflow as the 14-session *all-live* case, even if the 8 dead chats are deleted instantly. Prefix peels and session wipes cannot cover that. Compare `eq32k-s14-abandon`: 10 live × 32K ≈ 328K, which almost fits the cache once the 4 dead are wiped, so you see 0.61×. `tail_miss` 88 → 17 means a few full suffix misses were avoided; the other ~3M tokens are still recomputed because the 16 live chats do not fit.
