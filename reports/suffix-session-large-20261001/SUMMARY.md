# Large-session CPU sim + earlier 14×32K

Cache is always **19927 blocks × 16 = 318,816 tokens** (published synthetic GPU KV).
Policies: `vllm-native-lru`, `vllm-prefix-pruning`, `vllm-prefix-pruning-on-sessions`.

Session arms only beat prefix-pruning when sessions actually disappear (`abandon`).

## 14 sessions × 32K (WS 505,344 ≈ 1.6× cache)

| pattern | vllm-native-lru | prefix-pruning / on-sessions | vs native |
|---|---:|---:|---:|
| cyclic | 2,428,160 | **1,638,512** | **0.67×** |
| random | 1,011,552 | **920,208** | **0.91×** |
| pareto | 1,207,088 | **1,031,904** | **0.85×** |
| abandon (4 of 14) | 1,614,352 | tiers 920,928 / **session 909,056** | 0.57× / **0.56×** |

## 24 sessions × 32K (WS 866,304 ≈ 2.7× cache)

| pattern | vllm-native-lru | prefix-pruning / on-sessions | vs native |
|---|---:|---:|---:|
| cyclic | 4,162,560 | 4,147,776 | 0.996× |
| random | 3,611,120 | 3,601,408 | 0.997× |
| pareto | 3,915,760 | 3,862,048 | 0.986× |
| abandon (8 of 24) | 3,059,200 | pruning 2,818,880 / **on-sessions 2,762,240** | 0.921× / **0.903×** |

Cyclic at 2.7× is the known wipe regime: most returns are full tail misses
(native 120/120, prefix-pruning still 79/120). Spreading 4K per session cannot cover
~547K of overflow.

## 32 sessions × 32K (WS 1,155,072 ≈ 3.6× cache)

| pattern | vllm-native-lru | prefix-pruning / on-sessions | vs native |
|---|---:|---:|---:|
| cyclic | 5,550,080 | 5,543,424 | 0.999× |
| random | 5,479,200 | 5,440,992 | 0.993× |
| pareto | **5,413,600** | 5,445,888 | **1.006×** (prefix-pruning slightly worse) |

At 3.6× the prefix-first ladder does not help. Pareto even loses a little
because front gaps plus leftover tail misses cost more than native suffix
eviction.

## Takeaway for live testing

The CPU sim will not show a session-awareness win on cyclic/random/pareto
as long as every session keeps returning. The live Weka 7 concurrent / 12
total mix is the right next measurement: real think times, real abandon,
~1.9× KV at util 0.90.

Raw sweep: `sweep.json` in this directory.
Specs: `../suffix-session-large-specs/`.
