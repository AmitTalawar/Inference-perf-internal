# Weka Trace Replay Compilation Flow (Preprocessing Stage)

This document explains, in detail, the **compile/preprocessing** half of Weka trace replay in `inference-perf`: everything that happens before the first live HTTP request is sent to the model server.

It answers two questions at each stage:

- **What** does this stage do?
- **Why** is this stage needed?

---

## 1. Entry and Configuration Wiring

### 1.1 CLI entrypoint

The benchmark starts at `main_cli()` in `inference_perf/main.py`.

The CLI layer:

- parses command line arguments (`-c/--config_file`, log level, optional overrides),
- loads YAML config via `read_config(...)`,
- validates it against Pydantic models.

### 1.2 Replay-specific validation

For Weka replay, config validation enforces:

- `data.type: weka_trace_replay`
- `load.type: trace_session_replay`

This check exists because Weka replay is session-graph based and depends on session lifecycle APIs (`activate_session`, `check_session_completed`, etc.), not the stateless request generator path.

---

## 2. Datagen Construction Boundary

The heavy compile stage starts when `main_cli()` creates:

- `WekaTraceReplayDataGenerator(...)`

This constructor performs almost all graph and prompt preparation up front.

### 2.1 Parent runtime setup

`WekaTraceReplayDataGenerator` inherits `ReplayGraphSessionGeneratorBase`, so its constructor first sets up:

- output registry structures,
- session state containers,
- per-session scheduling containers,
- optional multiprocessing completion queue.

These are runtime scaffolds used later during live replay, but initialized now.

---

## 3. Corpus and Deterministic Token Source Setup

Weka traces store hash IDs and token counts, not raw text prompts. So the generator must synthesize text deterministically.

### 3.1 Corpus load

The generator loads corpus text (default Shakespeare file or configured corpus path), prepends a fixed base prompt, and tokenizes the full text once.

### 3.2 Deterministic hash RNG

`HashIdRandomGenerator` is initialized with `base_seed`.

At per-trace reconstruction time it is scoped by:

- base seed
- trace ID
- hash ID

The seed formula ensures deterministic mapping:

- same `(base_seed, trace_id, hash_id)` -> same token block,
- different trace ID can map same hash ID differently.

### 3.3 Cache dictionary

`self._cache: Dict[int, List[int]]` stores decoded token blocks per hash ID during one trace reconstruction.

Purpose:

- avoid recomputing token block lookup repeatedly,
- keep within-trace mapping stable and efficient.

---

## 4. Loading Weka Traces

`_load_weka_traces()` accepts one source:

- `trace_directory` (`.json` + `.jsonl` files),
- explicit `trace_files`,
- HF dataset path (`hf_hub_download` of `traces.jsonl`).

### 4.1 Validation

Each record is validated into `WekaTrace` (and nested request models).

### 4.2 Guardrails

- duplicate `trace.id` is rejected,
- invalid files can be skipped or fail-fast depending on `skip_invalid_files`.

Why this stage exists:

- normalize heterogeneous input files into one trusted typed representation,
- fail early on schema drift.

---

## 5. Per-Trace Session Build

`_build_sessions_from_traces(...)` loops all traces and builds one replayable session graph per trace.

For each trace:

1. reconstruct raw calls from Weka semantics (`_reconstruct_raw_calls`),
2. convert raw calls to DAG (`build_graph`),
3. wrap into `ReplaySession`.

After all sessions are built:

- sessions are shuffled with `base_seed` for deterministic randomized order.

Why:

- avoid replaying traces in static source ordering,
- improve stress behavior while preserving reproducibility.

---

## 6. Raw Call Reconstruction (Core Compile Logic)

`_reconstruct_raw_calls(trace)` is the most important compile-time function.

It converts one Weka trace (hashes/timings/tokens) into chronological `RawCall` objects containing:

- prompt messages,
- expected output message,
- token counts,
- timestamps,
- model identity.

### 6.1 Trace-local reset

At function start:

- hash cache is cleared,
- RNG trace scope is set (`set_trace_id(trace.id)`).

Why:

- guarantees deterministic mapping per trace,
- avoids leakage across traces.

### 6.2 Parent/subagent planning

Trace requests are separated into:

- normal parent requests,
- subagent entries with nested requests.

Subagent nested requests are packed into parallel streams (`_pack_into_streams`) based on overlap timing.

Why:

- preserves subagent internal concurrency semantics,
- creates stream-specific child session IDs.

### 6.3 Timing warping

`_build_trace_idle_timing(...)` builds adjusted timings.

If idle gaps are huge, `_IdleGapTimeWarp` caps gap effects (`trace_idle_gap_cap_seconds`) so replay does not spend most wall clock idling.

Why:

- retain temporal structure,
- avoid unrealistic long pauses from raw traces.

### 6.4 Deterministic block decode

`decode_block_tokens(hash_ids)`:

- for each hash ID:
  - reseed RNG for that hash,
  - pick deterministic start index in tokenized corpus,
  - slice exactly one block of size `trace_bs`,
  - cache result.

Why:

- convert hash IDs into stable token blocks,
- emulate block identity behavior required for KV-cache studies.

### 6.5 Conversation reconstruction engine

`ConversationReconstructor` builds turn-by-turn message history with block-level prefix semantics.

Important pieces:

- `longest_common_prefix(prev_hash_ids, curr_hash_ids)` detects branch/prefix reuse,
- `truncate_synth_buf_at_block(...)` rewinds divergent history,
- `init_turn_0(...)` bootstraps first prompt,
- `advance_turn(...)` appends assistant/user segments for next turn based on LCP and token lengths.

Why:

- Weka traces describe prompt evolution in hash-space; this maps it into role-based message history.

### 6.6 Expected output construction

For each call:

- prompt messages come from reconstructor snapshot,
- expected output text is computed:
  - for non-final turns: lookahead reconstructor infers assistant segment that should be inserted by next turn shape,
  - final turn: fallback sampled output tokens.

Why:

- every call must carry output text/tokens for graph construction and runtime metadata,
- non-final turns need structural continuity with subsequent turn composition.

### 6.7 Parent + child call assembly

Parent and subagent child calls are combined and sorted by `(t_start_ms, call_id)`.

This ordered call list is then handed to DAG compiler.

---

## 7. Graph Compilation from Raw Calls

`build_graph(raw_calls, source_file=...)` from `otel_trace_to_replay_graph.py` transforms chronological calls into a replay DAG.

### 7.1 Why graph compilation is needed

Raw calls only give call-local prompt/output/timing. Replay needs explicit dependency structure:

- which event must wait for which predecessor,
- where predecessor output appears in successor input,
- where shared prefix opportunities are.

### 7.2 Predecessor inference

For each call `i`, graph builder scans prior calls `j`:

- causal dependency checks (`get_causal_dep`) by content/tool-call matching,
- temporal fallback predecessor when no causal edge captures timing continuity.

Why:

- preserve both semantic causality and practical temporal sequencing.

### 7.3 Input decomposition

`decompose_input(...)` splits each call input into `InputSegment`s:

- `shared`: message prefix reused from a predecessor history,
- `output`: assistant message slot corresponding to predecessor output,
- `unique`: new content local to current call.

Why:

- later runtime substitution/wait logic needs this structured map, not just role labels.

### 7.4 Wait computation

`wait_ms` is computed from call timestamp minus latest predecessor end timestamp.

Why:

- replay should delay successor dispatch to preserve inter-call timing gaps.

### 7.5 Compiled session store

When `compiled_store_path` is set, each compiled `ReplayGraph` (including `__dupseed*` duplicates) is written through under `sessions/` with `session_headers`, plus a `manifest.json` that records compile identity. New stores use orjson + zstd (`*.orjson.zst`). Existing gzip-JSON stores remain readable via `manifest.codec`. Encode/decode of session files runs in a parent-process thread pool.

On a later run, if `duplicate_sessions_target` is set and the store already has at least that many artifacts with a matching identity, `__init__` **skips corpus tokenization, raw trace load, reconstruct, and `build_graph`**. It seeded-shuffles manifest ids, deserializes that many session files, then continues with `initialize_sessions` as today.

If the store has fewer artifacts than the target, the existing load/compile path runs for the gap only: unique traces are still loaded via `_load_weka_traces()`, store-aware expansion continues the global `__dupseed` counter from the max suffix already in the store, and only missing ids are reconstructed. Parent process writes new files after compile (workers never write the store).

Compile identity (tokenizer `name_or_path`, corpus path and byte size, `base_seed`, `default_block_size`, `trace_idle_gap_cap_seconds`, `parallelize_sibling_subagents`, `max_parallel_subagents`, schema version) is checked before reuse. Model names are not part of identity.

---

## 8. Session Schedule Materialization

After each session graph is built, `initialize_sessions(...)` finalizes runnable schedule state.

### 8.1 Optional duplication

If `duplicate_sessions_target` is set and source sessions are fewer:

- sessions are duplicated round-robin with `_dupN` suffix IDs.

Why:

- allows stress tests with higher session counts than source corpus.

### 8.2 Per-session event schedule

`_build_replay_schedule()` and `_build_session_schedule(session)` create `ReplaySessionEvent`s with:

- qualified event IDs (`session_id:event_id`),
- qualified predecessor IDs,
- qualified segment source IDs,
- capped `wait_ms` using `max_wait_ms`,
- expected output token metadata.

This schedule is what loadgen will enqueue later.

---

## 9. End of Compilation Stage

Compilation stage ends when `WekaTraceReplayDataGenerator.__init__` returns.

At that point, the system has:

- all sessions loaded,
- each session converted to a graph,
- each graph converted into replay events,
- session states initialized,
- optional duplicates generated.

No live HTTP traffic has been sent yet.

---

## 10. Why this heavy upfront compile stage exists

The design intentionally pushes complexity up front to make live replay mostly about:

- dependency orchestration,
- wait handling,
- HTTP dispatch.

Primary reasons:

- deterministic reproducibility from hash IDs,
- explicit causal graph for agentic/session replay,
- predictable token and timing metadata for benchmarking.

