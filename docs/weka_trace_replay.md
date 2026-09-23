# Weka Trace Replay

The Weka Trace Replay capability allows you to benchmark GenAI model servers by replaying complex, real-world multi-agent execution traces. It converts a raw trace into a **dependency graph of events** (parent-child turns, tool calls, subagent spawns) and replays it concurrently, maintaining full causal/dependency fidelity.

---

## 🏗️ How it Works

1. **Dataset Download**: At startup, `inference-perf` downloads the Weka trace dataset from Hugging Face (e.g. `semianalysisai/cc-traces-weka-with-subagents-060826-256k`).
2. **Graph Compilation**: The replay generator parses each trace session, compiling individual turns and events into an **Execution Graph** of nodes. Each node represents an event (an inference call or tool/subagent execution).
3. **Causal Propagation**: Nodes register parent-child relationships. The text output of parent nodes is dynamically cached and substituted into the prompt messages of child nodes at runtime (e.g. tool execution output is placed back in the next LLM call).
4. **Session-based Execution**: A thread pool runs sessions concurrently. Within each session, nodes are executed as soon as their parents complete.
5. **Think-Time Simulation**: Think times and idle gaps between turns are simulated and capped using `trace_idle_gap_cap_seconds`.
6. **Session Header Overrides**: Each trace session can optionally carry `session_headers` (for example auth headers). These are applied to all requests in that session and override global/static headers at request-construction time.

---

## ⚙️ Configuration

To use Weka Trace Replay, define the data and load sections in your configuration YAML:

```yaml
load:
  type: trace_session_replay
  stages:
    - concurrent_sessions: 16
      num_sessions: 391
  num_workers: 8
  worker_max_concurrency: 100

api:
  type: chat
  streaming: true

server:
  type: mock # or openai/vllm/sglang/tgi

tokenizer:
  pretrained_model_name_or_path: HuggingFaceTB/SmolLM2-135M-Instruct

data:
  type: weka_trace_replay
  weka_trace_replay:
    hf_dataset_path: "semianalysisai/cc-traces-weka-with-subagents-060826-256k"
    num_dataset_entries: 500
    use_static_model: true
    static_model_name: "mock-model"
    default_block_size: 64
    skip_invalid_files: true
    trace_idle_gap_cap_seconds: 1.0 # Caps think-time delay between turns to 1s
```

### Optional Trace Fields

Each Weka trace record can include:

```json
{
  "id": "trace-123",
  "session_headers": {
    "Authorization": "Bearer session-specific-token",
    "x-tenant-id": "tenant-a"
  }
}
```

When present, these headers are attached to every request in that replay session and take precedence over `server.api_key` and `api.headers`.

---

## Compiled session store

Set `compiled_store_path` to persist compiled `ReplayGraph`s (including `__dupseed*` duplicates) and reuse them on later runs.

```yaml
data:
  type: weka_trace_replay
  weka_trace_replay:
    trace_files:
      - /path/to/traces.jsonl
    num_dataset_entries: 3001
    duplicate_sessions_target: 3001
    compiled_store_path: /path/to/weka_compiled_store
    use_static_model: true
    static_model_name: "my-model"
```

Behavior:

- `compiled_store_path` is **optional and extra**. Exactly one of `trace_files` / `trace_directory` / `hf_dataset_path` is still required. Those sources are used on cache miss or partial fill. They are not opened on a full store hit.
- Sample size for a store-backed run is `duplicate_sessions_target`. The store is shuffled with `load.base_seed` and the first `duplicate_sessions_target` artifacts are loaded.
- **Full hit** (`stored_count >= duplicate_sessions_target`): skip Shakespeare/corpus tokenization and skip raw trace load. Reconstruct / `build_graph` do not run. The tokenizer is still created by `main.py` for live response token counting.
- **Partial miss / empty store**: tokenize the corpus, load traces from the configured source (`num_dataset_entries` still caps unique raw traces), compile only missing artifacts (including new `__dupseed*` ids), and write them through. Combined session count must reach `duplicate_sessions_target` or the run fails.
- If `duplicate_sessions_target` is unset, today's unique-trace compile path is used. When the store path is set, those unique ids are still looked up / written through after raw load (no skip-raw-load).
- Write-through is always on when `compiled_store_path` is set. There is no separate toggle. Only the parent process writes the store.
- New stores write `sessions/*.orjson.zst` (orjson + zstd level 1). Existing gzip-JSON stores keep `manifest.codec: json.gz` and stay readable. Session files are encoded/decoded in a parent-process thread pool (`min(n_files, cpu_count)`). Eager `initialize_sessions` is unchanged.

### Compile identity

A store built under one compile identity cannot be reused under another. Mismatch fails immediately:

- tokenizer `name_or_path`
- corpus path and corpus file byte size
- `load.base_seed`
- `default_block_size`
- `trace_idle_gap_cap_seconds`
- `parallelize_sibling_subagents`
- `max_parallel_subagents` (recorded as `0` when fan-out is off)
- store `schema_version`

Identity does **not** include `static_model_name`, `model_mapping`, or `use_static_model`. Live request payloads use `server.model_name`; stored `GraphCall.model` is ignored on the wire.

Older manifests that omit the fan-out keys still match runs with `parallelize_sibling_subagents: false` (defaults to off / `max_parallel_subagents: 0`). Enabling fan-out or changing the cap requires deleting or repathing `compiled_store_path` so graphs rebuild once.

### Sibling subagent fan-out

Weka traces often record sibling `type: subagent` entries as a serial pipeline (each starts when the previous ends) even when their first turns share a long hash prefix. Set:

```yaml
data:
  weka_trace_replay:
    parallelize_sibling_subagents: true
    max_parallel_subagents: 8          # 0 = whole wave
    max_inflight_requests: 32          # optional hard HTTP ceiling
```

**Wave definition:** consecutive subagents between two parent (`n`/`s`) turns. Intra-subagent nested turns stay serial; only sibling starts collapse into concurrent batches of `max_parallel_subagents`.

**Concurrency:** `load.stages[].concurrent_sessions` (how many session DAGs) and `max_parallel_subagents` (width of one wave) multiply. With both set to 8, peak in-flight can approach ~64 unless `max_inflight_requests` caps HTTP after predecessor wait.

**Experiment recipes:**

- Routing disagreement: `concurrent_sessions: 2–4`, `max_parallel_subagents: 8`, optional `max_inflight_requests: 32`
- Fidelity baseline: leave `parallelize_sibling_subagents` false (default)

### Headers

Each stored session file includes `session_headers`. On a full hit, raw traces are not loaded, so headers are restored from the payload and applied per `session_id` as today.

---

## 🏃 Running the Benchmark

Run the benchmark with the following command:

```bash
python3 inference_perf/main.py -c configs/weka_trace_replay.yaml
```

The execution results will be written to the `reports-...` directory and summarized in [benchmark-results.md](../benchmark-results.md).
