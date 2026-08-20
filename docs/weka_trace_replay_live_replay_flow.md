# Weka Trace Replay Live Replay Flow (Runtime Stage)

This document explains the **live replay/runtime** half of Weka trace replay in `inference-perf`: what happens after compile-time graph/session construction is complete and traffic starts flowing to the model server.

It focuses on:

- session dispatch,
- worker execution model,
- predecessor waits and substitution,
- HTTP request/response lifecycle,
- completion tracking and reporting.

---

## 1. Runtime Start Boundary

After `WekaTraceReplayDataGenerator` compilation completes, `main_cli()` creates:

- `LoadGenerator(datagen, load_config, session_metrics_collector)`,
- `InferencePerfRunner(...)`,

then calls `InferencePerfRunner.run()`.

In `run()`:

- request metrics collector context is entered,
- load generation begins via `loadgen.run(client)`.

For Weka replay with multiprocessing (`num_workers > 0`), this enters `LoadGenerator.mp_run(...)`.

---

## 2. Multiprocess Runtime Initialization

`mp_run(...)` sets up shared runtime primitives:

- `RequestQueue` with one channel per worker (because session replay requests worker affinity),
- shared counters:
  - active requests,
  - finished requests,
- sync events:
  - request phase,
  - cancel signal,
  - stop signal,
- stage barrier.

Then it spawns `Worker` processes.

Why:

- each worker independently runs async request execution loops,
- shared counters/events allow orchestration from main process.

---

## 3. Session Stage Orchestration in Main Process

Weka replay uses `load.type: trace_session_replay`, so each stage runs through `run_session_stage(...)`.

### 3.1 Stage pool model

Main process keeps:

- `pending_session_indices` (not started),
- `active_session_indices` (in flight),
- `completed_session_ids`.

Concurrency is controlled by `stage.concurrent_sessions`.

### 3.2 Session dispatch

`dispatch_session(session_idx)` does:

1. optional buildability check (`is_session_buildable`),
2. fetch session metadata (`get_session_info`),
3. activate session (`activate_session`),
4. fetch all session events (`get_session_events`),
5. enqueue all events immediately into request queue.

Why dispatch all events immediately:

- allows workers to naturally resolve intra-session dependencies by waiting on predecessors.

### 3.3 Session completion detection

Main loop repeatedly checks active sessions via `check_session_completed(session_id)`.

Completion status is driven by worker notifications consumed in `_process_completion_queue()`.

On completion:

- build session metric (`build_session_metric`),
- record metric in `SessionMetricsCollector`,
- cleanup session memory (`cleanup_session`),
- free concurrency slot for next pending session.

---

## 4. Worker Execution Model

Each worker runs `Worker.loop()` with:

- an asyncio event loop (`uvloop`),
- a semaphore sized by `worker_max_concurrency`.

### 4.1 Queue pull and lazy materialization

For each dequeued item:

1. semaphore acquired,
2. lazy event materialized by `LazyLoadDataMixin.get_request(...)`,
3. this calls datagen `load_lazy_data(...)` to build a concrete APIData object,
4. a task coroutine (`schedule_client`) is created.

Why lazy materialization:

- avoids serializing full request payloads across processes,
- allows worker-local construction of concrete API request objects.

### 4.2 Subtle but important concurrency behavior

Semaphore is acquired **before** predecessor wait logic.
So an event waiting for dependencies still occupies a worker slot.

Impact:

- high dependency depth can reduce immediately sendable HTTP requests,
- effective live request concurrency can appear below configured `concurrent_sessions`.

---

## 5. Per-Event Runtime Object Construction

`load_lazy_data(...)` in replay datagen builds `SessionChatCompletionAPIData` (or Anthropic variant) with:

- event ID and predecessor IDs,
- `wait_ms`,
- `input_segments`,
- original messages snapshot,
- output registry handle,
- worker tracker,
- completion queue handle,
- replay flags (`disable_output_substitution`, tool-call handling options),
- back-reference to generator for worker-side eviction.

This object is the core execution state machine for one graph event.

---

## 6. Pre-Send Dependency and Substitution Pipeline

Before HTTP send, worker task calls:

- `wait_for_predecessors_and_substitute()`.

This function performs:

### 6.1 Session fail-fast check

If worker tracker marks session failed, this event is skipped and failure is propagated.

### 6.2 Predecessor wait barrier

For each predecessor event ID:

- waits on `EventOutputRegistry.require_async(...)`.

If predecessor failed or timed out:

- event is marked failed/skipped via `_fail_and_notify(...)`.

### 6.3 Timing wait

If `wait_ms > 0`:

- sleeps required delay to preserve replay timing relation.

### 6.4 Segment substitution/injection

If enabled and relevant segments exist:

- `_build_messages_with_substitution()` rewrites messages by segment type:
  - `output`: replace recorded assistant slot with predecessor live output message/text,
  - `shared`: reuse predecessor input-prefix messages,
  - `unique`: keep local messages, optionally inject random session marker.

Resulting substituted messages become wire payload.

Why this stage exists:

- preserves dependency-correct prompt growth behavior,
- aligns replay with graph-level shared/output semantics.

---

## 7. HTTP Request Send Path

Worker then calls model client `process_request(...)`.

### 7.1 Client session creation/reuse

`openAIModelServerClient` lazily creates one `openAIModelServerClientSession` and reuses it.

### 7.2 Request body assembly

`data.to_request_body(...)` creates payload:

- model name,
- messages,
- max tokens,
- stream flags,
- tool definitions (if any).

Headers are assembled from:

- content type,
- API key/auth headers,
- configured static headers,
- per-request headers,
- optional session-id header.

### 7.3 Network call

Sends:

- `POST <base_url + route>` using `aiohttp.ClientSession.post(...)`.

---

## 8. Response Processing and Failure Handling

### 8.1 Success path

For HTTP 200:

- streaming: parse SSE chunks, chunk times, token timing, output text/tool calls,
- non-streaming: parse JSON body.

Session replay response parser then calls `on_completion(info)`.

### 8.2 Failure path

For non-200 or processing exceptions:

- builds `ErrorResponseInfo`,
- calls `process_failure(...)` on APIData.

For session replay events, `process_failure`:

- marks session failed in worker tracker,
- records failure in output registry,
- emits immediate completion/failure notification to main process queue.

Why immediate queue notification matters:

- prevents main process from waiting indefinitely,
- allows prompt stage/session teardown and slot refill.

---

## 9. Event Completion, Registry, and Cascade Mechanics

`on_completion(info)` is where dependency graph progresses.

It does:

1. `registry.record(event_id, output_text, input_messages, output_message)`,
2. mark event completion time in worker tracker,
3. if all events in session completed on this worker, push session completion payload to queue.

Downstream events waiting on this event wake via `require_async(...)`.

If an event fails:

- registry marks event failed,
- downstream waiters receive `EventFailedError`,
- they skip/fail fast instead of sending broken requests.

---

## 10. Worker-Side Drain and Memory Eviction

Every terminal event path (success/skip/failure) calls drain accounting:

- `_mark_drained_and_maybe_evict(session_id)`.

When drained count reaches total events in session:

- `evict_worker_session(session_id)` is called,
- worker frees graph/session state and registry entries.

Why:

- keeps worker memory bounded to active working set,
- avoids unbounded growth across thousands of sessions.

---

## 11. Main-Process Session Completion Integration

Main process receives session completion payloads via completion queue.

`_process_completion_queue()` updates `ReplaySessionState` with:

- completed event set and completion times,
- failed flag/reason/cancelled count,
- optional substitution telemetry.

Then `check_session_completed(...)` can return true, enabling:

- session metric build + recording,
- parent-side session cleanup,
- activation of next pending session.

---

## 12. Request Metrics Pipeline

For every request (success or failure), model client builds `RequestLifecycleMetric` and records it in metrics collector.

In multiprocess mode:

- metrics are pushed into collector queue from workers,
- collector task drains queue and stores final metrics list.

This is independent from session completion queue; both are needed:

- request metrics for per-request/per-stage performance reports,
- session queue for orchestration and session-lifecycle reports.

---

## 13. Stage End and Run End

At stage completion:

- request phase is cleared,
- queues are joined/drained,
- stage runtime metadata stored.

After all stages:

- loadgen exits,
- collector context closes and finalizes request metrics,
- reports are generated and written.

---

## 14. Why Runtime Stage Is Structured This Way

This design balances correctness and throughput:

- **Correctness**: explicit predecessor waiting + substitution enforces DAG causality.
- **Scalability**: multi-process workers and async HTTP maximize concurrent IO.
- **Observability**: separate request/session metrics and detailed failure propagation.
- **Memory safety**: per-session eviction in both parent and workers prevents accumulation.

---

## 15. Practical Debug Implications

When configured concurrency appears lower than expected, inspect:

- number of tasks blocked in predecessor waits,
- `wait_ms` distribution,
- worker-channel skew from preferred worker routing,
- semaphore saturation by waiting tasks.

In this architecture, "active session count" does not automatically equal "simultaneous HTTP requests in flight", because many enqueued events can be dependency-blocked.

