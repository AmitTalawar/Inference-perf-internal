#!/usr/bin/env python3
# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simulate a multi-turn workload against vLLM's real scheduler and KV cache.

The vLLM Scheduler and KVCacheManager run unmodified on CPU, so admission,
chunked prefill, eviction order and suffix hits are exact. The model is
replaced by a cost model: every scheduler step takes

  step_s + sum over scheduled token ranges [s, e) of b*(e - s) + c*(e^2 - s^2)

seconds (b, c from the Phase 1 fit), and every request that finishes its
prompt or decodes in a step samples a token.

Policies:
  vllm-native-lru                         APC + LRU (flag off)
  vllm-prefix-pruning:T[:S[:K[:idle]]]    prefix-first tiers
  vllm-prefix-pruning-on-sessions:T[:S[:dead[:k[:gaps]]]]
  vllm-prefix-pruning-on-sessions-global:T[:S[:dead[:q]]]
  fraction:F                              prefix fraction F

Must run with the Python environment of a vLLM checkout containing the
suffix-eviction patch (--vllm-src). Writes <out>/requests.jsonl and
<out>/summary.json.

Usage:
  .venv/bin/python suffix_eviction_sim.py --spec spec.json --policy tiers:4096 --out DIR
  suffix_eviction_sim.py --sweep --specs a.json,b.json --policies vllm-native-lru,vllm-prefix-pruning:4096 --out DIR
"""

from __future__ import annotations

import argparse
import functools
import heapq
import json
import os
import subprocess
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suffix_eviction_workload as wl  # noqa: E402

DEFAULT_VLLM_SRC = os.environ.get(
    "SUFFIX_SIM_VLLM_SRC", "/home/ubuntu/sneh/opensource/eviction/vllm"
)
DEFAULT_PYTHON = os.environ.get("SUFFIX_SIM_PYTHON", f"{DEFAULT_VLLM_SRC}/.venv/bin/python")


def policy_env(policy: str) -> dict[str, str]:
    # Clear prior knobs so in-process sweeps cannot leak across policies.
    base = {
        "VLLM_SUFFIX_PRESERVING_EVICTION": "0",
        "VLLM_SUFFIX_EVICTION_PREFIX_FRACTION": "0.5",
        "VLLM_SUFFIX_EVICTION_TIER_TOKENS": "0",
        "VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS": "0",
        "VLLM_SUFFIX_EVICTION_MAX_TIERS": "0",
        "VLLM_SUFFIX_EVICTION_IDLE_TTL_S": "0",
        "VLLM_SUFFIX_SESSION_DEAD_TTL_S": "0",
        "VLLM_SUFFIX_SESSION_EWA_ALPHA": "0.3",
        "VLLM_SUFFIX_SESSION_DEAD_K": "3",
        "VLLM_SUFFIX_SESSION_MIN_GAPS": "2",
        "VLLM_SUFFIX_SESSION_DETECTOR": "ewa",
        "VLLM_SUFFIX_SESSION_QUANTILE": "0.95",
        "VLLM_SUFFIX_SESSION_GAP_WINDOW": "256",
    }
    kind, _, rest = policy.partition(":")
    kind = {
        "stock": "vllm-native-lru",
        "tiers": "vllm-prefix-pruning",
        "session": "vllm-prefix-pruning-on-sessions",
        "sessiong": "vllm-prefix-pruning-on-sessions-global",
    }.get(kind, kind)
    if kind == "vllm-native-lru":
        return base
    env = {**base, "VLLM_SUFFIX_PRESERVING_EVICTION": "1"}
    if kind == "fraction":
        env["VLLM_SUFFIX_EVICTION_PREFIX_FRACTION"] = rest
    elif kind in (
        "vllm-prefix-pruning",
        "vllm-prefix-pruning-on-sessions",
        "vllm-prefix-pruning-on-sessions-global",
    ):
        parts = rest.split(":") if rest else ["4096"]
        env["VLLM_SUFFIX_EVICTION_TIER_TOKENS"] = parts[0]
        env["VLLM_SUFFIX_EVICTION_TIER_STEP_TOKENS"] = parts[1] if len(parts) > 1 else parts[0]
        if kind == "vllm-prefix-pruning":
            if len(parts) > 2:
                env["VLLM_SUFFIX_EVICTION_MAX_TIERS"] = parts[2]
            if len(parts) > 3:
                env["VLLM_SUFFIX_EVICTION_IDLE_TTL_S"] = parts[3]
        elif kind == "vllm-prefix-pruning-on-sessions":
            env["VLLM_SUFFIX_SESSION_DETECTOR"] = "ewa"
            env["VLLM_SUFFIX_SESSION_DEAD_TTL_S"] = parts[2] if len(parts) > 2 else "90"
            if len(parts) > 3:
                env["VLLM_SUFFIX_SESSION_DEAD_K"] = parts[3]
            if len(parts) > 4:
                env["VLLM_SUFFIX_SESSION_MIN_GAPS"] = parts[4]
        else:
            env["VLLM_SUFFIX_SESSION_DETECTOR"] = "global"
            env["VLLM_SUFFIX_SESSION_DEAD_TTL_S"] = parts[2] if len(parts) > 2 else "90"
            if len(parts) > 3:
                env["VLLM_SUFFIX_SESSION_QUANTILE"] = parts[3]
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return env


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(records: list[dict], makespan: float) -> dict:
    ret = [r for r in records if r["turn"] > 0]
    ttft = [r["ttft_s"] for r in ret]
    return {
        "requests": len(records),
        "returning_requests": len(ret),
        "makespan_s": makespan,
        "returning_recomputed_tokens": sum(r["prompt_tokens"] - r["cached_tokens"] for r in ret),
        "returning_prefill_s": sum(r["prefill_s"] for r in ret),
        "returning_full_hits": sum(r["prompt_tokens"] - r["cached_tokens"] <= r["new_tokens"] for r in ret),
        "returning_with_gap": sum(r["gap_tokens"] > 0 for r in ret),
        "returning_tail_miss": sum(r["tail_miss_tokens"] > 0 for r in ret),
        "mean_gap_tokens_when_gap": (
            sum(r["gap_tokens"] for r in ret if r["gap_tokens"]) / max(1, sum(r["gap_tokens"] > 0 for r in ret))
        ),
        "ttft_p50_s": percentile(ttft, 0.5),
        "ttft_p90_s": percentile(ttft, 0.9),
        "ttft_p99_s": percentile(ttft, 0.99),
        "ttft_mean_s": sum(ttft) / len(ttft) if ttft else None,
        "preemptions": sum(r["preemptions"] for r in records),
    }


def simulate(spec: dict, policy: str, num_blocks: int, b: float, c: float, step_s: float, max_num_seqs: int) -> dict:
    os.environ.update(policy_env(policy))
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    vllm_src = os.environ.get("SUFFIX_SIM_VLLM_SRC", DEFAULT_VLLM_SRC)
    sys.path.insert(0, vllm_src)
    sys.path.insert(0, f"{vllm_src}/tests/v1/core")
    import utils  # noqa: PLC0415
    from vllm.config import ModelConfig  # noqa: PLC0415

    max_model_len = 262144
    utils.ModelConfig = functools.partial(ModelConfig, max_model_len=max_model_len)
    from utils import create_scheduler  # noqa: PLC0415
    from vllm.sampling_params import SamplingParams  # noqa: PLC0415
    from vllm.utils.hashing import sha256  # noqa: PLC0415
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: PLC0415
    from vllm.v1.outputs import ModelRunnerOutput  # noqa: PLC0415
    from vllm.v1.request import Request  # noqa: PLC0415

    block_size = 16
    sched = create_scheduler(
        enable_prefix_caching=True,
        num_blocks=num_blocks,
        block_size=block_size,
        max_num_batched_tokens=2048,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
    )
    sched.max_model_len = max_model_len
    init_none_hash(sha256)
    hasher = get_request_block_hasher(block_size, sha256)

    kvm = sched.kv_cache_manager
    hit_info: dict[str, tuple[int, tuple[int, int] | None]] = {}
    orig_get_computed_blocks = kvm.get_computed_blocks

    def get_computed_blocks(request):
        res = orig_get_computed_blocks(request)
        hit_info[request.request_id] = (res[1], kvm.get_suffix_hit(res[0], res[1]))
        return res

    kvm.get_computed_blocks = get_computed_blocks

    sessions = {s["id"]: s for s in spec["sessions"]}
    prompts: dict[int, list[int]] = {}
    events: list[tuple[float, int, int]] = [(s["start_s"], s["id"], 0) for s in spec["sessions"]]
    heapq.heapify(events)
    live: dict[str, dict] = {}
    live_outputs: dict[int, list[int]] = {}
    records: list[dict] = []
    clock = 0.0
    # Drive idle-TTL demotion from sim time, not wall clock.
    kvm.block_pool._time_fn = lambda: clock
    out_tokens = spec["output_tokens"]

    while events or live:
        while events and events[0][0] <= clock:
            t, sid, turn = heapq.heappop(events)
            sess = sessions[sid]
            if turn == 0:
                prompt = wl.initial_prompt(spec, sess)
            else:
                prompt = wl.next_prompt(spec, sess, turn, prompts[sid], live_outputs[sid])
            prompts[sid] = prompt
            rid = f"s{sid}t{turn}"
            req = Request(
                request_id=rid,
                prompt_token_ids=prompt,
                sampling_params=SamplingParams(max_tokens=out_tokens, ignore_eos=True),
                pooling_params=None,
                block_hasher=hasher,
                session_id=str(sid),
            )
            sched.add_request(req)
            live[rid] = {
                "req": req, "session": sid, "turn": turn, "arrival_s": t, "prompt_tokens": len(prompt),
                "new_tokens": len(prompt) if turn == 0 else spec["user_tokens"] + out_tokens,
                "prefill_s": 0.0, "ttft_s": None,
            }  # fmt: skip
        if not live:
            clock = events[0][0]
            continue

        out = sched.schedule()
        starts = {nr.req_id: nr.num_computed_tokens for nr in out.scheduled_new_reqs}
        cr = out.scheduled_cached_reqs
        starts.update(zip(cr.req_ids, cr.num_computed_tokens, strict=False))
        step = step_s
        req_ids = list(out.num_scheduled_tokens)
        for rid in req_ids:
            s = starts[rid]
            e = s + out.num_scheduled_tokens[rid]
            cost = b * (e - s) + c * (e * e - s * s)
            step += cost
            if s < live[rid]["prompt_tokens"]:
                live[rid]["prefill_s"] += cost
        clock += step if req_ids else 0.0

        sampled = []
        for rid in req_ids:
            r = live[rid]["req"]
            if r.num_computed_tokens >= r.num_tokens:
                token = zlib.crc32(f"{rid}-{r.num_output_tokens}".encode()) % 50000
                sampled.append([wl.TOKEN_LO + token])
                if live[rid]["ttft_s"] is None:
                    live[rid]["ttft_s"] = clock - live[rid]["arrival_s"]
            else:
                sampled.append([])
        mro = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        sched.update_from_output(out, mro)

        for rid in [rid for rid, rec in live.items() if rec["req"].is_finished()]:
            rec = live.pop(rid)
            req = rec.pop("req")
            prefix_hit, suffix_hit = hit_info.pop(rid, (0, None))
            cached = prefix_hit + (suffix_hit[1] - suffix_hit[0] if suffix_hit else 0)
            gap = suffix_hit[0] - prefix_hit if suffix_hit else 0
            prev_len = rec["prompt_tokens"] - rec["new_tokens"]
            tail_miss = max(0, prev_len - cached - gap) if rec["turn"] > 0 else 0
            rec.update(
                finish_s=clock, cached_tokens=cached, prefix_hit_tokens=prefix_hit, gap_tokens=gap,
                tail_miss_tokens=tail_miss, preemptions=req.num_preemptions,
            )  # fmt: skip
            records.append(rec)
            live_outputs[rec["session"]] = list(req.output_token_ids)
            sess = sessions[rec["session"]]
            if rec["turn"] + 1 < spec["turns"]:
                heapq.heappush(events, (clock + sess["think_s"][rec["turn"]], rec["session"], rec["turn"] + 1))

    records.sort(key=lambda r: (r["arrival_s"], r["session"]))
    return {"records": records, "summary": summarize(records, clock)}


def run_one(
    spec_path: str, policy: str, out_dir: str, num_blocks: int, b: float, c: float, step_s: float, max_num_seqs: int
) -> dict:
    spec = json.loads(Path(spec_path).read_text())
    t0 = time.time()
    res = simulate(spec, policy, num_blocks, b, c, step_s, max_num_seqs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "requests.jsonl").open("w") as f:
        for r in res["records"]:
            f.write(json.dumps(r) + "\n")
    summary = {
        "spec": spec_path, "pattern": spec["pattern"], "policy": policy, "num_blocks": num_blocks,
        "b": b, "c": c, "step_s": step_s, "working_set_tokens": wl.working_set_tokens(spec),
        "cache_tokens": (num_blocks - 1) * 16, "wall_s": time.time() - t0, **res["summary"],
    }  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def sweep_worker(args: tuple) -> dict:
    spec_path, policy, out_dir, a = args
    cmd = [
        a.python, __file__, "--spec", spec_path, "--policy", policy, "--out", out_dir,
        "--num-blocks", str(a.num_blocks), "--b", str(a.b), "--c", str(a.c), "--step-s", str(a.step_s),
        "--max-num-seqs", str(a.max_num_seqs),
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"spec": spec_path, "policy": policy, "error": proc.stderr[-2000:]}
    return json.loads((Path(out_dir) / "summary.json").read_text())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec")
    p.add_argument("--policy")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--specs", help="Comma-separated spec files for --sweep")
    p.add_argument("--policies", help="Comma-separated policies for --sweep")
    p.add_argument("--jobs", type=int, default=os.cpu_count())
    p.add_argument("--python", default=DEFAULT_PYTHON, help="Interpreter for sweep workers")
    p.add_argument("--num-blocks", type=int, default=19927, help="GPU blocks including the null block")
    p.add_argument("--b", type=float, default=3.66e-5, help="Linear prefill cost, s/token")
    p.add_argument("--c", type=float, default=6.583e-10, help="Attention prefill cost, s/token^2")
    p.add_argument("--step-s", type=float, default=0.012, help="Fixed cost per scheduler step")
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    if not a.sweep:
        s = run_one(a.spec, a.policy, a.out, a.num_blocks, a.b, a.c, a.step_s, a.max_num_seqs)
        print(json.dumps(s))
        return 0

    jobs = []
    for spec_path in a.specs.split(","):
        for policy in a.policies.split(","):
            name = f"{Path(spec_path).stem}__{policy.replace(':', '-')}"
            jobs.append((spec_path, policy, str(Path(a.out) / name), a))
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        results = list(ex.map(sweep_worker, jobs))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "sweep.json").write_text(json.dumps(results, indent=2))
    cols = ["pattern", "policy", "returning_recomputed_tokens", "returning_prefill_s", "ttft_p50_s", "ttft_p99_s",
            "makespan_s", "returning_with_gap", "returning_tail_miss", "preemptions"]  # fmt: skip
    print("\t".join(cols))
    for r in results:
        if "error" in r:
            print(f"{r['spec']}\t{r['policy']}\tERROR {r['error'][-300:]}")
            continue
        print("\t".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in cols))
    return 0


if __name__ == "__main__":
    sys.exit(main())
