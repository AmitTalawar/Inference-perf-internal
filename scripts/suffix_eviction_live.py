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
"""Live replay of a suffix_eviction_workload spec against one vLLM server.

Stdlib only, so it runs inside the predictor pod against 127.0.0.1:8000.

  run   resets the prefix cache, replays every session (one thread each) with
        streaming /v1/completions on token-ID prompts, then replays the
        sampled requests cold (prefix cache reset before each) and compares
        greedy outputs. Writes requests.jsonl, correctness.jsonl, summary.json.
  gap   measures the fixed cost of a front gap: caches a session of --length
        tokens, fills the cache so exactly g front tokens of it are evicted
        (needs a single front-first arm, VLLM_SUFFIX_EVICTION_TIER_TOKENS >=
        max_model_len), then returns it with --turn-tokens new tokens.

Usage:
  suffix_eviction_live.py run --spec spec.json --out DIR
  suffix_eviction_live.py gap --gaps 0,256,1024,4096,16384 --reps 3 --out DIR
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suffix_eviction_workload as wl  # noqa: E402

BS = 16


class Client:
    def __init__(self, host: str, port: int, model: str):
        self.host, self.port, self.model = host, port, model

    def _conn(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=1800)

    def post(self, path: str, body: dict | None = None) -> tuple[int, str]:
        c = self._conn()
        c.request("POST", path, body=json.dumps(body or {}), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read().decode()
        c.close()
        return r.status, data

    def get(self, path: str) -> str:
        c = self._conn()
        c.request("GET", path)
        data = c.getresponse().read().decode()
        c.close()
        return data

    def reset_prefix_cache(self) -> None:
        for _ in range(60):
            status, body = self.post("/reset_prefix_cache")
            if status == 200 and "false" not in body.lower():
                return
            time.sleep(1)
        raise RuntimeError(f"reset_prefix_cache failed: {status} {body}")

    def num_gpu_blocks(self) -> int:
        m = re.search(r'vllm:cache_config_info\{[^}]*num_gpu_blocks="(\d+)"', self.get("/metrics"))
        if not m:
            raise RuntimeError("num_gpu_blocks not found in /metrics")
        return int(m.group(1))

    def complete(self, prompt: list[int], max_tokens: int, stream: bool = True) -> dict:
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        c = self._conn()
        t0 = time.monotonic()
        c.request("POST", "/v1/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {r.read()[:500]!r}")
        out: dict = {"token_ids": [], "ttft_s": None}
        if not stream:
            d = json.loads(r.read())
            out["token_ids"] = d["choices"][0]["token_ids"]
            out["usage"] = d.get("usage") or {}
            out["metrics"] = d.get("metrics")
        else:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                for ch in d.get("choices") or []:
                    if ch.get("token_ids"):
                        if out["ttft_s"] is None:
                            out["ttft_s"] = time.monotonic() - t0
                        out["token_ids"] += ch["token_ids"]
                if d.get("usage"):
                    out["usage"] = d["usage"]
                    out["metrics"] = d.get("metrics")
        c.close()
        out["e2e_s"] = time.monotonic() - t0
        usage = out.get("usage") or {}
        out["prompt_tokens"] = usage.get("prompt_tokens")
        out["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        return out


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def run(a: argparse.Namespace, cl: Client) -> int:
    with open(a.spec) as f:
        spec = json.load(f)
    os.makedirs(a.out, exist_ok=True)
    if not a.no_reset:
        cl.reset_prefix_cache()
    lock = threading.Lock()
    records: list[dict] = []
    sampled: dict[tuple[int, int], dict] = {}
    errors: list[str] = []
    t0 = time.monotonic()
    out_f = open(os.path.join(a.out, "requests.jsonl"), "w")

    def session(s: dict) -> None:
        try:
            time.sleep(max(0.0, s["start_s"] - (time.monotonic() - t0)))
            prompt = wl.initial_prompt(spec, s)
            for turn in range(spec["turns"]):
                arrival = time.monotonic() - t0
                res = cl.complete(prompt, spec["output_tokens"])
                rec = {
                    "session": s["id"],
                    "turn": turn,
                    "arrival_s": round(arrival, 4),
                    "finish_s": round(time.monotonic() - t0, 4),
                    "prompt_tokens": len(prompt),
                    "cached_tokens": res["cached_tokens"],
                    "new_tokens": spec["user_tokens"] + spec["output_tokens"] if turn else len(prompt),
                    "ttft_s": res["ttft_s"],
                    "e2e_s": res["e2e_s"],
                    "server": res.get("metrics"),
                    "output_len": len(res["token_ids"]),
                }
                with lock:
                    records.append(rec)
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()
                    if wl.is_sampled(spec, s["id"], turn):
                        sampled[(s["id"], turn)] = {"prompt": prompt, "warm": res["token_ids"]}
                if turn + 1 < spec["turns"]:
                    time.sleep(s["think_s"][turn])
                    prompt = wl.next_prompt(spec, s, turn + 1, prompt, res["token_ids"])
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(f"session {s['id']}: {e!r}")

    threads = [threading.Thread(target=session, args=(s,), daemon=True) for s in spec["sessions"]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    makespan = time.monotonic() - t0
    out_f.close()

    corr = []
    if not a.skip_correctness:
        with open(os.path.join(a.out, "correctness.jsonl"), "w") as f:
            for (sid, turn), v in sorted(sampled.items()):
                cl.reset_prefix_cache()
                cold = cl.complete(v["prompt"], spec["output_tokens"], stream=False)["token_ids"]
                warm = v["warm"]
                div = next((i for i, (x, y) in enumerate(zip(warm, cold, strict=False)) if x != y), None)
                row = {"session": sid, "turn": turn, "identical": warm == cold, "first_divergence": div}
                corr.append(row)
                f.write(json.dumps(row) + "\n")

    ret = [r for r in records if r["turn"] > 0]
    ttft = [r["ttft_s"] for r in ret if r["ttft_s"] is not None]
    srv = [r["server"]["time_to_first_token_ms"] / 1e3 for r in ret if (r.get("server") or {}).get("time_to_first_token_ms")]
    summary = {
        "spec": os.path.basename(a.spec),
        "label": a.label,
        "errors": errors,
        "requests": len(records),
        "returning_requests": len(ret),
        "makespan_s": makespan,
        "returning_recomputed_tokens": sum(r["prompt_tokens"] - r["cached_tokens"] for r in ret),
        "returning_full_hits": sum(r["prompt_tokens"] - r["cached_tokens"] <= r["new_tokens"] for r in ret),
        "ttft_p50_s": percentile(ttft, 0.5),
        "ttft_p90_s": percentile(ttft, 0.9),
        "ttft_p99_s": percentile(ttft, 0.99),
        "ttft_mean_s": sum(ttft) / len(ttft) if ttft else None,
        "server_ttft_p50_s": percentile(srv, 0.5),
        "server_ttft_p99_s": percentile(srv, 0.99),
        "server_ttft_sum_s": sum(srv),
        "correctness_checked": len(corr),
        "correctness_identical": sum(r["identical"] for r in corr),
        "correctness_first_divergences": sorted(r["first_divergence"] for r in corr if r["first_divergence"] is not None),
    }
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary))
    return 1 if errors else 0


def gap(a: argparse.Namespace, cl: Client) -> int:
    os.makedirs(a.out, exist_ok=True)
    usable = cl.num_gpu_blocks() - 1
    session_blocks = a.length // BS
    filler_blocks = a.filler_tokens // BS
    rng = random.Random(a.seed)

    def toks(n: int) -> list[int]:
        return [rng.randrange(wl.TOKEN_LO, wl.TOKEN_HI) for _ in range(n)]

    rows = []
    with open(os.path.join(a.out, "gap.jsonl"), "w") as f:
        for rep in range(a.reps):
            for g in [int(x) for x in a.gaps.split(",")]:
                cl.reset_prefix_cache()
                session = toks(a.length)
                cl.complete(session, 1, stream=False)
                # A filler of n * BS - 1 tokens holds n blocks while running
                # (room for the async-scheduled next token) and leaves one
                # partial block, which is uncached, back in the free pool.
                free = usable - session_blocks
                while free > filler_blocks:
                    cl.complete(toks(filler_blocks * BS - 1), 1, stream=False)
                    free -= filler_blocks - 1
                n = free + g // BS
                if n > 0:
                    cl.complete(toks(n * BS - 1), 1, stream=False)
                res = cl.complete(session + toks(a.turn_tokens), 1)
                row = {
                    "rep": rep,
                    "gap_tokens": g,
                    "cached_tokens": res["cached_tokens"],
                    "prompt_tokens": res["prompt_tokens"],
                    "client_ttft_s": res["ttft_s"],
                    "server": res.get("metrics"),
                }
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(json.dumps(row), flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("run", "gap"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="himanshu-llmd-test")
    p.add_argument("--out", required=True)
    p.add_argument("--spec")
    p.add_argument("--label", default="")
    p.add_argument("--skip-correctness", action="store_true")
    p.add_argument(
        "--no-reset",
        action="store_true",
        help="skip the initial prefix-cache reset (use on a fresh server: a reset leaves "
        "uncached blocks at their old queue positions, behind cached ones)",
    )
    p.add_argument("--gaps", default="0,256,1024,4096,16384")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--length", type=int, default=32768)
    p.add_argument("--turn-tokens", type=int, default=512)
    p.add_argument("--filler-tokens", type=int, default=32768)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    cl = Client(a.host, a.port, a.model)
    return run(a, cl) if a.mode == "run" else gap(a, cl)


if __name__ == "__main__":
    sys.exit(main())
