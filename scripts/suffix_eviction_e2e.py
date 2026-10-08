#!/usr/bin/env python3
# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Suffix-preserving eviction, Phase 2: two sessions under eviction pressure.

On one idle replica, per round:

  1. POST /reset_prefix_cache (needs VLLM_SERVER_DEV_MODE=1).
  2. Send sessions A and B, L tokens each (A first, so A is older).
  3. Send one filler prompt that leaves exactly L tokens of A and B cached.
  4. Re-send B + T new tokens, then A + T new tokens. Measure both.
  5. Reset, then send A + T and B + T cold as the recompute reference.

With vllm-native-lru, step 3 evicts all of A (Option 1): B returns fully cached
and A is recomputed. With VLLM_SUFFIX_PRESERVING_EVICTION=1 (fraction 0.5),
step 3 evicts the first half of both sessions (Option 3): each returns with
its second half cached and recomputes [0, L/2).

Reported per round: server prefill seconds (vllm:request_prefill_time_seconds
delta), usage cached_tokens, and the greedy tokens and first-token top-k
logprobs compared with the cold reference. Run once per arm with --label and
compare the JSON outputs.

Run inside the predictor (HTTP to 127.0.0.1:8000) with no other traffic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PREFILL_HIST = "vllm:request_prefill_time_seconds"
TTFT_HIST = "vllm:time_to_first_token_seconds"


def http(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    body: dict[str, Any] | None = None,
    timeout: float = 60,
) -> tuple[int, bytes]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def scrape(url: str, headers: dict[str, str]) -> tuple[dict[str, float], dict[str, str]]:
    code, raw = http("GET", url, headers=headers, timeout=20)
    if code != 200:
        raise RuntimeError(f"metrics HTTP {code}: {raw[:200]!r}")
    out: dict[str, float] = {}
    cache_info: dict[str, str] = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        labels: dict[str, str] = {}
        if "{" in line:
            name, rest = line.split("{", 1)
            labels_s, val_s = rest.rsplit("}", 1)
            for part in labels_s.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    labels[k.strip()] = v.strip().strip('"')
            value_tok = val_s.strip().split()[0]
        else:
            name, value_tok = line.rsplit(" ", 1)
        if name.endswith("_created"):
            continue
        if name == "vllm:cache_config_info":
            cache_info = labels
            continue
        try:
            val = float(value_tok)
        except ValueError:
            continue
        key = name.removesuffix("_total")
        out[key] = out.get(key, 0.0) + val
    return out, cache_info


class E2E:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        lo, hi = (int(x) for x in args.token_id_range.split(","))
        self.token_lo, self.token_hi = lo, hi
        self.rng = random.Random(args.seed)

    def metrics(self) -> dict[str, float]:
        return scrape(self.args.metrics_url, self.headers)[0]

    def wait_idle(self, timeout_s: float = 120) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            m = self.metrics()
            if not m.get("vllm:num_requests_running") and not m.get("vllm:num_requests_waiting"):
                return
            time.sleep(0.2)
        raise SystemExit("engine did not become idle; isolate this replica")

    def reset_cache(self) -> None:
        self.wait_idle()
        for _ in range(20):
            code, raw = http("POST", self.args.reset_url, headers=self.headers, timeout=60)
            if code == 404:
                raise SystemExit("/reset_prefix_cache returned 404; set VLLM_SERVER_DEV_MODE=1")
            if code == 200 and b"false" not in raw.lower():
                return
            time.sleep(0.5)
        raise SystemExit(f"/reset_prefix_cache kept failing: HTTP {code} {raw[:200]!r}")

    def tokens(self, n: int) -> list[int]:
        return [self.rng.randrange(self.token_lo, self.token_hi) for _ in range(n)]

    def send(self, prompt: list[int], max_tokens: int = 1) -> dict[str, Any]:
        self.wait_idle()
        before = self.metrics()
        body = {
            "model": self.args.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "logprobs": self.args.top_logprobs,
            "return_tokens_as_token_ids": True,
        }
        t0 = time.perf_counter()
        code, raw = http(
            "POST",
            self.args.base_url.rstrip("/") + "/completions",
            headers=self.headers,
            body=body,
            timeout=self.args.timeout,
        )
        e2e = time.perf_counter() - t0
        after = self.metrics()
        if code != 200:
            raise SystemExit(f"completion HTTP {code}: {raw[:300]!r}")
        resp = json.loads(raw)
        usage = resp.get("usage") or {}
        choice = resp["choices"][0]
        lp = choice.get("logprobs") or {}
        top = (lp.get("top_logprobs") or [None])[0] or {}
        return {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "cached_tokens": int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
            "prefill_s": after.get(f"{PREFILL_HIST}_sum", 0) - before.get(f"{PREFILL_HIST}_sum", 0),
            "ttft_s": after.get(f"{TTFT_HIST}_sum", 0) - before.get(f"{TTFT_HIST}_sum", 0),
            "ttft_count": after.get(f"{TTFT_HIST}_count", 0) - before.get(f"{TTFT_HIST}_count", 0),
            "e2e_s": e2e,
            "output_tokens": lp.get("tokens") or [],
            "first_top_logprobs": top,
        }

    def preflight(self) -> tuple[int, int]:
        code, raw = http("GET", self.args.base_url.rstrip("/") + "/models", headers=self.headers)
        if code != 200:
            raise SystemExit(f"models HTTP {code}: {raw[:300]!r}")
        max_model_len = 0
        for entry in json.loads(raw).get("data") or []:
            if entry.get("id") == self.args.model:
                max_model_len = int(entry.get("max_model_len") or 0)
        if not max_model_len:
            raise SystemExit(f"model {self.args.model!r} not served")
        _, cache_info = scrape(self.args.metrics_url, self.headers)
        if not cache_info.get("num_gpu_blocks"):
            raise SystemExit("vllm:cache_config_info has no num_gpu_blocks")
        # One block is the null block and never holds KV.
        usable_blocks = int(cache_info["num_gpu_blocks"]) - 1
        return max_model_len, usable_blocks

    def round(self, idx: int, bs: int, session: int, filler: int, turn: int) -> dict[str, Any]:
        a, b = self.tokens(session), self.tokens(session)
        ta, tb = self.tokens(turn), self.tokens(turn)
        gen = self.args.max_tokens

        self.reset_cache()
        self.send(a)
        self.send(b)
        filler_rec = self.send(self.tokens(filler))
        b_ret = self.send(b + tb, gen)
        a_ret = self.send(a + ta, gen)

        self.reset_cache()
        a_cold = self.send(a + ta, gen)
        b_cold = self.send(b + tb, gen)

        def compare(ret: dict[str, Any], cold: dict[str, Any]) -> dict[str, Any]:
            shared = set(ret["first_top_logprobs"]) & set(cold["first_top_logprobs"])
            diffs = [abs(ret["first_top_logprobs"][t] - cold["first_top_logprobs"][t]) for t in shared]
            n = min(len(ret["output_tokens"]), len(cold["output_tokens"]))
            prefix_match = next(
                (i for i in range(n) if ret["output_tokens"][i] != cold["output_tokens"][i]), n
            )
            return {
                "greedy_identical": ret["output_tokens"] == cold["output_tokens"],
                "greedy_matching_prefix": prefix_match,
                "first_token_top_logprob_max_abs_diff": max(diffs) if diffs else None,
                "first_token_top_shared": len(shared),
            }

        rec = {
            "round": idx,
            "label": self.args.label,
            "session_tokens": session,
            "filler_tokens": filler,
            "turn_tokens": turn,
            "filler_prefill_s": filler_rec["prefill_s"],
            "returning": {"A": a_ret, "B": b_ret},
            "cold": {"A": a_cold, "B": b_cold},
            "vs_cold": {"A": compare(a_ret, a_cold), "B": compare(b_ret, b_cold)},
            "returning_prefill_s": a_ret["prefill_s"] + b_ret["prefill_s"],
            "cold_prefill_s": a_cold["prefill_s"] + b_cold["prefill_s"],
        }
        clean = all(r["ttft_count"] == 1 for r in (a_ret, b_ret, a_cold, b_cold, filler_rec))
        rec["clean"] = clean
        return rec

    def run(self) -> int:
        bs = self.args.block_size
        max_model_len, usable_blocks = self.preflight()
        session = self.args.session_tokens - self.args.session_tokens % (2 * bs)
        turn = self.args.turn_tokens
        filler = (usable_blocks - session // bs) * bs
        problems = []
        if 2 * session > usable_blocks * bs:
            problems.append(f"2L={2 * session} exceeds KV capacity {usable_blocks * bs}")
        if filler + 1 > max_model_len:
            problems.append(
                f"filler {filler} exceeds max_model_len {max_model_len}; "
                f"raise --session-tokens to >= {usable_blocks * bs - max_model_len + bs}"
            )
        if session + turn + self.args.max_tokens > max_model_len:
            problems.append("session + turn + max_tokens exceeds max_model_len")
        if problems:
            raise SystemExit("; ".join(problems))
        print(
            f"label={self.args.label} usable_blocks={usable_blocks} L={session} "
            f"filler={filler} turn={turn} rounds={self.args.rounds}"
        )

        out = Path(self.args.out or f"reports/suffix-eviction-e2e-{self.args.label}-{time.strftime('%Y%m%d-%H%M%S')}")
        out.mkdir(parents=True, exist_ok=True)
        rounds = []
        with (out / "rounds.jsonl").open("a") as f:
            for i in range(self.args.rounds):
                rec = self.round(i, bs, session, filler, turn)
                rounds.append(rec)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                r, c, v = rec["returning"], rec["cold"], rec["vs_cold"]
                print(
                    f"round {i}: returning prefill A={r['A']['prefill_s']:.3f}s "
                    f"(cached {r['A']['cached_tokens']}) B={r['B']['prefill_s']:.3f}s "
                    f"(cached {r['B']['cached_tokens']}) total={rec['returning_prefill_s']:.3f}s | "
                    f"cold total={rec['cold_prefill_s']:.3f}s | "
                    f"greedy identical A={v['A']['greedy_identical']} B={v['B']['greedy_identical']} "
                    f"max|dlogprob| A={v['A']['first_token_top_logprob_max_abs_diff']} "
                    f"B={v['B']['first_token_top_logprob_max_abs_diff']}"
                    + ("" if rec["clean"] else " [other traffic seen]")
                )

        used = [r for r in rounds if r["clean"]] or rounds
        summary = {
            "label": self.args.label,
            "session_tokens": session,
            "turn_tokens": turn,
            "rounds": len(rounds),
            "clean_rounds": sum(r["clean"] for r in rounds),
            "median_returning_prefill_s": statistics.median(r["returning_prefill_s"] for r in used),
            "median_cold_prefill_s": statistics.median(r["cold_prefill_s"] for r in used),
            "median_cached_tokens": {
                k: statistics.median(r["returning"][k]["cached_tokens"] for r in used) for k in "AB"
            },
            "greedy_identical_fraction": {
                k: sum(r["vs_cold"][k]["greedy_identical"] for r in used) / len(used) for k in "AB"
            },
            "max_first_token_logprob_diff": max(
                (r["vs_cold"][k]["first_token_top_logprob_max_abs_diff"] or 0.0)
                for r in used
                for k in "AB"
            ),
        }
        summary["returning_over_cold"] = (
            summary["median_returning_prefill_s"] / summary["median_cold_prefill_s"]
            if summary["median_cold_prefill_s"] > 0
            else math.nan
        )
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        print(f"Wrote {out}")
        return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", required=True, help="Arm name recorded in the output, e.g. treatment")
    p.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument("--metrics-url", default=os.environ.get("METRICS_URL", "http://127.0.0.1:8000/metrics"))
    p.add_argument("--reset-url", default=os.environ.get("RESET_URL", "http://127.0.0.1:8000/reset_prefix_cache"))
    p.add_argument("--model", default=os.environ.get("MODEL", "himanshu-llmd-test"))
    p.add_argument("--session-tokens", type=int, default=65536, help="L, tokens per session")
    p.add_argument("--turn-tokens", type=int, default=256, help="New tokens appended on return")
    p.add_argument("--max-tokens", type=int, default=16, help="Greedy tokens generated on return")
    p.add_argument("--top-logprobs", type=int, default=5)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--token-id-range", default="1000,100000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--out", default=None)
    return E2E(p.parse_args(argv)).run()


if __name__ == "__main__":
    sys.exit(main())
