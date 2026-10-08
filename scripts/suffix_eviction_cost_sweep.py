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
"""Suffix-preserving eviction, Phase 1: recompute cost of prefix vs suffix.

A session holds 2M tokens. Two sessions must give up 2M tokens of KV cache:

  Option 1  evict one whole session     -> recompute [0, 2M) once
  Option 2  evict the last M of each    -> recompute [M, 2M) twice
  Option 3  evict the first M of each   -> recompute [0, M) twice

Each option is measured on one idle vLLM replica with the exact cache state it
implies, using random token-ID prompts so lengths are exact and nothing hits
the cache by accident:

  full    fresh 2M-token prompt, nothing cached             (Option 1, evicted session)
  hit     the same 2M-token prompt again, fully cached      (Option 1, intact session)
  suffix  M-token prompt P cached first, then P + S sent    (Option 2, per session)
  prefix  fresh M-token prompt                              (Option 3, per session)

Option 1 = full + hit, Option 2 = 2 * suffix, Option 3 = 2 * prefix, so every
option pays for two returning requests. "prefix" is exact for Option 3: tokens
[0, M) attend only to [0, M), and with the suffix still cached nothing after M
is computed.

Timing is the server-side prefill time (vllm:request_prefill_time_seconds
delta), which excludes queueing and HTTP. The cache state is checked with the
vllm:prefix_cache_hits delta; samples with the wrong hit count are rejected.

Attention-only theory predicts Option1/Option3 = 2 and Option2/Option3 = 3.
All options compute 2M new tokens, so the per-token (MLP / projection) cost is
equal across them, and the ratios approach 1 where that cost dominates. A fit
of prefill_s = a + b*new_tokens + c*(end^2 - start^2) over all samples reports
the context length where attention starts to dominate.

Run inside the predictor (HTTP to 127.0.0.1:8000) with CPU offload disabled
and no other traffic. Requires only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "himanshu-llmd-test"
BLOCK_SIZE = 16
DEFAULT_M_GRID = "1024,2048,4096,8192,16384,32768,65536,98304,122880"
KINDS = ("full", "hit", "suffix", "prefix")
PREFILL_HIST = "vllm:request_prefill_time_seconds"
TTFT_HIST = "vllm:time_to_first_token_seconds"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1"))
    p.add_argument(
        "--metrics-url",
        default=os.environ.get("METRICS_URL", "http://127.0.0.1:8000/metrics"),
    )
    p.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    p.add_argument(
        "--m-grid",
        default=DEFAULT_M_GRID,
        help="Comma-separated M values (tokens per half). Rounded down to a block multiple.",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    p.add_argument(
        "--token-id-range",
        default="1000,100000",
        help="Random prompt token IDs are drawn from [lo, hi). Keep clear of special tokens.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--out", default=None, help="Output directory (default: reports/suffix-eviction-cost-<ts>)")
    p.add_argument(
        "--summarize",
        default=None,
        help="Skip measuring; summarize an existing samples.jsonl",
    )
    return p.parse_args(argv)


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


def parse_prom_line(line: str) -> tuple[str, dict[str, str], float] | None:
    if not line or line.startswith("#"):
        return None
    labels: dict[str, str] = {}
    if "{" in line:
        name, rest = line.split("{", 1)
        labels_s, val_s = rest.rsplit("}", 1)
        for part in labels_s.split(","):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            labels[k.strip()] = v.strip().strip('"')
        value_tok = val_s.strip().split()[0]
    else:
        name, value_tok = line.rsplit(" ", 1)
    try:
        return name, labels, float(value_tok)
    except ValueError:
        return None


def scrape(url: str, headers: dict[str, str]) -> tuple[dict[str, float], dict[str, str]]:
    """Return summed metric values and the cache_config_info labels."""
    code, raw = http("GET", url, headers=headers, timeout=20)
    if code != 200:
        raise RuntimeError(f"metrics HTTP {code}: {raw[:200]!r}")
    out: dict[str, float] = {}
    cache_info: dict[str, str] = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        parsed = parse_prom_line(line)
        if parsed is None:
            continue
        name, labels, val = parsed
        if name.endswith("_created"):
            continue
        if name == "vllm:cache_config_info":
            cache_info = labels
            continue
        key = name.removesuffix("_total")
        out[key] = out.get(key, 0.0) + val
    return out, cache_info


class Sweep:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        lo, hi = (int(x) for x in args.token_id_range.split(","))
        self.token_lo, self.token_hi = lo, hi
        self.rng = random.Random(args.seed)
        self.out = Path(args.out or f"reports/suffix-eviction-cost-{time.strftime('%Y%m%d-%H%M%S')}")
        self.out.mkdir(parents=True, exist_ok=True)
        self.samples_path = self.out / "samples.jsonl"
        self.max_model_len = 0
        self.kv_capacity_tokens: int | None = None
        self.last_full: list[int] = []

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

    def tokens(self, n: int) -> list[int]:
        return [self.rng.randrange(self.token_lo, self.token_hi) for _ in range(n)]

    def complete(self, prompt: list[int]) -> dict[str, Any]:
        body = {
            "model": self.args.model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
        }
        t0 = time.perf_counter()
        code, raw = http(
            "POST",
            self.args.base_url.rstrip("/") + "/completions",
            headers=self.headers,
            body=body,
            timeout=self.args.timeout,
        )
        rec: dict[str, Any] = {"e2e_s": time.perf_counter() - t0, "http_status": code}
        if code != 200:
            rec["error"] = f"HTTP {code}: {raw[:300]!r}"
            return rec
        usage = json.loads(raw).get("usage") or {}
        rec["prompt_tokens"] = int(usage.get("prompt_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        if "cached_tokens" in details:
            rec["usage_cached_tokens"] = int(details["cached_tokens"] or 0)
        return rec

    def probe(self, prompt: list[int]) -> dict[str, Any]:
        self.wait_idle()
        before = self.metrics()
        rec = self.complete(prompt)
        after = self.metrics()

        def delta(key: str) -> float:
            return after.get(key, 0.0) - before.get(key, 0.0)

        rec.update(
            {
                "prefill_s": delta(f"{PREFILL_HIST}_sum"),
                "ttft_s": delta(f"{TTFT_HIST}_sum"),
                "ttft_count": delta(f"{TTFT_HIST}_count"),
                "cache_hits": delta("vllm:prefix_cache_hits"),
                "external_hits": delta("vllm:external_prefix_cache_hits"),
            }
        )
        return rec

    def measure(self, kind: str, m: int, repeat: int) -> dict[str, Any]:
        bs = self.args.block_size
        if kind == "full":
            self.last_full = self.tokens(2 * m)
            prompt, start, expect_hits = self.last_full, 0, 0
        elif kind == "hit":
            # The last token is always recomputed, so its block is too.
            prompt, start = self.last_full, 2 * m - bs
            expect_hits = start
        elif kind == "prefix":
            prompt, start, expect_hits = self.tokens(m), 0, 0
        else:
            prefix = self.tokens(m)
            setup = self.probe(prefix)
            if setup.get("error"):
                return {"kind": kind, "m": m, "repeat": repeat, "error": f"setup: {setup['error']}"}
            prompt, start, expect_hits = prefix + self.tokens(m), m, m
        rec = self.probe(prompt)
        rec.update(
            {
                "kind": kind,
                "m": m,
                "repeat": repeat,
                "start": start,
                "end": len(prompt),
                "new_tokens": len(prompt) - start,
                "expect_hits": expect_hits,
            }
        )
        if not rec.get("error"):
            problems = []
            if rec["ttft_count"] != 1:
                problems.append(f"ttft_count={rec['ttft_count']:g} (other traffic?)")
            if rec["cache_hits"] != expect_hits or rec["external_hits"]:
                problems.append(
                    f"cache_hits={rec['cache_hits']:g} external_hits={rec['external_hits']:g} "
                    f"expected {expect_hits} local"
                )
            if problems:
                rec["error"] = "; ".join(problems)
        return rec

    def preflight(self) -> list[int]:
        code, raw = http("GET", self.args.base_url.rstrip("/") + "/models", headers=self.headers)
        if code != 200:
            raise SystemExit(f"models HTTP {code}: {raw[:300]!r}")
        models = json.loads(raw).get("data") or []
        for entry in models:
            if entry.get("id") == self.args.model:
                self.max_model_len = int(entry.get("max_model_len") or 0)
        if not self.max_model_len:
            raise SystemExit(f"model {self.args.model!r} not served; have {[e.get('id') for e in models]}")
        _, cache_info = scrape(self.args.metrics_url, self.headers)
        if cache_info.get("num_gpu_blocks"):
            self.kv_capacity_tokens = int(cache_info["num_gpu_blocks"]) * self.args.block_size
        if cache_info.get("enable_prefix_caching", "True") != "True":
            raise SystemExit("prefix caching is disabled on the server")
        self.wait_idle(30)

        bs = self.args.block_size
        grid = []
        for m in (int(x) for x in self.args.m_grid.split(",") if x.strip()):
            m -= m % bs
            if m <= 0:
                continue
            if 2 * m + 1 > self.max_model_len:
                print(f"skip M={m}: 2M+1 exceeds max_model_len={self.max_model_len}")
                continue
            if self.kv_capacity_tokens is not None and 2 * m > self.kv_capacity_tokens:
                print(f"skip M={m}: 2M exceeds KV capacity of {self.kv_capacity_tokens} tokens")
                continue
            grid.append(m)
        print(
            f"max_model_len={self.max_model_len} kv_capacity_tokens={self.kv_capacity_tokens} "
            f"grid={grid} repeats={self.args.repeats}"
        )
        return grid

    def run(self) -> int:
        grid = self.preflight()
        (self.out / "config.json").write_text(
            json.dumps(
                {
                    **vars(self.args),
                    "grid": grid,
                    "max_model_len": self.max_model_len,
                    "kv_capacity_tokens": self.kv_capacity_tokens,
                },
                indent=2,
            )
        )
        for n in (512, 2048, 8192):
            if n < self.max_model_len:
                self.probe(self.tokens(n))
        with self.samples_path.open("a") as f:
            for m in grid:
                for repeat in range(self.args.repeats):
                    for kind in KINDS:
                        rec = self.measure(kind, m, repeat)
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        status = rec.get("error") or f"prefill {rec['prefill_s']:.4f}s"
                        print(f"M={m:>6} rep={repeat} {kind:<6} {status}")
        summarize(self.samples_path, self.out, self.args.block_size)
        return 0


def solve3(a: list[list[float]], b: list[float]) -> list[float] | None:
    m = [row[:] + [v] for row, v in zip(a, b)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-30:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(3):
            if r != col:
                f = m[r][col] / m[col][col]
                for c in range(col, 4):
                    m[r][c] -= f * m[col][c]
    return [m[i][3] / m[i][i] for i in range(3)]


def fit_cost_model(samples: list[dict[str, Any]]) -> dict[str, float] | None:
    """Least squares for prefill_s = a + b*new_tokens + c*(end^2 - start^2).

    Columns are scaled before solving so the normal equations stay well
    conditioned across 1k..250k-token prompts.
    """
    rows = [
        (1.0, float(s["new_tokens"]), float(s["end"] ** 2 - s["start"] ** 2), s["prefill_s"])
        for s in samples
    ]
    if len(rows) < 3:
        return None
    scale = [max(abs(r[i]) for r in rows) or 1.0 for i in range(3)]
    xs = [[r[i] / scale[i] for i in range(3)] for r in rows]
    ys = [r[3] for r in rows]
    ata = [[sum(x[i] * x[j] for x in xs) for j in range(3)] for i in range(3)]
    aty = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(3)]
    coef = solve3(ata, aty)
    if coef is None:
        return None
    a, b, c = (coef[i] / scale[i] for i in range(3))
    out = {"a_s": a, "b_s_per_token": b, "c_s_per_token_pair": c}
    if c > 0:
        # Marginal cost of token t is b + 2*c*t; attention dominates past b / (2c).
        out["attention_dominates_after_tokens"] = b / (2 * c)
    return out


def summarize(samples_path: Path, out: Path, block_size: int) -> dict[str, Any]:
    samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
    good = [s for s in samples if not s.get("error")]
    rejected = len(samples) - len(good)
    by: dict[tuple[int, str], list[float]] = {}
    for s in good:
        by.setdefault((s["m"], s["kind"]), []).append(s["prefill_s"])

    fit = fit_cost_model(good)
    rows = []
    for m in sorted({s["m"] for s in good}):
        med = {k: statistics.median(by[(m, k)]) for k in KINDS if (m, k) in by}
        row: dict[str, Any] = {"m": m, **{f"{k}_s": v for k, v in med.items()}}
        if len(med) == len(KINDS):
            opt1 = med["full"] + med["hit"]
            opt2, opt3 = 2 * med["suffix"], 2 * med["prefix"]
            row.update(
                {
                    "option1_s": opt1,
                    "option2_s": opt2,
                    "option3_s": opt3,
                    "opt1_over_opt3": opt1 / opt3,
                    "opt2_over_opt3": opt2 / opt3,
                }
            )
        if fit:
            a, b, c = fit["a_s"], fit["b_s_per_token"], fit["c_s_per_token_pair"]

            def cost(start: int, end: int) -> float:
                return a + b * (end - start) + c * (end * end - start * start)

            p1 = cost(0, 2 * m) + cost(2 * m - block_size, 2 * m)
            p2, p3 = 2 * cost(m, 2 * m), 2 * cost(0, m)
            row.update({"fit_opt1_over_opt3": p1 / p3, "fit_opt2_over_opt3": p2 / p3})
        rows.append(row)

    summary = {"samples": len(samples), "rejected": rejected, "fit": fit, "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "summary.csv").open("w") as f:
        cols = [
            "m", "full_s", "hit_s", "suffix_s", "prefix_s", "option1_s", "option2_s", "option3_s",
            "opt1_over_opt3", "opt2_over_opt3", "fit_opt1_over_opt3", "fit_opt2_over_opt3",
        ]  # fmt: skip
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(c) is None else f"{r[c]:.6g}" for c in cols) + "\n")

    print(f"\n{len(good)} samples used, {rejected} rejected. Median server prefill seconds:")
    print(
        f"{'M':>7} {'full[0,2M)':>11} {'hit':>8} {'suffix[M,2M)':>13} {'prefix[0,M)':>12} "
        f"{'Opt1/Opt3':>10} {'Opt2/Opt3':>10} {'fit 1/3':>8} {'fit 2/3':>8}"
    )

    def fmt(v: float | None, spec: str) -> str:
        return "-" if v is None else format(v, spec)

    for r in rows:
        print(
            f"{r['m']:>7} {fmt(r.get('full_s'), '11.4f')} {fmt(r.get('hit_s'), '8.4f')} "
            f"{fmt(r.get('suffix_s'), '13.4f')} "
            f"{fmt(r.get('prefix_s'), '12.4f')} {fmt(r.get('opt1_over_opt3'), '10.2f')} "
            f"{fmt(r.get('opt2_over_opt3'), '10.2f')} {fmt(r.get('fit_opt1_over_opt3'), '8.2f')} "
            f"{fmt(r.get('fit_opt2_over_opt3'), '8.2f')}"
        )
    print("Attention-only theory: Opt1/Opt3 = 2.00, Opt2/Opt3 = 3.00")
    if fit:
        print(
            f"Fit: a={fit['a_s']:.4g}s b={fit['b_s_per_token']:.4g}s/token "
            f"c={fit['c_s_per_token_pair']:.4g}s/token^2"
        )
        if "attention_dominates_after_tokens" in fit:
            print(f"Attention cost per token exceeds linear cost after ~{fit['attention_dominates_after_tokens']:.0f} tokens")
    print(f"Wrote {out}/summary.json and summary.csv")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summarize:
        path = Path(args.summarize)
        summarize(path, Path(args.out) if args.out else path.parent, args.block_size)
        return 0
    return Sweep(args).run()


if __name__ == "__main__":
    sys.exit(main())
