#!/usr/bin/env python3
"""Compare live suffix-eviction arms with the CPU simulator.

Usage:
  suffix_eviction_compare.py --live reports/suffix-eviction-live-20260928 \
      --sim reports/suffix-eviction-sim-20260928/sweep-default
"""

from __future__ import annotations

import argparse
import json
import os


def load(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--live", required=True)
    p.add_argument("--sim", required=True)
    p.add_argument("--arms", default="vllm-native-lru,vllm-prefix-pruning-262144,fraction-0.5,vllm-prefix-pruning-16384,vllm-prefix-pruning-4096,vllm-prefix-pruning-512")
    p.add_argument("--specs", default="eq32k-s14-cyclic,eq32k-s14-random,eq32k-s14-pareto")
    a = p.parse_args()
    rows = []
    for spec in a.specs.split(","):
        print(f"\n{spec}")
        print(
            f"  {'arm':14s} {'recomputed live/sim (vs native)':>36s}  {'p50 live/sim':>13s}  {'p99 live/sim':>13s}"
            f"  {'mean':>5s}  {'srv-sum':>7s}  {'identical':>9s}"
        )
        base = None
        for arm in a.arms.split(","):
            live = load(os.path.join(a.live, arm, spec, "summary.json"))
            sim = load(os.path.join(a.sim, f"{spec}__{arm}", "summary.json"))
            if live is None:
                continue
            base = base or live
            ratio = live["returning_recomputed_tokens"] / base["returning_recomputed_tokens"]
            sim_rec = sim["returning_recomputed_tokens"] if sim else float("nan")
            print(
                f"  {arm:14s} {live['returning_recomputed_tokens']:9d} / {sim_rec:9.0f} ({ratio:4.2f}x)       "
                f"  {live['ttft_p50_s']:5.2f}/{sim['ttft_p50_s'] if sim else float('nan'):5.2f}"
                f"  {live['ttft_p99_s']:5.2f}/{sim['ttft_p99_s'] if sim else float('nan'):5.2f}"
                f"  {live['ttft_mean_s']:5.2f}  {live['server_ttft_sum_s']:7.1f}"
                f"  {live['correctness_identical']}/{live['correctness_checked']}"
                f" {live['correctness_first_divergences']}"
            )
            rows.append({"spec": spec, "arm": arm, "live": live, "sim": sim})
    with open(os.path.join(a.live, "comparison.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
