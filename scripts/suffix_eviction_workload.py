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
"""Deterministic multi-turn session workloads for the suffix-eviction study.

Every session sends `turns` requests. Turn 0 is `initial_tokens` random token
IDs; turn k > 0 is the previous prompt, the previous turn's output tokens and
`user_tokens` new token IDs. Every request generates exactly `output_tokens`
tokens (ignore_eos). A session sends turn k + 1 `think_s[k]` seconds after
turn k finishes, so every session returns until its last turn.

Return-order patterns (the order in which idle sessions come back):
  cyclic  equal think times and staggered starts: sessions return in the order
          they went idle, so the oldest idle session is the next to return.
  random  exponential think times: return order independent of idle age.
  pareto  heavy-tailed think times (capped at 5x the mean): a session that has
          been idle long tends to stay idle, so the oldest returns last.

The same spec drives the CPU simulator and the live client. Only lengths
matter for cache behaviour, so the simulator's output tokens differ from the
model's without changing the eviction pattern.

Usage:
  suffix_eviction_workload.py --pattern cyclic --sessions 14 --out spec.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys

TOKEN_LO, TOKEN_HI = 1000, 100000


def tokens(seed: int, session: int, turn: int, n: int) -> list[int]:
    rng = random.Random(f"{seed}-{session}-{turn}")
    return [rng.randrange(TOKEN_LO, TOKEN_HI) for _ in range(n)]


def initial_prompt(spec: dict, session: dict) -> list[int]:
    return tokens(spec["seed"], session["id"], 0, session["initial_tokens"])


def next_prompt(spec: dict, session: dict, turn: int, prev_prompt: list[int], prev_output: list[int]) -> list[int]:
    return prev_prompt + prev_output + tokens(spec["seed"], session["id"], turn, spec["user_tokens"])


def prompt_len(spec: dict, session: dict, turn: int) -> int:
    return session["initial_tokens"] + turn * (spec["output_tokens"] + spec["user_tokens"])


def working_set_tokens(spec: dict) -> int:
    last = spec["turns"] - 1
    return sum(prompt_len(spec, s, last) + spec["output_tokens"] for s in spec["sessions"])


def is_sampled(spec: dict, session: int, turn: int) -> bool:
    """Requests replayed cold after the run to check output identity."""
    return turn > 0 and (session * spec["turns"] + turn) % spec["sample_every"] == 0


def build_spec(
    pattern: str,
    sessions: int,
    turns: int,
    initial_tokens: str,
    user_tokens: int,
    output_tokens: int,
    think_s: float,
    seed: int,
    sample_every: int,
) -> dict:
    rng = random.Random(f"spec-{seed}-{pattern}")
    len_rng = random.Random(f"len-{seed}")
    lo, _, hi = initial_tokens.partition("-")
    lo_n, hi_n = int(lo), int(hi or lo)
    out = []
    for sid in range(sessions):
        n0 = lo_n if lo_n == hi_n else len_rng.randrange(lo_n, hi_n + 1)
        n0 -= n0 % 16
        if pattern == "cyclic":
            start = sid * think_s / sessions
            thinks = [think_s] * (turns - 1)
        elif pattern == "random":
            start = rng.uniform(0, think_s)
            thinks = [rng.expovariate(1 / think_s) for _ in range(turns - 1)]
        elif pattern == "pareto":
            alpha = 1.5
            xm = think_s * (alpha - 1) / alpha
            start = rng.uniform(0, think_s)
            thinks = [min(xm * rng.paretovariate(alpha), 5 * think_s) for _ in range(turns - 1)]
        else:
            raise ValueError(pattern)
        out.append({"id": sid, "initial_tokens": n0, "start_s": round(start, 3), "think_s": [round(t, 3) for t in thinks]})
    return {
        "pattern": pattern,
        "seed": seed,
        "turns": turns,
        "user_tokens": user_tokens,
        "output_tokens": output_tokens,
        "sample_every": sample_every,
        "sessions": out,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", choices=("cyclic", "random", "pareto"), required=True)
    p.add_argument("--sessions", type=int, default=14)
    p.add_argument("--turns", type=int, default=6)
    p.add_argument("--initial-tokens", default="32768", help="N or LO-HI (uniform)")
    p.add_argument("--user-tokens", type=int, default=512)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--think-s", type=float, default=30.0, help="Mean think time")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sample-every", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    spec = build_spec(
        a.pattern, a.sessions, a.turns, a.initial_tokens, a.user_tokens, a.output_tokens, a.think_s, a.seed, a.sample_every
    )
    with open(a.out, "w") as f:
        json.dump(spec, f, indent=1)
    print(f"{a.out}: {len(spec['sessions'])} sessions, working set {working_set_tokens(spec)} tokens", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
