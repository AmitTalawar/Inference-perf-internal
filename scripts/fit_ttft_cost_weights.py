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
"""Fit Method IIB ttft-cost-scorer weights from Method I ALL p50 rows.

Fits token-space surfaces on isolated cold / GPU / CPU TTFT, then converts
to per-block coefficients for:

  cost = w_gpu*m_gpu + w_cpu*m_cpu + w1*(N-m) + w2*(N^2-m^2)

where N, m_* are KV blocks (N = n_tokens / block_size).

  T_cold(n) ≈ a + b*n + c*n^2
  T_gpu(n)  ≈ a_g + g*n
  T_xfer(n) = T_cpu(n) - T_gpu(n) ≈ d*n

  unmatchedLinear (w1)     = b * block_size
  unmatchedQuadratic (w2)  = c * block_size^2
  gpu (w_gpu)              = g * block_size
  cpu (w_cpu)              = d * block_size

Example:

  python3 scripts/fit_ttft_cost_weights.py \\
    --weights-csv reports/calibrate-cpu-weight-20260828-100914/weights.csv \\
    --block-size 16
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def load_all_rows(path: Path, pod: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if r.get("pod") != pod:
                continue
            try:
                n = float(r["n"])
                cold = float(r["ttft_cold_p50"])
                gpu = float(r["ttft_gpu_p50"])
                cpu = float(r["ttft_cpu_p50"])
            except (KeyError, TypeError, ValueError) as e:
                raise SystemExit(f"bad row in {path}: {r!r}: {e}") from e
            if n <= 0 or cold <= 0:
                continue
            rows.append({"n": n, "cold": cold, "gpu": gpu, "cpu": cpu})
    if len(rows) < 3:
        raise SystemExit(f"need >= 3 rows for pod={pod!r} in {path}, got {len(rows)}")
    rows.sort(key=lambda x: x["n"])
    return rows


def solve_normal(xtx: list[list[float]], xty: list[float]) -> list[float]:
    """Solve X^T X beta = X^T y via Gaussian elimination with partial pivoting."""
    n = len(xty)
    a = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-18:
            raise SystemExit("singular design matrix; check input lengths")
        a[col], a[piv] = a[piv], a[col]
        div = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            for j in range(col, n + 1):
                a[r][j] -= factor * a[col][j]
    return [a[i][n] for i in range(n)]


def fit_poly(xs: list[float], ys: list[float], degree: int) -> list[float]:
    """Least-squares polynomial coeffs [a0, a1, ..., adegree]."""
    m = degree + 1
    xtx = [[0.0] * m for _ in range(m)]
    xty = [0.0] * m
    for x, y in zip(xs, ys):
        powers = [1.0]
        for _ in range(degree):
            powers.append(powers[-1] * x)
        for i in range(m):
            xty[i] += powers[i] * y
            for j in range(m):
                xtx[i][j] += powers[i] * powers[j]
    return solve_normal(xtx, xty)


def fit_through_origin(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope for y ≈ d * x (no intercept)."""
    num = sum(x * y for x, y in zip(xs, ys))
    den = sum(x * x for x in xs)
    if den <= 0:
        return 0.0
    return num / den


def rmse(xs: list[float], ys: list[float], predict) -> float:
    if not xs:
        return 0.0
    err = [(predict(x) - y) ** 2 for x, y in zip(xs, ys)]
    return math.sqrt(sum(err) / len(err))


def cost_seconds(w_gpu: float, w_cpu: float, w1: float, w2: float,
                 n_blocks: int, m_gpu: int, m_cpu: int) -> float:
    m = m_gpu + m_cpu
    if m > n_blocks:
        m = n_blocks
    unmatched = n_blocks - m
    return (
        w_gpu * m_gpu
        + w_cpu * m_cpu
        + w1 * unmatched
        + w2 * (n_blocks * n_blocks - m * m)
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--weights-csv",
        type=Path,
        required=True,
        help="Method I weights.csv (expects pod,n,ttft_{cold,gpu,cpu}_p50)",
    )
    ap.add_argument("--pod", default="ALL", help="Which pod rows to fit (default ALL)")
    ap.add_argument("--block-size", type=int, default=16, help="KV block size in tokens")
    ap.add_argument(
        "--min-n",
        type=float,
        default=1024,
        help="Drop short-n points dominated by fixed overhead (default 1024)",
    )
    ap.add_argument(
        "--allow-gpu-nonzero",
        action="store_true",
        help="Keep fitted w_gpu instead of clamping to 0 (default clamps w_gpu=0)",
    )
    args = ap.parse_args()
    if args.block_size <= 0:
        raise SystemExit("--block-size must be > 0")
    force_gpu_zero = not args.allow_gpu_nonzero

    rows = [r for r in load_all_rows(args.weights_csv, args.pod) if r["n"] >= args.min_n]
    if len(rows) < 3:
        raise SystemExit(f"need >= 3 rows with n >= {args.min_n}, got {len(rows)}")

    ns = [r["n"] for r in rows]
    cold = [r["cold"] for r in rows]
    gpu = [r["gpu"] for r in rows]
    cpu = [r["cpu"] for r in rows]
    xfer = [c - g for c, g in zip(cpu, gpu)]

    a, b, c = fit_poly(ns, cold, degree=2)
    a_g, g = fit_poly(ns, gpu, degree=1)
    d = fit_through_origin(ns, xfer)

    b = max(0.0, b)
    c = max(0.0, c)
    g = max(0.0, g)
    d = max(0.0, d)

    bs = float(args.block_size)
    w1 = b * bs
    w2 = c * (bs * bs)
    w_gpu = 0.0 if force_gpu_zero else g * bs
    w_cpu = d * bs

    cold_rmse = rmse(ns, cold, lambda n: a + b * n + c * n * n)
    gpu_rmse = rmse(ns, gpu, lambda n: a_g + g * n)
    xfer_rmse = rmse(ns, xfer, lambda n: d * n)

    # Gate: N=98304 tokens, 96000 GPU vs full CPU.
    n_tok, gpu_tok = 98304.0, 96000.0
    n_b = int(n_tok / bs)
    m_gpu_b = int(gpu_tok / bs)
    cost_gpu_tail = cost_seconds(w_gpu, w_cpu, w1, w2, n_b, m_gpu_b, 0)
    cost_cpu_full = cost_seconds(w_gpu, w_cpu, w1, w2, n_b, 0, n_b)
    gate_ok = cost_cpu_full < cost_gpu_tail

    print(f"# fit from {args.weights_csv} pod={args.pod} min_n={args.min_n} block_size={args.block_size}")
    print(f"# points={len(rows)} n=[{ns[0]:.0f},{ns[-1]:.0f}]")
    print(f"# T_cold ≈ {a:.6g} + {b:.6g}*n + {c:.6g}*n^2   rmse={cold_rmse:.4g}s")
    print(f"# T_gpu  ≈ {a_g:.6g} + {g:.6g}*n               rmse={gpu_rmse:.4g}s")
    print(f"# T_xfer ≈ {d:.6g}*n                          rmse={xfer_rmse:.4g}s")
    print(f"# gate 96k-GPU vs full-CPU: cost_gpu_tail={cost_gpu_tail:.6g}s "
          f"cost_cpu_full={cost_cpu_full:.6g}s  {'PASS' if gate_ok else 'FAIL'}")
    print()
    print("# Paste into ttft-cost-scorer parameters:")
    print("mode: offline")
    print("weights:")
    print(f"  gpu: {w_gpu:.8g}")
    print(f"  cpu: {w_cpu:.8g}")
    print(f"  unmatchedLinear: {w1:.8g}")
    print(f"  unmatchedQuadratic: {w2:.8g}")

    if not gate_ok:
        print(
            "\n# WARNING: fitted weights fail the 96k-GPU vs full-CPU gate; "
            "check min-n / calibration table.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
