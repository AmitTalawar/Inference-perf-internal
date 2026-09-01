#!/usr/bin/env python3
"""Plot stacked percentile bar graphs across block sizes.

Example:
  python scripts/plot_block_size_percentiles.py --metrics ttft itl request_latency
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print(
        "[error] matplotlib is required. Install with: pip install matplotlib",
        file=sys.stderr,
    )
    sys.exit(1)


PERCENTILES = ("median", "p90", "p95", "p99")

# Map shorthand and canonical names to JSON paths.
METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "ttft": ("successes", "latency", "time_to_first_token"),
    "time_to_first_token": ("successes", "latency", "time_to_first_token"),
    "itl": ("successes", "latency", "inter_token_latency"),
    "inter_token_latency": ("successes", "latency", "inter_token_latency"),
    "tpot": ("successes", "latency", "time_per_output_token"),
    "time_per_output_token": ("successes", "latency", "time_per_output_token"),
    "ntpot": ("successes", "latency", "normalized_time_per_output_token"),
    "normalized_time_per_output_token": (
        "successes",
        "latency",
        "normalized_time_per_output_token",
    ),
    "request_latency": ("successes", "latency", "request_latency"),
    "schedule_delay": ("load_summary", "schedule_delay"),
}

PERCENTILE_COLORS = {
    "median": "#1f77b4",  # blue
    "p90": "#ff7f0e",  # orange
    "p95": "#2ca02c",  # green
    "p99": "#d62728",  # red
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw stacked percentile bar graphs for requested metrics across "
            "block-size sweep directories (BS=*)."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("benchmarks/block-size-sweep"),
        help="Base directory containing BS=<size> subdirectories.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/block-size-sweep/results"),
        help="Directory where graph images are written.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help=(
            "Metric shorthand names to plot, e.g. ttft itl tpot ntpot "
            "request_latency schedule_delay."
        ),
    )
    return parser.parse_args()


def parse_block_size(directory_name: str) -> int | None:
    if not directory_name.startswith("BS="):
        return None
    value = directory_name.split("=", 1)[1]
    return int(value) if value.isdigit() else None


def resolve_metric_path(metric_name: str) -> Tuple[str, ...]:
    key = metric_name.lower()
    if key not in METRIC_ALIASES:
        valid = ", ".join(sorted(METRIC_ALIASES))
        raise ValueError(f"Unknown metric '{metric_name}'. Supported: {valid}")
    return METRIC_ALIASES[key]


def nested_get(payload: dict, path: Tuple[str, ...]):
    cursor = payload
    for node in path:
        if not isinstance(cursor, dict) or node not in cursor:
            return None
        cursor = cursor[node]
    return cursor


def select_scale(max_value_seconds: float) -> Tuple[float, str]:
    if max_value_seconds >= 1.0:
        return 1.0, "s"
    if max_value_seconds >= 1e-3:
        return 1e3, "ms"
    return 1e6, "us"


def collect_metric_percentiles(
    base_dir: Path, metric_path: Tuple[str, ...]
) -> Tuple[List[int], Dict[str, List[float]]]:
    collected: List[Tuple[int, Dict[str, float]]] = []
    for candidate in sorted(base_dir.iterdir(), key=lambda p: p.name):
        if not candidate.is_dir():
            continue
        block_size = parse_block_size(candidate.name)
        if block_size is None:
            continue
        summary_file = candidate / "summary_lifecycle_metrics.json"
        if not summary_file.exists():
            continue

        with summary_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        metric_obj = nested_get(data, metric_path)
        if not isinstance(metric_obj, dict):
            continue

        try:
            percentiles = {p: float(metric_obj[p]) for p in PERCENTILES}
        except (KeyError, TypeError, ValueError):
            continue
        collected.append((block_size, percentiles))

    collected.sort(key=lambda item: item[0])
    block_sizes = [block for block, _ in collected]
    values_by_pct = {p: [entry[p] for _, entry in collected] for p in PERCENTILES}
    return block_sizes, values_by_pct


def plot_metric(
    metric_name: str,
    block_sizes: List[int],
    values_by_pct: Dict[str, List[float]],
    output_path: Path,
) -> None:
    if not block_sizes:
        print(f"[warn] No valid data found for metric '{metric_name}'. Skipping.")
        return

    max_v = max(values_by_pct["p99"])
    scale, unit = select_scale(max_v)
    scaled = {k: np.array(v, dtype=float) * scale for k, v in values_by_pct.items()}

    # Stack increments so each segment reflects percentile band.
    median = scaled["median"]
    p90 = np.maximum(scaled["p90"] - scaled["median"], 0.0)
    p95 = np.maximum(scaled["p95"] - scaled["p90"], 0.0)
    p99 = np.maximum(scaled["p99"] - scaled["p95"], 0.0)

    x = np.arange(len(block_sizes))
    width = 0.72

    plt.figure(figsize=(12, 7))
    plt.bar(x, median, width=width, color=PERCENTILE_COLORS["median"], label="median")
    plt.bar(
        x,
        p90,
        width=width,
        bottom=median,
        color=PERCENTILE_COLORS["p90"],
        label="p90",
    )
    plt.bar(
        x,
        p95,
        width=width,
        bottom=median + p90,
        color=PERCENTILE_COLORS["p95"],
        label="p95",
    )
    plt.bar(
        x,
        p99,
        width=width,
        bottom=median + p90 + p95,
        color=PERCENTILE_COLORS["p99"],
        label="p99",
    )

    # Mark block size with minimum latency for each percentile.
    percentile_series = {
        "median": scaled["median"],
        "p90": scaled["p90"],
        "p95": scaled["p95"],
        "p99": scaled["p99"],
    }
    marker_styles = {"median": "o", "p90": "s", "p95": "^", "p99": "D"}
    for pct in PERCENTILES:
        series = percentile_series[pct]
        min_idx = int(np.argmin(series))
        min_val = float(series[min_idx])
        marker_x = x[min_idx]
        marker_label = f"min {pct} (BS={block_sizes[min_idx]})"
        plt.scatter(
            marker_x,
            min_val,
            marker=marker_styles[pct],
            s=120,
            color=PERCENTILE_COLORS[pct],
            edgecolors="black",
            linewidths=0.9,
            zorder=4,
            label=marker_label,
        )

    plt.xticks(x, [str(bs) for bs in block_sizes], fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Block size", fontsize=14)
    plt.ylabel(f"Latency ({unit})", fontsize=14)
    plt.title(
        f"{metric_name} percentiles by block size", fontsize=16, pad=12
    )
    plt.legend(fontsize=10, ncol=2)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"[ok] Wrote {output_path}")


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    for metric in args.metrics:
        try:
            metric_path = resolve_metric_path(metric)
        except ValueError as exc:
            print(f"[error] {exc}")
            continue

        block_sizes, values_by_pct = collect_metric_percentiles(base_dir, metric_path)
        output_path = results_dir / f"{metric.lower()}.png"
        plot_metric(metric.lower(), block_sizes, values_by_pct, output_path)


if __name__ == "__main__":
    main()
