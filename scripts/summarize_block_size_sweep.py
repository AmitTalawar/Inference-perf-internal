#!/usr/bin/env python3
"""Summarize block-size sweep results from inference-perf + vLLM JSONL metrics.

Reads:
  - BS=*/summary_lifecycle_metrics.json
  - BS=*/vllm_metrics.jsonl

Writes:
  - JSON summary for all block sizes
  - CSV table for quick plotting/comparison
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize block-size sweep metrics.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("benchmarks/block-size-sweep"),
        help="Directory containing BS=<size> subdirectories.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("benchmarks/block-size-sweep/results/summary_combined.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("benchmarks/block-size-sweep/results/summary_combined.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def parse_block_size(path: Path) -> int | None:
    name = path.name
    if not name.startswith("BS="):
        return None
    raw = name.split("=", 1)[1]
    return int(raw) if raw.isdigit() else None


def nested_get(payload: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cursor: Any = payload
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_counter_sum(cumulative: dict[str, Any], metric_name: str) -> float:
    metric = cumulative.get(metric_name)
    if not isinstance(metric, dict):
        return 0.0
    total = 0.0
    for series in metric.get("series", []):
        value = series.get("value")
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def load_last_jsonl_record(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last_line = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                last_line = line
    if not last_line:
        return None
    return json.loads(last_line)


def summarize_vllm_metrics(path: Path) -> dict[str, Any]:
    record = load_last_jsonl_record(path)
    if not record:
        return {}

    cumulative = record.get("cumulative", {})
    prefix_hits = get_counter_sum(cumulative, "vllm:prefix_cache_hits")
    prefix_queries = get_counter_sum(cumulative, "vllm:prefix_cache_queries")
    ext_hits = get_counter_sum(cumulative, "vllm:external_prefix_cache_hits")
    ext_queries = get_counter_sum(cumulative, "vllm:external_prefix_cache_queries")
    prompt_cached = get_counter_sum(cumulative, "vllm:prompt_tokens_cached")
    prompt_recomputed = get_counter_sum(cumulative, "vllm:prompt_tokens_recomputed")

    load_bytes = get_counter_sum(cumulative, "vllm:kv_offload_load_bytes")
    load_time = get_counter_sum(cumulative, "vllm:kv_offload_load_time")
    store_bytes = get_counter_sum(cumulative, "vllm:kv_offload_store_bytes")
    store_time = get_counter_sum(cumulative, "vllm:kv_offload_store_time")
    alloc_failures = get_counter_sum(cumulative, "vllm:kv_offload_allocation_failure")

    prefix_hit_rate = (100.0 * prefix_hits / prefix_queries) if prefix_queries > 0 else None
    external_hit_rate = (100.0 * ext_hits / ext_queries) if ext_queries > 0 else None
    prompt_cached_ratio = (
        100.0 * prompt_cached / (prompt_cached + prompt_recomputed)
        if (prompt_cached + prompt_recomputed) > 0
        else None
    )
    load_bandwidth_gbps = (load_bytes / load_time / 1e9) if load_time > 0 else None
    store_bandwidth_gbps = (store_bytes / store_time / 1e9) if store_time > 0 else None

    return {
        "elapsed_s": to_float(record.get("elapsed_s")),
        "prefix_hits_tokens": prefix_hits,
        "prefix_queries_tokens": prefix_queries,
        "prefix_hit_rate_pct": prefix_hit_rate,
        "external_hits_tokens": ext_hits,
        "external_queries_tokens": ext_queries,
        "external_hit_rate_pct": external_hit_rate,
        "prompt_tokens_cached": prompt_cached,
        "prompt_tokens_recomputed": prompt_recomputed,
        "prompt_cached_ratio_pct": prompt_cached_ratio,
        "kv_offload_load_bytes": load_bytes,
        "kv_offload_load_seconds": load_time,
        "kv_offload_load_bandwidth_gbps": load_bandwidth_gbps,
        "kv_offload_store_bytes": store_bytes,
        "kv_offload_store_seconds": store_time,
        "kv_offload_store_bandwidth_gbps": store_bandwidth_gbps,
        "kv_offload_allocation_failures": alloc_failures,
    }


def summarize_lifecycle(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    def metric(path_keys: list[str]) -> float | None:
        return to_float(nested_get(data, path_keys))

    return {
        "benchmark_time_seconds": metric(["benchmark_time_seconds"]),
        "success_count": metric(["successes", "count"]),
        "failure_count": metric(["failures", "count"]),
        "schedule_delay_median_s": metric(["load_summary", "schedule_delay", "median"]),
        "schedule_delay_p90_s": metric(["load_summary", "schedule_delay", "p90"]),
        "request_latency_median_s": metric(["successes", "latency", "request_latency", "median"]),
        "request_latency_p90_s": metric(["successes", "latency", "request_latency", "p90"]),
        "ttft_median_s": metric(["successes", "latency", "time_to_first_token", "median"]),
        "ttft_p90_s": metric(["successes", "latency", "time_to_first_token", "p90"]),
        "ttft_p95_s": metric(["successes", "latency", "time_to_first_token", "p95"]),
        "ttft_p99_s": metric(["successes", "latency", "time_to_first_token", "p99"]),
        "tpot_median_s": metric(["successes", "latency", "time_per_output_token", "median"]),
        "tpot_p90_s": metric(["successes", "latency", "time_per_output_token", "p90"]),
        "tpot_p95_s": metric(["successes", "latency", "time_per_output_token", "p95"]),
        "itl_median_s": metric(["successes", "latency", "inter_token_latency", "median"]),
        "itl_p90_s": metric(["successes", "latency", "inter_token_latency", "p90"]),
        "requests_per_sec": metric(["successes", "throughput", "requests_per_sec"]),
        "input_tokens_per_sec": metric(["successes", "throughput", "input_tokens_per_sec"]),
        "output_tokens_per_sec": metric(["successes", "throughput", "output_tokens_per_sec"]),
        "prompt_len_median": metric(["successes", "prompt_len", "median"]),
        "prompt_len_p90": metric(["successes", "prompt_len", "p90"]),
        "output_len_median": metric(["successes", "output_len", "median"]),
        "goodput_percentage": metric(["successes", "goodput_metrics", "goodput_percentage"]),
        "ttft_attainment_percentage": metric(["successes", "goodput_metrics", "ttft_attainment_percentage"]),
        "itl_attainment_percentage": metric(["successes", "goodput_metrics", "itl_attainment_percentage"]),
    }


def discover_runs(base_dir: Path) -> list[tuple[int, Path]]:
    runs: list[tuple[int, Path]] = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        bs = parse_block_size(child)
        if bs is None:
            continue
        runs.append((bs, child))
    runs.sort(key=lambda x: x[0])
    return runs


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    rows: list[dict[str, Any]] = []
    for block_size, run_dir in discover_runs(base_dir):
        lifecycle = summarize_lifecycle(run_dir / "summary_lifecycle_metrics.json")
        vllm = summarize_vllm_metrics(run_dir / "vllm_metrics.jsonl")
        combined = {"block_size": block_size, "run_dir": str(run_dir), **lifecycle, **vllm}
        rows.append(combined)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    write_csv(rows, args.out_csv)

    print(f"Wrote {len(rows)} rows")
    print(f"JSON: {args.out_json}")
    print(f"CSV:  {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
