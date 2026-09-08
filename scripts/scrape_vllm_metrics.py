#!/usr/bin/env python3
"""Sample vLLM /metrics as a Prometheus-style time series.

Each scrape is one JSONL record. Counters and histograms are stored the way
Prometheus exposes them (cumulative), plus per-interval deltas so you can
plot rates and histogram increases without a Prometheus server.

By default every family on the endpoint is recorded (engine, KV offload,
HTTP, process). Narrow later with --include if a run is too large.

Example (run while inference-perf is in flight, stop with SIGINT/SIGTERM):

  python3 scripts/scrape_vllm_metrics.py \\
    --url http://127.0.0.1:8000/metrics \\
    --interval 10 \\
    --block-size 64 \\
    --out benchmarks/block-size-sweep/BS=64/vllm_metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Match everything the endpoint returns unless --include is set.
DEFAULT_INCLUDE = r"."

HELP_RE = re.compile(r"^#\s+HELP\s+(\S+)\s+(.*)$")
TYPE_RE = re.compile(r"^#\s+TYPE\s+(\S+)\s+(\S+)$")
SAMPLE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(.*)\})?\s+"
    r"(\+Inf|-Inf|NaN|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s+\d+)?"
    r"\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time-series scrape of vLLM /metrics with full histograms."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/metrics",
        help="vLLM Prometheus scrape URL.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="JSONL path. One scrape record per line.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between scrapes (default: 10).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Write a single scrape and exit.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after this many seconds. 0 means run until signal.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="Tag each record with the vLLM --block-size under test.",
    )
    parser.add_argument(
        "--include",
        default=DEFAULT_INCLUDE,
        help=(
            "Regex matched against Prometheus family names. "
            "Default '.' keeps every family on the endpoint."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="If set, also write each full /metrics body as a .prom file.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout per scrape in seconds.",
    )
    return parser.parse_args()


def _parse_prom_value(raw: str) -> float:
    if raw in ("+Inf", "Inf"):
        return float("inf")
    if raw == "-Inf":
        return float("-inf")
    if raw == "NaN":
        return float("nan")
    return float(raw)


def _parse_labels(raw: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    i = 0
    n = len(raw)
    while i < n:
        while i < n and raw[i] in ", \t":
            i += 1
        if i >= n:
            break
        eq = raw.find("=", i)
        if eq < 0:
            break
        key = raw[i:eq].strip()
        i = eq + 1
        if i >= n or raw[i] != '"':
            break
        i += 1
        value: list[str] = []
        while i < n:
            ch = raw[i]
            if ch == "\\":
                i += 1
                if i < n:
                    escaped = raw[i]
                    value.append({"n": "\n", "t": "\t", "\\": "\\", '"': '"'}.get(escaped, escaped))
                    i += 1
                continue
            if ch == '"':
                i += 1
                break
            value.append(ch)
            i += 1
        labels[key] = "".join(value)
    return labels


def _parse_prometheus_text(text: str) -> list[dict[str, Any]]:
    """Fallback text-format parser (no prometheus_client required)."""
    types: dict[str, str] = {}
    helps: dict[str, str] = {}
    samples: list[tuple[str, dict[str, str], float]] = []
    for line in text.splitlines():
        if not line:
            continue
        if line.startswith("#"):
            help_m = HELP_RE.match(line)
            if help_m:
                helps[help_m.group(1)] = help_m.group(2)
                continue
            type_m = TYPE_RE.match(line)
            if type_m:
                types[type_m.group(1)] = type_m.group(2).lower()
            continue
        sample_m = SAMPLE_RE.match(line)
        if not sample_m:
            continue
        name, label_raw, value_raw = sample_m.group(1), sample_m.group(2), sample_m.group(3)
        labels = _parse_labels(label_raw) if label_raw else {}
        samples.append((name, labels, _parse_prom_value(value_raw)))

    grouped: dict[str, dict[str, Any]] = {}

    def family_for(sample_name: str) -> tuple[str, str]:
        for suffix in ("_bucket", "_sum", "_count", "_created", "_total"):
            if sample_name.endswith(suffix):
                base = sample_name[: -len(suffix)]
                if base in types:
                    return base, types[base]
        if sample_name in types:
            return sample_name, types[sample_name]
        return sample_name, "untyped"

    for sample_name, labels, value in samples:
        if sample_name.endswith("_created"):
            continue
        fam_name, fam_type = family_for(sample_name)
        fam = grouped.setdefault(
            fam_name,
            {
                "name": fam_name,
                "type": fam_type,
                "help": helps.get(fam_name, ""),
                "samples": [],
            },
        )
        fam["samples"].append({"name": sample_name, "labels": labels, "value": value})
    return list(grouped.values())


def _parse_with_prometheus_client(text: str) -> list[dict[str, Any]] | None:
    try:
        from prometheus_client.parser import text_string_to_metric_families
    except ImportError:
        return None
    families: list[dict[str, Any]] = []
    for family in text_string_to_metric_families(text):
        samples = []
        for sample in family.samples:
            if sample.name.endswith("_created"):
                continue
            samples.append(
                {
                    "name": sample.name,
                    "labels": dict(sample.labels),
                    "value": float(sample.value),
                }
            )
        families.append(
            {
                "name": family.name,
                "type": family.type,
                "help": family.documentation,
                "samples": samples,
            }
        )
    return families


def parse_families(text: str) -> list[dict[str, Any]]:
    parsed = _parse_with_prometheus_client(text)
    if parsed is not None:
        return parsed
    return _parse_prometheus_text(text)


def _label_key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(labels.items()))


def _le_sort_key(le: str) -> float:
    if le == "+Inf":
        return float("inf")
    return float(le)


def _organize_family(family: dict[str, Any]) -> dict[str, Any]:
    fam_type = family["type"]
    if fam_type == "histogram":
        series: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
        for sample in family["samples"]:
            labels = dict(sample["labels"])
            le = labels.pop("le", None)
            slot = series.setdefault(
                _label_key(labels),
                {"labels": labels, "buckets": {}, "count": None, "sum": None},
            )
            name = sample["name"]
            if name.endswith("_bucket") and le is not None:
                slot["buckets"][le] = sample["value"]
            elif name.endswith("_count"):
                slot["count"] = sample["value"]
            elif name.endswith("_sum"):
                slot["sum"] = sample["value"]
        organized = []
        for slot in series.values():
            buckets = [
                {"le": le, "count": slot["buckets"][le]}
                for le in sorted(slot["buckets"], key=_le_sort_key)
            ]
            organized.append(
                {
                    "labels": slot["labels"],
                    "count": slot["count"],
                    "sum": slot["sum"],
                    "buckets": buckets,
                }
            )
        return {
            "type": "histogram",
            "help": family.get("help", ""),
            "series": organized,
        }

    series_out = []
    for sample in family["samples"]:
        series_out.append({"labels": sample["labels"], "value": sample["value"]})
    return {
        "type": fam_type,
        "help": family.get("help", ""),
        "series": series_out,
    }


def organize_metrics(
    families: Iterable[dict[str, Any]], include_re: re.Pattern[str]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family in families:
        if not include_re.search(family["name"]):
            continue
        out[family["name"]] = _organize_family(family)
    return out


def _series_map(metric: dict[str, Any]) -> dict[tuple[tuple[str, str], ...], dict[str, Any]]:
    return {_label_key(item["labels"]): item for item in metric.get("series", [])}


def _bucket_map(series: dict[str, Any] | None) -> dict[str, float]:
    if not series:
        return {}
    return {bucket["le"]: bucket["count"] for bucket in series.get("buckets", [])}


def _delta_histogram(curr: dict[str, Any], prev: dict[str, Any] | None) -> dict[str, Any]:
    prev_buckets = _bucket_map(prev)
    curr_buckets = _bucket_map(curr)
    prev_count = (prev or {}).get("count") or 0.0
    prev_sum = (prev or {}).get("sum") or 0.0
    buckets = [
        {"le": le, "count": curr_buckets[le] - prev_buckets.get(le, 0.0)}
        for le in (bucket["le"] for bucket in curr.get("buckets", []))
    ]
    return {
        "labels": curr["labels"],
        "count": (curr.get("count") or 0.0) - prev_count,
        "sum": (curr.get("sum") or 0.0) - prev_sum,
        "buckets": buckets,
    }


def interval_metrics(
    current: dict[str, Any], previous: dict[str, Any] | None
) -> dict[str, Any]:
    """Prometheus increase() over the last scrape interval (from 0 if first)."""
    out: dict[str, Any] = {}
    for name, metric in current.items():
        prev_metric = (previous or {}).get(name)
        prev_series = _series_map(prev_metric) if prev_metric else {}
        if metric["type"] == "histogram":
            series = [
                _delta_histogram(item, prev_series.get(_label_key(item["labels"])))
                for item in metric["series"]
            ]
            out[name] = {"type": "histogram", "series": series}
            continue
        if metric["type"] == "gauge":
            out[name] = metric
            continue
        series = []
        for item in metric["series"]:
            prev_item = prev_series.get(_label_key(item["labels"]))
            prev_value = (prev_item or {}).get("value") or 0.0
            series.append(
                {
                    "labels": item["labels"],
                    "value": item["value"] - prev_value,
                }
            )
        out[name] = {"type": metric["type"], "series": series}
    return out


def _sum_values(metric: dict[str, Any] | None) -> float:
    if not metric:
        return 0.0
    total = 0.0
    for item in metric.get("series", []):
        if "value" in item and item["value"] is not None:
            total += item["value"]
        elif item.get("count") is not None:
            total += item["count"]
    return total


def _sum_field(metric: dict[str, Any] | None, field: str) -> float:
    if not metric:
        return 0.0
    return sum((item.get(field) or 0.0) for item in metric.get("series", []))


def _hit_rate(hits: float, queries: float) -> float | None:
    if queries <= 0:
        return None
    return 100.0 * hits / queries


def _bandwidth(nbytes: float, seconds: float) -> float | None:
    if seconds <= 0:
        return None
    return nbytes / seconds


def derived_rates(
    cumulative: dict[str, Any], interval: dict[str, Any], interval_s: float
) -> dict[str, Any]:
    def pack(prefix: str, source: dict[str, Any]) -> dict[str, Any]:
        hits = _sum_values(source.get(f"{prefix}_hits"))
        queries = _sum_values(source.get(f"{prefix}_queries"))
        return {
            "hits_tokens": hits,
            "queries_tokens": queries,
            "hit_rate_pct": _hit_rate(hits, queries),
        }

    store_bytes = _sum_values(interval.get("vllm:kv_offload_store_bytes"))
    store_time = _sum_values(interval.get("vllm:kv_offload_store_time"))
    load_bytes = _sum_values(interval.get("vllm:kv_offload_load_bytes"))
    load_time = _sum_values(interval.get("vllm:kv_offload_load_time"))
    return {
        "prefix_cache": {
            "cumulative": pack("vllm:prefix_cache", cumulative),
            "interval": pack("vllm:prefix_cache", interval),
        },
        "external_prefix_cache": {
            "cumulative": pack("vllm:external_prefix_cache", cumulative),
            "interval": pack("vllm:external_prefix_cache", interval),
        },
        "kv_movement_interval": {
            "store_bytes": store_bytes,
            "store_seconds": store_time,
            "store_bytes_per_sec": _bandwidth(store_bytes, store_time),
            "load_bytes": load_bytes,
            "load_seconds": load_time,
            "load_bytes_per_sec": _bandwidth(load_bytes, load_time),
            "store_ops": _sum_field(interval.get("vllm:kv_offload_store_size"), "count"),
            "load_ops": _sum_field(interval.get("vllm:kv_offload_load_size"), "count"),
            "interval_s": interval_s,
        },
    }


def fetch_metrics(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_raw(raw_dir: Path, scrape_index: int, ts: str, body: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_ts = ts.replace(":", "").replace("+", "p")
    path = raw_dir / f"{scrape_index:06d}_{safe_ts}.prom"
    path.write_text(body, encoding="utf-8")


def build_record(
    *,
    scrape_index: int,
    started: float,
    now: float,
    prev_now: float | None,
    block_size: int | None,
    cumulative: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    interval_s = (now - prev_now) if prev_now is not None else (now - started)
    interval = interval_metrics(cumulative, previous)
    return {
        "scrape_index": scrape_index,
        "timestamp": utc_now(),
        "elapsed_s": now - started,
        "interval_s": interval_s,
        "block_size": block_size,
        "derived": derived_rates(cumulative, interval, interval_s),
        "cumulative": cumulative,
        "interval": interval,
    }


def _install_signal_handlers(stop_flag: dict[str, bool]) -> None:
    def _handle(_signum: int, _frame: Any) -> None:
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def _sleep_or_stop(seconds: float, stop_flag: dict[str, bool]) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline and not stop_flag["stop"]:
        time.sleep(min(0.25, max(0.0, deadline - time.time())))


def main() -> int:
    args = parse_args()
    include_re = re.compile(args.include)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir) if args.raw_dir else None

    stop_flag = {"stop": False}
    _install_signal_handlers(stop_flag)

    started = time.time()
    deadline = started + args.duration if args.duration > 0 else None
    previous: dict[str, Any] | None = None
    prev_now: float | None = None
    scrape_index = 0
    take_final = False

    print(
        f"scraping {args.url} every {args.interval}s -> {out_path}",
        file=sys.stderr,
    )

    with out_path.open("w", encoding="utf-8") as out_f:
        while True:
            now = time.time()
            try:
                body = fetch_metrics(args.url, args.timeout)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                print(f"scrape failed: {exc}", file=sys.stderr)
                if args.once or stop_flag["stop"] or take_final:
                    break
                _sleep_or_stop(args.interval, stop_flag)
                continue

            families = parse_families(body)
            cumulative = organize_metrics(families, include_re)
            record = build_record(
                scrape_index=scrape_index,
                started=started,
                now=now,
                prev_now=prev_now,
                block_size=args.block_size,
                cumulative=cumulative,
                previous=previous,
            )
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            if raw_dir is not None:
                write_raw(raw_dir, scrape_index, record["timestamp"], body)

            derived = record["derived"]
            gpu = derived["prefix_cache"]["cumulative"]["hit_rate_pct"]
            ext = derived["external_prefix_cache"]["cumulative"]["hit_rate_pct"]
            store_bps = derived["kv_movement_interval"]["store_bytes_per_sec"]
            extra = ""
            if scrape_index == 0:
                extra = f" families={len(cumulative)}"
            print(
                f"[{scrape_index}] elapsed={record['elapsed_s']:.1f}s "
                f"gpu_hit={gpu} ext_hit={ext} store_Bps={store_bps}{extra}",
                file=sys.stderr,
            )

            previous = cumulative
            prev_now = now
            scrape_index += 1

            if args.once or take_final:
                break
            if deadline is not None and time.time() >= deadline:
                break
            if stop_flag["stop"]:
                break
            _sleep_or_stop(args.interval, stop_flag)
            if stop_flag["stop"]:
                take_final = True

    print(f"wrote {scrape_index} scrapes to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
