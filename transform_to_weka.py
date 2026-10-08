#!/usr/bin/env python3
"""Transform raw session traces (File 1 format) into Weka-compatible traces (File 2).

Input  (per line): {"format_version", "week", "day", "session": {...}}
Output (per line): WekaTrace = {"id", "models", "block_size", "hash_id_scope",
                                "requests": [...], "session_headers"?}

Each raw session.request becomes a Weka request:
  - type: "s" (streaming, has ttft) if request.stream else "n" (normal)
  - t:        (request.start_ms - session.start_ms) / 1000   seconds
  - model:    request.model
  - in/out:   request.provider_tokens.prompt / .completion
  - hash_ids: request.block_ids passed through unchanged (global ids; same
              block keeps the same id across sessions/users for cross-session
              prefix sharing). hash_id_scope is "global".
  - api_time: (request.end_ms - request.start_ms) / 1000     seconds
  - ttft:     request.ttft_ms / 1000  (streaming only)

Streaming, line-by-line; constant memory regardless of input size.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time


DEFAULT_INPUT = "/home/ubuntu/sneh/opensource/eviction/Inference-perf-internal/release_all_raw.jsonl"
DEFAULT_OUTPUT = "/home/ubuntu/sneh/opensource/eviction/Inference-perf-internal/release_all_weka.jsonl"


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def transform_session(session: dict, block_size: int, keep_headers: bool) -> dict | None:
    """Convert one raw `session` dict into a WekaTrace dict. Returns None if unusable."""
    reqs_in = session.get("requests") or []
    if not reqs_in:
        return None

    session_start = session.get("start_ms")
    if session_start is None:
        # Fall back to the earliest request start.
        starts = [r.get("start_ms") for r in reqs_in if r.get("start_ms") is not None]
        if not starts:
            return None
        session_start = min(starts)

    models: list[str] = []
    seen_models: set[str] = set()
    out_reqs: list[dict] = []

    for r in reqs_in:
        model = r.get("model")
        if model and model not in seen_models:
            seen_models.add(model)
            models.append(model)

        start_ms = r.get("start_ms")
        end_ms = r.get("end_ms")
        t = round((start_ms - session_start) / 1000.0, 3) if start_ms is not None else 0.0
        api_time = (
            round((end_ms - start_ms) / 1000.0, 3)
            if (start_ms is not None and end_ms is not None)
            else None
        )

        ptok = r.get("provider_tokens") or {}
        in_tokens = int(ptok.get("prompt") or 0)
        out_tokens = int(ptok.get("completion") or 0)

        # Emit the raw GLOBAL block ids unchanged so the same block keeps the
        # same id across every session/user. This preserves cross-session /
        # cross-user prefix sharing (shared system prompt + tool definitions).
        # Within a session, equality/order are still preserved, so turn-to-turn
        # prefix structure is unchanged.
        hash_ids = [int(b) for b in (r.get("block_ids") or [])]
        is_stream = bool(r.get("stream"))

        entry: dict = {
            "t": t,
            "type": "s" if is_stream else "n",
            "model": model,
            "in": in_tokens,
            "out": out_tokens,
            "hash_ids": hash_ids,
            "api_time": api_time,
        }
        if is_stream:
            ttft_ms = r.get("ttft_ms")
            entry["ttft"] = round(ttft_ms / 1000.0, 3) if ttft_ms is not None else None

        out_reqs.append(entry)

    if not out_reqs:
        return None

    trace: dict = {
        "id": session.get("session_id"),
        "models": models,
        "block_size": block_size,
        "hash_id_scope": "global",
        "requests": out_reqs,
    }
    if keep_headers:
        trace["session_headers"] = None
    return trace


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--block-size",
        type=int,
        default=16,
        help=(
            "Tokens per hash-id block. Must match the granularity at which the "
            "source block_ids were generated (~16 tokens/block for these traces). "
            "inference-perf reads this field and uses in//block_size to decide how "
            "many hash_ids to consume, so a too-large value silently drops hash_ids."
        ),
    )
    ap.add_argument("--limit", type=int, default=0, help="Max input lines to process (0 = all).")
    ap.add_argument("--keep-headers", action="store_true", help="Emit session_headers: null.")
    ap.add_argument("--progress-every", type=int, default=50000, help="Print progress every N lines.")
    ap.add_argument("--min-free-gib", type=float, default=3.0, help="Abort if free disk drops below this.")
    args = ap.parse_args()

    use_stdin = args.input == "-"
    if not use_stdin and not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    min_free = args.min_free_gib * (1 << 30)

    t0 = time.time()
    read_lines = written = skipped = bad = 0
    bytes_out = 0

    fin = sys.stdin if use_stdin else open(args.input, "r")
    try:
        fout = open(args.output, "w")
        for line in fin:
            read_lines += 1
            if args.limit and read_lines > args.limit:
                read_lines -= 1
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue
            session = obj.get("session") if isinstance(obj, dict) else None
            if session is None:
                skipped += 1
                continue
            trace = transform_session(session, args.block_size, args.keep_headers)
            if trace is None or not trace.get("id"):
                skipped += 1
                continue
            s = json.dumps(trace, separators=(",", ":"))
            fout.write(s)
            fout.write("\n")
            written += 1
            bytes_out += len(s) + 1

            if args.progress_every and read_lines % args.progress_every == 0:
                rate = read_lines / (time.time() - t0 + 1e-9)
                free = shutil.disk_usage(out_dir).free
                print(
                    f"  read={read_lines:,} written={written:,} skipped={skipped:,} "
                    f"bad={bad:,} out={human(bytes_out)} "
                    f"({rate:,.0f} lines/s, free={human(free)})"
                )
                if free < min_free:
                    print(
                        f"ABORT: free disk {human(free)} below --min-free-gib; stopping.",
                        file=sys.stderr,
                    )
                    break
    finally:
        try:
            fout.close()
        except Exception:
            pass
        if not use_stdin:
            fin.close()

    dt = time.time() - t0
    print(
        f"\nDone in {dt:.1f}s. read={read_lines:,} written={written:,} "
        f"skipped={skipped:,} bad_json={bad:,} out={human(bytes_out)}"
    )
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
