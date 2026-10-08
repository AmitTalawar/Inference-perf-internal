#!/usr/bin/env python3
"""Merge per-day .jsonl.zst trace files from the release tree into one .jsonl.

Directory layout expected:

    <release-dir>/
        <week>/
            traces/
                YYYY-MM-DD.jsonl.zst
                ...

The script iterates every week folder, decompresses each daily .jsonl.zst
(streaming, via the `zstd` CLI) and appends its lines to a single output
.jsonl file. Each source file's content is newline-terminated so sessions
never get concatenated onto the same line.

NOTE: the full release tree decompresses to ~100 GiB. Use --limit / --max-sessions
to do a smaller run first.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys


DEFAULT_RELEASE_DIR = "/home/ubuntu/hf_directory/freeinference_trace/release"
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "release_all_raw.jsonl")

CHUNK = 1 << 20  # 1 MiB


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def find_zst_files(release_dir: str) -> list[str]:
    pattern = os.path.join(release_dir, "*", "traces", "*.jsonl.zst")
    return sorted(glob.glob(pattern))


def decompressed_size(files: list[str]) -> int:
    """Total decompressed size (bytes) via `zstd -l` frame headers; 0 on failure."""
    total = 0
    for f in files:
        try:
            # `zstd -l` prints a human table; easier to just sum via a quick probe.
            # Fall back to 0 if parsing fails for any file.
            out = subprocess.run(
                ["zstd", "-lv", f], capture_output=True, text=True, check=True
            ).stdout
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("Decompressed Size:"):
                    # e.g. "Decompressed Size: 41.3 MiB (43252345 B)"
                    inside = line.split("(")[-1].split("B")[0].strip()
                    total += int(inside)
                    break
        except Exception:
            return 0
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-dir", default=DEFAULT_RELEASE_DIR, help="Root release directory.")
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="Output merged .jsonl path.")
    ap.add_argument("--limit", type=int, default=0, help="Max number of .zst files to process (0 = all).")
    ap.add_argument("--max-sessions", type=int, default=0, help="Stop after writing this many lines/sessions (0 = no cap).")
    ap.add_argument("--min-free-gib", type=float, default=5.0, help="Abort if free disk would drop below this many GiB.")
    ap.add_argument("--dry-run", action="store_true", help="List what would be done, write nothing.")
    args = ap.parse_args()

    files = find_zst_files(args.release_dir)
    if args.limit > 0:
        files = files[: args.limit]

    if not files:
        print(f"No .jsonl.zst files found under {args.release_dir}", file=sys.stderr)
        return 1

    if shutil.which("zstd") is None:
        print("`zstd` CLI not found on PATH.", file=sys.stderr)
        return 1

    print(f"Found {len(files)} .jsonl.zst file(s) under {args.release_dir}")

    est = decompressed_size(files)
    if est:
        print(f"Estimated decompressed size: {human(est)}")
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(args.output))).free
        print(f"Free disk at output location: {human(free)}")
        if not args.dry_run and (free - est) < args.min_free_gib * (1 << 30):
            print(
                f"ABORT: writing ~{human(est)} would leave less than {args.min_free_gib} GiB free.\n"
                f"       Re-run with --limit / --max-sessions, or lower --min-free-gib to override.",
                file=sys.stderr,
            )
            return 2

    if args.dry_run:
        print("\n[dry-run] Files that would be merged (in order):")
        for f in files:
            print(f"  {f}")
        print(f"\n[dry-run] Would write to: {args.output}")
        return 0

    total_lines = 0
    bytes_written = 0
    with open(args.output, "wb") as out:
        for idx, f in enumerate(files, 1):
            last_byte = b"\n"
            proc = subprocess.Popen(["zstd", "-dc", f], stdout=subprocess.PIPE)
            assert proc.stdout is not None
            file_lines = 0
            buf = b""
            while True:
                chunk = proc.stdout.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                bytes_written += len(chunk)
                last_byte = chunk[-1:]
                # Count newlines across the chunk boundary correctly.
                file_lines += chunk.count(b"\n")
                buf = chunk  # keep last chunk only; used to detect trailing newline
                if args.max_sessions and (total_lines + file_lines) >= args.max_sessions:
                    break
            proc.stdout.close()
            proc.wait()
            # Ensure the file ends with a newline so the next file starts cleanly.
            if last_byte != b"\n":
                out.write(b"\n")
                bytes_written += 1
                file_lines += 1
            total_lines += file_lines
            print(
                f"[{idx}/{len(files)}] {os.path.relpath(f, args.release_dir)} "
                f"-> +{file_lines} lines (total {total_lines}, {human(bytes_written)})"
            )
            if args.max_sessions and total_lines >= args.max_sessions:
                print(f"Reached --max-sessions={args.max_sessions}; stopping.")
                break

    print(f"\nDone. Wrote {total_lines} lines ({human(bytes_written)}) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
