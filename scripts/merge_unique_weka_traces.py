#!/usr/bin/env python3
"""Merge Weka trace JSONL files and keep only unique sessions."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import re
from typing import Iterable, List


def _natural_key(value: str) -> List[object]:
    """Return a key for natural sorting (trace2 before trace10)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge trace JSONL files and keep only unique sessions by key."
    )
    parser.add_argument(
        "--input-glob",
        default="data/traces/trace*.jsonl",
        help="Glob pattern for source trace files.",
    )
    parser.add_argument(
        "--output",
        default="data/traces/trace.jsonl",
        help="Path to write the merged deduplicated JSONL.",
    )
    parser.add_argument(
        "--session-key",
        default="id",
        help="JSON key used to identify unique sessions.",
    )
    return parser.parse_args()


def resolve_input_files(pattern: str) -> List[Path]:
    files = [Path(path) for path in glob.glob(pattern)]
    files.sort(key=lambda p: _natural_key(str(p)))
    return files


def merge_unique_sessions(
    input_files: Iterable[Path], output_file: Path, session_key: str
) -> tuple[int, int, int]:
    seen: set[str] = set()
    total_lines = 0
    unique_lines = 0
    duplicate_lines = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as out_f:
        for file_path in input_files:
            with file_path.open("r", encoding="utf-8") as in_f:
                for line_number, raw_line in enumerate(in_f, start=1):
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    total_lines += 1
                    try:
                        session = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {file_path} line {line_number}: {exc}"
                        ) from exc

                    session_id = session.get(session_key)
                    if session_id is None:
                        raise ValueError(
                            f"Missing session key '{session_key}' in {file_path} line {line_number}"
                        )

                    session_id_str = str(session_id)
                    if session_id_str in seen:
                        duplicate_lines += 1
                        continue

                    seen.add(session_id_str)
                    out_f.write(stripped + "\n")
                    unique_lines += 1

    return total_lines, unique_lines, duplicate_lines


def main() -> None:
    args = parse_args()
    input_files = resolve_input_files(args.input_glob)
    if not input_files:
        raise FileNotFoundError(f"No input files matched: {args.input_glob}")

    output_path = Path(args.output)
    total, unique, duplicates = merge_unique_sessions(
        input_files=input_files,
        output_file=output_path,
        session_key=args.session_key,
    )

    print("Merged files:")
    for path in input_files:
        print(f"  - {path}")
    print(f"Output: {output_path}")
    print(f"Total sessions read: {total}")
    print(f"Unique sessions written: {unique}")
    print(f"Duplicate sessions skipped: {duplicates}")


if __name__ == "__main__":
    main()
