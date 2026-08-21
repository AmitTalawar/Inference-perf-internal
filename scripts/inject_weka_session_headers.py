#!/usr/bin/env python3
"""Inject per-session headers into Weka trace JSONL records.

This script updates each JSON object line by adding/updating:
  session_headers: { ... }

It is designed to stream large files line-by-line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Dict, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inject session_headers into each Weka trace record in a JSONL file. "
            "Adds API key header and optional extra per-session header."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input Weka trace JSONL file path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Required unless --in-place is set.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the input file in place (safe temp-file replace).",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="API key value to inject into session_headers.",
    )
    parser.add_argument(
        "--api-key-header",
        default="Authorization",
        help="Header key for API key injection (default: Authorization).",
    )
    parser.add_argument(
        "--api-key-format",
        choices=["bearer", "raw"],
        default="bearer",
        help="Format for API key header value: 'bearer' -> 'Bearer <api-key>', 'raw' -> '<api-key>'.",
    )
    parser.add_argument(
        "--session-header-key",
        default=None,
        help=(
            "Optional additional session header key to inject per trace. "
            "If set, --session-header-value-template must also be set."
        ),
    )
    parser.add_argument(
        "--session-header-value-template",
        default=None,
        help=(
            "Optional additional session header value template. "
            "Supports {id} placeholder from trace.id (example: sess-{id})."
        ),
    )
    parser.add_argument(
        "--overwrite-existing-session-headers",
        action="store_true",
        help=(
            "If set, replace existing trace.session_headers entirely. "
            "Default behavior merges and overwrites only keys being injected."
        ),
    )
    return parser.parse_args()


def _build_api_key_value(api_key: str, fmt: str) -> str:
    if fmt == "raw":
        return api_key
    return f"Bearer {api_key}"


def _merge_headers_case_insensitive(base: Dict[str, str], incoming: Dict[str, str]) -> Dict[str, str]:
    merged = dict(base)
    for in_key, in_val in incoming.items():
        in_key_lower = in_key.lower()
        existing_keys = [k for k in merged.keys() if k.lower() == in_key_lower]
        for existing in existing_keys:
            del merged[existing]
        merged[in_key] = in_val
    return merged


def _inject_headers_for_record(
    record: dict,
    api_key_header: str,
    api_key_value: str,
    session_header_key: Optional[str],
    session_header_value_template: Optional[str],
    overwrite_existing_session_headers: bool,
) -> dict:
    trace_id = str(record.get("id", ""))

    new_headers: Dict[str, str] = {
        api_key_header: api_key_value,
    }
    if session_header_key:
        assert session_header_value_template is not None
        new_headers[session_header_key] = session_header_value_template.format(id=trace_id)

    existing = record.get("session_headers")
    if existing is None or overwrite_existing_session_headers:
        record["session_headers"] = new_headers
        return record

    if not isinstance(existing, dict):
        raise ValueError("session_headers exists but is not a JSON object")

    existing_str = {str(k): str(v) for k, v in existing.items()}
    record["session_headers"] = _merge_headers_case_insensitive(existing_str, new_headers)
    return record


def transform_jsonl(
    input_path: Path,
    output_path: Path,
    api_key_header: str,
    api_key_value: str,
    session_header_key: Optional[str],
    session_header_value_template: Optional[str],
    overwrite_existing_session_headers: bool,
) -> tuple[int, int]:
    total = 0
    updated = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, raw in enumerate(src, start=1):
            stripped = raw.strip()
            if not stripped:
                dst.write(raw)
                continue

            total += 1
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at line {line_no}, got {type(record).__name__}")

            updated_record = _inject_headers_for_record(
                record=record,
                api_key_header=api_key_header,
                api_key_value=api_key_value,
                session_header_key=session_header_key,
                session_header_value_template=session_header_value_template,
                overwrite_existing_session_headers=overwrite_existing_session_headers,
            )
            dst.write(json.dumps(updated_record, ensure_ascii=False) + "\n")
            updated += 1

    return total, updated


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if args.session_header_key and not args.session_header_value_template:
        raise ValueError("--session-header-value-template is required when --session-header-key is set")
    if args.session_header_value_template and not args.session_header_key:
        raise ValueError("--session-header-key is required when --session-header-value-template is set")
    if not args.in_place and not args.output:
        raise ValueError("Either --output must be provided, or use --in-place")

    api_key_value = _build_api_key_value(args.api_key, args.api_key_format)

    if args.in_place:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(input_path.parent),
            prefix=f"{input_path.name}.tmp.",
            suffix=".jsonl",
        ) as tmp:
            temp_path = Path(tmp.name)

        try:
            total, updated = transform_jsonl(
                input_path=input_path,
                output_path=temp_path,
                api_key_header=args.api_key_header,
                api_key_value=api_key_value,
                session_header_key=args.session_header_key,
                session_header_value_template=args.session_header_value_template,
                overwrite_existing_session_headers=args.overwrite_existing_session_headers,
            )
            temp_path.replace(input_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
        output_path = input_path
    else:
        output_path = Path(args.output)
        total, updated = transform_jsonl(
            input_path=input_path,
            output_path=output_path,
            api_key_header=args.api_key_header,
            api_key_value=api_key_value,
            session_header_key=args.session_header_key,
            session_header_value_template=args.session_header_value_template,
            overwrite_existing_session_headers=args.overwrite_existing_session_headers,
        )

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Records processed: {total}")
    print(f"Records updated: {updated}")
    print(
        f"Injected header: {args.api_key_header} "
        f"(format={args.api_key_format}, value={'Bearer <api_key>' if args.api_key_format == 'bearer' else '<api_key>'})"
    )
    if args.session_header_key:
        print(
            "Injected extra session header: "
            f"{args.session_header_key}={args.session_header_value_template}"
        )


if __name__ == "__main__":
    main()
