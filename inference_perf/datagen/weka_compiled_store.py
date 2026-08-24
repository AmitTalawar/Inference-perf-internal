# Copyright 2026 The Kubernetes Authors.
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

"""On-disk compiled-session cache for Weka trace replay.

New stores use orjson + zstd (`orjson.zst`). Existing gzip-JSON stores remain
readable via `manifest.codec`. Session files are independent so the parent
process can encode/decode them in a thread pool.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence
from urllib.parse import quote, unquote

import orjson
import zstandard as zstd

from inference_perf.datagen.replay_graph_types import GraphCall, GraphEvent, InputSegment, ReplayGraph

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
CODEC_JSON_GZ = "json.gz"
CODEC_ORJSON_ZST = "orjson.zst"
DEFAULT_CODEC = CODEC_ORJSON_ZST
# zstd level 1 favours write/load speed over ratio; RAM/disk are not the limiter.
ZSTD_LEVEL = 1
_DUPSEED_RE = re.compile(r"__dupseed(\d+)$")
_CODEC_EXTENSIONS = {
    CODEC_JSON_GZ: ".json.gz",
    CODEC_ORJSON_ZST: ".orjson.zst",
}


class CompileIdentityMismatchError(ValueError):
    """Raised when a compiled-session store was built with different compile inputs."""


class UnknownSessionCodecError(ValueError):
    """Raised when a store manifest names a codec this build does not implement."""


@dataclass(frozen=True)
class CompileIdentity:
    tokenizer_name_or_path: str
    corpus_path: str
    corpus_byte_size: int
    base_seed: int
    default_block_size: int
    trace_idle_gap_cap_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "corpus_path": self.corpus_path,
            "corpus_byte_size": self.corpus_byte_size,
            "base_seed": self.base_seed,
            "default_block_size": self.default_block_size,
            "trace_idle_gap_cap_seconds": self.trace_idle_gap_cap_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompileIdentity":
        return cls(
            tokenizer_name_or_path=str(data["tokenizer_name_or_path"]),
            corpus_path=str(data["corpus_path"]),
            corpus_byte_size=int(data["corpus_byte_size"]),
            base_seed=int(data["base_seed"]),
            default_block_size=int(data["default_block_size"]),
            trace_idle_gap_cap_seconds=float(data["trace_idle_gap_cap_seconds"]),
        )

    def check_against(self, current: "CompileIdentity") -> None:
        diffs: List[str] = []
        for field_name in (
            "tokenizer_name_or_path",
            "corpus_path",
            "corpus_byte_size",
            "base_seed",
            "default_block_size",
            "trace_idle_gap_cap_seconds",
        ):
            stored_value = getattr(self, field_name)
            current_value = getattr(current, field_name)
            if stored_value != current_value:
                diffs.append(f"{field_name}: stored={stored_value!r} current={current_value!r}")
        if diffs:
            raise CompileIdentityMismatchError("Weka compiled session store identity mismatch (" + "; ".join(diffs) + ")")


@dataclass
class StoredSession:
    artifact_id: str
    source_trace_id: str
    session_headers: Optional[Dict[str, str]]
    graph: ReplayGraph
    schema_version: int = SCHEMA_VERSION


@dataclass
class StoreManifest:
    compile_identity: CompileIdentity
    artifact_ids: List[str]
    schema_version: int = SCHEMA_VERSION
    codec: str = DEFAULT_CODEC


class SessionCodec(Protocol):
    name: str

    def encode(self, payload: Dict[str, Any]) -> bytes: ...

    def decode(self, data: bytes) -> Dict[str, Any]: ...


class JsonGzCodec:
    """Backward-compatible gzip JSON. Uses orjson for parse/dump speed."""

    name = CODEC_JSON_GZ

    def encode(self, payload: Dict[str, Any]) -> bytes:
        return gzip.compress(orjson.dumps(payload), compresslevel=1)

    def decode(self, data: bytes) -> Dict[str, Any]:
        return orjson.loads(gzip.decompress(data))


class OrjsonZstCodec:
    """Default codec: orjson payload compressed with zstd."""

    name = CODEC_ORJSON_ZST

    def encode(self, payload: Dict[str, Any]) -> bytes:
        return zstd.compress(orjson.dumps(payload), level=ZSTD_LEVEL)

    def decode(self, data: bytes) -> Dict[str, Any]:
        return orjson.loads(zstd.decompress(data))


def get_codec(name: str) -> SessionCodec:
    if name == CODEC_JSON_GZ:
        return JsonGzCodec()
    if name == CODEC_ORJSON_ZST:
        return OrjsonZstCodec()
    raise UnknownSessionCodecError(f"Unknown compiled-session codec: {name!r}")


def codec_extension(codec_name: str) -> str:
    ext = _CODEC_EXTENSIONS.get(codec_name)
    if ext is None:
        raise UnknownSessionCodecError(f"Unknown compiled-session codec: {codec_name!r}")
    return ext


def safe_artifact_filename(artifact_id: str, codec_name: str = DEFAULT_CODEC) -> str:
    """Filesystem-safe file name for an artifact id. JSON always stores the raw id."""
    return quote(artifact_id, safe="-_.") + codec_extension(codec_name)


def artifact_id_from_filename(filename: str) -> str:
    for ext in _CODEC_EXTENSIONS.values():
        if filename.endswith(ext):
            return unquote(filename[: -len(ext)])
    return unquote(filename)


def store_io_workers(n_items: int) -> int:
    """Thread count for parent-process session file encode/decode."""
    if n_items <= 1:
        return 1
    return max(1, min(n_items, os.cpu_count() or 8))


def source_trace_id_from_artifact_id(artifact_id: str) -> str:
    match = _DUPSEED_RE.search(artifact_id)
    if match:
        return artifact_id[: match.start()]
    return artifact_id


def max_dupseed_suffix(artifact_ids: Iterable[str]) -> int:
    max_n = 0
    for artifact_id in artifact_ids:
        match = _DUPSEED_RE.search(artifact_id)
        if match:
            max_n = max(max_n, int(match.group(1)))
    return max_n


def _input_segment_to_dict(seg: InputSegment) -> Dict[str, Any]:
    return {
        "type": seg.type,
        "message_count": seg.message_count,
        "token_count": seg.token_count,
        "source_event_id": seg.source_event_id,
    }


def _input_segment_from_dict(data: Dict[str, Any]) -> InputSegment:
    return InputSegment(
        type=data["type"],
        message_count=int(data["message_count"]),
        token_count=int(data["token_count"]),
        source_event_id=data.get("source_event_id"),
    )


def _graph_call_to_dict(call: GraphCall) -> Dict[str, Any]:
    return {
        "call_id": call.call_id,
        "model": call.model,
        "messages": call.messages,
        "expected_output": call.expected_output,
        "input_segments": [_input_segment_to_dict(seg) for seg in call.input_segments],
        "total_input_tokens": call.total_input_tokens,
        "expected_output_tokens": call.expected_output_tokens,
        "temperature": call.temperature,
        "max_tokens_recorded": call.max_tokens_recorded,
        "tool_definitions": call.tool_definitions,
        "expected_output_is_tool_call": call.expected_output_is_tool_call,
        "expected_output_tool_names": call.expected_output_tool_names,
        "attributes": call.attributes,
    }


def _graph_call_from_dict(data: Dict[str, Any]) -> GraphCall:
    return GraphCall(
        call_id=str(data["call_id"]),
        model=str(data["model"]),
        messages=list(data.get("messages") or []),
        expected_output=str(data.get("expected_output") or ""),
        input_segments=[_input_segment_from_dict(seg) for seg in data.get("input_segments") or []],
        total_input_tokens=int(data["total_input_tokens"]),
        expected_output_tokens=int(data["expected_output_tokens"]),
        temperature=data.get("temperature"),
        max_tokens_recorded=data.get("max_tokens_recorded"),
        tool_definitions=data.get("tool_definitions"),
        expected_output_is_tool_call=bool(data.get("expected_output_is_tool_call", False)),
        expected_output_tool_names=data.get("expected_output_tool_names"),
        attributes=data.get("attributes"),
    )


def _graph_event_to_dict(event: GraphEvent) -> Dict[str, Any]:
    return {
        "event_id": event.event_id,
        "call": _graph_call_to_dict(event.call),
        "predecessor_event_ids": list(event.predecessor_event_ids),
        "predecessor_dependency_types": dict(event.predecessor_dependency_types),
        "wait_ms": event.wait_ms,
        "t_start_ms": event.t_start_ms,
        "t_end_ms": event.t_end_ms,
    }


def _graph_event_from_dict(data: Dict[str, Any]) -> GraphEvent:
    return GraphEvent(
        event_id=str(data["event_id"]),
        call=_graph_call_from_dict(data["call"]),
        predecessor_event_ids=list(data.get("predecessor_event_ids") or []),
        predecessor_dependency_types=dict(data.get("predecessor_dependency_types") or {}),
        wait_ms=int(data["wait_ms"]),
        t_start_ms=int(data["t_start_ms"]),
        t_end_ms=int(data["t_end_ms"]),
    )


def graph_to_payload(graph: ReplayGraph) -> Dict[str, Any]:
    return {
        "events": {event_id: _graph_event_to_dict(event) for event_id, event in graph.events.items()},
        "root_event_ids": list(graph.root_event_ids),
        "source_file": graph.source_file,
    }


def graph_from_payload(data: Dict[str, Any]) -> ReplayGraph:
    events_blob = data.get("events") or {}
    return ReplayGraph(
        events={event_id: _graph_event_from_dict(event) for event_id, event in events_blob.items()},
        root_event_ids=list(data.get("root_event_ids") or []),
        source_file=str(data.get("source_file") or ""),
    )


def stored_session_to_payload(session: StoredSession) -> Dict[str, Any]:
    return {
        "schema_version": session.schema_version,
        "artifact_id": session.artifact_id,
        "source_trace_id": session.source_trace_id,
        "session_headers": session.session_headers,
        "graph": graph_to_payload(session.graph),
    }


def stored_session_from_payload(data: Dict[str, Any]) -> StoredSession:
    schema_version = int(data.get("schema_version", SCHEMA_VERSION))
    if schema_version != SCHEMA_VERSION:
        raise CompileIdentityMismatchError(
            f"Weka compiled session payload schema_version={schema_version} is not supported (expected {SCHEMA_VERSION})"
        )
    artifact_id = str(data["artifact_id"])
    source_trace_id = str(data.get("source_trace_id") or source_trace_id_from_artifact_id(artifact_id))
    headers = data.get("session_headers")
    if headers is not None:
        headers = {str(k): str(v) for k, v in headers.items()}
    return StoredSession(
        schema_version=schema_version,
        artifact_id=artifact_id,
        source_trace_id=source_trace_id,
        session_headers=headers,
        graph=graph_from_payload(data["graph"]),
    )


def encode_session(session: StoredSession, codec: Optional[SessionCodec] = None) -> bytes:
    chosen = codec or get_codec(DEFAULT_CODEC)
    return chosen.encode(stored_session_to_payload(session))


def decode_session(data: bytes, codec: Optional[SessionCodec] = None) -> StoredSession:
    chosen = codec or get_codec(DEFAULT_CODEC)
    return stored_session_from_payload(chosen.decode(data))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_bytes(data)
    os.replace(tmp_path, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


class WekaCompiledSessionStore:
    """Directory-backed compiled ReplayGraph cache. Parent process writes only."""

    def __init__(self, root: Path, identity: CompileIdentity, codec_name: str = DEFAULT_CODEC) -> None:
        self.root = Path(root)
        self.identity = identity
        self.codec_name = codec_name
        self.manifest_path = self.root / "manifest.json"
        self.sessions_dir = self.root / "sessions"
        self.artifact_ids: List[str] = []
        self._artifact_id_set: set[str] = set()
        self._codec = get_codec(codec_name)

    def load(self) -> StoreManifest:
        if not self.manifest_path.is_file():
            logger.info("Weka compiled store: no manifest at %s; treating as empty", self.manifest_path)
            self.artifact_ids = []
            self._artifact_id_set = set()
            return StoreManifest(compile_identity=self.identity, artifact_ids=[])

        blob = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        schema_version = int(blob.get("schema_version", SCHEMA_VERSION))
        if schema_version != SCHEMA_VERSION:
            raise CompileIdentityMismatchError(
                f"Weka compiled session store schema_version={schema_version} is not supported (expected {SCHEMA_VERSION})"
            )
        codec_name = str(blob.get("codec") or "")
        self._codec = get_codec(codec_name)
        self.codec_name = codec_name
        stored_identity = CompileIdentity.from_dict(blob["compile_identity"])
        stored_identity.check_against(self.identity)
        artifact_ids = [str(x) for x in blob.get("artifact_ids") or []]
        self.artifact_ids = artifact_ids
        self._artifact_id_set = set(artifact_ids)
        logger.info(
            "Weka compiled store: identity ok path=%s codec=%s stored=%d tokenizer=%s "
            "corpus=%s corpus_bytes=%d base_seed=%s default_block_size=%s idle_gap_cap=%s",
            self.root,
            codec_name,
            len(artifact_ids),
            stored_identity.tokenizer_name_or_path,
            stored_identity.corpus_path,
            stored_identity.corpus_byte_size,
            stored_identity.base_seed,
            stored_identity.default_block_size,
            stored_identity.trace_idle_gap_cap_seconds,
        )
        return StoreManifest(
            schema_version=schema_version,
            codec=codec_name,
            compile_identity=stored_identity,
            artifact_ids=list(artifact_ids),
        )

    def session_path(self, artifact_id: str) -> Path:
        return self.sessions_dir / safe_artifact_filename(artifact_id, self.codec_name)

    def write_session(self, session: StoredSession) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(self.session_path(session.artifact_id), encode_session(session, self._codec))
        self._remember_artifact_id(session.artifact_id)

    def write_sessions(self, sessions: Sequence[StoredSession]) -> None:
        """Encode and write session files in a parent-process thread pool."""
        if not sessions:
            return
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        workers = store_io_workers(len(sessions))
        logger.info(
            "Weka compiled store: writing %d session(s) workers=%d codec=%s path=%s",
            len(sessions),
            workers,
            self.codec_name,
            self.root,
        )

        def _write_one(session: StoredSession) -> str:
            _atomic_write_bytes(self.session_path(session.artifact_id), encode_session(session, self._codec))
            return session.artifact_id

        if workers == 1:
            written_ids = [_write_one(session) for session in sessions]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                written_ids = list(executor.map(_write_one, sessions))
        for artifact_id in written_ids:
            self._remember_artifact_id(artifact_id)

    def read_session(self, artifact_id: str) -> StoredSession:
        path = self.session_path(artifact_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"Weka compiled session {artifact_id!r} is listed in the store manifest but file is missing: {path}"
            )
        stored = decode_session(path.read_bytes(), self._codec)
        if stored.artifact_id != artifact_id:
            raise ValueError(f"Compiled session file {path} has artifact_id={stored.artifact_id!r}, expected {artifact_id!r}")
        return stored

    def read_sessions(self, artifact_ids: Sequence[str]) -> List[StoredSession]:
        """Decode session files in a parent-process thread pool, preserving input order."""
        if not artifact_ids:
            return []
        workers = store_io_workers(len(artifact_ids))
        logger.info(
            "Weka compiled store: reading %d session(s) workers=%d codec=%s path=%s",
            len(artifact_ids),
            workers,
            self.codec_name,
            self.root,
        )
        if workers == 1:
            return [self.read_session(artifact_id) for artifact_id in artifact_ids]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self.read_session, artifact_ids))

    def _remember_artifact_id(self, artifact_id: str) -> None:
        if artifact_id not in self._artifact_id_set:
            self.artifact_ids.append(artifact_id)
            self._artifact_id_set.add(artifact_id)

    def write_manifest(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "codec": self.codec_name,
            "compile_identity": self.identity.to_dict(),
            "artifact_ids": list(self.artifact_ids),
        }
        _atomic_write_text(self.manifest_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        logger.info(
            "Weka compiled store: wrote manifest %s artifact_ids=%d",
            self.manifest_path,
            len(self.artifact_ids),
        )

    def contains(self, artifact_id: str) -> bool:
        return artifact_id in self._artifact_id_set
