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

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType
from inference_perf.config.datagen.replay import WekaTraceReplayConfig
from inference_perf.datagen.replay_graph_types import GraphCall, GraphEvent, InputSegment, ReplayGraph
from inference_perf.datagen.weka_compiled_store import (
    CODEC_JSON_GZ,
    CODEC_ORJSON_ZST,
    CompileIdentity,
    CompileIdentityMismatchError,
    StoredSession,
    UnknownSessionCodecError,
    WekaCompiledSessionStore,
    decode_session,
    encode_session,
    get_codec,
    max_dupseed_suffix,
    safe_artifact_filename,
    source_trace_id_from_artifact_id,
)
from inference_perf.datagen.weka_trace_replay_datagen import WekaTraceReplayDataGenerator


TWO_TURN_TRACE: Dict[str, Any] = {
    "id": "mock_trace_123",
    "models": ["stored-model-a"],
    "block_size": 2,
    "tool_tokens": 0,
    "system_tokens": 0,
    "session_headers": {"X-Session": "trace-token"},
    "requests": [
        {
            "t": 0.1,
            "type": "n",
            "model": "stored-model-a",
            "in": 4,
            "out": 2,
            "hash_ids": [10, 20],
            "api_time": 0.5,
        },
        {
            "t": 1.2,
            "type": "n",
            "model": "stored-model-a",
            "in": 8,
            "out": 4,
            "hash_ids": [10, 20, 30, 40],
            "api_time": 0.8,
        },
    ],
}


def _tiny_graph() -> ReplayGraph:
    call = GraphCall(
        call_id="parent_turn_0",
        model="stored-model-a",
        messages=[{"role": "user", "content": "hello"}],
        expected_output="world",
        input_segments=[InputSegment(type="unique", message_count=1, token_count=4)],
        total_input_tokens=4,
        expected_output_tokens=2,
        temperature=0.0,
        max_tokens_recorded=2,
    )
    event = GraphEvent(
        event_id="e0",
        call=call,
        predecessor_event_ids=[],
        predecessor_dependency_types={},
        wait_ms=0,
        t_start_ms=100,
        t_end_ms=600,
    )
    return ReplayGraph(events={"e0": event}, root_event_ids=["e0"], source_file="weka_trace_mock")


def test_encode_decode_session_roundtrip(tmp_path: Path) -> None:
    stored = StoredSession(
        artifact_id="abc__dupseed3",
        source_trace_id="abc",
        session_headers={"Authorization": "Bearer t"},
        graph=_tiny_graph(),
    )
    blob = encode_session(stored)
    restored = decode_session(blob)
    assert restored.artifact_id == "abc__dupseed3"
    assert restored.source_trace_id == "abc"
    assert restored.session_headers == {"Authorization": "Bearer t"}
    assert restored.graph == stored.graph


def test_json_gz_codec_round_trip() -> None:
    stored = StoredSession(
        artifact_id="abc",
        source_trace_id="abc",
        session_headers=None,
        graph=_tiny_graph(),
    )
    codec = get_codec(CODEC_JSON_GZ)
    restored = decode_session(encode_session(stored, codec), codec)
    assert restored.graph == stored.graph


def test_safe_artifact_filename_encodes_unsafe_chars() -> None:
    artifact_id = "trace::sa:agent/id"
    filename = safe_artifact_filename(artifact_id)
    assert "/" not in filename
    assert ":" not in filename
    assert filename.endswith(".orjson.zst")
    assert safe_artifact_filename(artifact_id, CODEC_JSON_GZ).endswith(".json.gz")


def test_source_trace_id_and_dupseed_suffix() -> None:
    assert source_trace_id_from_artifact_id("abc") == "abc"
    assert source_trace_id_from_artifact_id("abc__dupseed3") == "abc"
    assert max_dupseed_suffix(["abc", "abc__dupseed1", "other__dupseed7"]) == 7


def test_unknown_codec_fails() -> None:
    with pytest.raises(UnknownSessionCodecError):
        get_codec("pickle")


def test_identity_mismatch_message() -> None:
    stored = CompileIdentity(
        tokenizer_name_or_path="tok-a",
        corpus_path="/corpus",
        corpus_byte_size=10,
        base_seed=42,
        default_block_size=64,
        trace_idle_gap_cap_seconds=60.0,
    )
    current = CompileIdentity(
        tokenizer_name_or_path="tok-a",
        corpus_path="/corpus",
        corpus_byte_size=10,
        base_seed=43,
        default_block_size=64,
        trace_idle_gap_cap_seconds=60.0,
    )
    with pytest.raises(CompileIdentityMismatchError, match="base_seed"):
        stored.check_against(current)


def test_store_atomic_manifest_and_session_files(tmp_path: Path) -> None:
    identity = CompileIdentity(
        tokenizer_name_or_path="mock-tokenizer",
        corpus_path=str(tmp_path / "corpus.txt"),
        corpus_byte_size=1,
        base_seed=42,
        default_block_size=64,
        trace_idle_gap_cap_seconds=60.0,
    )
    store = WekaCompiledSessionStore(tmp_path / "store", identity)
    store.load()
    stored = StoredSession(
        artifact_id="orig-a__dupseed1",
        source_trace_id="orig-a",
        session_headers=None,
        graph=_tiny_graph(),
    )
    store.write_session(stored)
    store.write_manifest()
    reloaded = WekaCompiledSessionStore(tmp_path / "store", identity)
    manifest = reloaded.load()
    assert manifest.codec == CODEC_ORJSON_ZST
    assert manifest.artifact_ids == ["orig-a__dupseed1"]
    assert reloaded.read_session("orig-a__dupseed1").graph == stored.graph
    assert (tmp_path / "store" / "sessions" / "orig-a__dupseed1.orjson.zst").is_file()


def test_existing_json_gz_store_stays_on_gzip(tmp_path: Path) -> None:
    identity = CompileIdentity(
        tokenizer_name_or_path="mock-tokenizer",
        corpus_path=str(tmp_path / "corpus.txt"),
        corpus_byte_size=1,
        base_seed=42,
        default_block_size=64,
        trace_idle_gap_cap_seconds=60.0,
    )
    store = WekaCompiledSessionStore(tmp_path / "store", identity, codec_name=CODEC_JSON_GZ)
    store.load()
    first = StoredSession(
        artifact_id="orig-a",
        source_trace_id="orig-a",
        session_headers=None,
        graph=_tiny_graph(),
    )
    store.write_session(first)
    store.write_manifest()
    reloaded = WekaCompiledSessionStore(tmp_path / "store", identity)
    reloaded.load()
    assert reloaded.codec_name == CODEC_JSON_GZ
    second = StoredSession(
        artifact_id="orig-b",
        source_trace_id="orig-b",
        session_headers=None,
        graph=_tiny_graph(),
    )
    reloaded.write_sessions([second])
    reloaded.write_manifest()
    assert (tmp_path / "store" / "sessions" / "orig-a.json.gz").is_file()
    assert (tmp_path / "store" / "sessions" / "orig-b.json.gz").is_file()
    assert list((tmp_path / "store" / "sessions").glob("*.orjson.zst")) == []
    assert reloaded.read_sessions(["orig-a", "orig-b"])[0].artifact_id == "orig-a"


def _write_trace_file(path: Path, trace: Dict[str, Any]) -> Path:
    path.write_text(json.dumps(trace), encoding="utf-8")
    return path


def _make_tokenizer() -> Tuple[MagicMock, MagicMock, List[int]]:
    encode_calls: List[int] = []
    inner = MagicMock()
    inner.name_or_path = "mock-tokenizer"

    def encode(text: str) -> List[int]:
        encode_calls.append(1)
        return list(range(len(text)))

    inner.encode = encode
    inner.decode = lambda tokens: ",".join(str(i) for i in tokens)
    tokenizer = MagicMock()
    tokenizer.get_tokenizer.return_value = inner
    return tokenizer, inner, encode_calls


def _make_generator(
    trace_file: Path,
    *,
    store_path: Optional[Path] = None,
    duplicate_sessions_target: Optional[int] = None,
    num_dataset_entries: int = 100,
    base_seed: int = 42,
    use_static_model: bool = False,
    static_model_name: str = "",
    tokenizer: Optional[MagicMock] = None,
) -> Tuple[WekaTraceReplayDataGenerator, List[int]]:
    if tokenizer is None:
        tokenizer, _, encode_calls = _make_tokenizer()
    else:
        encode_calls = []

    api_cfg = APIConfig(type=APIType.Chat, streaming=False)
    data_cfg = DataConfig(type=DataGenType.WekaTraceReplay)
    kwargs: Dict[str, Any] = {
        "trace_files": [str(trace_file)],
        "default_block_size": 2,
        "num_dataset_entries": num_dataset_entries,
        "skip_invalid_files": True,
    }
    if store_path is not None:
        kwargs["compiled_store_path"] = str(store_path)
    if duplicate_sessions_target is not None:
        kwargs["duplicate_sessions_target"] = duplicate_sessions_target
    if use_static_model:
        kwargs["use_static_model"] = True
        kwargs["static_model_name"] = static_model_name
    data_cfg.weka_trace_replay = WekaTraceReplayConfig(**kwargs)
    gen = WekaTraceReplayDataGenerator(
        api_config=api_cfg,
        config=data_cfg,
        tokenizer=tokenizer,
        num_workers=1,
        base_seed=base_seed,
    )
    return gen, encode_calls


def _graphs_by_source(gen: WekaTraceReplayDataGenerator) -> Dict[str, ReplayGraph]:
    return {session.source_id: session.graph for session in gen.sessions if session is not None}


def _session_text(session: Any) -> str:
    parts: List[str] = []
    for event in session.graph.events.values():
        for msg in event.call.messages:
            parts.append(str(msg.get("content", "")))
        parts.append(event.call.expected_output)
    return "".join(parts)


def test_write_through_then_full_hit_skips_corpus_and_raw_load(tmp_path: Path) -> None:
    trace_file = _write_trace_file(tmp_path / "mock_trace.json", TWO_TURN_TRACE)
    store_path = tmp_path / "compiled_store"
    tokenizer1, _, encode_calls1 = _make_tokenizer()
    gen1, _ = _make_generator(
        trace_file,
        store_path=store_path,
        duplicate_sessions_target=1,
        tokenizer=tokenizer1,
    )
    assert encode_calls1
    graphs1 = _graphs_by_source(gen1)
    assert "mock_trace_123" in graphs1

    trace_file.unlink()
    tokenizer2, _, encode_calls2 = _make_tokenizer()
    gen2, _ = _make_generator(
        trace_file,
        store_path=store_path,
        duplicate_sessions_target=1,
        tokenizer=tokenizer2,
    )
    assert encode_calls2 == []
    graphs2 = _graphs_by_source(gen2)
    assert graphs1.keys() == graphs2.keys()
    for source_id, graph in graphs1.items():
        assert graphs2[source_id] == graph


def test_identity_mismatch_on_base_seed_raises(tmp_path: Path) -> None:
    trace_file = _write_trace_file(tmp_path / "mock_trace.json", TWO_TURN_TRACE)
    store_path = tmp_path / "compiled_store"
    _make_generator(trace_file, store_path=store_path, duplicate_sessions_target=1, base_seed=42)
    with pytest.raises(CompileIdentityMismatchError, match="base_seed"):
        _make_generator(trace_file, store_path=store_path, duplicate_sessions_target=1, base_seed=43)


def test_identity_ignores_static_model_name(tmp_path: Path) -> None:
    trace_file = _write_trace_file(tmp_path / "mock_trace.json", TWO_TURN_TRACE)
    store_path = tmp_path / "compiled_store"
    gen1, _ = _make_generator(
        trace_file,
        store_path=store_path,
        duplicate_sessions_target=1,
        use_static_model=True,
        static_model_name="model-a",
    )
    stored_models = {event.call.model for session in gen1.sessions if session for event in session.graph.events.values()}
    assert stored_models == {"model-a"}

    gen2, _ = _make_generator(
        trace_file,
        store_path=store_path,
        duplicate_sessions_target=1,
        use_static_model=True,
        static_model_name="model-b",
    )
    # Stored GraphCall.model is not overlaid; live requests use server.model_name.
    hit_models = {event.call.model for session in gen2.sessions if session for event in session.graph.events.values()}
    assert hit_models == {"model-a"}
    assert len(gen2.sessions) == 1


def test_duplicates_write_distinct_files_and_text(tmp_path: Path) -> None:
    trace_file = _write_trace_file(tmp_path / "mock_trace.json", TWO_TURN_TRACE)
    store_path = tmp_path / "compiled_store"
    gen, _ = _make_generator(
        trace_file,
        store_path=store_path,
        duplicate_sessions_target=3,
        num_dataset_entries=1,
    )
    session_files = sorted((store_path / "sessions").glob("*.orjson.zst"))
    assert len(session_files) == 3
    source_ids = sorted(session.source_id for session in gen.sessions if session is not None)
    assert source_ids == ["mock_trace_123", "mock_trace_123__dupseed1", "mock_trace_123__dupseed2"]
    texts = [_session_text(session) for session in gen.sessions if session is not None]
    assert len(set(texts)) == 3


def test_headers_round_trip_on_full_hit_without_raw_file(tmp_path: Path) -> None:
    trace_file = _write_trace_file(tmp_path / "mock_trace.json", TWO_TURN_TRACE)
    store_path = tmp_path / "compiled_store"
    _make_generator(trace_file, store_path=store_path, duplicate_sessions_target=1)
    trace_file.unlink()
    gen2, _ = _make_generator(trace_file, store_path=store_path, duplicate_sessions_target=1)
    assert gen2._session_headers_by_session_id
    assert any(headers.get("X-Session") == "trace-token" for headers in gen2._session_headers_by_session_id.values())


def test_partial_fill_then_full_hit(tmp_path: Path) -> None:
    trace_file = _write_trace_file(tmp_path / "mock_trace.json", TWO_TURN_TRACE)
    store_path = tmp_path / "compiled_store"
    _make_generator(trace_file, store_path=store_path, duplicate_sessions_target=1, num_dataset_entries=1)
    assert len(list((store_path / "sessions").glob("*.orjson.zst"))) == 1

    tokenizer_fill, _, encode_calls_fill = _make_tokenizer()
    gen_fill, _ = _make_generator(
        trace_file,
        store_path=store_path,
        duplicate_sessions_target=2,
        num_dataset_entries=1,
        tokenizer=tokenizer_fill,
    )
    assert encode_calls_fill
    assert len(list((store_path / "sessions").glob("*.orjson.zst"))) == 2
    assert len(gen_fill.sessions) == 2

    tokenizer_hit, _, encode_calls_hit = _make_tokenizer()
    gen_hit, _ = _make_generator(
        trace_file,
        store_path=store_path,
        duplicate_sessions_target=2,
        num_dataset_entries=1,
        tokenizer=tokenizer_hit,
    )
    assert encode_calls_hit == []
    assert len(gen_hit.sessions) == 2
