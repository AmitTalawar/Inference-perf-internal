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
"""Measurement-window filtering for time-boxed trace_session_replay stages."""

from unittest.mock import Mock

from inference_perf.client.server_metrics.base import StageRuntimeInfo, StageStatus
from inference_perf.reportgen.base import (
    build_stage_request_metadata,
    filter_metrics_to_measurement_window,
    measurement_window_bounds,
    request_in_measurement_window,
)


def _stage(
    *,
    warmup: float | None = 5.0,
    duration: float | None = 20.0,
    start_perf: float = 1000.0,
    end_perf: float = 1020.0,
) -> StageRuntimeInfo:
    return StageRuntimeInfo(
        stage_id=0,
        rate=0.0,
        start_time=1_700_000_000.0,
        end_time=1_700_000_020.0,
        status=StageStatus.COMPLETED,
        concurrency_level=4,
        duration=duration,
        warmup=warmup,
        start_time_perf=start_perf,
        end_time_perf=end_perf,
    )


def _metric(start_time: float, stage_id: int = 0) -> Mock:
    m = Mock()
    m.stage_id = stage_id
    m.start_time = start_time
    return m


def test_measurement_window_bounds_none_without_warmup() -> None:
    assert measurement_window_bounds(_stage(warmup=None)) is None
    assert measurement_window_bounds(_stage(warmup=0.0)) is None


def test_measurement_window_bounds_applies_warmup_and_duration() -> None:
    bounds = measurement_window_bounds(_stage(warmup=5.0, duration=20.0, start_perf=1000.0, end_perf=1025.0))
    assert bounds == (1005.0, 1020.0)  # end capped by duration, not late end_perf


def test_filter_excludes_warmup_requests() -> None:
    stage = _stage(warmup=5.0, duration=20.0, start_perf=1000.0, end_perf=1020.0)
    stages = {0: stage}
    metrics = [
        _metric(1001.0),  # warmup
        _metric(1005.0),  # measure start (inclusive)
        _metric(1010.0),  # measure
        _metric(1020.0),  # at/after end (exclusive)
    ]
    filtered = filter_metrics_to_measurement_window(metrics, stages)  # type: ignore[arg-type]
    assert [m.start_time for m in filtered] == [1005.0, 1010.0]


def test_request_in_measurement_window_passthrough_without_warmup() -> None:
    stage = _stage(warmup=None)
    assert request_in_measurement_window(_metric(1001.0), {0: stage})  # type: ignore[arg-type]


def test_build_stage_request_metadata_includes_window() -> None:
    meta = build_stage_request_metadata(_stage(warmup=5.0, duration=20.0))
    assert meta["duration_configured"] == 20.0
    assert meta["warmup_configured"] == 5.0
    assert meta["measurement_window_sec"]["start_offset"] == 5.0
    assert meta["measurement_window_sec"]["length"] == 15.0
