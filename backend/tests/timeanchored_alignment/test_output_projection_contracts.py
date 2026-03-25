from __future__ import annotations

import inspect

import pytest

from app.services.textflow.contracts import SubtitleBatch, SubtitleItem
from app.services.textflow.output_dispatch_adapter import OutputDispatchAdapter
from app.services.timeanchored_alignment.output_projection.output_projector import (
    OutputProjectionInput,
)
from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowChunkBinding,
    WindowCoverage,
)


def _build_coverage() -> WindowCoverage:
    return WindowCoverage(
        core_segments=((10.0, 12.0),),
        left_guard_sec=0.2,
        right_guard_sec=0.3,
        chunk_bindings=(
            WindowChunkBinding(
                chunk_id="chunk-10",
                chunk_index=10,
                chunk_start=10.0,
                chunk_end=11.0,
                overlap_ratio=0.7,
                role="owner",
                is_owner=True,
            ),
            WindowChunkBinding(
                chunk_id="chunk-11",
                chunk_index=11,
                chunk_start=10.9,
                chunk_end=12.0,
                overlap_ratio=0.6,
                role="core",
                is_owner=False,
            ),
        ),
    )


def _build_owner_batch() -> SubtitleBatch:
    return SubtitleBatch(
        chunk_id="chunk-10",
        chunk_index=10,
        items=(
            SubtitleItem(
                segment_id="timeanchored:window-10:seg:0",
                chunk_id="chunk-10",
                start=10.0,
                end=11.6,
                text="测试输出",
                source="render_core",
            ),
        ),
    )


def test_output_projection_min_input_requires_window_source_and_coverage() -> None:
    data = OutputProjectionInput(
        window_id="window-10",
        owner_chunk_id="chunk-10",
        owner_chunk_index=10,
        source_chunk_ids=("chunk-10", "chunk-11"),
        source_chunk_indices=(10, 11),
        coverage=_build_coverage(),
        owner_carrier_batch=_build_owner_batch(),
        decision_metadata={"route": "timeanchored"},
    )

    assert data.owner_carrier_batch.chunk_id == "chunk-10"
    assert data.window_id == "window-10"
    assert data.source_chunk_ids == ("chunk-10", "chunk-11")
    assert data.source_chunk_indices == (10, 11)
    assert len(data.coverage.chunk_bindings) == 2
    assert data.owner_carrier_role == "text_carrier_only"


def test_output_projection_rejects_owner_batch_without_source_provenance_alignment() -> None:
    with pytest.raises(ValueError, match="owner_carrier_batch.chunk_id"):
        OutputProjectionInput(
            window_id="window-10",
            owner_chunk_id="chunk-10",
            owner_chunk_index=10,
            source_chunk_ids=("chunk-10", "chunk-11"),
            source_chunk_indices=(10, 11),
            coverage=_build_coverage(),
            owner_carrier_batch=SubtitleBatch(
                chunk_id="chunk-999",
                chunk_index=999,
                items=(
                    SubtitleItem(
                        segment_id="seg-999",
                        chunk_id="chunk-999",
                        start=10.0,
                        end=11.0,
                        text="错误载体",
                        source="render_core",
                    ),
                ),
            ),
            decision_metadata={},
        )


def test_output_projection_keeps_southbound_replace_chunk_batch_contract() -> None:
    source = inspect.getsource(OutputDispatchAdapter.dispatch)
    assert "replace_chunk_batch(" in source
