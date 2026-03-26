from __future__ import annotations

from app.services.textflow.contracts import SubtitleBatch, SubtitleItem
from app.services.timeanchored_alignment.output_projection.output_projector import (
    OutputProjectionInput,
    OutputProjector,
)
from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowChunkBinding,
    WindowCoverage,
)


def _build_input() -> OutputProjectionInput:
    coverage = WindowCoverage(
        core_segments=((0.0, 3.0),),
        left_guard_sec=0.0,
        right_guard_sec=0.0,
        chunk_bindings=(
            WindowChunkBinding(
                chunk_id="chunk-0",
                chunk_index=0,
                chunk_start=0.0,
                chunk_end=1.5,
                overlap_ratio=0.5,
                role="owner",
                is_owner=True,
            ),
            WindowChunkBinding(
                chunk_id="chunk-1",
                chunk_index=1,
                chunk_start=1.5,
                chunk_end=3.0,
                overlap_ratio=0.5,
                role="core",
                is_owner=False,
            ),
            WindowChunkBinding(
                chunk_id="chunk-2",
                chunk_index=2,
                chunk_start=3.0,
                chunk_end=4.0,
                overlap_ratio=0.1,
                role="right_guard",
                is_owner=False,
            ),
        ),
    )
    owner_batch = SubtitleBatch(
        chunk_id="chunk-0",
        chunk_index=0,
        items=(
            SubtitleItem(
                segment_id="seg-0",
                chunk_id="chunk-0",
                start=0.1,
                end=1.3,
                text="第一句",
                source="render_core",
                trace={
                    "split_reason": "boundary_a",
                    "mapped_cut_time": 1.3,
                    "mapping_quality": "boundary",
                },
            ),
            SubtitleItem(
                segment_id="seg-1",
                chunk_id="chunk-0",
                start=1.7,
                end=2.8,
                text="第二句",
                source="render_core",
                trace={
                    "split_reason": "boundary_b",
                    "mapped_cut_time": 2.8,
                    "mapping_quality": "boundary",
                },
            ),
        ),
        diagnostics={"output_trace": [{"sentence_index": 0}, {"sentence_index": 1}]},
    )
    return OutputProjectionInput(
        window_id="window-0",
        owner_chunk_id="chunk-0",
        owner_chunk_index=0,
        source_chunk_ids=("chunk-0", "chunk-1", "chunk-2"),
        source_chunk_indices=(0, 1, 2),
        coverage=coverage,
        owner_carrier_batch=owner_batch,
        decision_metadata={"route": "timeanchored"},
    )


def test_output_projector_emits_single_output_group_batch_with_scope_metadata() -> None:
    projected = OutputProjector().project(_build_input())
    assert len(projected) == 1
    batch = projected[0]
    assert batch.chunk_id == "ow-window-0"
    assert batch.chunk_index is None
    assert [item.text for item in batch.items] == ["第一句", "第二句"]
    projection_meta = dict(batch.diagnostics.get("projection") or {})
    assert projection_meta["projection_mode"] == "window_group"
    assert projection_meta["replace_scope_chunk_ids"] == ["chunk-0", "chunk-1", "chunk-2"]
    assert projection_meta["replace_scope_chunk_indices"] == [0, 1, 2]


def test_output_projector_does_not_reinfer_boundaries_from_owner_items() -> None:
    projected = OutputProjector().project(_build_input())
    assert len(projected) == 1
    first = projected[0].items[0]
    second = projected[0].items[1]

    assert first.start == 0.1
    assert first.end == 1.3
    assert first.trace["split_reason"] == "boundary_a"
    assert first.trace["mapped_cut_time"] == 1.3
    assert second.start == 1.7
    assert second.end == 2.8
    assert second.trace["split_reason"] == "boundary_b"
    assert second.trace["mapped_cut_time"] == 2.8
