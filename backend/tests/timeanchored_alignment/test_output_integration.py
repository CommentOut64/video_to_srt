from __future__ import annotations

from app.services.alignment.types import OutputLayerInput
from app.services.textflow.output_dispatch_adapter import OutputLayerProcessor
from app.services.timeanchored_alignment.output_projection.output_projector import (
    OutputProjectionInput,
    OutputProjector,
)
from app.services.textflow.contracts import SubtitleBatch, SubtitleItem
from app.services.timeanchored_alignment.slow_window.contracts import WindowChunkBinding, WindowCoverage


class _DummySubtitleManager:
    def __init__(self) -> None:
        self.calls: list[tuple[int | str, int]] = []

    def replace_chunk_batch(self, subtitle_batch):
        self.calls.append((subtitle_batch.chunk_id, len(list(subtitle_batch.items or ()))))
        return list(range(len(list(subtitle_batch.items or ()))))


def _build_projection_input() -> OutputProjectionInput:
    return OutputProjectionInput(
        window_id="window-0",
        owner_chunk_id="chunk-0",
        owner_chunk_index=0,
        source_chunk_ids=("chunk-0", "chunk-1", "chunk-2"),
        source_chunk_indices=(0, 1, 2),
        coverage=WindowCoverage(
            core_segments=((0.0, 2.0),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-0",
                    chunk_index=0,
                    chunk_start=0.0,
                    chunk_end=1.0,
                    overlap_ratio=0.6,
                    role="owner",
                    is_owner=True,
                ),
                WindowChunkBinding(
                    chunk_id="chunk-1",
                    chunk_index=1,
                    chunk_start=1.0,
                    chunk_end=2.0,
                    overlap_ratio=0.8,
                    role="core",
                    is_owner=False,
                ),
                WindowChunkBinding(
                    chunk_id="chunk-2",
                    chunk_index=2,
                    chunk_start=2.0,
                    chunk_end=3.0,
                    overlap_ratio=0.1,
                    role="right_guard",
                    is_owner=False,
                ),
            ),
        ),
        owner_carrier_batch=SubtitleBatch(
            chunk_id="chunk-0",
            chunk_index=0,
            items=(
                SubtitleItem(
                    segment_id="seg-0",
                    chunk_id="chunk-0",
                    start=0.9,
                    end=1.35,
                    text="跨段",
                    source="render_core",
                    trace={"split_reason": "timeanchored"},
                ),
            ),
        ),
        decision_metadata={"route": "timeanchored"},
    )


def test_timeanchored_output_chain_dispatches_window_group_batch() -> None:
    projected_batches = OutputProjector().project(_build_projection_input())
    assert len(projected_batches) == 1

    subtitle_manager = _DummySubtitleManager()
    processor = OutputLayerProcessor(subtitle_manager=subtitle_manager)
    payloads = [
        processor.process(
            OutputLayerInput(
                chunk_index=batch.chunk_index if batch.chunk_index is not None else batch.chunk_id,
                sentence_segments=[],
                language="zh",
                injection_report={},
                segmentation_report={"route": "timeanchored"},
                output_traces=[],
                subtitle_batch=batch,
            )
        ).output_payload
        for batch in projected_batches
    ]

    assert subtitle_manager.calls == [("ow-window-0", 1)]
    assert payloads[0]["sentence_count"] == 1
    assert payloads[0]["source_chunk_ids"] == ["chunk-0", "chunk-1", "chunk-2"]
