from __future__ import annotations

from app.services.textflow.output_dispatch_adapter import OutputLayerProcessor
from app.services.timeanchored_alignment.chunk_projector import ChunkProjector, ChunkWindow
from app.services.timeanchored_alignment.output_adapter import OutputAdapter
from app.services.timeanchored_alignment.sentence_segmenter import SentenceSegmenter
from app.services.timeanchored_alignment.subtitle_assembler import SubtitleAssembler
from app.services.timeanchored_alignment.contracts import (
    AlignmentItem,
    AlignmentMetrics,
    FinalAlignmentResult,
)


class _DummySubtitleManager:
    def __init__(self) -> None:
        self.calls: list[tuple[int | str, int]] = []

    def replace_chunk_batch(self, subtitle_batch):
        self.calls.append((subtitle_batch.chunk_id, len(list(subtitle_batch.items or ()))))
        return list(range(len(list(subtitle_batch.items or ()))))


def _build_result() -> FinalAlignmentResult:
    items = (
        AlignmentItem(
            text="跨",
            start=0.9,
            end=1.05,
            status="direct",
            source="text_aligner",
            confidence=0.9,
        ),
        AlignmentItem(
            text="段",
            start=1.05,
            end=1.35,
            status="direct",
            source="text_aligner",
            confidence=0.9,
        ),
    )
    return FinalAlignmentResult(
        items=items,
        route="text",
        metrics=AlignmentMetrics(
            coverage=1.0,
            duration_ratio=1.0,
            failed_count=0,
            route_confidence=0.9,
        ),
    )


def test_timeanchored_output_chain_supports_empty_replace_chunk_cleanup() -> None:
    assembler = SubtitleAssembler()
    segmenter = SentenceSegmenter()
    projector = ChunkProjector()
    adapter = OutputAdapter()

    stream = assembler.assemble(base_result=_build_result())
    segments = segmenter.segment(stream=stream, language="zh")
    projections = projector.project(
        sentence_segments=segments,
        chunk_windows=(
            ChunkWindow(chunk_ref=0, start=0.0, end=1.0),
            ChunkWindow(chunk_ref=1, start=1.0, end=2.0),
            ChunkWindow(chunk_ref=2, start=2.0, end=3.0),
        ),
    )
    outputs = adapter.to_output_layer_inputs(
        projections=projections,
        language="zh",
        segmentation_report={"route": "timeanchored"},
    )

    subtitle_manager = _DummySubtitleManager()
    processor = OutputLayerProcessor(subtitle_manager=subtitle_manager)
    payloads = [processor.process(item).output_payload for item in outputs]

    assert subtitle_manager.calls == [("0", 0), ("1", 1), ("2", 0)]
    assert payloads[0]["sentence_count"] == 0
    assert payloads[1]["sentence_count"] == 1
    assert payloads[2]["sentence_count"] == 0
