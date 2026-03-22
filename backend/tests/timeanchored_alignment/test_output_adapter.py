from __future__ import annotations

from app.models.sensevoice_models import SentenceSegment
from app.services.timeanchored_alignment.chunk_projector import (
    ChunkProjection,
    ChunkWindow,
)
from app.services.timeanchored_alignment.output_adapter import OutputAdapter


def _segment(text: str, *, start: float, end: float, split_reason: str = "") -> SentenceSegment:
    sentence = SentenceSegment(text=text, text_clean=text, start=start, end=end)
    sentence.split_reason = split_reason
    return sentence


def test_output_adapter_converts_projection_to_output_layer_input() -> None:
    adapter = OutputAdapter()
    projection = ChunkProjection(
        chunk_window=ChunkWindow(chunk_ref="chunk-7", start=14.0, end=16.0),
        sentence_segments=(
            _segment("你好", start=14.1, end=14.9, split_reason="long_pause"),
            _segment("世界", start=15.0, end=15.7, split_reason="tail_flush"),
        ),
    )

    outputs = adapter.to_output_layer_inputs(
        projections=(projection,),
        language="zh",
        injection_report={"mapping_coverage": 1.0},
        segmentation_report={"route": "timeanchored"},
    )

    assert len(outputs) == 1
    payload = outputs[0]
    assert payload.chunk_index == "chunk-7"
    assert payload.language == "zh"
    assert len(payload.sentence_segments) == 2
    assert payload.injection_report == {"mapping_coverage": 1.0}
    assert payload.segmentation_report == {"route": "timeanchored"}
    assert payload.output_traces is not None
    assert payload.output_traces[0].split_reason == "long_pause"


def test_output_adapter_preserves_empty_projection_for_replace_chunk_clear() -> None:
    adapter = OutputAdapter()
    projection = ChunkProjection(
        chunk_window=ChunkWindow(chunk_ref=3, start=6.0, end=8.0),
        sentence_segments=(),
    )

    outputs = adapter.to_output_layer_inputs(projections=(projection,), language="zh")

    assert len(outputs) == 1
    assert outputs[0].chunk_index == 3
    assert outputs[0].sentence_segments == []
