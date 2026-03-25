from __future__ import annotations

from pathlib import Path

from app.services.alignment.types import DecisionLayerInput, DecisionLayerOutput
from app.services.punctuation.final_splitter import FinalSplitter
from app.services.textflow.decision_layer import SegmentationProcessor


def test_process_must_not_emit_legacy_fallback_route() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    expected = DecisionLayerOutput(
        sentence_segments=[],
        words_for_split=[],
        segmentation_report={},
    )

    class _DummyCore:
        def process(self, data, *, stream_id, chunk_index, is_last_chunk):
            return expected

    processor._segmentation_core = _DummyCore()
    processor._try_render_with_unified_pipeline = lambda **kwargs: None

    output = processor.process(DecisionLayerInput(annotated_words=[], vad_intervals=[]))

    assert output is expected
    assert output.segmentation_report.get("pipeline_route") != "legacy_fallback"


def test_async_pipeline_no_longer_keeps_sentence_segment_to_batch_final_bridge() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "app"
        / "pipelines"
        / "dual_pipeline"
        / "implementation.py"
    ).read_text(encoding="utf-8")

    assert "_build_output_subtitle_batch_from_sentences(" not in source
    assert "draft_segmenter_final" not in source
    assert "semantic_buffer_final" not in source
    assert "semantic_buffer_flush_final" not in source
