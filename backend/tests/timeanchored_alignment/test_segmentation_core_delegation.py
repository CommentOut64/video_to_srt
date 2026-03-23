from __future__ import annotations

from app.services.alignment.types import DecisionLayerInput, DecisionLayerOutput
from app.services.punctuation.final_splitter import FinalSplitter
from app.services.textflow.decision_layer import SegmentationProcessor
from app.services.textflow.segmentation_core import SegmentationCore


def test_segmentation_core_forwards_call_to_impl() -> None:
    recorded: dict[str, object] = {}

    def _impl(*, data, stream_id, chunk_index, is_last_chunk):
        recorded["data"] = data
        recorded["stream_id"] = stream_id
        recorded["chunk_index"] = chunk_index
        recorded["is_last_chunk"] = is_last_chunk
        return DecisionLayerOutput(sentence_segments=[], words_for_split=[])

    core = SegmentationCore(process_impl=_impl)
    payload = DecisionLayerInput(annotated_words=[], vad_intervals=[])
    result = core.process(
        payload,
        stream_id="unit-stream",
        chunk_index=7,
        is_last_chunk=True,
    )

    assert recorded["data"] is payload
    assert recorded["stream_id"] == "unit-stream"
    assert recorded["chunk_index"] == 7
    assert recorded["is_last_chunk"] is True
    assert isinstance(result, DecisionLayerOutput)


def test_decision_processor_process_delegates_to_segmentation_core() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    called: dict[str, object] = {}
    expected = DecisionLayerOutput(
        sentence_segments=[],
        words_for_split=[],
        segmentation_report={"delegated": True},
    )

    class _DummyCore:
        def process(self, data, *, stream_id, chunk_index, is_last_chunk):
            called["data"] = data
            called["stream_id"] = stream_id
            called["chunk_index"] = chunk_index
            called["is_last_chunk"] = is_last_chunk
            return expected

    processor._segmentation_core = _DummyCore()
    payload = DecisionLayerInput(annotated_words=[], vad_intervals=[])
    output = processor.process(
        payload,
        stream_id="legacy:main:none",
        chunk_index=1,
        is_last_chunk=False,
    )

    assert output is expected
    assert called["data"] is payload
    assert called["stream_id"] == "legacy:main:none"
    assert called["chunk_index"] == 1
    assert called["is_last_chunk"] is False
