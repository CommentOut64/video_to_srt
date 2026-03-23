from __future__ import annotations

from app.models.sensevoice_models import SentenceSegment, WordTimestamp
from app.services.alignment.types import DecisionLayerInput, DecisionLayerOutput, OutputTrace
from app.services.punctuation.final_splitter import FinalSplitter
from app.services.textflow.contracts import CanonicalTextStream, CoreToken
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


def test_decision_processor_marks_legacy_route_when_unified_render_unavailable() -> None:
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
    payload = DecisionLayerInput(annotated_words=[], vad_intervals=[])
    output = processor.process(payload)

    assert output is expected
    assert output.segmentation_report["pipeline_route"] == "legacy_fallback"
    assert output.segmentation_report["fallback_reason"] == "unified_render_unavailable"


def test_build_segmentation_result_from_legacy_segment_id_must_include_chunk_ref() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    canonical_stream = CanonicalTextStream(
        stream_id="timeanchored:main:none",
        chunk_ref="3",
        language="en",
        text_source="slow",
        tokens=(
            CoreToken(
                token_id="tk-0",
                index=0,
                text_core="Hello",
                normalized_text="hello",
                start=0.0,
                end=0.5,
                source="slow",
            ),
            CoreToken(
                token_id="tk-1",
                index=1,
                text_core="world",
                normalized_text="world",
                start=0.5,
                end=1.0,
                source="slow",
            ),
        ),
    )
    legacy_sentence = SentenceSegment(
        text="Hello world",
        text_clean="Hello world",
        start=0.0,
        end=1.0,
    )

    result = processor._build_segmentation_result_from_legacy(
        canonical_stream=canonical_stream,
        sentence_segments=[legacy_sentence],
        output_traces=[],
    )

    assert result.segments[0].segment_id == "timeanchored:main:none:3:seg:0"


def test_try_render_with_unified_pipeline_rebuilds_clean_words_instead_of_copying_legacy() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    legacy_output = DecisionLayerOutput(
        sentence_segments=[
            SentenceSegment(
                text="Hello world??",
                text_clean="Hello world??",
                start=0.0,
                end=1.0,
                words=[
                    WordTimestamp(word="Hello", start=0.0, end=0.5, confidence=0.9),
                    WordTimestamp(word="world??", start=0.5, end=1.0, confidence=0.8),
                ],
            )
        ],
        words_for_split=[
            WordTimestamp(word="Hello", start=0.0, end=0.5, confidence=0.9, confidence_source="slow"),
            WordTimestamp(word="world", start=0.5, end=1.0, confidence=0.8, confidence_source="slow"),
        ],
        output_traces=[OutputTrace(sentence_index=0, split_reason="legacy")],
    )

    output = processor._try_render_with_unified_pipeline(
        legacy_output=legacy_output,
        data=DecisionLayerInput(annotated_words=[], vad_intervals=[]),
        stream_id="timeanchored:main:none",
        chunk_index=3,
    )

    assert output is not None
    assert output.sentence_segments[0].segment_id == "timeanchored:main:none:3:seg:0"
    assert [word.word for word in output.sentence_segments[0].words] == ["Hello", "world"]


def test_strip_boundary_residual_weak_punct_removes_trailing_comma_but_keeps_question_mark() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    sentences = [
        SentenceSegment(
            text="Then tomorrow we can celebrate her birthday,",
            text_clean="Then tomorrow we can celebrate her birthday,",
            words=[WordTimestamp(word="birthday,", start=0.0, end=1.0, confidence=1.0)],
        ),
        SentenceSegment(
            text="and maybe even get her a lava lamp",
            text_clean="and maybe even get her a lava lamp",
            words=[WordTimestamp(word="and", start=1.0, end=1.1, confidence=1.0)],
        ),
        SentenceSegment(
            text="Who is my favorite DDLC character?",
            text_clean="Who is my favorite DDLC character?",
            words=[WordTimestamp(word="character?", start=2.0, end=3.0, confidence=1.0)],
        ),
        SentenceSegment(
            text="Probably Monika",
            text_clean="Probably Monika",
            words=[WordTimestamp(word="Probably", start=3.0, end=3.2, confidence=1.0)],
        ),
    ]

    processor._strip_boundary_residual_weak_punct(sentences)

    assert sentences[0].text == "Then tomorrow we can celebrate her birthday"
    assert sentences[0].text_clean == "Then tomorrow we can celebrate her birthday"
    assert sentences[0].words[-1].word == "birthday"
    assert sentences[2].text == "Who is my favorite DDLC character?"
    assert sentences[2].words[-1].word == "character?"
