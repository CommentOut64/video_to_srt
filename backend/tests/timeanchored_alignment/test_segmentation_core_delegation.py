from __future__ import annotations

import pytest

from app.models.sensevoice_models import SentenceSegment, WordTimestamp
from app.services.alignment.types import AnnotatedWord, DecisionLayerInput, DecisionLayerOutput, OutputTrace
from app.services.punctuation.final_splitter import FinalSplitter
from app.services.textflow.contracts import SegmentPlan, SegmentationResult
from app.services.textflow.decision_layer import SegmentationProcessor
from app.services.textflow.segmentation_core import SegmentationCore


def test_segmentation_core_forwards_call_to_processor_runtime() -> None:
    recorded: dict[str, object] = {}

    class _DummyProcessor:
        def _run_segmentation_core(self, *, data, stream_id, chunk_index, is_last_chunk):
            recorded["data"] = data
            recorded["stream_id"] = stream_id
            recorded["chunk_index"] = chunk_index
            recorded["is_last_chunk"] = is_last_chunk
            return DecisionLayerOutput(sentence_segments=[], words_for_split=[])

    core = SegmentationCore(processor=_DummyProcessor())
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
    assert output.segmentation_report["pipeline_route"] == "canonical_segmentation_render"


def test_decision_processor_marks_unified_route_even_when_output_is_empty() -> None:
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
    payload = DecisionLayerInput(annotated_words=[], vad_intervals=[])
    output = processor.process(payload)

    assert output is expected
    assert output.segmentation_report["pipeline_route"] == "canonical_segmentation_render"
    assert "fallback_reason" not in output.segmentation_report


def test_build_split_exit_segmentation_result_segment_id_must_include_chunk_ref() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    legacy_sentence = SentenceSegment(
        text="Hello world",
        text_clean="Hello world",
        start=0.0,
        end=1.0,
        words=[
            WordTimestamp(word="Hello", start=0.0, end=0.5, confidence=0.9),
            WordTimestamp(word="world", start=0.5, end=1.0, confidence=0.8),
        ],
    )

    result = processor._build_split_exit_segmentation_result(
        stream_id="timeanchored:main:none",
        chunk_ref="3",
        words_for_split=list(legacy_sentence.words),
        sentence_segments=[legacy_sentence],
        output_traces=[OutputTrace(sentence_index=0, split_reason="cut_plan")],
        source="cut_plan",
    )

    assert result.segments[0].segment_id == "timeanchored:main:none:3:seg:0"


def test_render_with_unified_pipeline_rebuilds_clean_words_instead_of_copying_legacy() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    segmentation_output = DecisionLayerOutput(
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
        segmentation_result=SegmentationResult(
            segments=(
                SegmentPlan(
                    segment_id="timeanchored:main:none:3:seg:0",
                    token_start=0,
                    token_end=1,
                    start=0.0,
                    end=1.0,
                    boundary_reason="legacy",
                    boundary_score=1.0,
                ),
            )
        ),
    )

    output = processor._render_with_unified_pipeline(
        segmentation_output=segmentation_output,
        data=DecisionLayerInput(annotated_words=[], vad_intervals=[]),
        stream_id="timeanchored:main:none",
        chunk_index=3,
    )

    assert output is not None
    assert output.sentence_segments[0].segment_id == "timeanchored:main:none:3:seg:0"
    assert [word.word for word in output.sentence_segments[0].words] == ["Hello", "world"]


def test_render_with_unified_pipeline_emits_subtitle_batch_without_legacy_sentence_bridge() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    segmentation_output = DecisionLayerOutput(
        sentence_segments=[
            SentenceSegment(
                text="你好世界",
                text_clean="你好世界",
                start=0.0,
                end=1.0,
            )
        ],
        words_for_split=[
            WordTimestamp(word="你好", start=0.0, end=0.4, confidence=0.9, confidence_source="slow"),
            WordTimestamp(word="世界", start=0.4, end=1.0, confidence=0.8, confidence_source="slow"),
        ],
        output_traces=[OutputTrace(sentence_index=0, split_reason="legacy")],
        segmentation_result=SegmentationResult(
            segments=(
                SegmentPlan(
                    segment_id="timeanchored:main:none:3:seg:0",
                    token_start=0,
                    token_end=1,
                    start=0.0,
                    end=1.0,
                    boundary_reason="legacy",
                    boundary_score=1.0,
                ),
            )
        ),
    )

    output = processor._render_with_unified_pipeline(
        segmentation_output=segmentation_output,
        data=DecisionLayerInput(annotated_words=[], vad_intervals=[]),
        stream_id="timeanchored:main:none",
        chunk_index=3,
    )

    assert output.subtitle_batch is not None
    assert output.subtitle_batch.chunk_id == "3"
    assert [item.text for item in output.subtitle_batch.items] == ["你好世界"]
    assert output.render_result is not None


def test_render_with_unified_pipeline_prefers_existing_segmentation_result() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    segmentation_output = DecisionLayerOutput(
        sentence_segments=[
            SentenceSegment(
                text="Hello world",
                text_clean="Hello world",
                start=0.0,
                end=1.0,
            )
        ],
        words_for_split=[
            WordTimestamp(word="Hello", start=0.0, end=0.5, confidence=0.9, confidence_source="slow"),
            WordTimestamp(word="world", start=0.5, end=1.0, confidence=0.8, confidence_source="slow"),
        ],
        output_traces=[OutputTrace(sentence_index=0, split_reason="segmentation_core")],
        segmentation_result=SegmentationResult(
            segments=(
                SegmentPlan(
                    segment_id="timeanchored:main:none:3:seg:0",
                    token_start=0,
                    token_end=1,
                    start=0.0,
                    end=1.0,
                    boundary_reason="segmentation_core",
                    boundary_score=1.0,
                ),
            )
        ),
    )

    output = processor._render_with_unified_pipeline(
        segmentation_output=segmentation_output,
        data=DecisionLayerInput(annotated_words=[], vad_intervals=[]),
        stream_id="timeanchored:main:none",
        chunk_index=3,
    )

    assert output.segmentation_result is not None
    assert output.subtitle_batch is not None
    assert [item.text for item in output.subtitle_batch.items] == ["Hello world"]


def test_render_with_unified_pipeline_requires_segmentation_result_from_segmentation_core() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    segmentation_output = DecisionLayerOutput(
        sentence_segments=[
            SentenceSegment(
                text="Hello world",
                text_clean="Hello world",
                start=0.0,
                end=1.0,
            )
        ],
        words_for_split=[
            WordTimestamp(word="Hello", start=0.0, end=0.5, confidence=0.9, confidence_source="slow"),
            WordTimestamp(word="world", start=0.5, end=1.0, confidence=0.8, confidence_source="slow"),
        ],
        output_traces=[OutputTrace(sentence_index=0, split_reason="cut_plan")],
        segmentation_result=None,
    )

    with pytest.raises(ValueError, match="segmentation_result"):
        processor._render_with_unified_pipeline(
            segmentation_output=segmentation_output,
            data=DecisionLayerInput(annotated_words=[], vad_intervals=[]),
            stream_id="timeanchored:main:none",
            chunk_index=3,
        )


def test_render_with_unified_pipeline_keeps_multiple_chinese_segments_after_canonical_rebuild() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    words = [
        WordTimestamp(word="幸运的他", start=0.0, end=0.5, confidence=0.98, confidence_source="fast"),
        WordTimestamp(word="躲过了", start=0.5, end=1.0, confidence=0.97, confidence_source="fast"),
        WordTimestamp(word="一劫", start=1.0, end=1.4, confidence=0.96, confidence_source="fast"),
        WordTimestamp(word="警方", start=1.5, end=1.9, confidence=0.95, confidence_source="fast"),
        WordTimestamp(word="一共", start=1.9, end=2.2, confidence=0.94, confidence_source="fast"),
        WordTimestamp(word="发现了毒可乐", start=2.2, end=2.9, confidence=0.93, confidence_source="fast"),
    ]
    segmentation_output = DecisionLayerOutput(
        sentence_segments=[
            SentenceSegment(
                text="幸运的他躲过了一劫",
                text_clean="幸运的他躲过了一劫",
                start=0.0,
                end=1.4,
                words=words[:3],
            ),
            SentenceSegment(
                text="警方一共发现了毒可乐",
                text_clean="警方一共发现了毒可乐",
                start=1.5,
                end=2.9,
                words=words[3:],
            ),
        ],
        words_for_split=words,
        output_traces=[
            OutputTrace(sentence_index=0, split_reason="cut_plan"),
            OutputTrace(sentence_index=1, split_reason="cut_plan"),
        ],
        segmentation_result=SegmentationResult(
            segments=(
                SegmentPlan(
                    segment_id="timeanchored:main:none:3:seg:0",
                    token_start=0,
                    token_end=2,
                    start=0.0,
                    end=1.4,
                    boundary_reason="cut_plan",
                    boundary_score=1.0,
                ),
                SegmentPlan(
                    segment_id="timeanchored:main:none:3:seg:1",
                    token_start=3,
                    token_end=5,
                    start=1.5,
                    end=2.9,
                    boundary_reason="cut_plan",
                    boundary_score=1.0,
                ),
            )
        ),
    )

    output = processor._render_with_unified_pipeline(
        segmentation_output=segmentation_output,
        data=DecisionLayerInput(annotated_words=[], vad_intervals=[]),
        stream_id="timeanchored:main:none",
        chunk_index=3,
    )

    assert [sentence.text for sentence in output.sentence_segments] == [
        "幸运的他躲过了一劫",
        "警方一共发现了毒可乐",
    ]
    assert output.subtitle_batch is not None
    assert [item.text for item in output.subtitle_batch.items] == [
        "幸运的他躲过了一劫",
        "警方一共发现了毒可乐",
    ]
    assert [item.start for item in output.subtitle_batch.items] == [0.0, 1.5]
    assert [item.end for item in output.subtitle_batch.items] == [1.4, 2.9]


def test_run_segmentation_core_populates_segmentation_result_for_render_stage() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    sentence = SentenceSegment(
        text="Hello world",
        text_clean="Hello world",
        start=0.0,
        end=1.0,
        words=[
            WordTimestamp(word="Hello", start=0.0, end=0.5, confidence=0.9),
            WordTimestamp(word="world", start=0.5, end=1.0, confidence=0.8),
        ],
    )

    processor._split_by_cut_plan = lambda **kwargs: (
        [sentence],
        [],
        [OutputTrace(sentence_index=0, split_reason="cut_plan")],
        processor._build_segmentation_result_from_sentences(
            stream_id="timeanchored:main:none",
            chunk_ref=3,
            words_for_split=sentence.words,
            sentence_segments=[sentence],
            output_traces=[OutputTrace(sentence_index=0, split_reason="cut_plan")],
        ),
    )
    processor._extract_cross_chunk_pending_tail = lambda sentences, policy_snapshot: (sentences, [], 0)
    processor._normalize_carried_article_sentence_case = lambda **kwargs: None
    processor._finalize_sentence_metadata = lambda sentences: None
    processor._build_canonical_stream_from_words = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("_run_segmentation_core 不应再通过 canonical rebuild 补 segmentation_result")
    )

    output = processor._run_segmentation_core(
        DecisionLayerInput(
            annotated_words=[
                AnnotatedWord(word="Hello", start=0.0, end=0.5),
                AnnotatedWord(word="world", start=0.5, end=1.0),
            ],
            vad_intervals=[],
        ),
        stream_id="timeanchored:main:none",
        chunk_index=3,
        is_last_chunk=True,
    )

    assert output.segmentation_result is not None
    assert output.segmentation_result.segments[0].segment_id == "timeanchored:main:none:3:seg:0"


def test_run_segmentation_core_prefers_segmentation_result_from_split_exit() -> None:
    processor = SegmentationProcessor(final_splitter=FinalSplitter())
    sentence = SentenceSegment(
        text="Hello world",
        text_clean="Hello world",
        start=0.0,
        end=1.0,
        words=[
            WordTimestamp(word="Hello", start=0.0, end=0.5, confidence=0.9),
            WordTimestamp(word="world", start=0.5, end=1.0, confidence=0.8),
        ],
    )
    expected_result = SegmentationResult(
        segments=(
            SegmentPlan(
                segment_id="timeanchored:main:none:3:seg:0",
                token_start=0,
                token_end=1,
                start=0.0,
                end=1.0,
                boundary_reason="cut_plan",
                boundary_score=1.0,
            ),
        )
    )

    processor._split_by_cut_plan = lambda **kwargs: (
        [sentence],
        [],
        [OutputTrace(sentence_index=0, split_reason="cut_plan")],
        expected_result,
    )
    processor._extract_cross_chunk_pending_tail = lambda sentences, policy_snapshot: (sentences, [], 0)
    processor._normalize_carried_article_sentence_case = lambda **kwargs: None
    processor._finalize_sentence_metadata = lambda sentences: None
    processor._build_segmentation_result_from_sentences = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("_run_segmentation_core 不应再自行兜底补构 segmentation_result")
    )

    output = processor._run_segmentation_core(
        DecisionLayerInput(
            annotated_words=[
                AnnotatedWord(word="Hello", start=0.0, end=0.5),
                AnnotatedWord(word="world", start=0.5, end=1.0),
            ],
            vad_intervals=[],
        ),
        stream_id="timeanchored:main:none",
        chunk_index=3,
        is_last_chunk=True,
    )

    assert output.segmentation_result is expected_result


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
