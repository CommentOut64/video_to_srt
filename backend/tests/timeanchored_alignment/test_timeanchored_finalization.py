from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from app.engines.dummy_engine import DummyEngine
from app.models.sensevoice_models import SentenceSegment, WordTimestamp
from app.pipelines.async_dual_pipeline import AsyncDualPipeline
from app.services.alignment.types import DecisionLayerOutput
from app.services.punctuation.base import PuncPosition
from app.services.timeanchored_alignment.contracts import AlignmentItem, BoundaryEvidence


def _item(text: str, *, start: float, end: float, source: str = "timeanchored") -> AlignmentItem:
    return AlignmentItem(
        text=text,
        start=start,
        end=end,
        status="direct",
        source=source,
        confidence=0.92,
    )


def test_finalize_timeanchored_stream_english_long_pause_stays_in_unified_split_layer() -> None:
    pipeline = AsyncDualPipeline(
        job_id="test_timeanchored_finalization_en_long_pause",
        draft_engine=DummyEngine(response_text="watch the stream. and keep listening", latency_ms=0),
        patch_engine=DummyEngine(response_text="watch the stream. and keep listening", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )

    final_stream = (
        _item("watch", start=0.0, end=0.18),
        _item("the", start=0.18, end=0.30),
        _item("stream.", start=0.30, end=0.56),
        _item("and", start=1.22, end=1.34),
        _item("keep", start=1.34, end=1.50),
        _item("listening", start=1.50, end=1.86),
    )
    boundary_evidences = (
        BoundaryEvidence(
            split_idx=2,
            event_time=0.89,
            left_end=0.56,
            right_start=1.22,
            reason="gap_pause",
            score=1.0,
            hard_flag=False,
        ),
        BoundaryEvidence(
            split_idx=2,
            event_time=0.89,
            left_end=0.56,
            right_start=1.22,
            reason="strong_punct",
            score=0.85,
            hard_flag=False,
        ),
    )

    run_result = pipeline._finalize_timeanchored_stream(
        final_stream=final_stream,
        boundary_evidences=boundary_evidences,
        detected_language="en",
        chosen_text_clean="watch the stream. and keep listening",
        punctuation_positions=[],
        punctuation_clean_text="watch the stream. and keep listening",
        speaker_id=None,
        turn_id=None,
        coverage=1.0,
        route_confidence=0.93,
        error_code="",
    )

    assert len(run_result.final_sentences) == 1
    assert run_result.final_sentences[0].text_clean == "watch the stream. and keep listening"
    assert run_result.final_sentences[0].confidence_source != "unknown"
    assert run_result.output_traces
    assert run_result.output_traces[0].split_reason in {"default_splitter", "tail_flush", "fallback_single_sentence"}


def test_finalize_timeanchored_stream_cjk_dense_char_candidates_do_not_fragment_sentences() -> None:
    pipeline = AsyncDualPipeline(
        job_id="test_timeanchored_finalization_cjk_dense_chars",
        draft_engine=DummyEngine(response_text="今天天气真不错我们出发吧", latency_ms=0),
        patch_engine=DummyEngine(response_text="今天天气真不错我们出发吧", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )

    tokens = ["今", "天", "天", "气", "真", "不", "错", "吧"]
    final_stream = tuple(
        _item(token, start=index * 0.12, end=index * 0.12 + 0.08)
        for index, token in enumerate(tokens)
    )
    boundary_evidences = tuple(
        BoundaryEvidence(
            split_idx=index,
            event_time=(final_stream[index].end + final_stream[index + 1].start) / 2.0,
            left_end=final_stream[index].end,
            right_start=final_stream[index + 1].start,
            reason="gap_pause",
            score=0.18,
            hard_flag=False,
        )
        for index in range(len(final_stream) - 1)
    )

    run_result = pipeline._finalize_timeanchored_stream(
        final_stream=final_stream,
        boundary_evidences=boundary_evidences,
        detected_language="zh",
        chosen_text_clean="今天天气真不错吧",
        punctuation_positions=[],
        punctuation_clean_text="今天天气真不错吧",
        speaker_id=None,
        turn_id=None,
        coverage=1.0,
        route_confidence=0.92,
        error_code="",
    )

    assert len(run_result.final_sentences) == 1
    assert run_result.final_sentences[0].text_clean == "今天天气真不错吧"


def test_finalize_timeanchored_stream_writes_back_punctuation_positions() -> None:
    pipeline = AsyncDualPipeline(
        job_id="test_timeanchored_finalization_writeback_punct",
        draft_engine=DummyEngine(response_text="你好，世界。", latency_ms=0),
        patch_engine=DummyEngine(response_text="你好，世界。", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )

    final_stream = (
        _item("你", start=0.00, end=0.08),
        _item("好", start=0.08, end=0.16),
        _item("世", start=0.16, end=0.24),
        _item("界", start=0.24, end=0.34),
    )

    run_result = pipeline._finalize_timeanchored_stream(
        final_stream=final_stream,
        boundary_evidences=(),
        detected_language="zh",
        chosen_text_clean="你好世界",
        punctuation_positions=[
            PuncPosition(char_index=1, punctuation="，", confidence=0.99),
            PuncPosition(char_index=3, punctuation="。", confidence=0.99),
        ],
        punctuation_clean_text="你好世界",
        speaker_id=None,
        turn_id=None,
        coverage=1.0,
        route_confidence=0.96,
        error_code="",
    )

    assert len(run_result.final_sentences) == 1
    assert run_result.final_sentences[0].text_clean == "你好，世界"
    assert run_result.words_for_split[-1].word.endswith("。")


def test_finalize_timeanchored_stream_blocks_punctuation_when_track_is_out_of_local_index_space() -> None:
    pipeline = AsyncDualPipeline(
        job_id="test_timeanchored_finalization_blocks_shifted_punct_track",
        draft_engine=DummyEngine(response_text="现了一瓶未开封的可乐这家", latency_ms=0),
        patch_engine=DummyEngine(response_text="现了一瓶未开封的可乐这家", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )

    final_stream = tuple(
        _item(token, start=index * 0.08, end=index * 0.08 + 0.06)
        for index, token in enumerate(("现", "了", "一", "瓶", "未", "开", "封", "的", "可", "乐", "这", "家"))
    )

    run_result = pipeline._finalize_timeanchored_stream(
        final_stream=final_stream,
        boundary_evidences=(),
        detected_language="zh",
        chosen_text_clean="现了一瓶未开封的可乐这家",
        punctuation_positions=[
            PuncPosition(char_index=10, punctuation="，", confidence=0.99),
        ],
        punctuation_clean_text="发现了一瓶未开封的可乐这家",
        speaker_id=None,
        turn_id=None,
        coverage=1.0,
        route_confidence=0.96,
        error_code="",
    )

    assert len(run_result.final_sentences) == 1
    assert run_result.final_sentences[0].text_clean == "现了一瓶未开封的可乐这家"
    assert all("，" not in word.word for word in run_result.words_for_split)
    assert run_result.injection_stats["injection_blocked"] == 1.0
    assert run_result.injection_stats["injection_error_code"] == "E_SCORING_INJECTION_BLOCKED"


def test_finalize_timeanchored_stream_cjk_gap_pause_can_promote_fast_draft_boundary() -> None:
    pipeline = AsyncDualPipeline(
        job_id="test_timeanchored_finalization_disable_cjk_fast_draft_fallback",
        draft_engine=DummyEngine(response_text="今天天气真不错我们出发吧", latency_ms=0),
        patch_engine=DummyEngine(response_text="今天天气真不错我们出发吧", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )

    tokens = ("今", "天", "天", "气", "真", "不", "错", "我", "们", "出", "发", "吧")
    time_rows = [
        (0.00, 0.08),
        (0.08, 0.16),
        (0.16, 0.24),
        (0.24, 0.32),
        (0.32, 0.40),
        (0.40, 0.48),
        (0.48, 0.56),
        (1.22, 1.30),
        (1.30, 1.38),
        (1.38, 1.46),
        (1.46, 1.54),
        (1.54, 1.62),
    ]
    final_stream = tuple(
        _item(token, start=start, end=end)
        for token, (start, end) in zip(tokens, time_rows)
    )
    boundary_evidences = (
        BoundaryEvidence(
            split_idx=6,
            event_time=0.89,
            left_end=0.56,
            right_start=1.22,
            reason="gap_pause",
            score=1.0,
            hard_flag=False,
        ),
    )

    run_result = pipeline._finalize_timeanchored_stream(
        final_stream=final_stream,
        boundary_evidences=boundary_evidences,
        detected_language="zh",
        chosen_text_clean="今天天气真不错我们出发吧",
        punctuation_positions=[],
        punctuation_clean_text="今天天气真不错我们出发吧",
        speaker_id=None,
        turn_id=None,
        coverage=1.0,
        route_confidence=0.92,
        error_code="",
    )

    assert len(run_result.final_sentences) == 2
    assert [sentence.text_clean for sentence in run_result.final_sentences] == [
        "今天天气真不错",
        "我们出发吧",
    ]
    assert [trace.split_reason for trace in run_result.output_traces] == ["fast_draft", "tail_flush"]
    assert int(run_result.split_stats.get("applied_decision_count", 0) or 0) >= 1


def test_finalize_timeanchored_stream_passes_soft_cut_plan_and_fast_draft_fallback_to_decision() -> None:
    pipeline = AsyncDualPipeline(
        job_id="test_timeanchored_finalization_passes_cut_plan",
        draft_engine=DummyEngine(response_text="今天天气真不错我们出发吧", latency_ms=0),
        patch_engine=DummyEngine(response_text="今天天气真不错我们出发吧", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )
    pipeline._is_enable_soft_cut = True

    captured: dict[str, Any] = {}

    def _mock_l6_process(data, **_kwargs: Any) -> DecisionLayerOutput:
        captured["cut_plan"] = data.cut_plan
        captured["allow_fast_draft_fallback"] = data.allow_fast_draft_fallback
        return DecisionLayerOutput(
            sentence_segments=[
                SentenceSegment(
                    text="今天天气真不错我们出发吧",
                    text_clean="今天天气真不错我们出发吧",
                    start=0.0,
                    end=1.62,
                    words=[
                        WordTimestamp(word="今天天气真不错", start=0.0, end=0.56, confidence=0.9),
                        WordTimestamp(word="我们出发吧", start=1.22, end=1.62, confidence=0.9),
                    ],
                )
            ],
            words_for_split=[
                WordTimestamp(word="今天天气真不错", start=0.0, end=0.56, confidence=0.9),
                WordTimestamp(word="我们出发吧", start=1.22, end=1.62, confidence=0.9),
            ],
            segmentation_report={"boundary_score_stats": {}, "soft_cut_stats": {}, "error_code": ""},
            applied_cut_plan=data.cut_plan,
        )

    pipeline._decision_processor.process = _mock_l6_process  # type: ignore[method-assign]

    final_stream = (
        _item("今天天气真不错", start=0.0, end=0.56),
        _item("我们出发吧", start=1.22, end=1.62),
    )
    boundary_evidences = (
        BoundaryEvidence(
            split_idx=0,
            event_time=0.89,
            left_end=0.56,
            right_start=1.22,
            reason="gap_pause",
            score=1.0,
            hard_flag=False,
        ),
    )

    pipeline._finalize_timeanchored_stream(
        final_stream=final_stream,
        boundary_evidences=boundary_evidences,
        detected_language="zh",
        chosen_text_clean="今天天气真不错我们出发吧",
        punctuation_positions=[],
        punctuation_clean_text="今天天气真不错我们出发吧",
        speaker_id="spk-1",
        turn_id="turn-1",
        coverage=1.0,
        route_confidence=0.92,
        error_code="",
    )

    assert captured["cut_plan"] is not None
    assert captured["allow_fast_draft_fallback"] is True
