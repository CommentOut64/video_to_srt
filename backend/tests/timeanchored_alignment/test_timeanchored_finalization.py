from __future__ import annotations

from unittest.mock import Mock

from app.engines.dummy_engine import DummyEngine
from app.pipelines.async_dual_pipeline import AsyncDualPipeline
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


def test_finalize_timeanchored_stream_cjk_gap_pause_is_not_direct_fast_draft_split() -> None:
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

    assert len(run_result.final_sentences) == 1
    assert run_result.final_sentences[0].text_clean == "今天天气真不错我们出发吧"
    assert all(trace.split_reason != "fast_draft" for trace in run_result.output_traces)
