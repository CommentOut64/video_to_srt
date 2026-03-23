from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from app.engines.dummy_engine import DummyEngine
from app.models.sensevoice_models import SentenceSegment
from app.pipelines.async_dual_pipeline import AsyncDualPipeline
from app.schemas.pipeline_context import ProcessingContext
from app.services.alignment.types import L2Output, PunctTrack, TextTrack, TextTrackBundle
from app.services.arbitration.arbiter import ArbitrationResult
from app.services.audio.chunk_engine import AudioChunk
from app.services.streaming_subtitle import remove_streaming_subtitle_manager
from app.services.timeanchored_alignment.contracts import (
    AlignmentItem,
    AlignmentMetrics,
    FinalAlignmentResult,
    TimeBasePackage,
    TimeBaseQuality,
    TimeBaseUnit,
)


def _build_chunk() -> AudioChunk:
    return AudioChunk(
        index=0,
        start=0.0,
        end=1.0,
        audio=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        language="zh",
    )


def _build_sv_result() -> dict:
    return {
        "text": "你 好 世 界",
        "text_clean": "你 好 世 界",
        "text_itn_raw": "你 好 世 界",
        "confidence": 0.8,
        "language": "zh",
        "words": [
            {"word": "你", "start": 0.0, "end": 0.1, "confidence": 0.9},
            {"word": "好", "start": 0.1, "end": 0.2, "confidence": 0.9},
            {"word": "世", "start": 0.2, "end": 0.3, "confidence": 0.9},
            {"word": "界", "start": 0.3, "end": 0.4, "confidence": 0.9},
        ],
    }


def _build_whisper_result() -> dict:
    return {
        "text": "你好世界",
        "text_clean": "你好世界",
        "text_itn_raw": "你好世界",
        "confidence": 0.9,
        "language": "zh",
        "raw_result": {"segments": [{"avg_logprob": -0.1}]},
    }


def _build_track(text: str, source: str) -> TextTrack:
    return TextTrack(
        raw_text=text,
        text_itn_raw=text,
        text_clean=text,
        char_mapping=[],
        raw_to_clean=[],
        clean_to_raw=[],
        language="zh",
        source=source,
    )


def _build_time_base() -> TimeBasePackage:
    units = (
        TimeBaseUnit(text="你", start=0.0, end=0.1, confidence=0.95, token_type="raw"),
        TimeBaseUnit(text="好", start=0.1, end=0.2, confidence=0.95, token_type="raw"),
        TimeBaseUnit(text="世", start=0.2, end=0.3, confidence=0.95, token_type="raw"),
        TimeBaseUnit(text="界", start=0.3, end=0.4, confidence=0.95, token_type="raw"),
    )
    return TimeBasePackage(
        raw_units=units,
        word_units=units,
        quality=TimeBaseQuality(blank_ratio=0.2, avg_max_prob=0.8, low_prob_ratio=0.1),
        language="zh",
    )


def _build_legacy_run_result() -> SimpleNamespace:
    sentence = SentenceSegment(
        text="legacy fallback",
        text_clean="legacy fallback",
        start=0.0,
        end=0.4,
        words=[],
    )
    return SimpleNamespace(
        alignment_result=SimpleNamespace(
            coverage=1.0,
            gap_ratio=0.0,
            alignment_score=0.9,
            gap_positions=[],
            resolution=None,
        ),
        aligned_facts=SimpleNamespace(
            annotated_words=[],
            speaker_turns=[],
            time_mappings=[],
        ),
        fused_evidence=SimpleNamespace(
            speaker_changes=[],
            pause_anchors=[],
            semantic_anchors=[],
            punctuation_anchors=[],
        ),
        words_for_split=[],
        injection_stats={},
        split_stats={},
        final_sentences=[sentence],
        output_traces=[],
        detected_language="zh",
    )


def _patch_common_alignment_inputs(
    *,
    monkeypatch,
    pipeline: AsyncDualPipeline,
    chosen_source: str = "slow",
) -> None:
    monkeypatch.setattr(pipeline, "_apply_whisper_full_sanitize", lambda _ctx: None)
    chosen_track = _build_track(
        "你 好 世 界" if chosen_source == "fast" else "你好世界",
        "chosen",
    )
    monkeypatch.setattr(
        pipeline,
        "_run_arbitration",
        lambda *_args, **_kwargs: L2Output(
            chosen_text_track=chosen_track,
            arbitration_result=ArbitrationResult(
                chosen_source=chosen_source,
                reason="test",
                sv_score=0.8,
                wh_score=0.9,
                coverage=1.0,
            ),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_punctuation_layer",
        AsyncMock(return_value=PunctTrack(clean_text_ref=chosen_track.text_clean, positions=[], source="none")),
    )


def test_alignment_stage_routes_to_timeanchored_main_chain_when_time_base_available(
    monkeypatch,
) -> None:
    job_id = "test_timeanchored_main_chain_route"
    pipeline = AsyncDualPipeline(
        job_id=job_id,
        draft_engine=DummyEngine(response_text="你好世界", latency_ms=0),
        patch_engine=DummyEngine(response_text="你好世界", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )
    pipeline._alignment_pipeline_mode = "default"

    _patch_common_alignment_inputs(monkeypatch=monkeypatch, pipeline=pipeline, chosen_source="slow")
    monkeypatch.setattr(
        pipeline,
        "_run_collection_scoring_decision_once",
        Mock(side_effect=AssertionError("timeanchored 分支命中后不应执行 legacy 四层")),
    )

    ctx = ProcessingContext(
        job_id=job_id,
        chunk_index=0,
        audio_chunk=_build_chunk(),
        sv_result=_build_sv_result(),
        whisper_result=_build_whisper_result(),
        time_base_chunk=_build_time_base(),
    )
    ctx.text_tracks = TextTrackBundle(
        sv_track=_build_track("你 好 世 界", source="sv"),
        whisper_track=_build_track("你好世界", source="whisper"),
    )

    asyncio.run(pipeline._run_alignment_stage(ctx))

    assert ctx.final_sentences
    assert ctx.finalization_metrics.get("timeanchored_enabled") == 1.0
    assert ctx.finalization_metrics.get("timeanchored_final_route") in {"text", "phonetic", "fast", "slow", "mixed", "error"}
    assert ctx.pronunciation_package is not None
    assert ctx.text_truth is not None

    remove_streaming_subtitle_manager(job_id)


def test_alignment_stage_raises_when_time_base_missing_and_legacy_disabled(
    monkeypatch,
) -> None:
    job_id = "test_timeanchored_missing_time_base_raise"
    pipeline = AsyncDualPipeline(
        job_id=job_id,
        draft_engine=DummyEngine(response_text="你好世界", latency_ms=0),
        patch_engine=DummyEngine(response_text="你好世界", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )
    pipeline._alignment_pipeline_mode = "default"

    _patch_common_alignment_inputs(monkeypatch=monkeypatch, pipeline=pipeline, chosen_source="slow")
    monkeypatch.setattr(
        pipeline,
        "_run_collection_scoring_decision_once",
        Mock(side_effect=AssertionError("legacy 链路已下线，不应被调用")),
    )

    ctx = ProcessingContext(
        job_id=job_id,
        chunk_index=0,
        audio_chunk=_build_chunk(),
        sv_result=_build_sv_result(),
        whisper_result=_build_whisper_result(),
        time_base_chunk=None,
    )
    ctx.text_tracks = TextTrackBundle(
        sv_track=_build_track("你 好 世 界", source="sv"),
        whisper_track=_build_track("你好世界", source="whisper"),
    )

    with pytest.raises(RuntimeError, match="legacy_alignment_pipeline_disabled"):
        asyncio.run(pipeline._run_alignment_stage(ctx))

    remove_streaming_subtitle_manager(job_id)


def test_alignment_stage_raises_when_timeanchored_main_chain_fails(
    monkeypatch,
) -> None:
    job_id = "test_timeanchored_exception_raise"
    pipeline = AsyncDualPipeline(
        job_id=job_id,
        draft_engine=DummyEngine(response_text="你好世界", latency_ms=0),
        patch_engine=DummyEngine(response_text="你好世界", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )
    pipeline._alignment_pipeline_mode = "default"

    _patch_common_alignment_inputs(monkeypatch=monkeypatch, pipeline=pipeline, chosen_source="slow")
    monkeypatch.setattr(
        pipeline,
        "_run_collection_scoring_decision_once",
        Mock(side_effect=AssertionError("legacy 链路已下线，不应被调用")),
    )
    monkeypatch.setattr(
        pipeline._alignment_stage_service._timeanchored_whisper_adapter,
        "build_text_truth_package",
        Mock(side_effect=RuntimeError("forced_timeanchored_error")),
    )
    ctx = ProcessingContext(
        job_id=job_id,
        chunk_index=0,
        audio_chunk=_build_chunk(),
        sv_result=_build_sv_result(),
        whisper_result=_build_whisper_result(),
        time_base_chunk=_build_time_base(),
    )
    ctx.text_tracks = TextTrackBundle(
        sv_track=_build_track("你 好 世 界", source="sv"),
        whisper_track=_build_track("你好世界", source="whisper"),
    )

    with pytest.raises(RuntimeError, match="legacy_alignment_pipeline_disabled"):
        asyncio.run(pipeline._run_alignment_stage(ctx))

    remove_streaming_subtitle_manager(job_id)


@pytest.mark.parametrize(
    ("mode", "chosen_source", "expected_route"),
    [
        ("force_fast", "fast", "fast"),
        ("force_slow", "slow", "slow"),
    ],
)
def test_alignment_stage_forwards_force_mode_to_timeanchored_edge_selector(
    monkeypatch,
    mode: str,
    chosen_source: str,
    expected_route: str,
) -> None:
    job_id = f"test_timeanchored_force_mode_{mode}"
    pipeline = AsyncDualPipeline(
        job_id=job_id,
        draft_engine=DummyEngine(response_text="你好世界", latency_ms=0),
        patch_engine=DummyEngine(response_text="你好世界", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        edge_selection_mode=mode,
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )
    pipeline._alignment_pipeline_mode = "default"

    _patch_common_alignment_inputs(
        monkeypatch=monkeypatch,
        pipeline=pipeline,
        chosen_source=chosen_source,
    )
    monkeypatch.setattr(
        pipeline,
        "_run_collection_scoring_decision_once",
        Mock(side_effect=AssertionError("timeanchored 分支命中后不应执行 legacy 四层")),
    )

    captured: dict[str, str] = {}

    def _mock_select(**kwargs):
        captured["mode"] = str(kwargs.get("edge_selection_mode", ""))
        return FinalAlignmentResult(
            items=(
                AlignmentItem(
                    text="你",
                    start=0.0,
                    end=0.1,
                    status="direct",
                    source="test_selector",
                    confidence=0.9,
                ),
            ),
            route=expected_route,
            metrics=AlignmentMetrics(
                coverage=1.0,
                duration_ratio=1.0,
                failed_count=0,
                route_confidence=1.0,
            ),
        )

    monkeypatch.setattr(
        pipeline._alignment_stage_service._timeanchored_edge_selector,
        "select",
        _mock_select,
    )

    ctx = ProcessingContext(
        job_id=job_id,
        chunk_index=0,
        audio_chunk=_build_chunk(),
        sv_result=_build_sv_result(),
        whisper_result=_build_whisper_result(),
        time_base_chunk=_build_time_base(),
    )
    ctx.text_tracks = TextTrackBundle(
        sv_track=_build_track("你 好 世 界", source="sv"),
        whisper_track=_build_track("你好世界", source="whisper"),
    )

    asyncio.run(pipeline._run_alignment_stage(ctx))

    assert captured.get("mode") == mode
    assert ctx.finalization_metrics.get("timeanchored_enabled") == 1.0
    assert ctx.finalization_metrics.get("timeanchored_edge_route") == expected_route

    remove_streaming_subtitle_manager(job_id)
