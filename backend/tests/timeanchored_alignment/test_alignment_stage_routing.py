from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from app.engines.dummy_engine import DummyEngine
from app.models.sensevoice_models import SentenceSegment
from app.pipelines.async_dual_pipeline import AsyncDualPipeline
from app.pipelines.dual_pipeline.services.alignment_stage_service import AlignmentStageService
from app.schemas.pipeline_context import ProcessingContext
from app.services.alignment.types import L2Output, PunctTrack, TextTrack, TextTrackBundle
from app.services.arbitration.arbiter import ArbitrationResult
from app.services.audio.chunk_engine import AudioChunk
from app.services.punctuation.base import PuncPosition
from app.services.streaming_subtitle import remove_streaming_subtitle_manager
from app.services.timeanchored_alignment.contracts import TimeBasePackage, TimeBaseQuality, TimeBaseUnit


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
        aligned_facts=SimpleNamespace(annotated_words=[], speaker_turns=[], time_mappings=[]),
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


def _patch_common_alignment_inputs(*, monkeypatch, pipeline: AsyncDualPipeline, chosen_source: str = "slow") -> None:
    monkeypatch.setattr(pipeline, "_apply_whisper_full_sanitize", lambda _ctx: None)
    chosen_track = _build_track(
        "你 好 世 界" if chosen_source == "fast" else "你好世界",
        "chosen",
    )
    arbitration_result = ArbitrationResult(
        chosen_source=chosen_source,
        reason="test",
        sv_score=0.8,
        wh_score=0.9,
        coverage=1.0,
    )
    monkeypatch.setattr(
        pipeline._alignment_stage_service._selection_service,
        "build_selection_inputs",
        Mock(return_value=SimpleNamespace(scope="selection-input")),
    )
    monkeypatch.setattr(
        pipeline._alignment_stage_service._selection_service,
        "select",
        Mock(
            return_value=SimpleNamespace(
                selected_text_truth=SimpleNamespace(
                    text=chosen_track.text_clean,
                    language_hint="zh",
                ),
                selection_decision=SimpleNamespace(chosen_source=chosen_source),
                selection_report=SimpleNamespace(primary_reason_code="test", summary={"layer": "selection"}),
                selection_inputs=SimpleNamespace(scope="selection-input"),
                arbitration_output=L2Output(
                    chosen_text_track=chosen_track,
                    arbitration_result=arbitration_result,
                ),
                arbitration_result=arbitration_result,
                chosen_track=chosen_track,
                chosen_text_clean=chosen_track.text_clean,
            )
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_punctuation_layer",
        AsyncMock(return_value=PunctTrack(clean_text_ref=chosen_track.text_clean, positions=[], source="none")),
    )


def _build_ctx(job_id: str) -> ProcessingContext:
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
    return ctx


def _build_pipeline(job_id: str, *, mode: str) -> AsyncDualPipeline:
    pipeline = AsyncDualPipeline(
        job_id=job_id,
        draft_engine=DummyEngine(response_text="你好世界", latency_ms=0),
        patch_engine=DummyEngine(response_text="你好世界", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )
    pipeline._alignment_pipeline_mode = mode
    pipeline._alignment_pipeline_shadow_sample_rate = 1.0
    return pipeline


def _patch_legacy_noop(monkeypatch, pipeline: AsyncDualPipeline, legacy_run: Mock) -> None:
    monkeypatch.setattr(pipeline, "_run_collection_scoring_decision_once", legacy_run)
    monkeypatch.setattr(pipeline, "_emit_layer_diagnostics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "_emit_layer_trace_full", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "_record_punct_retry_candidates", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "_emit_output_layer",
        lambda **_kwargs: SimpleNamespace(output_payload={"errors": []}),
    )


def test_alignment_stage_mode_legacy_is_coerced_to_default(monkeypatch) -> None:
    job_id = "test_phase7_route_legacy_coerce_default"
    pipeline = _build_pipeline(job_id, mode="legacy")
    _patch_common_alignment_inputs(monkeypatch=monkeypatch, pipeline=pipeline)

    monkeypatch.setattr(
        pipeline,
        "_run_collection_scoring_decision_once",
        Mock(side_effect=AssertionError("legacy 链路已下线，不应被调用")),
    )
    timeanchored_run = Mock(return_value=object())
    commit = Mock()
    record = Mock()
    monkeypatch.setattr(pipeline._alignment_stage_service, "_run_timeanchored_main_chain", timeanchored_run)
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_should_accept_timeanchored_result",
        Mock(return_value=(True, "default_prefer_timeanchored")),
    )
    monkeypatch.setattr(pipeline._alignment_stage_service, "_commit_timeanchored_main_chain_result", commit)
    monkeypatch.setattr(pipeline._alignment_stage_service, "_record_hetero_alignment_result", record)

    ctx = _build_ctx(job_id)
    asyncio.run(pipeline._run_alignment_stage(ctx))

    assert timeanchored_run.call_count == 1
    assert commit.call_count == 1
    assert record.call_count == 1
    assert record.call_args.kwargs.get("mode") == "default"
    remove_streaming_subtitle_manager(job_id)


def test_alignment_stage_mode_shadow_is_coerced_to_default(monkeypatch) -> None:
    job_id = "test_phase7_route_shadow_coerce_default"
    pipeline = _build_pipeline(job_id, mode="shadow")
    _patch_common_alignment_inputs(monkeypatch=monkeypatch, pipeline=pipeline)

    monkeypatch.setattr(
        pipeline,
        "_run_collection_scoring_decision_once",
        Mock(side_effect=AssertionError("legacy 链路已下线，不应被调用")),
    )
    timeanchored_run = Mock(return_value=object())
    commit = Mock()
    record = Mock()
    monkeypatch.setattr(pipeline._alignment_stage_service, "_run_timeanchored_main_chain", timeanchored_run)
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_should_accept_timeanchored_result",
        Mock(return_value=(True, "default_prefer_timeanchored")),
    )
    monkeypatch.setattr(pipeline._alignment_stage_service, "_commit_timeanchored_main_chain_result", commit)
    monkeypatch.setattr(pipeline._alignment_stage_service, "_record_hetero_alignment_result", record)

    ctx = _build_ctx(job_id)
    asyncio.run(pipeline._run_alignment_stage(ctx))

    assert timeanchored_run.call_count == 1
    assert commit.call_count == 1
    assert record.call_count == 1
    assert record.call_args.kwargs.get("mode") == "default"
    remove_streaming_subtitle_manager(job_id)


def test_alignment_stage_mode_active_uses_timeanchored_when_gate_passes(monkeypatch) -> None:
    job_id = "test_phase7_route_active_pass"
    pipeline = _build_pipeline(job_id, mode="active")
    _patch_common_alignment_inputs(monkeypatch=monkeypatch, pipeline=pipeline)

    monkeypatch.setattr(
        pipeline,
        "_run_collection_scoring_decision_once",
        Mock(side_effect=AssertionError("legacy 链路已下线，不应被调用")),
    )
    timeanchored_run = Mock(return_value=object())
    commit = Mock()
    record = Mock()
    monkeypatch.setattr(pipeline._alignment_stage_service, "_run_timeanchored_main_chain", timeanchored_run)
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_should_accept_timeanchored_result",
        Mock(return_value=(True, "active_gate_pass")),
    )
    monkeypatch.setattr(pipeline._alignment_stage_service, "_commit_timeanchored_main_chain_result", commit)
    monkeypatch.setattr(pipeline._alignment_stage_service, "_record_hetero_alignment_result", record)

    ctx = _build_ctx(job_id)
    asyncio.run(pipeline._run_alignment_stage(ctx))

    assert timeanchored_run.call_count == 1
    assert commit.call_count == 1
    assert record.call_count == 1
    remove_streaming_subtitle_manager(job_id)


def test_alignment_stage_mode_active_no_longer_forces_fast_direct_from_legacy_fallback_signal(
    monkeypatch,
) -> None:
    job_id = "test_phase7_route_active_decoder_failure_semantic_keeps_timeanchored_commit"
    pipeline = _build_pipeline(job_id, mode="active")
    _patch_common_alignment_inputs(monkeypatch=monkeypatch, pipeline=pipeline)

    monkeypatch.setattr(
        pipeline,
        "_run_collection_scoring_decision_once",
        Mock(side_effect=AssertionError("legacy 链路已下线，不应被调用")),
    )
    stage_result = SimpleNamespace(
        alignment_path=SimpleNamespace(aligned_tokens=(SimpleNamespace(source_chunk_ids=("chunk-0",)),)),
        alignment_report=SimpleNamespace(route="alignment_path", failure_semantic="alignment_low_confidence"),
    )
    timeanchored_run = Mock(return_value=stage_result)
    commit_timeanchored = Mock()
    commit_fast_direct = Mock()
    record = Mock()
    monkeypatch.setattr(pipeline._alignment_stage_service, "_run_timeanchored_main_chain", timeanchored_run)
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_should_accept_timeanchored_result",
        Mock(return_value=(True, "active_gate_pass")),
    )
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_commit_timeanchored_main_chain_result",
        commit_timeanchored,
    )
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_commit_fast_direct_result",
        commit_fast_direct,
    )
    monkeypatch.setattr(pipeline._alignment_stage_service, "_record_hetero_alignment_result", record)

    ctx = _build_ctx(job_id)
    asyncio.run(pipeline._run_alignment_stage(ctx))

    assert timeanchored_run.call_count == 1
    assert commit_fast_direct.call_count == 0
    assert commit_timeanchored.call_count == 1
    assert record.call_count == 1
    assert record.call_args.kwargs.get("reason") == "active_gate_pass"
    assert record.call_args.kwargs.get("selected") is True
    remove_streaming_subtitle_manager(job_id)


def test_alignment_stage_sensevoice_only_enters_unified_main_chain(monkeypatch) -> None:
    job_id = "test_phase7_route_sensevoice_only_unified"
    pipeline = AsyncDualPipeline(
        job_id=job_id,
        draft_engine=DummyEngine(response_text="你好世界", latency_ms=0),
        patch_engine=None,
        transcription_profile="sensevoice_only",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )
    pipeline._alignment_pipeline_mode = "default"
    _patch_common_alignment_inputs(monkeypatch=monkeypatch, pipeline=pipeline, chosen_source="fast")

    monkeypatch.setattr(
        pipeline,
        "_run_collection_scoring_decision_once",
        Mock(side_effect=AssertionError("统一主链命中后不应执行 legacy 四层")),
    )
    timeanchored_run = Mock(return_value=object())
    commit = Mock()
    commit_fast_direct = Mock()
    record = Mock()
    monkeypatch.setattr(pipeline._alignment_stage_service, "_run_timeanchored_main_chain", timeanchored_run)
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_should_accept_timeanchored_result",
        Mock(return_value=(True, "default_gate_pass")),
    )
    monkeypatch.setattr(pipeline._alignment_stage_service, "_commit_timeanchored_main_chain_result", commit)
    monkeypatch.setattr(pipeline._alignment_stage_service, "_commit_fast_direct_result", commit_fast_direct)
    monkeypatch.setattr(pipeline._alignment_stage_service, "_record_hetero_alignment_result", record)

    ctx = ProcessingContext(
        job_id=job_id,
        chunk_index=0,
        audio_chunk=_build_chunk(),
        sv_result=_build_sv_result(),
        whisper_result=None,
        time_base_chunk=_build_time_base(),
    )
    ctx.text_tracks = TextTrackBundle(
        sv_track=_build_track("你 好 世 界", source="sv"),
        whisper_track=None,
    )
    asyncio.run(pipeline._run_alignment_stage(ctx))

    assert timeanchored_run.call_count == 1
    assert commit.call_count == 1
    assert commit_fast_direct.call_count == 0
    assert record.call_count == 1
    assert isinstance(ctx.whisper_result, dict)
    remove_streaming_subtitle_manager(job_id)


def test_alignment_stage_whisper_skipped_still_enters_unified_main_chain(monkeypatch) -> None:
    job_id = "test_phase7_route_whisper_skipped_unified"
    pipeline = _build_pipeline(job_id, mode="default")
    _patch_common_alignment_inputs(monkeypatch=monkeypatch, pipeline=pipeline, chosen_source="fast")

    monkeypatch.setattr(
        pipeline,
        "_run_collection_scoring_decision_once",
        Mock(side_effect=AssertionError("统一主链命中后不应执行 legacy 四层")),
    )
    timeanchored_run = Mock(return_value=object())
    commit = Mock()
    finalize_sensevoice_only = Mock(side_effect=AssertionError("不应走 finalize_sensevoice_only 旁路"))
    monkeypatch.setattr(pipeline._alignment_stage_service, "_run_timeanchored_main_chain", timeanchored_run)
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_should_accept_timeanchored_result",
        Mock(return_value=(True, "default_gate_pass")),
    )
    monkeypatch.setattr(pipeline._alignment_stage_service, "_commit_timeanchored_main_chain_result", commit)
    monkeypatch.setattr(pipeline, "_finalize_sensevoice_only", finalize_sensevoice_only)
    monkeypatch.setattr(pipeline._alignment_stage_service, "_record_hetero_alignment_result", Mock())

    ctx = ProcessingContext(
        job_id=job_id,
        chunk_index=0,
        audio_chunk=_build_chunk(),
        sv_result=_build_sv_result(),
        whisper_result=None,
        time_base_chunk=_build_time_base(),
        whisper_skipped=True,
    )
    ctx.text_tracks = TextTrackBundle(
        sv_track=_build_track("你 好 世 界", source="sv"),
        whisper_track=None,
    )
    asyncio.run(pipeline._run_alignment_stage(ctx))

    assert timeanchored_run.call_count == 1
    assert commit.call_count == 1
    assert finalize_sensevoice_only.call_count == 0
    remove_streaming_subtitle_manager(job_id)


def test_alignment_stage_mode_default_raises_when_gate_fails(monkeypatch) -> None:
    job_id = "test_phase7_route_default_raise"
    pipeline = _build_pipeline(job_id, mode="default")
    _patch_common_alignment_inputs(monkeypatch=monkeypatch, pipeline=pipeline)

    monkeypatch.setattr(
        pipeline,
        "_run_collection_scoring_decision_once",
        Mock(side_effect=AssertionError("legacy 链路已下线，不应被调用")),
    )
    timeanchored_run = Mock(return_value=object())
    commit = Mock()
    record = Mock()
    monkeypatch.setattr(pipeline._alignment_stage_service, "_run_timeanchored_main_chain", timeanchored_run)
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_should_accept_timeanchored_result",
        Mock(return_value=(False, "default_gate_route_error")),
    )
    monkeypatch.setattr(pipeline._alignment_stage_service, "_commit_timeanchored_main_chain_result", commit)
    monkeypatch.setattr(pipeline._alignment_stage_service, "_record_hetero_alignment_result", record)

    ctx = _build_ctx(job_id)
    with pytest.raises(RuntimeError, match="legacy_alignment_pipeline_disabled"):
        asyncio.run(pipeline._run_alignment_stage(ctx))

    assert timeanchored_run.call_count == 1
    assert commit.call_count == 0
    assert record.call_count == 1
    remove_streaming_subtitle_manager(job_id)


def test_alignment_stage_mode_default_commits_fast_direct_when_selected_fast_rejects_slow(
    monkeypatch,
) -> None:
    job_id = "test_phase7_route_default_selection_reject_slow_fast_direct"
    pipeline = _build_pipeline(job_id, mode="default")
    _patch_common_alignment_inputs(monkeypatch=monkeypatch, pipeline=pipeline, chosen_source="fast")

    stage_result = SimpleNamespace(
        alignment_path=None,
        alignment_report=SimpleNamespace(
            route="selection_reject_slow",
            failure_semantic="selection_reject_slow",
        ),
        low_confidence_spans=(),
    )
    timeanchored_run = Mock(return_value=stage_result)
    commit_timeanchored = Mock()
    commit_fast_direct = Mock()
    record = Mock()
    monkeypatch.setattr(pipeline._alignment_stage_service, "_run_timeanchored_main_chain", timeanchored_run)
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_commit_timeanchored_main_chain_result",
        commit_timeanchored,
    )
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_commit_fast_direct_result",
        commit_fast_direct,
    )
    monkeypatch.setattr(pipeline._alignment_stage_service, "_record_hetero_alignment_result", record)

    ctx = _build_ctx(job_id)
    asyncio.run(pipeline._run_alignment_stage(ctx))

    assert timeanchored_run.call_count == 1
    assert commit_timeanchored.call_count == 0
    assert commit_fast_direct.call_count == 1
    assert commit_fast_direct.call_args.kwargs["reason"] == "default_selection_reject_slow"
    assert record.call_count == 1
    remove_streaming_subtitle_manager(job_id)


def test_alignment_stage_sensevoice_only_commits_fast_direct_when_timeanchored_rejects_slow(
    monkeypatch,
) -> None:
    job_id = "test_phase7_route_sensevoice_only_selection_reject_slow_fast_direct"
    pipeline = AsyncDualPipeline(
        job_id=job_id,
        draft_engine=DummyEngine(response_text="你好世界", latency_ms=0),
        patch_engine=None,
        transcription_profile="sensevoice_only",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )
    pipeline._alignment_pipeline_mode = "default"
    _patch_common_alignment_inputs(monkeypatch=monkeypatch, pipeline=pipeline, chosen_source="fast")

    stage_result = SimpleNamespace(
        alignment_path=None,
        alignment_report=SimpleNamespace(
            route="selection_reject_slow",
            failure_semantic="selection_reject_slow",
        ),
        low_confidence_spans=(),
    )
    timeanchored_run = Mock(return_value=stage_result)
    commit_timeanchored = Mock()
    commit_fast_direct = Mock()
    record = Mock()
    monkeypatch.setattr(pipeline._alignment_stage_service, "_run_timeanchored_main_chain", timeanchored_run)
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_commit_timeanchored_main_chain_result",
        commit_timeanchored,
    )
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_commit_fast_direct_result",
        commit_fast_direct,
    )
    monkeypatch.setattr(pipeline._alignment_stage_service, "_record_hetero_alignment_result", record)

    ctx = ProcessingContext(
        job_id=job_id,
        chunk_index=0,
        audio_chunk=_build_chunk(),
        sv_result=_build_sv_result(),
        whisper_result=None,
        time_base_chunk=_build_time_base(),
    )
    ctx.text_tracks = TextTrackBundle(
        sv_track=_build_track("你 好 世 界", source="sv"),
        whisper_track=None,
    )

    asyncio.run(pipeline._run_alignment_stage(ctx))

    assert timeanchored_run.call_count == 1
    assert commit_timeanchored.call_count == 0
    assert commit_fast_direct.call_count == 1
    assert commit_fast_direct.call_args.kwargs["reason"] == "default_selection_reject_slow"
    assert record.call_count == 1
    assert isinstance(ctx.whisper_result, dict)
    remove_streaming_subtitle_manager(job_id)


def test_should_accept_timeanchored_default_rejects_error_route() -> None:
    stage_result = SimpleNamespace(
        alignment_path=None,
        alignment_report=SimpleNamespace(route="alignment_no_path", failure_semantic="alignment_no_path"),
    )
    accepted, reason = AlignmentStageService._should_accept_timeanchored_result(
        stage_result=stage_result,
        mode="default",
    )

    assert accepted is False
    assert reason == "default_alignment_no_path"


def test_should_accept_timeanchored_default_requires_non_empty_stream() -> None:
    stage_result = SimpleNamespace(
        alignment_path=SimpleNamespace(aligned_tokens=tuple()),
        alignment_report=SimpleNamespace(route="alignment_path", failure_semantic="alignment_path_empty"),
    )
    accepted, reason = AlignmentStageService._should_accept_timeanchored_result(
        stage_result=stage_result,
        mode="default",
    )

    assert accepted is False
    assert reason == "default_alignment_path_empty"


def test_should_accept_timeanchored_default_accepts_valid_result_without_final_sentences() -> None:
    stage_result = SimpleNamespace(
        alignment_path=SimpleNamespace(aligned_tokens=(object(),)),
        alignment_report=SimpleNamespace(route="alignment_path", failure_semantic="none"),
    )
    accepted, reason = AlignmentStageService._should_accept_timeanchored_result(
        stage_result=stage_result,
        mode="default",
    )

    assert accepted is True
    assert reason == "default_gate_pass"


def test_alignment_stage_emits_punctuation_chain_health_metrics() -> None:
    metrics = AlignmentStageService._build_punctuation_chain_health_metrics(
        chosen_source="slow",
        punct_track=PunctTrack(
            clean_text_ref="你好世界",
            positions=[PuncPosition(char_index=3, punctuation="。", confidence=0.9)],
            source="fast",
        ),
        preparation_punctuation_count=0,
        punctuation_fact_count=0,
    )

    assert metrics["chosen_source"] == "slow"
    assert metrics["punct_track_positions_total"] == 1
    assert metrics["preparation_punctuation_count"] == 0
    assert metrics["punctuation_chain_broken_flag"] in (0, 1)
