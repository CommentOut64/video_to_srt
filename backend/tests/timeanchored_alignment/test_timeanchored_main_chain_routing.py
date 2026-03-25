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
from app.services.streaming_subtitle import remove_streaming_subtitle_manager
from app.services.textflow.contracts import SubtitleBatch, SubtitleItem
from app.services.timeanchored_alignment.contracts import (
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
        subtitle_batch=SubtitleBatch(
            chunk_id="0",
            chunk_index=0,
            items=(
                SubtitleItem(
                    segment_id="legacy:0:seg:0",
                    chunk_id="0",
                    start=0.0,
                    end=0.4,
                    text="legacy fallback",
                    source="aligned",
                ),
            ),
        ),
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
    monkeypatch.setattr(
        pipeline,
        "_finalize_timeanchored_stream",
        Mock(side_effect=AssertionError("AnchorMount 默认主链不应再回流到 finalize_timeanchored_stream")),
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
    assert ctx.finalization_metrics.get("timeanchored_final_route") in {"fast", "slow", "error"}
    assert ctx.alignment_preparation is not None
    assert ctx.window_time_base is not None

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


def test_alignment_stage_compat_window_time_base_rebases_relative_units() -> None:
    ready_window = SimpleNamespace(
        window_id="compat-window-001",
        source_chunk_ids=("chunk-0",),
        source_chunk_indices=(0,),
        coverage=SimpleNamespace(
            chunk_bindings=(
                SimpleNamespace(
                    chunk_id="chunk-0",
                    chunk_index=0,
                    chunk_start=12.5,
                    chunk_end=13.5,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
            )
        ),
    )
    ctx = ProcessingContext(
        job_id="job-compat-window-time-base",
        chunk_index=0,
        audio_chunk=_build_chunk(),
        time_base_chunk=TimeBasePackage(
            raw_units=(
                TimeBaseUnit(text="你", start=0.0, end=0.1, confidence=0.95, token_type="raw"),
            ),
            word_units=(
                TimeBaseUnit(text="你", start=0.0, end=0.1, confidence=0.95, token_type="word"),
            ),
            quality=TimeBaseQuality(blank_ratio=0.2, avg_max_prob=0.8, low_prob_ratio=0.1),
            language="zh",
        ),
    )

    compat_window_time_base = AlignmentStageService._build_compat_window_time_base(
        ctx=ctx,
        ready_window=ready_window,
        language_hint="zh",
    )

    assert tuple((unit.start, unit.end) for unit in compat_window_time_base.raw_units) == (
        (12.5, 12.6),
    )
    assert tuple((unit.start, unit.end) for unit in compat_window_time_base.word_units) == (
        (12.5, 12.6),
    )


def test_alignment_stage_force_fast_direct_path_without_time_base(
    monkeypatch,
) -> None:
    job_id = "test_timeanchored_force_fast_direct_without_time_base"
    pipeline = AsyncDualPipeline(
        job_id=job_id,
        draft_engine=DummyEngine(response_text="你好世界", latency_ms=0),
        patch_engine=DummyEngine(response_text="你好世界", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        edge_selection_mode="force_fast",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )
    pipeline._alignment_pipeline_mode = "default"

    monkeypatch.setattr(
        pipeline,
        "_run_collection_scoring_decision_once",
        Mock(side_effect=AssertionError("force_fast 直通不应进入 legacy 四层")),
    )
    finalize_mock = Mock(return_value=_build_legacy_run_result())
    monkeypatch.setattr(
        pipeline,
        "_finalize_timeanchored_stream",
        finalize_mock,
    )

    ctx = ProcessingContext(
        job_id=job_id,
        chunk_index=0,
        audio_chunk=_build_chunk(),
        sv_result=_build_sv_result(),
        whisper_result=_build_whisper_result(),
        time_base_chunk=None,
    )

    asyncio.run(pipeline._run_alignment_stage(ctx))

    assert ctx.final_sentences
    assert ctx.finalization_metrics.get("fast_direct_enabled") == 1.0
    assert ctx.finalization_metrics.get("alignment_pipeline_route") == "fast"
    finalize_mock.assert_called_once()

    remove_streaming_subtitle_manager(job_id)


def test_alignment_stage_timeanchored_main_chain_dispatches_projected_batches(
    monkeypatch,
) -> None:
    job_id = "test_timeanchored_output_projection_dispatch"
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

    projected_batches = (
        SubtitleBatch(
            chunk_id="chunk-0",
            chunk_index=0,
            items=(
                SubtitleItem(
                    segment_id="seg-0",
                    chunk_id="chunk-0",
                    text="第一句",
                    start=0.0,
                    end=0.8,
                    source="render_core",
                ),
            ),
        ),
        SubtitleBatch(
            chunk_id="chunk-1",
            chunk_index=1,
            items=(),
        ),
    )
    monkeypatch.setattr(
        type(pipeline._alignment_stage_service._output_projector),
        "project",
        lambda _self, _data: projected_batches,
    )

    emit_calls: list[tuple[object, str, int]] = []

    def _mock_emit_output_layer(**kwargs):
        subtitle_batch = kwargs["subtitle_batch"]
        emit_calls.append(
            (
                kwargs["chunk_index"],
                subtitle_batch.chunk_id,
                int(kwargs["segmentation_report"].get("projection_chunk_count", -1)),
            )
        )
        return SimpleNamespace(output_payload={"errors": []})

    monkeypatch.setattr(pipeline, "_emit_output_layer", _mock_emit_output_layer)

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

    assert emit_calls == [(0, "chunk-0", 2), (1, "chunk-1", 2)]
    assert ctx.finalization_metrics.get("timeanchored_projected_chunk_count") == 2.0
    assert ctx.finalization_metrics.get("l7_error_count") == 0.0

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
        pipeline._alignment_stage_service._timeanchored_preparation_assembler,
        "prepare",
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
        ("force_slow", "slow", "slow"),
        ("prefer_fast", "fast", "fast"),
        ("prefer_slow", "slow", "slow"),
    ],
)
def test_alignment_stage_forwards_edge_mode_to_timeanchored_edge_selector(
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

    assert ctx.finalization_metrics.get("timeanchored_enabled") == 1.0
    assert ctx.finalization_metrics.get("timeanchored_edge_route") == expected_route
    assert ctx.finalization_metrics.get("timeanchored_final_route") == expected_route

    remove_streaming_subtitle_manager(job_id)


@pytest.mark.parametrize("input_mode", ("auto", "prefer_fast", "prefer_slow", "force_slow"))
def test_sensevoice_only_always_overrides_edge_mode_to_force_fast(input_mode: str) -> None:
    pipeline = AsyncDualPipeline(
        job_id="test_sensevoice_only_edge_mode",
        draft_engine=DummyEngine(response_text="你好世界", latency_ms=0),
        patch_engine=None,
        transcription_profile="sensevoice_only",
        edge_selection_mode=input_mode,
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )
    assert pipeline._edge_selection_mode == "force_fast"
