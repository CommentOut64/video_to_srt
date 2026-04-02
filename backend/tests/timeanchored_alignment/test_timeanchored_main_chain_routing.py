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
from app.services.timeanchored_alignment.window_recovery_contracts import (
    RecoveredWindowPlan,
    WindowSpanDecision,
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


def test_alignment_stage_force_fast_without_time_base_still_uses_unified_main_chain(
    monkeypatch,
) -> None:
    job_id = "test_timeanchored_force_fast_without_time_base_unified"
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

    _patch_common_alignment_inputs(monkeypatch=monkeypatch, pipeline=pipeline, chosen_source="fast")
    run_collection_mock = Mock(side_effect=AssertionError("统一主链命中后不应执行 legacy 四层"))
    monkeypatch.setattr(
        pipeline,
        "_run_collection_scoring_decision_once",
        run_collection_mock,
    )
    timeanchored_run = Mock(return_value=object())
    commit = Mock()
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_run_timeanchored_main_chain",
        timeanchored_run,
    )
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_should_accept_timeanchored_result",
        Mock(return_value=(True, "default_gate_pass")),
    )
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_commit_timeanchored_main_chain_result",
        commit,
    )
    monkeypatch.setattr(pipeline._alignment_stage_service, "_record_hetero_alignment_result", Mock())
    commit_fast_direct = Mock()
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_commit_fast_direct_result",
        commit_fast_direct,
    )
    ctx = ProcessingContext(
        job_id=job_id,
        chunk_index=0,
        audio_chunk=_build_chunk(),
        sv_result=_build_sv_result(),
        whisper_result=None,
        time_base_chunk=None,
    )
    ctx.text_tracks = TextTrackBundle(
        sv_track=_build_track("你 好 世 界", source="sv"),
        whisper_track=None,
    )

    asyncio.run(pipeline._run_alignment_stage(ctx))

    assert timeanchored_run.call_count == 1
    assert commit.call_count == 1
    assert commit_fast_direct.call_count == 0
    assert run_collection_mock.call_count == 0
    assert ctx.time_base_chunk is not None
    assert isinstance(ctx.whisper_result, dict)

    remove_streaming_subtitle_manager(job_id)


def test_resolve_anchor_mount_routes_keeps_timeanchored_route_when_single_chunk_is_only_repairable() -> None:
    service = AlignmentStageService(
        host=SimpleNamespace(
            logger=Mock(),
            _edge_selection_mode="force_slow",
            _postprocess_trace_enabled=False,
            _postprocess_trace_level="summary",
            _anchor_mount_graph="off",
        )
    )
    stage_result = SimpleNamespace(
        decision_ingress=SimpleNamespace(
            anchored_token_units=(object(),),
            source_chunk_ids=("chunk-0",),
            timeline_validity="repairable",
        ),
        anchor_mount_result=SimpleNamespace(should_fallback=True, timeline_validity="repairable"),
    )

    text_route, edge_route, final_route, error_code = service._resolve_anchor_mount_routes(
        ctx=SimpleNamespace(edge_selection_mode="force_slow"),
        stage_result=stage_result,
    )

    assert text_route == "slow"
    assert edge_route == "slow"
    assert final_route == "slow"
    assert error_code is None


def test_resolve_anchor_mount_routes_keeps_slow_route_when_multi_chunk_anchor_mount_requests_fallback() -> None:
    service = AlignmentStageService(
        host=SimpleNamespace(
            logger=Mock(),
            _edge_selection_mode="force_slow",
            _postprocess_trace_enabled=False,
            _postprocess_trace_level="summary",
            _anchor_mount_graph="off",
        )
    )
    stage_result = SimpleNamespace(
        decision_ingress=SimpleNamespace(
            anchored_token_units=(object(),),
            source_chunk_ids=("chunk-19", "chunk-20"),
            timeline_validity="repairable",
        ),
        anchor_mount_result=SimpleNamespace(should_fallback=True, timeline_validity="repairable"),
    )

    text_route, edge_route, final_route, error_code = service._resolve_anchor_mount_routes(
        ctx=SimpleNamespace(edge_selection_mode="force_slow"),
        stage_result=stage_result,
    )

    assert text_route == "slow"
    assert edge_route == "slow"
    assert final_route == "slow"
    assert error_code is None


def test_resolve_anchor_mount_routes_marks_fatal_window_for_safe_fallback() -> None:
    service = AlignmentStageService(
        host=SimpleNamespace(
            logger=Mock(),
            _edge_selection_mode="force_slow",
            _postprocess_trace_enabled=False,
            _postprocess_trace_level="summary",
            _anchor_mount_graph="off",
        )
    )
    stage_result = SimpleNamespace(
        decision_ingress=SimpleNamespace(
            anchored_token_units=(object(),),
            source_chunk_ids=("chunk-19", "chunk-20"),
            timeline_validity="fatal",
        ),
        anchor_mount_result=SimpleNamespace(should_fallback=True, timeline_validity="fatal"),
    )

    text_route, edge_route, final_route, error_code = service._resolve_anchor_mount_routes(
        ctx=SimpleNamespace(edge_selection_mode="force_slow"),
        stage_result=stage_result,
    )

    assert text_route == "slow"
    assert edge_route == "safe_window_fallback"
    assert final_route == "safe_window_fallback"
    assert error_code is None


def test_resolve_anchor_mount_routes_prefers_partial_commit_for_quarantined_window_with_recoverable_spans() -> None:
    service = AlignmentStageService(
        host=SimpleNamespace(
            logger=Mock(),
            _edge_selection_mode="force_slow",
            _postprocess_trace_enabled=False,
            _postprocess_trace_level="summary",
            _anchor_mount_graph="off",
        )
    )
    stage_result = SimpleNamespace(
        decision_ingress=SimpleNamespace(
            anchored_token_units=(object(), object()),
            source_chunk_ids=("chunk-19", "chunk-20"),
            timeline_validity="quarantined",
        ),
        anchor_mount_result=SimpleNamespace(should_fallback=True, timeline_validity="quarantined"),
        window_recovery_plan=SimpleNamespace(has_recoverable_spans=True),
    )

    text_route, edge_route, final_route, error_code = service._resolve_anchor_mount_routes(
        ctx=SimpleNamespace(edge_selection_mode="force_slow"),
        stage_result=stage_result,
    )

    assert text_route == "slow"
    assert edge_route == "partial_commit"
    assert final_route == "partial_commit"
    assert error_code is None


def test_should_use_safe_window_fallback_is_false_when_quarantined_window_has_recoverable_spans() -> None:
    stage_result = SimpleNamespace(
        decision_ingress=SimpleNamespace(
            anchored_token_units=(object(),),
            source_chunk_ids=("chunk-19", "chunk-20"),
            timeline_validity="quarantined",
        ),
        anchor_mount_result=SimpleNamespace(should_fallback=True, timeline_validity="quarantined"),
        window_recovery_plan=SimpleNamespace(has_recoverable_spans=True),
    )

    assert (
        AlignmentStageService._should_use_safe_window_fallback(stage_result=stage_result) is False
    )


def test_build_partial_commit_spans_returns_only_decision_spans_and_skips_emergency_window() -> None:
    service = AlignmentStageService(
        host=SimpleNamespace(
            logger=Mock(),
            _edge_selection_mode="force_slow",
            _postprocess_trace_enabled=False,
            _postprocess_trace_level="summary",
            _anchor_mount_graph="off",
        )
    )
    stage_result = SimpleNamespace(
        window_recovery_plan=RecoveredWindowPlan(
            spans=(
                WindowSpanDecision(
                    span_id="span-decision",
                    span_kind="trusted",
                    route="decision",
                    token_start=1,
                    token_end=3,
                ),
                WindowSpanDecision(
                    span_id="span-fallback",
                    span_kind="fallback",
                    route="span_fallback",
                    token_start=3,
                    token_end=4,
                ),
            )
        )
    )
    emergency_stage_result = SimpleNamespace(
        window_recovery_plan=RecoveredWindowPlan(
            spans=(
                WindowSpanDecision(
                    span_id="span-emergency",
                    span_kind="trusted",
                    route="decision",
                    token_start=4,
                    token_end=6,
                ),
            ),
            emergency_fallback_only=True,
        )
    )

    spans = service._build_partial_commit_spans(stage_result=stage_result)

    assert [(span.span_id, span.token_start, span.token_end) for span in spans] == [
        ("span-decision", 1, 3),
    ]
    assert service._build_partial_commit_spans(stage_result=emergency_stage_result) == ()


def test_partial_commit_passes_span_selector_to_decision_ingress_adapter(monkeypatch) -> None:
    service = AlignmentStageService(
        host=SimpleNamespace(
            logger=Mock(),
            _edge_selection_mode="force_slow",
            _postprocess_trace_enabled=False,
            _postprocess_trace_level="summary",
            _anchor_mount_graph="off",
        )
    )
    stage_result = SimpleNamespace(
        decision_ingress=SimpleNamespace(window_id="window-1"),
        anchor_mount_result=SimpleNamespace(metrics={}),
        window_recovery_plan=RecoveredWindowPlan(
            spans=(
                WindowSpanDecision(
                    span_id="span-decision",
                    span_kind="trusted",
                    route="decision",
                    token_start=1,
                    token_end=3,
                ),
            )
        ),
    )
    ctx = SimpleNamespace(
        chunk_index=0,
        audio_chunk=_build_chunk(),
        arbitration_result=SimpleNamespace(chosen_source="slow"),
        alignment_preparation=SimpleNamespace(
            compat=SimpleNamespace(text_truth=SimpleNamespace(language="en")),
            slow_text=SimpleNamespace(window_text=SimpleNamespace(text="hello world")),
        ),
        text_tracks=None,
    )
    calls: list[dict] = []

    monkeypatch.setattr(
        service,
        "_resolve_anchor_mount_routes",
        lambda **_kwargs: ("slow", "partial_commit", "partial_commit", None),
    )

    def _mock_build(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("stop_after_adapter")

    monkeypatch.setattr(service._decision_ingress_adapter, "build", _mock_build)

    with pytest.raises(RuntimeError, match="stop_after_adapter"):
        service._commit_timeanchored_main_chain_result(
            ctx=ctx,
            stage_result=stage_result,
            whisper_result={"language": "en"},
            sv_result={},
            speaker_id=None,
            turn_id=None,
        )

    assert [call.get("span_selector") for call in calls] == [(1, 3)]


def test_sensevoice_only_pipeline_runs_alignment_stage_instead_of_finalize_shortcut(
    monkeypatch,
) -> None:
    job_id = "test_sensevoice_only_pipeline_unified_alignment_stage"
    pipeline = AsyncDualPipeline(
        job_id=job_id,
        draft_engine=DummyEngine(response_text="你好世界", latency_ms=0),
        patch_engine=None,
        transcription_profile="sensevoice_only",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )

    async def _mock_fast_process(ctx: ProcessingContext) -> None:
        ctx.sv_result = _build_sv_result()

    async def _mock_apply_fast_punctuation(_ctx: ProcessingContext, _normalized: Dict[str, Any]) -> None:
        return None

    captured_whisper_results: list[object] = []

    async def _mock_run_alignment_stage(ctx: ProcessingContext) -> None:
        captured_whisper_results.append(ctx.whisper_result)
        ctx.final_sentences = [
            SentenceSegment(text="统一主链定稿", text_clean="统一主链定稿", start=0.0, end=0.4, words=[])
        ]

    monkeypatch.setattr(pipeline.fast_worker, "process", _mock_fast_process)
    monkeypatch.setattr(pipeline, "_normalize_sensevoice_result", lambda _ctx: _build_sv_result())
    monkeypatch.setattr(pipeline, "_apply_fast_punctuation", _mock_apply_fast_punctuation)
    monkeypatch.setattr(pipeline, "_run_alignment_stage", _mock_run_alignment_stage)
    monkeypatch.setattr(
        pipeline,
        "_finalize_sensevoice_only",
        Mock(side_effect=AssertionError("极速模式不应再走 finalize_sensevoice_only 旁路")),
    )

    results = asyncio.run(
        pipeline._run_sensevoice_only(
            [_build_chunk()],
            full_audio_array=None,
            full_audio_sr=16000,
            job_dir=None,
            processed_indices=set(),
        )
    )

    assert len(results) == 1
    assert results[0].final_sentences
    assert results[0].final_sentences[0].text == "统一主链定稿"
    assert captured_whisper_results == [None]
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
