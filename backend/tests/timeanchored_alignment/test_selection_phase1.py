from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np

from app.engines.dummy_engine import DummyEngine
from app.models.sensevoice_models import SentenceSegment
from app.pipelines.async_dual_pipeline import AsyncDualPipeline
from app.pipelines.dual_pipeline.services.diagnostic_trace_service import DiagnosticTraceService
from app.schemas.pipeline_context import ProcessingContext
from app.services.alignment.types import AlignmentResult, L2Output, PunctTrack, TextTrack, TextTrackBundle
from app.services.arbitration.arbiter import ArbitrationResult
from app.services.audio.chunk_engine import AudioChunk
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


def _build_ctx(job_id: str) -> ProcessingContext:
    ctx = ProcessingContext(
        job_id=job_id,
        chunk_index=0,
        audio_chunk=_build_chunk(),
        job_dir=Path("F:/video_to_srt_gpu/backend/tests/.tmp-selection-phase1"),
        sv_result=_build_sv_result(),
        whisper_result=_build_whisper_result(),
        time_base_chunk=_build_time_base(),
    )
    ctx.text_tracks = TextTrackBundle(
        sv_track=_build_track("你 好 世 界", source="sv"),
        whisper_track=_build_track("你好世界", source="whisper"),
    )
    return ctx


def _build_pipeline(job_id: str) -> AsyncDualPipeline:
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
    pipeline._alignment_pipeline_shadow_sample_rate = 1.0
    return pipeline


def test_alignment_stage_uses_selection_service_instead_of_legacy_arbitration(monkeypatch) -> None:
    job_id = "test_phase1_selection_service_replaces_legacy_arbitration"
    pipeline = _build_pipeline(job_id)
    monkeypatch.setattr(pipeline, "_apply_whisper_full_sanitize", lambda _ctx: None)
    assert not hasattr(pipeline, "_run_arbitration")

    chosen_track = _build_track("你好世界", "chosen")
    arbitration_result = ArbitrationResult(
        chosen_source="slow",
        reason="auto_slow",
        sv_score=0.8,
        wh_score=0.9,
        coverage=1.0,
    )
    selection_outcome = SimpleNamespace(
        selected_text_truth=SimpleNamespace(
            text=chosen_track.text_clean,
            raw_text=chosen_track.raw_text,
            language_hint="zh",
            text_source="slow",
        ),
        selection_decision=SimpleNamespace(
            chosen_source="slow",
            reason_code="auto_slow",
        ),
        selection_report=SimpleNamespace(
            primary_reason_code="auto_slow",
            summary={"layer": "selection"},
        ),
        selection_inputs=SimpleNamespace(scope="selection-input"),
        arbitration_output=L2Output(
            chosen_text_track=chosen_track,
            arbitration_result=arbitration_result,
        ),
        arbitration_result=arbitration_result,
    )
    selection_service = SimpleNamespace(
        build_selection_inputs=Mock(return_value=selection_outcome.selection_inputs),
        select=Mock(return_value=selection_outcome),
        apply_runtime_selection=Mock(return_value=chosen_track.text_clean),
    )
    pipeline._alignment_stage_service._selection_service = selection_service
    writer = SimpleNamespace(
        enabled=True,
        level="summary",
        write_layer_summary=Mock(),
        write_stage=Mock(),
    )
    pipeline._alignment_stage_service._postprocess_trace_writer = writer
    monkeypatch.setattr(
        pipeline,
        "_run_punctuation_layer",
        AsyncMock(return_value=PunctTrack(clean_text_ref=chosen_track.text_clean, positions=[], source="none")),
    )
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_run_timeanchored_main_chain",
        Mock(return_value=object()),
    )
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_should_accept_timeanchored_result",
        Mock(return_value=(True, "default_prefer_timeanchored")),
    )
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_commit_timeanchored_main_chain_result",
        Mock(),
    )
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_record_hetero_alignment_result",
        Mock(),
    )

    ctx = _build_ctx(job_id)
    asyncio.run(pipeline._run_alignment_stage(ctx))

    assert selection_service.select.call_count == 1
    assert selection_service.apply_runtime_selection.call_count == 1
    assert writer.write_layer_summary.call_count == 1
    assert writer.write_stage.call_count == 2
    assert getattr(ctx, "selected_text_truth", None) is not None
    assert getattr(ctx, "selection_decision", None) is not None
    assert getattr(ctx.selection_decision, "chosen_source", None) == "slow"
    remove_streaming_subtitle_manager(job_id)


def test_diagnostic_trace_service_prefers_selection_outputs_when_arbitration_result_absent() -> None:
    service = DiagnosticTraceService(logger=Mock())
    captured: dict[str, object] = {}
    ctx = SimpleNamespace(
        job_id="job-selection-diag",
        chunk_index=4,
        job_dir=None,
        arbitration_result=None,
        selected_text_truth=SimpleNamespace(
            text="来自选择层的文本",
            text_source="fast",
        ),
        selection_decision=SimpleNamespace(chosen_source="fast"),
        selection_report=SimpleNamespace(primary_reason_code="hallucination"),
    )

    service.emit_layer_diagnostics(
        ctx,
        tracks=TextTrackBundle(
            sv_track=_build_track("快流文本", "sv"),
            whisper_track=_build_track("慢流文本", "whisper"),
            chosen_track=_build_track("旧轨道文本", "chosen"),
        ),
        punct_track=PunctTrack(clean_text_ref="来自选择层的文本", positions=[], source="none"),
        alignment_result=AlignmentResult(
            aligned_words=[],
            alignment_score=1.0,
            gap_ratio=0.0,
            gap_positions=[],
            coverage=1.0,
        ),
        injection_stats={},
        split_stats={},
        final_sentences=[
            SentenceSegment(
                text="快流文本",
                text_clean="快流文本",
                start=0.0,
                end=0.2,
            )
        ],
        append_debug_layer_diag_line=lambda _job_dir, payload, logger=None: captured.update(payload=payload),
    )

    payload = captured["payload"]
    assert payload["chosen_source"] == "fast"
    assert payload["arbitration_reason"] == "hallucination"
    assert payload.get("selection_reason") == "hallucination"
    assert payload["chosen_clean_len"] == len("来自选择层的文本")
    assert payload["punctuation_pre_clean_text_match"] is True
