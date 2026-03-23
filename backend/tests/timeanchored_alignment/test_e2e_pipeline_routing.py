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


def _patch_common_inputs(monkeypatch, pipeline: AsyncDualPipeline) -> None:
    monkeypatch.setattr(pipeline, "_apply_whisper_full_sanitize", lambda _ctx: None)
    chosen_track = _build_track("你好世界", "chosen")
    monkeypatch.setattr(
        pipeline,
        "_run_arbitration",
        lambda *_args, **_kwargs: L2Output(
            chosen_text_track=chosen_track,
            arbitration_result=ArbitrationResult(
                chosen_source="slow",
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


def _build_ctx(job_id: str, *, with_time_base: bool = True) -> ProcessingContext:
    ctx = ProcessingContext(
        job_id=job_id,
        chunk_index=0,
        audio_chunk=_build_chunk(),
        sv_result=_build_sv_result(),
        whisper_result=_build_whisper_result(),
        time_base_chunk=_build_time_base() if with_time_base else None,
    )
    ctx.text_tracks = TextTrackBundle(
        sv_track=_build_track("你 好 世 界", source="sv"),
        whisper_track=_build_track("你好世界", source="whisper"),
    )
    return ctx


def _patch_legacy_flow(monkeypatch, pipeline: AsyncDualPipeline, legacy_run: Mock) -> None:
    monkeypatch.setattr(pipeline, "_run_collection_scoring_decision_once", legacy_run)
    monkeypatch.setattr(pipeline, "_emit_layer_diagnostics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "_emit_layer_trace_full", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pipeline, "_record_punct_retry_candidates", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "_emit_output_layer",
        lambda **_kwargs: SimpleNamespace(output_payload={"errors": []}),
    )


def test_e2e_shadow_mode_is_coerced_to_default(monkeypatch) -> None:
    job_id = "test_phase7_e2e_shadow_coerce_default"
    pipeline = _build_pipeline(job_id, mode="shadow")
    _patch_common_inputs(monkeypatch, pipeline)

    monkeypatch.setattr(
        pipeline,
        "_run_collection_scoring_decision_once",
        Mock(side_effect=AssertionError("legacy 链路已下线，不应被调用")),
    )
    monkeypatch.setattr(pipeline._alignment_stage_service, "_run_timeanchored_main_chain", Mock(return_value=object()))
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_should_accept_timeanchored_result",
        Mock(return_value=(True, "default_prefer_timeanchored")),
    )

    def _mock_commit(**kwargs):
        ctx = kwargs["ctx"]
        ctx.final_sentences = [
            SentenceSegment(
                text="timeanchored selected",
                text_clean="timeanchored selected",
                start=0.0,
                end=0.4,
                words=[],
            )
        ]
        ctx.finalization_metrics = {"timeanchored_enabled": 1.0}

    monkeypatch.setattr(pipeline._alignment_stage_service, "_commit_timeanchored_main_chain_result", _mock_commit)
    monkeypatch.setattr(pipeline._alignment_stage_service, "_record_hetero_alignment_result", Mock())

    ctx = _build_ctx(job_id)
    asyncio.run(pipeline._run_alignment_stage(ctx))

    assert ctx.final_sentences
    assert ctx.final_sentences[0].text == "timeanchored selected"
    remove_streaming_subtitle_manager(job_id)


@pytest.mark.parametrize("gate_pass", [True, False])
def test_e2e_active_switches_by_gate(monkeypatch, gate_pass: bool) -> None:
    job_id = f"test_phase7_e2e_active_{int(gate_pass)}"
    pipeline = _build_pipeline(job_id, mode="active")
    _patch_common_inputs(monkeypatch, pipeline)

    monkeypatch.setattr(
        pipeline,
        "_run_collection_scoring_decision_once",
        Mock(side_effect=AssertionError("legacy 链路已下线，不应被调用")),
    )

    monkeypatch.setattr(pipeline._alignment_stage_service, "_run_timeanchored_main_chain", Mock(return_value=object()))
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_should_accept_timeanchored_result",
        Mock(return_value=(gate_pass, "active_gate_test")),
    )

    def _mock_commit(**kwargs):
        ctx = kwargs["ctx"]
        ctx.final_sentences = [
            SentenceSegment(
                text="timeanchored selected",
                text_clean="timeanchored selected",
                start=0.0,
                end=0.4,
                words=[],
            )
        ]
        ctx.finalization_metrics = {"timeanchored_enabled": 1.0}

    monkeypatch.setattr(pipeline._alignment_stage_service, "_commit_timeanchored_main_chain_result", _mock_commit)
    monkeypatch.setattr(pipeline._alignment_stage_service, "_record_hetero_alignment_result", Mock())

    ctx = _build_ctx(job_id)
    if gate_pass:
        asyncio.run(pipeline._run_alignment_stage(ctx))
        assert ctx.final_sentences
        assert ctx.final_sentences[0].text == "timeanchored selected"
    else:
        with pytest.raises(RuntimeError, match="legacy_alignment_pipeline_disabled"):
            asyncio.run(pipeline._run_alignment_stage(ctx))

    remove_streaming_subtitle_manager(job_id)


def test_e2e_default_raises_when_time_base_missing(monkeypatch) -> None:
    job_id = "test_phase7_e2e_default_raise"
    pipeline = _build_pipeline(job_id, mode="default")
    _patch_common_inputs(monkeypatch, pipeline)

    monkeypatch.setattr(
        pipeline,
        "_run_collection_scoring_decision_once",
        Mock(side_effect=AssertionError("legacy 链路已下线，不应被调用")),
    )
    monkeypatch.setattr(pipeline._alignment_stage_service, "_run_timeanchored_main_chain", Mock(return_value=None))
    monkeypatch.setattr(
        pipeline._alignment_stage_service,
        "_should_accept_timeanchored_result",
        Mock(return_value=(False, "default_timeanchored_failed")),
    )
    monkeypatch.setattr(pipeline._alignment_stage_service, "_record_hetero_alignment_result", Mock())

    ctx = _build_ctx(job_id, with_time_base=False)
    with pytest.raises(RuntimeError, match="legacy_alignment_pipeline_disabled"):
        asyncio.run(pipeline._run_alignment_stage(ctx))

    remove_streaming_subtitle_manager(job_id)
