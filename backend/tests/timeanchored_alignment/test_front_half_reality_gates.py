from __future__ import annotations

import asyncio
from dataclasses import is_dataclass
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

import app.services.bridge as bridge_pkg
from app.engines.dummy_engine import DummyEngine
from app.pipelines.async_dual_pipeline import AsyncDualPipeline
from app.pipelines.workers.slow_worker import SlowWorker
from app.schemas.pipeline_context import ProcessingContext
from app.services.audio.chunk_engine import AudioChunk
from app.services.timeanchored_alignment.contracts import TimeBasePackage, TimeBaseQuality, TimeBaseUnit
from app.services.timeanchored_alignment.slow_window.contracts import (
    DialogueShapeSnapshot,
    PromptSeed,
    ReadySlowWindow,
    WindowBatchHint,
    WindowChunkBinding,
    WindowCoverage,
    WindowLanguageProfile,
)
from app.services.timeanchored_alignment.window_time_base_assembler import WindowTimeBasePackage


def _build_audio_chunk(*, index: int, start: float, end: float) -> AudioChunk:
    return AudioChunk(
        index=index,
        start=start,
        end=end,
        audio=np.zeros(int((end - start) * 16000), dtype=np.float32),
        sample_rate=16000,
        language="zh",
    )


def _build_ctx(*, index: int, start: float, end: float) -> ProcessingContext:
    return ProcessingContext(
        job_id="job-front-half-gate",
        chunk_index=index,
        audio_chunk=_build_audio_chunk(index=index, start=start, end=end),
        full_audio_array=np.zeros(int(6 * 16000), dtype=np.float32),
        full_audio_sr=16000,
        sv_result={"text": f"chunk-{index}", "language": "zh"},
        time_base_chunk=TimeBasePackage(
            raw_units=(
                TimeBaseUnit(
                    text=f"字{index}",
                    start=start,
                    end=min(end, start + 0.2),
                    confidence=0.95,
                ),
            ),
            word_units=(
                TimeBaseUnit(
                    text=f"字{index}",
                    start=start,
                    end=min(end, start + 0.2),
                    confidence=0.95,
                ),
            ),
            quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.9, low_prob_ratio=0.05),
            language="zh",
        ),
    )


def test_processing_context_only_carries_window_level_stable_package_refs() -> None:
    field_names = set(ProcessingContext.__dataclass_fields__.keys())

    assert "ready_slow_window" in field_names
    assert "window_time_base" in field_names


def test_front_half_public_surface_no_longer_exposes_turn_group_entrypoints() -> None:
    assert not hasattr(SlowWorker, "process_turn_group")
    assert not hasattr(bridge_pkg, "TurnGroup")
    assert not hasattr(bridge_pkg, "TurnGroupBuilder")
    assert not hasattr(bridge_pkg, "TurnGroupEnvelope")


@pytest.mark.asyncio
async def test_ready_window_transport_does_not_split_window_whisper_result_back_to_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = AsyncDualPipeline(
        job_id="job-front-half-gate",
        draft_engine=DummyEngine(response_text="快流", latency_ms=0),
        patch_engine=DummyEngine(response_text="慢流", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )

    owner_ctx = _build_ctx(index=3, start=0.0, end=1.0)
    peer_ctx = _build_ctx(index=4, start=1.0, end=2.0)
    pipeline._context_cache = {3: owner_ctx, 4: peer_ctx}

    ready_window = ReadySlowWindow(
        window_id="sw-000001",
        owner_chunk_id="chunk-3",
        owner_chunk_index=3,
        window_mode="bootstrap",
        flush_reason="target_duration",
        audio_segments=((0.0, 1.0), (1.0, 2.0)),
        coverage=WindowCoverage(
            core_segments=((0.0, 2.0),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-3",
                    chunk_index=3,
                    chunk_start=0.0,
                    chunk_end=1.0,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
                WindowChunkBinding(
                    chunk_id="chunk-4",
                    chunk_index=4,
                    chunk_start=1.0,
                    chunk_end=2.0,
                    overlap_ratio=1.0,
                    role="core",
                    is_owner=False,
                ),
            ),
        ),
        source_semantic_chunk_ids=("semantic-3", "semantic-4"),
        source_chunk_ids=("chunk-3", "chunk-4"),
        source_chunk_indices=(3, 4),
        source_units=(),
        dialogue_shape=DialogueShapeSnapshot(
            shape="single_speaker",
            speaker_count=1,
            dominant_speaker_id="spk-1",
            dominant_speaker_ratio=1.0,
            speaker_switch_count=0,
            speaker_switch_density=0.0,
            turn_count=1,
            avg_turn_duration_sec=2.0,
        ),
        language_profile=WindowLanguageProfile(
            primary_language="zh",
            language_mix_state="single_language",
            decision_domains=("zh",),
            should_bypass_whisper=False,
        ),
        prompt_seed=PromptSeed(text="你好世界", keywords=("你好", "世界")),
        batch_hint=WindowBatchHint(
            duration_bucket="short",
            token_estimate=4,
            acoustic_density_hint="normal",
            queue_priority=0,
        ),
        created_at=0.0,
    )

    monkeypatch.setattr(
        pipeline,
        "_split_whisper_result_by_chunks",
        Mock(side_effect=AssertionError("window-first 主链不应再回切 chunk whisper")),
    )
    monkeypatch.setattr(
        pipeline.slow_worker,
        "process_ready_window",
        AsyncMock(
            return_value={
                "text": "你好世界",
                "raw_text": "你好世界",
                "min_clean_text": "你好世界",
                "confidence": 0.9,
                "language": "zh",
                "segments": [{"start": 0.0, "end": 2.0, "text": "你好世界"}],
                "raw_result": {"segments": [{"start": 0.0, "end": 2.0, "text": "你好世界"}]},
            }
        ),
    )
    pipeline.queue_final = asyncio.Queue()

    pause_requested = await pipeline._process_ready_slow_window(
        ready_window,
        job_dir=None,
        total_chunks=8,
        slow_processed_indices=set(),
        token=None,
    )

    assert pause_requested is False
    assert owner_ctx.whisper_result["text"] == "你好世界"
    assert peer_ctx.whisper_result is None
    assert owner_ctx.ready_slow_window is ready_window
    assert is_dataclass(owner_ctx.window_time_base)
    assert isinstance(owner_ctx.window_time_base, WindowTimeBasePackage)


@pytest.mark.asyncio
async def test_ready_window_marks_hallucination_and_resets_prompt_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = AsyncDualPipeline(
        job_id="job-front-half-hallucination",
        draft_engine=DummyEngine(response_text="快流", latency_ms=0),
        patch_engine=DummyEngine(response_text="慢流", latency_ms=0),
        transcription_profile="sv_whisper_dual",
        enable_cross_chunk_merge=False,
        enable_semantic_buffer=False,
        punctuation_service=Mock(),
    )

    owner_ctx = _build_ctx(index=6, start=0.0, end=1.0)
    owner_ctx.sv_result = {
        "text": "快流文本",
        "text_clean": "快流文本",
        "text_itn_raw": "快流文本",
        "language": "zh",
    }
    peer_ctx = _build_ctx(index=7, start=1.0, end=2.0)
    pipeline._context_cache = {6: owner_ctx, 7: peer_ctx}
    pipeline.previous_whisper_text = "历史慢流上文"

    ready_window = ReadySlowWindow(
        window_id="sw-hallucination",
        owner_chunk_id="chunk-6",
        owner_chunk_index=6,
        window_mode="steady",
        flush_reason="target_duration",
        audio_segments=((0.0, 1.0), (1.0, 2.0)),
        coverage=WindowCoverage(
            core_segments=((0.0, 2.0),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-6",
                    chunk_index=6,
                    chunk_start=0.0,
                    chunk_end=1.0,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
                WindowChunkBinding(
                    chunk_id="chunk-7",
                    chunk_index=7,
                    chunk_start=1.0,
                    chunk_end=2.0,
                    overlap_ratio=1.0,
                    role="core",
                    is_owner=False,
                ),
            ),
        ),
        source_semantic_chunk_ids=("semantic-6", "semantic-7"),
        source_chunk_ids=("chunk-6", "chunk-7"),
        source_chunk_indices=(6, 7),
        source_units=(),
        dialogue_shape=DialogueShapeSnapshot(
            shape="single_speaker",
            speaker_count=1,
            dominant_speaker_id="spk-1",
            dominant_speaker_ratio=1.0,
            speaker_switch_count=0,
            speaker_switch_density=0.0,
            turn_count=1,
            avg_turn_duration_sec=2.0,
        ),
        language_profile=WindowLanguageProfile(
            primary_language="zh",
            language_mix_state="single_language",
            decision_domains=("zh",),
            should_bypass_whisper=False,
        ),
        prompt_seed=PromptSeed(text="你好世界", keywords=("你好", "世界")),
        batch_hint=WindowBatchHint(
            duration_bucket="short",
            token_estimate=4,
            acoustic_density_hint="normal",
            queue_priority=0,
        ),
        created_at=0.0,
    )

    monkeypatch.setattr(
        pipeline.slow_worker,
        "process_ready_window",
        AsyncMock(
            return_value={
                "text": "Questions 19",
                "raw_text": "Questions 19",
                "min_clean_text": "Questions 19",
                "confidence": 0.9,
                "language": "zh",
                "segments": [{"start": 0.0, "end": 2.0, "text": "Questions 19"}],
                "raw_result": {"segments": [{"start": 0.0, "end": 2.0, "text": "Questions 19"}]},
            }
        ),
    )
    pipeline.queue_final = asyncio.Queue()

    pause_requested = await pipeline._process_ready_slow_window(
        ready_window,
        job_dir=None,
        total_chunks=8,
        slow_processed_indices=set(),
        token=None,
    )

    assert pause_requested is False
    assert owner_ctx.whisper_result["is_hallucination"] is True
    assert pipeline.previous_whisper_text == ""
