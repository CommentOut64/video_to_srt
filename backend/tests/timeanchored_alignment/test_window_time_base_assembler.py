from __future__ import annotations

import numpy as np

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
from app.services.timeanchored_alignment.window_time_base_assembler import WindowTimeBaseAssembler


def _build_ctx(index: int, start: float, end: float, text: str) -> ProcessingContext:
    chunk = AudioChunk(
        index=index,
        start=start,
        end=end,
        audio=np.zeros(int((end - start) * 16000), dtype=np.float32),
        sample_rate=16000,
        language="zh",
    )
    unit = TimeBaseUnit(text=text, start=start, end=end, confidence=0.95)
    return ProcessingContext(
        job_id="job-window-time-base",
        chunk_index=index,
        audio_chunk=chunk,
        time_base_chunk=TimeBasePackage(
            raw_units=(unit,),
            word_units=(unit,),
            quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.9, low_prob_ratio=0.05),
            language="zh",
        ),
    )


def _build_ready_window() -> ReadySlowWindow:
    bindings = (
        WindowChunkBinding(
            chunk_id="chunk-2",
            chunk_index=2,
            chunk_start=0.0,
            chunk_end=1.0,
            overlap_ratio=1.0,
            role="owner",
            is_owner=True,
        ),
        WindowChunkBinding(
            chunk_id="chunk-3",
            chunk_index=3,
            chunk_start=1.0,
            chunk_end=2.0,
            overlap_ratio=1.0,
            role="core",
            is_owner=False,
        ),
    )
    return ReadySlowWindow(
        window_id="sw-000777",
        owner_chunk_id="chunk-2",
        owner_chunk_index=2,
        window_mode="steady",
        flush_reason="target_duration",
        audio_segments=((0.0, 1.0), (1.0, 2.0)),
        coverage=WindowCoverage(core_segments=((0.0, 2.0),), left_guard_sec=0.0, right_guard_sec=0.0, chunk_bindings=bindings),
        source_semantic_chunk_ids=("semantic-2", "semantic-3"),
        source_chunk_ids=("chunk-2", "chunk-3"),
        source_chunk_indices=(2, 3),
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
            duration_bucket="medium",
            token_estimate=4,
            acoustic_density_hint="normal",
            queue_priority=0,
        ),
        created_at=0.0,
    )


def test_window_time_base_assembler_aggregates_multiple_chunk_time_bases() -> None:
    assembler = WindowTimeBaseAssembler()
    ready_window = _build_ready_window()
    contexts = [
        _build_ctx(index=2, start=0.0, end=0.6, text="你"),
        _build_ctx(index=3, start=1.0, end=1.5, text="好"),
    ]

    package = assembler.assemble(ready_window=ready_window, source_contexts=contexts)

    assert package.window_id == "sw-000777"
    assert package.source_chunk_ids == ("chunk-2", "chunk-3")
    assert package.source_chunk_indices == (2, 3)
    assert tuple(binding.chunk_index for binding in package.chunk_bindings) == (2, 3)
    assert tuple(unit.text for unit in package.raw_units) == ("你", "好")


def test_window_time_base_assembler_rebases_chunk_relative_units_to_absolute_timeline() -> None:
    assembler = WindowTimeBaseAssembler()
    ready_window = _build_ready_window()
    contexts = [
        _build_ctx(index=2, start=0.0, end=0.6, text="你"),
        _build_ctx(index=3, start=0.0, end=0.5, text="好"),
    ]

    package = assembler.assemble(ready_window=ready_window, source_contexts=contexts)

    assert tuple((unit.start, unit.end) for unit in package.raw_units) == (
        (0.0, 0.6),
        (1.0, 1.5),
    )
    assert tuple((unit.start, unit.end) for unit in package.word_units) == (
        (0.0, 0.6),
        (1.0, 1.5),
    )
