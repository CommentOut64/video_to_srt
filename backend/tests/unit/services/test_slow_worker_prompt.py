import numpy as np
import pytest

from app.core.asr.enums import ASRCapability, TimestampPrecision
from app.core.asr.models import ASRMetadata, ASRResult, Segment
from app.pipelines.workers.slow_worker import SlowWorker
from app.services.timeanchored_alignment.slow_window.contracts import (
    DialogueShapeSnapshot,
    PromptSeed,
    ReadySlowWindow,
    WindowBatchHint,
    WindowChunkBinding,
    WindowCoverage,
    WindowLanguageProfile,
)


class _DummyPatchEngine:
    def __init__(self) -> None:
        self.last_initial_prompt = "__unset__"

    async def transcribe(self, audio, language=None, **kwargs):  # type: ignore[no-untyped-def]
        self.last_initial_prompt = kwargs.get("initial_prompt")
        return ASRResult(
            text="测试文本",
            segments=[Segment(start=0.0, end=0.5, text="测试文本", confidence=0.8)],
            confidence=0.8,
            language=str(language or "zh"),
            metadata=ASRMetadata(
                engine="dummy",
                source="dummy",
                timestamp_precision=TimestampPrecision.SEGMENT,
                capabilities=[ASRCapability.LANGUAGE_DETECTION],
                raw_tags={},
            ),
        )


def _build_ready_window(*, prompt_seed_text: str) -> ReadySlowWindow:
    return ReadySlowWindow(
        window_id="sw-000001",
        owner_chunk_id="chunk-0",
        owner_chunk_index=0,
        window_mode="steady",
        flush_reason="target_duration",
        audio_segments=((0.0, 0.5),),
        coverage=WindowCoverage(
            core_segments=((0.0, 0.5),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-0",
                    chunk_index=0,
                    chunk_start=0.0,
                    chunk_end=0.5,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
            ),
        ),
        source_semantic_chunk_ids=("semantic-0",),
        source_chunk_ids=("chunk-0",),
        source_chunk_indices=(0,),
        source_units=(),
        dialogue_shape=DialogueShapeSnapshot(
            shape="single_speaker",
            speaker_count=1,
            dominant_speaker_id="spk-1",
            dominant_speaker_ratio=1.0,
            speaker_switch_count=0,
            speaker_switch_density=0.0,
            turn_count=1,
            avg_turn_duration_sec=0.5,
        ),
        language_profile=WindowLanguageProfile(
            primary_language="zh",
            language_mix_state="single_language",
            decision_domains=("zh",),
            should_bypass_whisper=False,
        ),
        prompt_seed=PromptSeed(text=prompt_seed_text, keywords=()),
        batch_hint=WindowBatchHint(
            duration_bucket="short",
            token_estimate=2,
            acoustic_density_hint="normal",
            queue_priority=0,
        ),
        created_at=0.0,
    )


@pytest.mark.asyncio
async def test_process_ready_window_uses_explicit_prompt_over_prompt_seed() -> None:
    engine = _DummyPatchEngine()
    worker = SlowWorker(
        patch_engine=engine,
        whisper_language="zh",
    )
    full_audio_array = np.zeros(16000, dtype=np.float32)

    await worker.process_ready_window(
        _build_ready_window(prompt_seed_text="窗口种子提示"),
        full_audio_array=full_audio_array,
        full_audio_sr=16000,
        prompt_text="显式策略提示",
    )

    assert engine.last_initial_prompt == "显式策略提示"
