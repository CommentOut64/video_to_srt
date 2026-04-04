from __future__ import annotations

from types import SimpleNamespace

from app.services.timeanchored_alignment.contracts import (
    SelectedTextTruth,
    TimeBasePackage,
    TimeBaseQuality,
    TimeBaseUnit,
)
from app.services.timeanchored_alignment.decoder.service import AlignmentDecoderService
from app.services.timeanchored_alignment.preparation.assembler import AlignmentPreparationAssembler
from app.services.timeanchored_alignment.slow_window.contracts import (
    DialogueShapeSnapshot,
    PromptSeed,
    ReadySlowWindow,
    WindowBatchHint,
    WindowChunkBinding,
    WindowCoverage,
    WindowLanguageProfile,
    WindowSourceUnit,
)
from app.services.timeanchored_alignment.time_base_builder import TimeBaseBuilder
from app.services.timeanchored_alignment.window_time_base_assembler import WindowTimeBasePackage


class MockFastTimeAdapter:
    @property
    def can_decode_ctc(self) -> bool:
        return False

    def build_time_base(
        self,
        *,
        ctc_logits,
        language: str,
        frame_stride: float = 0.06,
        encoder_out_lens: int | None = None,
        compact_acoustic_trace: dict | None = None,
        raw_tokens=None,
    ) -> TimeBasePackage:
        units = (
            TimeBaseUnit(text="你", start=0.0, end=0.1, confidence=0.92, token_type="word"),
            TimeBaseUnit(text="好", start=0.1, end=0.2, confidence=0.93, token_type="word"),
            TimeBaseUnit(text="世", start=0.2, end=0.3, confidence=0.94, token_type="word"),
            TimeBaseUnit(text="界", start=0.3, end=0.4, confidence=0.95, token_type="word"),
        )
        return TimeBasePackage(
            raw_units=units,
            word_units=units,
            quality=TimeBaseQuality(blank_ratio=0.0, avg_max_prob=0.93, low_prob_ratio=0.0),
            language=language,
            frame_stride=frame_stride,
            source="mock_fast_adapter",
        )


def _build_ready_window() -> ReadySlowWindow:
    return ReadySlowWindow(
        window_id="phase7-window-001",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        window_mode="steady",
        flush_reason="test",
        audio_segments=((0.0, 0.4),),
        coverage=WindowCoverage(
            core_segments=((0.0, 0.4),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-1",
                    chunk_index=1,
                    chunk_start=0.0,
                    chunk_end=0.4,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
            ),
        ),
        source_semantic_chunk_ids=("sem-1",),
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        source_units=(
            WindowSourceUnit(
                unit_id="unit-1",
                semantic_chunk_id="sem-1",
                text="你好世界",
                audio_start=0.0,
                audio_end=0.4,
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
                speaker_id="speaker-a",
                turn_id="turn-a",
                language="zh",
                arrived_at=0.4,
            ),
        ),
        dialogue_shape=DialogueShapeSnapshot(
            shape="single_speaker",
            speaker_count=1,
            dominant_speaker_id="speaker-a",
            dominant_speaker_ratio=1.0,
            speaker_switch_count=0,
            speaker_switch_density=0.0,
            turn_count=1,
            avg_turn_duration_sec=0.4,
        ),
        language_profile=WindowLanguageProfile(
            primary_language="zh",
            language_mix_state="single_language",
            decision_domains=("timeanchored_alignment",),
            should_bypass_whisper=False,
        ),
        prompt_seed=PromptSeed(text="你好世界"),
        batch_hint=WindowBatchHint(
            duration_bucket="short",
            token_estimate=4,
            acoustic_density_hint="medium",
            queue_priority=1,
        ),
        created_at=0.4,
    )


def test_decoder_mainline_accepts_mock_fast_adapter_observation_source() -> None:
    ready_window = _build_ready_window()
    ctx = SimpleNamespace(
        chunk_index=1,
        sv_result={
            "language": "zh",
            "raw_tokens": [{"text": "你"}, {"text": "好"}, {"text": "世"}, {"text": "界"}],
            "ctc_compact_trace": {"raw_tokens": [{"text": "你"}, {"text": "好"}, {"text": "世"}, {"text": "界"}]},
        },
    )
    time_base = TimeBaseBuilder(adapter=MockFastTimeAdapter()).build(ctx)
    assert time_base is not None

    window_time_base = WindowTimeBasePackage(
        window_id=ready_window.window_id,
        language="zh",
        raw_units=time_base.raw_units,
        word_units=time_base.word_units,
        quality=time_base.quality,
        source_chunk_ids=ready_window.source_chunk_ids,
        source_chunk_indices=ready_window.source_chunk_indices,
        chunk_bindings=ready_window.coverage.chunk_bindings,
        metadata=time_base.metadata,
        frame_stride=time_base.frame_stride,
        source="mock_fast_window",
    )
    preparation = AlignmentPreparationAssembler().prepare(
        ready_window=ready_window,
        window_time_base=window_time_base,
        selected_text_truth=SelectedTextTruth(
            text="你好世界",
            text_source="slow",
            language_hint="zh",
            source_chunk_ids=("chunk-1",),
            quality={"confidence": 0.95},
            metadata={"raw_text": "你好世界"},
        ),
        whisper_result={
            "text": "你好世界",
            "text_clean": "你好世界",
            "text_itn_raw": "你好世界",
            "language": "zh",
            "confidence": 0.95,
            "raw_result": {"segments": [{"avg_logprob": -0.1}]},
        },
        default_language="zh",
    )

    result = AlignmentDecoderService().execute(preparation=preparation)

    assert preparation.acoustic_observation_pack.adapter_type == "mock_fast_window"
    assert result.alignment_path is not None
    assert result.alignment_report.failure_semantic in {"none", "alignment_low_confidence"}
    assert result.alignment_report.metadata["window_id"] == ready_window.window_id
