from __future__ import annotations

import numpy as np

from app.models.sensevoice_models import WordTimestamp
from app.services.textflow.alignment_path_adapter import AlignmentPathAdapter
from app.services.textflow.fast_timed_ingress_adapter import FastTimedIngressAdapter
from app.services.timeanchored_alignment.contracts import (
    SelectedTextTruth,
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
from app.services.timeanchored_alignment.window_time_base_assembler import WindowTimeBasePackage
from app.services.audio.chunk_engine import AudioChunk
from app.services.punctuation.base import PuncPosition


def _build_chunk() -> AudioChunk:
    return AudioChunk(
        index=5,
        start=10.0,
        end=12.0,
        audio=np.zeros(32000, dtype=np.float32),
        sample_rate=16000,
        language="en",
    )


def _build_fast_words() -> list[WordTimestamp]:
    return [
        WordTimestamp(word="Wouldn't", start=10.0, end=10.35, confidence=0.95),
        WordTimestamp(word="it", start=10.35, end=10.55, confidence=0.94),
        WordTimestamp(word="make", start=10.55, end=10.95, confidence=0.93),
        WordTimestamp(word="sense", start=10.95, end=11.35, confidence=0.92),
    ]


def _build_speaker_turns() -> list[dict[str, object]]:
    return [
        {
            "turn_id": "turn-a",
            "speaker_id": "speaker-a",
            "start": 10.0,
            "end": 10.55,
            "boundary_confidence": 0.92,
        },
        {
            "turn_id": "turn-b",
            "speaker_id": "speaker-b",
            "start": 10.55,
            "end": 11.35,
            "boundary_confidence": 0.95,
        },
    ]


def _build_anchor_ready_window() -> ReadySlowWindow:
    return ReadySlowWindow(
        window_id="phase4-window-fast-ingress-shape",
        owner_chunk_id="chunk-5",
        owner_chunk_index=5,
        window_mode="steady",
        flush_reason="test",
        audio_segments=((10.0, 11.35),),
        coverage=WindowCoverage(
            core_segments=((10.0, 11.35),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-5",
                    chunk_index=5,
                    chunk_start=10.0,
                    chunk_end=11.35,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
            ),
        ),
        source_semantic_chunk_ids=("sem-5",),
        source_chunk_ids=("chunk-5",),
        source_chunk_indices=(5,),
        source_units=(
            WindowSourceUnit(
                unit_id="unit-5",
                semantic_chunk_id="sem-5",
                text="Wouldn't it make sense",
                audio_start=10.0,
                audio_end=11.35,
                source_chunk_ids=("chunk-5",),
                source_chunk_indices=(5,),
                speaker_id="speaker-a",
                turn_id="turn-a",
                language="en",
                arrived_at=11.35,
            ),
        ),
        dialogue_shape=DialogueShapeSnapshot(
            shape="two_party_stable",
            speaker_count=2,
            dominant_speaker_id="speaker-a",
            dominant_speaker_ratio=0.5,
            speaker_switch_count=1,
            speaker_switch_density=0.5,
            turn_count=2,
            avg_turn_duration_sec=0.675,
        ),
        language_profile=WindowLanguageProfile(
            primary_language="en",
            language_mix_state="single_language",
            decision_domains=("timeanchored_alignment",),
            should_bypass_whisper=False,
        ),
        prompt_seed=PromptSeed(text="Wouldn't it make sense"),
        batch_hint=WindowBatchHint(
            duration_bucket="short",
            token_estimate=4,
            acoustic_density_hint="medium",
            queue_priority=1,
        ),
        created_at=11.35,
    )


def _build_anchor_decision_input():
    ready_window = _build_anchor_ready_window()
    window_time_base = WindowTimeBasePackage(
        window_id=ready_window.window_id,
        language="en",
        raw_units=(
            TimeBaseUnit(text="Wouldn't", start=10.0, end=10.35, confidence=0.95, token_type="word"),
            TimeBaseUnit(text="it", start=10.35, end=10.55, confidence=0.95, token_type="word"),
            TimeBaseUnit(text="make", start=10.55, end=10.95, confidence=0.95, token_type="word"),
            TimeBaseUnit(text="sense", start=10.95, end=11.35, confidence=0.95, token_type="word"),
        ),
        word_units=(
            TimeBaseUnit(text="Wouldn't", start=10.0, end=10.35, confidence=0.95, token_type="word"),
            TimeBaseUnit(text="it", start=10.35, end=10.55, confidence=0.95, token_type="word"),
            TimeBaseUnit(text="make", start=10.55, end=10.95, confidence=0.95, token_type="word"),
            TimeBaseUnit(text="sense", start=10.95, end=11.35, confidence=0.95, token_type="word"),
        ),
        quality=TimeBaseQuality(blank_ratio=0.05, avg_max_prob=0.95, low_prob_ratio=0.02),
        source_chunk_ids=("chunk-5",),
        source_chunk_indices=(5,),
        chunk_bindings=ready_window.coverage.chunk_bindings,
    )
    preparation = AlignmentPreparationAssembler().prepare(
        ready_window=ready_window,
        window_time_base=window_time_base,
        selected_text_truth=SelectedTextTruth(
            text="Wouldn't it make sense",
            text_source="slow",
            language_hint="en",
            source_chunk_ids=("chunk-5",),
            quality={"confidence": 0.9},
            metadata={"raw_text": "Wouldn't it make sense"},
        ),
        whisper_result={
            "text": "Wouldn't it make sense",
            "text_clean": "Wouldn't it make sense",
            "text_itn_raw": "Wouldn't it make sense",
            "confidence": 0.9,
            "language": "en",
            "raw_result": {"segments": [{"avg_logprob": -0.1}]},
        },
        default_language="en",
        external_speaker_turns=(),
    )
    decoder_result = AlignmentDecoderService().execute(preparation=preparation)
    return AlignmentPathAdapter().build(
        preparation=preparation,
        decoder_result=decoder_result,
    ).decision_input


def test_fast_timed_ingress_adapter_builds_decision_input_from_fast_words() -> None:
    result = FastTimedIngressAdapter().build(
        chunk=_build_chunk(),
        fast_words=_build_fast_words(),
        language="en",
        speaker_turns=[],
        punctuation_positions=[
            PuncPosition(char_index=22, punctuation="?", confidence=0.9),
        ],
        fallback_clean_text="Wouldn't it make sense?",
        speaker_id="speaker-a",
        turn_id="turn-a",
    )

    decision_input = result.decision_input
    assert [item.word for item in decision_input.annotated_words] == ["Wouldn't", "it", "make", "sense"]
    assert decision_input.aligned_facts is not None
    assert decision_input.aligned_facts.time_axis_version == "fast_timed_direct"
    assert decision_input.fallback_clean_text_ref == "Wouldn't it make sense?"
    assert result.ingress_context.metadata["segmentation_input_version"] == "fast_timed_v1"
    assert result.compat_report["time_mapping_count"] == 4


def test_fast_timed_ingress_adapter_preserves_speaker_turn_facts_when_enabled() -> None:
    result = FastTimedIngressAdapter().build(
        chunk=_build_chunk(),
        fast_words=_build_fast_words(),
        language="en",
        speaker_turns=_build_speaker_turns(),
        punctuation_positions=[],
        fallback_clean_text="Wouldn't it make sense",
        speaker_id="speaker-a",
        turn_id="turn-a",
    )

    speaker_turns = result.decision_input.aligned_facts.speaker_turns
    assert [item["speaker_id"] for item in speaker_turns] == ["speaker-a", "speaker-b"]
    assert any(
        item["reason"] == "speaker_change" and item["decision_time"] == 10.55
        for item in result.decision_input.fused_evidence.speaker_changes
    )


def test_fast_timed_ingress_adapter_and_alignment_path_adapter_share_shape() -> None:
    fast_input = FastTimedIngressAdapter().build(
        chunk=_build_chunk(),
        fast_words=_build_fast_words(),
        language="en",
        speaker_turns=[],
        punctuation_positions=[],
        fallback_clean_text="Wouldn't it make sense",
        speaker_id="speaker-a",
        turn_id="turn-a",
    ).decision_input
    anchor_input = _build_anchor_decision_input()

    assert set(vars(fast_input).keys()) == set(vars(anchor_input).keys())
