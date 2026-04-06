from __future__ import annotations

from app.models.speaker_timeline_models import SpeakerTurn
from app.services.textflow.alignment_path_adapter import AlignmentPathAdapter
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


def _build_ready_window() -> ReadySlowWindow:
    return ReadySlowWindow(
        window_id="phase4-window-001",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        window_mode="steady",
        flush_reason="test",
        audio_segments=((0.0, 1.0), (1.0, 2.0)),
        coverage=WindowCoverage(
            core_segments=((0.0, 2.0),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-1",
                    chunk_index=1,
                    chunk_start=0.0,
                    chunk_end=1.0,
                    overlap_ratio=1.0,
                    role="owner",
                    is_owner=True,
                ),
                WindowChunkBinding(
                    chunk_id="chunk-2",
                    chunk_index=2,
                    chunk_start=1.0,
                    chunk_end=2.0,
                    overlap_ratio=1.0,
                    role="core",
                    is_owner=False,
                ),
            ),
        ),
        source_semantic_chunk_ids=("sem-1", "sem-2"),
        source_chunk_ids=("chunk-1", "chunk-2"),
        source_chunk_indices=(1, 2),
        source_units=(
            WindowSourceUnit(
                unit_id="unit-1",
                semantic_chunk_id="sem-1",
                text="你好",
                audio_start=0.0,
                audio_end=1.0,
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
                speaker_id="speaker-a",
                turn_id="turn-a",
                language="zh",
                arrived_at=1.0,
            ),
            WindowSourceUnit(
                unit_id="unit-2",
                semantic_chunk_id="sem-2",
                text="世界",
                audio_start=1.0,
                audio_end=2.0,
                source_chunk_ids=("chunk-2",),
                source_chunk_indices=(2,),
                speaker_id="speaker-b",
                turn_id="turn-b",
                language="zh",
                arrived_at=2.0,
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
            avg_turn_duration_sec=1.0,
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
        created_at=2.0,
    )


def _build_window_time_base() -> WindowTimeBasePackage:
    units = (
        TimeBaseUnit(text="你", start=0.0, end=0.1, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="好", start=0.1, end=0.2, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="世", start=1.0, end=1.1, confidence=0.95, token_type="word"),
        TimeBaseUnit(text="界", start=1.1, end=1.2, confidence=0.95, token_type="word"),
    )
    ready_window = _build_ready_window()
    return WindowTimeBasePackage(
        window_id=ready_window.window_id,
        language="zh",
        raw_units=units,
        word_units=units,
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.9, low_prob_ratio=0.05),
        source_chunk_ids=ready_window.source_chunk_ids,
        source_chunk_indices=ready_window.source_chunk_indices,
        chunk_bindings=ready_window.coverage.chunk_bindings,
    )


def _build_preparation_bundle(*, selected_source: str = "slow"):
    ready_window = _build_ready_window()
    selected_text_truth = SelectedTextTruth(
        text="你好世界",
        text_source=selected_source,
        language_hint="zh",
        source_chunk_ids=ready_window.source_chunk_ids,
        quality={"confidence": 0.9},
        metadata={"raw_text": "你好，世界！"},
    )
    return AlignmentPreparationAssembler().prepare(
        ready_window=ready_window,
        window_time_base=_build_window_time_base(),
        selected_text_truth=selected_text_truth,
        whisper_result={
            "text": "你好，世界！",
            "text_clean": "你好世界",
            "text_itn_raw": "你好，世界！",
            "confidence": 0.9,
            "language": "zh",
            "raw_result": {"segments": [{"avg_logprob": -0.1}]},
        },
        default_language="zh",
    )


def test_alignment_path_adapter_builds_decision_input_from_decoder_alignment_path() -> None:
    preparation = _build_preparation_bundle()
    decoder_result = AlignmentDecoderService().execute(preparation=preparation)

    result = AlignmentPathAdapter().build(
        preparation=preparation,
        decoder_result=decoder_result,
    )

    decision_input = result.decision_input
    assert [item.word for item in decision_input.annotated_words] == ["你", "好", "世", "界"]
    assert decision_input.fallback_clean_text_ref == preparation.canonical_sequence.normalized_text
    assert len(decision_input.canonical_candidate_boundaries) == 1
    assert decision_input.canonical_candidate_boundaries[0].reason == "gap_pause"
    assert any(item.punct_class == "sentence_end" for item in decision_input.canonical_punctuation_facts)
    assert decision_input.ingress_context is result.ingress_context
    assert result.ingress_context.chunk_id == preparation.window_id
    assert result.ingress_context.metadata["segmentation_input_version"] == "alignment_path_v1"
    assert result.ingress_context.metadata["alignment_route"] == "alignment_path"
    assert result.compat_report["alignment_boundary_count"] == 1
    assert result.compat_report["alignment_path_token_count"] == len(
        decoder_result.alignment_path.aligned_tokens
    )


def test_alignment_path_adapter_rejects_decoder_results_without_alignment_path() -> None:
    preparation = _build_preparation_bundle(selected_source="fast")
    decoder_result = AlignmentDecoderService().execute(preparation=preparation)

    try:
        AlignmentPathAdapter().build(
            preparation=preparation,
            decoder_result=decoder_result,
        )
    except ValueError as exc:
        assert "AlignmentPath" in str(exc)
    else:
        raise AssertionError("缺少 AlignmentPath 时必须拒绝建立正式切分入口")


def test_alignment_path_adapter_builds_speaker_change_anchor_from_timeline_turns_inside_single_source_unit() -> None:
    ready_window = ReadySlowWindow(
        window_id="phase4-window-timeline-001",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        window_mode="steady",
        flush_reason="test",
        audio_segments=((0.0, 1.2),),
        coverage=WindowCoverage(
            core_segments=((0.0, 1.2),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=(
                WindowChunkBinding(
                    chunk_id="chunk-1",
                    chunk_index=1,
                    chunk_start=0.0,
                    chunk_end=1.2,
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
                text="Wouldn't it make sense",
                audio_start=0.0,
                audio_end=1.2,
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
                speaker_id="speaker-a",
                turn_id="unit-turn-a",
                language="en",
                arrived_at=1.2,
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
            avg_turn_duration_sec=1.2,
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
        created_at=1.2,
    )
    window_time_base = WindowTimeBasePackage(
        window_id=ready_window.window_id,
        language="en",
        raw_units=(
            TimeBaseUnit(text="Wouldn't", start=0.0, end=0.25, confidence=0.95, token_type="word"),
            TimeBaseUnit(text="it", start=0.25, end=0.45, confidence=0.95, token_type="word"),
            TimeBaseUnit(text="make", start=0.45, end=0.75, confidence=0.95, token_type="word"),
            TimeBaseUnit(text="sense", start=0.75, end=1.2, confidence=0.95, token_type="word"),
        ),
        word_units=(
            TimeBaseUnit(text="Wouldn't", start=0.0, end=0.25, confidence=0.95, token_type="word"),
            TimeBaseUnit(text="it", start=0.25, end=0.45, confidence=0.95, token_type="word"),
            TimeBaseUnit(text="make", start=0.45, end=0.75, confidence=0.95, token_type="word"),
            TimeBaseUnit(text="sense", start=0.75, end=1.2, confidence=0.95, token_type="word"),
        ),
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.9, low_prob_ratio=0.05),
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        chunk_bindings=ready_window.coverage.chunk_bindings,
    )
    preparation = AlignmentPreparationAssembler().prepare(
        ready_window=ready_window,
        window_time_base=window_time_base,
        selected_text_truth=SelectedTextTruth(
            text="Wouldn't it make sense",
            text_source="slow",
            language_hint="en",
            source_chunk_ids=("chunk-1",),
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
        external_speaker_turns=(
            SpeakerTurn(
                turn_id="timeline-turn-1",
                block_id="timeline-block",
                speaker_id="speaker-a",
                start=0.0,
                end=0.45,
                boundary_confidence=0.92,
                is_overlap=False,
                source="merged",
            ),
            SpeakerTurn(
                turn_id="timeline-turn-2",
                block_id="timeline-block",
                speaker_id="speaker-b",
                start=0.45,
                end=1.2,
                boundary_confidence=0.94,
                is_overlap=False,
                source="merged",
            ),
        ),
    )
    decoder_result = AlignmentDecoderService().execute(preparation=preparation)

    result = AlignmentPathAdapter().build(
        preparation=preparation,
        decoder_result=decoder_result,
    )

    assert [item["speaker_id"] for item in result.decision_input.aligned_facts.speaker_turns] == [
        "speaker-a",
        "speaker-b",
    ]
    assert any(
        item["reason"] == "speaker_change" and item["decision_time"] == 0.45
        for item in result.decision_input.fused_evidence.speaker_changes
    )
