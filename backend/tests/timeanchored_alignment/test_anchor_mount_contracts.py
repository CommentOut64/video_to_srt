from __future__ import annotations

from dataclasses import fields

import pytest

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorMountResult,
    BoundaryHint,
    DecisionIngressPackage,
    DecisionToken,
)
from app.services.timeanchored_alignment.anchor_mount.ingress_validator import (
    IngressValidator,
)
from app.services.timeanchored_alignment.chunk_projector import ChunkWindow
from app.services.timeanchored_alignment.contracts import (
    LanguageRun,
    LanguageRunPackage,
    PronunciationPackage,
    TextTruthPackage,
    TextTruthQuality,
    TextTruthUnit,
    TimeBaseQuality,
    TimeBaseUnit,
)
from app.services.timeanchored_alignment.preparation.contracts import (
    AlignmentPreparationCompat,
    AlignmentPreparationPackage,
    FastHook,
    PreparedSlowText,
    SlowSlot,
    SlowWindowTextPackage,
)
from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowChunkBinding,
    WindowCoverage,
)
from app.services.timeanchored_alignment.window_time_base_assembler import (
    WindowTimeBasePackage,
)


def _build_preparation_for_ingress_validation(
    *,
    slots: tuple[SlowSlot, ...],
    fast_hooks: tuple[FastHook, ...],
    window_text: str = "hello world",
) -> AlignmentPreparationPackage:
    coverage = WindowCoverage(
        core_segments=((0.0, 1.0),),
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
        ),
    )
    time_base = WindowTimeBasePackage(
        window_id="window-1",
        language="en",
        raw_units=(
            TimeBaseUnit(text="hello", start=0.0, end=0.4, confidence=0.95, token_type="word"),
        ),
        word_units=(
            TimeBaseUnit(text="hello", start=0.0, end=0.4, confidence=0.95, token_type="word"),
        ),
        quality=TimeBaseQuality(blank_ratio=0.0, avg_max_prob=0.9, low_prob_ratio=0.0),
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        chunk_bindings=coverage.chunk_bindings,
    )
    return AlignmentPreparationPackage(
        window_id="window-1",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        slow_text=PreparedSlowText(
            window_text=SlowWindowTextPackage(text=window_text, display_text=window_text, source_language="en"),
            slots=slots,
            punctuation_evidences=tuple(),
            protected_units=tuple(),
            language_runs=(
                LanguageRun(
                    run_text=window_text,
                    run_language="en",
                    char_start=0,
                    char_end=len(window_text),
                ),
            ),
            pronunciation_hints=tuple(),
        ),
        fast_hooks=fast_hooks,
        coverage=coverage,
        compat=AlignmentPreparationCompat(
            time_base=time_base,
            text_truth=TextTruthPackage(
                units=(
                    TextTruthUnit(
                        text=window_text,
                        normalized_text=window_text,
                        confidence=0.95,
                        language="en",
                    ),
                ),
                quality=TextTruthQuality(
                    hallucination_risk=0.0,
                    repetition_ratio=0.0,
                    length_ratio=1.0,
                ),
                language="en",
                raw_text=window_text,
                normalized_text=window_text,
            ),
            protected_spans=tuple(),
            language_runs=LanguageRunPackage(
                runs=(
                    LanguageRun(
                        run_text=window_text,
                        run_language="en",
                        char_start=0,
                        char_end=len(window_text),
                    ),
                ),
                dominant_language="en",
                window_kind="single_language",
                foreign_run_ratio=0.0,
                source_text=window_text,
            ),
            pronunciation=PronunciationPackage(
                token_units=tuple(),
                phone_units=tuple(),
                token_to_phone_spans=tuple(),
                frontend_source="test",
                dependency_mode={},
                language="en",
            ),
            pronunciation_report={},
            chunk_window=ChunkWindow(chunk_ref="chunk-1", start=0.0, end=1.0),
        ),
    )


def test_anchor_mount_result_contract_exposes_mount_fact_and_boundary_fields() -> None:
    field_names = {field.name for field in fields(AnchorMountResult)}

    assert "items" in field_names
    assert "envelopes" in field_names
    assert "punctuation_facts" in field_names
    assert "punctuation_pair_states" in field_names
    assert "cross_chunk_locks" in field_names
    assert "boundary_hints" in field_names
    assert "hook_claims" in field_names


def test_decision_ingress_package_contract_exposes_tokens_punctuation_and_coverage() -> None:
    field_names = {field.name for field in fields(DecisionIngressPackage)}

    assert "tokens" in field_names
    assert "punctuation_facts" in field_names
    assert "punctuation_pair_states" in field_names
    assert "boundary_hints" in field_names
    assert "coverage" in field_names


def test_decision_token_rejects_none_start_end() -> None:
    with pytest.raises(ValueError, match="DecisionToken.start/end 不能为空"):
        DecisionToken(
            token_id="token-1",
            slot_index=0,
            text_core="你",
            display_text="你",
            normalized_text="你",
            start=None,
            end=0.1,
            left_bound=0.0,
            right_bound=0.1,
            speaker_id=None,
            turn_id=None,
            source_chunk_ids=("chunk-1",),
            source_chunk_indices=(1,),
            source_hook_ids=("hook-0",),
            metadata={},
        )


def test_boundary_hint_requires_decision_time() -> None:
    with pytest.raises(ValueError, match="BoundaryHint.decision_time 必须 >= 0"):
        BoundaryHint(
            split_after_slot_id="slot-0",
            decision_time=-0.1,
            score=0.9,
            reason="punctuation_sentence_end",
            hard_flag=True,
        )


def test_ingress_validator_rejects_slot_that_runs_past_window_text() -> None:
    preparation = _build_preparation_for_ingress_validation(
        slots=(
            SlowSlot(
                slot_id="slot-0",
                text="hello",
                char_start=0,
                char_end=12,
                speaker_id=None,
                turn_id=None,
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
            ),
        ),
        fast_hooks=(
            FastHook(
                hook_text="hello",
                start=0.0,
                end=0.4,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
            ),
        ),
    )

    with pytest.raises(ValueError, match="SlowSlot.char_end 越过 window_text 边界"):
        IngressValidator().validate(preparation=preparation, language="en")


def test_ingress_validator_rejects_slot_without_window_provenance() -> None:
    preparation = _build_preparation_for_ingress_validation(
        slots=(
            SlowSlot(
                slot_id="slot-0",
                text="hello",
                char_start=0,
                char_end=5,
                speaker_id=None,
                turn_id=None,
                source_chunk_ids=tuple(),
                source_chunk_indices=tuple(),
            ),
        ),
        fast_hooks=(
            FastHook(
                hook_text="hello",
                start=0.0,
                end=0.4,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
            ),
        ),
    )

    with pytest.raises(ValueError, match="SlowSlot 必须保留 source chunk provenance"):
        IngressValidator().validate(preparation=preparation, language="en")


def test_ingress_validator_rejects_non_monotonic_fast_hooks() -> None:
    preparation = _build_preparation_for_ingress_validation(
        slots=(
            SlowSlot(
                slot_id="slot-0",
                text="hello",
                char_start=0,
                char_end=5,
                speaker_id=None,
                turn_id=None,
                source_chunk_ids=("chunk-1",),
                source_chunk_indices=(1,),
            ),
        ),
        fast_hooks=(
            FastHook(
                hook_text="world",
                start=0.5,
                end=0.8,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
            ),
            FastHook(
                hook_text="hello",
                start=0.1,
                end=0.3,
                confidence=0.95,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
            ),
        ),
    )

    with pytest.raises(ValueError, match="FastHook 必须按时间单调排列"):
        IngressValidator().validate(preparation=preparation, language="en")
