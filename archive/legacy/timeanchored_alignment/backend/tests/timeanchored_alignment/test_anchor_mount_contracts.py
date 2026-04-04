from __future__ import annotations

from dataclasses import fields

import pytest

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorMountResult,
    AnchoredTokenUnit,
    DecisionIngressPackage,
)
from app.services.timeanchored_alignment.anchor_mount.ingress_validator import (
    IngressValidator,
)
from app.services.timeanchored_alignment.chunk_projector import ChunkWindow
from app.services.timeanchored_alignment.contracts import (
    AcousticObservationPack,
    AcousticObservationQuality,
    AcousticObservationSlice,
    LayerSummary,
    OBSERVATION_CAPABILITY_TIMESTAMP_ONLY,
    BoundaryEvidence,
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
    CanonicalSequence,
    CanonicalToken,
    ExternalStableFacts,
    FastHook,
    PreparationBundle,
    PreparationProvenance,
    PreparationReport,
    PreparationScope,
    PreparedSlowText,
    PreparedTokenUnit,
    PronunciationGraph,
    SlowWindowTextPackage,
)
from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowChunkBinding,
    WindowCoverage,
)
from app.services.timeanchored_alignment.window_time_base_assembler import (
    WindowTimeBasePackage,
)


def _build_phase2_fields(
    *,
    window_id: str,
    source_chunk_ids: tuple[str, ...],
    source_chunk_indices: tuple[int, ...],
    window_text: str,
    token_units: tuple[PreparedTokenUnit, ...],
    language_runs: tuple[LanguageRun, ...],
    time_base: WindowTimeBasePackage,
) -> dict[str, object]:
    return {
        "canonical_sequence": CanonicalSequence(
            original_text=window_text,
            normalized_text=window_text,
            tokens=tuple(
                CanonicalToken(
                    token_id=unit.unit_id,
                    text=unit.token_text,
                    normalized_text=unit.normalized_text,
                    char_start=unit.char_start,
                    char_end=unit.char_end,
                    language="en",
                    source_chunk_ids=unit.source_chunk_ids,
                    source_chunk_indices=unit.source_chunk_indices,
                )
                for unit in token_units
            ),
            protected_spans=tuple(),
            language_runs=language_runs,
            frontend_version="test",
            language_hint="en",
        ),
        "pronunciation_graph": PronunciationGraph(
            token_nodes=tuple(),
            state_nodes=tuple(),
            edges=tuple(),
        ),
        "acoustic_observation_pack": AcousticObservationPack(
            capability_level=OBSERVATION_CAPABILITY_TIMESTAMP_ONLY,
            adapter_type="test",
            source_chunk_ids=source_chunk_ids,
            source_chunk_indices=source_chunk_indices,
            absolute_time_range=(0.0, 1.0),
            slices=tuple(
                AcousticObservationSlice(
                    slice_id=f"slice-{index}",
                    start=float(unit.start),
                    end=float(unit.end),
                    primary_token=str(unit.text),
                    confidence=float(unit.confidence),
                )
                for index, unit in enumerate(time_base.word_units)
            ),
            quality=AcousticObservationQuality(
                slice_count=len(time_base.word_units),
                timestamp_coverage=1.0,
            ),
        ),
        "scope": PreparationScope(
            window_id=window_id,
            source_chunk_ids=source_chunk_ids,
            source_chunk_indices=source_chunk_indices,
            absolute_time_range=(0.0, 1.0),
        ),
        "provenance": PreparationProvenance(
            text_source="slow",
            observation_source="test",
        ),
        "external_stable_facts": ExternalStableFacts(),
        "report": PreparationReport(summary=LayerSummary(layer="preparation")),
    }


def _build_preparation_for_ingress_validation(
    *,
    token_units: tuple[PreparedTokenUnit, ...],
    fast_hooks: tuple[FastHook, ...],
    window_text: str = "hello world",
) -> PreparationBundle:
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
    return PreparationBundle(
        window_id="window-1",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        slow_text=PreparedSlowText(
            window_text=SlowWindowTextPackage(
                text=window_text,
                display_text=window_text,
                source_language="en",
            ),
            token_units=token_units,
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
        **_build_phase2_fields(
            window_id="window-1",
            source_chunk_ids=("chunk-1",),
            source_chunk_indices=(1,),
            window_text=window_text,
            token_units=token_units,
            language_runs=(
                LanguageRun(
                    run_text=window_text,
                    run_language="en",
                    char_start=0,
                    char_end=len(window_text),
                ),
            ),
            time_base=time_base,
        ),
    )


def test_anchor_mount_result_contract_exposes_mount_fact_and_boundary_fields() -> None:
    field_names = {field.name for field in fields(AnchorMountResult)}

    assert field_names >= {
        "items",
        "envelopes",
        "punctuation_facts",
        "punctuation_pair_states",
        "cross_chunk_locks",
        "boundary_evidences",
        "hook_claims",
    }
    assert "boundary_evidences" in field_names


def test_decision_ingress_package_contract_exposes_tokens_punctuation_and_coverage() -> None:
    field_names = {field.name for field in fields(DecisionIngressPackage)}

    assert field_names >= {
        "anchored_token_units",
        "punctuation_facts",
        "punctuation_pair_states",
        "boundary_evidences",
        "coverage",
    }


def test_anchored_token_unit_rejects_none_start_end() -> None:
    with pytest.raises(ValueError, match="AnchoredTokenUnit.start/end 不能为空"):
        AnchoredTokenUnit(
            unit_id="unit-1",
            token_text="你",
            normalized_text="你",
            start=None,
            end=0.1,
            left_bound=0.0,
            right_bound=0.1,
            speaker_id=None,
            turn_id=None,
            mount_status="anchored",
            anchor_kind="lexical",
            source_chunk_ids=("chunk-1",),
            source_chunk_indices=(1,),
            source_hook_ids=("hook-0",),
            match_confidence=0.9,
            cross_chunk_lock_ids=tuple(),
        )


def test_boundary_evidence_requires_non_negative_event_time() -> None:
    with pytest.raises(ValueError, match="BoundaryEvidence.event_time 必须 >= 0"):
        BoundaryEvidence(
            split_idx=0,
            event_time=-0.1,
            left_end=0.0,
            right_start=0.1,
            reason="lexical_boundary",
            score=0.9,
            hard_flag=False,
            metadata={},
        )


def test_ingress_validator_rejects_token_unit_that_runs_past_window_text() -> None:
    preparation = _build_preparation_for_ingress_validation(
        token_units=(
            PreparedTokenUnit(
                unit_id="unit-0",
                token_text="hello",
                normalized_text="hello",
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

    with pytest.raises(ValueError, match="PreparedTokenUnit.char_end 越过 window_text 边界"):
        IngressValidator().validate(preparation=preparation, language="en")


def test_ingress_validator_rejects_token_unit_without_window_provenance() -> None:
    preparation = _build_preparation_for_ingress_validation(
        token_units=(
            PreparedTokenUnit(
                unit_id="unit-0",
                token_text="hello",
                normalized_text="hello",
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

    with pytest.raises(ValueError, match="PreparedTokenUnit 必须保留 source chunk provenance"):
        IngressValidator().validate(preparation=preparation, language="en")


def test_ingress_validator_rejects_non_monotonic_fast_hooks() -> None:
    preparation = _build_preparation_for_ingress_validation(
        token_units=(
            PreparedTokenUnit(
                unit_id="unit-0",
                token_text="hello",
                normalized_text="hello",
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
