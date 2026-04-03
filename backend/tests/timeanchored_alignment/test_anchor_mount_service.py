from __future__ import annotations

from types import SimpleNamespace

from app.services.timeanchored_alignment.anchor_mount.chain_solver import ChainSolver
from app.services.timeanchored_alignment.anchor_mount.contracts import LocalAlignmentBlock
from app.services.timeanchored_alignment.anchor_mount.service import AnchorMountAlignmentService
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
    ProtectedUnit,
    PunctuationEvidence,
    PronunciationGraph,
    PronunciationHint,
    SlowWindowTextPackage,
)
from app.services.timeanchored_alignment.contracts import (
    AcousticObservationPack,
    AcousticObservationQuality,
    AcousticObservationSlice,
    LanguageRun,
    LanguageRunPackage,
    LayerSummary,
    OBSERVATION_CAPABILITY_TIMESTAMP_ONLY,
    PronunciationPackage,
    TextTruthPackage,
    TextTruthQuality,
    TextTruthUnit,
    TimeBaseQuality,
    TimeBaseUnit,
)
from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowChunkBinding,
    WindowCoverage,
)
from app.services.timeanchored_alignment.window_time_base_assembler import WindowTimeBasePackage


def _build_phase2_fields(
    *,
    window_id: str,
    source_chunk_ids: tuple[str, ...],
    source_chunk_indices: tuple[int, ...],
    window_text: str,
    token_units: tuple[PreparedTokenUnit, ...],
    language_runs: tuple[LanguageRun, ...],
    time_base: WindowTimeBasePackage,
    absolute_time_range: tuple[float, float],
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
                    is_protected=False,
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
            absolute_time_range=absolute_time_range,
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
            absolute_time_range=absolute_time_range,
        ),
        "provenance": PreparationProvenance(
            text_source="slow",
            observation_source="test",
        ),
        "external_stable_facts": ExternalStableFacts(),
        "report": PreparationReport(summary=LayerSummary(layer="preparation")),
    }


def _build_preparation_with_unresolved_middle_unit() -> PreparationBundle:
    coverage = WindowCoverage(
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
    )
    compat_time_base = WindowTimeBasePackage(
        window_id="window-service-001",
        language="en",
        raw_units=(
            TimeBaseUnit(text="hello", start=0.0, end=0.3, confidence=0.9, token_type="word"),
            TimeBaseUnit(text="world", start=0.9, end=1.2, confidence=0.9, token_type="word"),
        ),
        word_units=(
            TimeBaseUnit(text="hello", start=0.0, end=0.3, confidence=0.9, token_type="word"),
            TimeBaseUnit(text="world", start=0.9, end=1.2, confidence=0.9, token_type="word"),
        ),
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.9, low_prob_ratio=0.05),
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        chunk_bindings=coverage.chunk_bindings,
    )
    return PreparationBundle(
        window_id="window-service-001",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        slow_text=PreparedSlowText(
            window_text=SlowWindowTextPackage(
                text="hellobraveworld",
                display_text="hellobraveworld",
                source_language="en",
            ),
            token_units=(
                PreparedTokenUnit(
                    unit_id="unit-0",
                    token_text="hello",
                    normalized_text="hello",
                    char_start=0,
                    char_end=5,
                    speaker_id="speaker-a",
                    turn_id="turn-a",
                    source_chunk_ids=("chunk-1",),
                    source_chunk_indices=(1,),
                ),
                PreparedTokenUnit(
                    unit_id="unit-1",
                    token_text="brave",
                    normalized_text="brave",
                    char_start=5,
                    char_end=10,
                    speaker_id="speaker-a",
                    turn_id="turn-a",
                    source_chunk_ids=("chunk-1",),
                    source_chunk_indices=(1,),
                ),
                PreparedTokenUnit(
                    unit_id="unit-2",
                    token_text="world",
                    normalized_text="world",
                    char_start=10,
                    char_end=15,
                    speaker_id="speaker-a",
                    turn_id="turn-a",
                    source_chunk_ids=("chunk-1",),
                    source_chunk_indices=(1,),
                ),
            ),
            punctuation_evidences=(PunctuationEvidence(mark="!", source_char_index=14),),
            protected_units=(ProtectedUnit(start=5, end=10, kind="word", text="brave"),),
            language_runs=(
                LanguageRun(
                    run_text="hellobraveworld",
                    run_language="en",
                    char_start=0,
                    char_end=15,
                ),
            ),
            pronunciation_hints=(
                PronunciationHint(
                    token_text="brave",
                    reading_key="brave",
                    language="en",
                    char_start=5,
                    char_end=10,
                ),
            ),
        ),
        fast_hooks=(
            FastHook(
                hook_text="hello",
                start=0.0,
                end=0.3,
                confidence=0.9,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
            ),
            FastHook(
                hook_text="world",
                start=0.9,
                end=1.2,
                confidence=0.9,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
            ),
        ),
        coverage=coverage,
        compat=AlignmentPreparationCompat(
            time_base=compat_time_base,
            text_truth=TextTruthPackage(
                raw_text="hello brave world",
                normalized_text="hello brave world",
                units=(
                    TextTruthUnit(
                        text="hello",
                        normalized_text="hello",
                        confidence=0.9,
                        language="en",
                        source="slow",
                    ),
                    TextTruthUnit(
                        text="brave",
                        normalized_text="brave",
                        confidence=0.9,
                        language="en",
                        source="slow",
                    ),
                    TextTruthUnit(
                        text="world",
                        normalized_text="world",
                        confidence=0.9,
                        language="en",
                        source="slow",
                    ),
                ),
                protected_spans=tuple(),
                quality=TextTruthQuality(
                    hallucination_risk=0.0,
                    repetition_ratio=0.0,
                    length_ratio=1.0,
                ),
                language="en",
                is_hallucination=False,
            ),
            protected_spans=tuple(),
            language_runs=LanguageRunPackage(
                runs=(
                    LanguageRun(
                        run_text="hello brave world",
                        run_language="en",
                        char_start=0,
                        char_end=17,
                    ),
                ),
                dominant_language="en",
                window_kind="single_language",
                foreign_run_ratio=0.0,
                source_text="hello brave world",
            ),
            pronunciation=PronunciationPackage(
                token_units=tuple(),
                phone_units=tuple(),
                token_to_phone_spans=tuple(),
                frontend_source="test",
                dependency_mode={},
                language="en",
            ),
            pronunciation_report={"source": "test"},
            chunk_window=type("ChunkWindow", (), {"chunk_ref": 1, "start": 0.0, "end": 1.2})(),
        ),
        **_build_phase2_fields(
            window_id="window-service-001",
            source_chunk_ids=("chunk-1",),
            source_chunk_indices=(1,),
            window_text="hellobraveworld",
            token_units=(
                PreparedTokenUnit(
                    unit_id="unit-0",
                    token_text="hello",
                    normalized_text="hello",
                    char_start=0,
                    char_end=5,
                    speaker_id="speaker-a",
                    turn_id="turn-a",
                    source_chunk_ids=("chunk-1",),
                    source_chunk_indices=(1,),
                ),
                PreparedTokenUnit(
                    unit_id="unit-1",
                    token_text="brave",
                    normalized_text="brave",
                    char_start=5,
                    char_end=10,
                    speaker_id="speaker-a",
                    turn_id="turn-a",
                    source_chunk_ids=("chunk-1",),
                    source_chunk_indices=(1,),
                ),
                PreparedTokenUnit(
                    unit_id="unit-2",
                    token_text="world",
                    normalized_text="world",
                    char_start=10,
                    char_end=15,
                    speaker_id="speaker-a",
                    turn_id="turn-a",
                    source_chunk_ids=("chunk-1",),
                    source_chunk_indices=(1,),
                ),
            ),
            language_runs=(
                LanguageRun(
                    run_text="hellobraveworld",
                    run_language="en",
                    char_start=0,
                    char_end=15,
                ),
            ),
            time_base=compat_time_base,
            absolute_time_range=(0.0, 1.2),
        ),
    )


def _build_preparation_with_noisy_contiguous_run() -> PreparationBundle:
    coverage = WindowCoverage(
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
    )
    compat_time_base = WindowTimeBasePackage(
        window_id="window-service-002",
        language="en",
        raw_units=(
            TimeBaseUnit(text="probably", start=0.0, end=0.06, confidence=0.9, token_type="word"),
            TimeBaseUnit(text="more", start=0.48, end=0.54, confidence=0.9, token_type="word"),
            TimeBaseUnit(text="beneficial", start=0.9, end=0.96, confidence=0.9, token_type="word"),
        ),
        word_units=(
            TimeBaseUnit(text="probably", start=0.0, end=0.06, confidence=0.9, token_type="word"),
            TimeBaseUnit(text="more", start=0.48, end=0.54, confidence=0.9, token_type="word"),
            TimeBaseUnit(text="beneficial", start=0.9, end=0.96, confidence=0.9, token_type="word"),
        ),
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.9, low_prob_ratio=0.05),
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        chunk_bindings=coverage.chunk_bindings,
    )
    return PreparationBundle(
        window_id="window-service-002",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        slow_text=PreparedSlowText(
            window_text=SlowWindowTextPackage(
                text="probably more beneficial",
                display_text="probably more beneficial",
                source_language="en",
            ),
            token_units=(
                PreparedTokenUnit(
                    unit_id="unit-0",
                    token_text="probably",
                    normalized_text="probably",
                    char_start=0,
                    char_end=8,
                    speaker_id="speaker-a",
                    turn_id="turn-a",
                    source_chunk_ids=("chunk-1",),
                    source_chunk_indices=(1,),
                ),
                PreparedTokenUnit(
                    unit_id="unit-1",
                    token_text="more",
                    normalized_text="more",
                    char_start=9,
                    char_end=13,
                    speaker_id="speaker-a",
                    turn_id="turn-a",
                    source_chunk_ids=("chunk-1",),
                    source_chunk_indices=(1,),
                ),
                PreparedTokenUnit(
                    unit_id="unit-2",
                    token_text="beneficial",
                    normalized_text="beneficial",
                    char_start=14,
                    char_end=24,
                    speaker_id="speaker-a",
                    turn_id="turn-a",
                    source_chunk_ids=("chunk-1",),
                    source_chunk_indices=(1,),
                ),
            ),
            punctuation_evidences=tuple(),
            protected_units=tuple(),
            language_runs=(
                LanguageRun(
                    run_text="probably more beneficial",
                    run_language="en",
                    char_start=0,
                    char_end=24,
                ),
            ),
            pronunciation_hints=tuple(),
        ),
        fast_hooks=(
            FastHook(
                hook_text="probably",
                start=0.0,
                end=0.06,
                confidence=0.9,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
            ),
            FastHook(
                hook_text="more",
                start=0.48,
                end=0.54,
                confidence=0.9,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
            ),
            FastHook(
                hook_text="beneficial",
                start=0.9,
                end=0.96,
                confidence=0.9,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
            ),
        ),
        coverage=coverage,
        compat=AlignmentPreparationCompat(
            time_base=compat_time_base,
            text_truth=TextTruthPackage(
                raw_text="probably more beneficial",
                normalized_text="probably more beneficial",
                units=(
                    TextTruthUnit(
                        text="probably",
                        normalized_text="probably",
                        confidence=0.9,
                        language="en",
                        source="slow",
                    ),
                    TextTruthUnit(
                        text="more",
                        normalized_text="more",
                        confidence=0.9,
                        language="en",
                        source="slow",
                    ),
                    TextTruthUnit(
                        text="beneficial",
                        normalized_text="beneficial",
                        confidence=0.9,
                        language="en",
                        source="slow",
                    ),
                ),
                protected_spans=tuple(),
                quality=TextTruthQuality(
                    hallucination_risk=0.0,
                    repetition_ratio=0.0,
                    length_ratio=1.0,
                ),
                language="en",
                is_hallucination=False,
            ),
            protected_spans=tuple(),
            language_runs=LanguageRunPackage(
                runs=(
                    LanguageRun(
                        run_text="probably more beneficial",
                        run_language="en",
                        char_start=0,
                        char_end=24,
                    ),
                ),
                dominant_language="en",
                window_kind="single_language",
                foreign_run_ratio=0.0,
                source_text="probably more beneficial",
            ),
            pronunciation=PronunciationPackage(
                token_units=tuple(),
                phone_units=tuple(),
                token_to_phone_spans=tuple(),
                frontend_source="test",
                dependency_mode={},
                language="en",
            ),
            pronunciation_report={"source": "test"},
            chunk_window=type("ChunkWindow", (), {"chunk_ref": 1, "start": 0.0, "end": 1.2})(),
        ),
        **_build_phase2_fields(
            window_id="window-service-002",
            source_chunk_ids=("chunk-1",),
            source_chunk_indices=(1,),
            window_text="probably more beneficial",
            token_units=(
                PreparedTokenUnit(
                    unit_id="unit-0",
                    token_text="probably",
                    normalized_text="probably",
                    char_start=0,
                    char_end=8,
                    speaker_id="speaker-a",
                    turn_id="turn-a",
                    source_chunk_ids=("chunk-1",),
                    source_chunk_indices=(1,),
                ),
                PreparedTokenUnit(
                    unit_id="unit-1",
                    token_text="more",
                    normalized_text="more",
                    char_start=9,
                    char_end=13,
                    speaker_id="speaker-a",
                    turn_id="turn-a",
                    source_chunk_ids=("chunk-1",),
                    source_chunk_indices=(1,),
                ),
                PreparedTokenUnit(
                    unit_id="unit-2",
                    token_text="beneficial",
                    normalized_text="beneficial",
                    char_start=14,
                    char_end=24,
                    speaker_id="speaker-a",
                    turn_id="turn-a",
                    source_chunk_ids=("chunk-1",),
                    source_chunk_indices=(1,),
                ),
            ),
            language_runs=(
                LanguageRun(
                    run_text="probably more beneficial",
                    run_language="en",
                    char_start=0,
                    char_end=24,
                ),
            ),
            time_base=compat_time_base,
            absolute_time_range=(0.0, 1.2),
        ),
    )


def test_anchor_mount_service_finalizes_hook_claims_and_marks_fallback_for_unresolved_gap() -> None:
    service = AnchorMountAlignmentService()
    preparation = _build_preparation_with_unresolved_middle_unit()

    result = service.align(preparation=preparation, language="en")

    assert result.anchor_mount_result.should_fallback is True
    assert all(claim.finalized for claim in result.anchor_mount_result.hook_claims)
    assert result.anchor_mount_result.envelopes[1].envelope_kind in {"inferred", "unresolved"}
    assert result.decision_ingress.anchored_token_units[1].start is not None
    assert result.decision_ingress.anchored_token_units[1].end is not None


def test_anchor_mount_service_metrics_include_documented_quality_signals() -> None:
    service = AnchorMountAlignmentService()
    preparation = _build_preparation_with_unresolved_middle_unit()

    result = service.align(preparation=preparation, language="en")

    metrics = result.anchor_mount_result.metrics
    assert "largest_unresolved_span" in metrics
    assert "hook_waste_ratio" in metrics
    assert "soft_anchor_ratio" in metrics
    assert "hook_claim_conflict_count" in metrics
    assert "cross_chunk_lock_count" in metrics
    assert "duplicate_candidate_hook_count" in metrics
    assert "unmapped_punctuation_count" in metrics


def test_chain_solver_rolls_back_suffix_when_later_block_has_higher_score() -> None:
    solver = ChainSolver()

    result = solver.solve(
        input_view=SimpleNamespace(token_units=(SimpleNamespace(), SimpleNamespace())),
        blocks=(
            LocalAlignmentBlock(
                block_id="block-0",
                unit_indices=(0,),
                hook_indices=(1,),
                score=0.30,
                block_kind="anchored",
                anchor_kind="exact",
            ),
            LocalAlignmentBlock(
                block_id="block-1",
                unit_indices=(1,),
                hook_indices=(0,),
                score=0.95,
                block_kind="anchored",
                anchor_kind="exact",
            ),
        ),
    )

    assert result.reseed_count == 1
    assert [block.block_id for block in result.committed_blocks] == ["block-1"]


def test_anchor_mount_service_projects_alignment_block_identity_into_items() -> None:
    service = AnchorMountAlignmentService()
    preparation = _build_preparation_with_unresolved_middle_unit()

    result = service.align(preparation=preparation, language="en")

    block_ids = [item.alignment_block_id for item in result.anchor_mount_result.items]

    assert any(block_id for block_id in block_ids)


def test_anchor_mount_service_merges_contiguous_exact_run_and_smooths_internal_time_gaps() -> None:
    service = AnchorMountAlignmentService()
    preparation = _build_preparation_with_noisy_contiguous_run()

    result = service.align(preparation=preparation, language="en")

    items = result.anchor_mount_result.items
    block_ids = {item.alignment_block_id for item in items}
    assert len(block_ids) == 1

    boundary_reasons = {evidence.reason for evidence in result.anchor_mount_result.boundary_evidences}
    assert "anchor_block_close" not in boundary_reasons

    spans = [
        (token.start, token.end)
        for token in result.decision_ingress.anchored_token_units
    ]
    assert spans[0][0] == 0.0
    assert spans[-1][1] == 0.96
    assert spans[0][1] <= spans[1][0]
    assert spans[1][1] <= spans[2][0]
