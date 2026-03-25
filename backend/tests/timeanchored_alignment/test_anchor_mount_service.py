from __future__ import annotations

from types import SimpleNamespace

from app.services.timeanchored_alignment.anchor_mount.chain_solver import ChainSolver
from app.services.timeanchored_alignment.anchor_mount.contracts import LocalAlignmentBlock
from app.services.timeanchored_alignment.anchor_mount.service import AnchorMountAlignmentService
from app.services.timeanchored_alignment.preparation.contracts import (
    AlignmentPreparationPackage,
    AlignmentPreparationCompat,
    FastHook,
    PreparedSlowText,
    PunctuationEvidence,
    PronunciationHint,
    ProtectedUnit,
    SlowSlot,
    SlowWindowTextPackage,
)
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
from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowChunkBinding,
    WindowCoverage,
)
from app.services.timeanchored_alignment.window_time_base_assembler import WindowTimeBasePackage


def _build_preparation_with_unresolved_middle_slot() -> AlignmentPreparationPackage:
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
    return AlignmentPreparationPackage(
        window_id="window-service-001",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        slow_text=PreparedSlowText(
            window_text=SlowWindowTextPackage(text="hellobraveworld", display_text="hellobraveworld", source_language="en"),
            slots=(
                SlowSlot(
                    slot_id="slot-0",
                    text="hello",
                    char_start=0,
                    char_end=5,
                    speaker_id="speaker-a",
                    turn_id="turn-a",
                    source_chunk_ids=("chunk-1",),
                    source_chunk_indices=(1,),
                ),
                SlowSlot(
                    slot_id="slot-1",
                    text="brave",
                    char_start=5,
                    char_end=10,
                    speaker_id="speaker-a",
                    turn_id="turn-a",
                    source_chunk_ids=("chunk-1",),
                    source_chunk_indices=(1,),
                ),
                SlowSlot(
                    slot_id="slot-2",
                    text="world",
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
    )


def test_anchor_mount_service_finalizes_hook_claims_and_marks_fallback_for_unresolved_gap() -> None:
    service = AnchorMountAlignmentService()
    preparation = _build_preparation_with_unresolved_middle_slot()

    result = service.align(preparation=preparation, language="en")

    assert result.anchor_mount_result.should_fallback is True
    assert all(claim.finalized for claim in result.anchor_mount_result.hook_claims)
    assert result.anchor_mount_result.envelopes[1].envelope_kind in {"inferred", "unresolved"}
    assert result.decision_ingress.tokens[1].start is not None
    assert result.decision_ingress.tokens[1].end is not None


def test_anchor_mount_service_metrics_include_documented_quality_signals() -> None:
    service = AnchorMountAlignmentService()
    preparation = _build_preparation_with_unresolved_middle_slot()

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
        input_view=SimpleNamespace(slots=(SimpleNamespace(), SimpleNamespace())),
        blocks=(
            LocalAlignmentBlock(
                block_id="block-0",
                slot_indices=(0,),
                hook_indices=(1,),
                score=0.30,
                block_kind="anchored",
                anchor_kind="exact",
            ),
            LocalAlignmentBlock(
                block_id="block-1",
                slot_indices=(1,),
                hook_indices=(0,),
                score=0.95,
                block_kind="anchored",
                anchor_kind="exact",
            ),
        ),
    )

    assert result.reseed_count == 1
    assert [block.block_id for block in result.committed_blocks] == ["block-1"]
