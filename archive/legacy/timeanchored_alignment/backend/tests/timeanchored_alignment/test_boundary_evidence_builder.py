from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.boundary_evidence_builder import (
    BoundaryEvidenceBuilder,
)
from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorMountItem,
    CrossChunkLock,
    PunctuationFact,
    TemporalEnvelope,
)


def _build_items() -> tuple[AnchorMountItem, ...]:
    return (
        AnchorMountItem(
            unit_id="unit-0",
            unit_index=0,
            token_text="hello",
            display_text="hello",
            normalized_text="hello",
            speaker_id="speaker-a",
            turn_id="turn-a",
            source_chunk_ids=("chunk-1",),
            source_chunk_indices=(1,),
            mount_status="anchored",
            anchor_kind="lexical",
            envelope_index=0,
            source_hook_ids=("hook-0",),
            match_confidence=0.98,
            cross_chunk_lock_ids=tuple(),
        ),
        AnchorMountItem(
            unit_id="unit-1",
            unit_index=1,
            token_text="world",
            display_text="world",
            normalized_text="world",
            speaker_id="speaker-b",
            turn_id="turn-b",
            source_chunk_ids=("chunk-2",),
            source_chunk_indices=(2,),
            mount_status="anchored",
            anchor_kind="lexical",
            envelope_index=1,
            source_hook_ids=("hook-1",),
            match_confidence=0.97,
            cross_chunk_lock_ids=tuple(),
        ),
    )


def _build_envelopes() -> tuple[TemporalEnvelope, ...]:
    return (
        TemporalEnvelope(
            unit_id="unit-0",
            envelope_kind="anchored",
            left_bound=0.0,
            right_bound=0.5,
            preferred_start=0.0,
            preferred_end=0.4,
            provisional_start=0.0,
            provisional_end=0.4,
            confidence=0.95,
            source_hook_ids=("hook-0",),
            source_chunk_ids=("chunk-1",),
            source_chunk_indices=(1,),
            cross_chunk_lock=False,
        ),
        TemporalEnvelope(
            unit_id="unit-1",
            envelope_kind="anchored",
            left_bound=0.5,
            right_bound=1.0,
            preferred_start=0.6,
            preferred_end=1.0,
            provisional_start=0.6,
            provisional_end=1.0,
            confidence=0.95,
            source_hook_ids=("hook-1",),
            source_chunk_ids=("chunk-2",),
            source_chunk_indices=(2,),
            cross_chunk_lock=False,
        ),
    )


def test_boundary_evidence_builder_only_emits_alignment_specific_reasons() -> None:
    evidences = BoundaryEvidenceBuilder().build(
        items=_build_items(),
        envelopes=_build_envelopes(),
        punctuation_facts=(
            PunctuationFact(
                fact_id="fact-0",
                left_token_index=0,
                right_token_index=1,
                attach_mode="after",
                normalized_text="!",
                punct_class="sentence_end",
                confidence=0.99,
                source="slow",
                group_id=None,
                boundary_weight=1.0,
                render_default=True,
            ),
        ),
        cross_chunk_locks=tuple(),
    )

    assert evidences
    assert {item.reason for item in evidences} <= {"lexical_boundary", "anchor_block_close"}


def test_boundary_evidence_builder_does_not_emit_punctuation_gap_or_speaker_reasons() -> None:
    evidences = BoundaryEvidenceBuilder().build(
        items=_build_items(),
        envelopes=_build_envelopes(),
        punctuation_facts=(
            PunctuationFact(
                fact_id="fact-0",
                left_token_index=0,
                right_token_index=1,
                attach_mode="after",
                normalized_text="!",
                punct_class="sentence_end",
                confidence=0.99,
                source="slow",
                group_id=None,
                boundary_weight=1.0,
                render_default=True,
            ),
        ),
        cross_chunk_locks=(
            CrossChunkLock(
                lock_id="lock-0",
                unit_ids=("unit-0", "unit-1"),
                hook_ids=("hook-0", "hook-1"),
                reason="test",
                source_chunk_ids=("chunk-1", "chunk-2"),
                source_chunk_indices=(1, 2),
            ),
        ),
    )

    reasons = {item.reason for item in evidences}
    assert "punctuation_sentence_end" not in reasons
    assert "punctuation_soft" not in reasons
    assert "gap_pause" not in reasons
    assert "blank_valley" not in reasons
    assert "speaker_change" not in reasons


def test_boundary_evidence_builder_ignores_per_token_hook_churn_inside_same_block() -> None:
    items = (
        AnchorMountItem(
            unit_id="unit-0",
            unit_index=0,
            token_text="killinging",
            display_text="killinging",
            normalized_text="killinging",
            speaker_id="speaker-a",
            turn_id="turn-a",
            source_chunk_ids=("chunk-1",),
            source_chunk_indices=(1,),
            mount_status="anchored",
            anchor_kind="lexical",
            envelope_index=0,
            source_hook_ids=("hook-30",),
            match_confidence=1.0,
            cross_chunk_lock_ids=tuple(),
            alignment_block_id="block-a",
        ),
        AnchorMountItem(
            unit_id="unit-1",
            unit_index=1,
            token_text="five",
            display_text="five",
            normalized_text="five",
            speaker_id="speaker-a",
            turn_id="turn-a",
            source_chunk_ids=("chunk-1",),
            source_chunk_indices=(1,),
            mount_status="anchored",
            anchor_kind="lexical",
            envelope_index=1,
            source_hook_ids=("hook-31",),
            match_confidence=1.0,
            cross_chunk_lock_ids=tuple(),
            alignment_block_id="block-a",
        ),
    )
    envelopes = (
        TemporalEnvelope(
            unit_id="unit-0",
            envelope_kind="anchored",
            left_bound=18.2,
            right_bound=18.68,
            preferred_start=18.2,
            preferred_end=18.68,
            provisional_start=18.2,
            provisional_end=18.68,
            confidence=1.0,
            source_hook_ids=("hook-30",),
            source_chunk_ids=("chunk-1",),
            source_chunk_indices=(1,),
            cross_chunk_lock=False,
        ),
        TemporalEnvelope(
            unit_id="unit-1",
            envelope_kind="anchored",
            left_bound=18.74,
            right_bound=19.52,
            preferred_start=18.74,
            preferred_end=19.52,
            provisional_start=18.74,
            provisional_end=19.52,
            confidence=1.0,
            source_hook_ids=("hook-31",),
            source_chunk_ids=("chunk-1",),
            source_chunk_indices=(1,),
            cross_chunk_lock=False,
        ),
    )

    evidences = BoundaryEvidenceBuilder().build(
        items=items,
        envelopes=envelopes,
        punctuation_facts=tuple(),
        cross_chunk_locks=tuple(),
    )

    assert {item.reason for item in evidences} == set()


def test_boundary_evidence_builder_skips_lexical_boundary_when_adjacent_provenance_overlaps() -> None:
    items = (
        AnchorMountItem(
            unit_id="unit-0",
            unit_index=0,
            token_text="beneficial",
            display_text="beneficial",
            normalized_text="beneficial",
            speaker_id="speaker-a",
            turn_id="turn-a",
            source_chunk_ids=("chunk-10", "chunk-11"),
            source_chunk_indices=(10, 11),
            mount_status="anchored",
            anchor_kind="lexical",
            envelope_index=0,
            source_hook_ids=("hook-36",),
            match_confidence=1.0,
            cross_chunk_lock_ids=tuple(),
            alignment_block_id="block-a",
        ),
        AnchorMountItem(
            unit_id="unit-1",
            unit_index=1,
            token_text="for",
            display_text="for",
            normalized_text="for",
            speaker_id="speaker-a",
            turn_id="turn-a",
            source_chunk_ids=("chunk-11", "chunk-12"),
            source_chunk_indices=(11, 12),
            mount_status="anchored",
            anchor_kind="lexical",
            envelope_index=1,
            source_hook_ids=("hook-37",),
            match_confidence=1.0,
            cross_chunk_lock_ids=tuple(),
            alignment_block_id="block-a",
        ),
    )
    envelopes = (
        TemporalEnvelope(
            unit_id="unit-0",
            envelope_kind="anchored",
            left_bound=21.08,
            right_bound=21.62,
            preferred_start=21.08,
            preferred_end=21.62,
            provisional_start=21.08,
            provisional_end=21.62,
            confidence=1.0,
            source_hook_ids=("hook-36",),
            source_chunk_ids=("chunk-10", "chunk-11"),
            source_chunk_indices=(10, 11),
            cross_chunk_lock=False,
        ),
        TemporalEnvelope(
            unit_id="unit-1",
            envelope_kind="anchored",
            left_bound=21.70,
            right_bound=21.82,
            preferred_start=21.70,
            preferred_end=21.82,
            provisional_start=21.70,
            provisional_end=21.82,
            confidence=1.0,
            source_hook_ids=("hook-37",),
            source_chunk_ids=("chunk-11", "chunk-12"),
            source_chunk_indices=(11, 12),
            cross_chunk_lock=False,
        ),
    )

    evidences = BoundaryEvidenceBuilder().build(
        items=items,
        envelopes=envelopes,
        punctuation_facts=tuple(),
        cross_chunk_locks=tuple(),
    )

    assert {item.reason for item in evidences} == set()
