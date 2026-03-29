from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorMountItem,
    AnchorMountResult,
    TemporalEnvelope,
)
from app.services.timeanchored_alignment.anchor_mount.timeline_validity_validator import (
    TimelineValidityValidator,
)


def _build_result(
    *,
    envelopes: tuple[TemporalEnvelope, ...],
    metrics: dict[str, float | int],
) -> AnchorMountResult:
    items = tuple(
        AnchorMountItem(
            unit_id=f"unit-{index}",
            unit_index=index,
            token_text=f"token-{index}",
            display_text=f"token-{index}",
            normalized_text=f"token-{index}",
            speaker_id=None,
            turn_id=None,
            source_chunk_ids=("chunk-1",),
            source_chunk_indices=(1,),
            mount_status=envelope.envelope_kind,
            anchor_kind="exact" if envelope.envelope_kind == "anchored" else "none",
            envelope_index=index,
            source_hook_ids=envelope.source_hook_ids,
            match_confidence=envelope.confidence,
            cross_chunk_lock_ids=tuple(),
        )
        for index, envelope in enumerate(envelopes)
    )
    return AnchorMountResult(
        items=items,
        envelopes=envelopes,
        punctuation_facts=tuple(),
        punctuation_pair_states=tuple(),
        hook_claims=tuple(),
        cross_chunk_locks=tuple(),
        boundary_evidences=tuple(),
        metrics=metrics,
    )


def _build_envelope(
    *,
    unit_id: str,
    envelope_kind: str,
    start: float,
    end: float,
    gap_state: str = "resolved",
) -> TemporalEnvelope:
    return TemporalEnvelope(
        unit_id=unit_id,
        envelope_kind=envelope_kind,
        left_bound=start,
        right_bound=end,
        preferred_start=start if envelope_kind == "anchored" else None,
        preferred_end=end if envelope_kind == "anchored" else None,
        provisional_start=start,
        provisional_end=end,
        confidence=0.9 if envelope_kind == "anchored" else 0.4,
        source_hook_ids=("hook-1",) if envelope_kind == "anchored" else tuple(),
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        cross_chunk_lock=False,
        gap_state=gap_state,
    )


def test_timeline_validity_validator_marks_clean_monotonic_result_valid() -> None:
    result = _build_result(
        envelopes=(
            _build_envelope(unit_id="unit-0", envelope_kind="anchored", start=0.0, end=0.3),
            _build_envelope(unit_id="unit-1", envelope_kind="anchored", start=0.3, end=0.6),
        ),
        metrics={
            "coverage_ratio": 1.0,
            "unresolved_count": 0,
            "inferred_count": 0,
            "largest_unresolved_span": 0,
            "hook_claim_conflict_count": 0,
        },
    )

    report = TimelineValidityValidator().validate(result=result)

    assert report.state == "valid"
    assert report.reasons == tuple()


def test_timeline_validity_validator_marks_residual_gap_result_repairable() -> None:
    result = _build_result(
        envelopes=(
            _build_envelope(unit_id="unit-0", envelope_kind="anchored", start=0.0, end=0.3),
            _build_envelope(
                unit_id="unit-1",
                envelope_kind="inferred",
                start=0.3,
                end=0.5,
                gap_state="residual",
            ),
        ),
        metrics={
            "coverage_ratio": 0.8,
            "unresolved_count": 0,
            "inferred_count": 1,
            "largest_unresolved_span": 1,
            "hook_claim_conflict_count": 0,
        },
    )

    report = TimelineValidityValidator().validate(result=result)

    assert report.state == "repairable"
    assert "inferred_gap_present" in report.reasons


def test_timeline_validity_validator_marks_monotonic_break_as_quarantined() -> None:
    result = _build_result(
        envelopes=(
            _build_envelope(unit_id="unit-0", envelope_kind="anchored", start=0.0, end=0.4),
            _build_envelope(unit_id="unit-1", envelope_kind="anchored", start=0.2, end=0.5),
        ),
        metrics={
            "coverage_ratio": 0.9,
            "unresolved_count": 0,
            "inferred_count": 0,
            "largest_unresolved_span": 0,
            "hook_claim_conflict_count": 0,
        },
    )

    report = TimelineValidityValidator().validate(result=result)

    assert report.state == "quarantined"
    assert "monotonic_violation" in report.reasons
