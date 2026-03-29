from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.ambiguity_cluster_builder import (
    AmbiguityCluster,
)
from app.services.timeanchored_alignment.anchor_mount.anchor_trust_evaluator import (
    AnchorTrustEvaluator,
)
from app.services.timeanchored_alignment.anchor_mount.contracts import AnchorCandidate


def test_anchor_trust_evaluator_promotes_unique_exact_candidate_to_primary() -> None:
    candidate = AnchorCandidate(
        candidate_id="exact:0:0",
        unit_indices=(0,),
        hook_indices=(0,),
        anchor_kind="exact",
        score=1.0,
        is_hard=True,
    )

    report = AnchorTrustEvaluator().evaluate_candidate(candidate)

    assert report.trust_tier == "primary"
    assert report.reject_reason is None
    assert report.trust_score >= 0.8


def test_anchor_trust_evaluator_rejects_non_contiguous_merge() -> None:
    candidate = AnchorCandidate(
        candidate_id="merge:0-2:1",
        unit_indices=(0, 2),
        hook_indices=(1,),
        anchor_kind="merge",
        score=0.82,
        is_hard=True,
    )

    report = AnchorTrustEvaluator().evaluate_candidate(candidate)

    assert report.trust_tier == "rejected"
    assert report.reject_reason == "non_contiguous_merge"


def test_anchor_trust_evaluator_penalizes_boundary_cross_risk_cluster() -> None:
    candidate = AnchorCandidate(
        candidate_id="merge:0-1:0",
        unit_indices=(0, 1),
        hook_indices=(0,),
        anchor_kind="merge",
        score=0.82,
        is_hard=True,
    )
    cluster = AmbiguityCluster(
        cluster_id="cluster-0",
        hook_span=(0,),
        unit_span=(0, 1),
        candidate_ids=("merge:0-1:0", "merge:0-1:1"),
        candidate_unit_spans=((0, 1), (0, 1)),
        candidate_hook_spans=((0,), (1,)),
        cluster_tags=("span_width_conflict",),
        conflict_basis=("boundary_cross_risk",),
    )

    report = AnchorTrustEvaluator().evaluate_candidate(candidate, cluster=cluster)

    assert report.trust_tier == "rejected"
    assert report.reject_reason == "hard_boundary_cross"
