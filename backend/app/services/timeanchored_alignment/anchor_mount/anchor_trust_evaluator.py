"""候选可信度评估。"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.timeanchored_alignment.anchor_mount.ambiguity_cluster_builder import (
    AmbiguityCluster,
)
from app.services.timeanchored_alignment.anchor_mount.contracts import AnchorCandidate

_KIND_PRIOR = {
    "exact": 1.00,
    "normalized": 0.90,
    "sequence": 0.82,
    "merge": 0.74,
    "soft_pronunciation": 0.58,
}


@dataclass(frozen=True)
class AnchorTrustReport:
    candidate_id: str
    trust_score: float
    trust_tier: str
    reject_reason: str | None
    diagnostics: dict[str, float | str]


class AnchorTrustEvaluator:
    """把“是否值得进主线”与“同簇里谁更优”分离开来。"""

    def evaluate_candidate(
        self,
        candidate: AnchorCandidate,
        *,
        cluster: AmbiguityCluster | None = None,
        competing_candidates: tuple[AnchorCandidate, ...] = tuple(),
    ) -> AnchorTrustReport:
        reject_reason = self._hard_reject(
            candidate=candidate,
            cluster=cluster,
            competing_candidates=competing_candidates,
        )
        if reject_reason is not None:
            return AnchorTrustReport(
                candidate_id=candidate.candidate_id,
                trust_score=0.0,
                trust_tier="rejected",
                reject_reason=reject_reason,
                diagnostics={"hard_reject": reject_reason},
            )

        kind_prior = _KIND_PRIOR.get(candidate.anchor_kind, 0.50)
        uniqueness = self._uniqueness_margin(candidate=candidate, competing_candidates=competing_candidates)
        information_value = self._information_value(candidate=candidate)
        boundary_consistency = self._boundary_consistency(cluster=cluster)
        provenance_consistency = 1.0 if candidate.is_hard else 0.55
        temporal_plausibility = self._temporal_plausibility(candidate=candidate)
        ambiguity_penalty = self._ambiguity_penalty(cluster=cluster, competing_candidates=competing_candidates)
        low_information_penalty = 0.25 if information_value < 0.35 else 0.0

        score = 0.0
        score += 0.30 * kind_prior
        score += 0.20 * uniqueness
        score += 0.15 * information_value
        score += 0.10 * boundary_consistency
        score += 0.10 * provenance_consistency
        score += 0.05 * temporal_plausibility
        score -= ambiguity_penalty
        score -= low_information_penalty
        normalized_score = max(0.0, min(1.0, score))

        if normalized_score >= 0.80 and uniqueness >= 0.70:
            trust_tier = "primary"
        elif normalized_score >= 0.62:
            trust_tier = "secondary"
        elif normalized_score >= 0.45:
            trust_tier = "boundary_only"
        else:
            trust_tier = "rejected"

        reject_reason = None if trust_tier != "rejected" else "trust_below_threshold"
        return AnchorTrustReport(
            candidate_id=candidate.candidate_id,
            trust_score=normalized_score,
            trust_tier=trust_tier,
            reject_reason=reject_reason,
            diagnostics={
                "kind_prior": round(kind_prior, 4),
                "uniqueness_margin": round(uniqueness, 4),
                "information_value": round(information_value, 4),
                "boundary_consistency": round(boundary_consistency, 4),
                "temporal_plausibility": round(temporal_plausibility, 4),
                "ambiguity_penalty": round(ambiguity_penalty, 4),
                "low_information_penalty": round(low_information_penalty, 4),
            },
        )

    @staticmethod
    def _hard_reject(
        *,
        candidate: AnchorCandidate,
        cluster: AmbiguityCluster | None,
        competing_candidates: tuple[AnchorCandidate, ...],
    ) -> str | None:
        if len(candidate.unit_indices) > 1 and any(
            right != left + 1
            for left, right in zip(candidate.unit_indices, candidate.unit_indices[1:], strict=False)
        ):
            return "non_contiguous_merge"
        if len(candidate.unit_indices) > 2:
            return "merge_width_exceeded"
        if cluster is not None and "boundary_cross_risk" in cluster.conflict_basis and len(candidate.unit_indices) > 1:
            return "hard_boundary_cross"
        if len(competing_candidates) > 0 and len(set(candidate.unit_indices)) == 1 and len(set(candidate.hook_indices)) == 1:
            if len(candidate.anchor_kind) <= 1:
                return "low_information_duplicate"
        return None

    @staticmethod
    def _uniqueness_margin(
        *,
        candidate: AnchorCandidate,
        competing_candidates: tuple[AnchorCandidate, ...],
    ) -> float:
        if not competing_candidates:
            return 1.0
        same_unit = sum(1 for item in competing_candidates if item.unit_indices == candidate.unit_indices)
        same_hook = sum(1 for item in competing_candidates if item.hook_indices == candidate.hook_indices)
        penalties = 0.0
        if same_unit:
            penalties += 0.35
        if same_hook:
            penalties += 0.35
        if any(item.anchor_kind == candidate.anchor_kind for item in competing_candidates):
            penalties += 0.10
        return max(0.0, 1.0 - penalties)

    @staticmethod
    def _information_value(*, candidate: AnchorCandidate) -> float:
        span_width = max(len(candidate.unit_indices), len(candidate.hook_indices))
        if span_width >= 2:
            return 0.95
        if candidate.anchor_kind in {"exact", "normalized"}:
            return 0.72
        if candidate.anchor_kind == "soft_pronunciation":
            return 0.40
        return 0.55

    @staticmethod
    def _boundary_consistency(*, cluster: AmbiguityCluster | None) -> float:
        if cluster is None:
            return 1.0
        if "boundary_cross_risk" in cluster.conflict_basis:
            return 0.15
        return 0.85

    @staticmethod
    def _temporal_plausibility(*, candidate: AnchorCandidate) -> float:
        unit_span = len(candidate.unit_indices)
        hook_span = len(candidate.hook_indices)
        delta = abs(unit_span - hook_span)
        if delta == 0:
            return 1.0
        if delta == 1:
            return 0.75
        return 0.35

    @staticmethod
    def _ambiguity_penalty(
        *,
        cluster: AmbiguityCluster | None,
        competing_candidates: tuple[AnchorCandidate, ...],
    ) -> float:
        penalty = 0.0
        if cluster is not None:
            penalty += 0.08 * float(len(cluster.candidate_ids) - 1)
            if "low_information_risk" in cluster.conflict_basis:
                penalty += 0.08
            if "merge_width_competition" in cluster.conflict_basis:
                penalty += 0.06
        if len(competing_candidates) >= 2:
            penalty += 0.05
        return penalty
