"""AnchorMount 核心主线编排。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.services.timeanchored_alignment.anchor_mount.ambiguity_cluster_builder import (
    AmbiguityCluster,
    AmbiguityClusterBuilder,
)
from app.services.timeanchored_alignment.anchor_mount.anchor_trust_evaluator import (
    AnchorTrustEvaluator,
)
from app.services.timeanchored_alignment.anchor_mount.boundary_evidence_builder import (
    BoundaryEvidenceBuilder,
)
from app.services.timeanchored_alignment.anchor_mount.chain_solver import ChainSolver
from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorCandidate,
    AnchorMountInputView,
)
from app.services.timeanchored_alignment.anchor_mount.cross_chunk_lock_builder import (
    CrossChunkLockBuilder,
)
from app.services.timeanchored_alignment.anchor_mount.densification_promoter import (
    DensificationPromoter,
)
from app.services.timeanchored_alignment.anchor_mount.gap_rescue_aligner import (
    GapRescueAligner,
)
from app.services.timeanchored_alignment.anchor_mount.hook_claim_resolver import (
    HookClaimResolver,
)
from app.services.timeanchored_alignment.anchor_mount.local_extension import LocalExtension
from app.services.timeanchored_alignment.anchor_mount.low_complexity_mask import (
    LowComplexityMask,
)
from app.services.timeanchored_alignment.anchor_mount.punctuation_fact_mapper import (
    PunctuationFactMapper,
)
from app.services.timeanchored_alignment.anchor_mount.seed_discovery import SeedDiscovery
from app.services.timeanchored_alignment.anchor_mount.temporal_envelope_builder import (
    TemporalEnvelopeBuilder,
)
from app.services.timeanchored_alignment.anchor_mount.window_alignment_state import (
    WindowAlignmentState,
)


class AnchorMountCorePipeline:
    """当前唯一的 AnchorMount 核心算法主线。"""

    def __init__(
        self,
        *,
        punctuation_fact_mapper: PunctuationFactMapper | None = None,
        seed_discovery: SeedDiscovery | None = None,
        low_complexity_mask: LowComplexityMask | None = None,
        ambiguity_cluster_builder: AmbiguityClusterBuilder | None = None,
        anchor_trust_evaluator: AnchorTrustEvaluator | None = None,
        local_extension: LocalExtension | None = None,
        chain_solver: ChainSolver | None = None,
        gap_rescue_aligner: GapRescueAligner | None = None,
        densification_promoter: DensificationPromoter | None = None,
        temporal_envelope_builder: TemporalEnvelopeBuilder | None = None,
        hook_claim_resolver: HookClaimResolver | None = None,
        cross_chunk_lock_builder: CrossChunkLockBuilder | None = None,
        boundary_evidence_builder: BoundaryEvidenceBuilder | None = None,
    ) -> None:
        self._punctuation_fact_mapper = punctuation_fact_mapper or PunctuationFactMapper()
        self._seed_discovery = seed_discovery or SeedDiscovery()
        self._low_complexity_mask = low_complexity_mask or LowComplexityMask()
        self._ambiguity_cluster_builder = ambiguity_cluster_builder or AmbiguityClusterBuilder()
        self._anchor_trust_evaluator = anchor_trust_evaluator or AnchorTrustEvaluator()
        self._local_extension = local_extension or LocalExtension()
        self._chain_solver = chain_solver or ChainSolver()
        self._gap_rescue_aligner = gap_rescue_aligner or GapRescueAligner()
        self._densification_promoter = densification_promoter or DensificationPromoter(
            trust_evaluator=self._anchor_trust_evaluator,
        )
        self._temporal_envelope_builder = temporal_envelope_builder or TemporalEnvelopeBuilder()
        self._hook_claim_resolver = hook_claim_resolver or HookClaimResolver()
        self._cross_chunk_lock_builder = cross_chunk_lock_builder or CrossChunkLockBuilder()
        self._boundary_evidence_builder = boundary_evidence_builder or BoundaryEvidenceBuilder()

    def run(self, input_view: AnchorMountInputView) -> WindowAlignmentState:
        state = WindowAlignmentState.bootstrap(input_view)

        punctuation_facts, punctuation_pair_states, punctuation_diagnostics = self._punctuation_fact_mapper.map(
            window_text=input_view.window_text,
            token_units=input_view.token_units,
            punctuation_evidences=input_view.punctuation_evidences,
        )
        raw_candidates = self._seed_discovery.discover(input_view=input_view)
        masked_candidates = self._low_complexity_mask.apply(
            candidates=raw_candidates,
            input_view=input_view,
        )
        ambiguity_clusters = self._ambiguity_cluster_builder.build(
            input_view=input_view,
            candidates=masked_candidates,
        )
        resolved_candidates = self._apply_trust_and_cluster_resolution(
            candidates=masked_candidates,
            ambiguity_clusters=ambiguity_clusters,
        )
        primary_candidates = tuple(
            candidate for candidate in resolved_candidates if candidate.trust_tier == "primary"
        )
        block_candidates = tuple(
            candidate
            for candidate in resolved_candidates
            if candidate.trust_tier in {"primary", "secondary", "boundary_only"}
        )
        blocks = self._local_extension.build(
            input_view=input_view,
            candidates=block_candidates,
        )
        solve_result = self._chain_solver.solve(input_view=input_view, blocks=blocks)
        diagnostics = self._build_solver_diagnostics(
            state=state,
            raw_candidates=raw_candidates,
            masked_candidates=masked_candidates,
            ambiguity_clusters=ambiguity_clusters,
            primary_candidates=primary_candidates,
            block_candidates=block_candidates,
            blocks=blocks,
            solve_result=solve_result,
        )
        state = state.with_updates(
            raw_seed_candidates=tuple(resolved_candidates),
            ambiguity_clusters=ambiguity_clusters,
            primary_candidates=primary_candidates,
            local_blocks=blocks,
            main_chain=solve_result,
            punctuation_facts=punctuation_facts,
            punctuation_pair_states=punctuation_pair_states,
            punctuation_diagnostics=punctuation_diagnostics,
            diagnostics=diagnostics,
        )
        state = self._gap_rescue_aligner.run(state=state)
        state = self._densification_promoter.run(state=state)
        if state.promoted_secondary_anchors:
            promoted_candidates = tuple(
                self._promoted_anchor_to_candidate(promoted_anchor)
                for promoted_anchor in state.promoted_secondary_anchors
            )
            promoted_block_candidates = block_candidates + promoted_candidates
            blocks = self._local_extension.build(
                input_view=input_view,
                candidates=promoted_block_candidates,
            )
            solve_result = self._chain_solver.solve(input_view=input_view, blocks=blocks)
            diagnostics = self._build_solver_diagnostics(
                state=state,
                raw_candidates=raw_candidates,
                masked_candidates=masked_candidates,
                ambiguity_clusters=ambiguity_clusters,
                primary_candidates=primary_candidates,
                block_candidates=promoted_block_candidates,
                blocks=blocks,
                solve_result=solve_result,
            )
            state = state.with_updates(
                local_blocks=blocks,
                main_chain=solve_result,
                diagnostics=diagnostics,
            )
            state = self._gap_rescue_aligner.run(
                state=state,
                attempt_rescue=False,
                increment_round=False,
            )
        items, envelopes = self._temporal_envelope_builder.build(
            input_view=input_view,
            solve_result=state.main_chain,
        )
        hook_claims = self._hook_claim_resolver.resolve(
            input_view=input_view,
            envelopes=envelopes,
        )
        cross_chunk_locks = self._cross_chunk_lock_builder.build(
            items=items,
            envelopes=envelopes,
        )
        boundary_evidences = self._boundary_evidence_builder.build(
            items=items,
            envelopes=envelopes,
            punctuation_facts=punctuation_facts,
            cross_chunk_locks=cross_chunk_locks,
        )
        diagnostics = dict(state.diagnostics)
        diagnostics["boundary_evidence_count"] = len(boundary_evidences)
        return state.with_updates(
            items=items,
            envelopes=envelopes,
            hook_claims=hook_claims,
            cross_chunk_locks=cross_chunk_locks,
            boundary_evidences=boundary_evidences,
            diagnostics=diagnostics,
        )

    def _apply_trust_and_cluster_resolution(
        self,
        *,
        candidates: tuple[AnchorCandidate, ...],
        ambiguity_clusters: tuple[AmbiguityCluster, ...],
    ) -> tuple[AnchorCandidate, ...]:
        cluster_by_candidate_id = {
            candidate_id: cluster
            for cluster in ambiguity_clusters
            for candidate_id in cluster.candidate_ids
        }
        cluster_reports: dict[str, dict[str, Any]] = {}
        updated_candidates: list[AnchorCandidate] = []
        for candidate in candidates:
            cluster = cluster_by_candidate_id.get(candidate.candidate_id)
            competing_candidates = self._collect_competing_candidates(
                candidate=candidate,
                candidates=candidates,
                cluster=cluster,
            )
            report = self._anchor_trust_evaluator.evaluate_candidate(
                candidate,
                cluster=cluster,
                competing_candidates=competing_candidates,
            )
            cluster_id = cluster.cluster_id if cluster is not None else None
            updated_candidates.append(
                replace(
                    candidate,
                    ambiguity_cluster_id=cluster_id,
                    trust_tier=report.trust_tier,
                    reject_reason=report.reject_reason,
                )
            )
            if cluster_id is not None:
                cluster_reports.setdefault(cluster_id, {})[candidate.candidate_id] = {
                    "trust_score": report.trust_score,
                    "trust_tier": report.trust_tier,
                }

        if not ambiguity_clusters:
            return tuple(updated_candidates)

        winner_ids: set[str] = set()
        for cluster in ambiguity_clusters:
            eligible = [
                candidate
                for candidate in updated_candidates
                if candidate.candidate_id in cluster.candidate_ids
                and candidate.trust_tier != "rejected"
            ]
            if not eligible:
                continue
            winner = max(
                eligible,
                key=lambda item: (
                    float(cluster_reports[cluster.cluster_id][item.candidate_id]["trust_score"]),
                    float(item.score),
                    -len(item.unit_indices),
                ),
            )
            winner_ids.add(winner.candidate_id)

        normalized_candidates: list[AnchorCandidate] = []
        for candidate in updated_candidates:
            if candidate.ambiguity_cluster_id is None:
                normalized_candidates.append(candidate)
                continue
            if candidate.candidate_id in winner_ids:
                normalized_candidates.append(candidate)
                continue
            if candidate.trust_tier == "primary":
                normalized_candidates.append(replace(candidate, trust_tier="secondary"))
                continue
            normalized_candidates.append(candidate)
        return tuple(normalized_candidates)

    @staticmethod
    def _collect_competing_candidates(
        *,
        candidate: AnchorCandidate,
        candidates: tuple[AnchorCandidate, ...],
        cluster: AmbiguityCluster | None,
    ) -> tuple[AnchorCandidate, ...]:
        if cluster is None:
            return tuple(
                other
                for other in candidates
                if other.candidate_id != candidate.candidate_id
                and (
                    other.unit_indices == candidate.unit_indices
                    or other.hook_indices == candidate.hook_indices
                )
            )
        return tuple(
            other
            for other in candidates
            if other.candidate_id != candidate.candidate_id
            and other.candidate_id in cluster.candidate_ids
        )

    @staticmethod
    def _build_solver_diagnostics(
        *,
        state: WindowAlignmentState,
        raw_candidates: tuple[AnchorCandidate, ...],
        masked_candidates: tuple[AnchorCandidate, ...],
        ambiguity_clusters: tuple[AmbiguityCluster, ...],
        primary_candidates: tuple[AnchorCandidate, ...],
        block_candidates: tuple[AnchorCandidate, ...],
        blocks,
        solve_result,
    ) -> dict[str, Any]:
        diagnostics = dict(state.diagnostics)
        diagnostics.update(
            {
                "seed_candidate_count": len(raw_candidates),
                "masked_candidate_count": len(masked_candidates),
                "ambiguity_cluster_count": len(ambiguity_clusters),
                "primary_candidate_count": len(primary_candidates),
                "block_candidate_count": len(block_candidates),
                "block_count": len(blocks),
                "compatibility_edge_count": int(
                    getattr(solve_result, "compatibility_edge_count", 0) or 0
                ),
                "solver_elapsed_ms": float(
                    getattr(solve_result, "solver_elapsed_ms", 0.0) or 0.0
                ),
                "avg_in_degree": float(getattr(solve_result, "avg_in_degree", 0.0) or 0.0),
                "max_in_degree": int(getattr(solve_result, "max_in_degree", 0) or 0),
            }
        )
        return diagnostics

    @staticmethod
    def _promoted_anchor_to_candidate(promoted_anchor) -> AnchorCandidate:
        return AnchorCandidate(
            candidate_id=str(promoted_anchor.candidate_id),
            unit_indices=tuple(int(item) for item in promoted_anchor.unit_indices),
            hook_indices=tuple(int(item) for item in promoted_anchor.hook_indices),
            anchor_kind=str(promoted_anchor.anchor_kind),
            score=min(max(float(promoted_anchor.trust_score), 0.0), 1.0),
            is_hard=False,
            trust_tier=str(promoted_anchor.trust_tier),
            reject_reason=promoted_anchor.reject_reason,
        )
