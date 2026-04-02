"""AnchorMountAlignment 层主入口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.language_policy import build_language_policy_snapshot
from app.services.timeanchored_alignment.anchor_mount.boundary_evidence_builder import (
    BoundaryEvidenceBuilder,
)
from app.services.timeanchored_alignment.anchor_mount.chain_solver import ChainSolveResult, ChainSolver
from app.services.timeanchored_alignment.anchor_mount.core_pipeline import (
    AnchorMountCorePipeline,
)
from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorMountResult,
    DecisionIngressPackage,
)
from app.services.timeanchored_alignment.anchor_mount.cross_chunk_lock_builder import (
    CrossChunkLockBuilder,
)
from app.services.timeanchored_alignment.anchor_mount.decision_ingress_assembler import (
    DecisionIngressAssembler,
)
from app.services.timeanchored_alignment.anchor_mount.densification_promoter import (
    DensificationPromoter,
)
from app.services.timeanchored_alignment.anchor_mount.fallback_gate import FallbackGate
from app.services.timeanchored_alignment.anchor_mount.gap_rescue_aligner import (
    GapRescueAligner,
)
from app.services.timeanchored_alignment.anchor_mount.hook_claim_resolver import (
    HookClaimResolver,
)
from app.services.timeanchored_alignment.anchor_mount.ingress_validator import (
    IngressValidator,
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
from app.services.timeanchored_alignment.anchor_mount.timeline_validity_validator import (
    TimelineValidityValidator,
)
from app.services.timeanchored_alignment.anchor_mount.window_alignment_state import (
    WindowAlignmentState,
)
from app.services.timeanchored_alignment.contracts import LayerReport, PipelineReport
from app.services.timeanchored_alignment.preparation.contracts import AlignmentPreparationPackage


@dataclass(frozen=True)
class AnchorMountStageResult:
    """Chunk3 对外阶段结果。"""

    decision_ingress: DecisionIngressPackage
    anchor_mount_result: AnchorMountResult
    pipeline_report: PipelineReport
    window_recovery_plan: Any | None = None


class AnchorMountAlignmentService:
    """完整执行 AnchorMountAlignment。"""

    def __init__(
        self,
        *,
        ingress_validator: IngressValidator | None = None,
        punctuation_fact_mapper: PunctuationFactMapper | None = None,
        seed_discovery: SeedDiscovery | None = None,
        low_complexity_mask: LowComplexityMask | None = None,
        local_extension: LocalExtension | None = None,
        chain_solver: ChainSolver | None = None,
        gap_rescue_aligner: GapRescueAligner | None = None,
        densification_promoter: DensificationPromoter | None = None,
        temporal_envelope_builder: TemporalEnvelopeBuilder | None = None,
        hook_claim_resolver: HookClaimResolver | None = None,
        cross_chunk_lock_builder: CrossChunkLockBuilder | None = None,
        boundary_evidence_builder: BoundaryEvidenceBuilder | None = None,
        timeline_validity_validator: TimelineValidityValidator | None = None,
        fallback_gate: FallbackGate | None = None,
        decision_ingress_assembler: DecisionIngressAssembler | None = None,
    ) -> None:
        self._ingress_validator = ingress_validator or IngressValidator()
        self._core_pipeline = AnchorMountCorePipeline(
            punctuation_fact_mapper=punctuation_fact_mapper or PunctuationFactMapper(),
            seed_discovery=seed_discovery or SeedDiscovery(),
            low_complexity_mask=low_complexity_mask or LowComplexityMask(),
            local_extension=local_extension or LocalExtension(),
            chain_solver=chain_solver or ChainSolver(),
            gap_rescue_aligner=gap_rescue_aligner or GapRescueAligner(),
            densification_promoter=densification_promoter or DensificationPromoter(),
            temporal_envelope_builder=temporal_envelope_builder or TemporalEnvelopeBuilder(),
            hook_claim_resolver=hook_claim_resolver or HookClaimResolver(),
            cross_chunk_lock_builder=cross_chunk_lock_builder or CrossChunkLockBuilder(),
            boundary_evidence_builder=boundary_evidence_builder or BoundaryEvidenceBuilder(),
        )
        self._timeline_validity_validator = (
            timeline_validity_validator or TimelineValidityValidator()
        )
        self._fallback_gate = fallback_gate or FallbackGate(
            timeline_validity_validator=self._timeline_validity_validator,
        )
        self._decision_ingress_assembler = decision_ingress_assembler or DecisionIngressAssembler()

    def align(
        self,
        *,
        preparation: AlignmentPreparationPackage,
        language: str,
        policy_snapshot: Any | None = None,
    ) -> AnchorMountStageResult:
        resolved_policy = policy_snapshot
        if resolved_policy is None:
            try:
                resolved_policy = build_language_policy_snapshot(language_hint=language)
            except Exception:
                resolved_policy = None
        input_view = self._ingress_validator.validate(
            preparation=preparation,
            language=language,
            policy_snapshot=resolved_policy,
        )
        state = self._core_pipeline.run(input_view)
        metrics = self._build_metrics(
            input_view=input_view,
            solve_result=state.main_chain,
            candidates=state.raw_seed_candidates,
            items=state.items,
            envelopes=state.envelopes,
            hook_claims=state.hook_claims,
            cross_chunk_locks=state.cross_chunk_locks,
            boundary_evidence_count=len(state.boundary_evidences),
            punctuation_fact_count=len(state.punctuation_facts),
            punctuation_diagnostics=state.punctuation_diagnostics,
        )
        metrics.update(
            {
                key: value
                for key, value in dict(state.diagnostics).items()
                if key
                in {
                    "seed_candidate_count",
                    "masked_candidate_count",
                    "ambiguity_cluster_count",
                    "primary_candidate_count",
                    "block_candidate_count",
                    "block_count",
                    "compatibility_edge_count",
                    "solver_elapsed_ms",
                    "avg_in_degree",
                    "max_in_degree",
                    "promoted_anchor_count",
                    "anchor_island_count",
                    "open_gap_count",
                    "gap_rescue_match_count",
                }
            }
        )
        provisional_result = self._build_result_from_state(
            state=state,
            metrics=metrics,
        )
        validity_report = self._timeline_validity_validator.validate(result=provisional_result)
        metrics = self._merge_validity_metrics(
            metrics=metrics,
            validity_report=validity_report,
        )
        should_fallback = self._fallback_gate.should_fallback(validity_report=validity_report)
        anchor_mount_result = AnchorMountResult(
            items=provisional_result.items,
            envelopes=provisional_result.envelopes,
            punctuation_facts=provisional_result.punctuation_facts,
            punctuation_pair_states=provisional_result.punctuation_pair_states,
            hook_claims=provisional_result.hook_claims,
            cross_chunk_locks=provisional_result.cross_chunk_locks,
            boundary_evidences=provisional_result.boundary_evidences,
            metrics=metrics,
            should_fallback=should_fallback,
            timeline_validity=validity_report.state,
            validity_reasons=validity_report.reasons,
        )
        decision_ingress = self._decision_ingress_assembler.build(
            input_view=input_view,
            result=anchor_mount_result,
        )
        route = "slow" if decision_ingress.anchored_token_units else "error"
        pipeline_report = PipelineReport(
            alignment_report=LayerReport(
                name="anchor_mount_alignment",
                status="ok" if decision_ingress.anchored_token_units else "error",
                metrics={
                    "route": route,
                    "unit_count": len(state.items),
                    "anchored_count": metrics["anchored_count"],
                    "reseed_count": metrics["reseed_count"],
                    "compatibility_edge_count": metrics.get("compatibility_edge_count", 0),
                    "solver_elapsed_ms": metrics.get("solver_elapsed_ms", 0.0),
                    "should_fallback": should_fallback,
                    "timeline_validity": validity_report.state,
                },
            ),
            segmentation_report=LayerReport(
                name="anchor_mount_boundary",
                status="ok",
                metrics={
                    "boundary_candidate_count": len(decision_ingress.boundary_evidences),
                    "punctuation_fact_count": len(state.punctuation_facts),
                },
            ),
            output_report=LayerReport(
                name="decision_ingress",
                status="ok",
                metrics={"token_count": len(decision_ingress.anchored_token_units)},
            ),
        )
        return AnchorMountStageResult(
            decision_ingress=decision_ingress,
            anchor_mount_result=anchor_mount_result,
            pipeline_report=pipeline_report,
        )

    def execute(
        self,
        *,
        preparation: AlignmentPreparationPackage,
        language: str,
        policy_snapshot: Any | None = None,
        **_: Any,
    ) -> AnchorMountStageResult:
        """与旧阶段服务保持调用形状兼容。"""

        return self.align(
            preparation=preparation,
            language=language,
            policy_snapshot=policy_snapshot,
        )

    @staticmethod
    def _build_result_from_state(
        *,
        state: WindowAlignmentState,
        metrics: dict[str, Any],
    ) -> AnchorMountResult:
        return AnchorMountResult(
            items=state.items,
            envelopes=state.envelopes,
            punctuation_facts=state.punctuation_facts,
            punctuation_pair_states=state.punctuation_pair_states,
            hook_claims=state.hook_claims,
            cross_chunk_locks=state.cross_chunk_locks,
            boundary_evidences=state.boundary_evidences,
            metrics=metrics,
            fuse_events=state.fuse_events,
        )

    @staticmethod
    def _merge_validity_metrics(
        *,
        metrics: dict[str, Any],
        validity_report: Any,
    ) -> dict[str, Any]:
        merged = dict(metrics)
        merged["timeline_validity"] = str(getattr(validity_report, "state", "repairable"))
        merged["timeline_validity_reasons"] = list(
            getattr(validity_report, "reasons", ()) or ()
        )
        merged["timeline_monotonic_violation_count"] = int(
            getattr(validity_report, "monotonic_violation_count", 0) or 0
        )
        merged["timeline_compressed_run_count"] = int(
            getattr(validity_report, "compressed_run_count", 0) or 0
        )
        merged["timeline_residual_gap_token_count"] = int(
            getattr(validity_report, "residual_gap_token_count", 0) or 0
        )
        merged["timeline_invalid_gap_ids"] = list(
            getattr(validity_report, "invalid_gap_ids", ()) or ()
        )
        return merged

    @staticmethod
    def _build_metrics(
        *,
        input_view: Any,
        solve_result: ChainSolveResult,
        candidates: tuple[Any, ...],
        items: tuple[Any, ...],
        envelopes: tuple[Any, ...],
        hook_claims: tuple[Any, ...],
        cross_chunk_locks: tuple[Any, ...],
        boundary_evidence_count: int,
        punctuation_fact_count: int,
        punctuation_diagnostics: Any,
    ) -> dict[str, Any]:
        unit_count = len(items)
        anchored_count = sum(
            1 for envelope in envelopes if envelope.envelope_kind in {"anchored", "merged"}
        )
        unresolved_count = sum(
            1 for envelope in envelopes if envelope.envelope_kind == "unresolved"
        )
        inferred_count = sum(
            1 for envelope in envelopes if envelope.envelope_kind == "inferred"
        )
        hard_anchor_count = sum(
            1 for block in solve_result.committed_blocks if block.block_kind == "anchored"
        )
        soft_anchor_count = sum(
            1 for kind in solve_result.unit_anchor_kinds if str(kind) == "soft_pronunciation"
        )
        total_hooks = len(getattr(input_view, "fast_hooks", ()) or ())
        claimed_hook_ids = {
            str(claim.hook_id)
            for claim in hook_claims
        }
        duplicate_candidate_hook_count = max(
            sum(len(candidate.hook_indices) for candidate in candidates)
            - len(
                {
                    int(hook_index)
                    for candidate in candidates
                    for hook_index in candidate.hook_indices
                }
            ),
            0,
        )
        hook_claim_conflict_count = AnchorMountAlignmentService._count_hook_claim_conflicts(
            items=items,
        )
        coverage_ratio = (
            float(anchored_count) / float(unit_count)
            if unit_count
            else 0.0
        )
        alignment_score = max(
            0.0,
            1.0 - float(unresolved_count) / max(float(unit_count), 1.0),
        )
        return {
            "unit_count": unit_count,
            "anchored_count": anchored_count,
            "unresolved_count": unresolved_count,
            "inferred_count": inferred_count,
            "hard_anchor_count": hard_anchor_count,
            "reseed_count": solve_result.reseed_count,
            "boundary_evidence_count": int(boundary_evidence_count),
            "punctuation_fact_count": int(punctuation_fact_count),
            "largest_unresolved_span": AnchorMountAlignmentService._largest_unresolved_span(
                envelopes=envelopes,
            ),
            "hook_waste_ratio": (
                max(float(total_hooks - len(claimed_hook_ids)), 0.0) / float(total_hooks)
                if total_hooks
                else 0.0
            ),
            "soft_anchor_ratio": (
                float(soft_anchor_count) / float(unit_count)
                if unit_count
                else 0.0
            ),
            "hook_claim_conflict_count": hook_claim_conflict_count,
            "cross_chunk_lock_count": len(cross_chunk_locks),
            "duplicate_candidate_hook_count": duplicate_candidate_hook_count,
            "unmapped_punctuation_count": len(
                getattr(punctuation_diagnostics, "unmapped", ()) or ()
            ),
            "dropped_punctuation_count": len(
                getattr(punctuation_diagnostics, "dropped_with_reason", ()) or ()
            ),
            "coverage_ratio": coverage_ratio,
            "alignment_score": alignment_score,
            "punctuation_unmapped": [
                dict(item) for item in (getattr(punctuation_diagnostics, "unmapped", ()) or ())
            ],
            "punctuation_dropped_with_reason": [
                dict(item)
                for item in (getattr(punctuation_diagnostics, "dropped_with_reason", ()) or ())
            ],
        }

    @staticmethod
    def _largest_unresolved_span(*, envelopes: tuple[Any, ...]) -> int:
        current = 0
        largest = 0
        for envelope in envelopes:
            if str(envelope.envelope_kind) in {"unresolved", "inferred"}:
                current += 1
                largest = max(largest, current)
                continue
            current = 0
        return largest

    @staticmethod
    def _count_hook_claim_conflicts(*, items: tuple[Any, ...]) -> int:
        hook_to_chunk_sets: dict[str, set[tuple[str, ...]]] = {}
        for item in items:
            for hook_id in getattr(item, "source_hook_ids", ()) or ():
                hook_to_chunk_sets.setdefault(str(hook_id), set()).add(
                    tuple(str(chunk_id) for chunk_id in getattr(item, "source_chunk_ids", ()) or ())
                )
        return sum(max(len(chunk_sets) - 1, 0) for chunk_sets in hook_to_chunk_sets.values())
