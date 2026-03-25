"""AnchorMountAlignment 层主入口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.language_policy import build_language_policy_snapshot
from app.services.timeanchored_alignment.anchor_mount.boundary_hint_assembler import (
    BoundaryHintAssembler,
)
from app.services.timeanchored_alignment.anchor_mount.chain_solver import ChainSolveResult, ChainSolver
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
from app.services.timeanchored_alignment.anchor_mount.fallback_gate import FallbackGate
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
from app.services.timeanchored_alignment.contracts import LayerReport, PipelineReport
from app.services.timeanchored_alignment.preparation.contracts import AlignmentPreparationPackage


@dataclass(frozen=True)
class AnchorMountStageResult:
    """Chunk3 对外阶段结果。"""

    decision_ingress: DecisionIngressPackage
    anchor_mount_result: AnchorMountResult
    pipeline_report: PipelineReport


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
        temporal_envelope_builder: TemporalEnvelopeBuilder | None = None,
        hook_claim_resolver: HookClaimResolver | None = None,
        cross_chunk_lock_builder: CrossChunkLockBuilder | None = None,
        boundary_hint_assembler: BoundaryHintAssembler | None = None,
        fallback_gate: FallbackGate | None = None,
        decision_ingress_assembler: DecisionIngressAssembler | None = None,
    ) -> None:
        self._ingress_validator = ingress_validator or IngressValidator()
        self._punctuation_fact_mapper = punctuation_fact_mapper or PunctuationFactMapper()
        self._seed_discovery = seed_discovery or SeedDiscovery()
        self._low_complexity_mask = low_complexity_mask or LowComplexityMask()
        self._local_extension = local_extension or LocalExtension()
        self._chain_solver = chain_solver or ChainSolver()
        self._temporal_envelope_builder = temporal_envelope_builder or TemporalEnvelopeBuilder()
        self._hook_claim_resolver = hook_claim_resolver or HookClaimResolver()
        self._cross_chunk_lock_builder = cross_chunk_lock_builder or CrossChunkLockBuilder()
        self._boundary_hint_assembler = boundary_hint_assembler or BoundaryHintAssembler()
        self._fallback_gate = fallback_gate or FallbackGate()
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
        punctuation_facts, punctuation_pair_states, punctuation_diagnostics = self._punctuation_fact_mapper.map(
            window_text=input_view.window_text,
            slots=input_view.slots,
            punctuation_evidences=input_view.punctuation_evidences,
        )
        candidates = self._seed_discovery.discover(input_view=input_view)
        masked_candidates = self._low_complexity_mask.apply(
            candidates=candidates,
            input_view=input_view,
        )
        blocks = self._local_extension.build(candidates=masked_candidates)
        solve_result = self._chain_solver.solve(input_view=input_view, blocks=blocks)
        items, envelopes = self._temporal_envelope_builder.build(
            input_view=input_view,
            solve_result=solve_result,
        )
        hook_claims = self._hook_claim_resolver.resolve(
            input_view=input_view,
            envelopes=envelopes,
        )
        cross_chunk_locks = self._cross_chunk_lock_builder.build(
            items=items,
            envelopes=envelopes,
        )
        boundary_hints = self._boundary_hint_assembler.build(
            items=items,
            envelopes=envelopes,
            punctuation_facts=punctuation_facts,
            cross_chunk_locks=cross_chunk_locks,
        )
        metrics = self._build_metrics(
            input_view=input_view,
            solve_result=solve_result,
            candidates=masked_candidates,
            items=items,
            envelopes=envelopes,
            hook_claims=hook_claims,
            cross_chunk_locks=cross_chunk_locks,
            boundary_hint_count=len(boundary_hints),
            punctuation_fact_count=len(punctuation_facts),
            punctuation_diagnostics=punctuation_diagnostics,
        )
        provisional_result = AnchorMountResult(
            items=items,
            envelopes=envelopes,
            punctuation_facts=punctuation_facts,
            punctuation_pair_states=punctuation_pair_states,
            hook_claims=hook_claims,
            cross_chunk_locks=cross_chunk_locks,
            boundary_hints=boundary_hints,
            metrics=metrics,
            should_fallback=False,
        )
        should_fallback = self._fallback_gate.should_fallback(result=provisional_result)
        anchor_mount_result = AnchorMountResult(
            items=provisional_result.items,
            envelopes=provisional_result.envelopes,
            punctuation_facts=provisional_result.punctuation_facts,
            punctuation_pair_states=provisional_result.punctuation_pair_states,
            hook_claims=provisional_result.hook_claims,
            cross_chunk_locks=provisional_result.cross_chunk_locks,
            boundary_hints=provisional_result.boundary_hints,
            metrics=provisional_result.metrics,
            should_fallback=should_fallback,
        )
        decision_ingress = self._decision_ingress_assembler.build(
            input_view=input_view,
            result=anchor_mount_result,
        )
        route = "slow" if decision_ingress.tokens else "error"
        pipeline_report = PipelineReport(
            alignment_report=LayerReport(
                name="anchor_mount_alignment",
                status="ok" if decision_ingress.tokens else "error",
                metrics={
                    "route": route,
                    "slot_count": len(items),
                    "anchored_count": metrics["anchored_count"],
                    "reseed_count": metrics["reseed_count"],
                    "should_fallback": should_fallback,
                },
            ),
            segmentation_report=LayerReport(
                name="anchor_mount_boundary",
                status="ok",
                metrics={
                    "boundary_candidate_count": len(decision_ingress.boundary_hints),
                    "punctuation_fact_count": len(punctuation_facts),
                },
            ),
            output_report=LayerReport(
                name="decision_ingress",
                status="ok",
                metrics={"token_count": len(decision_ingress.tokens)},
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
    def _build_metrics(
        *,
        input_view: Any,
        solve_result: ChainSolveResult,
        candidates: tuple[Any, ...],
        items: tuple[Any, ...],
        envelopes: tuple[Any, ...],
        hook_claims: tuple[Any, ...],
        cross_chunk_locks: tuple[Any, ...],
        boundary_hint_count: int,
        punctuation_fact_count: int,
        punctuation_diagnostics: Any,
    ) -> dict[str, Any]:
        slot_count = len(items)
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
            1 for kind in solve_result.slot_anchor_kinds if str(kind) == "soft_pronunciation"
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
            float(anchored_count) / float(slot_count)
            if slot_count
            else 0.0
        )
        alignment_score = max(
            0.0,
            1.0 - float(unresolved_count) / max(float(slot_count), 1.0),
        )
        return {
            "slot_count": slot_count,
            "anchored_count": anchored_count,
            "unresolved_count": unresolved_count,
            "inferred_count": inferred_count,
            "hard_anchor_count": hard_anchor_count,
            "reseed_count": solve_result.reseed_count,
            "boundary_hint_count": int(boundary_hint_count),
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
                float(soft_anchor_count) / float(slot_count)
                if slot_count
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
