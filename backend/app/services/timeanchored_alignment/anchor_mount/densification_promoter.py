"""把 gap rescue 的局部匹配晋升为 secondary anchors。"""

from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.anchor_trust_evaluator import (
    AnchorTrustEvaluator,
)
from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorCandidate,
    PromotedAnchor,
)
from app.services.timeanchored_alignment.anchor_mount.window_alignment_state import (
    WindowAlignmentState,
)


class DensificationPromoter:
    """只允许 rescue 命中的局部唯一点晋升，且不能覆盖 primary。"""

    def __init__(
        self,
        *,
        trust_evaluator: AnchorTrustEvaluator | None = None,
    ) -> None:
        self._trust_evaluator = trust_evaluator or AnchorTrustEvaluator()

    def run(self, *, state: WindowAlignmentState) -> WindowAlignmentState:
        occupied_units, occupied_hooks = self._collect_primary_spans(state=state)
        promoted: list[PromotedAnchor] = []
        for gap in state.open_gaps:
            for match in getattr(gap, "rescue_matches", ()) or ():
                if set(match.unit_indices) & occupied_units:
                    continue
                if set(match.hook_indices) & occupied_hooks:
                    continue
                candidate = AnchorCandidate(
                    candidate_id=(
                        f"promoted:{gap.gap_id}:"
                        f"{'-'.join(str(item) for item in match.unit_indices)}:"
                        f"{'-'.join(str(item) for item in match.hook_indices)}"
                    ),
                    unit_indices=match.unit_indices,
                    hook_indices=match.hook_indices,
                    anchor_kind=match.anchor_kind,
                    score=min(max(float(match.score), 0.0), 1.0),
                    is_hard=False,
                )
                report = self._trust_evaluator.evaluate_candidate(candidate)
                if report.trust_tier not in {"primary", "secondary"}:
                    continue
                promoted.append(
                    PromotedAnchor(
                        candidate_id=candidate.candidate_id,
                        unit_indices=candidate.unit_indices,
                        hook_indices=candidate.hook_indices,
                        anchor_kind=candidate.anchor_kind,
                        source_gap_id=gap.gap_id,
                        trust_score=max(float(report.trust_score), float(candidate.score)),
                        trust_tier="secondary",
                        reject_reason=report.reject_reason,
                    )
                )
                occupied_units.update(match.unit_indices)
                occupied_hooks.update(match.hook_indices)

        diagnostics = dict(state.diagnostics)
        diagnostics["promoted_anchor_count"] = len(promoted)
        return state.with_updates(
            promoted_secondary_anchors=tuple(promoted),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _collect_primary_spans(
        *,
        state: WindowAlignmentState,
    ) -> tuple[set[int], set[int]]:
        occupied_units: set[int] = set()
        occupied_hooks: set[int] = set()
        for candidate in state.primary_candidates:
            occupied_units.update(int(item) for item in candidate.unit_indices)
            occupied_hooks.update(int(item) for item in candidate.hook_indices)
        solve_result = state.main_chain
        for block in getattr(solve_result, "committed_blocks", ()) or ():
            if str(getattr(block, "trust_tier", "")) != "primary":
                continue
            occupied_units.update(int(item) for item in block.unit_indices)
            occupied_hooks.update(int(item) for item in block.hook_indices)
        return occupied_units, occupied_hooks
