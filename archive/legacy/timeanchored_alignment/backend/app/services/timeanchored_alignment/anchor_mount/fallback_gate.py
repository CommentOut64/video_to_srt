"""AnchorMount 质量门。"""

from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.contracts import AnchorMountResult


class FallbackGate:
    """根据挂载质量判断是否需要后续 fallback。"""

    def should_fallback(self, *, result: AnchorMountResult) -> bool:
        anchored = int(result.metrics.get("anchored_count", 0) or 0)
        total = int(result.metrics.get("unit_count", 0) or 0)
        unresolved = int(result.metrics.get("unresolved_count", 0) or 0)
        inferred = int(result.metrics.get("inferred_count", 0) or 0)
        largest_unresolved_span = int(result.metrics.get("largest_unresolved_span", 0) or 0)
        hook_claim_conflict_count = int(
            result.metrics.get("hook_claim_conflict_count", 0) or 0
        )
        unmapped_punctuation_count = int(
            result.metrics.get("unmapped_punctuation_count", 0) or 0
        )
        hook_waste_ratio = float(result.metrics.get("hook_waste_ratio", 0.0) or 0.0)
        if total <= 0:
            return True
        if anchored <= 0:
            return True
        if unresolved > 0:
            return True
        if inferred > 0:
            return True
        if largest_unresolved_span > 1:
            return True
        if hook_claim_conflict_count > 0:
            return True
        if unmapped_punctuation_count > 0:
            return True
        if hook_waste_ratio >= 0.5:
            return True
        return False
