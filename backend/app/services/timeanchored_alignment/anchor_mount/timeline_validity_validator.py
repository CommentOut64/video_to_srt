"""时间轴合法性校验。"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.timeanchored_alignment.anchor_mount.contracts import AnchorMountResult


@dataclass(frozen=True)
class TimelineValidityReport:
    state: str
    monotonic_violation_count: int
    compressed_run_count: int
    residual_gap_token_count: int
    invalid_gap_ids: tuple[str, ...]
    reasons: tuple[str, ...]


class TimelineValidityValidator:
    """根据当前挂载结果生成结构化 validity，而不是单布尔 fallback。"""

    _OVERLAP_TOLERANCE = 1e-3
    _FATAL_COVERAGE_RATIO = 0.35
    _QUARANTINED_COVERAGE_RATIO = 0.50

    def validate(self, *, result: AnchorMountResult) -> TimelineValidityReport:
        monotonic_violation_count = 0
        compressed_run_count = 0
        invalid_gap_ids: list[str] = []
        previous_end: float | None = None
        for envelope in result.envelopes:
            start = float(envelope.provisional_start)
            end = float(envelope.provisional_end)
            if previous_end is not None and start + self._OVERLAP_TOLERANCE < previous_end:
                monotonic_violation_count += 1
                invalid_gap_ids.append(str(envelope.unit_id))
            if end - start <= self._OVERLAP_TOLERANCE:
                compressed_run_count += 1
                invalid_gap_ids.append(str(envelope.unit_id))
            previous_end = max(previous_end or start, end)

        unresolved_count = int(result.metrics.get("unresolved_count", 0) or 0)
        inferred_count = int(result.metrics.get("inferred_count", 0) or 0)
        largest_unresolved_span = int(result.metrics.get("largest_unresolved_span", 0) or 0)
        hook_claim_conflict_count = int(result.metrics.get("hook_claim_conflict_count", 0) or 0)
        coverage_ratio = float(result.metrics.get("coverage_ratio", 0.0) or 0.0)
        residual_gap_token_count = sum(
            1
            for envelope in result.envelopes
            if str(envelope.envelope_kind) in {"unresolved", "inferred"}
            or str(getattr(envelope, "gap_state", "")) in {"residual", "large_residual", "invalid"}
        )

        reasons: list[str] = []
        if not result.items or not result.envelopes:
            reasons.append("empty_timeline")
        if monotonic_violation_count > 0:
            reasons.append("monotonic_violation")
        if compressed_run_count > 0:
            reasons.append("compressed_run")
        if hook_claim_conflict_count > 0:
            reasons.append("hook_claim_conflict")
        if largest_unresolved_span > 1:
            reasons.append("large_residual_gap")
        if inferred_count > 0:
            reasons.append("inferred_gap_present")
        if unresolved_count > 0:
            reasons.append("unresolved_gap_present")
        if coverage_ratio < self._FATAL_COVERAGE_RATIO:
            reasons.append("coverage_too_low")

        state = "valid"
        if "empty_timeline" in reasons or "coverage_too_low" in reasons:
            state = "fatal"
        elif (
            "monotonic_violation" in reasons
            or "compressed_run" in reasons
            or "hook_claim_conflict" in reasons
            or (coverage_ratio < self._QUARANTINED_COVERAGE_RATIO and residual_gap_token_count > 0)
        ):
            state = "quarantined"
        elif residual_gap_token_count > 0 or unresolved_count > 0 or inferred_count > 0:
            state = "repairable"

        return TimelineValidityReport(
            state=state,
            monotonic_violation_count=monotonic_violation_count,
            compressed_run_count=compressed_run_count,
            residual_gap_token_count=residual_gap_token_count,
            invalid_gap_ids=tuple(dict.fromkeys(invalid_gap_ids)),
            reasons=tuple(dict.fromkeys(reasons)),
        )
