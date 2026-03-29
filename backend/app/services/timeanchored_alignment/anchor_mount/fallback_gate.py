"""AnchorMount 质量门。"""

from __future__ import annotations

from app.services.timeanchored_alignment.anchor_mount.contracts import AnchorMountResult
from app.services.timeanchored_alignment.anchor_mount.timeline_validity_validator import (
    TimelineValidityReport,
    TimelineValidityValidator,
)


class FallbackGate:
    """兼容旧 fallback 布尔，同时让核心语义落到 timeline_validity。"""

    def __init__(
        self,
        *,
        timeline_validity_validator: TimelineValidityValidator | None = None,
    ) -> None:
        self._timeline_validity_validator = timeline_validity_validator or TimelineValidityValidator()

    def evaluate(self, *, result: AnchorMountResult) -> TimelineValidityReport:
        report = self._timeline_validity_validator.validate(result=result)
        if report.state != "valid":
            return report

        anchored = int(result.metrics.get("anchored_count", 0) or 0)
        total = int(result.metrics.get("unit_count", 0) or 0)
        hook_waste_ratio = float(result.metrics.get("hook_waste_ratio", 0.0) or 0.0)
        unmapped_punctuation_count = int(result.metrics.get("unmapped_punctuation_count", 0) or 0)
        reasons = list(report.reasons)
        state = report.state
        if total <= 0 or anchored <= 0:
            state = "fatal"
            reasons.append("no_primary_coverage")
        elif hook_waste_ratio >= 0.5 or unmapped_punctuation_count > 0:
            state = "repairable"
            if hook_waste_ratio >= 0.5:
                reasons.append("hook_waste_high")
            if unmapped_punctuation_count > 0:
                reasons.append("punctuation_unmapped")
        return TimelineValidityReport(
            state=state,
            monotonic_violation_count=report.monotonic_violation_count,
            compressed_run_count=report.compressed_run_count,
            residual_gap_token_count=report.residual_gap_token_count,
            invalid_gap_ids=report.invalid_gap_ids,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def should_fallback(
        self,
        *,
        result: AnchorMountResult | None = None,
        validity_report: TimelineValidityReport | None = None,
    ) -> bool:
        report = validity_report
        if report is None:
            if result is None:
                raise ValueError("FallbackGate.should_fallback 需要 result 或 validity_report")
            report = self.evaluate(result=result)
        return report.state != "valid"
