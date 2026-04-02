"""Window partial-commit recovery contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WindowSpanDecision:
    span_id: str
    span_kind: str
    route: str
    token_start: int = -1
    token_end: int = -1
    char_start: int = -1
    char_end: int = -1
    time_start: float = 0.0
    time_end: float = 0.0
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SpanFallbackPlan:
    span_id: str
    text: str = ""
    route: str = "span_fallback"
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RecoveredWindowPlan:
    spans: tuple[WindowSpanDecision, ...] = field(default_factory=tuple)
    fallback_plans: tuple[SpanFallbackPlan, ...] = field(default_factory=tuple)
    emergency_fallback_only: bool = False
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_recoverable_spans(self) -> bool:
        if self.emergency_fallback_only:
            return False
        return any(
            str(span.span_kind or "").strip().lower() in {"trusted", "interpolated"}
            for span in self.spans
        )
