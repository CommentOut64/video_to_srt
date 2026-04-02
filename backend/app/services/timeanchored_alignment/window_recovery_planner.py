"""Build span-level recovery plans for timeanchored windows."""

from __future__ import annotations

from dataclasses import replace

from app.services.timeanchored_alignment.window_recovery_contracts import (
    RecoveredWindowPlan,
    WindowSpanDecision,
)


class WindowRecoveryPlanner:
    """Classify a window into trusted/interpolated/fallback spans."""

    _TRUSTED_STATUSES = {"anchored", "merged", "promoted"}
    _INTERPOLATED_STATUSES = {"inferred", "interpolated"}

    def build(self, *, stage_result: object) -> RecoveredWindowPlan:
        timeline_validity = self._resolve_timeline_validity(stage_result=stage_result)
        reason_codes = self._resolve_reason_codes(stage_result=stage_result)
        decision_ingress = getattr(stage_result, "decision_ingress", None)
        token_units = tuple(getattr(decision_ingress, "anchored_token_units", ()) or ())
        if not token_units:
            return RecoveredWindowPlan(reason_codes=reason_codes)

        spans: list[WindowSpanDecision] = []
        current_kind: str | None = None
        current_tokens: list[object] = []
        for token in token_units:
            token_kind = self._classify_token(token=token)
            if current_kind is None or token_kind == current_kind:
                current_kind = token_kind
                current_tokens.append(token)
                continue
            spans.append(self._build_span(span_index=len(spans), span_kind=current_kind, tokens=current_tokens))
            current_kind = token_kind
            current_tokens = [token]
        if current_kind is not None and current_tokens:
            spans.append(self._build_span(span_index=len(spans), span_kind=current_kind, tokens=current_tokens))
        normalized_spans = self._demote_backtracking_spans(tuple(spans))
        emergency_fallback_only = False
        if (
            timeline_validity in {"fatal", "quarantined"}
            and "monotonic_violation" in reason_codes
            and not self._spans_form_monotonic_commit(normalized_spans)
        ):
            emergency_fallback_only = True
        return RecoveredWindowPlan(
            spans=normalized_spans,
            emergency_fallback_only=emergency_fallback_only,
            reason_codes=reason_codes,
        )

    def _classify_token(self, *, token: object) -> str:
        mount_status = str(getattr(token, "mount_status", "") or "").strip().lower()
        if mount_status in self._TRUSTED_STATUSES:
            return "trusted"
        if mount_status in self._INTERPOLATED_STATUSES:
            return "interpolated"
        return "fallback"

    @staticmethod
    def _build_span(
        *,
        span_index: int,
        span_kind: str,
        tokens: list[object],
    ) -> WindowSpanDecision:
        first = tokens[0]
        last = tokens[-1]
        route = "decision" if span_kind in {"trusted", "interpolated"} else "span_fallback"
        return WindowSpanDecision(
            span_id=f"span-{span_index}",
            span_kind=span_kind,
            route=route,
            token_start=int(getattr(first, "token_index", span_index)),
            token_end=int(getattr(last, "token_index", span_index)) + 1,
            char_start=int(getattr(first, "char_start", -1)),
            char_end=int(getattr(last, "char_end", -1)),
            time_start=float(getattr(first, "start", 0.0) or 0.0),
            time_end=float(getattr(last, "end", 0.0) or 0.0),
            reason_codes=(f"mount_status:{span_kind}",),
            provenance={
                "unit_ids": [str(getattr(token, "unit_id", "") or "") for token in tokens],
                "token_times": [
                    {
                        "start": float(getattr(token, "start", 0.0) or 0.0),
                        "end": float(getattr(token, "end", getattr(token, "start", 0.0)) or 0.0),
                    }
                    for token in tokens
                ],
            },
        )

    @staticmethod
    def _resolve_timeline_validity(*, stage_result: object) -> str:
        anchor_mount_result = getattr(stage_result, "anchor_mount_result", None)
        return str(getattr(anchor_mount_result, "timeline_validity", "") or "").strip().lower()

    @staticmethod
    def _resolve_reason_codes(*, stage_result: object) -> tuple[str, ...]:
        anchor_mount_result = getattr(stage_result, "anchor_mount_result", None)
        raw_reason_codes = tuple(getattr(anchor_mount_result, "validity_reasons", ()) or ())
        return tuple(str(item).strip().lower() for item in raw_reason_codes if str(item).strip())

    @classmethod
    def _demote_backtracking_spans(
        cls,
        spans: tuple[WindowSpanDecision, ...],
    ) -> tuple[WindowSpanDecision, ...]:
        normalized: list[WindowSpanDecision] = []
        for span in spans:
            if span.span_kind not in {"trusted", "interpolated"}:
                normalized.append(span)
                continue
            if not cls._span_has_backtracking(span):
                normalized.append(span)
                continue
            normalized.append(
                replace(
                    span,
                    span_kind="fallback",
                    route="span_fallback",
                    reason_codes=tuple(span.reason_codes) + ("monotonic_backtrack",),
                )
            )
        return tuple(normalized)

    @staticmethod
    def _span_has_backtracking(span: WindowSpanDecision) -> bool:
        token_times = tuple(
            item
            for item in list((span.provenance or {}).get("token_times") or [])
            if isinstance(item, dict)
        )
        previous_end: float | None = None
        for item in token_times:
            start = float(item.get("start", 0.0) or 0.0)
            end = float(item.get("end", start) or start)
            if previous_end is not None and start < previous_end:
                return True
            previous_end = max(end, start)
        return False

    @staticmethod
    def _spans_form_monotonic_commit(spans: tuple[WindowSpanDecision, ...]) -> bool:
        commit_spans = [span for span in spans if span.span_kind in {"trusted", "interpolated"}]
        if not commit_spans:
            return False
        previous_end: float | None = None
        for span in commit_spans:
            if previous_end is not None and float(span.time_start) < previous_end:
                return False
            previous_end = max(float(span.time_end), float(span.time_start))
        return True
