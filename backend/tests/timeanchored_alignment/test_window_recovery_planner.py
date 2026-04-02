from __future__ import annotations

from types import SimpleNamespace

from app.services.timeanchored_alignment.window_recovery_contracts import (
    RecoveredWindowPlan,
    WindowSpanDecision,
)
from app.services.timeanchored_alignment.window_recovery_planner import WindowRecoveryPlanner


def _token(
    *,
    unit_id: str,
    token_index: int,
    char_start: int,
    char_end: int,
    start: float,
    end: float,
    mount_status: str,
):
    return SimpleNamespace(
        unit_id=unit_id,
        token_index=token_index,
        char_start=char_start,
        char_end=char_end,
        start=start,
        end=end,
        mount_status=mount_status,
    )


def _stage_result(
    *,
    tokens,
    timeline_validity: str = "quarantined",
    validity_reasons: tuple[str, ...] = ("unresolved_gap_present",),
):
    return SimpleNamespace(
        decision_ingress=SimpleNamespace(
            anchored_token_units=tuple(tokens),
            boundary_evidences=tuple(),
            punctuation_facts=tuple(),
        ),
        anchor_mount_result=SimpleNamespace(
            timeline_validity=timeline_validity,
            validity_reasons=validity_reasons,
            metrics={"largest_unresolved_span": 1},
        ),
    )


def test_recovered_window_plan_exposes_span_level_routes() -> None:
    plan = RecoveredWindowPlan(
        spans=(
            WindowSpanDecision(span_id="s0", span_kind="trusted", route="decision"),
            WindowSpanDecision(span_id="s1", span_kind="fallback", route="span_fallback"),
        )
    )

    assert [item.span_kind for item in plan.spans] == ["trusted", "fallback"]


def test_window_recovery_planner_splits_prefix_suffix_fallback_and_middle_trusted() -> None:
    planner = WindowRecoveryPlanner()
    stage_result = _stage_result(
        tokens=(
            _token(
                unit_id="u0",
                token_index=0,
                char_start=0,
                char_end=5,
                start=0.0,
                end=0.2,
                mount_status="unresolved",
            ),
            _token(
                unit_id="u1",
                token_index=1,
                char_start=6,
                char_end=11,
                start=0.2,
                end=0.4,
                mount_status="anchored",
            ),
            _token(
                unit_id="u2",
                token_index=2,
                char_start=12,
                char_end=17,
                start=0.4,
                end=0.6,
                mount_status="anchored",
            ),
            _token(
                unit_id="u3",
                token_index=3,
                char_start=18,
                char_end=23,
                start=0.6,
                end=0.8,
                mount_status="unresolved",
            ),
        ),
    )

    plan = planner.build(stage_result=stage_result)

    assert [span.span_kind for span in plan.spans] == ["fallback", "trusted", "fallback"]
    assert plan.has_recoverable_spans is True


def test_window_recovery_plan_does_not_treat_monotonic_violation_island_as_recoverable() -> None:
    planner = WindowRecoveryPlanner()
    stage_result = _stage_result(
        tokens=(
            _token(
                unit_id="u0",
                token_index=0,
                char_start=0,
                char_end=3,
                start=0.0,
                end=0.3,
                mount_status="unresolved",
            ),
            _token(
                unit_id="u1",
                token_index=1,
                char_start=4,
                char_end=7,
                start=0.3,
                end=0.6,
                mount_status="merged",
            ),
            _token(
                unit_id="u2",
                token_index=2,
                char_start=8,
                char_end=11,
                start=0.2,
                end=0.4,
                mount_status="anchored",
            ),
            _token(
                unit_id="u3",
                token_index=3,
                char_start=12,
                char_end=15,
                start=0.6,
                end=0.9,
                mount_status="unresolved",
            ),
        ),
        timeline_validity="fatal",
        validity_reasons=("monotonic_violation", "coverage_too_low"),
    )

    plan = planner.build(stage_result=stage_result)

    assert plan.has_recoverable_spans is False
    assert plan.emergency_fallback_only is True


def test_window_recovery_plan_demotes_backtracking_span_to_fallback() -> None:
    planner = WindowRecoveryPlanner()
    stage_result = _stage_result(
        tokens=(
            _token(
                unit_id="u0",
                token_index=0,
                char_start=0,
                char_end=2,
                start=0.0,
                end=0.2,
                mount_status="unresolved",
            ),
            _token(
                unit_id="u1",
                token_index=1,
                char_start=3,
                char_end=5,
                start=0.2,
                end=0.4,
                mount_status="merged",
            ),
            _token(
                unit_id="u2",
                token_index=2,
                char_start=6,
                char_end=8,
                start=0.1,
                end=0.3,
                mount_status="anchored",
            ),
            _token(
                unit_id="u3",
                token_index=3,
                char_start=9,
                char_end=11,
                start=0.5,
                end=0.7,
                mount_status="unresolved",
            ),
        ),
        timeline_validity="fatal",
        validity_reasons=("monotonic_violation",),
    )

    plan = planner.build(stage_result=stage_result)

    assert [span.span_kind for span in plan.spans] == ["fallback", "fallback", "fallback"]
