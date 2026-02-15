"""
软切决策引擎（Phase C）。
V3.2.0+dev.20260214.09
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .types import (
    AnchorScore,
    AnchorType,
    CutDecision,
    CutPlan,
    CutWindow,
    CutWindowState,
    DeferredCut,
    DeferredCutState,
    EvidenceLevel,
)


@dataclass
class WindowDecisionContext:
    """窗口决策上下文。"""

    current_sentence_word_count: int = 0
    waiting_word_count: int = 0
    depends_on_fast_draft: bool = False


@dataclass
class DecisionEngineConfig:
    """决策引擎参数。"""

    mid_high_quality_threshold: float = 0.60
    mid_medium_quality_threshold: float = 0.40
    length_pressure_soft: int = 15
    length_pressure_hard: int = 25
    deferred_max_wait_sec: float = 2.5
    deferred_max_words: int = 18
    deferred_force_min_word_boundary_gap: float = 0.08
    max_deferred_windows: int = 3


class SoftCutDecisionEngine:
    """
    软切决策引擎（State Machine Pattern）。

    Why:
    - 把 high/mid/low 决策矩阵与 deferred 生命周期集中管理，避免规则散落。
    - 输出固定 `CutPlan` 契约，保证后续 Phase D 集成时调用边界稳定。
    """

    def __init__(self, config: Optional[DecisionEngineConfig] = None) -> None:
        self.config = config or DecisionEngineConfig()

    def decide(
        self,
        *,
        block_id: str,
        cut_windows: Sequence[CutWindow],
        window_contexts: Optional[dict[str, WindowDecisionContext]] = None,
        previous_deferred: Optional[Sequence[DeferredCut]] = None,
        current_time: Optional[float] = None,
    ) -> CutPlan:
        """根据窗口与上下文生成 CutPlan。"""
        contexts = window_contexts or {}
        existing_deferred = list(previous_deferred or [])
        ordered_windows = sorted(cut_windows, key=lambda item: (item.trigger_time, item.window_id))
        decision_time = self._resolve_decision_time(cut_windows=ordered_windows, current_time=current_time)

        decisions: list[CutDecision] = []
        deferred_records: list[DeferredCut] = []
        report = {
            "window_count": int(len(ordered_windows)),
            "decision_count": 0,
            "high_forced_count": 0,
            "mid_immediate_count": 0,
            "mid_deferred_count": 0,
            "low_queued_count": 0,
            "length_pressure_cut_count": 0,
            "deferred_forced_count": 0,
            "overflow_forced_count": 0,
            "deferred_pending_count": 0,
        }

        pending_deferred = self._resolve_existing_deferred(
            previous_deferred=existing_deferred,
            contexts=contexts,
            decision_time=decision_time,
            decisions=decisions,
            report=report,
            deferred_records=deferred_records,
        )

        for window in ordered_windows:
            context = contexts.get(window.window_id, WindowDecisionContext())
            best_anchor = window.candidate_anchors[0] if window.candidate_anchors else None

            if window.trigger_level == EvidenceLevel.HIGH:
                self._handle_high_window(
                    window=window,
                    best_anchor=best_anchor,
                    context=context,
                    decision_time=decision_time,
                    decisions=decisions,
                    pending_deferred=pending_deferred,
                    deferred_records=deferred_records,
                    report=report,
                )
                continue

            if window.trigger_level == EvidenceLevel.MID:
                self._handle_mid_window(
                    window=window,
                    best_anchor=best_anchor,
                    context=context,
                    decisions=decisions,
                    pending_deferred=pending_deferred,
                    deferred_records=deferred_records,
                    report=report,
                )
                continue

            # LOW: 进入 evidence_queue（当前通过 pending deferred 建模）。
            pending_deferred.append(self._create_deferred(window, context=context))
            deferred_records.append(pending_deferred[-1])
            report["low_queued_count"] += 1

        self._enforce_deferred_capacity(
            pending_deferred=pending_deferred,
            contexts=contexts,
            decisions=decisions,
            report=report,
            decision_time=decision_time,
        )

        decisions.sort(key=lambda item: (item.time, item.window_id))
        report["decision_count"] = int(len(decisions))
        report["deferred_pending_count"] = int(
            sum(1 for item in deferred_records if item.state == DeferredCutState.PENDING)
        )
        plan = CutPlan(
            plan_id=f"{block_id}-plan-{int(decision_time * 1000):010d}",
            block_id=block_id,
            decisions=decisions,
            deferred_cuts=deferred_records,
            generation_report=report,
        )
        return plan

    def _resolve_existing_deferred(
        self,
        *,
        previous_deferred: Sequence[DeferredCut],
        contexts: dict[str, WindowDecisionContext],
        decision_time: float,
        decisions: list[CutDecision],
        report: dict[str, int],
        deferred_records: list[DeferredCut],
    ) -> list[DeferredCut]:
        pending: list[DeferredCut] = []
        for deferred in previous_deferred:
            if deferred.state != DeferredCutState.PENDING:
                deferred_records.append(deferred)
                continue

            context = contexts.get(deferred.window_id, WindowDecisionContext())
            depends_on_fast_draft = bool(
                context.depends_on_fast_draft or getattr(deferred, "depends_on_fast_draft", False)
            )
            is_timed_out = decision_time >= deferred.expected_resolve_by
            is_word_overflow = context.waiting_word_count >= self.config.deferred_max_words
            if is_timed_out or is_word_overflow:
                time_range = self._resolve_deferred_time_range(deferred)
                forced_decision = self._build_forced_decision(
                    window_id=deferred.window_id,
                    time_range=time_range,
                    decision_time=decision_time,
                    reason="deferred_forced",
                    risk="deferred",
                    depends_on_fast_draft=depends_on_fast_draft,
                )
                deferred.state = DeferredCutState.FORCED
                deferred.resolution_decision = forced_decision
                decisions.append(forced_decision)
                report["deferred_forced_count"] += 1
            else:
                pending.append(deferred)
            deferred_records.append(deferred)
        return pending

    def _handle_high_window(
        self,
        *,
        window: CutWindow,
        best_anchor: Optional[AnchorScore],
        context: WindowDecisionContext,
        decision_time: float,
        decisions: list[CutDecision],
        pending_deferred: list[DeferredCut],
        deferred_records: list[DeferredCut],
        report: dict[str, int],
    ) -> None:
        if best_anchor is not None:
            decisions.append(
                self._build_anchor_decision(
                    window=window,
                    anchor=best_anchor,
                    reason=self._resolve_anchor_reason(best_anchor),
                    risk=None,
                    depends_on_fast_draft=context.depends_on_fast_draft,
                )
            )
            window.state = CutWindowState.RESOLVED
            return

        expected_resolve_by = window.trigger_time + self.config.deferred_max_wait_sec
        if decision_time >= expected_resolve_by:
            decisions.append(
                self._build_forced_decision(
                    window_id=window.window_id,
                    time_range=(window.start_time, window.end_time),
                    decision_time=decision_time,
                    reason="high_forced",
                    risk="deferred",
                    depends_on_fast_draft=context.depends_on_fast_draft,
                )
            )
            window.state = CutWindowState.FORCED
            report["high_forced_count"] += 1
            return

        pending = self._create_deferred(window, context=context)
        pending_deferred.append(pending)
        deferred_records.append(pending)

    def _handle_mid_window(
        self,
        *,
        window: CutWindow,
        best_anchor: Optional[AnchorScore],
        context: WindowDecisionContext,
        decisions: list[CutDecision],
        pending_deferred: list[DeferredCut],
        deferred_records: list[DeferredCut],
        report: dict[str, int],
    ) -> None:
        action = self._decide_mid_action(
            anchor_score=best_anchor.final_score if best_anchor is not None else 0.0,
            current_sentence_word_count=context.current_sentence_word_count,
        )
        if action == "defer":
            pending = self._create_deferred(window, context=context)
            pending_deferred.append(pending)
            deferred_records.append(pending)
            report["mid_deferred_count"] += 1
            return

        if best_anchor is None:
            pending = self._create_deferred(window, context=context)
            pending_deferred.append(pending)
            deferred_records.append(pending)
            report["mid_deferred_count"] += 1
            return

        reason = "length_pressure" if action == "length_pressure_cut" else self._resolve_anchor_reason(best_anchor)
        decisions.append(
            self._build_anchor_decision(
                window=window,
                anchor=best_anchor,
                reason=reason,
                risk=None,
                depends_on_fast_draft=context.depends_on_fast_draft,
            )
        )
        window.state = CutWindowState.RESOLVED
        report["mid_immediate_count"] += 1
        if reason == "length_pressure":
            report["length_pressure_cut_count"] += 1

    def _enforce_deferred_capacity(
        self,
        *,
        pending_deferred: list[DeferredCut],
        contexts: dict[str, WindowDecisionContext],
        decisions: list[CutDecision],
        report: dict[str, int],
        decision_time: float,
    ) -> None:
        if self.config.max_deferred_windows <= 0:
            return

        pending_deferred.sort(key=lambda item: (item.created_at, item.window_id))
        while len(pending_deferred) > self.config.max_deferred_windows:
            oldest = pending_deferred.pop(0)
            context = contexts.get(oldest.window_id, WindowDecisionContext())
            depends_on_fast_draft = bool(
                context.depends_on_fast_draft or getattr(oldest, "depends_on_fast_draft", False)
            )
            forced_decision = self._build_forced_decision(
                window_id=oldest.window_id,
                time_range=self._resolve_deferred_time_range(oldest),
                decision_time=decision_time,
                reason="deferred_forced",
                risk="overflow_forced",
                depends_on_fast_draft=depends_on_fast_draft,
            )
            oldest.state = DeferredCutState.FORCED
            oldest.resolution_decision = forced_decision
            decisions.append(forced_decision)
            report["deferred_forced_count"] += 1
            report["overflow_forced_count"] += 1

    def _build_anchor_decision(
        self,
        *,
        window: CutWindow,
        anchor: AnchorScore,
        reason: str,
        risk: Optional[str],
        depends_on_fast_draft: bool,
    ) -> CutDecision:
        return CutDecision(
            time=float(anchor.anchor_time),
            window_id=window.window_id,
            reason=reason,
            risk=risk,
            anchor_type=anchor.anchor_type,
            anchor_score=float(anchor.final_score),
            depends_on_fast_draft=depends_on_fast_draft,
            time_range=(float(window.start_time), float(window.end_time)),
        )

    def _build_forced_decision(
        self,
        *,
        window_id: str,
        time_range: tuple[float, float],
        decision_time: float,
        reason: str,
        risk: Optional[str],
        depends_on_fast_draft: bool,
    ) -> CutDecision:
        start, end = time_range
        target_time = min(max(decision_time, start), end if end > start else decision_time)
        target_time = max(start + self.config.deferred_force_min_word_boundary_gap, target_time)
        if end > start:
            target_time = min(target_time, end)
        return CutDecision(
            time=float(target_time),
            window_id=window_id,
            reason=reason,
            risk=risk,
            anchor_type=AnchorType.WORD_BOUNDARY,
            anchor_score=0.0,
            depends_on_fast_draft=depends_on_fast_draft,
            time_range=(float(start), float(end)),
        )

    def _create_deferred(
        self,
        window: CutWindow,
        *,
        context: Optional[WindowDecisionContext] = None,
    ) -> DeferredCut:
        active_context = context or WindowDecisionContext()
        return DeferredCut(
            deferred_id=f"{window.window_id}-deferred",
            window_id=window.window_id,
            created_at=float(window.trigger_time),
            expected_resolve_by=float(window.trigger_time + self.config.deferred_max_wait_sec),
            state=DeferredCutState.PENDING,
            window_start=float(window.start_time),
            window_end=float(window.end_time),
            trigger_level=str(getattr(window.trigger_level, "value", window.trigger_level)),
            depends_on_fast_draft=bool(active_context.depends_on_fast_draft),
        )

    @staticmethod
    def _resolve_deferred_time_range(deferred: DeferredCut) -> tuple[float, float]:
        start = (
            float(deferred.window_start)
            if getattr(deferred, "window_start", None) is not None
            else float(deferred.created_at)
        )
        end = (
            float(deferred.window_end)
            if getattr(deferred, "window_end", None) is not None
            else float(deferred.expected_resolve_by)
        )
        if end <= start:
            end = max(float(deferred.expected_resolve_by), start + 1e-3)
        return start, end

    def _decide_mid_action(self, *, anchor_score: float, current_sentence_word_count: int) -> str:
        if anchor_score >= self.config.mid_high_quality_threshold:
            return "cut"
        if anchor_score >= self.config.mid_medium_quality_threshold:
            if current_sentence_word_count > self.config.length_pressure_soft:
                return "cut"
            return "defer"
        if current_sentence_word_count > self.config.length_pressure_hard:
            return "length_pressure_cut"
        return "defer"

    @staticmethod
    def _resolve_anchor_reason(anchor: AnchorScore) -> str:
        if anchor.anchor_type == AnchorType.PAUSE_ANCHOR:
            return "pause"
        return "speaker_change"

    @staticmethod
    def _resolve_decision_time(
        *,
        cut_windows: Sequence[CutWindow],
        current_time: Optional[float],
    ) -> float:
        if current_time is not None:
            return float(current_time)
        if not cut_windows:
            return 0.0
        return float(max(window.end_time for window in cut_windows))


__all__ = [
    "DecisionEngineConfig",
    "SoftCutDecisionEngine",
    "WindowDecisionContext",
]
