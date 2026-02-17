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
    SplitEvidenceSource,
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

    cut_threshold_base: float = 0.62
    cut_threshold_relax_start_words: int = 12
    cut_threshold_relax_span_words: int = 18
    cut_threshold_relax_max: float = 0.10
    score_bonus_high_level: float = 0.08
    score_bonus_mid_level: float = 0.03
    score_bonus_punctuation_source: float = 0.12
    score_bonus_pause_source: float = 0.04
    score_bonus_speaker_source: float = 0.02
    score_bonus_semantic_source: float = 0.02
    score_bonus_llm_source: float = 0.10
    score_bonus_punctuation_presence: float = 0.08
    deferred_max_wait_sec: float = 2.5
    deferred_max_wait_cap_sec: float = 4.2
    deferred_dynamic_relax_min_words: int = 14
    deferred_dynamic_relax_long_sentence_sec: float = 0.6
    deferred_dynamic_relax_punctuation_sec: float = 0.5
    deferred_force_min_word_boundary_gap: float = 0.08
    hard_limit_trigger_score_weight: float = 0.20
    speaker_left_anchor_prefer_max_gap_sec: float = 0.30
    speaker_right_anchor_tolerance_sec: float = 0.06


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
            "immediate_cut_count": 0,
            "deferred_queued_count": 0,
            "low_queued_count": 0,
            "hard_limit_forced_count": 0,
            "deferred_pending_count": 0,
        }

        pending_deferred = self._resolve_existing_deferred(
            previous_deferred=existing_deferred,
            deferred_records=deferred_records,
        )
        candidate_map: dict[
            str,
            tuple[CutWindow, Optional[AnchorScore], Optional[AnchorScore], WindowDecisionContext],
        ] = {}

        for window in ordered_windows:
            context = contexts.get(window.window_id, WindowDecisionContext())
            scored_anchor = window.candidate_anchors[0] if window.candidate_anchors else None
            selected_anchor = self._select_cut_anchor(
                window=window,
                scored_anchor=scored_anchor,
            )
            candidate_map[str(window.window_id)] = (window, scored_anchor, selected_anchor, context)

            if window.trigger_level == EvidenceLevel.HIGH:
                self._handle_scored_window(
                    window=window,
                    scored_anchor=scored_anchor,
                    selected_anchor=selected_anchor,
                    context=context,
                    decisions=decisions,
                    pending_deferred=pending_deferred,
                    deferred_records=deferred_records,
                    report=report,
                )
                continue

            if window.trigger_level == EvidenceLevel.MID:
                self._handle_scored_window(
                    window=window,
                    scored_anchor=scored_anchor,
                    selected_anchor=selected_anchor,
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
            report["deferred_queued_count"] += 1
            report["low_queued_count"] += 1

        self._apply_hard_limit_force(
            pending_deferred=pending_deferred,
            candidate_map=candidate_map,
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
        deferred_records: list[DeferredCut],
    ) -> list[DeferredCut]:
        pending: list[DeferredCut] = []
        for deferred in previous_deferred:
            if deferred.state != DeferredCutState.PENDING:
                deferred_records.append(deferred)
                continue
            pending.append(deferred)
            deferred_records.append(deferred)
        return pending

    def _handle_scored_window(
        self,
        *,
        window: CutWindow,
        scored_anchor: Optional[AnchorScore],
        selected_anchor: Optional[AnchorScore],
        context: WindowDecisionContext,
        decisions: list[CutDecision],
        pending_deferred: list[DeferredCut],
        deferred_records: list[DeferredCut],
        report: dict[str, int],
    ) -> None:
        if scored_anchor is None:
            pending = self._create_deferred(window, context=context)
            pending_deferred.append(pending)
            deferred_records.append(pending)
            report["deferred_queued_count"] += 1
            return

        if self._should_cut_window(
            window=window,
            anchor=scored_anchor,
            context=context,
        ):
            cut_anchor = selected_anchor or scored_anchor
            decisions.append(
                self._build_anchor_decision(
                    window=window,
                    anchor=cut_anchor,
                    reason=self._resolve_anchor_reason(cut_anchor),
                    risk=None,
                    depends_on_fast_draft=context.depends_on_fast_draft,
                )
            )
            window.state = CutWindowState.RESOLVED
            report["immediate_cut_count"] += 1
            return

        pending = self._create_deferred(window, context=context)
        pending_deferred.append(pending)
        deferred_records.append(pending)
        report["deferred_queued_count"] += 1

    def _apply_hard_limit_force(
        self,
        *,
        pending_deferred: list[DeferredCut],
        candidate_map: dict[
            str,
            tuple[CutWindow, Optional[AnchorScore], Optional[AnchorScore], WindowDecisionContext],
        ],
        contexts: dict[str, WindowDecisionContext],
        decisions: list[CutDecision],
        report: dict[str, int],
        decision_time: float,
    ) -> None:
        if not pending_deferred:
            return
        # Why: 同轮已有稳定切点时，不再叠加硬上限强切，避免“一段文本被多刀”。
        if decisions:
            return
        timeout_indices = [
            index
            for index, item in enumerate(pending_deferred)
            if decision_time >= float(getattr(item, "expected_resolve_by", item.created_at))
        ]
        if not timeout_indices:
            return

        force_index = self._select_hard_limit_target_index(
            pending_deferred=pending_deferred,
            candidate_map=candidate_map,
            candidate_indices=timeout_indices,
        )
        target = pending_deferred.pop(force_index)
        context = contexts.get(target.window_id, WindowDecisionContext())
        depends_on_fast_draft = bool(
            context.depends_on_fast_draft or getattr(target, "depends_on_fast_draft", False)
        )
        candidate = candidate_map.get(str(target.window_id))
        forced_decision: CutDecision
        if candidate is not None and candidate[2] is not None:
            window, _, anchor, _ = candidate
            forced_decision = self._build_anchor_decision(
                window=window,
                anchor=anchor,
                reason="hard_limit_forced",
                risk="hard_limit",
                depends_on_fast_draft=depends_on_fast_draft,
            )
        else:
            forced_decision = self._build_forced_decision(
                window_id=target.window_id,
                time_range=self._resolve_deferred_time_range(target),
                decision_time=decision_time,
                reason="hard_limit_forced",
                risk="hard_limit",
                depends_on_fast_draft=depends_on_fast_draft,
            )
        target.state = DeferredCutState.FORCED
        target.resolution_decision = forced_decision
        decisions.append(forced_decision)
        report["hard_limit_forced_count"] += 1

    def _select_hard_limit_target_index(
        self,
        *,
        pending_deferred: Sequence[DeferredCut],
        candidate_map: dict[
            str,
            tuple[CutWindow, Optional[AnchorScore], Optional[AnchorScore], WindowDecisionContext],
        ],
        candidate_indices: Sequence[int],
    ) -> int:
        candidate_index: Optional[int] = None
        candidate_key: Optional[tuple[float, float]] = None
        for index in candidate_indices:
            if index < 0 or index >= len(pending_deferred):
                continue
            item = pending_deferred[index]
            candidate = candidate_map.get(str(item.window_id))
            if candidate is None or candidate[1] is None:
                score_key = (-1.0, float(item.created_at))
            else:
                window, anchor, _, _ = candidate
                score_key = (
                    self._resolve_hard_limit_priority(window=window, anchor=anchor),
                    float(item.created_at),
                )
            if candidate_key is None or score_key > candidate_key:
                candidate_key = score_key
                candidate_index = index

        if candidate_index is None:
            return 0
        return int(candidate_index)

    def _resolve_hard_limit_priority(
        self,
        *,
        window: CutWindow,
        anchor: AnchorScore,
    ) -> float:
        base_score = self._score_window(window=window, anchor=anchor)
        trigger_score = max(0.0, float(getattr(window, "trigger_score", 0.0)))
        return base_score + (trigger_score * float(self.config.hard_limit_trigger_score_weight))

    def _select_cut_anchor(
        self,
        *,
        window: CutWindow,
        scored_anchor: Optional[AnchorScore],
    ) -> Optional[AnchorScore]:
        if scored_anchor is None:
            return None
        trigger_source = getattr(window, "trigger_source", SplitEvidenceSource.SPEAKER)
        trigger_source_text = str(getattr(trigger_source, "value", trigger_source))
        if trigger_source_text != SplitEvidenceSource.SPEAKER.value:
            return scored_anchor
        return self._select_speaker_anchor(window=window, default_anchor=scored_anchor)

    def _select_speaker_anchor(
        self,
        *,
        window: CutWindow,
        default_anchor: AnchorScore,
    ) -> AnchorScore:
        candidates = list(window.candidate_anchors or [])
        if not candidates:
            return default_anchor

        trigger_time = float(window.trigger_time)
        epsilon = 1e-6
        left_candidates = [
            item for item in candidates if float(item.anchor_time) <= trigger_time + epsilon
        ]
        if left_candidates:
            prefer_gap_sec = max(0.0, float(self.config.speaker_left_anchor_prefer_max_gap_sec))
            near_left_candidates = [
                item
                for item in left_candidates
                if (trigger_time - float(item.anchor_time)) <= prefer_gap_sec
            ]
            target_pool = near_left_candidates or left_candidates
            return min(
                target_pool,
                key=lambda item: (
                    trigger_time - float(item.anchor_time),
                    -float(item.final_score),
                    -float(item.anchor_time),
                ),
            )

        right_tolerance_sec = max(0.0, float(self.config.speaker_right_anchor_tolerance_sec))
        if right_tolerance_sec > 0.0:
            right_candidates = [
                item
                for item in candidates
                if 0.0 < (float(item.anchor_time) - trigger_time) <= (right_tolerance_sec + epsilon)
            ]
            if right_candidates:
                return min(
                    right_candidates,
                    key=lambda item: (
                        float(item.anchor_time) - trigger_time,
                        -float(item.final_score),
                        float(item.anchor_time),
                    ),
                )

        return default_anchor

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
            source=str(getattr(anchor.evidence_source, "value", "") or ""),
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
            source=SplitEvidenceSource.FORCE.value,
        )

    def _create_deferred(
        self,
        window: CutWindow,
        *,
        context: Optional[WindowDecisionContext] = None,
    ) -> DeferredCut:
        active_context = context or WindowDecisionContext()
        max_wait_sec = self._resolve_deferred_max_wait_sec(
            window=window,
            context=active_context,
        )
        return DeferredCut(
            deferred_id=f"{window.window_id}-deferred",
            window_id=window.window_id,
            created_at=float(window.trigger_time),
            expected_resolve_by=float(window.trigger_time + max_wait_sec),
            state=DeferredCutState.PENDING,
            window_start=float(window.start_time),
            window_end=float(window.end_time),
            trigger_level=str(getattr(window.trigger_level, "value", window.trigger_level)),
            depends_on_fast_draft=bool(active_context.depends_on_fast_draft),
        )

    def _resolve_deferred_max_wait_sec(
        self,
        *,
        window: CutWindow,
        context: WindowDecisionContext,
    ) -> float:
        wait_sec = max(0.1, float(self.config.deferred_max_wait_sec))
        long_sentence_words = max(0, int(self.config.deferred_dynamic_relax_min_words))
        if int(context.current_sentence_word_count) >= long_sentence_words:
            wait_sec += max(0.0, float(self.config.deferred_dynamic_relax_long_sentence_sec))
            has_punctuation_anchor = any(
                item.evidence_source == SplitEvidenceSource.PUNCTUATION
                or item.anchor_type == AnchorType.PUNCTUATION_ANCHOR
                for item in list(window.candidate_anchors or [])
            )
            if has_punctuation_anchor:
                wait_sec += max(0.0, float(self.config.deferred_dynamic_relax_punctuation_sec))

        wait_cap = max(0.1, float(self.config.deferred_max_wait_cap_sec))
        return min(wait_sec, wait_cap)

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

    def _resolve_cut_threshold(self, *, current_sentence_word_count: int) -> float:
        start_words = max(0, int(self.config.cut_threshold_relax_start_words))
        span_words = max(1, int(self.config.cut_threshold_relax_span_words))
        relax_max = max(0.0, float(self.config.cut_threshold_relax_max))
        overflow_words = max(0, int(current_sentence_word_count) - start_words)
        relax_ratio = min(1.0, float(overflow_words) / float(span_words))
        return float(self.config.cut_threshold_base) - (relax_ratio * relax_max)

    def _should_cut_window(
        self,
        *,
        window: CutWindow,
        anchor: AnchorScore,
        context: WindowDecisionContext,
    ) -> bool:
        score = self._score_window(window=window, anchor=anchor)
        threshold = self._resolve_cut_threshold(
            current_sentence_word_count=context.current_sentence_word_count,
        )
        return score >= threshold

    def _score_window(
        self,
        *,
        window: CutWindow,
        anchor: AnchorScore,
    ) -> float:
        score = float(anchor.final_score)
        score += self._resolve_level_bonus(window.trigger_level)
        score += self._resolve_source_bonus(anchor)
        has_punctuation_anchor = any(
            item.evidence_source == SplitEvidenceSource.PUNCTUATION
            for item in list(window.candidate_anchors or [])
        )
        if has_punctuation_anchor:
            score += max(0.0, float(self.config.score_bonus_punctuation_presence))
        return max(0.0, score)

    def _resolve_level_bonus(self, level: EvidenceLevel) -> float:
        if level == EvidenceLevel.HIGH:
            return max(0.0, float(self.config.score_bonus_high_level))
        if level == EvidenceLevel.MID:
            return max(0.0, float(self.config.score_bonus_mid_level))
        return 0.0

    def _resolve_source_bonus(self, anchor: AnchorScore) -> float:
        source = getattr(anchor, "evidence_source", None)
        if source == SplitEvidenceSource.PUNCTUATION:
            return max(0.0, float(self.config.score_bonus_punctuation_source))
        if source == SplitEvidenceSource.PAUSE:
            return max(0.0, float(self.config.score_bonus_pause_source))
        if source == SplitEvidenceSource.SEMANTIC:
            return max(0.0, float(self.config.score_bonus_semantic_source))
        if source == SplitEvidenceSource.LLM:
            return max(0.0, float(self.config.score_bonus_llm_source))
        if source == SplitEvidenceSource.SPEAKER:
            return max(0.0, float(self.config.score_bonus_speaker_source))
        return 0.0

    @staticmethod
    def _resolve_anchor_reason(anchor: AnchorScore) -> str:
        source = getattr(anchor, "evidence_source", None)
        if source == SplitEvidenceSource.PAUSE:
            return "pause"
        if source == SplitEvidenceSource.PUNCTUATION:
            return "punctuation"
        if source == SplitEvidenceSource.SEMANTIC:
            return "semantic"
        if source == SplitEvidenceSource.LLM:
            return "llm_semantic"
        if source == SplitEvidenceSource.FORCE:
            return "forced_guard"
        if anchor.anchor_type == AnchorType.PAUSE_ANCHOR:
            return "pause"
        if anchor.anchor_type == AnchorType.PUNCTUATION_ANCHOR:
            return "punctuation"
        if anchor.anchor_type == AnchorType.SEMANTIC_ANCHOR:
            return "semantic"
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
