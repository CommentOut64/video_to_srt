"""
软切决策引擎（Phase C）。
V3.2.0+dev.20260220.01
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
    EvidenceLevel,
    SplitEvidenceSource,
)


@dataclass
class WindowDecisionContext:
    """窗口决策上下文。"""

    current_sentence_word_count: int = 0
    waiting_word_count: int = 0
    depends_on_fast_draft: bool = False
    cut_threshold_override: Optional[float] = None
    is_allow_low_level_semantic: bool = False
    allow_low_level_sources: frozenset[SplitEvidenceSource] = frozenset()


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
    score_bonus_fast_draft_source: float = 0.08
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
    - 把 high/mid/low 的即时裁决规则集中管理，避免分支散落。
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
        _ = previous_deferred  # 兼容旧调用签名：deferred 输入已停用。
        ordered_windows = sorted(cut_windows, key=lambda item: (item.trigger_time, item.window_id))
        decision_time = self._resolve_decision_time(cut_windows=ordered_windows, current_time=current_time)

        decisions: list[CutDecision] = []
        report = {
            "window_count": int(len(ordered_windows)),
            "decision_count": 0,
            "immediate_cut_count": 0,
            "deferred_queued_count": 0,  # 兼容字段：固定 0
            "low_queued_count": 0,  # 兼容字段：固定 0
            "hard_limit_forced_count": 0,  # 兼容字段：固定 0
            "deferred_pending_count": 0,  # 兼容字段：固定 0
            "skipped_no_anchor_count": 0,
            "skipped_below_threshold_count": 0,
            "skipped_low_level_count": 0,
        }

        for window in ordered_windows:
            context = contexts.get(window.window_id, WindowDecisionContext())
            scored_anchor = window.candidate_anchors[0] if window.candidate_anchors else None
            selected_anchor = self._select_cut_anchor(
                window=window,
                scored_anchor=scored_anchor,
            )
            if (
                window.trigger_level == EvidenceLevel.LOW
                and not self._is_allow_low_level_window(window=window, context=context)
            ):
                report["skipped_low_level_count"] += 1
                continue
            self._handle_scored_window(
                window=window,
                scored_anchor=scored_anchor,
                selected_anchor=selected_anchor,
                context=context,
                decisions=decisions,
                report=report,
            )

        decisions.sort(key=lambda item: (item.time, item.window_id))
        report["decision_count"] = int(len(decisions))
        plan = CutPlan(
            plan_id=f"{block_id}-plan-{int(decision_time * 1000):010d}",
            block_id=block_id,
            decisions=decisions,
            deferred_cuts=[],
            generation_report=report,
        )
        return plan

    def _handle_scored_window(
        self,
        *,
        window: CutWindow,
        scored_anchor: Optional[AnchorScore],
        selected_anchor: Optional[AnchorScore],
        context: WindowDecisionContext,
        decisions: list[CutDecision],
        report: dict[str, int],
    ) -> None:
        if scored_anchor is None:
            report["skipped_no_anchor_count"] += 1
            return

        if self._should_cut_window(window=window, anchor=scored_anchor, context=context):
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

        report["skipped_below_threshold_count"] += 1

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

    def _resolve_cut_threshold(
        self,
        *,
        current_sentence_word_count: int,
        cut_threshold_override: Optional[float] = None,
    ) -> float:
        base_threshold = float(self.config.cut_threshold_base)
        if cut_threshold_override is not None:
            base_threshold = max(0.0, min(1.0, float(cut_threshold_override)))
        start_words = max(0, int(self.config.cut_threshold_relax_start_words))
        span_words = max(1, int(self.config.cut_threshold_relax_span_words))
        relax_max = max(0.0, float(self.config.cut_threshold_relax_max))
        overflow_words = max(0, int(current_sentence_word_count) - start_words)
        relax_ratio = min(1.0, float(overflow_words) / float(span_words))
        return base_threshold - (relax_ratio * relax_max)

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
            cut_threshold_override=context.cut_threshold_override,
        )
        return score >= threshold

    @staticmethod
    def _is_allow_low_level_window(
        *,
        window: CutWindow,
        context: WindowDecisionContext,
    ) -> bool:
        source = getattr(window, "trigger_source", None)
        if context.allow_low_level_sources:
            return source in set(context.allow_low_level_sources)
        if not bool(context.is_allow_low_level_semantic):
            return False
        return source in {SplitEvidenceSource.SEMANTIC, SplitEvidenceSource.LLM}

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
        if source == SplitEvidenceSource.FAST_DRAFT:
            return max(0.0, float(self.config.score_bonus_fast_draft_source))
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
        if source == SplitEvidenceSource.FAST_DRAFT:
            return "fast_draft"
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
