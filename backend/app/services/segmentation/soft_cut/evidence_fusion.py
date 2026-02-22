"""
软切证据融合服务（Phase C）。
V3.2.0+dev.20260220.01
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Dict, Mapping, Optional, Sequence

from app.services.alignment.types import FusedEvidence
from .evidence_builder import AnchorCandidate
from .types import (
    AnchorScore,
    AnchorType,
    CutWindow,
    EvidenceLevel,
    SoftCutPriorityProfile,
    SourcePriorityRule,
    SplitEvidenceSource,
    resolve_priority_profile,
)


@dataclass
class EvidenceFusionConfig:
    """证据融合参数。"""

    anchor_distance_penalty_factor_m1: float = 0.10
    anchor_distance_penalty_factor_m2: float = 0.06
    word_time_confidence_weight: float = 0.35
    priority_active_profile: str = "punct_boost_transition"
    priority_profiles: dict[str, dict[str, object]] = field(default_factory=dict)
    anchor_base_scores: dict[AnchorType, float] = field(
        default_factory=lambda: {
            AnchorType.PAUSE_ANCHOR: 0.8,
            AnchorType.WORD_BOUNDARY: 0.6,
            AnchorType.SEMANTIC_ANCHOR: 0.5,
            AnchorType.PUNCTUATION_ANCHOR: 0.3,
        }
    )


@dataclass
class EvidenceFusionResult:
    """融合输出。"""

    fused_windows: list[CutWindow]
    generation_report: dict[str, float | int | str | dict[str, Any]] = field(default_factory=dict)


class EvidenceFusion:
    """
    证据融合器（Facade Pattern）。

    Why:
    - 把多来源锚点融合与重评分收口在单点，避免决策层依赖各来源细节。
    - Phase A 扩展：统一来源优先级配置，实现“调权不改代码”。
    """

    def __init__(self, config: Optional[EvidenceFusionConfig] = None) -> None:
        self.config = config or EvidenceFusionConfig()
        self._priority_profile: SoftCutPriorityProfile = resolve_priority_profile(
            active_profile=self.config.priority_active_profile,
            profile_overrides=self.config.priority_profiles,
        )

    def fuse(
        self,
        *,
        cut_windows: Sequence[CutWindow],
        pause_anchors: Sequence[AnchorCandidate] = (),
        word_anchors: Sequence[AnchorCandidate] = (),
        semantic_anchors: Sequence[AnchorCandidate] = (),
        punctuation_anchors: Sequence[AnchorCandidate] = (),
        time_axis_version: str = "m1_legacy",
        word_time_confidence: Optional[Sequence[float]] = None,
    ) -> EvidenceFusionResult:
        """融合多来源锚点并输出按优先级收敛后的窗口。"""
        normalized_windows = [self._clone_window(window) for window in cut_windows]
        merged_windows, merged_window_count = self._resolve_window_conflicts(normalized_windows)

        all_new_candidates = [
            *pause_anchors,
            *word_anchors,
            *semantic_anchors,
            *punctuation_anchors,
        ]

        total_output_anchors = 0
        source_score_report: dict[str, dict[str, float | int]] = {}
        for window in merged_windows:
            existing_candidates = self._convert_existing_scores_to_candidates(window.candidate_anchors)
            combined_candidates = [*existing_candidates, *all_new_candidates]
            window.candidate_anchors = self._build_fused_anchor_scores(
                trigger_time=window.trigger_time,
                start_time=window.start_time,
                end_time=window.end_time,
                candidates=combined_candidates,
                fallback_source=window.trigger_source,
                time_axis_version=time_axis_version,
                word_time_confidence=word_time_confidence,
                source_score_report=source_score_report,
            )
            total_output_anchors += len(window.candidate_anchors)

        generation_report = {
            "input_window_count": int(len(cut_windows)),
            "output_window_count": int(len(merged_windows)),
            "merged_window_count": int(merged_window_count),
            "input_anchor_count": int(
                sum(len(item.candidate_anchors) for item in cut_windows) + len(all_new_candidates)
            ),
            "output_anchor_count": int(total_output_anchors),
            "time_axis_version": time_axis_version,
            "has_word_time_confidence": int(bool(word_time_confidence)),
            "priority_profile": self._priority_profile.profile_name,
            "source_score_report": source_score_report,
        }
        return EvidenceFusionResult(
            fused_windows=merged_windows,
            generation_report=generation_report,
        )

    def _build_fused_anchor_scores(
        self,
        *,
        trigger_time: float,
        start_time: float,
        end_time: float,
        candidates: Sequence[AnchorCandidate],
        fallback_source: SplitEvidenceSource,
        time_axis_version: str,
        word_time_confidence: Optional[Sequence[float]],
        source_score_report: dict[str, dict[str, float | int]],
    ) -> list[AnchorScore]:
        scores: list[AnchorScore] = []
        seen: set[tuple[str, int, str]] = set()
        for candidate in candidates:
            if candidate.anchor_time < start_time or candidate.anchor_time > end_time:
                continue
            source = self._resolve_candidate_source(
                candidate,
                fallback_source=fallback_source,
            )
            rule = self._priority_profile.source_rule(source)
            confidence = self._resolve_candidate_confidence(candidate)
            if not rule.enabled:
                self._append_source_report(source_score_report, source=source, skipped_disabled=1)
                continue
            if confidence < rule.min_confidence:
                self._append_source_report(source_score_report, source=source, skipped_low_conf=1)
                continue

            score = self._score_anchor_candidate(
                trigger_time=trigger_time,
                candidate=candidate,
                source=source,
                source_rule=rule,
                time_axis_version=time_axis_version,
                word_time_confidence=word_time_confidence,
            )
            dedupe_key = (
                score.anchor_type.value,
                int(round(score.anchor_time * 1000)),
                score.source,
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            scores.append(score)
            self._append_source_report(
                source_score_report,
                source=source,
                accepted=1,
                total_score=score.final_score,
            )

        return sorted(
            scores,
            key=lambda item: (
                -item.final_score,
                self._priority_profile.source_rank(item.evidence_source),
                abs(item.anchor_time - trigger_time),
                item.anchor_time,
            ),
        )

    def _score_anchor_candidate(
        self,
        *,
        trigger_time: float,
        candidate: AnchorCandidate,
        source: SplitEvidenceSource,
        source_rule: SourcePriorityRule,
        time_axis_version: str,
        word_time_confidence: Optional[Sequence[float]],
    ) -> AnchorScore:
        base_score = float(
            candidate.base_score_override
            if candidate.base_score_override is not None
            else self.config.anchor_base_scores.get(candidate.anchor_type, 0.0)
        )

        confidence = self._resolve_candidate_confidence(candidate)
        base_score *= confidence
        base_score *= max(0.0, min(1.0, float(source_rule.weight)))

        if word_time_confidence:
            clamped = [max(0.0, min(1.0, float(value))) for value in word_time_confidence]
            mean_confidence = float(fmean(clamped))
            weight = max(0.0, min(1.0, self.config.word_time_confidence_weight))
            base_score *= (1.0 - weight) + (mean_confidence * weight)

        distance_penalty_factor = (
            self.config.anchor_distance_penalty_factor_m2
            if time_axis_version == "m2_nw_v2"
            else self.config.anchor_distance_penalty_factor_m1
        )
        distance_penalty = abs(candidate.anchor_time - trigger_time) * distance_penalty_factor
        final_score = max(0.0, base_score - distance_penalty)
        return AnchorScore(
            anchor_type=candidate.anchor_type,
            anchor_time=float(candidate.anchor_time),
            base_score=float(base_score),
            distance_penalty=float(distance_penalty),
            final_score=float(final_score),
            source=candidate.source,
            evidence_source=source,
        )

    def _resolve_window_conflicts(
        self,
        windows: Sequence[CutWindow],
    ) -> tuple[list[CutWindow], int]:
        if not windows:
            return [], 0

        ordered = sorted(windows, key=lambda item: (item.start_time, item.trigger_time))
        resolved: list[CutWindow] = []
        merged_count = 0
        for current in ordered:
            if not resolved:
                resolved.append(current)
                continue

            previous = resolved[-1]
            is_overlap = current.start_time < previous.end_time and previous.start_time < current.end_time
            if not is_overlap:
                resolved.append(current)
                continue

            merged_count += 1
            preferred = self._pick_preferred_window(previous, current)
            if preferred is current:
                current.candidate_anchors.extend(previous.candidate_anchors)
                current.start_time = min(current.start_time, previous.start_time)
                current.end_time = max(current.end_time, previous.end_time)
                resolved[-1] = current
            else:
                previous.candidate_anchors.extend(current.candidate_anchors)
                previous.start_time = min(previous.start_time, current.start_time)
                previous.end_time = max(previous.end_time, current.end_time)
                resolved[-1] = previous
        return resolved, merged_count

    def _pick_preferred_window(self, left: CutWindow, right: CutWindow) -> CutWindow:
        left_priority = self._level_priority(left.trigger_level)
        right_priority = self._level_priority(right.trigger_level)
        if left_priority != right_priority:
            return left if left_priority > right_priority else right

        if abs(float(left.trigger_score) - float(right.trigger_score)) > 1e-9:
            return left if left.trigger_score > right.trigger_score else right

        left_rank = self._priority_profile.source_rank(left.trigger_source)
        right_rank = self._priority_profile.source_rank(right.trigger_source)
        if left_rank != right_rank:
            return left if left_rank < right_rank else right

        left_best = left.candidate_anchors[0].final_score if left.candidate_anchors else 0.0
        right_best = right.candidate_anchors[0].final_score if right.candidate_anchors else 0.0
        if left_best != right_best:
            return left if left_best > right_best else right
        return left if left.trigger_time <= right.trigger_time else right

    @staticmethod
    def _level_priority(level: EvidenceLevel) -> int:
        if level == EvidenceLevel.HIGH:
            return 3
        if level == EvidenceLevel.MID:
            return 2
        return 1

    @staticmethod
    def _clone_window(window: CutWindow) -> CutWindow:
        return CutWindow(
            window_id=window.window_id,
            trigger_time=window.trigger_time,
            trigger_level=window.trigger_level,
            start_time=window.start_time,
            end_time=window.end_time,
            chunk_id=window.chunk_id,
            trigger_source=window.trigger_source,
            trigger_score=window.trigger_score,
            candidate_anchors=list(window.candidate_anchors),
            state=window.state,
        )

    @staticmethod
    def _convert_existing_scores_to_candidates(
        scores: Sequence[AnchorScore],
    ) -> list[AnchorCandidate]:
        return [
            AnchorCandidate(
                anchor_type=item.anchor_type,
                anchor_time=item.anchor_time,
                source=item.source,
                base_score_override=item.base_score,
            )
            for item in scores
        ]

    @staticmethod
    def _resolve_candidate_confidence(candidate: AnchorCandidate) -> float:
        if candidate.confidence is None:
            return 1.0
        return max(0.0, min(1.0, float(candidate.confidence)))

    @staticmethod
    def _resolve_candidate_source(
        candidate: AnchorCandidate,
        *,
        fallback_source: Optional[SplitEvidenceSource] = None,
    ) -> SplitEvidenceSource:
        source_text = str(candidate.source or "").strip().lower()
        if "llm" in source_text:
            return SplitEvidenceSource.LLM
        if "punct" in source_text:
            return SplitEvidenceSource.PUNCTUATION
        if "semantic" in source_text:
            return SplitEvidenceSource.SEMANTIC
        if "fast_draft" in source_text or "fastdraft" in source_text:
            return SplitEvidenceSource.FAST_DRAFT
        if "pause" in source_text:
            return SplitEvidenceSource.PAUSE
        if "speaker" in source_text:
            return SplitEvidenceSource.SPEAKER
        if candidate.anchor_type == AnchorType.PAUSE_ANCHOR:
            return SplitEvidenceSource.PAUSE
        if candidate.anchor_type == AnchorType.PUNCTUATION_ANCHOR:
            return SplitEvidenceSource.PUNCTUATION
        if candidate.anchor_type == AnchorType.SEMANTIC_ANCHOR:
            return SplitEvidenceSource.SEMANTIC
        # Why: WORD_BOUNDARY 是落点类型而非来源，优先继承触发窗口来源，避免 split_reason 漂移。
        if candidate.anchor_type == AnchorType.WORD_BOUNDARY and fallback_source is not None:
            return fallback_source
        return SplitEvidenceSource.SPEAKER

    @staticmethod
    def _append_source_report(
        report: dict[str, dict[str, float | int]],
        *,
        source: SplitEvidenceSource,
        accepted: int = 0,
        skipped_disabled: int = 0,
        skipped_low_conf: int = 0,
        total_score: float = 0.0,
    ) -> None:
        key = source.value
        item = report.setdefault(
            key,
            {
                "accepted": 0,
                "skipped_disabled": 0,
                "skipped_low_conf": 0,
                "total_score": 0.0,
            },
        )
        item["accepted"] = int(item.get("accepted", 0)) + int(accepted)
        item["skipped_disabled"] = int(item.get("skipped_disabled", 0)) + int(skipped_disabled)
        item["skipped_low_conf"] = int(item.get("skipped_low_conf", 0)) + int(skipped_low_conf)
        item["total_score"] = float(item.get("total_score", 0.0)) + float(total_score)

    def to_fused_evidence(
        self,
        *,
        speaker_changes: Sequence[Dict[str, Any]],
        pause_anchors: Sequence[Dict[str, Any]],
        semantic_anchors: Sequence[Dict[str, Any]],
        punctuation_anchors: Sequence[Dict[str, Any]],
        generation_report: Optional[Dict[str, Any]] = None,
    ) -> FusedEvidence:
        """将融合结果封装为阶段4统一契约。"""
        return FusedEvidence(
            speaker_changes=list(speaker_changes or []),
            pause_anchors=list(pause_anchors or []),
            semantic_anchors=list(semantic_anchors or []),
            punctuation_anchors=list(punctuation_anchors or []),
            evidence_report=dict(generation_report or {}),
        )


__all__ = [
    "EvidenceFusion",
    "EvidenceFusionConfig",
    "EvidenceFusionResult",
]
