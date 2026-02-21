"""
软切证据构建器（Phase B）。
V3.2.0+dev.20260220.01
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .types import (
    AnchorScore,
    AnchorType,
    CutWindow,
    EvidenceLevel,
    SoftCutPriorityProfile,
    SpeakerChangeEvidence,
    SpeakerChangeTag,
    SplitEvidenceSource,
    resolve_priority_profile,
)


@dataclass(frozen=True)
class SpeakerChangeFact:
    """说话人变化原始事实（来自 timeline/diarization）。"""

    time: float
    from_speaker: str
    to_speaker: str
    pyannote_confidence: float
    pause_duration: float
    embedding_distance: float
    embedding_threshold: float
    is_abrupt_energy_shift: bool
    tags: set[SpeakerChangeTag] = field(default_factory=set)


@dataclass(frozen=True)
class AnchorCandidate:
    """锚点候选事实（来自停顿/词边界/语义/标点）。"""

    anchor_type: AnchorType
    anchor_time: float
    source: str
    confidence: Optional[float] = None
    base_score_override: Optional[float] = None


@dataclass
class EvidenceBuilderConfig:
    """证据构建参数。"""

    speaker_change_high_threshold: float = 0.80
    speaker_change_mid_threshold: float = 0.50
    pause_bonus_threshold_sec: float = 0.30
    pause_bonus: float = 0.15
    embedding_bonus: float = 0.20
    energy_bonus: float = 0.05
    debounce_cooldown_ms: int = 300
    window_before_sec: float = 0.50
    window_after_sec: float = 0.30
    anchor_window_before_sec: float = 0.35
    anchor_window_after_sec: float = 0.25
    anchor_trigger_high_threshold: float = 0.72
    anchor_trigger_mid_threshold: float = 0.45
    anchor_distance_penalty_factor: float = 0.10
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
class EvidenceBuildResult:
    """证据构建输出。"""

    speaker_changes: list[SpeakerChangeEvidence]
    cut_windows: list[CutWindow]
    generation_report: dict[str, float | int | str] = field(default_factory=dict)


class EvidenceBuilder:
    """
    证据构建器（Builder Pattern）。

    Why:
    - 把“事实 -> 证据 -> 窗口候选”的转换收口在单点，避免 pipeline 中散落评分口径。
    - Phase A 扩展：支持非 speaker 来源（标点/语义）独立触发窗口，避免冲突路径吞切分。
    """

    def __init__(self, config: Optional[EvidenceBuilderConfig] = None) -> None:
        self.config = config or EvidenceBuilderConfig()
        self._priority_profile: SoftCutPriorityProfile = resolve_priority_profile(
            active_profile=self.config.priority_active_profile,
            profile_overrides=self.config.priority_profiles,
        )

    def build(
        self,
        *,
        chunk_id: str,
        chunk_start: float,
        chunk_end: float,
        speaker_change_facts: Sequence[SpeakerChangeFact],
        anchor_candidates: Sequence[AnchorCandidate],
    ) -> EvidenceBuildResult:
        """构建分级证据、窗口与窗口内锚点候选。"""
        classified_changes = self.build_speaker_change_evidences(speaker_change_facts)
        debounced_changes = self.debounce_speaker_changes(classified_changes)
        speaker_windows = self.build_cut_windows(
            chunk_id=chunk_id,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            speaker_changes=debounced_changes,
        )
        anchor_trigger_windows = self.build_anchor_trigger_windows(
            chunk_id=chunk_id,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            anchor_candidates=anchor_candidates,
        )
        merged_windows = self._resolve_mixed_window_conflicts(
            windows=[*speaker_windows, *anchor_trigger_windows]
        )
        self.attach_anchor_scores(
            cut_windows=merged_windows,
            anchor_candidates=anchor_candidates,
        )
        generation_report = self._build_generation_report(
            input_change_count=len(speaker_change_facts),
            output_changes=debounced_changes,
            speaker_window_count=len(speaker_windows),
            anchor_window_count=len(anchor_trigger_windows),
            merged_window_count=len(merged_windows),
            anchor_candidate_count=len(anchor_candidates),
        )
        return EvidenceBuildResult(
            speaker_changes=debounced_changes,
            cut_windows=merged_windows,
            generation_report=generation_report,
        )

    def build_speaker_change_evidences(
        self,
        speaker_change_facts: Sequence[SpeakerChangeFact],
    ) -> list[SpeakerChangeEvidence]:
        """按统一评分口径把原始事实转为 high/mid/low 证据。"""
        evidences: list[SpeakerChangeEvidence] = []
        for fact in sorted(speaker_change_facts, key=lambda item: item.time):
            level = self.classify_speaker_change(fact)
            evidences.append(
                SpeakerChangeEvidence(
                    time=float(fact.time),
                    from_speaker=fact.from_speaker,
                    to_speaker=fact.to_speaker,
                    pyannote_confidence=float(fact.pyannote_confidence),
                    pause_duration=float(fact.pause_duration),
                    embedding_distance=float(fact.embedding_distance),
                    embedding_threshold=float(fact.embedding_threshold),
                    is_abrupt_energy_shift=bool(fact.is_abrupt_energy_shift),
                    level=level,
                    tags=set(fact.tags),
                )
            )
        return evidences

    def classify_speaker_change(self, fact: SpeakerChangeFact) -> EvidenceLevel:
        """speaker_change 分级。"""
        score = float(fact.pyannote_confidence)
        if fact.pause_duration > self.config.pause_bonus_threshold_sec:
            score += self.config.pause_bonus
        if fact.embedding_distance > fact.embedding_threshold:
            score += self.config.embedding_bonus
        if fact.is_abrupt_energy_shift:
            score += self.config.energy_bonus

        if score >= self.config.speaker_change_high_threshold:
            return EvidenceLevel.HIGH
        if score >= self.config.speaker_change_mid_threshold:
            return EvidenceLevel.MID
        return EvidenceLevel.LOW

    def debounce_speaker_changes(
        self,
        speaker_changes: Sequence[SpeakerChangeEvidence],
    ) -> list[SpeakerChangeEvidence]:
        """去抖：短间隔 A->B->A 回弹合并，快速对话保留。"""
        ordered = sorted(speaker_changes, key=lambda item: item.time)
        if not ordered:
            return []

        cooldown_sec = max(0.0, self.config.debounce_cooldown_ms / 1000.0)
        result: list[SpeakerChangeEvidence] = []
        for change in ordered:
            if not result:
                result.append(change)
                continue

            last = result[-1]
            gap_sec = max(0.0, change.time - last.time)
            is_within_cooldown = gap_sec < cooldown_sec
            if not is_within_cooldown:
                result.append(change)
                continue

            is_rebound = (
                change.from_speaker == last.to_speaker
                and change.to_speaker == last.from_speaker
            )
            if is_rebound:
                last.tags.add(SpeakerChangeTag.REBOUND_MERGED)
                continue

            # 非回弹短间隔按“真实快速对话”保留。
            result.append(change)
        return result

    def build_cut_windows(
        self,
        *,
        chunk_id: str,
        chunk_start: float,
        chunk_end: float,
        speaker_changes: Sequence[SpeakerChangeEvidence],
    ) -> list[CutWindow]:
        """围绕 speaker_change 构建待切窗口，并做冲突合并。"""
        if chunk_end <= chunk_start:
            return []

        window_pairs: list[tuple[CutWindow, SpeakerChangeEvidence]] = []
        ordered_changes = sorted(speaker_changes, key=lambda item: item.time)
        for index, change in enumerate(ordered_changes):
            raw_start = change.time - self.config.window_before_sec
            raw_end = change.time + self.config.window_after_sec
            start_time = max(chunk_start, raw_start)
            end_time = min(chunk_end, raw_end)
            if end_time <= start_time:
                continue

            is_cross_chunk = raw_start < chunk_start or raw_end > chunk_end
            if is_cross_chunk:
                change.tags.add(SpeakerChangeTag.CROSS_CHUNK_DEFERRED)

            window_pairs.append(
                (
                    CutWindow(
                        window_id=f"{chunk_id}-w-{index:04d}",
                        trigger_time=change.time,
                        trigger_level=change.level,
                        start_time=start_time,
                        end_time=end_time,
                        chunk_id=chunk_id,
                        trigger_source=SplitEvidenceSource.SPEAKER,
                        trigger_score=max(
                            0.0,
                            min(1.0, float(change.pyannote_confidence)),
                        ),
                    ),
                    change,
                )
            )

        resolved_pairs = self._resolve_speaker_window_conflicts(window_pairs)
        return [window for window, _ in resolved_pairs]

    def build_anchor_trigger_windows(
        self,
        *,
        chunk_id: str,
        chunk_start: float,
        chunk_end: float,
        anchor_candidates: Sequence[AnchorCandidate],
    ) -> list[CutWindow]:
        """
        基于非 speaker 证据触发窗口。

        Why:
        - 解决“无 speaker_change 时标点/语义证据无法触发切分”的主链缺口。
        """
        if chunk_end <= chunk_start:
            return []

        windows: list[CutWindow] = []
        index = 0
        for candidate in sorted(anchor_candidates, key=lambda item: item.anchor_time):
            source = self._resolve_candidate_source(candidate)
            # Why: 词边界默认仅作落点候选；fast_draft 词边界是显式切点，可作为触发窗口。
            if (
                candidate.anchor_type == AnchorType.WORD_BOUNDARY
                and source != SplitEvidenceSource.FAST_DRAFT
            ):
                continue
            rule = self._priority_profile.source_rule(source)
            if not rule.enabled:
                continue

            confidence = self._resolve_candidate_confidence(candidate)
            if confidence < rule.min_confidence:
                continue

            trigger_score = confidence * rule.weight
            if trigger_score < rule.trigger_threshold:
                continue

            trigger_time = float(candidate.anchor_time)
            if trigger_time < chunk_start or trigger_time > chunk_end:
                continue
            raw_start = trigger_time - self.config.anchor_window_before_sec
            raw_end = trigger_time + self.config.anchor_window_after_sec
            start_time = max(chunk_start, raw_start)
            end_time = min(chunk_end, raw_end)
            if end_time <= start_time:
                continue

            windows.append(
                CutWindow(
                    window_id=f"{chunk_id}-a-{source.value}-{index:04d}",
                    trigger_time=trigger_time,
                    trigger_level=self._build_anchor_trigger_level(trigger_score),
                    start_time=start_time,
                    end_time=end_time,
                    chunk_id=chunk_id,
                    trigger_source=source,
                    trigger_score=trigger_score,
                )
            )
            index += 1
        return windows

    def attach_anchor_scores(
        self,
        *,
        cut_windows: Sequence[CutWindow],
        anchor_candidates: Sequence[AnchorCandidate],
    ) -> None:
        """为每个窗口附着候选锚点评分。"""
        ordered_candidates = sorted(anchor_candidates, key=lambda item: item.anchor_time)
        for window in cut_windows:
            scores: list[AnchorScore] = []
            for candidate in ordered_candidates:
                is_in_window = window.start_time <= candidate.anchor_time <= window.end_time
                if not is_in_window:
                    continue
                scores.append(
                    self._build_anchor_score(
                        trigger_time=window.trigger_time,
                        candidate=candidate,
                        window_trigger_source=window.trigger_source,
                    )
                )
            scores.sort(
                key=lambda item: (
                    -item.final_score,
                    self._priority_profile.source_rank(item.evidence_source),
                    abs(item.anchor_time - window.trigger_time),
                    item.anchor_time,
                )
            )
            window.candidate_anchors = scores

    def _build_anchor_score(
        self,
        *,
        trigger_time: float,
        candidate: AnchorCandidate,
        window_trigger_source: SplitEvidenceSource,
    ) -> AnchorScore:
        source = self._resolve_candidate_source(
            candidate,
            fallback_source=window_trigger_source,
        )
        base_score = self._resolve_anchor_base_score(candidate)
        distance_penalty = abs(candidate.anchor_time - trigger_time) * self.config.anchor_distance_penalty_factor
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

    def _resolve_anchor_base_score(self, candidate: AnchorCandidate) -> float:
        if candidate.base_score_override is not None:
            score = float(candidate.base_score_override)
        else:
            score = float(self.config.anchor_base_scores.get(candidate.anchor_type, 0.0))

        if candidate.confidence is None:
            return max(0.0, score)

        clamped_confidence = max(0.0, min(1.0, float(candidate.confidence)))
        return max(0.0, score * clamped_confidence)

    def _resolve_speaker_window_conflicts(
        self,
        window_pairs: Sequence[tuple[CutWindow, SpeakerChangeEvidence]],
    ) -> list[tuple[CutWindow, SpeakerChangeEvidence]]:
        if not window_pairs:
            return []

        ordered = sorted(window_pairs, key=lambda item: item[0].trigger_time)
        resolved: list[tuple[CutWindow, SpeakerChangeEvidence]] = []
        for window, evidence in ordered:
            if not resolved:
                resolved.append((window, evidence))
                continue

            prev_window, prev_evidence = resolved[-1]
            is_overlap = window.start_time < prev_window.end_time and prev_window.start_time < window.end_time
            if not is_overlap:
                resolved.append((window, evidence))
                continue

            current_priority = self._level_priority(evidence.level)
            previous_priority = self._level_priority(prev_evidence.level)
            if current_priority > previous_priority:
                evidence.tags.add(SpeakerChangeTag.SUSPECTED_MISSED)
                resolved[-1] = (window, evidence)
                continue
            if current_priority < previous_priority:
                prev_evidence.tags.add(SpeakerChangeTag.SUSPECTED_MISSED)
                continue

            # 同级冲突按 pyannote 置信度优先，其余场景保持先到先得。
            if evidence.pyannote_confidence > prev_evidence.pyannote_confidence:
                evidence.tags.add(SpeakerChangeTag.SUSPECTED_MISSED)
                resolved[-1] = (window, evidence)
            else:
                prev_evidence.tags.add(SpeakerChangeTag.SUSPECTED_MISSED)
        return resolved

    def _resolve_mixed_window_conflicts(
        self,
        *,
        windows: Sequence[CutWindow],
    ) -> list[CutWindow]:
        if not windows:
            return []
        ordered = sorted(windows, key=lambda item: (item.start_time, item.trigger_time, item.window_id))
        resolved: list[CutWindow] = []
        merge_gap_sec = max(0.02, float(self._priority_profile.merge_window_ms) / 1000.0)
        for current in ordered:
            if not resolved:
                resolved.append(current)
                continue
            previous = resolved[-1]
            is_overlap = current.start_time <= previous.end_time and previous.start_time <= current.end_time
            is_near = abs(float(current.trigger_time) - float(previous.trigger_time)) <= merge_gap_sec
            if not is_overlap and not is_near:
                resolved.append(current)
                continue

            preferred = self._pick_preferred_window(previous, current)
            if preferred is previous:
                previous.start_time = min(previous.start_time, current.start_time)
                previous.end_time = max(previous.end_time, current.end_time)
                continue
            current.start_time = min(previous.start_time, current.start_time)
            current.end_time = max(previous.end_time, current.end_time)
            resolved[-1] = current
        return resolved

    def _pick_preferred_window(self, left: CutWindow, right: CutWindow) -> CutWindow:
        left_level = self._level_priority(left.trigger_level)
        right_level = self._level_priority(right.trigger_level)
        if left_level != right_level:
            return left if left_level > right_level else right
        if abs(float(left.trigger_score) - float(right.trigger_score)) > 1e-9:
            return left if left.trigger_score > right.trigger_score else right
        left_rank = self._priority_profile.source_rank(left.trigger_source)
        right_rank = self._priority_profile.source_rank(right.trigger_source)
        if left_rank != right_rank:
            return left if left_rank < right_rank else right
        return left if left.trigger_time <= right.trigger_time else right

    @staticmethod
    def _level_priority(level: EvidenceLevel) -> int:
        if level == EvidenceLevel.HIGH:
            return 3
        if level == EvidenceLevel.MID:
            return 2
        return 1

    def _build_anchor_trigger_level(self, trigger_score: float) -> EvidenceLevel:
        if trigger_score >= self.config.anchor_trigger_high_threshold:
            return EvidenceLevel.HIGH
        if trigger_score >= self.config.anchor_trigger_mid_threshold:
            return EvidenceLevel.MID
        return EvidenceLevel.LOW

    @staticmethod
    def _resolve_candidate_confidence(candidate: AnchorCandidate) -> float:
        if candidate.confidence is None:
            return 0.0
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
        # Why: WORD_BOUNDARY 只是落点类型，来源应继承当前窗口触发来源，避免归因为 speaker_change。
        if candidate.anchor_type == AnchorType.WORD_BOUNDARY and fallback_source is not None:
            return fallback_source
        return SplitEvidenceSource.SPEAKER

    def _build_generation_report(
        self,
        *,
        input_change_count: int,
        output_changes: Sequence[SpeakerChangeEvidence],
        speaker_window_count: int,
        anchor_window_count: int,
        merged_window_count: int,
        anchor_candidate_count: int,
    ) -> dict[str, float | int | str]:
        high_count = sum(1 for item in output_changes if item.level == EvidenceLevel.HIGH)
        mid_count = sum(1 for item in output_changes if item.level == EvidenceLevel.MID)
        low_count = sum(1 for item in output_changes if item.level == EvidenceLevel.LOW)
        return {
            "input_change_count": int(input_change_count),
            "output_change_count": int(len(output_changes)),
            "high_count": int(high_count),
            "mid_count": int(mid_count),
            "low_count": int(low_count),
            "speaker_window_count": int(speaker_window_count),
            "anchor_window_count": int(anchor_window_count),
            "window_count": int(merged_window_count),
            "anchor_candidate_count": int(anchor_candidate_count),
            "priority_profile": self._priority_profile.profile_name,
        }


__all__ = [
    "AnchorCandidate",
    "EvidenceBuildResult",
    "EvidenceBuilder",
    "EvidenceBuilderConfig",
    "SpeakerChangeFact",
]
