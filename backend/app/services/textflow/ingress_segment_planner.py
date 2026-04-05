from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from app.models.sensevoice_models import WordTimestamp
from app.services.alignment.types import DecisionLayerInput
from app.services.segmentation.soft_cut.types import AnchorType, CutDecision, CutPlan
from app.services.text_protection import is_sentence_end_punct


@dataclass(frozen=True)
class _BoundaryFeature:
    split_idx: int
    event_time: float
    left_end: float
    right_start: float
    reasons: tuple[str, ...]
    scores: tuple[float, ...]


@dataclass(frozen=True)
class _SegmentCandidate:
    split_idx: int
    event_time: float
    left_end: float
    right_start: float
    left_text: str
    right_text: str
    gap_sec: float
    duration_sec: float
    word_count: int
    reasons: tuple[str, ...]
    reason: str
    anchor_type: AnchorType
    closure_score: float
    score: float
    constraint_only: bool
    short_fragment_hit: bool
    overflow_sec: float
    pause_only: bool


@dataclass(frozen=True)
class _TailHoldCandidate:
    segment_start_idx: int
    segment_end_idx: int
    duration_sec: float
    word_count: int
    score: float
    reason: str = "tail_hold"


class IngressSegmentPlanner:
    """timeanchored 入口的滚动式 segment planner。"""

    _DEFAULT_MAX_SEGMENT_SEC = 5.8
    _CJK_MAX_SEGMENT_SEC = 6.2
    _DEFAULT_MAX_OVERFLOW_SEC = 0.35
    _CJK_MAX_OVERFLOW_SEC = 0.45
    _DEFAULT_ACCEPT_THRESHOLD = 0.55
    _CJK_ACCEPT_THRESHOLD = 0.48
    _ENGLISH_TAIL_HOLD_MARGIN = 0.08
    _SENTENCE_END_PUNCT = ("。", "！", "？", ".", "!", "?")
    _WEAK_CLOSURE_PUNCT = ",，、;；:："
    _LEFT_OPEN_ENGLISH_TOKENS = frozenset(
        {
            "about",
            "across",
            "after",
            "against",
            "along",
            "among",
            "around",
            "as",
            "at",
            "because",
            "before",
            "between",
            "by",
            "for",
            "from",
            "if",
            "into",
            "like",
            "of",
            "onto",
            "over",
            "than",
            "through",
            "to",
            "toward",
            "towards",
            "under",
            "until",
            "with",
            "without",
        }
    )
    _RIGHT_CONTINUATION_TOKENS = frozenset(
        {
            "a",
            "an",
            "and",
            "as",
            "at",
            "because",
            "but",
            "for",
            "from",
            "if",
            "into",
            "of",
            "or",
            "so",
            "than",
            "that",
            "the",
            "to",
            "when",
            "while",
            "with",
            "without",
        }
    )

    def build_plan(
        self,
        *,
        processor: Any,
        data: DecisionLayerInput,
        stream_id: str,
        chunk_index: Optional[int],
        words_for_split: Sequence[WordTimestamp],
    ) -> CutPlan:
        language_is_cjk = bool(processor._is_cjk_policy_language(data.policy_snapshot))
        max_segment_sec = (
            self._CJK_MAX_SEGMENT_SEC if language_is_cjk else self._DEFAULT_MAX_SEGMENT_SEC
        )
        max_overflow_sec = (
            self._CJK_MAX_OVERFLOW_SEC if language_is_cjk else self._DEFAULT_MAX_OVERFLOW_SEC
        )
        accept_threshold = (
            self._CJK_ACCEPT_THRESHOLD if language_is_cjk else self._DEFAULT_ACCEPT_THRESHOLD
        )
        (
            feature_map,
            feature_stats,
            rejection_stats,
            input_boundary_count,
        ) = self._collect_features(
            processor=processor,
            data=data,
            words_for_split=words_for_split,
        )

        decisions: list[CutDecision] = []
        candidate_diagnostics: list[dict[str, Any]] = []
        planner_rejections: Counter[str] = Counter()
        tail_hold_stats = {
            "considered_count": 0,
            "selected_count": 0,
        }
        constraint_stats = {
            "hard_limit_hits": 0,
            "rolling_resets": 0,
        }
        segment_start_idx = 0
        while segment_start_idx < len(words_for_split) - 1:
            candidate, considered_candidates, tail_hold, suppressed_split_idx = self._select_best_segment_end(
                processor=processor,
                words_for_split=words_for_split,
                segment_start_idx=segment_start_idx,
                feature_map=feature_map,
                accept_threshold=accept_threshold,
                max_segment_sec=max_segment_sec,
                max_overflow_sec=max_overflow_sec,
                language_is_cjk=language_is_cjk,
                rejection_stats=planner_rejections,
            )
            if tail_hold is not None:
                tail_hold_stats["considered_count"] += 1
            if suppressed_split_idx is not None:
                tail_hold_stats["selected_count"] += 1
            candidate_diagnostics.extend(
                self._serialize_candidate_diagnostics(
                    segment_start_idx=segment_start_idx,
                    considered_candidates=considered_candidates,
                    selected_candidate=candidate,
                    suppressed_split_idx=suppressed_split_idx,
                    tail_hold=tail_hold,
                )
            )
            if candidate is None:
                break

            decisions.append(
                CutDecision(
                    time=float(candidate.event_time),
                    window_id=f"{stream_id}:cut-{candidate.split_idx}",
                    reason=str(candidate.reason),
                    risk=None,
                    anchor_type=candidate.anchor_type,
                    anchor_score=float(candidate.score),
                    depends_on_fast_draft=False,
                    time_range=(
                        min(float(candidate.left_end), float(candidate.right_start)),
                        max(float(candidate.left_end), float(candidate.right_start)),
                    ),
                    source="timeanchored_segment_planner",
                    pyannote_frame_time=float(candidate.event_time),
                    mapped_cut_time=float(
                        processor._resolve_split_boundary_time(
                            words_for_split=words_for_split,
                            split_idx=int(candidate.split_idx),
                        )
                    ),
                    mapping_quality="boundary",
                    mapping_reason="timeanchored_segment_planner",
                )
            )
            if candidate.constraint_only:
                constraint_stats["hard_limit_hits"] += 1
            constraint_stats["rolling_resets"] += 1
            segment_start_idx = int(candidate.split_idx) + 1

        fallback_reason = ""
        if not decisions and (input_boundary_count > 0 or sum(rejection_stats.values()) > 0):
            fallback_reason = "no_boundary_passed_scoring"
        elif not decisions and len(words_for_split) > 1:
            fallback_reason = "no_candidate_boundaries"
        reason_stats = Counter(str(item.reason) for item in decisions)

        chunk_part = int(chunk_index) if chunk_index is not None else -1
        return CutPlan(
            plan_id=f"{stream_id}-segment-planner-{chunk_part}",
            block_id=f"{stream_id}:{chunk_part}",
            decisions=decisions,
            deferred_cuts=[],
            generation_report={
                "generated_by": "timeanchored_segment_planner",
                "fallback_reason": fallback_reason,
                "input_boundary_count": int(input_boundary_count),
                "decision_count": int(len(decisions)),
                "reason_stats": dict(reason_stats),
                "feature_stats": dict(feature_stats),
                "constraint_stats": dict(constraint_stats),
                "tail_hold_stats": dict(tail_hold_stats),
                "rejection_stats": {
                    **dict(rejection_stats),
                    **{
                        key: int(planner_rejections[key]) + int(rejection_stats.get(key, 0))
                        for key in planner_rejections
                    },
                },
                "candidate_diagnostics": candidate_diagnostics,
            },
        )

    def _collect_features(
        self,
        *,
        processor: Any,
        data: DecisionLayerInput,
        words_for_split: Sequence[WordTimestamp],
    ) -> tuple[dict[int, _BoundaryFeature], Counter[str], Counter[str], int]:
        raw_alignment = list(data.canonical_candidate_boundaries or ())
        raw_punctuation = list(
            processor._build_punctuation_fact_boundaries(
                punctuation_facts=list(getattr(data, "canonical_punctuation_facts", ()) or ()),
                words_for_split=words_for_split,
            )
        )
        raw_gap = list(processor._build_gap_boundaries(words_for_split=words_for_split))
        raw_speaker = list(
            processor._build_turn_change_boundaries(
                turns=list(getattr(getattr(data, "aligned_facts", None), "speaker_turns", ()) or ()),
                words_for_split=words_for_split,
            )
        )
        raw_boundaries = (
            [("alignment", item) for item in raw_alignment]
            + [("punctuation", item) for item in raw_punctuation]
            + [("gap", item) for item in raw_gap]
            + [("speaker", item) for item in raw_speaker]
        )

        feature_stats: Counter[str] = Counter()
        rejection_stats: Counter[str] = Counter()
        feature_map: dict[int, dict[str, Any]] = {}
        for source_kind, boundary in raw_boundaries:
            reject_reason = processor._resolve_ingress_boundary_reject_reason(
                boundary=boundary,
                words_for_split=words_for_split,
            )
            if reject_reason:
                rejection_stats[reject_reason] += 1
                continue

            try:
                split_idx = int(getattr(boundary, "split_idx", -1))
            except (TypeError, ValueError):
                rejection_stats["invalid_split_idx"] += 1
                continue
            if split_idx < 0 or split_idx >= len(words_for_split) - 1:
                rejection_stats["invalid_split_idx"] += 1
                continue

            payload = feature_map.setdefault(
                split_idx,
                {
                    "event_time": float(getattr(boundary, "event_time", 0.0) or 0.0),
                    "left_end": float(getattr(boundary, "left_end", 0.0) or 0.0),
                    "right_start": float(getattr(boundary, "right_start", 0.0) or 0.0),
                    "reasons": [],
                    "scores": [],
                },
            )
            payload["event_time"] = float(getattr(boundary, "event_time", payload["event_time"]) or payload["event_time"])
            payload["left_end"] = float(getattr(boundary, "left_end", payload["left_end"]) or payload["left_end"])
            payload["right_start"] = float(
                getattr(boundary, "right_start", payload["right_start"]) or payload["right_start"]
            )
            payload["reasons"].append(str(getattr(boundary, "reason", "") or "boundary_hint"))
            payload["scores"].append(float(getattr(boundary, "score", 0.0) or 0.0))
            feature_stats[f"{source_kind}_feature_count"] += 1

        normalized_map: dict[int, _BoundaryFeature] = {}
        for split_idx, payload in feature_map.items():
            normalized_map[split_idx] = _BoundaryFeature(
                split_idx=int(split_idx),
                event_time=float(payload["event_time"]),
                left_end=float(payload["left_end"]),
                right_start=float(payload["right_start"]),
                reasons=tuple(str(item) for item in payload["reasons"]),
                scores=tuple(float(item) for item in payload["scores"]),
            )
        return normalized_map, feature_stats, rejection_stats, len(raw_boundaries)

    def _select_best_segment_end(
        self,
        *,
        processor: Any,
        words_for_split: Sequence[WordTimestamp],
        segment_start_idx: int,
        feature_map: dict[int, _BoundaryFeature],
        accept_threshold: float,
        max_segment_sec: float,
        max_overflow_sec: float,
        language_is_cjk: bool,
        rejection_stats: Counter[str],
    ) -> tuple[
        Optional[_SegmentCandidate],
        list[_SegmentCandidate],
        Optional[_TailHoldCandidate],
        Optional[int],
    ]:
        segment_start_time = float(getattr(words_for_split[segment_start_idx], "start", 0.0) or 0.0)
        last_feasible_split: Optional[int] = None
        last_overflow_split: Optional[int] = None
        for split_idx in range(segment_start_idx, len(words_for_split) - 1):
            left_end = float(getattr(words_for_split[split_idx], "end", segment_start_time) or segment_start_time)
            duration_sec = max(0.0, left_end - segment_start_time)
            if duration_sec <= max_segment_sec:
                last_feasible_split = split_idx
            if duration_sec <= max_segment_sec + max_overflow_sec:
                last_overflow_split = split_idx
                continue
            break
        if last_feasible_split is None:
            return None, [], None, None

        must_split = float(getattr(words_for_split[-1], "end", segment_start_time) or segment_start_time) - segment_start_time > max_segment_sec
        candidate_end_idx = last_feasible_split
        if must_split and last_overflow_split is not None:
            candidate_end_idx = last_overflow_split
        tail_hold = None
        if not language_is_cjk and not must_split:
            tail_hold = self._build_tail_hold_candidate(
                words_for_split=words_for_split,
                segment_start_idx=segment_start_idx,
                max_segment_sec=max_segment_sec,
            )
        best_candidate: Optional[_SegmentCandidate] = None
        considered_candidates: list[_SegmentCandidate] = []
        for split_idx in range(segment_start_idx, candidate_end_idx + 1):
            candidate = self._build_candidate(
                processor=processor,
                words_for_split=words_for_split,
                segment_start_idx=segment_start_idx,
                split_idx=split_idx,
                feature=feature_map.get(split_idx),
                constraint_only=bool(must_split and split_idx >= last_feasible_split and split_idx not in feature_map),
                max_segment_sec=max_segment_sec,
            )
            if candidate is None:
                continue
            considered_candidates.append(candidate)
            if best_candidate is None or self._should_replace_best_candidate(
                current=best_candidate,
                challenger=candidate,
                prefer_later_pause_tie=not language_is_cjk,
            ):
                best_candidate = candidate

        if best_candidate is None:
            return None, considered_candidates, tail_hold, None
        if (
            tail_hold is not None
            and best_candidate.pause_only
            and best_candidate.score <= tail_hold.score + self._ENGLISH_TAIL_HOLD_MARGIN
        ):
            rejection_stats["tail_hold_preferred"] += 1
            return None, considered_candidates, tail_hold, int(best_candidate.split_idx)
        if best_candidate.score >= accept_threshold or must_split:
            return best_candidate, considered_candidates, tail_hold, None
        if best_candidate.short_fragment_hit:
            rejection_stats["short_fragment"] += 1
        else:
            rejection_stats["low_closure"] += 1
        return None, considered_candidates, tail_hold, None

    def _build_candidate(
        self,
        *,
        processor: Any,
        words_for_split: Sequence[WordTimestamp],
        segment_start_idx: int,
        split_idx: int,
        feature: Optional[_BoundaryFeature],
        constraint_only: bool,
        max_segment_sec: float,
    ) -> Optional[_SegmentCandidate]:
        if split_idx < segment_start_idx or split_idx >= len(words_for_split) - 1:
            return None

        left_word = words_for_split[split_idx]
        right_word = words_for_split[split_idx + 1]
        left_text = str(getattr(left_word, "word", "") or "").strip()
        right_text = str(getattr(right_word, "word", "") or "").strip()
        left_end = float(getattr(left_word, "end", 0.0) or 0.0)
        right_start = float(getattr(right_word, "start", left_end) or left_end)
        gap_sec = max(0.0, right_start - left_end)
        segment_start_time = float(getattr(words_for_split[segment_start_idx], "start", 0.0) or 0.0)
        duration_sec = max(0.0, left_end - segment_start_time)
        word_count = split_idx - segment_start_idx + 1
        overflow_sec = max(0.0, duration_sec - max_segment_sec)

        reasons = tuple(feature.reasons) if feature is not None else tuple()
        has_sentence_end = (
            "punctuation_sentence_end" in reasons
            or is_sentence_end_punct(
                left_text,
                right_text,
                sentence_end_chars=self._SENTENCE_END_PUNCT,
            )
        )
        has_soft_punct = bool(left_text.rstrip().endswith(tuple(self._WEAK_CLOSURE_PUNCT)))
        has_speaker_change = "speaker_change" in reasons
        has_gap_pause = "gap_pause" in reasons
        has_blank_valley = "blank_valley" in reasons
        has_alignment = any(reason in {"anchor_block_close", "lexical_boundary"} for reason in reasons)

        closure_score = 0.0
        primary_reason = "rolling_constraint"
        anchor_type = AnchorType.WORD_BOUNDARY
        if has_sentence_end:
            closure_score = 1.0
            primary_reason = "punctuation_sentence_end"
            anchor_type = AnchorType.PUNCTUATION_ANCHOR
        elif has_speaker_change:
            closure_score = 0.92
            primary_reason = "speaker_change"
        elif has_gap_pause:
            closure_score = min(0.96, 0.42 + (gap_sec * 0.45))
            primary_reason = "gap_pause"
            anchor_type = AnchorType.PAUSE_ANCHOR
        elif has_blank_valley:
            closure_score = min(0.65, 0.24 + (gap_sec * 0.80))
            primary_reason = "blank_valley"
            anchor_type = AnchorType.PAUSE_ANCHOR
        elif "punctuation_soft" in reasons:
            closure_score = 0.58
            primary_reason = "punctuation_soft"
            anchor_type = AnchorType.PUNCTUATION_ANCHOR
        elif has_alignment:
            closure_score = 0.34
            primary_reason = "anchor_block_close" if "anchor_block_close" in reasons else "lexical_boundary"
            anchor_type = AnchorType.SEMANTIC_ANCHOR
        elif constraint_only:
            closure_score = min(0.54, 0.20 + (gap_sec * 0.30))

        score = closure_score
        if has_soft_punct:
            score += 0.10
        if word_count >= 4:
            score += 0.06
        if 1.0 <= duration_sec <= 4.6:
            score += 0.06
        if has_sentence_end:
            short_fragment_penalty = 0.0
        elif word_count <= 1:
            short_fragment_penalty = 1.10 if gap_sec < 0.90 else 0.55
            if has_soft_punct:
                short_fragment_penalty -= 0.18
        elif word_count == 2:
            short_fragment_penalty = 0.88 if gap_sec < 0.90 else 0.42
            if has_soft_punct:
                short_fragment_penalty -= 0.12
        elif word_count == 3 and duration_sec < 2.2:
            short_fragment_penalty = 0.26
        else:
            short_fragment_penalty = 0.0
        pause_only = (
            primary_reason in {"gap_pause", "blank_valley"}
            and not has_sentence_end
            and not has_speaker_change
            and not has_soft_punct
            and "punctuation_soft" not in reasons
        )
        score -= max(0.0, short_fragment_penalty)
        score -= min(0.35, overflow_sec * 0.45)
        score -= self._continuation_penalty(
            left_text=left_text,
            right_text=right_text,
            has_sentence_end=has_sentence_end,
            has_speaker_change=has_speaker_change,
        )

        event_time = (
            float(feature.event_time)
            if feature is not None
            else (float(left_end) + float(right_start)) / 2.0
        )
        return _SegmentCandidate(
            split_idx=int(split_idx),
            event_time=float(event_time),
            left_end=float(left_end),
            right_start=float(right_start),
            left_text=left_text,
            right_text=right_text,
            gap_sec=float(gap_sec),
            duration_sec=float(duration_sec),
            word_count=int(word_count),
            reasons=tuple(reasons),
            reason=primary_reason,
            anchor_type=anchor_type,
            closure_score=float(closure_score),
            score=float(score),
            constraint_only=bool(constraint_only),
            short_fragment_hit=short_fragment_penalty >= 0.42,
            overflow_sec=float(overflow_sec),
            pause_only=bool(pause_only),
        )

    @classmethod
    def _serialize_candidate_diagnostics(
        cls,
        *,
        segment_start_idx: int,
        considered_candidates: Sequence[_SegmentCandidate],
        selected_candidate: Optional[_SegmentCandidate],
        suppressed_split_idx: Optional[int],
        tail_hold: Optional[_TailHoldCandidate],
    ) -> list[dict[str, Any]]:
        selected_split_idx = (
            int(selected_candidate.split_idx)
            if selected_candidate is not None
            else None
        )
        best_split_idx = (
            max(
                considered_candidates,
                key=lambda item: float(item.score),
            ).split_idx
            if considered_candidates
            else None
        )
        rows: list[dict[str, Any]] = []
        for item in considered_candidates:
            status = "candidate"
            if suppressed_split_idx is not None and int(item.split_idx) == int(suppressed_split_idx):
                status = "suppressed_by_tail_hold"
            elif selected_split_idx is not None and int(item.split_idx) == selected_split_idx:
                status = "selected"
            elif best_split_idx is not None and int(item.split_idx) == int(best_split_idx):
                status = "best_rejected"
            rows.append(
                {
                    "segment_start_idx": int(segment_start_idx),
                    "split_idx": int(item.split_idx),
                    "left_text": str(item.left_text),
                    "right_text": str(item.right_text),
                    "gap_sec": float(item.gap_sec),
                    "duration_sec": float(item.duration_sec),
                    "word_count": int(item.word_count),
                    "reasons": [str(reason) for reason in item.reasons],
                    "primary_reason": str(item.reason),
                    "closure_score": float(item.closure_score),
                    "final_score": float(item.score),
                    "constraint_only": bool(item.constraint_only),
                    "short_fragment_hit": bool(item.short_fragment_hit),
                    "overflow_sec": float(item.overflow_sec),
                    "pause_only": bool(item.pause_only),
                    "selection_status": status,
                    "tail_hold_score": float(tail_hold.score) if tail_hold is not None else None,
                    "tail_hold_margin": float(cls._ENGLISH_TAIL_HOLD_MARGIN) if tail_hold is not None else None,
                }
            )
        return rows

    @classmethod
    def _build_tail_hold_candidate(
        cls,
        *,
        words_for_split: Sequence[WordTimestamp],
        segment_start_idx: int,
        max_segment_sec: float,
    ) -> Optional[_TailHoldCandidate]:
        if segment_start_idx < 0 or segment_start_idx >= len(words_for_split):
            return None
        segment_start_time = float(getattr(words_for_split[segment_start_idx], "start", 0.0) or 0.0)
        segment_end_time = float(getattr(words_for_split[-1], "end", segment_start_time) or segment_start_time)
        duration_sec = max(0.0, segment_end_time - segment_start_time)
        if duration_sec <= 0.0 or duration_sec > max_segment_sec:
            return None
        word_count = len(words_for_split) - int(segment_start_idx)
        end_token = cls._normalize_english_token(str(getattr(words_for_split[-1], "word", "") or ""))

        score = 0.56
        if word_count >= 4:
            score += 0.10
        if word_count >= 7:
            score += 0.08
        if 1.8 <= duration_sec <= max_segment_sec:
            score += 0.08
        if end_token and end_token not in cls._LEFT_OPEN_ENGLISH_TOKENS:
            score += 0.06
        if end_token and end_token not in cls._RIGHT_CONTINUATION_TOKENS:
            score += 0.04
        if word_count <= 2:
            score -= 0.10

        return _TailHoldCandidate(
            segment_start_idx=int(segment_start_idx),
            segment_end_idx=len(words_for_split) - 1,
            duration_sec=float(duration_sec),
            word_count=int(word_count),
            score=float(score),
        )

    @staticmethod
    def _should_replace_best_candidate(
        *,
        current: _SegmentCandidate,
        challenger: _SegmentCandidate,
        prefer_later_pause_tie: bool,
    ) -> bool:
        score_delta = float(challenger.score) - float(current.score)
        if score_delta > 1e-6:
            return True
        if abs(score_delta) > 1e-6:
            return False
        if (
            prefer_later_pause_tie
            and current.pause_only
            and challenger.pause_only
            and challenger.split_idx > current.split_idx
        ):
            return True
        return False

    @classmethod
    def _continuation_penalty(
        cls,
        *,
        left_text: str,
        right_text: str,
        has_sentence_end: bool,
        has_speaker_change: bool,
    ) -> float:
        if has_sentence_end or has_speaker_change:
            return 0.0
        left_core = cls._normalize_english_token(left_text)
        right_core = cls._normalize_english_token(right_text)
        penalty = 0.0
        if left_core in cls._LEFT_OPEN_ENGLISH_TOKENS:
            penalty += 0.48
        if right_core in cls._RIGHT_CONTINUATION_TOKENS:
            penalty += 0.42
        return min(0.70, penalty)

    @staticmethod
    def _normalize_english_token(token: str) -> str:
        normalized = str(token or "").strip().lower()
        while normalized and not normalized[0].isalnum():
            normalized = normalized[1:]
        while normalized and not normalized[-1].isalnum():
            normalized = normalized[:-1]
        return normalized
