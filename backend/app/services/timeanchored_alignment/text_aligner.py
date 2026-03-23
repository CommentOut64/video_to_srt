"""TextAligner：文本主判，发音证据仅做局部共判。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

from app.models.confidence_models import AlignedWord, AlignmentStatus
from app.services.alignment.gap_resolver import GapResolution, GapResolver
from app.services.alignment.nw_v2_core import NeedlemanWunschV2Core
from app.services.language_policy.compiler import build_language_policy_snapshot
from app.services.language_policy.types import (
    WINDOW_KIND_DOMINANT_WITH_ISLANDS,
    WINDOW_KIND_SINGLE_LANGUAGE,
    WINDOW_KIND_TRUE_MIXED,
)
from app.services.timeanchored_alignment.contracts import (
    AlignmentItem,
    AlignmentMetrics,
    FinalAlignmentResult,
    LanguageRunPackage,
    PronunciationPackage,
    ProtectedSpan,
    TextTruthPackage,
    TimeBasePackage,
    TimeBaseUnit,
)
from app.services.timeanchored_alignment.text_alignment_scoring import (
    TextAlignmentScorer,
    default_token_normalizer,
)

if TYPE_CHECKING:
    from app.services.timeanchored_alignment.phonetic_aligner import PhoneticAligner, PhoneticRescueTrace


@dataclass(frozen=True)
class TextAlignerThresholds:
    text_direct_ratio_min: float
    text_estimated_ratio_max: float
    foreign_run_max_ratio: float
    pronunciation_evidence_min_score: float


class TextAligner:
    """Phase4 文本主判器。"""

    def __init__(
        self,
        *,
        threshold_overrides: Optional[Mapping[str, float]] = None,
        nw_core: Optional[NeedlemanWunschV2Core] = None,
        gap_resolver: Optional[GapResolver] = None,
        scorer: Optional[TextAlignmentScorer] = None,
        phonetic_aligner: Optional["PhoneticAligner"] = None,
    ) -> None:
        self._threshold_overrides = dict(threshold_overrides or {})
        self._nw_core = nw_core or NeedlemanWunschV2Core()
        self._gap_resolver = gap_resolver or GapResolver()
        self._scorer = scorer or TextAlignmentScorer()
        self._normalizer = default_token_normalizer
        self._phonetic_aligner = phonetic_aligner
        self._last_phonetic_report: dict[str, object] = {}
        self._last_phonetic_traces: tuple["PhoneticRescueTrace", ...] = tuple()

    @property
    def last_phonetic_report(self) -> dict[str, object]:
        return dict(self._last_phonetic_report)

    @property
    def last_phonetic_traces(self) -> tuple["PhoneticRescueTrace", ...]:
        return tuple(self._last_phonetic_traces)

    def align_window(
        self,
        *,
        time_base: TimeBasePackage,
        text_truth: TextTruthPackage,
        language_runs: LanguageRunPackage,
        pronunciation: PronunciationPackage,
    ) -> FinalAlignmentResult:
        self._last_phonetic_report = {}
        self._last_phonetic_traces = tuple()
        if language_runs.window_kind == WINDOW_KIND_TRUE_MIXED:
            self._last_phonetic_report = {"skipped": True, "reason": "true_mixed_window"}
            return self._error_result(
                route="mixed",
                error_code="TRUE_MIXED_WINDOW",
                item_count=len(text_truth.units),
            )
        if language_runs.window_kind not in {WINDOW_KIND_SINGLE_LANGUAGE, WINDOW_KIND_DOMINANT_WITH_ISLANDS}:
            self._last_phonetic_report = {"skipped": True, "reason": "unsupported_window_kind"}
            return self._error_result(
                route="error",
                error_code="UNSUPPORTED_WINDOW_KIND",
                item_count=len(text_truth.units),
            )
        if not text_truth.units:
            self._last_phonetic_report = {"skipped": True, "reason": "empty_text_truth"}
            return self._error_result(route="error", error_code="EMPTY_TEXT_TRUTH", item_count=0)

        dominant_language = self._resolve_language(time_base, text_truth, language_runs, pronunciation)
        thresholds = self._resolve_thresholds(dominant_language)
        if (
            language_runs.window_kind == WINDOW_KIND_DOMINANT_WITH_ISLANDS
            and float(language_runs.foreign_run_ratio) > float(thresholds.foreign_run_max_ratio)
        ):
            self._last_phonetic_report = {"skipped": True, "reason": "foreign_run_ratio_exceeded"}
            return self._error_result(
                route="mixed",
                error_code="FOREIGN_RUN_RATIO_EXCEEDED",
                item_count=len(text_truth.units),
            )

        time_units, error_code = self._select_time_units(time_base=time_base, language=dominant_language)
        if error_code:
            self._last_phonetic_report = {"skipped": True, "reason": str(error_code).lower()}
            return self._error_result(route="error", error_code=error_code, item_count=len(text_truth.units))

        seq1 = [unit.text for unit in time_units]
        seq2 = [unit.text for unit in text_truth.units]
        mapping = self._align_indices(
            seq1=seq1,
            seq2=seq2,
            seq1_confidences=[unit.confidence for unit in time_units],
            seq2_confidences=[unit.confidence for unit in text_truth.units],
        )

        unit_spans = self._resolve_text_unit_spans(text_truth)
        pronunciation_tokens = {
            self._normalizer(token.token_text)
            for token in pronunciation.token_units
            if self._normalizer(token.token_text)
        }
        ja_pron_fallback = self._is_ja_pronunciation_fallback(
            language=dominant_language,
            pronunciation=pronunciation,
        )

        direct_count = 0
        estimated_count = 0
        failed_count = 0
        raw_items: list[AlignmentItem] = []
        aligned_words: list[AlignedWord] = []

        for text_index, text_unit in enumerate(text_truth.units):
            mapped_time_index = mapping.get(text_index)
            if mapped_time_index is None:
                start = self._fallback_start(raw_items)
                end = start + self._fallback_duration(time_base)
                raw_items.append(
                    AlignmentItem(
                        text=text_unit.text,
                        start=start,
                        end=end,
                        status="failed",
                        source="text_aligner",
                        confidence=0.0,
                        reason="unmapped",
                    )
                )
                aligned_words.append(
                    AlignedWord(
                        word=text_unit.text,
                        start=start,
                        end=end,
                        final_confidence=0.0,
                        confidence_source="text_aligner",
                        alignment_status=AlignmentStatus.INSERTED,
                        is_pseudo=True,
                    )
                )
                failed_count += 1
                continue

            time_unit = time_units[mapped_time_index]
            normalized_token = self._normalizer(text_unit.text)
            score = self._scorer.score(
                time_token=time_unit.text,
                text_token=text_unit.text,
                is_protected_span=self._is_protected_unit(
                    unit_index=text_index,
                    unit_spans=unit_spans,
                    protected_spans=text_truth.protected_spans,
                ),
                pronunciation_match=(
                    bool(normalized_token)
                    and (normalized_token in pronunciation_tokens)
                    and (not ja_pron_fallback)
                ),
                top_candidates=time_unit.top_candidates,
            )
            confidence = max(0.0, min(1.0, float(score.total_score)))
            if score.text_exact > 0:
                status = "direct"
                reason = "text_exact"
                direct_count += 1
                align_status = AlignmentStatus.MATCHED
            else:
                status = "estimated"
                reason = "text_normalized_or_tiebreak"
                if ja_pron_fallback and dominant_language == "ja":
                    reason = f"{reason}|ja_pron_fallback"
                estimated_count += 1
                align_status = AlignmentStatus.SUBSTITUTED

            raw_items.append(
                AlignmentItem(
                    text=text_unit.text,
                    start=time_unit.start,
                    end=time_unit.end,
                    status=status,
                    source="text_aligner",
                    confidence=confidence,
                    reason=reason,
                )
            )
            aligned_words.append(
                AlignedWord(
                    word=text_unit.text,
                    start=time_unit.start,
                    end=time_unit.end,
                    final_confidence=confidence,
                    confidence_source="text_aligner",
                    alignment_status=align_status,
                    is_pseudo=False,
                )
            )

        items = list(raw_items)
        if failed_count > 0:
            gap_result = self._gap_resolver.resolve_gaps(aligned_words)
            if gap_result.resolution in {GapResolution.INTERPOLATED, GapResolution.MERGED}:
                for idx, word in enumerate(gap_result.words):
                    if items[idx].status != "failed":
                        continue
                    items[idx] = AlignmentItem(
                        text=items[idx].text,
                        start=word.start,
                        end=word.end,
                        status="interpolated",
                        source=items[idx].source,
                        confidence=items[idx].confidence,
                        reason=f"gap_{gap_result.resolution.value}",
                    )
                    failed_count -= 1
                    estimated_count += 1

        total = len(text_truth.units)
        direct_ratio = float(direct_count) / float(total)
        estimated_ratio = float(estimated_count) / float(total)
        coverage = float(total - failed_count) / float(total)

        route = "text"
        error_code = None
        if direct_ratio < thresholds.text_direct_ratio_min:
            route = "error"
            error_code = "TEXT_DIRECT_RATIO_LOW"
        elif estimated_ratio > thresholds.text_estimated_ratio_max:
            route = "error"
            error_code = "TEXT_ESTIMATED_RATIO_HIGH"
        elif failed_count > 0:
            route = "error"
            error_code = "TEXT_ALIGNMENT_FAILED"

        text_result = FinalAlignmentResult(
            items=tuple(items),
            route=route,
            metrics=AlignmentMetrics(
                coverage=max(0.0, min(1.0, coverage)),
                duration_ratio=1.0,
                failed_count=max(0, int(failed_count)),
                route_confidence=max(0.0, min(1.0, direct_ratio)),
            ),
            error_code=error_code,
        )
        if self._phonetic_aligner is None:
            self._last_phonetic_report = {"skipped": True, "reason": "phonetic_aligner_not_configured"}
            return text_result
        if not self._should_trigger_phonetic(text_result=text_result, thresholds=thresholds):
            self._last_phonetic_report = {"skipped": True, "reason": "trigger_not_met"}
            return text_result
        rescue = self._phonetic_aligner.rescue_window(
            text_alignment=text_result,
            time_base=time_base,
            text_truth=text_truth,
            language_runs=language_runs,
            pronunciation=pronunciation,
        )
        self._last_phonetic_report = dict(rescue.report or {})
        self._last_phonetic_traces = tuple(rescue.traces or tuple())
        return rescue.alignment

    def _resolve_thresholds(self, language: str) -> TextAlignerThresholds:
        snapshot = build_language_policy_snapshot(
            language_hint=language,
            feature_scope="timeanchored_alignment",
        )
        merged = dict(snapshot.thresholds or {})
        merged.update(self._threshold_overrides)
        return TextAlignerThresholds(
            text_direct_ratio_min=self._safe_float(merged.get("text_direct_ratio_min"), 0.65),
            text_estimated_ratio_max=self._safe_float(merged.get("text_estimated_ratio_max"), 0.35),
            foreign_run_max_ratio=self._safe_float(merged.get("foreign_run_max_ratio"), 0.50),
            pronunciation_evidence_min_score=self._safe_float(merged.get("pronunciation_evidence_min_score"), 0.45),
        )

    def _resolve_language(
        self,
        time_base: TimeBasePackage,
        text_truth: TextTruthPackage,
        language_runs: LanguageRunPackage,
        pronunciation: PronunciationPackage,
    ) -> str:
        for candidate in (
            language_runs.dominant_language,
            text_truth.language,
            time_base.language,
            pronunciation.language,
        ):
            normalized = str(candidate or "").strip().lower()
            if normalized in {"zh", "ja", "en"}:
                return normalized
        return "zh"

    def _select_time_units(
        self,
        *,
        time_base: TimeBasePackage,
        language: str,
    ) -> tuple[Sequence[TimeBaseUnit], Optional[str]]:
        if language == "en":
            if not time_base.word_units:
                return tuple(), "EMPTY_EN_WORD_UNITS"
            if any(self._looks_like_subword(unit.text) for unit in time_base.word_units):
                return tuple(), "EN_SUBWORD_MAIN_PATH_FORBIDDEN"
            return tuple(time_base.word_units), None

        if time_base.raw_units:
            return tuple(time_base.raw_units), None
        if time_base.word_units:
            return tuple(time_base.word_units), None
        return tuple(), "EMPTY_TIME_BASE_UNITS"

    def _align_indices(
        self,
        *,
        seq1: Sequence[str],
        seq2: Sequence[str],
        seq1_confidences: Sequence[float],
        seq2_confidences: Sequence[float],
    ) -> dict[int, int]:
        path = self._nw_core.align(
            seq1=seq1,
            seq2=seq2,
            seq1_confidences=seq1_confidences,
            seq2_confidences=seq2_confidences,
            match_fn=lambda left, right: (
                left == right or (self._normalizer(left) and self._normalizer(left) == self._normalizer(right))
            ),
        )
        mapped: dict[int, int] = {}
        for left_index, right_index in path:
            if left_index is None or right_index is None:
                continue
            mapped.setdefault(int(right_index), int(left_index))
        return mapped

    @staticmethod
    def _is_protected_unit(
        *,
        unit_index: int,
        unit_spans: Sequence[tuple[int, int]],
        protected_spans: Sequence[ProtectedSpan],
    ) -> bool:
        if unit_index >= len(unit_spans):
            return False
        unit_start, unit_end = unit_spans[unit_index]
        for span in protected_spans:
            if unit_end <= int(span.start):
                continue
            if unit_start >= int(span.end):
                continue
            return True
        return False

    @staticmethod
    def _resolve_text_unit_spans(text_truth: TextTruthPackage) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        source_text = str(text_truth.raw_text or "")
        cursor = 0
        for unit in text_truth.units:
            token = str(unit.text or "")
            if source_text and token:
                found = source_text.find(token, cursor)
                if found >= 0:
                    spans.append((found, found + len(token)))
                    cursor = found + len(token)
                    continue
            spans.append((cursor, cursor + len(token)))
            cursor += len(token)
        return spans

    @staticmethod
    def _looks_like_subword(token: str) -> bool:
        text = str(token or "")
        return text.startswith("##") or text.startswith("▁")

    @staticmethod
    def _fallback_start(items: Sequence[AlignmentItem]) -> float:
        if not items:
            return 0.0
        return float(items[-1].end)

    @staticmethod
    def _fallback_duration(time_base: TimeBasePackage) -> float:
        stride = float(time_base.frame_stride)
        if stride <= 0.0:
            return 0.06
        return max(0.03, stride)

    @staticmethod
    def _safe_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _should_trigger_phonetic(
        *,
        text_result: FinalAlignmentResult,
        thresholds: TextAlignerThresholds,
    ) -> bool:
        if not text_result.items:
            return False
        if all(item.status == "direct" for item in text_result.items):
            return False
        if any(item.status in {"failed", "estimated", "interpolated"} for item in text_result.items):
            return True
        for item in text_result.items:
            if item.confidence is None:
                continue
            if float(item.confidence) < float(thresholds.pronunciation_evidence_min_score):
                return True
        return False

    @staticmethod
    def _is_ja_pronunciation_fallback(
        *,
        language: str,
        pronunciation: PronunciationPackage,
    ) -> bool:
        if language != "ja":
            return False
        mode = str(pronunciation.dependency_mode.get("ja") or "").strip().lower()
        return bool(mode) and mode != "sudachi"

    @staticmethod
    def _error_result(
        *,
        route: str,
        error_code: str,
        item_count: int,
    ) -> FinalAlignmentResult:
        return FinalAlignmentResult(
            items=tuple(),
            route=route,
            metrics=AlignmentMetrics(
                coverage=0.0,
                duration_ratio=1.0,
                failed_count=max(0, int(item_count)),
                route_confidence=0.0,
            ),
            error_code=error_code,
        )


__all__ = ["TextAligner", "TextAlignerThresholds"]
