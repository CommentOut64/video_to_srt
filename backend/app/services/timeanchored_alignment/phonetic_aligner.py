"""Phase5 轻量发音证据层。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from app.services.homophone.tokenizers import HomophoneTokenizer
from app.services.language_policy.types import WINDOW_KIND_TRUE_MIXED
from app.services.timeanchored_alignment.contracts import (
    AcousticCandidate,
    AlignmentItem,
    AlignmentMetrics,
    FinalAlignmentResult,
    LanguageRunPackage,
    PronunciationPackage,
    ProtectedSpan,
    TextTruthPackage,
    TimeBasePackage,
)
from app.services.timeanchored_alignment.phonetic_matchers import (
    align_monotonic_sequences,
    project_token_phone_keys,
)


@dataclass(frozen=True)
class PhoneticAlignerConfig:
    rescue_score_min: float = 0.62
    low_confidence_threshold: float = 0.55
    ambiguous_top_delta_max: float = 0.08
    top_candidate_bonus: float = 0.08


@dataclass(frozen=True)
class PhoneticRescueTrace:
    index: int
    token_text: str
    language: str
    rescued: bool
    evidence_score: float
    reason: str
    source_status: str
    target_status: str
    fallback_reason: str = ""


@dataclass(frozen=True)
class PhoneticRescueResult:
    alignment: FinalAlignmentResult
    traces: Tuple[PhoneticRescueTrace, ...] = tuple()
    report: Dict[str, Any] = field(default_factory=dict)


class PhoneticAligner:
    """仅处理局部 unresolved/低置信片段的发音证据层。"""

    def __init__(
        self,
        *,
        tokenizer: Optional[HomophoneTokenizer] = None,
        config: Optional[PhoneticAlignerConfig] = None,
        zh_reading_overrides: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._tokenizer = tokenizer or HomophoneTokenizer()
        self._config = config or PhoneticAlignerConfig()
        self._zh_reading_overrides = {
            str(key): str(value).strip().lower()
            for key, value in dict(zh_reading_overrides or {}).items()
            if str(key).strip()
        }

    def rescue_window(
        self,
        *,
        text_alignment: FinalAlignmentResult,
        time_base: TimeBasePackage,
        text_truth: TextTruthPackage,
        language_runs: LanguageRunPackage,
        pronunciation: PronunciationPackage,
    ) -> PhoneticRescueResult:
        if language_runs.window_kind == WINDOW_KIND_TRUE_MIXED:
            return PhoneticRescueResult(
                alignment=text_alignment,
                traces=tuple(),
                report={
                    "skipped": True,
                    "fallback_reason": "true_mixed_window",
                    "candidate_count": 0,
                    "rescued_count": 0,
                },
            )

        language = self._resolve_language(time_base, text_truth, language_runs, pronunciation)
        time_units = tuple(time_base.word_units if language == "en" and time_base.word_units else time_base.raw_units)
        if language == "en" and any(self._looks_like_subword(unit.text) for unit in time_units):
            return PhoneticRescueResult(
                alignment=text_alignment,
                traces=tuple(),
                report={
                    "language": language,
                    "candidate_count": 0,
                    "rescued_count": 0,
                    "fallback_reason": "en_subword_input_forbidden",
                },
            )
        text_tokens = [unit.text for unit in text_truth.units]
        unit_spans = self._resolve_text_unit_spans(text_truth)
        unit_languages = self._resolve_unit_languages(
            unit_spans=unit_spans,
            language_runs=language_runs,
            default_language=language,
        )
        text_keys = tuple(
            self._token_key(token, language=unit_languages[idx], fuzzy=False)
            for idx, token in enumerate(text_tokens)
        )
        text_fuzzy = tuple(
            self._token_key(token, language=unit_languages[idx], fuzzy=True)
            for idx, token in enumerate(text_tokens)
        )
        pron_keys, pron_fuzzy = self._build_pronunciation_keys(pronunciation=pronunciation, default_language=language)
        pron_index_map = self._build_pron_index_map(text_keys=text_keys, pron_keys=pron_keys)

        items = list(text_alignment.items)
        traces: list[PhoneticRescueTrace] = []
        rescued_count = 0
        candidate_count = 0

        for index, item in enumerate(items):
            if not self._should_process(index=index, item=item, time_units=time_units):
                continue
            candidate_count += 1

            if self._is_protected_unit(index=index, unit_spans=unit_spans, protected_spans=text_truth.protected_spans):
                traces.append(
                    PhoneticRescueTrace(
                        index=index,
                        token_text=item.text,
                        language=language,
                        rescued=False,
                        evidence_score=0.0,
                        reason="protected_span_skip",
                        source_status=item.status,
                        target_status=item.status,
                    )
                )
                continue

            time_unit = time_units[index] if index < len(time_units) else None
            strict_text = text_keys[index] if index < len(text_keys) else ""
            fuzzy_text = text_fuzzy[index] if index < len(text_fuzzy) else ""
            unit_language = unit_languages[index] if index < len(unit_languages) else language
            strict_time = (
                self._token_key(time_unit.text, language=unit_language, fuzzy=False)
                if time_unit is not None
                else ""
            )
            fuzzy_time = (
                self._token_key(time_unit.text, language=unit_language, fuzzy=True)
                if time_unit is not None
                else ""
            )
            pron_idx = pron_index_map.get(index)
            strict_pron = pron_keys[pron_idx] if pron_idx is not None and pron_idx < len(pron_keys) else ""
            fuzzy_pron = pron_fuzzy[pron_idx] if pron_idx is not None and pron_idx < len(pron_fuzzy) else ""

            score, reason = self._score_local_evidence(
                strict_text=strict_text,
                fuzzy_text=fuzzy_text,
                strict_time=strict_time,
                fuzzy_time=fuzzy_time,
                strict_pron=strict_pron,
                fuzzy_pron=fuzzy_pron,
                top_candidates=time_unit.top_candidates if time_unit is not None else tuple(),
                original_text=item.text,
            )
            if score < self._config.rescue_score_min:
                traces.append(
                    PhoneticRescueTrace(
                        index=index,
                        token_text=item.text,
                        language=language,
                        rescued=False,
                        evidence_score=score,
                        reason=reason or "insufficient_evidence",
                        source_status=item.status,
                        target_status=item.status,
                    )
                )
                continue

            rescued_count += 1
            rescued_item = self._build_rescued_item(
                index=index,
                item=item,
                time_units=time_units,
                frame_stride=time_base.frame_stride,
                score=score,
                reason=reason or "phonetic_rescue",
            )
            items[index] = rescued_item
            traces.append(
                PhoneticRescueTrace(
                    index=index,
                    token_text=item.text,
                    language=language,
                    rescued=True,
                    evidence_score=score,
                    reason=reason or "phonetic_rescue",
                    source_status=item.status,
                    target_status=rescued_item.status,
                )
            )

        failed_count = sum(1 for row in items if row.status == "failed")
        coverage = float(len(items) - failed_count) / float(max(len(items), 1))
        route = text_alignment.route
        error_code = text_alignment.error_code
        if rescued_count > 0:
            route = "phonetic"
            if failed_count == 0:
                error_code = None

        fallback_reason = ""
        if language == "ja":
            ja_mode = str(pronunciation.dependency_mode.get("ja") or "").strip().lower()
            if ja_mode and ja_mode != "sudachi":
                fallback_reason = f"ja_{ja_mode}"

        alignment = FinalAlignmentResult(
            items=tuple(items),
            route=route,
            metrics=AlignmentMetrics(
                coverage=max(0.0, min(1.0, coverage)),
                duration_ratio=text_alignment.metrics.duration_ratio,
                failed_count=failed_count,
                route_confidence=max(
                    text_alignment.metrics.route_confidence,
                    float(rescued_count) / float(max(len(items), 1)),
                ),
            ),
            error_code=error_code,
        )
        report = {
            "language": language,
            "candidate_count": candidate_count,
            "rescued_count": rescued_count,
            "fallback_reason": fallback_reason,
        }
        return PhoneticRescueResult(
            alignment=alignment,
            traces=tuple(traces),
            report=report,
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

    def _token_key(self, token: str, *, language: str, fuzzy: bool) -> str:
        value = str(token or "").strip()
        if not value:
            return ""
        if language == "zh" and value in self._zh_reading_overrides:
            strict = self._zh_reading_overrides[value]
            if fuzzy:
                return re.sub(r"[1-5]$", "", strict)
            return strict
        try:
            return str(
                self._tokenizer.build_query_key(
                    query_text=value,
                    language=language,  # type: ignore[arg-type]
                    is_fuzzy=fuzzy,
                )
                or ""
            )
        except Exception as exc:
            raise RuntimeError(f"token_key 构建失败: language={language} token={value}") from exc

    def _build_pronunciation_keys(
        self,
        *,
        pronunciation: PronunciationPackage,
        default_language: str,
    ) -> tuple[Tuple[str, ...], Tuple[str, ...]]:
        token_count = len(pronunciation.token_units)
        phone_keys = tuple(str(item.phone_text or "") for item in pronunciation.phone_units)
        spans = tuple(
            (int(item.token_index), int(item.phone_start), int(item.phone_end))
            for item in pronunciation.token_to_phone_spans
        )
        projected = project_token_phone_keys(
            token_count=token_count,
            phone_keys=phone_keys,
            token_to_phone_spans=spans,
        )
        strict_keys: list[str] = []
        fuzzy_keys: list[str] = []
        for index, token in enumerate(pronunciation.token_units):
            language = str(token.language or default_language).strip().lower() or default_language
            strict = self._token_key(token.token_text, language=language, fuzzy=False)
            fuzzy = self._token_key(token.token_text, language=language, fuzzy=True)
            if not strict and projected[index]:
                strict = str(projected[index]).lower()
            if not fuzzy and projected[index]:
                fuzzy = str(projected[index]).lower()
            strict_keys.append(strict)
            fuzzy_keys.append(fuzzy)
        return tuple(strict_keys), tuple(fuzzy_keys)

    @staticmethod
    def _resolve_unit_languages(
        *,
        unit_spans: Sequence[tuple[int, int]],
        language_runs: LanguageRunPackage,
        default_language: str,
    ) -> Tuple[str, ...]:
        resolved: list[str] = []
        runs = tuple(language_runs.runs)
        for start, end in unit_spans:
            best_overlap = 0
            best_language = default_language
            for run in runs:
                run_start = int(run.char_start)
                run_end = int(run.char_end)
                overlap = min(end, run_end) - max(start, run_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    language = str(run.run_language or "").strip().lower()
                    if language in {"zh", "ja", "en"}:
                        best_language = language
            resolved.append(best_language)
        return tuple(resolved)

    @staticmethod
    def _build_pron_index_map(
        *,
        text_keys: Sequence[str],
        pron_keys: Sequence[str],
    ) -> Dict[int, int]:
        if not text_keys or not pron_keys:
            return {}
        mapping: Dict[int, int] = {}
        for row in align_monotonic_sequences(left_keys=text_keys, right_keys=pron_keys):
            if row.left_index is None or row.right_index is None:
                continue
            mapping.setdefault(int(row.left_index), int(row.right_index))
        return mapping

    def _should_process(
        self,
        *,
        index: int,
        item: AlignmentItem,
        time_units: Sequence[Any],
    ) -> bool:
        if item.status in {"failed", "estimated"}:
            return True
        if item.confidence is not None and float(item.confidence) < self._config.low_confidence_threshold:
            return True
        if index >= len(time_units):
            return False
        return self._is_ambiguous_top_candidates(time_units[index].top_candidates)

    def _is_ambiguous_top_candidates(self, top_candidates: Sequence[AcousticCandidate]) -> bool:
        if len(top_candidates) < 2:
            return False
        ordered = sorted(top_candidates, key=lambda item: float(item.score), reverse=True)
        delta = float(ordered[0].score) - float(ordered[1].score)
        return abs(delta) <= self._config.ambiguous_top_delta_max

    @staticmethod
    def _resolve_text_unit_spans(text_truth: TextTruthPackage) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        source_text = str(text_truth.raw_text or "")
        cursor = 0
        for unit in text_truth.units:
            if unit.start is not None and unit.end is not None:
                spans.append((int(unit.start), int(unit.end)))
                continue
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
    def _is_protected_unit(
        *,
        index: int,
        unit_spans: Sequence[tuple[int, int]],
        protected_spans: Sequence[ProtectedSpan],
    ) -> bool:
        if index >= len(unit_spans):
            return False
        start, end = unit_spans[index]
        for span in protected_spans:
            if end <= int(span.start):
                continue
            if start >= int(span.end):
                continue
            return True
        return False

    def _score_local_evidence(
        self,
        *,
        strict_text: str,
        fuzzy_text: str,
        strict_time: str,
        fuzzy_time: str,
        strict_pron: str,
        fuzzy_pron: str,
        top_candidates: Sequence[AcousticCandidate],
        original_text: str,
    ) -> tuple[float, str]:
        score = 0.0
        reason = ""
        if strict_text and strict_time and strict_text == strict_time:
            return 1.0, "strict_time_match"
        if fuzzy_text and fuzzy_time and fuzzy_text == fuzzy_time:
            score = max(score, 0.74)
            reason = "fuzzy_time_match"
        if score > 0.0 and strict_text and strict_pron and strict_text == strict_pron:
            score = min(1.0, score + 0.05)
            reason = f"{reason}+strict_pron_support" if reason else "strict_pron_support"
        if score > 0.0 and fuzzy_text and fuzzy_pron and fuzzy_text == fuzzy_pron:
            score = min(1.0, score + 0.03)
            reason = f"{reason}+fuzzy_pron_support" if reason else "fuzzy_pron_support"
        if self._top_candidate_supports_text(top_candidates=top_candidates, text=original_text):
            score = min(1.0, score + self._config.top_candidate_bonus)
            reason = f"{reason}+top_candidate" if reason else "top_candidate_only"
        return score, reason

    @staticmethod
    def _top_candidate_supports_text(
        *,
        top_candidates: Sequence[AcousticCandidate],
        text: str,
    ) -> bool:
        if len(top_candidates) < 1:
            return False
        normalized = str(text or "").strip()
        for row in top_candidates:
            if str(row.text or "").strip() == normalized:
                return True
        return False

    @staticmethod
    def _looks_like_subword(token: str) -> bool:
        text = str(token or "")
        return text.startswith("##") or text.startswith("▁")

    @staticmethod
    def _build_rescued_item(
        *,
        index: int,
        item: AlignmentItem,
        time_units: Sequence[Any],
        frame_stride: float,
        score: float,
        reason: str,
    ) -> AlignmentItem:
        if index < len(time_units):
            unit = time_units[index]
            return AlignmentItem(
                text=item.text,
                start=float(unit.start),
                end=float(unit.end),
                status="phonetic",
                source="phonetic_aligner",
                confidence=max(0.0, min(1.0, score)),
                reason=reason,
            )

        stride = max(0.03, float(frame_stride or 0.06))
        prev_end = float(item.start)
        end = prev_end + stride
        return AlignmentItem(
            text=item.text,
            start=prev_end,
            end=end,
            status="phonetic",
            source="phonetic_aligner",
            confidence=max(0.0, min(1.0, score)),
            reason=f"{reason}|single_gap_interp",
        )


__all__ = [
    "PhoneticAligner",
    "PhoneticAlignerConfig",
    "PhoneticRescueResult",
    "PhoneticRescueTrace",
]
