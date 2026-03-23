"""SentenceSegmenter：基于终稿流的独立切分层。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from app.models.sensevoice_models import SentenceSegment, WordTimestamp
from app.services.timeanchored_alignment.contracts import AlignmentItem, ProtectedSpan


_STRONG_END_PUNCT = frozenset({"。", "！", "？", ".", "!", "?"})
_CJK_LANGS = frozenset({"zh", "ja", "ko"})
_CJK_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_WORD_CHAR_RE = re.compile(r"[A-Za-z0-9]")
_NO_SPACE_BEFORE = frozenset({",", ".", "!", "?", ";", ":", "%", ")", "]", "}", "。", "，", "！", "？", "；", "：", "）", "】", "》", "」", "』"})
_NO_SPACE_AFTER = frozenset({"(", "[", "{", "（", "【", "《", "「", "『", "$", "#", "@", "¥", "£"})
_FORCE_SPACE_AFTER = frozenset({",", ".", "!", "?", ";", ")", "]", "}", "，", "。", "！", "？", "；", "）", "】", "》", "」", "』"})


@dataclass(frozen=True)
class SentenceSegmenterConfig:
    long_pause_gap_sec: float = 0.65
    max_segment_duration_sec: float = 8.0
    max_tokens_per_segment: int = 40


class SentenceSegmenter:
    """终稿切分器。"""

    def __init__(self, *, config: SentenceSegmenterConfig | None = None) -> None:
        self._config = config or SentenceSegmenterConfig()

    def compose_text(
        self,
        *,
        stream: Sequence[AlignmentItem],
        language: str = "auto",
    ) -> str:
        text, _ = self._render_stream_text_and_boundaries(stream=stream, language=language)
        return text

    def segment(
        self,
        *,
        stream: Sequence[AlignmentItem],
        language: str = "auto",
        protected_spans: Sequence[ProtectedSpan] = (),
        speaker_boundaries: Sequence[float] = (),
        blank_valley_boundaries: Sequence[float] = (),
    ) -> list[SentenceSegment]:
        if not stream:
            return []

        text, boundary_chars = self._render_stream_text_and_boundaries(stream=stream, language=language)
        if protected_spans:
            resolved_spans = tuple(protected_spans)
        else:
            from app.services.text_protection import extract_protected_spans

            resolved_spans = tuple(extract_protected_spans(text))
        speaker_marks = tuple(float(v) for v in speaker_boundaries)
        blank_marks = tuple(float(v) for v in blank_valley_boundaries)

        sentences: list[SentenceSegment] = []
        bucket: list[AlignmentItem] = []
        for index, item in enumerate(stream):
            bucket.append(item)
            if index >= len(stream) - 1:
                continue

            next_item = stream[index + 1]
            boundary_char = boundary_chars[index]
            split_reason = self._resolve_split_reason(
                bucket=bucket,
                current=item,
                next_item=next_item,
                boundary_char=boundary_char,
                protected_spans=resolved_spans,
                speaker_boundaries=speaker_marks,
                blank_valley_boundaries=blank_marks,
            )
            if not split_reason:
                continue
            sentences.append(self._build_sentence(bucket, split_reason=split_reason, language=language))
            bucket = []

        if bucket:
            sentences.append(self._build_sentence(bucket, split_reason="tail_flush", language=language))
        return sentences

    def _resolve_split_reason(
        self,
        *,
        bucket: Sequence[AlignmentItem],
        current: AlignmentItem,
        next_item: AlignmentItem,
        boundary_char: int,
        protected_spans: Sequence[ProtectedSpan],
        speaker_boundaries: Sequence[float],
        blank_valley_boundaries: Sequence[float],
    ) -> str | None:
        protected = self._is_boundary_protected(boundary_char=boundary_char, protected_spans=protected_spans)
        if protected:
            return None

        if self._has_boundary_time_mark(
            marks=speaker_boundaries,
            left=float(current.end),
            right=float(next_item.start),
        ):
            return "speaker_change"
        if self._has_boundary_time_mark(
            marks=blank_valley_boundaries,
            left=float(current.end),
            right=float(next_item.start),
        ):
            return "blank_valley"

        gap = float(next_item.start) - float(current.end)
        if gap >= float(self._config.long_pause_gap_sec):
            return "long_pause"

        if self._is_strong_end(current.text):
            return "strong_punct"

        first = bucket[0]
        duration = float(current.end) - float(first.start)
        if duration >= float(self._config.max_segment_duration_sec):
            return "duration_budget"
        if len(bucket) >= int(self._config.max_tokens_per_segment):
            return "token_budget"
        return None

    def _render_stream_text_and_boundaries(
        self,
        *,
        stream: Sequence[AlignmentItem],
        language: str,
    ) -> tuple[str, list[int]]:
        parts: list[str] = []
        boundary_positions: list[int] = []
        cursor = 0
        prev_token = ""
        normalized_language = self._normalize_language_tag(language)
        for index, item in enumerate(stream):
            token = str(item.text or "")
            if index > 0 and self._should_insert_space(prev_token=prev_token, current_token=token, language=normalized_language):
                parts.append(" ")
                cursor += 1
            parts.append(token)
            cursor += len(token)
            if index < len(stream) - 1:
                boundary_positions.append(cursor)
            prev_token = token
        return "".join(parts), boundary_positions

    @staticmethod
    def _normalize_language_tag(language: str) -> str:
        value = str(language or "auto").strip().lower()
        if not value:
            return "auto"
        if "-" in value:
            value = value.split("-", 1)[0]
        if "_" in value:
            value = value.split("_", 1)[0]
        return value or "auto"

    @staticmethod
    def _contains_cjk(token: str) -> bool:
        return bool(_CJK_CHAR_RE.search(str(token or "")))

    @staticmethod
    def _has_word_char(token: str) -> bool:
        return bool(_WORD_CHAR_RE.search(str(token or "")))

    def _should_insert_space(
        self,
        *,
        prev_token: str,
        current_token: str,
        language: str,
    ) -> bool:
        language = self._normalize_language_tag(language)
        if language in _CJK_LANGS:
            return False

        prev = str(prev_token or "").strip()
        curr = str(current_token or "").strip()
        if not prev or not curr:
            return False

        prev_last = prev[-1]
        curr_first = curr[0]

        if curr_first in _NO_SPACE_BEFORE:
            return False
        if prev_last in _NO_SPACE_AFTER:
            return False

        # 时间表达（7:28）和版本号（v1.2）不插入空格。
        if prev_last == ":" and curr_first.isdigit():
            return False
        if prev_last == "." and curr_first.isdigit() and any(ch.isdigit() for ch in prev):
            return False

        # 英文缩写收口：there 's -> there's
        if curr_first in {"'", "’"} and self._has_word_char(prev):
            return False

        # CJK 邻接保留无空格，避免破坏中日文连写习惯。
        if self._contains_cjk(prev) or self._contains_cjk(curr):
            return False

        if self._has_word_char(prev) and self._has_word_char(curr):
            return True
        if prev_last in _FORCE_SPACE_AFTER and (self._has_word_char(curr) or self._contains_cjk(curr)):
            return True
        return False

    def _build_boundary_char_positions(self, stream: Sequence[AlignmentItem], *, language: str) -> list[int]:
        positions: list[int] = []
        _, positions = self._render_stream_text_and_boundaries(stream=stream, language=language)
        return positions

    @staticmethod
    def _is_boundary_protected(*, boundary_char: int, protected_spans: Sequence[ProtectedSpan]) -> bool:
        for span in protected_spans:
            if int(span.start) < int(boundary_char) < int(span.end):
                return True
        return False

    @staticmethod
    def _has_boundary_time_mark(*, marks: Iterable[float], left: float, right: float) -> bool:
        low = min(left, right)
        high = max(left, right)
        for value in marks:
            if low <= float(value) <= high:
                return True
        return False

    @staticmethod
    def _is_strong_end(text: str) -> bool:
        token = str(text or "").rstrip()
        if not token:
            return False
        return token[-1] in _STRONG_END_PUNCT

    def _build_sentence(
        self,
        items: Sequence[AlignmentItem],
        *,
        split_reason: str,
        language: str,
    ) -> SentenceSegment:
        words: list[WordTimestamp] = []
        confidences: list[float] = []
        for row in items:
            words.append(
                WordTimestamp(
                    word=row.text,
                    start=float(row.start),
                    end=float(row.end),
                    confidence=row.confidence,
                    confidence_source=row.source,
                    is_pseudo=row.status in {"failed", "interpolated"},
                )
            )
            if row.confidence is not None:
                confidences.append(float(row.confidence))

        confidence = sum(confidences) / len(confidences) if confidences else 1.0
        text, _ = self._render_stream_text_and_boundaries(stream=items, language=language)
        return SentenceSegment(
            text=text,
            text_clean=text,
            start=float(items[0].start),
            end=float(items[-1].end),
            words=words,
            confidence=confidence,
            confidence_source="timeanchored_alignment",
            split_reason=split_reason,
            whisper_text=text if language else None,
        )


__all__ = ["SentenceSegmenter", "SentenceSegmenterConfig"]
