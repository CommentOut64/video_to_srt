"""SentenceSegmenter：基于终稿流的独立切分层。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from app.models.sensevoice_models import SentenceSegment, WordTimestamp
from app.services.timeanchored_alignment.contracts import AlignmentItem, ProtectedSpan


_STRONG_END_PUNCT = frozenset({"。", "！", "？", ".", "!", "?"})


@dataclass(frozen=True)
class SentenceSegmenterConfig:
    long_pause_gap_sec: float = 0.65
    max_segment_duration_sec: float = 8.0
    max_tokens_per_segment: int = 40


class SentenceSegmenter:
    """终稿切分器。"""

    def __init__(self, *, config: SentenceSegmenterConfig | None = None) -> None:
        self._config = config or SentenceSegmenterConfig()

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

        text = "".join(item.text for item in stream)
        if protected_spans:
            resolved_spans = tuple(protected_spans)
        else:
            from app.services.text_protection import extract_protected_spans

            resolved_spans = tuple(extract_protected_spans(text))
        boundary_chars = self._build_boundary_char_positions(stream)
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

    @staticmethod
    def _build_boundary_char_positions(stream: Sequence[AlignmentItem]) -> list[int]:
        positions: list[int] = []
        cursor = 0
        for index, item in enumerate(stream):
            cursor += len(str(item.text or ""))
            if index < len(stream) - 1:
                positions.append(cursor)
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

    @staticmethod
    def _build_sentence(
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
        text = "".join(row.text for row in items)
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
