"""
语义注入器（词边界标点注入）。
V3.2.0+dev.20260203.03
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from app.models.confidence_models import AlignedWord
from app.models.sensevoice_models import WordTimestamp
from app.services.punctuation.base import PuncPosition, apply_punctuation
from app.services.punctuation.postprocess import build_clean_text


_LEFT_QUOTES = set("“‘「『《（【")
_RIGHT_QUOTES = set("”’」』》）】")
_AMBIGUOUS_QUOTES = set("\"'")
_PUNCTUATION_SET = set(",.!?;:\"()[]{}，。！？；：、（）【】《》“”‘’「」『』")


@dataclass
class AnnotatedWord:
    """语义注入后的词信息。"""

    word: str
    start: Optional[float]
    end: Optional[float]
    trailing_punct: str = ""


@dataclass
class SemanticInjectionResult:
    """语义注入结果。"""

    annotated_words: List[AnnotatedWord]
    punctuated_text: str
    unmatched_positions: List[PuncPosition] = field(default_factory=list)
    mapping_coverage: float = 0.0
    is_mapping_blocked: bool = False


class SemanticInjector:
    """语义注入器（管道模式：按规则顺序注入标点，保持时间戳不变）。"""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        min_mapping_coverage: float = 0.6,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._min_mapping_coverage = min_mapping_coverage

    def inject(
        self,
        aligned_words: List[AlignedWord],
        clean_text: str,
        positions: List[PuncPosition],
    ) -> SemanticInjectionResult:
        """将标点注入到对齐词流（仅改文本，不改时间戳）。"""
        if not aligned_words:
            return SemanticInjectionResult(annotated_words=[], punctuated_text=clean_text)

        if not clean_text or not positions:
            annotated = [
                AnnotatedWord(word=word.word, start=word.start, end=word.end)
                for word in aligned_words
            ]
            return SemanticInjectionResult(annotated_words=annotated, punctuated_text=clean_text)

        char_to_word, coverage = self._build_char_to_word_map(clean_text, aligned_words)
        if coverage < self._min_mapping_coverage:
            annotated = [
                AnnotatedWord(word=word.word, start=word.start, end=word.end)
                for word in aligned_words
            ]
            self._logger.debug(
                "语义注入映射覆盖率过低=%.2f，跳过标点注入",
                coverage,
            )
            return SemanticInjectionResult(
                annotated_words=annotated,
                punctuated_text=clean_text,
                unmatched_positions=positions,
                mapping_coverage=coverage,
                is_mapping_blocked=True,
            )
        leading_marks: List[List[str]] = [[] for _ in aligned_words]
        trailing_marks: List[List[str]] = [[] for _ in aligned_words]
        unmatched: List[PuncPosition] = []
        quote_stack: List[str] = []

        for position in sorted(positions, key=lambda item: item.char_index):
            idx = position.char_index
            if idx < 0 or idx >= len(clean_text):
                unmatched.append(position)
                continue

            word_index = char_to_word[idx] if idx < len(char_to_word) else None
            if word_index is None:
                word_index = self._fallback_word_index(char_to_word, idx)
            if word_index is None:
                unmatched.append(position)
                continue

            punct = position.punctuation
            if punct in _LEFT_QUOTES:
                leading_marks[word_index].append(punct)
                quote_stack.append(punct)
                continue
            if punct in _RIGHT_QUOTES:
                trailing_marks[word_index].append(punct)
                if quote_stack:
                    quote_stack.pop()
                continue
            if punct in _AMBIGUOUS_QUOTES:
                if quote_stack:
                    trailing_marks[word_index].append(punct)
                    quote_stack.pop()
                else:
                    leading_marks[word_index].append(punct)
                    quote_stack.append(punct)
                continue

            trailing_marks[word_index].append(punct)

        annotated_words: List[AnnotatedWord] = []
        for idx, word in enumerate(aligned_words):
            leading = "".join(leading_marks[idx])
            trailing = "".join(trailing_marks[idx])
            annotated_words.append(
                AnnotatedWord(
                    word=f"{leading}{word.word}",
                    start=word.start,
                    end=word.end,
                    trailing_punct=trailing,
                )
            )

        punctuated_text = apply_punctuation(clean_text, positions)
        if unmatched:
            self._logger.debug(
                "语义注入未命中标点=%d，clean_text_len=%d",
                len(unmatched),
                len(clean_text),
            )
        return SemanticInjectionResult(
            annotated_words=annotated_words,
            punctuated_text=punctuated_text,
            unmatched_positions=unmatched,
            mapping_coverage=coverage,
        )

    @staticmethod
    def build_word_timestamps(
        aligned_words: List[AlignedWord],
        annotated_words: List[AnnotatedWord],
    ) -> List[WordTimestamp]:
        """将注入后的词信息转换为 WordTimestamp 列表。"""
        words: List[WordTimestamp] = []
        for source, annotated in zip(aligned_words, annotated_words):
            word_text = f"{annotated.word}{annotated.trailing_punct}"
            words.append(
                WordTimestamp(
                    word=word_text,
                    start=source.start,
                    end=source.end,
                    confidence=source.final_confidence,
                    is_pseudo=source.is_pseudo,
                )
            )
        return words

    @staticmethod
    def extract_raw_punctuation_positions(raw_text: str) -> Tuple[str, List[PuncPosition]]:
        """从原文中提取标点位置（以 clean_text 为基准）。"""
        if not raw_text:
            return "", []
        clean_text, _, raw_to_clean = build_clean_text(raw_text)
        positions: List[PuncPosition] = []
        for raw_idx, char in enumerate(raw_text):
            if char not in _PUNCTUATION_SET:
                continue
            if raw_to_clean[raw_idx] is not None:
                continue
            prev_clean = None
            for back in range(raw_idx - 1, -1, -1):
                prev_clean = raw_to_clean[back]
                if prev_clean is not None:
                    break
            if prev_clean is None:
                continue
            positions.append(PuncPosition(char_index=prev_clean, punctuation=char, confidence=1.0))
        return clean_text, positions

    def _build_char_to_word_map(
        self,
        clean_text: str,
        aligned_words: List[AlignedWord],
    ) -> tuple[List[Optional[int]], float]:
        mapping: List[Optional[int]] = [None] * len(clean_text)
        total_chars = sum(1 for ch in clean_text if not ch.isspace())
        matched_chars = 0
        cursor = 0
        for idx, word in enumerate(aligned_words):
            token = word.word
            if not token:
                continue
            cursor = self._skip_whitespace(clean_text, cursor)
            if cursor >= len(clean_text):
                break
            if clean_text.startswith(token, cursor):
                for offset in range(len(token)):
                    if cursor + offset < len(mapping):
                        mapping[cursor + offset] = idx
                matched_chars += self._count_nonspace(clean_text[cursor:cursor + len(token)])
                cursor += len(token)
                continue

            found = clean_text.find(token, cursor)
            if found != -1:
                for offset in range(len(token)):
                    if found + offset < len(mapping):
                        mapping[found + offset] = idx
                matched_chars += self._count_nonspace(clean_text[found:found + len(token)])
                cursor = found + len(token)
                continue

            for offset in range(len(token)):
                if cursor + offset < len(mapping):
                    mapping[cursor + offset] = idx
            cursor = min(len(clean_text), cursor + len(token))

        last_idx: Optional[int] = None
        for i, value in enumerate(mapping):
            if value is not None:
                last_idx = value
            elif last_idx is not None:
                mapping[i] = last_idx
        coverage = matched_chars / max(total_chars, 1)
        return mapping, coverage

    @staticmethod
    def _count_nonspace(text: str) -> int:
        return sum(1 for ch in text if not ch.isspace())

    @staticmethod
    def _skip_whitespace(text: str, cursor: int) -> int:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        return cursor

    @staticmethod
    def _fallback_word_index(mapping: List[Optional[int]], idx: int) -> Optional[int]:
        for back in range(idx, -1, -1):
            if mapping[back] is not None:
                return mapping[back]
        return None
