"""
语义注入器（词边界标点注入）。
V3.2.0+dev.20260205.06
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from app.models.confidence_models import AlignedWord
from app.models.sensevoice_models import WordTimestamp
from app.services.punctuation.base import PuncPosition, WordTimestampLike, apply_punctuation
from app.services.punctuation.postprocess import build_clean_text
from app.services.text_protection import (
    can_merge_word_tokens,
    should_skip_raw_punctuation,
)


_LEFT_QUOTES = set("“‘「『《（【")
_RIGHT_QUOTES = set("”’」』》）】")
_AMBIGUOUS_QUOTES = set("\"'")
_PUNCTUATION_SET = set(",.!?;:\"()[]{}，。！？；：、（）【】《》“”‘’「」『』")
_WEAK_PUNCTUATION_SET = set(",，、;；:：")


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
        *,
        language: Optional[str] = None,
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

        char_to_word, coverage = self._build_char_to_word_map(
            clean_text,
            aligned_words,
            language,
        )
        if coverage < self._min_mapping_coverage:
            annotated = [
                AnnotatedWord(word=word.word, start=word.start, end=word.end)
                for word in aligned_words
            ]
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
                    confidence_source=source.confidence_source,
                    is_pseudo=source.is_pseudo,
                )
            )
        return words

    # V3.2.0+dev.20260203.08: 支持外部传入 clean_text/raw_to_clean，统一清洗口径。
    @staticmethod
    def extract_raw_punctuation_positions(
        raw_text: str,
        *,
        clean_text: Optional[str] = None,
        raw_to_clean: Optional[Sequence[Optional[int]]] = None,
    ) -> Tuple[str, List[PuncPosition]]:
        """从原文中提取标点位置（以 clean_text 为基准）。"""
        if not raw_text:
            return "", []
        if clean_text is None or raw_to_clean is None:
            clean_text, _, raw_to_clean = build_clean_text(raw_text)
        if raw_to_clean is None:
            return clean_text or "", []
        positions: List[PuncPosition] = []
        for raw_idx, char in enumerate(raw_text):
            if char not in _PUNCTUATION_SET:
                continue
            if should_skip_raw_punctuation(raw_text, raw_idx, char):
                continue
            if raw_idx >= len(raw_to_clean):
                continue
            if raw_to_clean[raw_idx] is not None:
                continue
            prev_clean = None
            for back in range(raw_idx - 1, -1, -1):
                if back >= len(raw_to_clean):
                    continue
                prev_clean = raw_to_clean[back]
                if prev_clean is not None:
                    break
            if prev_clean is None:
                continue
            positions.append(PuncPosition(char_index=prev_clean, punctuation=char, confidence=1.0))
        return clean_text, positions

    # V3.2.0+dev.20260204.10: Whisper 词级标点提取（基于词时间戳对齐 clean_text）。
    @staticmethod
    def extract_word_punctuation_positions(
        word_timestamps: Sequence[WordTimestampLike],
        *,
        clean_text: str,
        min_mapping_coverage: float = 0.6,
        max_weak_ratio: float = 0.8,  # 保留参数以保持向后兼容，但不再使用
        language: Optional[str] = None,  # 保留参数以保持向后兼容，但不再使用
    ) -> Tuple[str, List[PuncPosition], float]:
        """基于词级时间戳提取标点位置（以 clean_text 为基准）。
        
        注意：max_weak_ratio 和 language 参数保留用于向后兼容，但自 V3.2.0+dev.20260205.06 起已不再使用。
        弱标点过滤逻辑已移除，因为密集标点问题已从根源修复。
        """
        """基于词级时间戳提取标点位置（以 clean_text 为基准）。"""
        if not clean_text or not word_timestamps:
            return clean_text or "", [], 0.0

        tokens: List[Tuple[str, str, str, float]] = []
        for item in word_timestamps:
            raw = SemanticInjector._get_word_text(item)
            token = raw.replace("▁", " ").strip()
            if not token:
                tokens.append(("", "", "", 0.0))
                continue
            leading, core, trailing = SemanticInjector._split_word_token(token)
            confidence = SemanticInjector._get_word_confidence(item)
            tokens.append((leading, core, trailing, confidence))

        spans: List[Optional[Tuple[int, int]]] = [None] * len(tokens)
        cursor = 0
        matched_chars = 0
        matched_tokens = 0
        total_chars = SemanticInjector._count_nonspace(clean_text)
        for idx, (_, core, _, _) in enumerate(tokens):
            if not core:
                continue
            cursor = SemanticInjector._skip_whitespace(clean_text, cursor)
            if cursor >= len(clean_text):
                break
            match_idx = SemanticInjector._find_match_index(clean_text, core, cursor)
            if match_idx == -1:
                continue
            start = max(match_idx, 0)
            end = min(match_idx + len(core) - 1, len(clean_text) - 1)
            spans[idx] = (start, end)
            matched_tokens += 1
            matched_chars += SemanticInjector._count_nonspace(clean_text[start:end + 1])
            cursor = min(end + 1, len(clean_text))

        coverage = matched_chars / max(total_chars, 1)
        # V3.2.0+dev.20260205.03: 输出词级标点映射细节，定位弱标点密集来源
        logger = logging.getLogger(__name__)
        leading_tokens = sum(1 for leading, _, _, _ in tokens if leading)
        trailing_tokens = sum(1 for _, _, trailing, _ in tokens if trailing)
        logger.debug(
            "词级标点映射统计: clean_len=%d tokens=%d matched_tokens=%d coverage=%.2f leading_tokens=%d trailing_tokens=%d",
            len(clean_text),
            len(tokens),
            matched_tokens,
            coverage,
            leading_tokens,
            trailing_tokens,
        )
        if coverage < min_mapping_coverage or matched_tokens <= 0:
            logger.debug(
                "词级标点映射覆盖不足: coverage=%.2f min_cov=%.2f matched_tokens=%d",
                coverage,
                min_mapping_coverage,
                matched_tokens,
            )
            return clean_text, [], coverage

        positions: List[PuncPosition] = []
        weak_count = 0
        for idx, (leading, _, trailing, confidence) in enumerate(tokens):
            span = spans[idx]
            if span is None:
                continue
            start, end = span
            core = tokens[idx][1]
            prev_core = tokens[idx - 1][1] if idx > 0 else ""
            next_core = tokens[idx + 1][1] if idx + 1 < len(tokens) else ""
            for ch in leading:
                if ch in _PUNCTUATION_SET:
                    # V3.2.0+dev.20260218.03: 保护规则 - 跳过跨 token 小数点前缀注入（如 1 | .4）。
                    if can_merge_word_tokens(prev_core, f"{ch}{core}"):
                        continue
                    positions.append(PuncPosition(char_index=start, punctuation=ch, confidence=confidence))
            for ch in trailing:
                if ch in _PUNCTUATION_SET:
                    # V3.2.0+dev.20260218.03: 保护规则 - 跳过跨 token 小数点后缀注入（如 0. | 15）。
                    if can_merge_word_tokens(f"{core}{ch}", next_core):
                        continue
                    if ch in _WEAK_PUNCTUATION_SET:
                        weak_count += 1
                    positions.append(PuncPosition(char_index=end, punctuation=ch, confidence=confidence))

        # V3.2.0+dev.20260205.06: 移除弱标点过滤逻辑
        # 原因：密集标点问题已从根源修复（Prompt格式 + condition_on_previous_text禁用）
        # 弱标点过滤会误伤英文等语言的正常逗号，且标点前置域后处理已有规则引擎过滤异常标点
        # 保留 weak_count 统计用于调试，但不再执行过滤

        if not positions:
            return clean_text, [], coverage

        deduped: List[PuncPosition] = []
        seen: set[tuple[int, str]] = set()
        for position in sorted(positions, key=lambda item: item.char_index):
            key = (int(position.char_index), str(position.punctuation))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(position)
        return clean_text, deduped, coverage

    @staticmethod
    def _get_word_text(item: WordTimestampLike) -> str:
        if isinstance(item, dict):
            return str(item.get("word", "") or "")
        return str(getattr(item, "word", "") or "")

    @staticmethod
    def _get_word_confidence(item: WordTimestampLike) -> float:
        if isinstance(item, dict):
            value = item.get("confidence")
            if value is None:
                value = item.get("probability")
            return float(value or 0.0)
        value = getattr(item, "confidence", None)
        if value is None:
            value = getattr(item, "probability", None)
        return float(value or 0.0)

    @staticmethod
    def _split_word_token(token: str) -> Tuple[str, str, str]:
        if not token:
            return "", "", ""
        start = 0
        end = len(token)
        leading: List[str] = []
        trailing: List[str] = []
        while start < end and token[start] in _PUNCTUATION_SET:
            leading.append(token[start])
            start += 1
        while end > start and token[end - 1] in _PUNCTUATION_SET:
            trailing.append(token[end - 1])
            end -= 1
        core = token[start:end]
        trailing.reverse()
        return "".join(leading), core, "".join(trailing)

    @staticmethod
    def _count_nonspace(text: str) -> int:
        return sum(1 for ch in text if not ch.isspace())

    @staticmethod
    def _skip_whitespace(text: str, cursor: int) -> int:
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        return cursor

    # V3.2.0+dev.20260206.02: 词级标点映射优先选择“最早可匹配位置”，并支持大小写无关匹配，
    # 防止 p.m. -> PM 这类缩写因大小写差异漂移到后续词（例如误落到 plenty/time）。
    @staticmethod
    def _find_match_index(text: str, token: str, cursor: int) -> int:
        if not token:
            return -1
        exact_idx = SemanticInjector._fallback_match(
            text,
            token,
            cursor,
            ignore_case=False,
        )
        ignore_case_idx = SemanticInjector._fallback_match(
            text,
            token,
            cursor,
            ignore_case=True,
        )
        candidates = [idx for idx in (exact_idx, ignore_case_idx) if idx != -1]
        if not candidates:
            return -1
        return min(candidates)

    @staticmethod
    def _fallback_match(
        text: str,
        token: str,
        cursor: int,
        *,
        ignore_case: bool = False,
    ) -> int:
        if not token:
            return -1
        if cursor >= len(text):
            return -1
        if not ignore_case:
            if text[cursor: cursor + len(token)] == token:
                return cursor
            return text.find(token, cursor)

        token_lower = token.lower()
        if text[cursor: cursor + len(token)].lower() == token_lower:
            return cursor
        upper_bound = len(text) - len(token) + 1
        for idx in range(cursor, max(cursor, upper_bound)):
            if text[idx: idx + len(token)].lower() == token_lower:
                return idx
        return -1

    # V3.2.0+dev.20260203.09: 语言感知映射（跳过空白 token + 规范化匹配 + 拼接兜底）。
    def _build_char_to_word_map(
        self,
        clean_text: str,
        aligned_words: List[AlignedWord],
        language: Optional[str],
    ) -> tuple[List[Optional[int]], float]:
        mapping: List[Optional[int]] = [None] * len(clean_text)
        total_chars = sum(1 for ch in clean_text if not ch.isspace())
        matched_chars = 0
        cursor = 0
        for idx, word in enumerate(aligned_words):
            raw_token = word.word or ""
            if self._is_whitespace_token(raw_token):
                cursor = self._skip_whitespace(clean_text, cursor)
                continue

            token = self._normalize_token_for_match(raw_token, language)
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
        if coverage < self._min_mapping_coverage:
            fallback_mapping, fallback_coverage = self._build_char_to_word_map_by_concat(
                clean_text,
                aligned_words,
            )
            if fallback_coverage > coverage:
                return fallback_mapping, fallback_coverage
        return mapping, coverage

    @staticmethod
    def _is_whitespace_token(token: str) -> bool:
        return bool(token) and token.isspace()

    @staticmethod
    def _is_cjk_language(language: Optional[str]) -> bool:
        lang = (language or "auto").lower()
        return lang.startswith(("zh", "yue", "ja", "jp", "ko"))

    def _normalize_token_for_match(self, token: str, language: Optional[str]) -> str:
        normalized = token.replace("▁", " ").strip()
        if not normalized:
            return ""
        if self._is_cjk_language(language):
            normalized = normalized.replace(" ", "")
        normalized = normalized.strip("".join(_PUNCTUATION_SET))
        return normalized

    def _build_char_to_word_map_by_concat(
        self,
        clean_text: str,
        aligned_words: List[AlignedWord],
    ) -> tuple[List[Optional[int]], float]:
        concat = "".join(word.word for word in aligned_words)
        if concat != clean_text:
            return [None] * len(clean_text), 0.0
        mapping: List[Optional[int]] = [None] * len(clean_text)
        cursor = 0
        for idx, word in enumerate(aligned_words):
            token = word.word or ""
            if not token:
                continue
            for _ in token:
                if cursor >= len(mapping):
                    break
                mapping[cursor] = idx
                cursor += 1
        coverage = 1.0 if clean_text else 0.0
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
