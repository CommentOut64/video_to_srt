"""
统一切分器（草稿/定稿统一出口）。
V3.2.0+dev.20260203.02
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp
from app.services.model_runtime_config_service import get_model_runtime_config_service
from app.services.punctuation.base import PuncPosition
from app.services.punctuation.final_splitter import FinalSplitter, FinalSplitConfig


_STRONG_END_PUNCT = set("。？！.!?")
_PUNCTUATION_SET = set(",.!?;:\"()[]{}，。！？；：、（）【】《》“”‘’「」『』")
_EN_WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")


@dataclass
class DraftSplitConfig:
    """草稿切分配置（对齐 SemanticBuffer 规则）。"""

    min_subtitle_chars_zh_ja: int = 4
    min_subtitle_words_en: int = 4
    min_subtitle_duration_sec: float = 0.8
    fast_delay_budget_sec: float = 2.0
    min_sentence_gap: float = 0.3
    hard_limit_duration: float = 20.0
    force_split_on_sentence_end_punct: bool = True
    soft_pause: float = 0.5
    long_pause: float = 1.0
    min_tokens: int = 1
    max_tokens: int = 60
    min_mapping_coverage: float = 0.6
    keep_sentence_end_punct: bool = False

    @classmethod
    def from_runtime_config(cls) -> "DraftSplitConfig":
        runtime = get_model_runtime_config_service().get_effective_runtime_global()
        punct = runtime.get("effective", {}).get("punctuation", {})
        return cls(
            min_subtitle_chars_zh_ja=int(punct.get("min_subtitle_chars_zh_ja", cls.min_subtitle_chars_zh_ja)),
            min_subtitle_words_en=int(punct.get("min_subtitle_words_en", cls.min_subtitle_words_en)),
            min_subtitle_duration_sec=float(punct.get("min_subtitle_duration_sec", cls.min_subtitle_duration_sec)),
            fast_delay_budget_sec=float(punct.get("fast_delay_budget_sec", cls.fast_delay_budget_sec)),
            min_sentence_gap=float(punct.get("min_sentence_gap", cls.min_sentence_gap)),
            hard_limit_duration=float(punct.get("hard_limit_duration", cls.hard_limit_duration)),
            force_split_on_sentence_end_punct=bool(
                punct.get("force_split_on_sentence_end_punct", cls.force_split_on_sentence_end_punct)
            ),
            keep_sentence_end_punct=bool(
                punct.get("keep_sentence_end_punct", cls.keep_sentence_end_punct)
            ),
        )


class UnifiedSplitter:
    """统一切分器：草稿/定稿共用最终切分入口。"""

    def __init__(
        self,
        *,
        draft_config: Optional[DraftSplitConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.draft_config = draft_config or DraftSplitConfig.from_runtime_config()
        self._draft_splitter = FinalSplitter(
            FinalSplitConfig(
                min_tokens=max(1, self.draft_config.min_tokens),
                max_tokens=max(10, self.draft_config.max_tokens),
                min_duration=max(0.0, self.draft_config.min_subtitle_duration_sec),
                max_duration=max(1.0, self.draft_config.hard_limit_duration),
                soft_pause=max(0.0, self.draft_config.soft_pause),
                long_pause=max(self.draft_config.soft_pause, self.draft_config.long_pause),
            ),
            logger=self.logger,
        )
        self._final_splitter = FinalSplitter(logger=self.logger)

    def split_draft_from_sv(
        self,
        sv_result: Dict[str, Any],
        *,
        chunk_start: float,
        chunk_end: float,
        is_final_output: bool,
        language: Optional[str] = None,
    ) -> List[SentenceSegment]:
        """使用草稿配置切分 SenseVoice 输出。"""
        text_clean = sv_result.get("text_clean", "")
        text_display = sv_result.get("text_itn_raw") or text_clean
        words_data = sv_result.get("words", [])
        words = self._build_words(words_data)
        if not words:
            return self._build_fallback_sentence(
                text_display=text_display,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                confidence=float(sv_result.get("confidence", 0.5)),
                is_final_output=is_final_output,
            )

        detected_language = language or sv_result.get("language", "auto")
        self._draft_splitter.set_language(detected_language)
        punct_positions, punct_clean_text = self._extract_punctuation_positions(sv_result, text_clean)
        if self.draft_config.keep_sentence_end_punct:
            words = self._inject_punctuation_into_words(words, punct_clean_text, punct_positions)
        sentences = self._draft_splitter.split(
            words,
            clean_text=punct_clean_text,
            punctuation_positions=punct_positions,
        )
        sentences = self._merge_short_sentences(sentences, detected_language)
        self._apply_sentence_flags(
            sentences,
            chunk_start=chunk_start,
            is_final_output=is_final_output,
        )
        if not self.draft_config.keep_sentence_end_punct:
            self._strip_sentence_end_punct(sentences)
        return sentences

    def split_final_from_sv(
        self,
        sv_result: Dict[str, Any],
        *,
        chunk_start: float,
        chunk_end: float,
        language: Optional[str] = None,
    ) -> List[SentenceSegment]:
        """定稿路径使用最终切分器处理 SenseVoice 输出。"""
        text_clean = sv_result.get("text_clean", "")
        text_display = sv_result.get("text_itn_raw") or text_clean
        words_data = sv_result.get("words", [])
        words = self._build_words(words_data)
        if not words:
            return self._build_fallback_sentence(
                text_display=text_display,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                confidence=float(sv_result.get("confidence", 0.5)),
                is_final_output=True,
            )

        detected_language = language or sv_result.get("language", "auto")
        self._final_splitter.set_language(detected_language)
        punct_positions, punct_clean_text = self._extract_punctuation_positions(sv_result, text_clean)
        sentences = self._final_splitter.split(
            words,
            clean_text=punct_clean_text,
            punctuation_positions=punct_positions,
        )
        self._apply_sentence_flags(
            sentences,
            chunk_start=chunk_start,
            is_final_output=True,
        )
        if not self.draft_config.keep_sentence_end_punct:
            self._strip_sentence_end_punct(sentences)
        return sentences

    def _merge_short_sentences(
        self,
        sentences: List[SentenceSegment],
        language: str,
    ) -> List[SentenceSegment]:
        if len(sentences) <= 1:
            return sentences

        locked_flags = self._build_locked_flags(sentences)
        merged: List[SentenceSegment] = []
        merged_locked: List[bool] = []
        i = 0
        strategy = self._get_language_strategy(language)

        while i < len(sentences):
            current = sentences[i]
            current_locked = locked_flags[i] if i < len(locked_flags) else False

            if not self._is_short_sentence(current, language):
                merged.append(current)
                merged_locked.append(current_locked)
                i += 1
                continue

            prev = merged[-1] if merged else None
            prev_locked = merged_locked[-1] if merged_locked else False
            next_sent = sentences[i + 1] if i + 1 < len(sentences) else None
            next_locked = locked_flags[i + 1] if i + 1 < len(locked_flags) else False

            prefer_forward = self._prefer_forward_merge(
                current,
                next_sent,
                strategy=strategy,
            )

            if (
                prev
                and not prefer_forward
                and not prev_locked
                and self._can_merge_sentences(prev, current)
            ):
                merged[-1] = self._merge_sentence_pair(prev, current)
                merged_locked[-1] = current_locked
                i += 1
                continue

            if (
                next_sent
                and not current_locked
                and self._can_merge_sentences(current, next_sent)
            ):
                merged.append(self._merge_sentence_pair(current, next_sent))
                merged_locked.append(next_locked)
                i += 2
                continue

            if prev and not prev_locked and self._can_merge_sentences(prev, current):
                merged[-1] = self._merge_sentence_pair(prev, current)
                merged_locked[-1] = current_locked
                i += 1
                continue

            merged.append(current)
            merged_locked.append(current_locked)
            i += 1

        return merged

    def _build_locked_flags(self, sentences: List[SentenceSegment]) -> List[bool]:
        if not self.draft_config.force_split_on_sentence_end_punct:
            return [False for _ in sentences]
        flags: List[bool] = []
        for sentence in sentences:
            text = (sentence.text or "").strip()
            flags.append(bool(text) and text[-1] in _STRONG_END_PUNCT)
        return flags

    def _is_short_sentence(self, sentence: SentenceSegment, language: str) -> bool:
        duration = max(sentence.end - sentence.start, 0.0)
        return self._is_short_segment(sentence.text_clean or sentence.text, duration, language)

    def _is_short_segment(self, text: str, duration: float, language: str) -> bool:
        if duration < self.draft_config.min_subtitle_duration_sec:
            return True
        unit_count = self._count_units(text, language)
        if self._resolve_language_group(text, language) == "en":
            return unit_count < self.draft_config.min_subtitle_words_en
        return unit_count < self.draft_config.min_subtitle_chars_zh_ja

    def _count_units(self, text: str, language: str) -> int:
        normalized = self._strip_trailing_punctuation(text).strip()
        if not normalized:
            return 0
        if self._resolve_language_group(normalized, language) == "en":
            return len(_EN_WORD_PATTERN.findall(normalized))
        return sum(1 for ch in normalized if ch not in _PUNCTUATION_SET and not ch.isspace())

    @staticmethod
    def _resolve_language_group(text: str, language: str) -> str:
        lang = (language or "auto").lower()
        if lang.startswith(("zh", "yue", "ja", "jp")):
            return "cjk"
        if lang.startswith("en"):
            return "en"
        if lang == "auto":
            if _is_likely_english(text):
                return "en"
            if any("\u4e00" <= char <= "\u9fff" for char in text):
                return "cjk"
        return "en"

    def _prefer_forward_merge(
        self,
        current: SentenceSegment,
        next_sent: Optional[SentenceSegment],
        *,
        strategy: Optional[Any],
    ) -> bool:
        if strategy:
            try:
                if strategy.is_incomplete_ending((current.text_clean or current.text).strip()):
                    return True
            except Exception:
                pass
        if next_sent:
            gap = max(next_sent.start - current.end, 0.0)
            if gap < self.draft_config.min_sentence_gap:
                return True
            if strategy:
                try:
                    if strategy.is_continuation((next_sent.text_clean or next_sent.text).strip()):
                        return True
                except Exception:
                    pass
        return False

    def _can_merge_sentences(self, left: SentenceSegment, right: SentenceSegment) -> bool:
        merged_duration = max(right.end - left.start, 0.0)
        if self.draft_config.hard_limit_duration > 0.0 and merged_duration > self.draft_config.hard_limit_duration:
            return False
        return True

    @staticmethod
    def _merge_sentence_pair(left: SentenceSegment, right: SentenceSegment) -> SentenceSegment:
        words = list(left.words) + list(right.words)
        return SentenceSegment(
            text=f"{left.text}{right.text}",
            text_clean=f"{(left.text_clean or left.text)}{(right.text_clean or right.text)}",
            start=left.start,
            end=right.end,
            words=words,
            confidence=min(left.confidence or 0.0, right.confidence or 0.0),
            confidence_display_raw=left.confidence_display_raw,
            confidence_source=left.confidence_source or right.confidence_source,
            source=left.source,
            is_draft=left.is_draft,
            is_finalized=left.is_finalized,
            warning_type=left.warning_type or right.warning_type,
        )

    @staticmethod
    def _strip_trailing_punctuation(text: str) -> str:
        if not text:
            return text
        idx = len(text) - 1
        while idx >= 0 and text[idx] in _PUNCTUATION_SET:
            idx -= 1
        return text[: idx + 1]

    @staticmethod
    def _get_language_strategy(language: str) -> Optional[Any]:
        try:
            from app.services.sentence_splitter import get_language_strategy
        except Exception:
            return None
        try:
            return get_language_strategy(language)
        except Exception:
            return None

    def _apply_sentence_flags(
        self,
        sentences: List[SentenceSegment],
        *,
        chunk_start: float,
        is_final_output: bool,
    ) -> None:
        for sentence in sentences:
            sentence.start += chunk_start
            sentence.end += chunk_start
            sentence.source = TextSource.SENSEVOICE
            sentence.is_finalized = is_final_output
            sentence.is_draft = not is_final_output
            for word in sentence.words:
                word.start += chunk_start
                word.end += chunk_start

    @staticmethod
    def _strip_sentence_end_punct(sentences: List[SentenceSegment]) -> None:
        for sentence in sentences:
            sentence.text = _strip_trailing_sentence_punct(sentence.text)
            if sentence.text_clean:
                sentence.text_clean = _strip_trailing_sentence_punct(sentence.text_clean)
            if sentence.words:
                last_word = sentence.words[-1]
                last_word.word = _strip_trailing_sentence_punct(last_word.word)

    @staticmethod
    def _build_words(words_data: List[Dict[str, Any]]) -> List[WordTimestamp]:
        words: List[WordTimestamp] = []
        for word in words_data:
            words.append(
                WordTimestamp(
                    word=word.get("word", ""),
                    start=word.get("start", 0.0),
                    end=word.get("end", 0.0),
                    confidence=word.get("confidence", 1.0),
                    confidence_raw=word.get("confidence_raw"),
                    confidence_display_raw=word.get("confidence_display_raw"),
                    token_type=word.get("token_type"),
                )
            )
        return words

    def _build_fallback_sentence(
        self,
        *,
        text_display: str,
        chunk_start: float,
        chunk_end: float,
        confidence: float,
        is_final_output: bool,
    ) -> List[SentenceSegment]:
        if not text_display or not text_display.strip():
            self.logger.warning("SenseVoice 结果没有字级时间戳且无文本，无法分句")
            return []
        self.logger.warning(
            "SenseVoice 没有字级时间戳但有文本，创建兜底单句: text='%s...', chunk=[%.2fs, %.2fs]",
            text_display[:50],
            chunk_start,
            chunk_end,
        )
        return [
            SentenceSegment(
                text=text_display.strip(),
                start=chunk_start,
                end=chunk_end,
                words=[],
                source=TextSource.SENSEVOICE,
                confidence=confidence,
                is_finalized=is_final_output,
                is_draft=not is_final_output,
            )
        ]

    @staticmethod
    def _extract_punctuation_positions(
        sv_result: Dict[str, Any],
        text_clean: str,
    ) -> Tuple[List[PuncPosition], Optional[str]]:
        metadata = sv_result.get("metadata", {})
        punct = metadata.get("punctuation", {}) if isinstance(metadata, dict) else {}
        positions_raw = punct.get("punctuation_positions", [])
        positions: List[PuncPosition] = []
        if isinstance(positions_raw, list):
            for item in positions_raw:
                if not isinstance(item, dict):
                    continue
                positions.append(
                    PuncPosition(
                        char_index=int(item.get("char_index", -1)),
                        punctuation=str(item.get("punctuation", "")),
                        confidence=float(item.get("confidence", 1.0)),
                    )
                )
        return positions, text_clean or None

    def _inject_punctuation_into_words(
        self,
        words: List[WordTimestamp],
        clean_text: Optional[str],
        positions: Sequence[PuncPosition],
    ) -> List[WordTimestamp]:
        """将标点位置回写到词级 token，保证草稿输出保留句末标点。"""
        if not words or not clean_text or not positions:
            return words

        mapping, coverage = self._build_char_to_word_map(clean_text, words)
        if coverage < self.draft_config.min_mapping_coverage:
            self.logger.debug(
                "草稿标点回写覆盖率过低=%.2f，跳过回写",
                coverage,
            )
            return words

        by_word: Dict[int, List[str]] = {}
        for pos in sorted(positions, key=lambda item: item.char_index):
            if pos.char_index < 0 or pos.char_index >= len(mapping):
                continue
            word_index = mapping[pos.char_index]
            if word_index is None:
                word_index = self._fallback_word_index(mapping, pos.char_index)
            if word_index is None:
                continue
            by_word.setdefault(word_index, []).append(pos.punctuation)

        for idx, punct_list in by_word.items():
            if idx < 0 or idx >= len(words):
                continue
            token = words[idx].word or ""
            for punct in punct_list:
                if token.endswith(punct):
                    continue
                token = f"{token}{punct}"
            words[idx].word = token

        return words

    def _build_char_to_word_map(
        self,
        clean_text: str,
        words: List[WordTimestamp],
    ) -> Tuple[List[Optional[int]], float]:
        mapping: List[Optional[int]] = [None] * len(clean_text)
        total_chars = self._count_nonspace(clean_text)
        matched_chars = 0
        cursor = 0
        for idx, word in enumerate(words):
            token = self._normalize_token_for_match(word.word or "")
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
    def _normalize_token_for_match(token: str) -> str:
        token = token.replace("▁", " ").strip()
        token = token.rstrip("".join(_PUNCTUATION_SET))
        return token

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


def _is_likely_english(text: str) -> bool:
    if not text:
        return False
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return False
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 3:
        return False
    ascii_letters = sum(1 for char in letters if char.isascii())
    return (ascii_letters / len(letters)) >= 0.8


def _strip_trailing_sentence_punct(text: Optional[str]) -> str:
    """
    句末标点清理（保留问号/感叹号）。

    说明：对齐层与草稿切分保持一致，仅移除句号/逗号/顿号/分号。
    """
    if not text:
        return ""
    punct_to_remove = {"。", ".", "，", ",", "、", "；", ";"}
    idx = len(text) - 1
    while idx >= 0 and text[idx] in punct_to_remove:
        idx -= 1
    return text[: idx + 1]
