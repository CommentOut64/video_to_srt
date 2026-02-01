"""
语义缓冲器（Phase C 语义缓冲实现）。
V3.2.0+dev.20260131.04
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Sequence

from app.models.sensevoice_models import SentenceSegment, TextSource, WordTimestamp, WarningType
from app.services.model_runtime_config_service import get_model_runtime_config_service
from app.services.punctuation.base import PunctuationResult, SplitPoint


def _is_decimal_point(text: str, index: int) -> bool:
    """
    判断指定位置的 '.'/'。' 是否为小数点（如 2.2、3.14）。
    V3.2.0+dev.20260201.01
    """
    # V3.2.0+dev.20260131.05: 兼容中文句号被误归一化为小数点场景
    decimal_marks = {".", "。"}
    if index < 0 or index >= len(text):
        return False
    if text[index] not in decimal_marks:
        return False

    # 检查左边是否有数字
    left_idx = index - 1
    while left_idx >= 0 and text[left_idx] in _PUNCTUATION_SET:
        left_idx -= 1
    if left_idx < 0 or not text[left_idx].isdigit():
        return False

    # 检查右边是否有数字
    right_idx = index + 1
    while right_idx < len(text) and text[right_idx] in _PUNCTUATION_SET and text[right_idx] not in decimal_marks:
        right_idx += 1
    if right_idx >= len(text) or not text[right_idx].isdigit():
        return False

    return True


_WEAK_PUNCTUATION = set("，、；,;:")
_PUNCTUATION_SET = set(",.!?;:\"()[]{}，。！？；：、（）【】《》""「」『』")
_TRAILING_PUNCTUATION = set(",，;:；：、。.") 


def _is_removable_punctuation(text: str, index: int) -> bool:
    """
    判断该位置字符是否应从 clean_text 中移除。
    V3.2.0+dev.20260201.14: 句末句号可移除，但小数点必须保留。
    """
    if index < 0 or index >= len(text):
        return False
    char = text[index]
    if char not in _PUNCTUATION_SET:
        return False
    if char in {".", "。"} and _is_decimal_point(text, index):
        return False
    return True
_CONTRACTION_MAP: Dict[str, str] = {
    "im": "i'm",
    "ive": "i've",
    "id": "i'd",
    "youre": "you're",
    "youve": "you've",
    "youd": "you'd",
    "theyre": "they're",
    "theyve": "they've",
    "theyd": "they'd",
    "theres": "there's",
    "thats": "that's",
    "whats": "what's",
    "whos": "who's",
    "wheres": "where's",
    "whens": "when's",
    "whys": "why's",
    "hows": "how's",
    "dont": "don't",
    "doesnt": "doesn't",
    "didnt": "didn't",
    "wont": "won't",
    "shouldnt": "shouldn't",
    "wouldnt": "wouldn't",
    "couldnt": "couldn't",
    "mustnt": "mustn't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hadnt": "hadn't",
    "hes": "he's",
    "shes": "she's",
    "yall": "y'all",
}
_CONTRACTION_PATTERN = re.compile(r"\b[A-Za-z]+\b")
_EN_WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")


def _get_language_strategy(language: str):
    """延迟导入语言策略，避免循环依赖。"""
    try:
        from app.services.sentence_splitter import get_language_strategy
        return get_language_strategy(language)
    except ImportError:
        return None


def _apply_contraction_case(source: str, replacement: str) -> str:
    if not source:
        return replacement
    if source.isupper():
        return replacement.upper()
    if source[0].isupper() and source[1:].islower():
        return replacement[0].upper() + replacement[1:]
    return replacement.lower()


def _restore_english_contractions(text: str) -> str:
    if not text:
        return text

    def repl(match: re.Match[str]) -> str:
        word = match.group(0)
        replacement = _CONTRACTION_MAP.get(word.lower())
        if not replacement:
            return word
        return _apply_contraction_case(word, replacement)

    return _CONTRACTION_PATTERN.sub(repl, text)


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


def _should_restore_contractions(text: str, language: Optional[str]) -> bool:
    if not text:
        return False
    lang = (language or "").lower()
    if lang.startswith("en"):
        return True
    if lang and lang != "auto":
        return False
    return _is_likely_english(text)


@dataclass
class PunctuationDecision:
    """标点调度决策（用于快慢流透传）。"""

    is_slow_requested: bool
    reason: str
    mode: str

    @classmethod
    def from_dict(cls, raw: dict) -> "PunctuationDecision":
        return cls(
            is_slow_requested=bool(raw.get("is_slow_requested", False)),
            reason=str(raw.get("reason", "")),
            mode=str(raw.get("mode", "")),
        )


@dataclass
class SemanticBufferInput:
    """语义缓冲输入数据。"""

    chunk_id: str
    text: str
    audio_range: Tuple[float, float]
    language: str = "auto"
    punctuation_result: Optional[PunctuationResult] = None
    punctuation_decision: Optional[PunctuationDecision] = None
    word_timestamps: Optional[List[Dict[str, Any]]] = None
    raw_tokens: Optional[List[Dict[str, Any]]] = None
    source_chunks: List[str] = field(default_factory=list)
    speaker_id: Optional[str] = None


@dataclass
class SemanticChunk:
    """语义切分后的 Chunk。"""

    chunk_id: str
    text: str
    sentences: List[SentenceSegment]
    punctuation_result: Optional[PunctuationResult]
    punctuation_decision: Optional[PunctuationDecision]
    pending_tail: str
    audio_range: Tuple[float, float]
    language: str
    source_chunks: List[str]
    speaker_id: Optional[str] = None  # V3.2.0+dev.20260201.04: 透传说话人标识
    word_timestamps: List[Dict[str, Any]] = field(default_factory=list)


class SemanticBuffer:
    """语义缓冲器：基于标点切分与边界待定区输出语义块。"""

    def __init__(
        self,
        max_pending_chars: int = 100,
        max_pending_duration: float = 3.0,
        force_flush_duration: float = 10.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._runtime_service = get_model_runtime_config_service()
        self._max_pending_chars = max(1, int(max_pending_chars))
        self._max_pending_duration = max(0.0, float(max_pending_duration))
        self._force_flush_duration = max(0.0, float(force_flush_duration))
        self._min_subtitle_chars_zh_ja = 4
        self._min_subtitle_words_en = 4
        self._min_subtitle_duration_sec = 0.8
        self._fast_delay_budget_sec = 2.0
        self._min_sentence_gap = 0.3
        self._hard_limit_duration = self._force_flush_duration
        self._force_split_on_sentence_end_punct = True
        self._pending_text = ""
        self._pending_audio_start = 0.0
        self._pending_audio_end = 0.0
        self._pending_language = "auto"
        self._pending_speaker_id: Optional[str] = None  # V3.2.0+dev.20260201.04: 记录说话人
        self._pending_source_chunks: List[str] = []
        self._pending_decision: Optional[PunctuationDecision] = None
        self._pending_punctuation_result: Optional[PunctuationResult] = None
        self._last_speaker_id: Optional[str] = None
        self._pending_words: List[Dict[str, Any]] = []

    def add(self, item: SemanticBufferInput) -> List[SemanticChunk]:
        """添加快流标点结果，返回可输出的语义块。"""
        if not item.text:
            return []
        self._refresh_runtime_config()
        outputs: List[SemanticChunk] = []

        if self._should_flush_for_speaker(item.speaker_id):
            outputs.extend(self.flush(reason="speaker_change"))

        text = item.text
        if item.punctuation_result and item.punctuation_result.text:
            text = item.punctuation_result.text

        combined_text = f"{self._pending_text}{text}"
        combined_start = self._pending_audio_start if self._pending_text else float(item.audio_range[0])
        combined_end = float(item.audio_range[1])
        combined_language = item.language or self._pending_language or "auto"
        combined_speaker_id = item.speaker_id or self._pending_speaker_id
        combined_sources = self._pending_source_chunks + (item.source_chunks or [item.chunk_id])
        selected_words = self._select_words(item.word_timestamps, item.raw_tokens, item.language)
        current_words = self._normalize_words(selected_words, item.audio_range)
        combined_words = self._pending_words + current_words

        decision = item.punctuation_decision or self._pending_decision
        punct_result = item.punctuation_result or self._pending_punctuation_result
        outputs.extend(
            self._split_and_update(
                combined_text=combined_text,
                audio_range=(combined_start, combined_end),
                language=combined_language,
                speaker_id=combined_speaker_id,
                source_chunks=combined_sources,
                decision=decision,
                punct_result=punct_result,
                pending_clean_len=self._clean_length(self._pending_text),
                combined_words=combined_words,
            )
        )
        self._last_speaker_id = item.speaker_id or self._last_speaker_id
        return outputs

    def flush(self, reason: str = "force") -> List[SemanticChunk]:
        """强制输出所有缓冲内容。"""
        self._refresh_runtime_config()
        if not self._pending_text:
            return []
        chunk = self._build_chunk(
            text=self._pending_text,
            audio_range=(self._pending_audio_start, self._pending_audio_end),
            language=self._pending_language,
            speaker_id=self._pending_speaker_id,
            source_chunks=self._pending_source_chunks,
            decision=self._pending_decision,
            split_end_indices=[len(self._pending_text) - 1],
            locked_end_indices=set(),
            pending_tail="",
            punctuation_result=self._pending_punctuation_result,
            word_timestamps=self._pending_words,
        )
        self._reset_pending()
        self._logger.debug("语义缓冲强制刷新: reason=%s", reason)
        return [chunk]

    def _refresh_runtime_config(self) -> None:
        """刷新运行参数（用于字幕切分与合并阈值）。"""
        try:
            runtime = self._runtime_service.get_effective_runtime_global()
        except Exception as exc:
            self._logger.debug("SemanticBuffer 运行参数读取失败（忽略）: %s", exc)
            return

        punct = runtime.get("effective", {}).get("punctuation", {})

        def _to_int(value: Any, fallback: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return fallback

        def _to_float(value: Any, fallback: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback

        self._min_subtitle_chars_zh_ja = max(1, _to_int(
            punct.get("min_subtitle_chars_zh_ja", self._min_subtitle_chars_zh_ja),
            self._min_subtitle_chars_zh_ja,
        ))
        self._min_subtitle_words_en = max(1, _to_int(
            punct.get("min_subtitle_words_en", self._min_subtitle_words_en),
            self._min_subtitle_words_en,
        ))
        self._min_subtitle_duration_sec = max(0.0, _to_float(
            punct.get("min_subtitle_duration_sec", self._min_subtitle_duration_sec),
            self._min_subtitle_duration_sec,
        ))
        self._fast_delay_budget_sec = max(0.0, _to_float(
            punct.get("fast_delay_budget_sec", self._fast_delay_budget_sec),
            self._fast_delay_budget_sec,
        ))
        force_split = punct.get("force_split_on_sentence_end_punct")
        if force_split is not None:
            self._force_split_on_sentence_end_punct = bool(force_split)
        self._min_sentence_gap = max(0.0, _to_float(
            punct.get("min_sentence_gap", self._min_sentence_gap),
            self._min_sentence_gap,
        ))
        hard_limit = _to_float(
            punct.get("hard_limit_duration", self._hard_limit_duration),
            self._hard_limit_duration,
        )
        self._hard_limit_duration = max(0.0, hard_limit)

        max_buffer = punct.get("max_buffer_duration")
        if max_buffer is not None:
            self._max_pending_duration = max(
                self._min_subtitle_duration_sec,
                _to_float(max_buffer, self._max_pending_duration),
            )
        self._force_flush_duration = max(
            self._force_flush_duration,
            self._hard_limit_duration,
            self._min_subtitle_duration_sec,
        )

    def _split_and_update(
        self,
        *,
        combined_text: str,
        audio_range: Tuple[float, float],
        language: str,
        speaker_id: Optional[str],
        source_chunks: List[str],
        decision: Optional[PunctuationDecision],
        punct_result: Optional[PunctuationResult],
        pending_clean_len: int,
        combined_words: List[Dict[str, Any]],
    ) -> List[SemanticChunk]:
        outputs: List[SemanticChunk] = []
        if not combined_text:
            return outputs

        split_clean_indices = self._build_split_indices(punct_result, pending_clean_len)
        if split_clean_indices:
            split_end_indices = self._map_clean_indices_to_text_end_indices(
                combined_text,
                split_clean_indices,
            )
            if split_end_indices:
                locked_end_indices = set(split_end_indices)
                split_end_indices = self._trim_trailing_short_boundaries(
                    text=combined_text,
                    end_indices=split_end_indices,
                    audio_range=audio_range,
                    word_timestamps=combined_words,
                    language=language,
                    locked_end_indices=locked_end_indices,
                )
            if split_end_indices:
                last_end = split_end_indices[-1]
                chunk_text = combined_text[: last_end + 1]
                chunk_range, pending_range = self._split_audio_range(audio_range, combined_text, last_end)
                split_time = chunk_range[1]
                chunk_words, pending_words = self._split_words_by_boundary(
                    text=combined_text,
                    words=combined_words,
                    split_text_index=last_end,
                    split_time=split_time,
                )
                chunk = self._build_chunk(
                    text=chunk_text,
                    audio_range=chunk_range,
                    language=language,
                    speaker_id=speaker_id,
                    source_chunks=source_chunks,
                    decision=decision,
                    split_end_indices=split_end_indices,
                    locked_end_indices=locked_end_indices if split_end_indices else set(),
                    pending_tail=combined_text[last_end + 1 :],
                    punctuation_result=self._clone_result(punct_result, chunk_text, split_end_indices, chunk_range),
                    word_timestamps=chunk_words,
                )
                outputs.append(chunk)
                self._update_pending(
                    text=combined_text[last_end + 1 :],
                    audio_range=pending_range,
                    language=language,
                    speaker_id=speaker_id,
                    source_chunks=source_chunks,
                    decision=decision,
                    punctuation_result=punct_result,
                    pending_words=pending_words,
                )
            else:
                self._update_pending(
                    text=combined_text,
                    audio_range=audio_range,
                    language=language,
                    speaker_id=speaker_id,
                    source_chunks=source_chunks,
                    decision=decision,
                    punctuation_result=punct_result,
                    pending_words=combined_words,
                )
        else:
            self._update_pending(
                text=combined_text,
                audio_range=audio_range,
                language=language,
                speaker_id=speaker_id,
                source_chunks=source_chunks,
                decision=decision,
                punctuation_result=punct_result,
                pending_words=combined_words,
            )

        outputs.extend(self._force_split_if_needed())
        return outputs

    def _build_split_indices(
        self,
        punct_result: Optional[PunctuationResult],
        pending_clean_len: int,
    ) -> List[int]:
        if not punct_result or not punct_result.split_points:
            return []
        indices: List[int] = []
        for point in punct_result.split_points:
            indices.append(int(point.char_index) + pending_clean_len)
        return sorted(set(idx for idx in indices if idx >= 0))

    def _map_clean_indices_to_text_end_indices(
        self,
        text: str,
        clean_indices: List[int],
    ) -> List[int]:
        clean_map = self._build_clean_index_map(text)
        if not clean_map:
            return []
        results: List[int] = []
        last_end = -1
        for clean_idx in clean_indices:
            if clean_idx < 0:
                continue
            if clean_idx >= len(clean_map):
                base_index = clean_map[-1]
            else:
                base_index = clean_map[clean_idx]
            end_index = base_index
            # V3.2.0+dev.20260201.01: 向右扩展标点符号时，跳过小数点（如 2.2）
            while end_index + 1 < len(text) and text[end_index + 1] in _PUNCTUATION_SET:
                # 检查下一个字符是否为小数点
                if _is_decimal_point(text, end_index + 1):
                    break
                end_index += 1
            if end_index <= last_end:
                continue
            results.append(end_index)
            last_end = end_index
        return results

    def _trim_trailing_short_boundaries(
        self,
        *,
        text: str,
        end_indices: List[int],
        audio_range: Tuple[float, float],
        word_timestamps: List[Dict[str, Any]],
        language: str,
        locked_end_indices: Optional[set[int]] = None,
    ) -> List[int]:
        if not end_indices:
            return []
        indices = list(end_indices)
        locked = locked_end_indices or set()
        if self._force_split_on_sentence_end_punct and indices and indices[-1] in locked:
            return indices
        if self._fast_delay_budget_sec <= 0.0:
            return indices

        clean_len = self._clean_length(text)
        clean_map = self._build_clean_index_map(text)
        clean_text = self._build_clean_text(text)
        clean_to_word = self._build_clean_to_word_map(clean_text, word_timestamps) if word_timestamps else []
        audio_start, audio_end = audio_range

        while indices:
            start_idx = indices[-2] + 1 if len(indices) > 1 else 0
            end_idx = indices[-1]
            segment_text = text[start_idx:end_idx + 1]
            if not segment_text.strip():
                indices.pop()
                continue

            seg_start, seg_end = self._estimate_sentence_time(
                text=text,
                clean_len=clean_len,
                clean_map=clean_map,
                clean_to_word=clean_to_word,
                word_timestamps=word_timestamps,
                audio_start=audio_start,
                audio_end=audio_end,
                start_text_index=start_idx,
                end_text_index=end_idx,
            )
            duration = max(seg_end - seg_start, 0.0)
            if not self._is_short_segment(segment_text, duration, language):
                break

            pending_duration = max(audio_end - seg_start, 0.0)
            if pending_duration > self._fast_delay_budget_sec:
                self._logger.debug(
                    "短句延迟预算超限: pending=%.2fs > budget=%.2fs，保持切分点",
                    pending_duration,
                    self._fast_delay_budget_sec,
                )
                break

            self._logger.debug(
                "短句延迟输出: duration=%.2fs, text='%s'",
                duration,
                segment_text.strip()[:30],
            )
            indices.pop()

        return indices

    def _merge_short_sentences(
        self,
        sentences: List[SentenceSegment],
        language: str,
        locked_end_flags: Optional[List[bool]] = None,
    ) -> List[SentenceSegment]:
        if len(sentences) <= 1:
            return sentences

        merged: List[SentenceSegment] = []
        merged_locked: List[bool] = []
        i = 0
        strategy = _get_language_strategy(language)
        locked_flags = locked_end_flags or []

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
                merged[-1] = self._merge_sentence_pair(prev, current, language)
                merged_locked[-1] = current_locked
                i += 1
                continue

            if (
                next_sent
                and not current_locked
                and self._can_merge_sentences(current, next_sent)
            ):
                merged.append(self._merge_sentence_pair(current, next_sent, language))
                merged_locked.append(next_locked)
                i += 2
                continue

            if prev and not prev_locked and self._can_merge_sentences(prev, current):
                merged[-1] = self._merge_sentence_pair(prev, current, language)
                merged_locked[-1] = current_locked
                i += 1
                continue

            merged.append(current)
            merged_locked.append(current_locked)
            i += 1

        return merged

    @staticmethod
    def _build_locked_flags(
        end_indices_used: List[int],
        locked_end_indices: Optional[set[int]],
    ) -> List[bool]:
        if not end_indices_used:
            return []
        locked = locked_end_indices or set()
        return [end_idx in locked for end_idx in end_indices_used]

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
            if gap < self._min_sentence_gap:
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
        if self._hard_limit_duration > 0.0 and merged_duration > self._hard_limit_duration:
            return False
        return True

    def _merge_sentence_pair(
        self,
        left: SentenceSegment,
        right: SentenceSegment,
        language: str,
    ) -> SentenceSegment:
        merged_text = f"{left.text}{right.text}"
        merged_clean = self._strip_trailing_punctuation(merged_text).strip()
        if _should_restore_contractions(merged_clean, language):
            merged_clean = _restore_english_contractions(merged_clean)

        merged_words = list(left.words) + list(right.words)
        strict_confidence, display_raw = self._compute_sentence_confidence(merged_words)
        merged_sentence = SentenceSegment(
            text=merged_text,
            text_clean=merged_clean,
            start=left.start,
            end=right.end,
            words=merged_words,
            confidence=strict_confidence if merged_words else left.confidence,
            confidence_display_raw=display_raw,
            confidence_source=left.confidence_source or right.confidence_source,
            source=left.source,
            is_draft=left.is_draft,
            is_finalized=left.is_finalized,
            warning_type=(
                left.warning_type
                if left.warning_type != WarningType.NONE
                else right.warning_type
            ),
        )
        return merged_sentence

    def _is_short_sentence(self, sentence: SentenceSegment, language: str) -> bool:
        duration = max(sentence.end - sentence.start, 0.0)
        return self._is_short_segment(sentence.text_clean or sentence.text, duration, language)

    def _is_short_segment(self, text: str, duration: float, language: str) -> bool:
        if duration < self._min_subtitle_duration_sec:
            return True
        unit_count = self._count_units(text, language)
        if self._resolve_language_group(text, language) == "en":
            return unit_count < self._min_subtitle_words_en
        return unit_count < self._min_subtitle_chars_zh_ja

    def _count_units(self, text: str, language: str) -> int:
        normalized = self._strip_trailing_punctuation(text).strip()
        if not normalized:
            return 0
        if self._resolve_language_group(normalized, language) == "en":
            return self._count_english_words(normalized)
        return self._count_cjk_chars(normalized)

    @staticmethod
    def _count_english_words(text: str) -> int:
        return len(_EN_WORD_PATTERN.findall(text))

    @staticmethod
    def _count_cjk_chars(text: str) -> int:
        return sum(1 for char in text if char not in _PUNCTUATION_SET and not char.isspace())

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

    def _force_split_if_needed(self) -> List[SemanticChunk]:
        outputs: List[SemanticChunk] = []
        if not self._pending_text:
            return outputs
        pending_clean_len = self._clean_length(self._pending_text)
        pending_duration = max(self._pending_audio_end - self._pending_audio_start, 0.0)

        if self._force_flush_duration > 0.0 and pending_duration >= self._force_flush_duration:
            outputs.extend(self.flush(reason="force_duration"))
            return outputs

        if pending_clean_len < self._max_pending_chars and pending_duration < self._max_pending_duration:
            return outputs

        weak_index = self._find_last_weak_punctuation(self._pending_text)
        if weak_index is None:
            outputs.extend(self.flush(reason="max_pending"))
            return outputs

        chunk_text = self._pending_text[: weak_index + 1]
        chunk_range, pending_range = self._split_audio_range(
            (self._pending_audio_start, self._pending_audio_end),
            self._pending_text,
            weak_index,
        )
        chunk_words, pending_words = self._split_words_by_time(self._pending_words, chunk_range[1])
        chunk = self._build_chunk(
            text=chunk_text,
            audio_range=chunk_range,
            language=self._pending_language,
            speaker_id=self._pending_speaker_id,
            source_chunks=self._pending_source_chunks,
            decision=self._pending_decision,
            split_end_indices=[weak_index],
            locked_end_indices=set(),
            pending_tail=self._pending_text[weak_index + 1 :],
            punctuation_result=self._clone_result(
                self._pending_punctuation_result,
                chunk_text,
                [weak_index],
                chunk_range,
            ),
            word_timestamps=chunk_words,
        )
        outputs.append(chunk)
        self._update_pending(
            text=self._pending_text[weak_index + 1 :],
            audio_range=pending_range,
            language=self._pending_language,
            speaker_id=self._pending_speaker_id,
            source_chunks=self._pending_source_chunks,
            decision=self._pending_decision,
            punctuation_result=self._pending_punctuation_result,
            pending_words=pending_words,
        )
        return outputs

    def _build_chunk(
        self,
        *,
        text: str,
        audio_range: Tuple[float, float],
        language: str,
        speaker_id: Optional[str],
        source_chunks: List[str],
        decision: Optional[PunctuationDecision],
        split_end_indices: List[int],
        locked_end_indices: Optional[set[int]] = None,
        pending_tail: str,
        punctuation_result: Optional[PunctuationResult],
        word_timestamps: Optional[List[Dict[str, Any]]] = None,
    ) -> SemanticChunk:
        confidence = punctuation_result.confidence if punctuation_result else 1.0
        sentences, end_indices_used = self._build_sentences(
            text,
            audio_range,
            split_end_indices,
            confidence,
            word_timestamps or [],
            language,
        )
        locked_flags = self._build_locked_flags(
            end_indices_used,
            locked_end_indices if self._force_split_on_sentence_end_punct else None,
        )
        sentences = self._merge_short_sentences(sentences, language, locked_flags)
        return SemanticChunk(
            chunk_id=self._build_chunk_id(source_chunks),
            text=text,
            sentences=sentences,
            punctuation_result=punctuation_result,
            punctuation_decision=decision,
            pending_tail=pending_tail,
            audio_range=audio_range,
            language=language,
            speaker_id=speaker_id,
            source_chunks=source_chunks,
            word_timestamps=word_timestamps or [],
        )

    def _build_sentences(
        self,
        text: str,
        audio_range: Tuple[float, float],
        split_end_indices: List[int],
        confidence: float,
        word_timestamps: List[Dict[str, Any]],
        language: str,
    ) -> Tuple[List[SentenceSegment], List[int]]:
        if not text:
            return [], []
        if not split_end_indices:
            split_end_indices = [len(text) - 1]
        sentences: List[SentenceSegment] = []
        end_indices_used: List[int] = []
        start_idx = 0
        clean_len = self._clean_length(text)
        clean_map = self._build_clean_index_map(text)
        clean_text = self._build_clean_text(text)
        clean_to_word = self._build_clean_to_word_map(clean_text, word_timestamps) if word_timestamps else []
        audio_start, audio_end = audio_range
        last_end_time = audio_start
        for end_idx in split_end_indices:
            end_idx = min(max(end_idx, start_idx), len(text) - 1)
            sentence_text = text[start_idx : end_idx + 1]
            if not sentence_text.strip():
                start_idx = end_idx + 1
                continue
            seg_start, seg_end = self._estimate_sentence_time(
                text=text,
                clean_len=clean_len,
                clean_map=clean_map,
                clean_to_word=clean_to_word,
                word_timestamps=word_timestamps,
                audio_start=audio_start,
                audio_end=audio_end,
                start_text_index=start_idx,
                end_text_index=end_idx,
            )
            if seg_start < last_end_time:
                seg_start = last_end_time
            if seg_end < seg_start:
                seg_end = seg_start
            if seg_start > audio_end:
                seg_start = audio_end
            if seg_end > audio_end:
                seg_end = audio_end
            sentence_words = self._select_words_by_time(word_timestamps, seg_start, seg_end)
            strict_confidence, display_raw = self._compute_sentence_confidence(sentence_words)
            sentence_confidence = strict_confidence if sentence_words else confidence
            sentence_word_models = self._build_word_models(sentence_words)
            display_text = self._strip_trailing_punctuation(sentence_text).strip()
            # V3.2.0+dev.20260131.04: 仅修复英文展示文本的缩写撇号，避免影响时间戳与切分。
            if _should_restore_contractions(display_text, language):
                display_text = _restore_english_contractions(display_text)
            sentences.append(
                SentenceSegment(
                    text=sentence_text,
                    text_clean=display_text,
                    start=seg_start,
                    end=seg_end,
                    words=sentence_word_models,
                    confidence=sentence_confidence,
                    confidence_display_raw=display_raw,
                    source=TextSource.SENSEVOICE,
                    is_draft=True,
                    is_finalized=False,
                )
            )
            end_indices_used.append(end_idx)
            last_end_time = seg_end
            start_idx = end_idx + 1
        return sentences, end_indices_used

    @staticmethod
    def _select_words_by_time(
        words: Sequence[Dict[str, Any]],
        seg_start: float,
        seg_end: float,
    ) -> List[Dict[str, Any]]:
        """根据时间范围选取词级时间戳，避免整段无高亮。"""
        if not words:
            return []
        selected: List[Dict[str, Any]] = []
        for word in words:
            start = float(word.get("start", 0.0) or 0.0)
            end = float(word.get("end", start) or start)
            mid = (start + end) / 2.0
            if seg_start <= mid <= seg_end:
                selected.append(word)
        return selected

    @staticmethod
    def _build_word_models(words: Sequence[Dict[str, Any]]) -> List[WordTimestamp]:
        """将词级字典转换为 WordTimestamp，确保 SentenceSegment.to_dict 可用。"""
        models: List[WordTimestamp] = []
        for word in words:
            if isinstance(word, WordTimestamp):
                models.append(word)
                continue
            if not isinstance(word, dict):
                continue
            models.append(
                WordTimestamp(
                    word=str(word.get("word", "")),
                    start=float(word.get("start", 0.0) or 0.0),
                    end=float(word.get("end", 0.0) or 0.0),
                    confidence=word.get("confidence"),
                    confidence_raw=word.get("confidence_raw"),
                    confidence_display_raw=word.get("confidence_display_raw"),
                    token_type=word.get("token_type"),
                    is_pseudo=bool(word.get("is_pseudo", False)),
                )
            )
        return models

    @staticmethod
    def _compute_sentence_confidence(
        words: Sequence[Dict[str, Any]],
    ) -> Tuple[float, Optional[float]]:
        """计算句级置信度（严格口径 + 显示口径）。"""
        if not words:
            return 0.0, None

        raw_values: List[float] = []
        display_values: List[float] = []
        display_weights: List[float] = []

        for word in words:
            conf = word.get("confidence") if isinstance(word, dict) else getattr(word, "confidence", None)
            if conf is not None:
                raw_values.append(float(conf))

            display_conf = (
                word.get("confidence_display_raw")
                if isinstance(word, dict)
                else getattr(word, "confidence_display_raw", None)
            )
            if display_conf is None:
                display_conf = conf
            if display_conf is None:
                continue

            if isinstance(word, dict):
                start = float(word.get("start", 0.0) or 0.0)
                end = float(word.get("end", start) or start)
            else:
                start = float(getattr(word, "start", 0.0) or 0.0)
                end = float(getattr(word, "end", start) or start)
            duration = max(end - start, 0.0)
            weight = duration if duration > 0.0 else 1.0
            display_values.append(float(display_conf))
            display_weights.append(weight)

        strict_confidence = min(raw_values) if raw_values else 0.0

        if not display_values:
            return strict_confidence, None

        weighted_sum = sum(val * w for val, w in zip(display_values, display_weights))
        total_weight = sum(display_weights)
        display_raw = weighted_sum / total_weight if total_weight > 0.0 else None
        return strict_confidence, display_raw

    def _split_audio_range(
        self,
        audio_range: Tuple[float, float],
        text: str,
        end_index: int,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        audio_start, audio_end = audio_range
        clean_len = self._clean_length(text)
        split_time = self._estimate_time_by_clean_index(
            audio_start=audio_start,
            audio_end=audio_end,
            text=text,
            clean_len=clean_len,
            text_index=end_index,
            include_current=True,
        )
        return (audio_start, split_time), (split_time, audio_end)

    def _estimate_time_by_clean_index(
        self,
        *,
        audio_start: float,
        audio_end: float,
        text: str,
        clean_len: int,
        text_index: int,
        include_current: bool,
    ) -> float:
        if clean_len <= 0:
            return audio_end if include_current else audio_start
        clean_index = self._text_index_to_clean_index(text, text_index)
        position = clean_index + (1 if include_current else 0)
        position = min(max(position, 0), clean_len)
        duration = max(audio_end - audio_start, 0.0)
        ratio = min(max(position / clean_len, 0.0), 1.0)
        return audio_start + duration * ratio

    def _update_pending(
        self,
        *,
        text: str,
        audio_range: Tuple[float, float],
        language: str,
        speaker_id: Optional[str],
        source_chunks: List[str],
        decision: Optional[PunctuationDecision],
        punctuation_result: Optional[PunctuationResult],
        pending_words: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._pending_text = text
        self._pending_audio_start = float(audio_range[0])
        self._pending_audio_end = float(audio_range[1])
        self._pending_language = language or self._pending_language
        self._pending_speaker_id = speaker_id or self._pending_speaker_id
        self._pending_source_chunks = list(source_chunks)
        self._pending_decision = decision
        self._pending_punctuation_result = punctuation_result
        self._pending_words = pending_words or []

    def _reset_pending(self) -> None:
        self._pending_text = ""
        self._pending_audio_start = 0.0
        self._pending_audio_end = 0.0
        self._pending_language = "auto"
        self._pending_speaker_id = None
        self._pending_source_chunks = []
        self._pending_decision = None
        self._pending_punctuation_result = None
        self._pending_words = []

    def _should_flush_for_speaker(self, speaker_id: Optional[str]) -> bool:
        if not self._pending_text:
            return False
        if speaker_id is None or self._last_speaker_id is None:
            return False
        return speaker_id != self._last_speaker_id

    @staticmethod
    def _build_chunk_id(source_chunks: List[str]) -> str:
        if not source_chunks:
            return "semantic-unknown"
        if len(source_chunks) == 1:
            return source_chunks[0]
        return f"{source_chunks[0]}+{source_chunks[-1]}"

    @staticmethod
    def _clean_length(text: str) -> int:
        """
        计算去除标点后的文本长度，但保留小数点（如 2.2）。
        V3.2.0+dev.20260201.14
        """
        return sum(1 for idx, _ in enumerate(text) if not _is_removable_punctuation(text, idx))

    @staticmethod
    def _build_clean_index_map(text: str) -> List[int]:
        """
        构建 clean_text 索引映射，保留小数点（如 2.2）。
        V3.2.0+dev.20260201.14
        """
        return [idx for idx, _ in enumerate(text) if not _is_removable_punctuation(text, idx)]

    @staticmethod
    def _build_clean_text(text: str) -> str:
        """
        构建去除标点的文本，但保留小数点（如 2.2）。
        V3.2.0+dev.20260201.14
        """
        return "".join(char for idx, char in enumerate(text) if not _is_removable_punctuation(text, idx))

    @staticmethod
    def _find_last_weak_punctuation(text: str) -> Optional[int]:
        for idx in range(len(text) - 1, -1, -1):
            if text[idx] in _WEAK_PUNCTUATION:
                return idx
        return None

    @staticmethod
    def _strip_trailing_punctuation(text: str) -> str:
        """
        移除尾部标点符号，但跳过小数点（如 2.2、3.14）。
        V3.2.0+dev.20260201.01
        """
        if not text:
            return text
        end = len(text)
        while end > 0 and text[end - 1] in _TRAILING_PUNCTUATION:
            # V3.2.0+dev.20260201.01: 检查是否为小数点，如果是则停止移除
            if _is_decimal_point(text, end - 1):
                break
            end -= 1
        return text[:end]

    def _select_words(
        self,
        words: Optional[Sequence[Dict[str, Any]]],
        raw_tokens: Optional[Sequence[Dict[str, Any]]],
        language: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        选择用于时间戳估算的词列表。

        当 word_timestamps 粒度过粗时，回退使用 raw_tokens 重新合并。
        """
        if words and (len(words) > 1 or not raw_tokens):
            return list(words)
        if raw_tokens:
            try:
                from app.services.token_merge_service import merge_tokens

                merge_result = merge_tokens(list(raw_tokens), language=language)
                if merge_result.words:
                    return merge_result.words
            except Exception as exc:
                self._logger.debug("SemanticBuffer 回退合并失败: %s", exc)
        return list(words) if words else []

    @staticmethod
    def _normalize_words(
        words: Optional[Sequence[Dict[str, Any]]],
        audio_range: Tuple[float, float],
    ) -> List[Dict[str, Any]]:
        if not words:
            return []
        audio_start = float(audio_range[0])
        normalized: List[Dict[str, Any]] = []
        for word in words:
            token = str(word.get("word", "") or "")
            token = token.lstrip(" ▁")
            if not token or token in _PUNCTUATION_SET:
                continue
            start = word.get("start", 0.0)
            end = word.get("end", 0.0)
            try:
                start_val = float(start)
                end_val = float(end)
            except (TypeError, ValueError):
                start_val = 0.0
                end_val = 0.0
            normalized.append(
                {
                    "word": token,
                    "start": audio_start + start_val,
                    "end": audio_start + end_val,
                    "confidence": word.get("confidence", 1.0),
                    "confidence_display_raw": word.get("confidence_display_raw"),
                    "token_type": word.get("token_type"),
                }
            )
        return normalized

    @staticmethod
    def _build_clean_to_word_map(
        clean_text: str,
        words: Sequence[Dict[str, Any]],
    ) -> List[Optional[int]]:
        mapping: List[Optional[int]] = [None] * len(clean_text)
        if not clean_text or not words:
            return mapping
        cursor = 0
        text_len = len(clean_text)
        for idx, word in enumerate(words):
            token = str(word.get("word", "") or "")
            token = token.strip()
            if not token:
                continue
            match_idx = clean_text.find(token, cursor)
            if match_idx == -1:
                match_idx = SemanticBuffer._fallback_match(clean_text, token, cursor)
            if match_idx == -1:
                match_idx = min(cursor, max(text_len - 1, 0))
            start = max(match_idx, 0)
            end = min(match_idx + len(token) - 1, max(text_len - 1, 0))
            for pos in range(start, end + 1):
                if 0 <= pos < len(mapping):
                    mapping[pos] = idx
            cursor = min(end + 1, text_len)
        return mapping

    @staticmethod
    def _fallback_match(text: str, token: str, cursor: int) -> int:
        if not token:
            return -1
        if token.isspace():
            for idx in range(cursor, len(text)):
                if text[idx].isspace():
                    return idx
            return -1
        if cursor < len(text) and text[cursor: cursor + len(token)] == token:
            return cursor
        for idx in range(cursor, len(text)):
            if text[idx: idx + len(token)] == token:
                return idx
        return -1

    @staticmethod
    def _find_clean_index_right(clean_map: List[int], text_index: int) -> Optional[int]:
        for clean_idx, raw_idx in enumerate(clean_map):
            if raw_idx >= text_index:
                return clean_idx
        return None

    @staticmethod
    def _find_clean_index_left(clean_map: List[int], text_index: int) -> Optional[int]:
        for clean_idx in range(len(clean_map) - 1, -1, -1):
            if clean_map[clean_idx] <= text_index:
                return clean_idx
        return None

    @staticmethod
    def _find_nearest_word_index(
        clean_to_word: Sequence[Optional[int]],
        clean_index: int,
        direction: int,
    ) -> Optional[int]:
        if not clean_to_word:
            return None
        if 0 <= clean_index < len(clean_to_word):
            current = clean_to_word[clean_index]
            if current is not None:
                return current
        if direction >= 0:
            for idx in range(clean_index + 1, len(clean_to_word)):
                current = clean_to_word[idx]
                if current is not None:
                    return current
        else:
            for idx in range(clean_index - 1, -1, -1):
                current = clean_to_word[idx]
                if current is not None:
                    return current
        return None

    def _estimate_sentence_time(
        self,
        *,
        text: str,
        clean_len: int,
        clean_map: List[int],
        clean_to_word: Sequence[Optional[int]],
        word_timestamps: Sequence[Dict[str, Any]],
        audio_start: float,
        audio_end: float,
        start_text_index: int,
        end_text_index: int,
    ) -> Tuple[float, float]:
        if word_timestamps and clean_map and clean_to_word:
            clean_start = self._find_clean_index_right(clean_map, start_text_index)
            clean_end = self._find_clean_index_left(clean_map, end_text_index)
            if clean_start is not None and clean_end is not None:
                start_word_idx = self._find_nearest_word_index(clean_to_word, clean_start, direction=1)
                end_word_idx = self._find_nearest_word_index(clean_to_word, clean_end, direction=-1)
                if start_word_idx is not None and end_word_idx is not None:
                    start_word = word_timestamps[start_word_idx]
                    end_word = word_timestamps[end_word_idx]
                    seg_start = float(start_word.get("start", audio_start) or audio_start)
                    seg_end = float(end_word.get("end", audio_end) or audio_end)
                    return seg_start, seg_end
        seg_start = self._estimate_time_by_clean_index(
            audio_start=audio_start,
            audio_end=audio_end,
            text=text,
            clean_len=clean_len,
            text_index=start_text_index,
            include_current=False,
        )
        seg_end = self._estimate_time_by_clean_index(
            audio_start=audio_start,
            audio_end=audio_end,
            text=text,
            clean_len=clean_len,
            text_index=end_text_index,
            include_current=True,
        )
        return seg_start, seg_end

    def _split_words_by_boundary(
        self,
        *,
        text: str,
        words: List[Dict[str, Any]],
        split_text_index: int,
        split_time: float,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not words:
            return [], []
        clean_text = self._build_clean_text(text)
        clean_to_word = self._build_clean_to_word_map(clean_text, words)
        clean_map = self._build_clean_index_map(text)
        clean_end = self._find_clean_index_left(clean_map, split_text_index)
        if clean_end is None or not clean_to_word:
            return self._split_words_by_time(words, split_time)
        end_word_idx = self._find_nearest_word_index(clean_to_word, clean_end, direction=-1)
        if end_word_idx is None:
            return self._split_words_by_time(words, split_time)
        return words[: end_word_idx + 1], words[end_word_idx + 1 :]

    @staticmethod
    def _split_words_by_time(
        words: List[Dict[str, Any]],
        split_time: float,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not words:
            return [], []
        head: List[Dict[str, Any]] = []
        tail: List[Dict[str, Any]] = []
        for word in words:
            start = float(word.get("start", 0.0) or 0.0)
            end = float(word.get("end", 0.0) or 0.0)
            mid = (start + end) / 2.0
            if end <= split_time or mid <= split_time:
                head.append(word)
            else:
                tail.append(word)
        return head, tail

    @staticmethod
    def _clone_result(
        result: Optional[PunctuationResult],
        text: str,
        split_end_indices: List[int],
        audio_range: Tuple[float, float],
    ) -> Optional[PunctuationResult]:
        if result is None:
            return None
        split_points: List[SplitPoint] = []
        for end_idx in split_end_indices:
            clean_idx = SemanticBuffer._text_index_to_clean_index(text, end_idx)
            if clean_idx < 0:
                continue
            if end_idx < len(text) and text[end_idx] in _PUNCTUATION_SET:
                punct = text[end_idx]
            else:
                punct = ""
            relative_time = max(
                SemanticBuffer._estimate_time_static(audio_range, text, end_idx) - audio_range[0],
                0.0,
            )
            split_points.append(
                SplitPoint(
                    char_index=clean_idx,
                    relative_time=relative_time,
                    punctuation=punct,
                    confidence=1.0,
                )
            )
        return PunctuationResult(
            text=text,
            model_id=result.model_id,
            split_points=split_points,
            punctuation_positions=[],
            confidence=result.confidence,
            processing_time_ms=result.processing_time_ms,
        )

    @staticmethod
    def _estimate_time_static(
        audio_range: Tuple[float, float],
        text: str,
        index: int,
    ) -> float:
        audio_start, audio_end = audio_range
        clean_len = SemanticBuffer._clean_length(text)
        if clean_len <= 0:
            return audio_end
        clean_index = SemanticBuffer._text_index_to_clean_index(text, index)
        position = min(max(clean_index + 1, 0), clean_len)
        duration = max(audio_end - audio_start, 0.0)
        ratio = min(max(position / clean_len, 0.0), 1.0)
        return audio_start + duration * ratio

    @staticmethod
    def _text_index_to_clean_index(text: str, text_index: int) -> int:
        clean_idx = -1
        for idx, char in enumerate(text):
            if _is_removable_punctuation(text, idx):
                continue
            clean_idx += 1
            if idx >= text_index:
                break
        return max(clean_idx, 0)
