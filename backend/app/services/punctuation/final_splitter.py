"""
最终切分器（Phase G-4）。
V3.2.0+dev.20260219.13
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from app.models.sensevoice_models import SentenceSegment, WordTimestamp
from app.services.punctuation.base import PuncPosition
from app.services.sentence_splitter import SentenceSplitter, SplitConfig
from app.services.text_protection import (
    is_sentence_end_punct,
    merge_protected_word_tokens,
)


_STRONG_PUNCT = set("。？！.!?")
_WEAK_PUNCT = set("，、,;；:：")


@dataclass
class FinalSplitConfig:
    """最终切分配置（多因子联合切分）。"""

    min_tokens: int = 5
    max_tokens: int = 50
    min_duration: float = 0.5
    max_duration: float = 10.0
    soft_pause: float = 0.35
    long_pause: float = 0.8
    is_force_split_on_sentence_end_punct: bool = True
    language: str = "auto"
    enable_balance: bool = True
    min_mapping_coverage: float = 0.6
    is_enable_cjk_weak_punct_split: bool = True
    is_enable_cjk_semantic_split: bool = True
    cjk_weak_punct_min_tokens: int = 8
    cjk_weak_punct_fallback_min_tokens: int = 16
    cjk_weak_punct_fallback_min_duration: float = 2.4
    cjk_semantic_split_min_tokens: int = 10
    cjk_semantic_split_min_duration: float = 1.2


class FinalSplitter:
    """最终切分器：基于标点 + 停顿 + 长度的联合切分。"""
    _CONTINUATION_MERGE_MAX_GAP_SEC = 0.55
    _CJK_CONTINUATION_MERGE_MAX_WORDS = 4
    _CJK_CONTINUATION_MERGE_MAX_DURATION_SEC = 2.20
    _CJK_DISCOURSE_BREAK_MARKERS = (
        "然而",
        "但是",
        "不过",
        "可是",
        "与此同时",
        "另外",
        "此外",
    )

    def __init__(
        self,
        config: Optional[FinalSplitConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or FinalSplitConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.last_split_stats: Optional[Dict[str, float]] = None
        self._semantic_anchor_words: Set[str] = set()
        self._sentence_builder = SentenceSplitter(
            SplitConfig(
                language=self.config.language,
                min_chars=0,
                enable_hard_limit=False,
                merge_short_sentences=False,
                trim_leading_silence=True,
                trim_trailing_silence=True,
                max_boundary_gap=0.3,
            )
        )

    def split(
        self,
        words: Sequence[WordTimestamp],
        *,
        clean_text: Optional[str] = None,
        punctuation_positions: Optional[Sequence[PuncPosition]] = None,
    ) -> List[SentenceSegment]:
        """执行最终切分并返回 SentenceSegment 列表。"""
        if not words:
            return []
        protected_words = merge_protected_word_tokens(list(words))
        if not protected_words:
            return []

        external_strength, stats = self._build_external_strength(
            clean_text=clean_text,
            words=list(protected_words),
            positions=punctuation_positions or [],
        )
        segments, split_stats = self._split_words(
            list(protected_words),
            min_tokens=self.config.min_tokens,
            external_strength=external_strength,
        )
        stats.update(split_stats)
        if not self.config.enable_balance:
            self.last_split_stats = stats
            return segments

        segments = self._merge_short_segments(segments)
        segments = self._split_long_segments(segments)
        segments = self._merge_short_segments(segments)
        segments, continuation_merge_count = self._merge_continuation_segments(segments)
        stats["continuation_merge_count"] = float(continuation_merge_count)
        self.last_split_stats = stats
        return segments

    def set_language(self, language: str) -> None:
        """同步切分语言设置到内部构建器。"""
        self.config.language = language
        self._sentence_builder.config.language = language

    def set_semantic_anchor_words(self, words: Optional[Sequence[str]]) -> None:
        """同步语言适配层语义锚词，用于无标点语义断点切分。"""
        normalized: Set[str] = set()
        for item in words or []:
            token = self._normalize_boundary_token(str(item or "")).lower()
            if token:
                normalized.add(token)
        self._semantic_anchor_words = normalized

    def set_cjk_split_mode(
        self,
        *,
        is_enable_weak_punct: bool,
        is_enable_semantic: bool,
    ) -> None:
        """设置 CJK 扩展切分开关。"""
        self.config.is_enable_cjk_weak_punct_split = bool(is_enable_weak_punct)
        self.config.is_enable_cjk_semantic_split = bool(is_enable_semantic)

    def _split_words(
        self,
        words: List[WordTimestamp],
        min_tokens: int,
        external_strength: Optional[Dict[int, int]] = None,
    ) -> Tuple[List[SentenceSegment], Dict[str, float]]:
        segments: List[SentenceSegment] = []
        start_idx = 0
        last_candidate_idx: Optional[int] = None
        last_candidate_strength = 0
        pause_split_blocked_count = 0
        cjk_weak_punct_split_count = 0
        semantic_anchor_split_count = 0
        external_strength = external_strength or {}

        for idx, word in enumerate(words):
            tokens = idx - start_idx + 1
            duration = self._segment_duration(words, start_idx, idx)
            punct_strength = self._get_punct_strength(words, idx, external_strength)
            pause_strength = self._pause_strength(words, idx)
            boundary_strength = max(punct_strength, pause_strength)
            is_semantic_boundary = self._is_semantic_anchor_boundary(
                words=words,
                boundary_idx=idx,
            )
            strategy = self._sentence_builder.config.get_strategy()

            if boundary_strength > 0:
                if boundary_strength > last_candidate_strength:
                    last_candidate_idx = idx
                    last_candidate_strength = boundary_strength
                else:
                    last_candidate_idx = idx

            is_force = tokens >= self.config.max_tokens or duration >= self.config.max_duration
            if is_force:
                split_idx = self._select_force_split(
                    start_idx,
                    idx,
                    last_candidate_idx,
                    min_tokens,
                )
                segments.append(self._build_sentence(words, start_idx, split_idx))
                start_idx = split_idx + 1
                last_candidate_idx = None
                last_candidate_strength = 0
                continue

            if punct_strength >= 3 and self.config.is_force_split_on_sentence_end_punct:
                segments.append(self._build_sentence(words, start_idx, idx))
                start_idx = idx + 1
                last_candidate_idx = None
                last_candidate_strength = 0
                continue

            if punct_strength >= 3:
                if tokens >= min_tokens or duration >= self.config.min_duration or idx == len(words) - 1:
                    segments.append(self._build_sentence(words, start_idx, idx))
                    start_idx = idx + 1
                    last_candidate_idx = None
                    last_candidate_strength = 0
                continue

            if (
                punct_strength >= 1
                and self._is_enable_cjk_weak_punct_split()
                and tokens >= max(min_tokens, self.config.cjk_weak_punct_min_tokens)
                and duration >= self.config.min_duration
                and (
                    pause_strength >= 1
                    or is_semantic_boundary
                    or tokens >= self.config.cjk_weak_punct_fallback_min_tokens
                    or duration >= self.config.cjk_weak_punct_fallback_min_duration
                )
            ):
                if not is_semantic_boundary and self._should_block_pause_split(words, start_idx, idx):
                    pause_split_blocked_count += 1
                    continue
                segments.append(self._build_sentence(words, start_idx, idx))
                cjk_weak_punct_split_count += 1
                start_idx = idx + 1
                last_candidate_idx = None
                last_candidate_strength = 0
                continue

            if (
                self._is_enable_cjk_semantic_split()
                and is_semantic_boundary
                and tokens >= max(min_tokens, self.config.cjk_semantic_split_min_tokens)
                and duration >= max(self.config.min_duration, self.config.cjk_semantic_split_min_duration)
            ):
                current_token = self._normalize_boundary_token(words[idx].word or "")
                if current_token and strategy.is_incomplete_ending(current_token):
                    continue
                segments.append(self._build_sentence(words, start_idx, idx))
                semantic_anchor_split_count += 1
                start_idx = idx + 1
                last_candidate_idx = None
                last_candidate_strength = 0
                continue

            if pause_strength >= 2 and tokens >= min_tokens and duration >= self.config.min_duration:
                if not is_semantic_boundary and self._should_block_pause_split(words, start_idx, idx):
                    pause_split_blocked_count += 1
                    continue
                segments.append(self._build_sentence(words, start_idx, idx))
                start_idx = idx + 1
                last_candidate_idx = None
                last_candidate_strength = 0
                continue

            if (
                pause_strength >= 1
                and tokens >= min_tokens
                and duration >= self.config.min_duration
                and (punct_strength >= 1 or tokens >= self.config.max_tokens)
            ):
                if not is_semantic_boundary and self._should_block_pause_split(words, start_idx, idx):
                    pause_split_blocked_count += 1
                    continue
                segments.append(self._build_sentence(words, start_idx, idx))
                start_idx = idx + 1
                last_candidate_idx = None
                last_candidate_strength = 0

        if start_idx < len(words):
            segments.append(self._build_sentence(words, start_idx, len(words) - 1))

        return [seg for seg in segments if seg and seg.words], {
            "pause_split_blocked_count": float(pause_split_blocked_count),
            "cjk_weak_punct_split_count": float(cjk_weak_punct_split_count),
            "semantic_anchor_split_count": float(semantic_anchor_split_count),
        }

    def _is_enable_cjk_weak_punct_split(self) -> bool:
        """CJK 语言允许弱标点作为切分触发，缓解“无停顿但长句连写”问题。"""
        if not bool(self.config.is_enable_cjk_weak_punct_split):
            return False
        language = str(self.config.language or "").strip().lower()
        return language.startswith(("zh", "yue", "ja", "jp", "ko"))

    def _is_enable_cjk_semantic_split(self) -> bool:
        """仅在 CJK 且存在语义锚词时启用无标点语义断点切分。"""
        language = str(self.config.language or "").strip().lower()
        return (
            bool(self.config.is_enable_cjk_semantic_split)
            and language.startswith(("zh", "yue", "ja", "jp", "ko"))
            and bool(self._semantic_anchor_words)
        )

    def _is_semantic_anchor_boundary(
        self,
        *,
        words: Sequence[WordTimestamp],
        boundary_idx: int,
    ) -> bool:
        if not self._semantic_anchor_words:
            return False
        next_idx = self._next_real_word_index(list(words), boundary_idx)
        if next_idx is None:
            return False
        right_tokens: List[str] = []
        idx = next_idx
        while idx < len(words) and len(right_tokens) < 8:
            if getattr(words[idx], "is_pseudo", False):
                idx += 1
                continue
            token = self._normalize_boundary_token(words[idx].word or "").lower()
            if token:
                right_tokens.append(token)
            idx += 1
        if not right_tokens:
            return False
        right_compact = "".join(right_tokens)
        right_spaced = " ".join(right_tokens)
        first_token = right_tokens[0]
        for anchor in self._semantic_anchor_words:
            normalized = self._normalize_boundary_token(anchor).lower()
            if not normalized:
                continue
            if " " in normalized:
                if right_spaced.startswith(normalized):
                    return True
                continue
            if self._is_cjk_text(normalized):
                if right_compact.startswith(normalized):
                    return True
                continue
            if first_token == normalized:
                return True
        return False

    def _should_block_pause_split(
        self,
        words: List[WordTimestamp],
        start_idx: int,
        boundary_idx: int,
    ) -> bool:
        """停顿切分保护：续接词/不完整结尾时避免句中误切。"""
        next_idx = self._next_real_word_index(words, boundary_idx)
        if next_idx is None:
            return False

        strategy = self._sentence_builder.config.get_strategy()
        next_probe = self._build_probe_text(words, next_idx)
        if next_probe and strategy.is_continuation(next_probe):
            return True

        current_token = self._normalize_boundary_token(words[boundary_idx].word or "")
        if current_token and strategy.is_incomplete_ending(current_token):
            return True

        # V3.2.0+dev.20260210.01: 英文兜底规则。
        # 对无强句末标点且后句小写开头的边界，判定为更可能是句内续写，优先阻断停顿切分。
        prev_raw = str(words[boundary_idx].word or "").strip()
        next_first = self._normalize_boundary_token(words[next_idx].word or "")
        if (
            next_first
            and self._is_ascii_word(next_first)
            and next_first[:1].islower()
            and not is_sentence_end_punct(prev_raw, words[next_idx].word or "")
            and boundary_idx >= start_idx
        ):
            return True

        return False

    def _select_force_split(
        self,
        start_idx: int,
        current_idx: int,
        candidate_idx: Optional[int],
        min_tokens: int,
    ) -> int:
        if candidate_idx is None:
            return current_idx
        if candidate_idx < start_idx:
            return current_idx
        if candidate_idx - start_idx + 1 >= min_tokens:
            return candidate_idx
        return current_idx

    def _split_long_segments(self, segments: List[SentenceSegment]) -> List[SentenceSegment]:
        result: List[SentenceSegment] = []
        for segment in segments:
            if not segment.words:
                continue
            if len(segment.words) <= self.config.max_tokens and segment.end - segment.start <= self.config.max_duration:
                result.append(segment)
                continue
            words = segment.words
            sub_segments = self._split_long_words(words)
            result.extend(sub_segments)
        return result

    def _split_long_words(self, words: List[WordTimestamp]) -> List[SentenceSegment]:
        segments: List[SentenceSegment] = []
        start_idx = 0
        while start_idx < len(words):
            limit_idx = self._find_limit_index(words, start_idx)
            if limit_idx >= len(words) - 1:
                segments.append(self._build_sentence(words, start_idx, len(words) - 1))
                break
            boundary_idx = self._find_best_boundary(words, start_idx, limit_idx)
            min_tokens = max(1, self.config.min_tokens // 2)
            if boundary_idx is None or boundary_idx - start_idx + 1 < min_tokens:
                boundary_idx = limit_idx
            segments.append(self._build_sentence(words, start_idx, boundary_idx))
            start_idx = boundary_idx + 1
        return [seg for seg in segments if seg and seg.words]

    def _find_limit_index(self, words: List[WordTimestamp], start_idx: int) -> int:
        for idx in range(start_idx, len(words)):
            tokens = idx - start_idx + 1
            duration = self._segment_duration(words, start_idx, idx)
            if tokens >= self.config.max_tokens or duration >= self.config.max_duration:
                return idx
        return len(words) - 1

    def _find_best_boundary(
        self,
        words: List[WordTimestamp],
        start_idx: int,
        limit_idx: int,
    ) -> Optional[int]:
        best_weak: Optional[int] = None
        for idx in range(limit_idx, start_idx - 1, -1):
            strength = self._get_boundary_strength(words, idx)
            if strength >= 3:
                return idx
            if strength >= 1 and best_weak is None:
                best_weak = idx
        return best_weak

    def _merge_short_segments(self, segments: List[SentenceSegment]) -> List[SentenceSegment]:
        if len(segments) <= 1:
            return segments

        merged: List[SentenceSegment] = []
        idx = 0
        while idx < len(segments):
            current = segments[idx]
            if not current.words:
                idx += 1
                continue

            tokens = len(current.words)
            duration = current.end - current.start
            is_short = tokens < self.config.min_tokens or duration < self.config.min_duration

            if is_short and merged:
                candidate = self._merge_two_segments(merged[-1], current)
                if len(candidate.words) <= self.config.max_tokens and (
                    candidate.end - candidate.start <= self.config.max_duration * 1.2
                ):
                    merged[-1] = candidate
                    idx += 1
                    continue

            if is_short and idx + 1 < len(segments):
                candidate = self._merge_two_segments(current, segments[idx + 1])
                if len(candidate.words) <= self.config.max_tokens:
                    merged.append(candidate)
                    idx += 2
                    continue

            merged.append(current)
            idx += 1

        return merged

    def _merge_continuation_segments(self, segments: List[SentenceSegment]) -> Tuple[List[SentenceSegment], int]:
        """切分后回并：处理“续接词开头”导致的误切片段。"""
        if len(segments) <= 1:
            return segments, 0

        merged: List[SentenceSegment] = [segments[0]]
        merge_count = 0
        for current in segments[1:]:
            previous = merged[-1]
            if self._should_merge_continuation_pair(previous, current):
                candidate = self._merge_two_segments(previous, current)
                if (
                    len(candidate.words) <= self.config.max_tokens
                    and (candidate.end - candidate.start) <= self.config.max_duration * 1.2
                ):
                    merged[-1] = candidate
                    merge_count += 1
                    continue
            merged.append(current)

        return merged, merge_count

    def _should_merge_continuation_pair(self, left: SentenceSegment, right: SentenceSegment) -> bool:
        if not left.words or not right.words:
            return False
        strategy = self._sentence_builder.config.get_strategy()
        left_last = self._normalize_boundary_token(left.words[-1].word or "")
        right_probe = self._build_probe_text(right.words, 0)
        left_end = float(getattr(left.words[-1], "end", 0.0) or 0.0)
        right_start = float(getattr(right.words[0], "start", left_end) or left_end)
        gap_sec = max(0.0, right_start - left_end)
        if gap_sec > self._CONTINUATION_MERGE_MAX_GAP_SEC:
            return False

        right_real_word_count = sum(1 for item in right.words if not getattr(item, "is_pseudo", False))
        if right_real_word_count <= 0:
            return False
        right_duration = max(
            0.0,
            float(getattr(right, "end", right_start) or right_start)
            - float(getattr(right, "start", right_start) or right_start),
        )
        right_probe_compact = str(right_probe or "").replace(" ", "")
        is_cjk_context = self._is_cjk_text(left_last) or self._is_cjk_text(right_probe_compact)
        is_cjk_discourse_break = bool(
            is_cjk_context
            and any(
                right_probe_compact.startswith(marker)
                for marker in self._CJK_DISCOURSE_BREAK_MARKERS
            )
        )

        if right_probe and strategy.is_continuation(right_probe):
            if is_cjk_discourse_break:
                return False
            if is_cjk_context:
                return (
                    right_real_word_count <= self._CJK_CONTINUATION_MERGE_MAX_WORDS
                    and right_duration <= self._CJK_CONTINUATION_MERGE_MAX_DURATION_SEC
                )
            return True
        if left_last and strategy.is_incomplete_ending(left_last):
            if is_cjk_context:
                if is_cjk_discourse_break:
                    return False
                return (
                    right_real_word_count <= (self._CJK_CONTINUATION_MERGE_MAX_WORDS + 1)
                    and right_duration <= (self._CJK_CONTINUATION_MERGE_MAX_DURATION_SEC + 0.8)
                )
            return True
        return False

    def _merge_two_segments(self, left: SentenceSegment, right: SentenceSegment) -> SentenceSegment:
        words = left.words + right.words
        return self._build_sentence(words, 0, len(words) - 1)

    @staticmethod
    def _next_real_word_index(words: List[WordTimestamp], boundary_idx: int) -> Optional[int]:
        next_idx = boundary_idx + 1
        while next_idx < len(words) and getattr(words[next_idx], "is_pseudo", False):
            next_idx += 1
        if next_idx >= len(words):
            return None
        return next_idx

    def _build_probe_text(
        self,
        words: Sequence[WordTimestamp],
        start_idx: int,
        window_size: int = 3,
    ) -> str:
        tokens: List[str] = []
        idx = start_idx
        while idx < len(words) and len(tokens) < window_size:
            if getattr(words[idx], "is_pseudo", False):
                idx += 1
                continue
            token = self._normalize_boundary_token(words[idx].word or "")
            if token:
                tokens.append(token)
            idx += 1
        return " ".join(tokens)

    @staticmethod
    def _normalize_boundary_token(token: str) -> str:
        trim_chars = "".join(_STRONG_PUNCT | _WEAK_PUNCT) + "\"'”’）)]}】」』"
        return token.replace("▁", " ").strip().strip(trim_chars)

    @staticmethod
    def _is_ascii_word(token: str) -> bool:
        return bool(token) and all(ch.isascii() for ch in token)

    @staticmethod
    def _is_cjk_text(text: str) -> bool:
        for char in str(text or ""):
            if "\u4e00" <= char <= "\u9fff":
                return True
            if "\u3040" <= char <= "\u30ff":
                return True
            if "\uac00" <= char <= "\ud7af":
                return True
        return False

    def _build_sentence(
        self,
        words: List[WordTimestamp],
        start_idx: int,
        end_idx: int,
    ) -> SentenceSegment:
        start_idx = max(0, start_idx)
        end_idx = min(len(words) - 1, end_idx)
        slice_words = words[start_idx:end_idx + 1]
        sentence = self._sentence_builder._create_sentence(slice_words, force_create=True)
        if sentence is None:
            start = slice_words[0].start
            end = slice_words[-1].end
            sentence = SentenceSegment(
                text="".join(word.word for word in slice_words),
                text_clean="".join(word.word for word in slice_words),
                start=start,
                end=end,
                words=slice_words,
            )
        return sentence

    def _segment_duration(self, words: List[WordTimestamp], start_idx: int, end_idx: int) -> float:
        start = words[start_idx].start
        end = words[end_idx].end
        return max(0.0, end - start)

    def _get_punct_strength(
        self,
        words: List[WordTimestamp],
        idx: int,
        external_strength: Dict[int, int],
    ) -> int:
        word = words[idx].word or ""
        next_word = words[idx + 1].word if idx + 1 < len(words) else None
        token_strength = self._trailing_punct_strength(word, next_word)
        return max(token_strength, external_strength.get(idx, 0))

    # V3.2.0+dev.20260206.03: 长句二次切分需要统一边界强度函数。
    # 这里复用“词尾标点 + 时间停顿”的联合评分，避免 _find_best_boundary 调用缺失。
    def _get_boundary_strength(self, words: List[WordTimestamp], idx: int) -> int:
        next_word = words[idx + 1].word if idx + 1 < len(words) else None
        token_strength = self._trailing_punct_strength(words[idx].word or "", next_word)
        pause_strength = self._pause_strength(words, idx)
        return max(token_strength, pause_strength)

    def _pause_strength(self, words: List[WordTimestamp], idx: int) -> int:
        if idx >= len(words) - 1:
            return 0
        # V3.2.0+dev.20260205.10: 伪对齐词可能压扁时间戳，跳过连续伪词以恢复真实停顿切分能力。
        # 伪对齐词可能会“填平”真实停顿，导致 pause_strength 失效。
        # 这里跳过连续的 is_pseudo 词，使用下一个非伪词作为边界参考。
        # 关键策略：只在“下一段非伪词的前一个词”处触发边界（即 next_idx - 1），
        # 这样能把中间的伪词一起归入前一句，避免产生“伪词单独成句”的碎片化结果，
        # 同时也避免把边界错误地提前到更早的真实词上。
        next_idx = idx + 1
        while next_idx < len(words) and getattr(words[next_idx], "is_pseudo", False):
            next_idx += 1
        if next_idx >= len(words):
            return 0
        boundary_idx = next_idx - 1
        if idx != boundary_idx:
            return 0

        current_end = float(words[idx].end or 0.0)
        if getattr(words[idx], "is_pseudo", False):
            for prev in range(idx - 1, -1, -1):
                if not getattr(words[prev], "is_pseudo", False):
                    current_end = float(words[prev].end or current_end)
                    break

        gap = float(words[next_idx].start or 0.0) - current_end
        if gap >= self.config.long_pause:
            return 2
        if gap >= self.config.soft_pause:
            return 1
        return 0

    def _trailing_punct_strength(self, token: str, next_token: Optional[str] = None) -> int:
        if not token:
            return 0
        if is_sentence_end_punct(token, next_token):
            return 3
        trailing = self._extract_trailing_punct(token)
        if not trailing:
            return 0
        if any(ch in _STRONG_PUNCT for ch in trailing):
            return 3
        if any(ch in _WEAK_PUNCT for ch in trailing):
            return 1
        return 0

    def _extract_trailing_punct(self, token: str) -> str:
        trailing = []
        idx = len(token) - 1
        while idx >= 0:
            char = token[idx]
            if char in _STRONG_PUNCT or char in _WEAK_PUNCT:
                if char in {".", "。"} and self._is_decimal_char(token, idx):
                    break
                trailing.append(char)
                idx -= 1
                continue
            if char.isspace():
                idx -= 1
                continue
            break
        return "".join(reversed(trailing))

    @staticmethod
    def _is_decimal_char(token: str, idx: int) -> bool:
        if idx <= 0 or idx >= len(token) - 1:
            return False
        if token[idx] not in {".", "。"}:
            return False
        return token[idx - 1].isdigit() and token[idx + 1].isdigit()

    def _build_external_strength(
        self,
        *,
        clean_text: Optional[str],
        words: List[WordTimestamp],
        positions: Sequence[PuncPosition],
    ) -> tuple[Dict[int, int], Dict[str, float]]:
        stats = {
            "mapping_coverage": 0.0,
            "writeback_ratio": 0.0,
            "writeback_used": 0.0,
            "writeback_blocked": 0.0,
        }
        if not clean_text or not positions or not words:
            return {}, stats
        char_to_word, coverage = self._build_char_to_word_map(clean_text, words)
        stats["mapping_coverage"] = coverage
        strength_map: Dict[int, int] = {}
        mapped_positions = 0
        for pos in positions:
            if pos.char_index < 0 or pos.char_index >= len(clean_text):
                continue
            word_index = char_to_word[pos.char_index] if pos.char_index < len(char_to_word) else None
            if word_index is None:
                word_index = self._fallback_word_index(char_to_word, pos.char_index)
            if word_index is None:
                continue
            mapped_positions += 1
            strength = 3 if pos.punctuation in _STRONG_PUNCT else 1
            prev = strength_map.get(word_index, 0)
            if strength > prev:
                strength_map[word_index] = strength
        stats["writeback_ratio"] = mapped_positions / max(len(positions), 1)
        if coverage < self.config.min_mapping_coverage:
            stats["writeback_blocked"] = 1.0
            self.logger.debug(
                "外部标点映射覆盖率过低=%.2f，禁用回写",
                coverage,
            )
            return {}, stats
        stats["writeback_used"] = 1.0 if strength_map else 0.0
        return strength_map, stats

    def _build_char_to_word_map(
        self,
        clean_text: str,
        words: List[WordTimestamp],
    ) -> tuple[List[Optional[int]], float]:
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
        token = token.rstrip("".join(_STRONG_PUNCT | _WEAK_PUNCT))
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
