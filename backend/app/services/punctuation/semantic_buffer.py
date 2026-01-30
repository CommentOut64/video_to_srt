"""
语义缓冲器（Phase C 语义缓冲实现）。
V3.2.0+dev.20260130.08
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Sequence

from app.models.sensevoice_models import SentenceSegment, TextSource
from app.services.punctuation.base import PunctuationResult, SplitPoint


_WEAK_PUNCTUATION = set("，、；,;:")
_PUNCTUATION_SET = set(",.!?;:\"()[]{}，。！？；：、（）【】《》“”「」『』")
_TRAILING_PUNCTUATION = set(",，;:；：、。.")


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
        self._max_pending_chars = max(1, int(max_pending_chars))
        self._max_pending_duration = max(0.0, float(max_pending_duration))
        self._force_flush_duration = max(0.0, float(force_flush_duration))
        self._pending_text = ""
        self._pending_audio_start = 0.0
        self._pending_audio_end = 0.0
        self._pending_language = "auto"
        self._pending_source_chunks: List[str] = []
        self._pending_decision: Optional[PunctuationDecision] = None
        self._pending_punctuation_result: Optional[PunctuationResult] = None
        self._last_speaker_id: Optional[str] = None
        self._pending_words: List[Dict[str, Any]] = []

    def add(self, item: SemanticBufferInput) -> List[SemanticChunk]:
        """添加快流标点结果，返回可输出的语义块。"""
        if not item.text:
            return []
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
        combined_sources = self._pending_source_chunks + (item.source_chunks or [item.chunk_id])
        current_words = self._normalize_words(item.word_timestamps, item.audio_range)
        combined_words = self._pending_words + current_words

        decision = item.punctuation_decision or self._pending_decision
        punct_result = item.punctuation_result or self._pending_punctuation_result
        outputs.extend(
            self._split_and_update(
                combined_text=combined_text,
                audio_range=(combined_start, combined_end),
                language=combined_language,
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
        if not self._pending_text:
            return []
        chunk = self._build_chunk(
            text=self._pending_text,
            audio_range=(self._pending_audio_start, self._pending_audio_end),
            language=self._pending_language,
            source_chunks=self._pending_source_chunks,
            decision=self._pending_decision,
            split_end_indices=[len(self._pending_text) - 1],
            pending_tail="",
            punctuation_result=self._pending_punctuation_result,
            word_timestamps=self._pending_words,
        )
        self._reset_pending()
        self._logger.debug("语义缓冲强制刷新: reason=%s", reason)
        return [chunk]

    def _split_and_update(
        self,
        *,
        combined_text: str,
        audio_range: Tuple[float, float],
        language: str,
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
                    source_chunks=source_chunks,
                    decision=decision,
                    split_end_indices=split_end_indices,
                    pending_tail=combined_text[last_end + 1 :],
                    punctuation_result=self._clone_result(punct_result, chunk_text, split_end_indices, chunk_range),
                    word_timestamps=chunk_words,
                )
                outputs.append(chunk)
                self._update_pending(
                    text=combined_text[last_end + 1 :],
                    audio_range=pending_range,
                    language=language,
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
            while end_index + 1 < len(text) and text[end_index + 1] in _PUNCTUATION_SET:
                end_index += 1
            if end_index <= last_end:
                continue
            results.append(end_index)
            last_end = end_index
        return results

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
            source_chunks=self._pending_source_chunks,
            decision=self._pending_decision,
            split_end_indices=[weak_index],
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
        source_chunks: List[str],
        decision: Optional[PunctuationDecision],
        split_end_indices: List[int],
        pending_tail: str,
        punctuation_result: Optional[PunctuationResult],
        word_timestamps: Optional[List[Dict[str, Any]]] = None,
    ) -> SemanticChunk:
        confidence = punctuation_result.confidence if punctuation_result else 1.0
        sentences = self._build_sentences(text, audio_range, split_end_indices, confidence, word_timestamps or [])
        return SemanticChunk(
            chunk_id=self._build_chunk_id(source_chunks),
            text=text,
            sentences=sentences,
            punctuation_result=punctuation_result,
            punctuation_decision=decision,
            pending_tail=pending_tail,
            audio_range=audio_range,
            language=language,
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
    ) -> List[SentenceSegment]:
        if not text:
            return []
        if not split_end_indices:
            split_end_indices = [len(text) - 1]
        sentences: List[SentenceSegment] = []
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
            sentences.append(
                SentenceSegment(
                    text=sentence_text,
                    text_clean=self._strip_trailing_punctuation(sentence_text).strip(),
                    start=seg_start,
                    end=seg_end,
                    words=[],
                    confidence=confidence,
                    source=TextSource.SENSEVOICE,
                    is_draft=True,
                    is_finalized=False,
                )
            )
            last_end_time = seg_end
            start_idx = end_idx + 1
        return sentences

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
        source_chunks: List[str],
        decision: Optional[PunctuationDecision],
        punctuation_result: Optional[PunctuationResult],
        pending_words: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._pending_text = text
        self._pending_audio_start = float(audio_range[0])
        self._pending_audio_end = float(audio_range[1])
        self._pending_language = language or self._pending_language
        self._pending_source_chunks = list(source_chunks)
        self._pending_decision = decision
        self._pending_punctuation_result = punctuation_result
        self._pending_words = pending_words or []

    def _reset_pending(self) -> None:
        self._pending_text = ""
        self._pending_audio_start = 0.0
        self._pending_audio_end = 0.0
        self._pending_language = "auto"
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
        return sum(1 for char in text if char not in _PUNCTUATION_SET)

    @staticmethod
    def _build_clean_index_map(text: str) -> List[int]:
        return [idx for idx, char in enumerate(text) if char not in _PUNCTUATION_SET]

    @staticmethod
    def _build_clean_text(text: str) -> str:
        return "".join(char for char in text if char not in _PUNCTUATION_SET)

    @staticmethod
    def _find_last_weak_punctuation(text: str) -> Optional[int]:
        for idx in range(len(text) - 1, -1, -1):
            if text[idx] in _WEAK_PUNCTUATION:
                return idx
        return None

    @staticmethod
    def _strip_trailing_punctuation(text: str) -> str:
        if not text:
            return text
        end = len(text)
        while end > 0 and text[end - 1] in _TRAILING_PUNCTUATION:
            end -= 1
        return text[:end]

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
            if not token:
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
            if char in _PUNCTUATION_SET:
                continue
            clean_idx += 1
            if idx >= text_index:
                break
        return max(clean_idx, 0)
