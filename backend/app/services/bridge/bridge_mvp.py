"""
Bridge 层最小可行版本（Phase D）。
V3.2.0+dev.20260201.02
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from app.models.sensevoice_models import SentenceSegment
from app.services.punctuation.semantic_buffer import SemanticChunk, PunctuationDecision


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _join_sentences(sentences: List[SentenceSegment]) -> str:
    parts = [sentence.text for sentence in sentences if sentence and sentence.text]
    if not parts:
        return ""
    if _has_cjk("".join(parts)):
        return "".join(parts)
    return " ".join(part.strip() for part in parts if part.strip())


@dataclass
class BridgeBatch:
    """Bridge 输出的批次。"""

    batch_id: str
    text: str
    sentences: List[SentenceSegment]
    audio_range: Tuple[float, float]
    source_chunks: List[str]
    prompt: str
    punctuation_decision: Optional[PunctuationDecision]
    created_at: float = field(default_factory=time.time)


class BridgeMVP:
    """Bridge 层最小可行版本：批次构建与 Prompt 生成。"""

    def __init__(
        self,
        batch_duration: float = 20.0,
        batch_sentences: int = 6,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.queue: List[SemanticChunk] = []
        self.last_batch_tail: str = ""
        self.batch_duration = max(1.0, float(batch_duration))
        self.batch_sentences = max(1, int(batch_sentences))
        self._logger = logger or logging.getLogger(__name__)

    def add(self, chunk: SemanticChunk) -> Optional[BridgeBatch]:
        """添加语义 Chunk，满足条件时返回批次。"""
        if not chunk:
            return None
        self.queue.append(chunk)
        if self._should_flush():
            return self._build_batch()
        return None

    def flush(self) -> Optional[BridgeBatch]:
        """强制刷新所有缓冲内容。"""
        if not self.queue:
            return None
        return self._build_batch()

    def _should_flush(self) -> bool:
        if not self.queue:
            return False
        total_sentences = sum(len(item.sentences) for item in self.queue if item.sentences)
        if total_sentences >= self.batch_sentences:
            return True
        start, end = self._compute_audio_range(self.queue)
        duration = max(0.0, end - start)
        return duration >= self.batch_duration

    def _build_batch(self) -> BridgeBatch:
        ordered = sorted(self.queue, key=lambda item: item.audio_range[0])
        self.queue.clear()

        sentences: List[SentenceSegment] = []
        source_chunks: List[str] = []
        for item in ordered:
            sentences.extend(item.sentences or [])
            source_chunks.extend(item.source_chunks or [])

        text = _join_sentences(sentences)
        audio_range = self._compute_audio_range(ordered)
        prompt = self.last_batch_tail or ""
        decision = self._merge_decision(ordered)

        self.last_batch_tail = self._build_tail_prompt(sentences)
        batch_id = uuid.uuid4().hex

        self._logger.debug(
            "BridgeMVP 批次构建: batch_id=%s, sentences=%d, duration=%.2fs",
            batch_id,
            len(sentences),
            max(0.0, audio_range[1] - audio_range[0]),
        )

        return BridgeBatch(
            batch_id=batch_id,
            text=text,
            sentences=sentences,
            audio_range=audio_range,
            source_chunks=source_chunks,
            prompt=prompt,
            punctuation_decision=decision,
        )

    @staticmethod
    def _compute_audio_range(chunks: List[SemanticChunk]) -> Tuple[float, float]:
        if not chunks:
            return 0.0, 0.0
        starts = [item.audio_range[0] for item in chunks]
        ends = [item.audio_range[1] for item in chunks]
        return min(starts), max(ends)

    @staticmethod
    def _build_tail_prompt(sentences: List[SentenceSegment], max_sentences: int = 2) -> str:
        if not sentences:
            return ""
        tail = sentences[-max_sentences:]
        return _join_sentences(tail)

    @staticmethod
    def _merge_decision(chunks: List[SemanticChunk]) -> Optional[PunctuationDecision]:
        decisions = [item.punctuation_decision for item in chunks if item.punctuation_decision]
        if not decisions:
            return None
        slow_decisions = [decision for decision in decisions if decision.is_slow_requested]
        if slow_decisions:
            reason = ";".join({decision.reason for decision in slow_decisions if decision.reason})
            mode = slow_decisions[0].mode
            return PunctuationDecision(is_slow_requested=True, reason=reason, mode=mode)
        return decisions[0]
