"""
语义缓冲器（Phase A 基础实现）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from app.services.punctuation.base import PunctuationResult


@dataclass
class SemanticChunk:
    """语义块（简化版）。"""

    text: str
    punctuation: PunctuationResult
    duration: float


class SemanticBuffer:
    """语义缓冲器：按时长与标点触发切分。"""

    def __init__(
        self,
        max_buffer_duration: float = 15.0,
        hard_limit_duration: float = 20.0,
        min_chunk_duration: float = 1.0,
        prefer_punctuation_split: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._max_buffer_duration = max_buffer_duration
        self._hard_limit_duration = hard_limit_duration
        self._min_chunk_duration = min_chunk_duration
        self._prefer_punctuation_split = prefer_punctuation_split
        self._buffer_texts: List[str] = []
        self._buffer_duration = 0.0
        self._last_result: Optional[PunctuationResult] = None

    def add(self, text: str, result: PunctuationResult, duration: float) -> List[SemanticChunk]:
        """添加一段文本并返回可输出的语义块。"""
        if not text:
            return []
        self._buffer_texts.append(text)
        self._buffer_duration += max(duration, 0.0)
        self._last_result = result

        if self._should_flush(result):
            return self.flush()
        return []

    def flush(self) -> List[SemanticChunk]:
        """强制输出缓冲区内容。"""
        if not self._buffer_texts or not self._last_result:
            return []

        text = " ".join(self._buffer_texts)
        chunk = SemanticChunk(
            text=text,
            punctuation=self._last_result,
            duration=self._buffer_duration,
        )
        self._buffer_texts = []
        self._buffer_duration = 0.0
        self._last_result = None
        return [chunk]

    def _should_flush(self, result: PunctuationResult) -> bool:
        if self._buffer_duration >= self._hard_limit_duration:
            return True
        if self._buffer_duration >= self._max_buffer_duration:
            return True
        if self._prefer_punctuation_split and result.split_points and self._buffer_duration >= self._min_chunk_duration:
            return True
        return False
