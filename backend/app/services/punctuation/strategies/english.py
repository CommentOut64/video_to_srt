"""
英文标点策略（DistilBERT 适配）。
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence

from app.services.punctuation.base import (
    PuncPosition,
    PunctuationResult,
    PunctuationStrategy,
    WordTimestampLike,
    apply_punctuation,
    build_split_points,
)
from app.services.punctuation.config import get_punctuation_config
from app.services.punctuation.models import DistilBertPunctOnnxAdapter


class EnglishPunctuationStrategy(PunctuationStrategy):
    """英文标点策略（策略模式：独立封装英文标点恢复流程）。"""

    def __init__(self) -> None:
        config = get_punctuation_config().get("english", {})
        self._model_id = config.get("model_id", "punct-distilbert-en")
        self._overlap_words = int(config.get("overlap_words", 10))
        self._adapter = DistilBertPunctOnnxAdapter(self._model_id)

    @property
    def supported_languages(self) -> List[str]:
        return ["en"]

    @property
    def model_id(self) -> str:
        return self._model_id

    async def restore(
        self,
        text: str,
        word_timestamps: Optional[Sequence[WordTimestampLike]] = None,
        context: Optional[str] = None,
    ) -> PunctuationResult:
        start = time.perf_counter()
        if not text:
            return PunctuationResult(text="", model_id=self._model_id)

        context_text = self._select_context(context)
        if context_text:
            input_text = f"{context_text} {text}"
            context_len = len(context_text) + 1
        else:
            input_text = text
            context_len = 0

        positions = self._adapter.predict(input_text)
        mapped = self._trim_context_positions(positions, context_len)
        punct_text = apply_punctuation(text, mapped)

        split_points = build_split_points(
            text=punct_text,
            positions=mapped,
            word_timestamps=word_timestamps,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return PunctuationResult(
            text=punct_text,
            split_points=split_points,
            punctuation_positions=mapped,
            confidence=self._estimate_confidence(mapped),
            model_id=self._model_id,
            processing_time_ms=elapsed_ms,
        )

    def get_split_suggestion(
        self,
        result: PunctuationResult,
        min_sentence_length: int = 5,
        max_sentence_length: int = 50,
    ) -> List[int]:
        return [point.char_index for point in result.split_points]

    def _select_context(self, context: Optional[str]) -> str:
        if not context:
            return ""
        words = context.strip().split()
        if not words:
            return ""
        if len(words) <= self._overlap_words:
            return " ".join(words)
        return " ".join(words[-self._overlap_words :])

    @staticmethod
    def _trim_context_positions(
        positions: List[PuncPosition],
        context_len: int,
    ) -> List[PuncPosition]:
        if context_len <= 0:
            return positions
        trimmed: List[PuncPosition] = []
        for pos in positions:
            if pos.char_index < context_len:
                continue
            trimmed.append(
                PuncPosition(
                    char_index=pos.char_index - context_len,
                    punctuation=pos.punctuation,
                    confidence=pos.confidence,
                )
            )
        return trimmed


    @staticmethod
    def _estimate_confidence(positions: List[PuncPosition]) -> float:
        if not positions:
            return 1.0
        return sum(pos.confidence for pos in positions) / len(positions)
