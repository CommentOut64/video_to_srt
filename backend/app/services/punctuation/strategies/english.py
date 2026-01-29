"""
英文标点恢复策略：Edge-Punct-Casing + 重叠上下文。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional, Sequence

from app.services.model_manager_v2 import get_model_manager_v2
from app.services.punctuation.base import (
    PuncPosition,
    PunctuationModelError,
    PunctuationResult,
    PunctuationStrategy,
    WordTimestampLike,
    apply_punctuation,
    build_split_points,
    measure_processing_ms,
)
from app.services.punctuation.models.edge_punct_onnx import EdgePunctOnnxAdapter


class EnglishPunctuationStrategy(PunctuationStrategy):
    """英文标点恢复策略。"""

    def __init__(
        self,
        model_id: str = "punct-edge-punct-en",
        overlap_words: int = 10,
        logger: Optional[logging.Logger] = None,
        adapter: Optional[EdgePunctOnnxAdapter] = None,
    ) -> None:
        self._model_id = model_id
        self._overlap_words = overlap_words
        self._logger = logger or logging.getLogger(__name__)
        self._adapter = adapter or EdgePunctOnnxAdapter(model_id=model_id, logger=self._logger)
        self._prev_tail: Optional[str] = None
        self._loaded = False

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
        start_time = time.time()
        if not text:
            return PunctuationResult(
                text="",
                model_id=self.model_id,
                processing_time_ms=measure_processing_ms(start_time),
            )

        try:
            await self._ensure_loaded()
        except PunctuationModelError as exc:
            self._logger.warning("英文标点模型不可用，降级为原文: %s", exc)
            return PunctuationResult(
                text=text,
                model_id="fallback",
                processing_time_ms=measure_processing_ms(start_time),
            )

        input_text, context_len = self._build_input(text, context)
        positions = await self._predict_async(input_text)
        trimmed = self._trim_context_positions(positions, context_len)
        final_text = apply_punctuation(text, trimmed)
        split_points = build_split_points(text, trimmed, word_timestamps)
        confidence = self._aggregate_confidence(trimmed)
        self._prev_tail = self._extract_tail(text)

        return PunctuationResult(
            text=final_text,
            split_points=split_points,
            punctuation_positions=trimmed,
            confidence=confidence,
            model_id=self.model_id,
            processing_time_ms=measure_processing_ms(start_time),
        )

    def get_split_suggestion(
        self,
        result: PunctuationResult,
        min_sentence_length: int = 5,
        max_sentence_length: int = 50,
    ) -> List[int]:
        suggestions: List[int] = []
        last_index = 0
        for point in result.split_points:
            length = point.char_index - last_index + 1
            if length < min_sentence_length:
                continue
            if length > max_sentence_length:
                continue
            suggestions.append(point.char_index)
            last_index = point.char_index + 1
        return suggestions

    def reset_context(self) -> None:
        """手动重置上下文缓存。"""
        self._prev_tail = None

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        manager = get_model_manager_v2()
        try:
            manager.ensure_available(self.model_id)
            self._adapter.load("")
        except Exception as exc:
            raise PunctuationModelError(f"英文标点模型加载失败: {exc}") from exc
        self._loaded = True

    async def _predict_async(self, text: str) -> List[PuncPosition]:
        return await asyncio.to_thread(self._adapter.predict, text)

    def _build_input(self, text: str, context: Optional[str]) -> tuple[str, int]:
        if context:
            input_text = f"{context} {text}"
            return input_text, len(context) + 1
        if self._prev_tail:
            input_text = f"{self._prev_tail} {text}"
            return input_text, len(self._prev_tail) + 1
        return text, 0

    @staticmethod
    def _trim_context_positions(positions: List[PuncPosition], context_len: int) -> List[PuncPosition]:
        if context_len <= 0:
            return positions
        trimmed: List[PuncPosition] = []
        for position in positions:
            if position.char_index < context_len:
                continue
            trimmed.append(
                PuncPosition(
                    char_index=position.char_index - context_len,
                    punctuation=position.punctuation,
                    confidence=position.confidence,
                )
            )
        return trimmed

    def _extract_tail(self, text: str) -> str:
        words = text.split()
        if len(words) <= self._overlap_words:
            return text
        return " ".join(words[-self._overlap_words :])

    @staticmethod
    def _aggregate_confidence(positions: Sequence[PuncPosition]) -> float:
        if not positions:
            return 0.0
        return sum(p.confidence for p in positions) / len(positions)
