"""
日文标点恢复策略：字符级 BERT。
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
from app.services.punctuation.models.char_bert_onnx import CharBertOnnxAdapter


class JapanesePunctuationStrategy(PunctuationStrategy):
    """日文标点恢复策略。"""

    def __init__(
        self,
        model_id: str = "punct-char-bert-ja",
        logger: Optional[logging.Logger] = None,
        adapter: Optional[CharBertOnnxAdapter] = None,
    ) -> None:
        self._model_id = model_id
        self._logger = logger or logging.getLogger(__name__)
        self._adapter = adapter or CharBertOnnxAdapter(model_id=model_id, logger=self._logger)
        self._loaded = False

    @property
    def supported_languages(self) -> List[str]:
        return ["ja"]

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
            self._logger.warning("日文标点模型不可用，降级为原文: %s", exc)
            return PunctuationResult(
                text=text,
                model_id="fallback",
                processing_time_ms=measure_processing_ms(start_time),
            )

        positions = await self._predict_async(text)
        final_text = apply_punctuation(text, positions)
        split_points = build_split_points(text, positions, word_timestamps)
        confidence = self._aggregate_confidence(positions)
        return PunctuationResult(
            text=final_text,
            split_points=split_points,
            punctuation_positions=positions,
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

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        manager = get_model_manager_v2()
        try:
            manager.ensure_available(self.model_id)
            self._adapter.load("")
        except Exception as exc:
            raise PunctuationModelError(f"日文标点模型加载失败: {exc}") from exc
        self._loaded = True

    async def _predict_async(self, text: str) -> List[PuncPosition]:
        return await asyncio.to_thread(self._adapter.predict, text)

    @staticmethod
    def _aggregate_confidence(positions: Sequence[PuncPosition]) -> float:
        if not positions:
            return 0.0
        return sum(p.confidence for p in positions) / len(positions)
