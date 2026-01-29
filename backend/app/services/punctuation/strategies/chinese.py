"""
中文标点恢复策略：WeTextProcessing + CT-Transformer。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional, Sequence, Tuple

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
from app.services.punctuation.models.ct_transformer_onnx import CTTransformerOnnxAdapter
from app.services.punctuation.models.wetext_processor import TextNormalizationResult, WeTextProcessor


class ChinesePunctuationStrategy(PunctuationStrategy):
    """中文标点恢复策略。"""

    def __init__(
        self,
        model_id: str = "punct-ct-transformer-zh",
        itn_first: bool = True,
        fallback_on_oov: bool = True,
        logger: Optional[logging.Logger] = None,
        wetext_processor: Optional[WeTextProcessor] = None,
        adapter: Optional[CTTransformerOnnxAdapter] = None,
    ) -> None:
        self._model_id = model_id
        self._itn_first = itn_first
        self._fallback_on_oov = fallback_on_oov
        self._logger = logger or logging.getLogger(__name__)
        self._wetext = wetext_processor
        self._adapter = adapter or CTTransformerOnnxAdapter(model_id=model_id, logger=self._logger)
        self._loaded = False

    @property
    def supported_languages(self) -> List[str]:
        return ["zh", "yue"]

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
            self._logger.warning("中文标点模型不可用，降级为原文: %s", exc)
            return PunctuationResult(
                text=text,
                model_id="fallback",
                processing_time_ms=measure_processing_ms(start_time),
            )

        if self._itn_first:
            itn_result = await self._wetext.process(text)
            normalized_text = itn_result.normalized_text
            positions = await self._predict_async(normalized_text)
            mapped, is_complete = self._map_back_positions(positions, itn_result)
            if not is_complete and self._fallback_on_oov:
                self._logger.info("中文 ITN 映射不完整，回退到原文标点策略")
                mapped = await self._predict_async(text)
            final_text = apply_punctuation(text, mapped)
            split_points = build_split_points(text, mapped, word_timestamps)
            confidence = self._aggregate_confidence(mapped)
            return PunctuationResult(
                text=final_text,
                split_points=split_points,
                punctuation_positions=mapped,
                confidence=confidence,
                model_id=self.model_id,
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
            raise PunctuationModelError(f"中文标点模型加载失败: {exc}") from exc
        if self._wetext is None:
            try:
                self._wetext = WeTextProcessor(logger=self._logger)
            except Exception as exc:
                self._logger.warning("WeTextProcessor 初始化失败，回退身份映射: %s", exc)
                self._wetext = WeTextProcessor(logger=self._logger)
        self._loaded = True

    async def _predict_async(self, text: str) -> List[PuncPosition]:
        return await asyncio.to_thread(self._adapter.predict, text)

    @staticmethod
    def _aggregate_confidence(positions: Sequence[PuncPosition]) -> float:
        if not positions:
            return 0.0
        return sum(p.confidence for p in positions) / len(positions)

    @staticmethod
    def _map_back_positions(
        positions: Sequence[PuncPosition],
        itn_result: TextNormalizationResult,
    ) -> Tuple[List[PuncPosition], bool]:
        if not positions:
            return [], True

        mapped: List[PuncPosition] = []
        is_complete = True
        for position in positions:
            mapped_index = _map_index(position.char_index, itn_result)
            if mapped_index is None:
                is_complete = False
                mapped_index = position.char_index
            mapped.append(
                PuncPosition(
                    char_index=mapped_index,
                    punctuation=position.punctuation,
                    confidence=position.confidence,
                )
            )
        return mapped, is_complete


def _map_index(
    normalized_index: int,
    itn_result: TextNormalizationResult,
) -> Optional[int]:
    for mapping in itn_result.char_mapping:
        start, end = mapping.normalized_range
        if start <= normalized_index < end:
            if mapping.mapping_type == "expand":
                return mapping.original_range[0]
            return max(mapping.original_range[1] - 1, 0)
    return None
