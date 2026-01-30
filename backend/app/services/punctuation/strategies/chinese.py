"""
中文标点策略（WeTextProcessing + CT-Transformer 适配）。
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
from app.services.punctuation.models import CTTransformerOnnxAdapter, WeTextProcessor
from app.services.punctuation.models.wetext_processor import CharMapping


class ChinesePunctuationStrategy(PunctuationStrategy):
    """中文标点策略（策略模式：独立封装中文标点恢复流程）。"""

    def __init__(self) -> None:
        config = get_punctuation_config().get("chinese", {})
        self._model_id = config.get("model_id", "punct-ct-transformer-zh")
        self._itn_first = bool(config.get("itn_first", True))
        self._fallback_on_oov = bool(config.get("fallback_on_oov", True))
        self._adapter = CTTransformerOnnxAdapter(self._model_id)
        self._wetext = WeTextProcessor()

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
        start = time.perf_counter()
        if not text:
            return PunctuationResult(text="", model_id=self._model_id)

        if self._itn_first:
            itn_result = await self._wetext.process(text)
            positions = self._adapter.predict(itn_result.normalized_text)
            mapped = self._map_positions(
                positions,
                itn_result.char_mapping,
                len(itn_result.original_text),
            )
            punct_text = apply_punctuation(itn_result.original_text, mapped)
        else:
            positions = self._adapter.predict(text)
            mapped = positions
            punct_text = apply_punctuation(text, positions)

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

    def _map_positions(
        self,
        positions: List[PuncPosition],
        mappings: List[CharMapping],
        original_length: int,
    ) -> List[PuncPosition]:
        if not positions or not mappings:
            return positions
        mapped: List[PuncPosition] = []
        for pos in positions:
            mapped_index = self._map_char_index(pos.char_index, mappings)
            if mapped_index is None:
                if self._fallback_on_oov:
                    mapped_index = pos.char_index
                else:
                    continue
            mapped_index = max(0, min(mapped_index, max(original_length - 1, 0)))
            mapped.append(
                PuncPosition(
                    char_index=mapped_index,
                    punctuation=pos.punctuation,
                    confidence=pos.confidence,
                )
            )
        return mapped

    @staticmethod
    def _map_char_index(char_index: int, mappings: List[CharMapping]) -> Optional[int]:
        for mapping in mappings:
            start, end = mapping.normalized_range
            if start <= char_index < end:
                if mapping.mapping_type == "identity":
                    return mapping.original_range[0] + (char_index - start)
                if mapping.mapping_type == "expand":
                    return mapping.original_range[0]
                if mapping.mapping_type == "collapse":
                    return max(mapping.original_range[1] - 1, mapping.original_range[0])
                return mapping.original_range[0]
        return None

    @staticmethod
    def _estimate_confidence(positions: List[PuncPosition]) -> float:
        if not positions:
            return 1.0
        return sum(pos.confidence for pos in positions) / len(positions)
