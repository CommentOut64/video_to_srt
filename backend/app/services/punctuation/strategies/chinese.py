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
from app.services.punctuation.models import CTTransformerOnnxAdapter


class ChinesePunctuationStrategy(PunctuationStrategy):
    """中文标点策略（策略模式：独立封装中文标点恢复流程）。"""

    def __init__(self) -> None:
        config = get_punctuation_config().get("chinese", {})
        self._model_id = config.get("model_id", "punct-ct-transformer-zh")
        self._adapter = CTTransformerOnnxAdapter(self._model_id)

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

        positions = self._adapter.predict(text)
        punct_text = apply_punctuation(text, positions)
        mapped = positions

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

    @staticmethod
    def _estimate_confidence(positions: List[PuncPosition]) -> float:
        if not positions:
            return 1.0
        return sum(pos.confidence for pos in positions) / len(positions)
