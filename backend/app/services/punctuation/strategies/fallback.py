"""
兜底标点策略（规则化复用 ASR 原始标点）。
V3.2.0+dev.20260129.02
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence

from app.services.punctuation.base import (
    PuncPosition,
    PunctuationResult,
    PunctuationStrategy,
    WordTimestampLike,
    build_split_points,
)
from app.services.text_protection import should_skip_raw_punctuation


class RuleBasedPunctuationStrategy(PunctuationStrategy):
    """基于已有标点的规则化策略（策略模式：用于统一不同语言的规则处理）。"""

    def __init__(
        self,
        *,
        model_id: str,
        supported_languages: List[str],
        punctuation_chars: str,
        sentence_end_chars: str,
    ) -> None:
        self._model_id = model_id
        self._supported_languages = supported_languages
        self._punctuation_chars = set(punctuation_chars)
        self._sentence_end_chars = set(sentence_end_chars)

    @property
    def supported_languages(self) -> List[str]:
        return self._supported_languages

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
            return PunctuationResult(
                text="",
                split_points=[],
                punctuation_positions=[],
                confidence=1.0,
                model_id=self._model_id,
                processing_time_ms=0.0,
            )

        positions = self._extract_positions(text)
        split_points = build_split_points(
            text=text,
            positions=positions,
            word_timestamps=word_timestamps,
            sentence_end_chars=self._sentence_end_chars,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        return PunctuationResult(
            text=text,
            split_points=split_points,
            punctuation_positions=positions,
            confidence=1.0,
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

    def _extract_positions(self, text: str) -> List[PuncPosition]:
        positions: List[PuncPosition] = []
        for idx, char in enumerate(text):
            if char in self._punctuation_chars:
                if should_skip_raw_punctuation(text, idx, char):
                    continue
                positions.append(PuncPosition(char_index=idx, punctuation=char, confidence=1.0))
        return positions

class FallbackPunctuationStrategy(RuleBasedPunctuationStrategy):
    """默认兜底策略（策略模式：使用 ASR 默认标点作为安全降级）。"""

    def __init__(self) -> None:
        super().__init__(
            model_id="asr_fallback",
            supported_languages=["*"],
            punctuation_chars=",.!?;:'\"()[]{}，。！？；：、（）【】《》",
            sentence_end_chars="。！？.!?",
        )
