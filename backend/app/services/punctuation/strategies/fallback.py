"""
兜底策略：使用 ASR 默认标点透传输出。
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

from app.services.punctuation.base import (
    PunctuationResult,
    PunctuationStrategy,
    WordTimestampLike,
    measure_processing_ms,
)


class FallbackPunctuationStrategy(PunctuationStrategy):
    """兜底策略（ASR 默认标点透传）。"""

    # 策略模式：提供无模型依赖的兜底实现，确保缺失模型时仍可输出可读文本。

    @property
    def supported_languages(self) -> List[str]:
        return ["fallback", "auto"]

    @property
    def model_id(self) -> str:
        return "asr_fallback"

    async def restore(
        self,
        text: str,
        word_timestamps: Optional[Sequence[WordTimestampLike]] = None,
        context: Optional[str] = None,
    ) -> PunctuationResult:
        start_time = time.time()
        return PunctuationResult(
            text=text or "",
            model_id=self.model_id,
            processing_time_ms=measure_processing_ms(start_time),
        )

    def get_split_suggestion(
        self,
        result: PunctuationResult,
        min_sentence_length: int = 5,
        max_sentence_length: int = 50,
    ) -> List[int]:
        return []
