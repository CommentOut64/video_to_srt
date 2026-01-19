"""
Dummy 引擎（测试用）。
V3.2.0+dev.20260119.02
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

import numpy as np

from app.core.asr.engine import ASREngine
from app.core.asr.enums import ASRCapability, TimestampPrecision
from app.core.asr.models import ASRMetadata, ASRResult, Segment, WordTimestamp


class DummyEngine(ASREngine):
    """哑引擎：用于无模型环境的快速测试。"""

    def __init__(
        self,
        response_text: str = "这是测试文本",
        confidence: float = 0.95,
        latency_ms: int = 100,
        simulate_failure: bool = False,
    ) -> None:
        self.response_text = response_text
        self.confidence = confidence
        self.latency_ms = latency_ms
        self.simulate_failure = simulate_failure
        self._call_count = 0

    def get_capabilities(self) -> List[ASRCapability]:
        return [
            ASRCapability.WORD_TIMESTAMPS,
            ASRCapability.CONFIDENCE_SCORES,
            ASRCapability.LANGUAGE_DETECTION,
        ]

    def get_timestamp_precision(self) -> TimestampPrecision:
        return TimestampPrecision.WORD

    async def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        **kwargs: object,
    ) -> ASRResult:
        self._call_count += 1

        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)

        if self.simulate_failure:
            raise RuntimeError("DummyEngine 模拟失败")

        words = self._build_words(self.response_text)
        segments = [
            Segment(
                start=0.0,
                end=words[-1].end if words else 0.0,
                text=self.response_text,
                confidence=self.confidence,
                words=words,
            )
        ]

        return ASRResult(
            text=self.response_text,
            segments=segments,
            words=words,
            confidence=self.confidence,
            language=language or "zh",
            metadata=ASRMetadata(
                engine="dummy",
                source="dummy",
                timestamp_precision=TimestampPrecision.WORD,
                capabilities=self.get_capabilities(),
                raw_tags={"call_count": self._call_count},
            ),
        )

    def estimate_confidence(self, raw_output: object) -> float:
        return self.confidence

    def get_call_count(self) -> int:
        """获取调用次数（测试用）。"""
        return self._call_count

    def _build_words(self, text: str) -> List[WordTimestamp]:
        words: List[WordTimestamp] = []
        current_time = 0.0
        for char in text:
            words.append(
                WordTimestamp(
                    word=char,
                    start=current_time,
                    end=current_time + 0.1,
                    confidence=self.confidence,
                    is_pseudo=True,
                )
            )
            current_time += 0.1
        return words
