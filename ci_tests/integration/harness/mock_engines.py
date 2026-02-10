# -*- coding: utf-8 -*-
"""
Mock ASR 引擎工厂。

从 test_fast_bridge_slow_flow.py 提取的测试引擎替身，
以及新增的 ConfigurableEngine。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from app.core.asr.engine import ASREngine
from app.core.asr.enums import ASRCapability, TimestampPrecision
from app.core.asr.models import ASRMetadata, ASRResult, Segment, WordTimestamp
from app.engines.dummy_engine import DummyEngine


class RecordingWhisperEngine(ASREngine):
    """记录 Prompt 的慢流引擎替身。

    接收固定的 segments 列表，每次 transcribe 时返回这些 segments
    并记录传入的 initial_prompt。
    """

    def __init__(self, segments: List[dict]) -> None:
        self._segments = segments
        self.prompts: List[Optional[str]] = []

    def get_capabilities(self) -> List[ASRCapability]:
        return [
            ASRCapability.WORD_TIMESTAMPS,
            ASRCapability.CONFIDENCE_SCORES,
        ]

    def get_timestamp_precision(self) -> TimestampPrecision:
        return TimestampPrecision.WORD

    async def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        **kwargs: object,
    ) -> ASRResult:
        self.prompts.append(kwargs.get("initial_prompt"))
        segments: List[Segment] = []
        raw_segments: List[dict] = []
        words: List[WordTimestamp] = []
        full_text_parts: List[str] = []

        for item in self._segments:
            start = float(item["start"])
            end = float(item["end"])
            text = str(item["text"])
            full_text_parts.append(text)
            word = WordTimestamp(
                word=text,
                start=start,
                end=end,
                confidence=0.9,
                probability=0.9,
                is_pseudo=False,
            )
            words.append(word)
            segments.append(
                Segment(
                    start=start,
                    end=end,
                    text=text,
                    confidence=0.9,
                    words=[word],
                )
            )
            raw_segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "avg_logprob": -0.1,
                    "no_speech_prob": 0.0,
                    "words": [
                        {
                            "word": text,
                            "start": start,
                            "end": end,
                            "probability": 0.9,
                        }
                    ],
                }
            )

        return ASRResult(
            text="".join(full_text_parts),
            segments=segments,
            words=words,
            confidence=0.9,
            language=language or "zh",
            metadata=ASRMetadata(
                engine="recording",
                source="recording",
                timestamp_precision=TimestampPrecision.WORD,
                capabilities=self.get_capabilities(),
                raw_tags={"raw_result": {"text": "".join(full_text_parts), "segments": raw_segments}},
            ),
        )

    def estimate_confidence(self, raw_output: object) -> float:
        return 0.9


class ConfigurableEngine(ASREngine):
    """按 chunk_index 返回不同文本的可配置引擎。

    用于测试需要不同 chunk 返回不同内容的场景。
    """

    def __init__(self, responses: Dict[int, str], default_text: str = "默认文本") -> None:
        """
        Args:
            responses: {chunk_index: response_text} 映射
            default_text: 未映射的 chunk 默认返回文本
        """
        self._responses = responses
        self._default_text = default_text
        self._call_index = 0

    def get_capabilities(self) -> List[ASRCapability]:
        return [
            ASRCapability.WORD_TIMESTAMPS,
            ASRCapability.CONFIDENCE_SCORES,
        ]

    def get_timestamp_precision(self) -> TimestampPrecision:
        return TimestampPrecision.WORD

    async def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        **kwargs: object,
    ) -> ASRResult:
        text = self._responses.get(self._call_index, self._default_text)
        self._call_index += 1

        words: List[WordTimestamp] = []
        current_time = 0.0
        for char in text:
            words.append(
                WordTimestamp(
                    word=char,
                    start=current_time,
                    end=current_time + 0.1,
                    confidence=0.9,
                    is_pseudo=True,
                )
            )
            current_time += 0.1

        segment = Segment(
            start=0.0,
            end=current_time,
            text=text,
            confidence=0.9,
            words=words,
        )

        return ASRResult(
            text=text,
            segments=[segment],
            words=words,
            confidence=0.9,
            language=language or "zh",
            metadata=ASRMetadata(
                engine="configurable",
                source="configurable",
                timestamp_precision=TimestampPrecision.WORD,
                capabilities=self.get_capabilities(),
            ),
        )

    def estimate_confidence(self, raw_output: object) -> float:
        return 0.9


class MockEngineFactory:
    """测试引擎工厂，提供快捷构建方法。"""

    @staticmethod
    def create_dummy_draft(
        text: str = "测试文本",
        confidence: float = 0.95,
        latency_ms: int = 0,
    ) -> DummyEngine:
        """创建 Dummy 草稿引擎。"""
        return DummyEngine(
            response_text=text,
            confidence=confidence,
            latency_ms=latency_ms,
        )

    @staticmethod
    def create_recording_whisper(segments: List[dict]) -> RecordingWhisperEngine:
        """创建录制 Prompt 的 Whisper 替身。"""
        return RecordingWhisperEngine(segments=segments)

    @staticmethod
    def create_configurable_engine(
        responses: Dict[int, str],
        default_text: str = "默认文本",
    ) -> ConfigurableEngine:
        """创建按 chunk_index 返回不同文本的引擎。"""
        return ConfigurableEngine(responses=responses, default_text=default_text)
