# -*- coding: utf-8 -*-
"""
Mock 服务集合。

从 test_fast_bridge_slow_flow.py 提取的测试服务替身。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.models.sensevoice_models import SentenceSegment
from app.services.audio.chunk_engine import AudioChunk
from app.services.punctuation.base import PunctuationResult, PuncPosition
from app.services.punctuation.scheduler import PunctuationMode


class StubPunctuationService:
    """记录标点调用的轻量替身。"""

    def __init__(self) -> None:
        self.calls: List[dict] = []

    async def restore(
        self,
        text: str,
        language: str = "zh",
        word_timestamps: Optional[Sequence[dict]] = None,
        context: Optional[str] = None,
        fallback_text: Optional[str] = None,
    ) -> PunctuationResult:
        self.calls.append(
            {
                "text": text,
                "language": language,
                "word_timestamps": word_timestamps,
            }
        )
        if not text:
            return PunctuationResult(text="", model_id="stub")
        end_index = max(len(text) - 1, 0)
        positions = [PuncPosition(char_index=end_index, punctuation="。", confidence=0.9)]
        return PunctuationResult(
            text=text,
            model_id="stub",
            split_points=[],
            punctuation_positions=positions,
            confidence=0.9,
            processing_time_ms=1.0,
        )


class StubAligner:
    """对齐阶段替身，避免真实对齐依赖。"""

    @dataclass
    class _Level:
        value: str

    @dataclass
    class _StubAlignmentService:
        """模拟 alignment_service 属性。"""
        gap_resolver: Any = None

    @dataclass
    class _StubFinalSplitter:
        """模拟 final_splitter 属性。"""
        config: Any = None

    def __init__(self) -> None:
        self.alignment_service = self._StubAlignmentService()
        self.final_splitter = self._StubFinalSplitter()

    async def align(self, whisper_result: dict, sv_result: dict, chunk: AudioChunk):
        text = whisper_result.get("text", "")
        sentence = SentenceSegment(
            text=text,
            text_clean=text,
            start=chunk.start,
            end=chunk.end,
            is_finalized=True,
            is_draft=False,
        )
        return [sentence], self._Level("stub")

    def split_sensevoice_only(self, sv_result: dict, chunk: AudioChunk):
        text = sv_result.get("text_clean") or sv_result.get("text") or ""
        sentence = SentenceSegment(
            text=text,
            text_clean=text,
            start=chunk.start,
            end=chunk.end,
            is_finalized=True,
            is_draft=False,
        )
        return [sentence]


class StubDecision:
    """调度决策替身。"""

    def __init__(self, is_slow_requested: bool, reason: str) -> None:
        self.is_slow_requested = is_slow_requested
        self.reason = reason


class StubPolicy:
    """调度策略替身。"""

    def __init__(self) -> None:
        self.mode = PunctuationMode.DUAL


class StubScheduler:
    """强制触发慢流标点的调度器替身。"""

    def __init__(self) -> None:
        self.policy = StubPolicy()

    def evaluate_fast(self, fast_result: Any, sv_confidence: Any = None) -> StubDecision:
        return StubDecision(True, "forced")


class NoOpProgressEmitter:
    """空操作进度发射器，接收所有 emit 调用但不做任何事。"""

    def emit(self, *args: Any, **kwargs: Any) -> None:
        pass

    def emit_phase(self, *args: Any, **kwargs: Any) -> None:
        pass

    def emit_overall(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        """对所有未定义方法返回空操作。"""
        return lambda *a, **kw: None
