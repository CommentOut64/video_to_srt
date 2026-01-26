"""
ASR 引擎抽象接口。
V3.2.0+dev.20260119.01
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import numpy as np

from app.core.asr.enums import ASRCapability, TimestampPrecision
from app.core.asr.models import ASRResult


class ASREngine(ABC):
    """ASR 引擎抽象基类。"""

    @abstractmethod
    def get_capabilities(self) -> List[ASRCapability]:
        """返回引擎能力列表。"""
        raise NotImplementedError

    @abstractmethod
    def get_timestamp_precision(self) -> TimestampPrecision:
        """返回时间戳精度等级。"""
        raise NotImplementedError

    @abstractmethod
    async def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> ASRResult:
        """转录音频并返回统一结果。"""
        raise NotImplementedError

    @abstractmethod
    def estimate_confidence(self, raw_output: Any) -> float:
        """估算置信度（由具体引擎实现）。"""
        raise NotImplementedError

    def get_engine_name(self) -> str:
        """返回引擎名称。"""
        return self.__class__.__name__
