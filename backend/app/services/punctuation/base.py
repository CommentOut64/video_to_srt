"""
标点恢复基础数据结构与抽象接口。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence


class WordTimestampLike(Protocol):
    """字级时间戳协议（兼容多来源对象）。"""

    word: str
    start: float
    end: float
    confidence: Optional[float]


@dataclass
class PuncPosition:
    """标点位置描述（字符索引 + 标点）。"""

    char_index: int
    punctuation: str
    confidence: float = 0.5


@dataclass
class SplitPoint:
    """切分点建议。"""

    char_index: int
    relative_time: float
    punctuation: str
    confidence: float = 0.5


@dataclass
class PunctuationResult:
    """标点恢复结果。"""

    text: str
    split_points: List[SplitPoint] = field(default_factory=list)
    punctuation_positions: List[PuncPosition] = field(default_factory=list)
    confidence: float = 0.0
    model_id: str = "unknown"
    processing_time_ms: float = 0.0


class PunctuationError(RuntimeError):
    """标点处理异常。"""


class PunctuationModelError(PunctuationError):
    """模型加载或推理错误。"""


class PunctuationUnavailableError(PunctuationError):
    """模型不可用异常。"""


class PunctuationModelAdapter(ABC):
    """标点模型适配器基类。"""

    def __init__(self, model_id: str):
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @abstractmethod
    def load(self, model_path: str) -> None:
        """加载模型。"""

    @abstractmethod
    def predict(self, text: str) -> List[PuncPosition]:
        """预测标点位置。"""

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型元信息。"""

    @abstractmethod
    def unload(self) -> None:
        """卸载模型释放资源。"""


class PunctuationStrategy(ABC):
    """标点恢复策略基类。"""

    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """支持语言列表。"""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """模型标识符。"""

    @abstractmethod
    async def restore(
        self,
        text: str,
        word_timestamps: Optional[Sequence[WordTimestampLike]] = None,
        context: Optional[str] = None,
    ) -> PunctuationResult:
        """恢复标点。"""

    @abstractmethod
    def get_split_suggestion(
        self,
        result: PunctuationResult,
        min_sentence_length: int = 5,
        max_sentence_length: int = 50,
    ) -> List[int]:
        """返回切分建议（字符索引列表）。"""


def apply_punctuation(text: str, positions: Sequence[PuncPosition]) -> str:
    """按字符索引插入标点。"""
    if not text or not positions:
        return text

    sorted_positions = sorted(positions, key=lambda p: p.char_index)
    chars = list(text)
    offset = 0
    for position in sorted_positions:
        index = max(0, min(len(chars), position.char_index + 1 + offset))
        chars.insert(index, position.punctuation)
        offset += 1
    return "".join(chars)


def estimate_relative_time(
    text: str,
    char_index: int,
    word_timestamps: Optional[Sequence[WordTimestampLike]] = None,
) -> float:
    """估算相对时间戳，默认按文本长度线性映射。"""
    if not text:
        return 0.0

    if word_timestamps:
        start = word_timestamps[0].start
        end = word_timestamps[-1].end
        duration = max(end - start, 0.0)
    else:
        duration = 0.0

    ratio = char_index / max(len(text), 1)
    return max(duration * ratio, 0.0)


def build_split_points(
    text: str,
    positions: Sequence[PuncPosition],
    word_timestamps: Optional[Sequence[WordTimestampLike]] = None,
) -> List[SplitPoint]:
    """根据标点位置生成切分点。"""
    split_points: List[SplitPoint] = []
    for position in positions:
        relative_time = estimate_relative_time(text, position.char_index, word_timestamps)
        split_points.append(
            SplitPoint(
                char_index=position.char_index,
                relative_time=relative_time,
                punctuation=position.punctuation,
                confidence=position.confidence,
            )
        )
    return split_points


def measure_processing_ms(start_time: float) -> float:
    """计算处理耗时（毫秒）。"""
    return max((time.time() - start_time) * 1000.0, 0.0)
