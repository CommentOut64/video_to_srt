"""
标点服务基础结构与抽象接口。
V3.2.0+dev.20260129.02
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence


class PunctuationError(RuntimeError):
    """标点系统基础异常。"""


class PunctuationUnavailableError(PunctuationError):
    """标点不可用（模型缺失或关闭）。"""


class PunctuationModelError(PunctuationError):
    """标点模型异常。"""


class PunctuationModelAdapter(ABC):
    """标点模型适配器基类。"""

    @abstractmethod
    def load(self, model_path: str) -> None:
        """加载模型。"""

    @abstractmethod
    def predict(self, text: str) -> List["PuncPosition"]:
        """预测标点位置。"""

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """返回模型信息。"""

    @abstractmethod
    def unload(self) -> None:
        """卸载模型。"""


class WordTimestampLike(Protocol):
    """词级时间戳协议（兼容 dict/对象）。"""

    word: str
    start: float
    end: float


@dataclass
class PuncPosition:
    """标点位置。"""

    char_index: int
    punctuation: str
    confidence: float = 1.0


@dataclass
class SplitPoint:
    """切分点。"""

    char_index: int
    relative_time: float
    punctuation: str
    confidence: float = 1.0


@dataclass
class PunctuationResult:
    """标点恢复结果。"""

    text: str
    model_id: str = "unknown"
    split_points: List[SplitPoint] = field(default_factory=list)
    punctuation_positions: List[PuncPosition] = field(default_factory=list)
    confidence: float = 1.0
    processing_time_ms: float = 0.0


class PunctuationStrategy(ABC):
    """标点恢复策略基类。"""

    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """支持的语言列表。"""

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
        """返回切分建议的字符位置。"""


_PUNCTUATION_SET = set(",.!?;:\"()[]{}，。！？；：、（）【】《》“”‘’「」『』")
# V3.2.0+dev.20260131.05: 保护小数点（中文句号误判）
_DECIMAL_PUNCTUATION = {".", "。"}


def _is_decimal_char(text: str, index: int) -> bool:
    if index <= 0 or index >= len(text) - 1:
        return False
    if text[index] not in _DECIMAL_PUNCTUATION:
        return False
    return text[index - 1].isdigit() and text[index + 1].isdigit()


def apply_punctuation(text: str, positions: Sequence[PuncPosition]) -> str:
    """将标点按位置插入到文本中。"""
    if not text or not positions:
        return text

    by_index: Dict[int, List[str]] = {}
    for position in sorted(positions, key=lambda item: item.char_index):
        idx = position.char_index
        if idx < 0:
            continue
        by_index.setdefault(idx, []).append(position.punctuation)

    chunks: List[str] = []
    last_index = len(text) - 1
    for idx, char in enumerate(text):
        chunks.append(char)
        for mark in by_index.get(idx, []):
            chunks.append(mark)

    # 允许追加到末尾
    if len(text) == 0:
        tail_marks = by_index.get(0, [])
    else:
        tail_marks = by_index.get(last_index + 1, [])
    if tail_marks:
        chunks.extend(tail_marks)

    return "".join(chunks)


def _build_clean_index_map(text: str) -> List[int]:
    """建立 clean_index -> text_index 的映射。"""
    mapping: List[int] = []
    for idx, char in enumerate(text):
        if char in _PUNCTUATION_SET and not _is_decimal_char(text, idx):
            continue
        mapping.append(idx)
    return mapping


def _is_decimal_punctuation(text: str, clean_index: int, punct: str, clean_map: List[int]) -> bool:
    """判断标点是否为数字小数点，避免将 2.5 等拆分成断句。"""
    if punct not in _DECIMAL_PUNCTUATION:
        return False
    if clean_index < 0 or clean_index >= len(clean_map):
        return False
    text_index = clean_map[clean_index]
    if text_index < 0 or text_index >= len(text):
        return False
    left_char = text[text_index]
    if not left_char.isdigit():
        return False

    cursor = text_index + 1
    while cursor < len(text) and text[cursor] in _PUNCTUATION_SET:
        cursor += 1
    if cursor >= len(text):
        return False
    right_char = text[cursor]
    if not right_char.isdigit():
        return False

    punct_cluster = text[text_index + 1:cursor]
    return punct in punct_cluster


def build_split_points(
    text: str,
    positions: Sequence[PuncPosition],
    word_timestamps: Optional[Sequence[WordTimestampLike]] = None,
    sentence_end_chars: Optional[Iterable[str]] = None,
) -> List[SplitPoint]:
    """根据标点位置生成切分点。"""
    if not text or not positions:
        return []
    sentence_end = set(sentence_end_chars or "。！？.!?")
    relative_time = _estimate_relative_time(word_timestamps)
    clean_map = _build_clean_index_map(text)
    split_points: List[SplitPoint] = []
    for position in positions:
        if position.punctuation not in sentence_end:
            continue
        if _is_decimal_punctuation(text, position.char_index, position.punctuation, clean_map):
            continue
        split_points.append(
            SplitPoint(
                char_index=position.char_index,
                relative_time=relative_time,
                punctuation=position.punctuation,
                confidence=position.confidence,
            )
        )
    return split_points


def _estimate_relative_time(word_timestamps: Optional[Sequence[WordTimestampLike]]) -> float:
    if not word_timestamps:
        return 0.0
    last = word_timestamps[-1]
    end_time = getattr(last, "end", None)
    if end_time is None and isinstance(last, dict):
        end_time = last.get("end")
    return float(end_time or 0.0)
