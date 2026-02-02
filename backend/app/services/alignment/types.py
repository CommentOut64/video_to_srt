"""
对齐/规范化共享类型定义。
V3.2.0+dev.20260202.03
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.models.confidence_models import AlignedWord as AlignedWord


@dataclass
class CharMapping:
    """字符映射信息（raw -> clean）。"""

    raw_idx: int
    clean_idx: Optional[int]
    punct: Optional[str] = None


@dataclass
class NormalizationResult:
    """统一规范化输出（分轨文本）。"""

    text_itn_raw: str
    text_clean: str
    char_mapping: List[CharMapping]
    raw_to_clean: List[Optional[int]]
    clean_to_raw: List[int]


@dataclass
class AnnotatedWord:
    """语义注入后的词信息（预留）。"""

    word: str
    start: Optional[float]
    end: Optional[float]
    trailing_punct: str = ""


__all__ = [
    "AlignedWord",
    "CharMapping",
    "NormalizationResult",
    "AnnotatedWord",
]
