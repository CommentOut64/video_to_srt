"""
对齐/规范化共享类型定义。
V3.2.0+dev.20260203.03
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from app.models.confidence_models import AlignedWord as AlignedWord
from app.services.punctuation.base import PuncPosition


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
    itn_fallback: bool = False
    itn_fallback_reason: Optional[str] = None


@dataclass
class TextTrack:
    """统一文本轨道（单轨）。"""

    raw_text: str
    text_itn_raw: str
    text_clean: str
    char_mapping: List[CharMapping]
    raw_to_clean: List[Optional[int]]
    clean_to_raw: List[int]
    language: str = "auto"
    source: str = ""
    itn_fallback: bool = False
    itn_fallback_reason: Optional[str] = None
    clean_to_word: List[Optional[int]] = field(default_factory=list)
    punct_positions: List[PuncPosition] = field(default_factory=list)
    mapping_coverage: float = 0.0


@dataclass
class TextTrackBundle:
    """三轨文本结构（sv/whisper/chosen）。"""

    sv_track: Optional[TextTrack] = None
    whisper_track: Optional[TextTrack] = None
    chosen_track: Optional[TextTrack] = None


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
    "TextTrack",
    "TextTrackBundle",
    "AnnotatedWord",
]
