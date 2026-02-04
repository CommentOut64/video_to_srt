"""
对齐/规范化共享类型定义。
V3.2.0+dev.20260204.03
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from app.models.confidence_models import AlignedWord as AlignedWord
from app.services.punctuation.base import PuncPosition

if TYPE_CHECKING:
    from app.services.arbitration.arbiter import ArbitrationResult


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
    mapping_coverage: float = 0.0


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
class L1Input:
    """L1 规范化输入。"""

    sv_raw_result: Optional[Dict[str, Any]]
    wh_raw_result: Optional[Dict[str, Any]]
    language_hint: Optional[str] = None


@dataclass
class L1Output:
    """L1 规范化输出。"""

    sv_track: Optional[TextTrack] = None
    whisper_track: Optional[TextTrack] = None


@dataclass
class QualitySignals:
    """L2 仲裁质量信号（统一口径）。"""

    length_ratio: float = 1.0
    confidence_fast: float = 0.0
    confidence_slow: float = 0.0
    is_repetition: bool = False
    is_hallucination: bool = False
    is_itn_fallback: bool = False
    mapping_coverage: float = 0.0
    alignment_score: float = 0.0
    gap_ratio: float = 0.0

    @property
    def repetition_flag(self) -> bool:
        """兼容文档口径：repetition_flag。"""
        return self.is_repetition

    @property
    def hallucination_flag(self) -> bool:
        """兼容文档口径：hallucination_flag。"""
        return self.is_hallucination

    @property
    def itn_fallback_flag(self) -> bool:
        """兼容文档口径：itn_fallback_flag。"""
        return self.is_itn_fallback


@dataclass
class L2Input:
    """L2 文本仲裁输入。"""

    sv_track: Optional[TextTrack]
    whisper_track: Optional[TextTrack]
    quality_signals: QualitySignals


@dataclass
class L2Output:
    """L2 文本仲裁输出。"""

    chosen_text_track: Optional[TextTrack]
    arbitration_result: "ArbitrationResult"


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
    "L1Input",
    "L1Output",
    "L2Input",
    "L2Output",
    "NormalizationResult",
    "QualitySignals",
    "TextTrack",
    "TextTrackBundle",
    "AnnotatedWord",
]
