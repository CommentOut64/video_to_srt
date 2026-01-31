"""
ASR 抽象数据模型定义。
V3.2.0+dev.20260119.01
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.asr.enums import ASRCapability, TimestampPrecision


@dataclass
class WordTimestamp:
    """字/词级时间戳最小单位。"""

    word: str
    start: float
    end: float
    confidence: Optional[float] = None
    confidence_raw: Optional[float] = None
    confidence_display_raw: Optional[float] = None
    is_pseudo: bool = False
    probability: Optional[float] = None
    token_type: Optional[str] = None

    def __post_init__(self) -> None:
        """兼容旧数据：confidence_raw 默认对齐 confidence。"""
        if self.confidence_raw is None:
            self.confidence_raw = self.confidence


@dataclass
class Segment:
    """段级转录结果。"""

    start: float
    end: float
    text: str
    confidence: Optional[float] = None
    words: Optional[List[WordTimestamp]] = None
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ASRMetadata:
    """引擎元数据（保留引擎特有信息）。"""

    engine: str
    source: str
    timestamp_precision: TimestampPrecision
    capabilities: List[ASRCapability]
    model_version: Optional[str] = None
    processing_time: Optional[float] = None
    raw_tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ASRResult:
    """统一转录结果。"""

    text: str
    segments: List[Segment]
    confidence: float
    language: str
    metadata: ASRMetadata
    text_clean: Optional[str] = None
    words: Optional[List[WordTimestamp]] = None
    raw_tokens: Optional[List[Dict[str, Any]]] = None
    emotion: Optional[str] = None
    event_tags: Optional[List[str]] = None
