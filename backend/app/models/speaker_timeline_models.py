"""
说话人时间线核心契约模型。

Phase 0 仅建立数据契约，不引入具体算法实现。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal


TimelineSource = Literal["cluster", "segmentation", "merged"]


@dataclass
class SpeakerTurn:
    """单个说话轮次。"""

    turn_id: str
    block_id: str
    speaker_id: str
    start: float
    end: float
    boundary_confidence: float
    is_overlap: bool
    source: TimelineSource

    @property
    def duration(self) -> float:
        """轮次持续时长（秒）。"""
        return self.end - self.start


@dataclass
class SpeakerProfile:
    """说话人聚合画像。"""

    speaker_id: str
    embedding_centroid: list[float]
    sample_count: int
    quality_score: float


@dataclass
class SpeakerTimeline:
    """块级说话人时间线。"""

    block_id: str
    turns: list[SpeakerTurn] = field(default_factory=list)
    speakers: list[SpeakerProfile] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

