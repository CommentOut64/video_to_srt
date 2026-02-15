"""
软切决策契约定义（Phase A）。
V3.2.0+dev.20260214.07
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class SpeakerChangeTag(str, Enum):
    """SpeakerChangeEvidence 标签枚举，禁止自由文本。"""

    REBOUND_MERGED = "rebound_merged"
    SUSPECTED_MISSED = "suspected_missed"
    CROSS_CHUNK_DEFERRED = "cross_chunk_deferred"


class EvidenceLevel(str, Enum):
    """说话人变化证据等级。"""

    HIGH = "high"
    MID = "mid"
    LOW = "low"


class AnchorType(str, Enum):
    """锚点类型。"""

    PAUSE_ANCHOR = "pause_anchor"
    WORD_BOUNDARY = "word_boundary"
    SEMANTIC_ANCHOR = "semantic_anchor"
    PUNCTUATION_ANCHOR = "punctuation_anchor"


class CutWindowState(str, Enum):
    """待切窗口状态。"""

    OPEN = "open"
    RESOLVED = "resolved"
    FORCED = "forced"
    EXPIRED = "expired"


class DeferredCutState(str, Enum):
    """延迟切分状态。"""

    PENDING = "pending"
    RESOLVED = "resolved"
    FORCED = "forced"
    EXPIRED = "expired"


@dataclass
class SpeakerChangeEvidence:
    """说话人变化证据。"""

    time: float
    from_speaker: str
    to_speaker: str
    pyannote_confidence: float
    pause_duration: float
    embedding_distance: float
    embedding_threshold: float
    is_abrupt_energy_shift: bool
    level: EvidenceLevel
    tags: set[SpeakerChangeTag] = field(default_factory=set)


@dataclass
class AnchorScore:
    """锚点评分。"""

    anchor_type: AnchorType
    anchor_time: float
    base_score: float
    distance_penalty: float
    final_score: float
    source: str


@dataclass
class CutWindow:
    """待切窗口。"""

    window_id: str
    trigger_time: float
    trigger_level: EvidenceLevel
    start_time: float
    end_time: float
    chunk_id: str
    candidate_anchors: list[AnchorScore] = field(default_factory=list)
    state: CutWindowState = CutWindowState.OPEN


@dataclass
class CutDecision:
    """切分决策。"""

    time: float
    window_id: str
    reason: str
    risk: Optional[str]
    anchor_type: AnchorType
    anchor_score: float
    depends_on_fast_draft: bool
    time_range: tuple[float, float]

    def __post_init__(self) -> None:
        if not self.window_id:
            raise ValueError("CutDecision.window_id 不能为空")


@dataclass
class DeferredCut:
    """延迟切分。"""

    deferred_id: str
    window_id: str
    created_at: float
    expected_resolve_by: float
    state: DeferredCutState = DeferredCutState.PENDING
    resolution_decision: Optional[CutDecision] = None


@dataclass
class CutPlan:
    """切分计划。"""

    plan_id: str
    block_id: str
    decisions: list[CutDecision] = field(default_factory=list)
    deferred_cuts: list[DeferredCut] = field(default_factory=list)
    generation_report: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "AnchorScore",
    "AnchorType",
    "CutDecision",
    "CutPlan",
    "CutWindow",
    "CutWindowState",
    "DeferredCut",
    "DeferredCutState",
    "EvidenceLevel",
    "SpeakerChangeEvidence",
    "SpeakerChangeTag",
]
