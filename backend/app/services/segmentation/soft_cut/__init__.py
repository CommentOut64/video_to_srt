"""
软切模块导出。
"""

from .evidence_builder import (
    AnchorCandidate,
    EvidenceBuildResult,
    EvidenceBuilder,
    EvidenceBuilderConfig,
    SpeakerChangeFact,
)
from .evidence_fusion import EvidenceFusion, EvidenceFusionConfig, EvidenceFusionResult
from .decision_engine import DecisionEngineConfig, SoftCutDecisionEngine, WindowDecisionContext
from .types import (
    AnchorScore,
    AnchorType,
    CutDecision,
    CutPlan,
    CutWindow,
    CutWindowState,
    DeferredCut,
    DeferredCutState,
    EvidenceLevel,
    SpeakerChangeEvidence,
    SpeakerChangeTag,
)

__all__ = [
    "AnchorCandidate",
    "AnchorScore",
    "AnchorType",
    "CutDecision",
    "CutPlan",
    "CutWindow",
    "CutWindowState",
    "DeferredCut",
    "DeferredCutState",
    "DecisionEngineConfig",
    "EvidenceBuildResult",
    "EvidenceBuilder",
    "EvidenceBuilderConfig",
    "EvidenceFusion",
    "EvidenceFusionConfig",
    "EvidenceFusionResult",
    "EvidenceLevel",
    "SoftCutDecisionEngine",
    "SpeakerChangeFact",
    "SpeakerChangeEvidence",
    "SpeakerChangeTag",
    "WindowDecisionContext",
]
