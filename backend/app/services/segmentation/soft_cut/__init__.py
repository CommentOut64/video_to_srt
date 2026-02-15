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
from .plan_provider import (
    M1InternalSoftCutPlanProvider,
    SoftCutPlanProvider,
    resolve_soft_cut_plan_provider,
)
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
    "M1InternalSoftCutPlanProvider",
    "SoftCutPlanProvider",
    "EvidenceLevel",
    "resolve_soft_cut_plan_provider",
    "SoftCutDecisionEngine",
    "SpeakerChangeFact",
    "SpeakerChangeEvidence",
    "SpeakerChangeTag",
    "WindowDecisionContext",
]
