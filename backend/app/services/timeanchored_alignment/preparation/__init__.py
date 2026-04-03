"""AlignmentPreparation 公共入口。"""

from app.services.timeanchored_alignment.preparation.assembler import (
    AlignmentPreparationAssembler,
)
from app.services.timeanchored_alignment.preparation.contracts import (
    AlignmentPreparationCompat,
    CanonicalSequence,
    CanonicalToken,
    ExternalStableFacts,
    FastHook,
    PreparationBundle,
    PreparationProvenance,
    PreparationReport,
    PreparationScope,
    PreparedSlowText,
    PreparedTokenUnit,
    PronunciationEdge,
    PronunciationGraph,
    PronunciationHint,
    PronunciationStateNode,
    PronunciationTokenNode,
    PronunciationVariant,
    ProtectedUnit,
    PunctuationEvidence,
    SlowWindowTextPackage,
)
from app.services.timeanchored_alignment.preparation.token_provenance_binder import (
    TokenProvenanceBinder,
)

__all__ = [
    "AlignmentPreparationAssembler",
    "AlignmentPreparationCompat",
    "CanonicalSequence",
    "CanonicalToken",
    "ExternalStableFacts",
    "FastHook",
    "PreparationBundle",
    "PreparationProvenance",
    "PreparationReport",
    "PreparationScope",
    "PreparedSlowText",
    "PreparedTokenUnit",
    "PronunciationEdge",
    "PronunciationGraph",
    "PronunciationHint",
    "PronunciationStateNode",
    "PronunciationTokenNode",
    "PronunciationVariant",
    "ProtectedUnit",
    "PunctuationEvidence",
    "SlowWindowTextPackage",
    "TokenProvenanceBinder",
]
