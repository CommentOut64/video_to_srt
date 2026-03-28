"""AlignmentPreparation 公共入口。"""

from app.services.timeanchored_alignment.preparation.assembler import (
    AlignmentPreparationAssembler,
)
from app.services.timeanchored_alignment.preparation.contracts import (
    AlignmentPreparationCompat,
    AlignmentPreparationPackage,
    FastHook,
    PreparedSlowText,
    PreparedTokenUnit,
    PronunciationHint,
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
    "AlignmentPreparationPackage",
    "FastHook",
    "PreparedSlowText",
    "PreparedTokenUnit",
    "PronunciationHint",
    "ProtectedUnit",
    "PunctuationEvidence",
    "SlowWindowTextPackage",
    "TokenProvenanceBinder",
]
