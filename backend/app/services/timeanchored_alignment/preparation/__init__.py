"""AlignmentPreparation 公共入口。"""

from app.services.timeanchored_alignment.preparation.assembler import (
    AlignmentPreparationAssembler,
)
from app.services.timeanchored_alignment.preparation.contracts import (
    AlignmentPreparationCompat,
    AlignmentPreparationPackage,
    FastHook,
    PreparedSlowText,
    PronunciationHint,
    ProtectedUnit,
    PunctuationEvidence,
    SlowSlot,
    SlowWindowTextPackage,
)

__all__ = [
    "AlignmentPreparationAssembler",
    "AlignmentPreparationCompat",
    "AlignmentPreparationPackage",
    "FastHook",
    "PreparedSlowText",
    "PronunciationHint",
    "ProtectedUnit",
    "PunctuationEvidence",
    "SlowSlot",
    "SlowWindowTextPackage",
]
