"""Timeanchored Alignment 服务入口。"""

from app.services.timeanchored_alignment.contracts import (
    CONTRACT_VERSION,
    ALLOWED_ALIGNMENT_ROUTES,
    ALLOWED_ALIGNMENT_STATUS,
    AcousticCandidate,
    AlignmentItem,
    BoundaryEvidence,
    AlignmentMetrics,
    FinalAlignmentResult,
    LanguageRun,
    LanguageRunPackage,
    LayerReport,
    PhoneUnit,
    PronunciationPackage,
    TokenToPhoneSpan,
    TokenUnit,
    PhoneticPackage,
    PhoneticUnit,
    PipelineReport,
    ProtectedSpan,
    SlowInferenceWindow,
    SlowInferenceWindowEnvelope,
    TextTruthPackage,
    TextTruthQuality,
    TextTruthUnit,
    TimeBasePackage,
    TimeBaseQuality,
    TimeBaseUnit,
)
from app.services.timeanchored_alignment.hint_builder import HintBuilder
from app.services.timeanchored_alignment.language_run_frontend import LanguageRunFrontend
from app.services.timeanchored_alignment.pronunciation_frontend import PronunciationFrontend
from app.services.timeanchored_alignment.phonetic_aligner import (
    PhoneticAligner,
    PhoneticAlignerConfig,
    PhoneticRescueResult,
    PhoneticRescueTrace,
)
from app.services.timeanchored_alignment.phonetic_matchers import (
    PhoneticAlignmentPair,
    align_monotonic_sequences,
    project_token_phone_keys,
)
from app.services.timeanchored_alignment.slow_window import (
    DialogueShapeSnapshot,
    PromptSeed,
    ReadySlowWindow,
    SlowWindowBuilder,
    SlowWindowBuilderConfig,
    SlowWindowIngressAdapter,
    SlowWindowIngressUnit,
    WindowBatchHint,
    WindowChunkBinding,
    WindowCoverage,
    WindowLanguageProfile,
    WindowSourceUnit,
)
from app.services.timeanchored_alignment.subtitle_assembler import (
    FinalAlignedStream,
    SubtitleAssembler,
)
from app.services.timeanchored_alignment.sentence_segmenter import (
    SentenceSegmenter,
    SentenceSegmenterConfig,
)
from app.services.timeanchored_alignment.chunk_projector import (
    ChunkProjection,
    ChunkProjector,
    ChunkWindow,
)
from app.services.timeanchored_alignment.output_adapter import OutputAdapter
from app.services.timeanchored_alignment.stage_service import (
    TimeanchoredAlignmentStageService,
    TimeanchoredStageResult,
)
from app.services.timeanchored_alignment.text_aligner import TextAligner, TextAlignerThresholds
from app.services.timeanchored_alignment.text_alignment_scoring import (
    TextAlignmentScoreBreakdown,
    TextAlignmentScoreConfig,
    TextAlignmentScorer,
)
from app.services.timeanchored_alignment.hallucination_gate import (
    BOTH_SIDES_UNAVAILABLE_ERROR,
    GateDecision,
    GateEvaluation,
    HallucinationGate,
    HallucinationGateConfig,
)
from app.services.timeanchored_alignment.edge_selector import (
    EdgeSelector,
    EdgeSelectorConfig,
    FailedSpan,
)
from app.services.timeanchored_alignment.time_base_builder import TimeBaseBuilder
from app.services.timeanchored_alignment.window_time_base_assembler import (
    WindowTimeBaseAssembler,
    WindowTimeBasePackage,
)
from app.services.timeanchored_alignment.window_language_classifier import (
    MAIN_CHAIN_DECISION_DOMAINS,
    MIXED_DECISION_DOMAINS,
    WindowLanguageClassifier,
    WindowLanguageDecision,
)

__all__ = [
    "CONTRACT_VERSION",
    "ALLOWED_ALIGNMENT_ROUTES",
    "ALLOWED_ALIGNMENT_STATUS",
    "AcousticCandidate",
    "AlignmentItem",
    "BoundaryEvidence",
    "AlignmentMetrics",
    "FinalAlignmentResult",
    "LanguageRun",
    "LanguageRunPackage",
    "LayerReport",
    "PhoneUnit",
    "PronunciationPackage",
    "TokenToPhoneSpan",
    "TokenUnit",
    "PhoneticPackage",
    "PhoneticUnit",
    "PipelineReport",
    "ProtectedSpan",
    "HintBuilder",
    "LanguageRunFrontend",
    "PronunciationFrontend",
    "PhoneticAligner",
    "PhoneticAlignerConfig",
    "PhoneticRescueResult",
    "PhoneticRescueTrace",
    "PhoneticAlignmentPair",
    "align_monotonic_sequences",
    "project_token_phone_keys",
    "TextAligner",
    "TextAlignerThresholds",
    "BOTH_SIDES_UNAVAILABLE_ERROR",
    "GateDecision",
    "GateEvaluation",
    "HallucinationGate",
    "HallucinationGateConfig",
    "EdgeSelector",
    "EdgeSelectorConfig",
    "FailedSpan",
    "TextAlignmentScorer",
    "TextAlignmentScoreConfig",
    "TextAlignmentScoreBreakdown",
    "DialogueShapeSnapshot",
    "FinalAlignedStream",
    "PromptSeed",
    "ReadySlowWindow",
    "SubtitleAssembler",
    "SentenceSegmenter",
    "SentenceSegmenterConfig",
    "SlowWindowBuilder",
    "SlowWindowBuilderConfig",
    "SlowWindowIngressAdapter",
    "SlowWindowIngressUnit",
    "ChunkProjection",
    "ChunkProjector",
    "ChunkWindow",
    "OutputAdapter",
    "TimeanchoredAlignmentStageService",
    "TimeanchoredStageResult",
    "SlowInferenceWindow",
    "SlowInferenceWindowEnvelope",
    "TextTruthPackage",
    "TextTruthQuality",
    "TextTruthUnit",
    "TimeBaseBuilder",
    "TimeBasePackage",
    "TimeBaseQuality",
    "TimeBaseUnit",
    "WindowBatchHint",
    "WindowChunkBinding",
    "WindowCoverage",
    "WindowLanguageClassifier",
    "WindowLanguageDecision",
    "WindowLanguageProfile",
    "WindowSourceUnit",
    "WindowTimeBaseAssembler",
    "WindowTimeBasePackage",
    "MIXED_DECISION_DOMAINS",
    "MAIN_CHAIN_DECISION_DOMAINS",
]
