"""Timeanchored Alignment 核心契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


CONTRACT_VERSION = "1.0"
ALLOWED_ALIGNMENT_ROUTES = frozenset({"text", "phonetic", "mixed", "fast", "slow", "error"})
ALLOWED_ALIGNMENT_STATUS = frozenset({"direct", "phonetic", "estimated", "interpolated", "failed"})
OBSERVATION_CAPABILITY_FRAME_POSTERIOR = "frame_posterior_capable"
OBSERVATION_CAPABILITY_TOKEN_TOPK = "token_topk_capable"
OBSERVATION_CAPABILITY_TIMESTAMP_ONLY = "timestamp_only_capable"
ALLOWED_OBSERVATION_CAPABILITIES = frozenset(
    {
        OBSERVATION_CAPABILITY_FRAME_POSTERIOR,
        OBSERVATION_CAPABILITY_TOKEN_TOPK,
        OBSERVATION_CAPABILITY_TIMESTAMP_ONLY,
    }
)
ACOUSTIC_OBSERVATION_MIN_REQUIRED_FIELDS = (
    "capability_level",
    "adapter_type",
    "source_chunk_ids",
    "source_chunk_indices",
    "absolute_time_range",
    "slices",
    "quality",
)
PTFA_LAYER_OBJECT_REGISTRY = {
    "selection": {
        "inputs": ("FastObservationSummary", "SlowTextCandidateSet", "WindowSelectionScope", "SlowQualitySignals"),
        "outputs": ("SelectedTextTruth", "SelectionDecision", "SelectionReport"),
    },
    "preparation": {
        "inputs": ("SelectedTextTruth", "ObservationAdapterOutput", "ExternalStableFacts", "PreparationScope"),
        "outputs": (
            "CanonicalSequence",
            "PronunciationGraph",
            "AcousticObservationPack",
            "PreparationBundle",
            "PreparationReport",
        ),
    },
    "alignment": {
        "inputs": ("PreparationBundle",),
        "outputs": ("AlignmentPath", "BoundaryCandidates", "LowConfidenceSpans", "AlignmentReport"),
    },
    "segmentation": {
        "inputs": ("AlignmentPath", "BoundaryCandidates", "SegmentationFacts", "SegmentationBudgetPolicy"),
        "outputs": ("AlignedSentence", "SegmentationReport"),
    },
    "output": {
        "inputs": ("AlignedSentence", "OutputProjectionScope", "OutputDeliveryConfig"),
        "outputs": ("SentenceRecord", "ChunkSentenceIndex", "SubtitleBatchCompat", "OutputReport"),
    },
}


def _ensure_probability(name: str, value: float) -> float:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} 必须在 [0, 1] 区间内，当前为 {value}")
    return float(value)


def _ensure_time_span(start: float, end: float, *, field_name: str) -> None:
    if float(end) < float(start):
        raise ValueError(f"{field_name} 非法：end({end}) < start({start})")


@dataclass(frozen=True)
class AcousticCandidate:
    text: str
    score: float
    token_id: Optional[int] = None

    def __post_init__(self) -> None:
        _ensure_probability("AcousticCandidate.score", self.score)


@dataclass(frozen=True)
class TimeBaseUnit:
    text: str
    start: float
    end: float
    confidence: float
    token_type: str = "raw"
    top_candidates: Tuple[AcousticCandidate, ...] = field(default_factory=tuple)
    source: str = "sensevoice"

    def __post_init__(self) -> None:
        _ensure_time_span(self.start, self.end, field_name="TimeBaseUnit")
        _ensure_probability("TimeBaseUnit.confidence", self.confidence)


@dataclass(frozen=True)
class TimeBaseQuality:
    blank_ratio: float
    avg_max_prob: float
    low_prob_ratio: float
    unit_count: int = 0
    word_count: int = 0

    def __post_init__(self) -> None:
        _ensure_probability("TimeBaseQuality.blank_ratio", self.blank_ratio)
        _ensure_probability("TimeBaseQuality.avg_max_prob", self.avg_max_prob)
        _ensure_probability("TimeBaseQuality.low_prob_ratio", self.low_prob_ratio)


@dataclass(frozen=True)
class ProtectedSpan:
    start: int
    end: int
    kind: str
    text: str = ""

    def __post_init__(self) -> None:
        if int(self.end) <= int(self.start):
            raise ValueError(f"ProtectedSpan 非法：end({self.end}) 必须大于 start({self.start})")


@dataclass(frozen=True)
class TimeBasePackage:
    raw_units: Tuple[TimeBaseUnit, ...]
    word_units: Tuple[TimeBaseUnit, ...]
    quality: TimeBaseQuality
    language: str
    frame_stride: float = 0.06
    source: str = "sensevoice"
    contract_version: str = CONTRACT_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextTruthUnit:
    text: str
    normalized_text: str
    confidence: float
    language: str
    start: Optional[float] = None
    end: Optional[float] = None
    source: str = "whisper"

    def __post_init__(self) -> None:
        _ensure_probability("TextTruthUnit.confidence", self.confidence)
        if self.start is not None and self.end is not None:
            _ensure_time_span(self.start, self.end, field_name="TextTruthUnit")


@dataclass(frozen=True)
class TextTruthQuality:
    hallucination_risk: float
    repetition_ratio: float
    length_ratio: float

    def __post_init__(self) -> None:
        _ensure_probability("TextTruthQuality.hallucination_risk", self.hallucination_risk)
        _ensure_probability("TextTruthQuality.repetition_ratio", self.repetition_ratio)
        if float(self.length_ratio) <= 0.0:
            raise ValueError("TextTruthQuality.length_ratio 必须大于 0")


@dataclass(frozen=True)
class TextTruthPackage:
    units: Tuple[TextTruthUnit, ...]
    quality: TextTruthQuality
    language: str
    raw_text: str = ""
    normalized_text: str = ""
    is_hallucination: bool = False
    quality_signals: Dict[str, float] = field(default_factory=dict)
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "whisper"
    contract_version: str = CONTRACT_VERSION
    protected_spans: Tuple[ProtectedSpan, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LanguageRun:
    run_text: str
    run_language: str
    char_start: int
    char_end: int
    is_protected: bool = False
    is_foreign_island: bool = False

    def __post_init__(self) -> None:
        if int(self.char_end) <= int(self.char_start):
            raise ValueError(f"LanguageRun 非法：char_end({self.char_end}) 必须大于 char_start({self.char_start})")


@dataclass(frozen=True)
class LanguageRunPackage:
    runs: Tuple[LanguageRun, ...]
    dominant_language: str
    window_kind: str
    foreign_run_ratio: float
    source_text: str = ""
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _ensure_probability("LanguageRunPackage.foreign_run_ratio", self.foreign_run_ratio)


@dataclass(frozen=True)
class TokenUnit:
    token_text: str
    language: str
    char_start: int
    char_end: int
    is_protected: bool = False
    is_foreign_island: bool = False

    def __post_init__(self) -> None:
        if int(self.char_end) <= int(self.char_start):
            raise ValueError(f"TokenUnit 非法：char_end({self.char_end}) 必须大于 char_start({self.char_start})")


@dataclass(frozen=True)
class PhoneUnit:
    phone_text: str
    language: str
    source: str = "homophone_tokenizer"


@dataclass(frozen=True)
class TokenToPhoneSpan:
    token_index: int
    phone_start: int
    phone_end: int

    def __post_init__(self) -> None:
        if int(self.phone_end) < int(self.phone_start):
            raise ValueError("TokenToPhoneSpan 非法：phone_end 必须 >= phone_start")


@dataclass(frozen=True)
class PronunciationPackage:
    token_units: Tuple[TokenUnit, ...]
    phone_units: Tuple[PhoneUnit, ...]
    token_to_phone_spans: Tuple[TokenToPhoneSpan, ...]
    frontend_source: str
    dependency_mode: Dict[str, str]
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class PhoneticUnit:
    token: str
    reading_key: str
    language: str
    confidence: float
    source: str = "phonetic_frontend"

    def __post_init__(self) -> None:
        _ensure_probability("PhoneticUnit.confidence", self.confidence)


@dataclass(frozen=True)
class PhoneticPackage:
    units: Tuple[PhoneticUnit, ...]
    language: str
    source: str = "phonetic_frontend"
    coverage: float = 0.0
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _ensure_probability("PhoneticPackage.coverage", self.coverage)


@dataclass(frozen=True)
class AlignmentItem:
    text: str
    start: float
    end: float
    status: str
    source: str
    confidence: Optional[float] = None
    reason: str = ""

    def __post_init__(self) -> None:
        _ensure_time_span(self.start, self.end, field_name="AlignmentItem")
        if self.status not in ALLOWED_ALIGNMENT_STATUS:
            raise ValueError(f"AlignmentItem.status 不支持: {self.status}")
        if self.confidence is not None:
            _ensure_probability("AlignmentItem.confidence", self.confidence)


@dataclass(frozen=True)
class BoundaryEvidence:
    split_idx: int
    event_time: float
    left_end: float
    right_start: float
    reason: str
    score: float
    hard_flag: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.split_idx) < 0:
            raise ValueError("BoundaryEvidence.split_idx 必须 >= 0")
        _ensure_probability("BoundaryEvidence.score", self.score)
        if float(self.event_time) < 0.0:
            raise ValueError("BoundaryEvidence.event_time 必须 >= 0")
        if float(self.left_end) < 0.0 or float(self.right_start) < 0.0:
            raise ValueError("BoundaryEvidence.left_end/right_start 必须 >= 0")


@dataclass(frozen=True)
class AlignmentMetrics:
    coverage: float = 0.0
    duration_ratio: float = 1.0
    failed_count: int = 0
    route_confidence: float = 0.0

    def __post_init__(self) -> None:
        _ensure_probability("AlignmentMetrics.coverage", self.coverage)
        _ensure_probability("AlignmentMetrics.route_confidence", self.route_confidence)
        if float(self.duration_ratio) <= 0.0:
            raise ValueError("AlignmentMetrics.duration_ratio 必须大于 0")


@dataclass(frozen=True)
class FinalAlignmentResult:
    items: Tuple[AlignmentItem, ...]
    route: str
    metrics: AlignmentMetrics
    contract_version: str = CONTRACT_VERSION
    error_code: Optional[str] = None

    def __post_init__(self) -> None:
        if self.route not in ALLOWED_ALIGNMENT_ROUTES:
            raise ValueError(f"FinalAlignmentResult.route 不支持: {self.route}")


@dataclass(frozen=True)
class SlowInferenceWindow:
    window_id: str
    start: float
    end: float
    language: str = "auto"
    chunk_indices: Tuple[int, ...] = field(default_factory=tuple)
    hints: Tuple[str, ...] = field(default_factory=tuple)
    window_time_base: Optional[TimeBasePackage] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_time_span(self.start, self.end, field_name="SlowInferenceWindow")


@dataclass(frozen=True)
class SlowInferenceWindowEnvelope:
    window: SlowInferenceWindow
    aggregated_time_base: Optional[TimeBasePackage] = None
    source_contexts: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LayerReport:
    name: str
    status: str = "ok"
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PipelineReport:
    contract_version: str = CONTRACT_VERSION
    time_base_report: Optional[LayerReport] = None
    text_truth_report: Optional[LayerReport] = None
    phonetic_report: Optional[LayerReport] = None
    alignment_report: Optional[LayerReport] = None
    segmentation_report: Optional[LayerReport] = None
    output_report: Optional[LayerReport] = None
    warnings: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PhysicalChunk:
    chunk_id: str
    chunk_index: int
    start: float
    end: float
    language_hint: str = "auto"
    observation_ref: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("PhysicalChunk.chunk_id 不能为空")
        if int(self.chunk_index) < 0:
            raise ValueError("PhysicalChunk.chunk_index 必须 >= 0")
        _ensure_time_span(self.start, self.end, field_name="PhysicalChunk")


@dataclass(frozen=True)
class SelectedTextTruth:
    text: str
    text_source: str
    language_hint: str
    source_chunk_ids: Tuple[str, ...]
    quality: Dict[str, float] = field(default_factory=dict)
    rejection_reasons: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("SelectedTextTruth.text 不能为空")
        if self.text_source not in {"fast", "slow", "mixed"}:
            raise ValueError(f"SelectedTextTruth.text_source 不支持: {self.text_source}")
        if not self.source_chunk_ids:
            raise ValueError("SelectedTextTruth.source_chunk_ids 不能为空")

    @property
    def normalized_text(self) -> str:
        return self.text

    @property
    def raw_text(self) -> str:
        return str(self.metadata.get("raw_text") or self.text)

    @property
    def source(self) -> str:
        return self.text_source

    @property
    def language(self) -> str:
        return self.language_hint


@dataclass(frozen=True)
class SelectionDecision:
    decision: str
    accepted_text_source: str
    reason_codes: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision not in {
            "accept_slow",
            "accept_fast",
            "force_fast",
            "force_slow",
            "mixed",
        }:
            raise ValueError(f"SelectionDecision.decision 不支持: {self.decision}")
        if self.accepted_text_source not in {"fast", "slow", "mixed"}:
            raise ValueError(
                "SelectionDecision.accepted_text_source 不支持: "
                f"{self.accepted_text_source}"
            )

    @property
    def chosen_source(self) -> str:
        return self.accepted_text_source

    @property
    def reason_code(self) -> str:
        if self.reason_codes:
            return str(self.reason_codes[0] or "")
        return ""


@dataclass(frozen=True)
class SelectionReport:
    chosen_source: str
    primary_reason_code: str
    decision: str
    summary: LayerSummary
    reason_codes: Tuple[str, ...] = field(default_factory=tuple)
    warnings: Tuple[LayerWarning, ...] = field(default_factory=tuple)
    errors: Tuple[LayerError, ...] = field(default_factory=tuple)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.chosen_source not in {"fast", "slow", "mixed"}:
            raise ValueError(f"SelectionReport.chosen_source 不支持: {self.chosen_source}")
        if not self.primary_reason_code:
            raise ValueError("SelectionReport.primary_reason_code 不能为空")
        if self.summary.layer != "selection":
            raise ValueError("SelectionReport.summary.layer 必须为 selection")


@dataclass(frozen=True)
class LayerWarning:
    code: str
    message: str
    layer: str
    job_id: str | None = None
    window_id: str | None = None
    chunk_id: str | None = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("LayerWarning.code 不能为空")
        if not self.message:
            raise ValueError("LayerWarning.message 不能为空")
        if not self.layer:
            raise ValueError("LayerWarning.layer 不能为空")


@dataclass(frozen=True)
class LayerError:
    code: str
    message: str
    layer: str
    job_id: str | None = None
    window_id: str | None = None
    chunk_id: str | None = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("LayerError.code 不能为空")
        if not self.message:
            raise ValueError("LayerError.message 不能为空")
        if not self.layer:
            raise ValueError("LayerError.layer 不能为空")


@dataclass(frozen=True)
class LayerSummary:
    layer: str
    status: str = "ok"
    counters: Dict[str, Any] = field(default_factory=dict)
    warnings: Tuple[LayerWarning, ...] = field(default_factory=tuple)
    errors: Tuple[LayerError, ...] = field(default_factory=tuple)
    debug_enabled: bool = False
    artifact_refs: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.layer not in PTFA_LAYER_OBJECT_REGISTRY:
            raise ValueError(f"LayerSummary.layer 不支持: {self.layer}")


@dataclass(frozen=True)
class PostprocessDebugConfig:
    enabled: bool = False
    level: str = "summary"
    root_dir: str = "debug/postprocess"

    def __post_init__(self) -> None:
        if self.level not in {"summary", "full"}:
            raise ValueError(f"PostprocessDebugConfig.level 不支持: {self.level}")


@dataclass(frozen=True)
class AcousticObservationTokenCandidate:
    token: str
    score: float
    token_id: int | None = None

    def __post_init__(self) -> None:
        _ensure_probability("AcousticObservationTokenCandidate.score", self.score)


@dataclass(frozen=True)
class AcousticObservationSlice:
    slice_id: str
    start: float
    end: float
    primary_token: str = ""
    top_candidates: Tuple[AcousticObservationTokenCandidate, ...] = field(default_factory=tuple)
    blank_score: float | None = None
    confidence: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.slice_id:
            raise ValueError("AcousticObservationSlice.slice_id 不能为空")
        _ensure_time_span(self.start, self.end, field_name="AcousticObservationSlice")
        if self.blank_score is not None:
            _ensure_probability("AcousticObservationSlice.blank_score", self.blank_score)
        if self.confidence is not None:
            _ensure_probability("AcousticObservationSlice.confidence", self.confidence)


@dataclass(frozen=True)
class AcousticObservationQuality:
    slice_count: int
    blank_coverage: float = 0.0
    topk_coverage: float = 0.0
    timestamp_coverage: float = 1.0
    capability_degraded: bool = False
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if int(self.slice_count) < 0:
            raise ValueError("AcousticObservationQuality.slice_count 必须 >= 0")
        _ensure_probability("AcousticObservationQuality.blank_coverage", self.blank_coverage)
        _ensure_probability("AcousticObservationQuality.topk_coverage", self.topk_coverage)
        _ensure_probability(
            "AcousticObservationQuality.timestamp_coverage",
            self.timestamp_coverage,
        )


@dataclass(frozen=True)
class AcousticObservationPack:
    capability_level: str
    adapter_type: str
    source_chunk_ids: Tuple[str, ...]
    source_chunk_indices: Tuple[int, ...]
    absolute_time_range: Tuple[float, float]
    slices: Tuple[AcousticObservationSlice, ...]
    quality: AcousticObservationQuality
    blank_track: Tuple[float, ...] = field(default_factory=tuple)
    topk_limit: int | None = None
    control_meta: Dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.capability_level not in ALLOWED_OBSERVATION_CAPABILITIES:
            raise ValueError(
                "AcousticObservationPack.capability_level 不支持: "
                f"{self.capability_level}"
            )
        if not self.adapter_type:
            raise ValueError("AcousticObservationPack.adapter_type 不能为空")
        if len(self.source_chunk_ids) != len(self.source_chunk_indices):
            raise ValueError(
                "AcousticObservationPack.source_chunk_ids/source_chunk_indices 长度必须一致"
            )
        if not self.source_chunk_ids:
            raise ValueError("AcousticObservationPack.source_chunk_ids 不能为空")
        _ensure_time_span(
            self.absolute_time_range[0],
            self.absolute_time_range[1],
            field_name="AcousticObservationPack.absolute_time_range",
        )
        if self.topk_limit is not None and int(self.topk_limit) <= 0:
            raise ValueError("AcousticObservationPack.topk_limit 必须 > 0")


@dataclass(frozen=True)
class BoundaryCandidate:
    split_token_index: int
    event_time: float
    reason: str
    score: float
    hard_boundary: bool = False
    source_chunk_ids: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.split_token_index) < 0:
            raise ValueError("BoundaryCandidate.split_token_index 必须 >= 0")
        if float(self.event_time) < 0.0:
            raise ValueError("BoundaryCandidate.event_time 必须 >= 0")
        _ensure_probability("BoundaryCandidate.score", self.score)


@dataclass(frozen=True)
class AlignedToken:
    token_id: str
    text: str
    start: float
    end: float
    source_chunk_ids: Tuple[str, ...]
    confidence: float = 0.0
    trace: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.token_id:
            raise ValueError("AlignedToken.token_id 不能为空")
        if not self.text:
            raise ValueError("AlignedToken.text 不能为空")
        if not self.source_chunk_ids:
            raise ValueError("AlignedToken.source_chunk_ids 不能为空")
        _ensure_time_span(self.start, self.end, field_name="AlignedToken")
        _ensure_probability("AlignedToken.confidence", self.confidence)


@dataclass(frozen=True)
class LowConfidenceSpan:
    start_token_index: int
    end_token_index: int
    reason: str
    score: float

    def __post_init__(self) -> None:
        if int(self.start_token_index) < 0:
            raise ValueError("LowConfidenceSpan.start_token_index 必须 >= 0")
        if int(self.end_token_index) < int(self.start_token_index):
            raise ValueError("LowConfidenceSpan.end_token_index 必须 >= start_token_index")
        _ensure_probability("LowConfidenceSpan.score", self.score)


@dataclass(frozen=True)
class AlignmentPath:
    path_id: str
    aligned_tokens: Tuple[AlignedToken, ...]
    route_confidence: float
    low_confidence_spans: Tuple[LowConfidenceSpan, ...]
    summary: LayerSummary
    source_chunk_ids: Tuple[str, ...]
    boundary_candidates: Tuple[BoundaryCandidate, ...] = field(default_factory=tuple)
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not self.path_id:
            raise ValueError("AlignmentPath.path_id 不能为空")
        if not self.source_chunk_ids:
            raise ValueError("AlignmentPath.source_chunk_ids 不能为空")
        _ensure_probability("AlignmentPath.route_confidence", self.route_confidence)


@dataclass(frozen=True)
class AlignmentReport:
    summary: LayerSummary
    route: str
    failure_semantic: str = "none"
    warnings: Tuple[LayerWarning, ...] = field(default_factory=tuple)
    errors: Tuple[LayerError, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.route not in {"alignment_path", "selection_reject_slow", "alignment_no_path", "alignment_low_confidence", "true_mixed_unresolved"}:
            raise ValueError(f"AlignmentReport.route 不支持: {self.route}")
