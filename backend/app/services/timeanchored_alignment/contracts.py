"""Timeanchored Alignment 核心契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


CONTRACT_VERSION = "1.0"
ALLOWED_ALIGNMENT_ROUTES = frozenset({"text", "phonetic", "mixed", "fast", "slow", "error"})
ALLOWED_ALIGNMENT_STATUS = frozenset({"direct", "phonetic", "estimated", "interpolated", "failed"})


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
