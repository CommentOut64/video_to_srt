"""统一切分层与输出层协议契约（Phase 1）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import unicodedata

from app.services.timeanchored_alignment.contracts import BoundaryEvidence, ProtectedSpan


CONTRACT_VERSION = "1.0"
ALLOWED_TEXT_SOURCES = frozenset({"fast", "slow", "aligned"})
ALLOWED_PUNCT_SOURCES = frozenset({"fast", "slow", "aligned", "injected"})
ALLOWED_ATTACH_MODES = frozenset({"leading", "trailing", "between", "standalone"})
ALLOWED_PUNCT_CLASSES = frozenset({"sentence_end", "weak", "quote", "bracket", "other"})
ALLOWED_WIDTH_POLICIES = frozenset({"auto", "fullwidth", "halfwidth", "none"})
ALLOWED_INGRESS_UNIT_KINDS = frozenset({"chunk", "slow_window", "turn_group"})


def _ensure_probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} 必须在 [0, 1] 区间内，当前为 {value}")
    return value


def _ensure_time_span(start: float, end: float, *, field_name: str) -> None:
    if float(end) < float(start):
        raise ValueError(f"{field_name} 非法：end({end}) < start({start})")


def _is_punctuation(char: str) -> bool:
    return bool(char) and unicodedata.category(char).startswith("P")


def _ensure_core_only_text(text: str) -> None:
    if not text or not text.strip():
        raise ValueError("CoreToken.text_core 不能为空")
    if _is_punctuation(text[0]) or _is_punctuation(text[-1]):
        raise ValueError(f"CoreToken.text_core 必须为 core-only token，当前为 {text!r}")


@dataclass(frozen=True)
class CoreToken:
    token_id: str
    index: int
    text_core: str
    normalized_text: str
    start: float
    end: float
    confidence: Optional[float] = None
    source: str = "unknown"
    is_pseudo: bool = False
    speaker_id: Optional[str] = None
    turn_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.index) < 0:
            raise ValueError("CoreToken.index 必须 >= 0")
        _ensure_time_span(self.start, self.end, field_name="CoreToken")
        _ensure_core_only_text(self.text_core)
        if not self.normalized_text:
            raise ValueError("CoreToken.normalized_text 不能为空")
        if self.confidence is not None:
            _ensure_probability("CoreToken.confidence", self.confidence)


@dataclass(frozen=True)
class PunctuationFact:
    fact_id: str
    left_token_index: Optional[int]
    right_token_index: Optional[int]
    attach_mode: str
    raw_text: str
    normalized_text: str
    punct_class: str
    source: str
    priority: int = 0
    is_sentence_end: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.attach_mode not in ALLOWED_ATTACH_MODES:
            raise ValueError(f"PunctuationFact.attach_mode 不支持: {self.attach_mode}")
        if self.punct_class not in ALLOWED_PUNCT_CLASSES:
            raise ValueError(f"PunctuationFact.punct_class 不支持: {self.punct_class}")
        if self.source not in ALLOWED_PUNCT_SOURCES:
            raise ValueError(f"PunctuationFact.source 不支持: {self.source}")
        if self.left_token_index is not None and int(self.left_token_index) < 0:
            raise ValueError("PunctuationFact.left_token_index 必须 >= 0")
        if self.right_token_index is not None and int(self.right_token_index) < 0:
            raise ValueError("PunctuationFact.right_token_index 必须 >= 0")
        if not self.raw_text:
            raise ValueError("PunctuationFact.raw_text 不能为空")
        if not self.normalized_text:
            raise ValueError("PunctuationFact.normalized_text 不能为空")


@dataclass(frozen=True)
class CanonicalStreamDiagnostics:
    raw_fast_text: str = ""
    raw_slow_text: str = ""
    raw_aligned_text: str = ""
    raw_fast_punctuation: Tuple[PunctuationFact, ...] = field(default_factory=tuple)
    raw_slow_punctuation: Tuple[PunctuationFact, ...] = field(default_factory=tuple)
    raw_aligned_punctuation: Tuple[PunctuationFact, ...] = field(default_factory=tuple)
    dedup_log: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    raw_mount_trace: Dict[str, Any] = field(default_factory=dict)
    token_mapping_trace: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    ingress_context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SegmentationIngressContext:
    unit_kind: str
    unit_id: str
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    slow_window_id: Optional[str] = None
    turn_group_id: Optional[str] = None
    window_coverage: Optional[float] = None
    source_chunk_ids: Tuple[str, ...] = field(default_factory=tuple)
    projection_chunk_ids: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.unit_kind not in ALLOWED_INGRESS_UNIT_KINDS:
            raise ValueError(f"SegmentationIngressContext.unit_kind 不支持: {self.unit_kind}")
        if not self.unit_id:
            raise ValueError("SegmentationIngressContext.unit_id 不能为空")
        if self.chunk_index is not None and int(self.chunk_index) < 0:
            raise ValueError("SegmentationIngressContext.chunk_index 必须 >= 0")
        if self.window_coverage is not None:
            _ensure_probability("SegmentationIngressContext.window_coverage", self.window_coverage)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "unit_kind": self.unit_kind,
            "unit_id": self.unit_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "slow_window_id": self.slow_window_id,
            "turn_group_id": self.turn_group_id,
            "window_coverage": self.window_coverage,
            "source_chunk_ids": [str(item) for item in self.source_chunk_ids],
            "projection_chunk_ids": [str(item) for item in self.projection_chunk_ids],
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class CanonicalTextStream:
    stream_id: str
    chunk_ref: str
    language: str
    text_source: str
    tokens: Tuple[CoreToken, ...] = field(default_factory=tuple)
    punctuation_facts: Tuple[PunctuationFact, ...] = field(default_factory=tuple)
    candidate_boundaries: Tuple[BoundaryEvidence, ...] = field(default_factory=tuple)
    protected_spans: Tuple[ProtectedSpan, ...] = field(default_factory=tuple)
    cross_chunk_context: Dict[str, Any] = field(default_factory=dict)
    diagnostics: CanonicalStreamDiagnostics = field(default_factory=CanonicalStreamDiagnostics)
    metadata: Dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not self.stream_id:
            raise ValueError("CanonicalTextStream.stream_id 不能为空")
        if not self.chunk_ref:
            raise ValueError("CanonicalTextStream.chunk_ref 不能为空")
        if self.text_source not in ALLOWED_TEXT_SOURCES:
            raise ValueError(f"CanonicalTextStream.text_source 不支持: {self.text_source}")
        previous_index = -1
        for token in self.tokens:
            if token.index <= previous_index:
                raise ValueError("CanonicalTextStream.tokens 必须按 index 严格递增")
            previous_index = token.index


@dataclass(frozen=True)
class SegmentationOptions:
    min_tokens: int = 2
    max_tokens: int = 26
    min_duration: float = 0.3
    max_duration: float = 8.0
    soft_pause: float = 0.45
    long_pause: float = 0.9
    split_on_inner_punctuation: bool = True
    enable_cjk_weak_punct_split: bool = True
    enable_cjk_semantic_split: bool = True
    continuation_words: Tuple[str, ...] = field(default_factory=tuple)
    sentence_end_chars: Tuple[str, ...] = (".", "!", "?", "。", "！", "？")
    policy_snapshot_ref: Optional[str] = None

    def __post_init__(self) -> None:
        if int(self.min_tokens) <= 0:
            raise ValueError("SegmentationOptions.min_tokens 必须 > 0")
        if int(self.max_tokens) < int(self.min_tokens):
            raise ValueError("SegmentationOptions.max_tokens 必须 >= min_tokens")
        if float(self.min_duration) < 0.0:
            raise ValueError("SegmentationOptions.min_duration 必须 >= 0")
        if float(self.max_duration) < float(self.min_duration):
            raise ValueError("SegmentationOptions.max_duration 必须 >= min_duration")
        if float(self.soft_pause) < 0.0 or float(self.long_pause) < 0.0:
            raise ValueError("SegmentationOptions.pause 阈值必须 >= 0")


@dataclass(frozen=True)
class ConsumedBoundaryPunct:
    fact_id: str
    raw_text: str
    normalized_text: str
    punct_class: str
    source: str
    render_hint: str = ""

    def __post_init__(self) -> None:
        if self.punct_class not in ALLOWED_PUNCT_CLASSES:
            raise ValueError(f"ConsumedBoundaryPunct.punct_class 不支持: {self.punct_class}")
        if self.source not in ALLOWED_PUNCT_SOURCES:
            raise ValueError(f"ConsumedBoundaryPunct.source 不支持: {self.source}")
        if not self.raw_text:
            raise ValueError("ConsumedBoundaryPunct.raw_text 不能为空")
        if not self.normalized_text:
            raise ValueError("ConsumedBoundaryPunct.normalized_text 不能为空")


@dataclass(frozen=True)
class SegmentPlan:
    segment_id: str
    token_start: int
    token_end: int
    start: float
    end: float
    boundary_reason: str = ""
    boundary_score: float = 0.0
    hard_boundary: bool = False
    consumed_boundary_punct: Optional[ConsumedBoundaryPunct] = None
    trace: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.token_start) < 0:
            raise ValueError("SegmentPlan.token_start 必须 >= 0")
        if int(self.token_end) < int(self.token_start):
            raise ValueError("SegmentPlan.token_end 必须 >= token_start")
        _ensure_time_span(self.start, self.end, field_name="SegmentPlan")
        _ensure_probability("SegmentPlan.boundary_score", self.boundary_score)


@dataclass(frozen=True)
class SegmentationResult:
    segments: Tuple[SegmentPlan, ...]
    boundary_traces: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    segmentation_report: Dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class RenderPolicy:
    show_inner_punctuation: bool = True
    show_terminal_period: bool = False
    show_terminal_non_period_punct: bool = True
    language_punctuation_standardization: bool = True
    fullwidth_halfwidth_policy: str = "auto"

    def __post_init__(self) -> None:
        if self.fullwidth_halfwidth_policy not in ALLOWED_WIDTH_POLICIES:
            raise ValueError(
                "RenderPolicy.fullwidth_halfwidth_policy 不支持: "
                f"{self.fullwidth_halfwidth_policy}"
            )


@dataclass(frozen=True)
class RenderedSubtitle:
    segment_id: str
    start: float
    end: float
    text_display: str
    text_core_joined: str
    terminal_punct: Optional[str] = None
    speaker_id: Optional[str] = None
    turn_id: Optional[str] = None
    text_source: str = "unknown"
    trace: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_time_span(self.start, self.end, field_name="RenderedSubtitle")


@dataclass(frozen=True)
class RenderResult:
    subtitles: Tuple[RenderedSubtitle, ...]
    render_report: Dict[str, Any] = field(default_factory=dict)
    output_trace: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class SubtitleItem:
    segment_id: str
    chunk_id: str
    start: float
    end: float
    text: str
    status: str = "final"
    source: str = "unknown"
    speaker_id: Optional[str] = None
    turn_id: Optional[str] = None
    trace: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("SubtitleItem.segment_id 不能为空")
        if not self.chunk_id:
            raise ValueError("SubtitleItem.chunk_id 不能为空")
        if not self.text:
            raise ValueError("SubtitleItem.text 不能为空")
        _ensure_time_span(self.start, self.end, field_name="SubtitleItem")


@dataclass(frozen=True)
class SubtitleBatch:
    chunk_id: str
    items: Tuple[SubtitleItem, ...]
    chunk_index: Optional[int] = None
    render_report: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("SubtitleBatch.chunk_id 不能为空")
        if self.chunk_index is not None and int(self.chunk_index) < 0:
            raise ValueError("SubtitleBatch.chunk_index 必须 >= 0")


__all__ = [
    "CONTRACT_VERSION",
    "CoreToken",
    "PunctuationFact",
    "CanonicalStreamDiagnostics",
    "SegmentationIngressContext",
    "CanonicalTextStream",
    "SegmentationOptions",
    "ConsumedBoundaryPunct",
    "SegmentPlan",
    "SegmentationResult",
    "RenderPolicy",
    "RenderedSubtitle",
    "RenderResult",
    "SubtitleItem",
    "SubtitleBatch",
]
