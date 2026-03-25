"""AnchorMountAlignment 契约定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.timeanchored_alignment.preparation.contracts import (
    FastHook,
    PunctuationEvidence,
    PronunciationHint,
    SlowSlot,
    SlowWindowTextPackage,
)
from app.services.timeanchored_alignment.slow_window.contracts import WindowCoverage


def _ensure_probability(name: str, value: float) -> float:
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} 必须在 [0, 1] 区间内，当前为 {value}")
    return normalized


def _ensure_non_negative_int(name: str, value: int) -> int:
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} 必须 >= 0，当前为 {value}")
    return normalized


def _ensure_time_span(name: str, start: float, end: float) -> None:
    if start is None or end is None:
        raise ValueError(f"{name}.start/end 不能为空")
    if float(end) < float(start):
        raise ValueError(f"{name} 非法：end({end}) < start({start})")


@dataclass(frozen=True)
class AnchorMountInputView:
    window_id: str
    owner_chunk_id: str
    owner_chunk_index: int
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    language: str
    slots: tuple[SlowSlot, ...]
    window_text: SlowWindowTextPackage
    punctuation_evidences: tuple[PunctuationEvidence, ...]
    fast_hooks: tuple[FastHook, ...]
    pronunciation_hints: tuple[PronunciationHint, ...]
    policy_snapshot: Any | None
    coverage: WindowCoverage


@dataclass(frozen=True)
class AnchorCandidate:
    slot_indices: tuple[int, ...]
    hook_indices: tuple[int, ...]
    anchor_kind: str
    score: float
    is_hard: bool

    def __post_init__(self) -> None:
        if not self.slot_indices:
            raise ValueError("AnchorCandidate.slot_indices 不能为空")
        if not self.hook_indices:
            raise ValueError("AnchorCandidate.hook_indices 不能为空")
        _ensure_probability("AnchorCandidate.score", self.score)


@dataclass(frozen=True)
class LocalAlignmentBlock:
    block_id: str
    slot_indices: tuple[int, ...]
    hook_indices: tuple[int, ...]
    score: float
    block_kind: str
    anchor_kind: str

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("LocalAlignmentBlock.block_id 不能为空")
        _ensure_probability("LocalAlignmentBlock.score", self.score)


@dataclass(frozen=True)
class AnchorMountItem:
    slot_id: str
    slot_index: int
    text_core: str
    display_text: str
    normalized_text: str
    speaker_id: str | None
    turn_id: str | None
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    mount_status: str
    anchor_kind: str
    envelope_index: int
    source_hook_ids: tuple[str, ...]
    match_confidence: float
    cross_chunk_lock_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _ensure_non_negative_int("AnchorMountItem.slot_index", self.slot_index)
        _ensure_non_negative_int("AnchorMountItem.envelope_index", self.envelope_index)
        _ensure_probability("AnchorMountItem.match_confidence", self.match_confidence)


@dataclass(frozen=True)
class TemporalEnvelope:
    slot_id: str
    envelope_kind: str
    left_bound: float
    right_bound: float
    preferred_start: float | None
    preferred_end: float | None
    provisional_start: float
    provisional_end: float
    confidence: float
    source_hook_ids: tuple[str, ...]
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    cross_chunk_lock: bool

    def __post_init__(self) -> None:
        _ensure_time_span("TemporalEnvelope", self.left_bound, self.right_bound)
        _ensure_time_span(
            "TemporalEnvelope.provisional",
            self.provisional_start,
            self.provisional_end,
        )
        _ensure_probability("TemporalEnvelope.confidence", self.confidence)


@dataclass(frozen=True)
class HookClaimRecord:
    hook_id: str
    owner_window_id: str
    claim_level: str
    claim_reason: str
    overlap_ratio: float
    anchor_score: float
    finalized: bool

    def __post_init__(self) -> None:
        _ensure_probability("HookClaimRecord.overlap_ratio", self.overlap_ratio)
        _ensure_probability("HookClaimRecord.anchor_score", self.anchor_score)


@dataclass(frozen=True)
class CrossChunkLock:
    lock_id: str
    slot_ids: tuple[str, ...]
    hook_ids: tuple[str, ...]
    reason: str
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]


@dataclass(frozen=True)
class BoundaryHint:
    split_after_slot_id: str
    decision_time: float
    score: float
    reason: str
    hard_flag: bool
    blocked_by_lock: bool = False

    def __post_init__(self) -> None:
        if float(self.decision_time) < 0.0:
            raise ValueError("BoundaryHint.decision_time 必须 >= 0")
        _ensure_probability("BoundaryHint.score", self.score)


@dataclass(frozen=True)
class PunctuationFact:
    fact_id: str
    left_slot_index: int | None
    right_slot_index: int | None
    attach_mode: str
    normalized_text: str
    punct_class: str
    confidence: float
    source: str
    group_id: str | None
    boundary_weight: float
    render_default: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.left_slot_index is not None:
            _ensure_non_negative_int("PunctuationFact.left_slot_index", self.left_slot_index)
        if self.right_slot_index is not None:
            _ensure_non_negative_int("PunctuationFact.right_slot_index", self.right_slot_index)
        _ensure_probability("PunctuationFact.confidence", self.confidence)
        _ensure_probability("PunctuationFact.boundary_weight", self.boundary_weight)


@dataclass(frozen=True)
class PunctuationPairState:
    group_id: str
    pair_kind: str
    open_fact_id: str
    close_fact_id: str | None
    state: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PunctuationMappingDiagnostics:
    unmapped: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    dropped_with_reason: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AnchorMountResult:
    items: tuple[AnchorMountItem, ...]
    envelopes: tuple[TemporalEnvelope, ...]
    punctuation_facts: tuple[PunctuationFact, ...]
    punctuation_pair_states: tuple[PunctuationPairState, ...]
    hook_claims: tuple[HookClaimRecord, ...]
    cross_chunk_locks: tuple[CrossChunkLock, ...]
    boundary_hints: tuple[BoundaryHint, ...]
    metrics: dict[str, Any]
    should_fallback: bool


@dataclass(frozen=True)
class DecisionToken:
    token_id: str
    slot_index: int
    text_core: str
    display_text: str
    normalized_text: str
    start: float
    end: float
    left_bound: float
    right_bound: float
    speaker_id: str | None
    turn_id: str | None
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    source_hook_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_negative_int("DecisionToken.slot_index", self.slot_index)
        _ensure_time_span("DecisionToken", self.start, self.end)
        _ensure_time_span("DecisionToken.bounds", self.left_bound, self.right_bound)


@dataclass(frozen=True)
class DecisionIngressPackage:
    window_id: str
    owner_chunk_id: str
    owner_chunk_index: int
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    language: str
    policy_snapshot: Any | None
    tokens: tuple[DecisionToken, ...]
    punctuation_facts: tuple[PunctuationFact, ...]
    punctuation_pair_states: tuple[PunctuationPairState, ...]
    boundary_hints: tuple[BoundaryHint, ...]
    cross_chunk_locks: tuple[CrossChunkLock, ...]
    coverage: WindowCoverage
    quality_metrics: dict[str, Any]
    should_fallback: bool
