"""AnchorMountAlignment 契约定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.timeanchored_alignment.contracts import BoundaryEvidence
from app.services.timeanchored_alignment.preparation.contracts import (
    FastHook,
    PreparedTokenUnit,
    PunctuationEvidence,
    PronunciationHint,
    SlowWindowTextPackage,
)
from app.services.timeanchored_alignment.slow_window.contracts import WindowCoverage

_ALLOWED_TRUST_TIERS = {
    "unreviewed",
    "primary",
    "secondary",
    "boundary_only",
    "rejected",
}
_ALLOWED_GAP_STATES = {
    "unexamined",
    "resolved",
    "residual",
    "large_residual",
    "invalid",
}
_ALLOWED_TIMELINE_VALIDITY = {
    "valid",
    "repairable",
    "quarantined",
    "fatal",
}


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


def _ensure_choice(name: str, value: str, allowed: set[str]) -> str:
    normalized = str(value or "").strip()
    if normalized not in allowed:
        raise ValueError(
            f"{name} 必须属于 {sorted(allowed)}，当前为 {value}"
        )
    return normalized


@dataclass(frozen=True)
class AnchorMountInputView:
    window_id: str
    owner_chunk_id: str
    owner_chunk_index: int
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    language: str
    token_units: tuple[PreparedTokenUnit, ...]
    window_text: SlowWindowTextPackage
    punctuation_evidences: tuple[PunctuationEvidence, ...]
    fast_hooks: tuple[FastHook, ...]
    pronunciation_hints: tuple[PronunciationHint, ...]
    policy_snapshot: Any | None
    coverage: WindowCoverage


@dataclass(frozen=True)
class AnchorCandidate:
    candidate_id: str
    unit_indices: tuple[int, ...]
    hook_indices: tuple[int, ...]
    anchor_kind: str
    score: float
    is_hard: bool
    ambiguity_cluster_id: str | None = None
    trust_tier: str = "unreviewed"
    reject_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("AnchorCandidate.candidate_id 不能为空")
        if not self.unit_indices:
            raise ValueError("AnchorCandidate.unit_indices 不能为空")
        if not self.hook_indices:
            raise ValueError("AnchorCandidate.hook_indices 不能为空")
        _ensure_probability("AnchorCandidate.score", self.score)
        _ensure_choice("AnchorCandidate.trust_tier", self.trust_tier, _ALLOWED_TRUST_TIERS)


@dataclass(frozen=True)
class LocalAlignmentBlock:
    block_id: str
    unit_indices: tuple[int, ...]
    hook_indices: tuple[int, ...]
    score: float
    block_kind: str
    anchor_kind: str
    candidate_ids: tuple[str, ...] = field(default_factory=tuple)
    ambiguity_cluster_ids: tuple[str, ...] = field(default_factory=tuple)
    trust_tier: str = "unreviewed"

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("LocalAlignmentBlock.block_id 不能为空")
        _ensure_probability("LocalAlignmentBlock.score", self.score)
        _ensure_choice("LocalAlignmentBlock.trust_tier", self.trust_tier, _ALLOWED_TRUST_TIERS)


@dataclass(frozen=True)
class PromotedAnchor:
    candidate_id: str
    unit_indices: tuple[int, ...]
    hook_indices: tuple[int, ...]
    anchor_kind: str
    source_gap_id: str
    trust_score: float
    trust_tier: str
    reject_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("PromotedAnchor.candidate_id 不能为空")
        if not self.source_gap_id:
            raise ValueError("PromotedAnchor.source_gap_id 不能为空")
        if not self.unit_indices:
            raise ValueError("PromotedAnchor.unit_indices 不能为空")
        if not self.hook_indices:
            raise ValueError("PromotedAnchor.hook_indices 不能为空")
        _ensure_probability("PromotedAnchor.trust_score", self.trust_score)
        _ensure_choice("PromotedAnchor.trust_tier", self.trust_tier, _ALLOWED_TRUST_TIERS)


@dataclass(frozen=True)
class AnchorMountItem:
    unit_id: str
    unit_index: int
    token_text: str
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
    alignment_block_id: str | None = None
    trust_tier: str = "unreviewed"
    ambiguity_cluster_id: str | None = None

    def __post_init__(self) -> None:
        _ensure_non_negative_int("AnchorMountItem.unit_index", self.unit_index)
        _ensure_non_negative_int("AnchorMountItem.envelope_index", self.envelope_index)
        _ensure_probability("AnchorMountItem.match_confidence", self.match_confidence)
        _ensure_choice("AnchorMountItem.trust_tier", self.trust_tier, _ALLOWED_TRUST_TIERS)


@dataclass(frozen=True)
class TemporalEnvelope:
    unit_id: str
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
    gap_state: str = "unexamined"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_time_span("TemporalEnvelope", self.left_bound, self.right_bound)
        _ensure_time_span(
            "TemporalEnvelope.provisional",
            self.provisional_start,
            self.provisional_end,
        )
        _ensure_probability("TemporalEnvelope.confidence", self.confidence)
        _ensure_choice("TemporalEnvelope.gap_state", self.gap_state, _ALLOWED_GAP_STATES)


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
    unit_ids: tuple[str, ...]
    hook_ids: tuple[str, ...]
    reason: str
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]


@dataclass(frozen=True)
class FuseEvent:
    level: str
    reason: str
    gap_id: str | None
    round_index: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_non_negative_int("FuseEvent.round_index", self.round_index)


@dataclass(frozen=True)
class PunctuationFact:
    fact_id: str
    left_token_index: int | None
    right_token_index: int | None
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
        if self.left_token_index is not None:
            _ensure_non_negative_int("PunctuationFact.left_token_index", self.left_token_index)
        if self.right_token_index is not None:
            _ensure_non_negative_int("PunctuationFact.right_token_index", self.right_token_index)
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
    boundary_evidences: tuple[BoundaryEvidence, ...]
    metrics: dict[str, Any]
    should_fallback: bool = False
    timeline_validity: str = "valid"
    validity_reasons: tuple[str, ...] = field(default_factory=tuple)
    fuse_events: tuple[FuseEvent, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _ensure_choice(
            "AnchorMountResult.timeline_validity",
            self.timeline_validity,
            _ALLOWED_TIMELINE_VALIDITY,
        )


@dataclass(frozen=True)
class AnchoredTokenUnit:
    unit_id: str
    token_text: str
    normalized_text: str
    start: float
    end: float
    left_bound: float
    right_bound: float
    speaker_id: str | None
    turn_id: str | None
    mount_status: str
    anchor_kind: str
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    source_hook_ids: tuple[str, ...]
    match_confidence: float
    cross_chunk_lock_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _ensure_time_span("AnchoredTokenUnit", self.start, self.end)
        _ensure_time_span("AnchoredTokenUnit.bounds", self.left_bound, self.right_bound)
        _ensure_probability("AnchoredTokenUnit.match_confidence", self.match_confidence)


@dataclass(frozen=True)
class DecisionIngressPackage:
    window_id: str
    owner_chunk_id: str
    owner_chunk_index: int
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    language: str
    policy_snapshot: Any | None
    anchored_token_units: tuple[AnchoredTokenUnit, ...]
    punctuation_facts: tuple[PunctuationFact, ...]
    punctuation_pair_states: tuple[PunctuationPairState, ...]
    boundary_evidences: tuple[BoundaryEvidence, ...]
    cross_chunk_locks: tuple[CrossChunkLock, ...]
    coverage: WindowCoverage
    quality_metrics: dict[str, Any]
    timeline_validity: str = "valid"
    should_fallback: bool = False

    def __post_init__(self) -> None:
        _ensure_choice(
            "DecisionIngressPackage.timeline_validity",
            self.timeline_validity,
            _ALLOWED_TIMELINE_VALIDITY,
        )
