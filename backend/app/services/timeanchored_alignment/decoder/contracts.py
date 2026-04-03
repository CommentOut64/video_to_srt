"""Phase 3 decoder 内部契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.timeanchored_alignment.contracts import (
    AlignmentPath,
    AlignmentReport,
    BoundaryCandidate,
    LowConfidenceSpan,
)


@dataclass(frozen=True)
class LatticeCandidate:
    token_index: int
    slice_index: int
    score: float
    lexical_exact: bool
    pronunciation_match: bool
    top_candidate_match: bool
    language_consistent: bool
    blank_support: float
    synthetic: bool
    blocked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationLattice:
    token_ids: tuple[str, ...]
    slice_ids: tuple[str, ...]
    candidates_by_token: tuple[tuple[LatticeCandidate, ...], ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecoderStep:
    token_index: int
    slice_index: int
    score: float
    confidence: float
    lexical_exact: bool
    synthetic: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecoderPath:
    steps: tuple[DecoderStep, ...]
    total_score: float
    matched_token_count: int
    direct_token_count: int
    coverage_ratio: float
    direct_ratio: float


@dataclass(frozen=True)
class DecoderShadowResult:
    alignment_path: AlignmentPath | None
    boundary_candidates: tuple[BoundaryCandidate, ...]
    low_confidence_spans: tuple[LowConfidenceSpan, ...]
    alignment_report: AlignmentReport
    lattice: ObservationLattice | None = None
    decode_path: DecoderPath | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
