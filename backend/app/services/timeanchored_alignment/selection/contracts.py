from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from app.services.alignment.types import L2Output, L2Input, QualitySignals, TextTrack
from app.services.arbitration.arbiter import ArbitrationResult
from app.services.timeanchored_alignment.contracts import (
    SelectedTextTruth,
    SelectionDecision,
    SelectionReport,
)


@dataclass(frozen=True)
class FastObservationSummary:
    text_track: Optional[TextTrack]
    confidence: float = 0.0
    language_hint: str = "auto"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SlowTextCandidateSet:
    primary_track: Optional[TextTrack]
    confidence: float = 0.0
    language_hint: str = "auto"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SlowQualitySignals:
    quality_signals: QualitySignals
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_l2_quality_signals(self) -> QualitySignals:
        return self.quality_signals


@dataclass(frozen=True)
class WindowSelectionScope:
    job_id: str
    chunk_index: int
    window_id: str = ""
    source_chunk_ids: Tuple[str, ...] = field(default_factory=tuple)
    source_chunk_indices: Tuple[int, ...] = field(default_factory=tuple)
    edge_selection_mode: str = "auto"


@dataclass(frozen=True)
class SelectionInputs:
    fast_summary: FastObservationSummary
    slow_candidates: SlowTextCandidateSet
    selection_scope: WindowSelectionScope
    slow_quality_signals: SlowQualitySignals

    def to_l2_input(self) -> L2Input:
        return L2Input(
            sv_track=self.fast_summary.text_track,
            whisper_track=self.slow_candidates.primary_track,
            quality_signals=self.slow_quality_signals.to_l2_quality_signals(),
            edge_selection_mode=self.selection_scope.edge_selection_mode,
        )


@dataclass(frozen=True)
class SelectionOutcome:
    selected_text_truth: SelectedTextTruth
    selection_decision: SelectionDecision
    selection_report: SelectionReport
    selection_inputs: SelectionInputs
    arbitration_output: L2Output
    arbitration_result: ArbitrationResult
    chosen_track: Optional[TextTrack]
    quality_signals: QualitySignals

    @property
    def chosen_text_clean(self) -> str:
        if not self.chosen_track:
            return ""
        return str(
            self.chosen_track.text_clean
            or self.chosen_track.text_itn_raw
            or self.chosen_track.raw_text
            or ""
        )
