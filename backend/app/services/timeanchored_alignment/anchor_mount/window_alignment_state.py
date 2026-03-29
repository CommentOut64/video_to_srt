"""AnchorMount 核心状态对象。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchorCandidate,
    AnchorMountInputView,
    AnchorMountItem,
    CrossChunkLock,
    FuseEvent,
    HookClaimRecord,
    LocalAlignmentBlock,
    PunctuationFact,
    PunctuationMappingDiagnostics,
    PunctuationPairState,
    PromotedAnchor,
    TemporalEnvelope,
)


@dataclass(frozen=True)
class WindowAlignmentState:
    """统一承载 seed -> chain -> rescue -> validity 的单向状态流。"""

    input_view: AnchorMountInputView
    raw_seed_candidates: tuple[AnchorCandidate, ...] = field(default_factory=tuple)
    ambiguity_clusters: tuple["AmbiguityCluster", ...] = field(default_factory=tuple)
    primary_candidates: tuple[AnchorCandidate, ...] = field(default_factory=tuple)
    local_blocks: tuple[LocalAlignmentBlock, ...] = field(default_factory=tuple)
    main_chain: Any | None = None
    anchor_islands: tuple[Any, ...] = field(default_factory=tuple)
    open_gaps: tuple[Any, ...] = field(default_factory=tuple)
    promoted_secondary_anchors: tuple[PromotedAnchor, ...] = field(default_factory=tuple)
    punctuation_facts: tuple[PunctuationFact, ...] = field(default_factory=tuple)
    punctuation_pair_states: tuple[PunctuationPairState, ...] = field(default_factory=tuple)
    punctuation_diagnostics: PunctuationMappingDiagnostics = field(
        default_factory=PunctuationMappingDiagnostics
    )
    items: tuple[AnchorMountItem, ...] = field(default_factory=tuple)
    envelopes: tuple[TemporalEnvelope, ...] = field(default_factory=tuple)
    hook_claims: tuple[HookClaimRecord, ...] = field(default_factory=tuple)
    cross_chunk_locks: tuple[CrossChunkLock, ...] = field(default_factory=tuple)
    boundary_evidences: tuple[Any, ...] = field(default_factory=tuple)
    rescue_round_index: int = 0
    fuse_events: tuple[FuseEvent, ...] = field(default_factory=tuple)
    timeline_validity: str = "repairable"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.rescue_round_index) < 0:
            raise ValueError("WindowAlignmentState.rescue_round_index 必须 >= 0")
        if not isinstance(self.diagnostics, dict):
            raise ValueError("WindowAlignmentState.diagnostics 必须是 dict")

    @classmethod
    def bootstrap(cls, input_view: AnchorMountInputView) -> "WindowAlignmentState":
        """从 ingress 构建最小初始状态。"""

        return cls(
            input_view=input_view,
            diagnostics={
                "window_id": str(input_view.window_id),
                "source_chunk_ids": tuple(str(item) for item in input_view.source_chunk_ids),
                "seed_candidate_count": 0,
                "primary_candidate_count": 0,
                "promoted_anchor_count": 0,
            },
        )

    def with_updates(self, **changes: Any) -> "WindowAlignmentState":
        """返回带指定更新的新状态，保持调用方显式原子提交。"""

        if "diagnostics" in changes and not isinstance(changes["diagnostics"], dict):
            raise ValueError("WindowAlignmentState.diagnostics 必须是 dict")
        return replace(self, **changes)
