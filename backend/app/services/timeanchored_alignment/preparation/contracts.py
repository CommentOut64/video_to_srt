"""AlignmentPreparation window-first 契约定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.timeanchored_alignment.chunk_projector import ChunkWindow
from app.services.timeanchored_alignment.contracts import (
    AcousticObservationPack,
    CONTRACT_VERSION,
    LanguageRun,
    LayerSummary,
    LanguageRunPackage,
    PronunciationPackage,
    ProtectedSpan,
    TextTruthPackage,
    TimeBasePackage,
)
from app.services.timeanchored_alignment.slow_window.contracts import WindowCoverage


def _ensure_non_negative_int(name: str, value: int) -> int:
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} 必须 >= 0，当前为 {value}")
    return normalized


def _ensure_time_span(name: str, start: float, end: float) -> None:
    if float(end) < float(start):
        raise ValueError(f"{name} 非法：end({end}) < start({start})")


@dataclass(frozen=True)
class SlowWindowTextPackage:
    text: str
    display_text: str
    source_language: str = "auto"
    contract_version: str = CONTRACT_VERSION


@dataclass(frozen=True)
class PreparedTokenUnit:
    unit_id: str
    token_text: str
    normalized_text: str
    char_start: int
    char_end: int
    speaker_id: str | None
    turn_id: str | None
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    source_unit_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _ensure_non_negative_int("PreparedTokenUnit.char_start", self.char_start)
        if int(self.char_end) <= int(self.char_start):
            raise ValueError(
                "PreparedTokenUnit 非法："
                f"char_end({self.char_end}) 必须大于 char_start({self.char_start})"
            )
        if len(self.source_chunk_ids) != len(self.source_chunk_indices):
            raise ValueError(
                "PreparedTokenUnit.source_chunk_ids/source_chunk_indices 长度必须一致"
            )


@dataclass(frozen=True)
class PunctuationEvidence:
    mark: str
    source_char_index: int
    attach_side: str = "after"
    evidence_source: str = "slow_text"

    def __post_init__(self) -> None:
        _ensure_non_negative_int("PunctuationEvidence.source_char_index", self.source_char_index)
        if self.attach_side not in {"before", "after"}:
            raise ValueError(f"PunctuationEvidence.attach_side 不支持: {self.attach_side}")


@dataclass(frozen=True)
class ProtectedUnit:
    start: int
    end: int
    kind: str
    text: str = ""

    def __post_init__(self) -> None:
        _ensure_non_negative_int("ProtectedUnit.start", self.start)
        if int(self.end) <= int(self.start):
            raise ValueError(
                f"ProtectedUnit 非法：end({self.end}) 必须大于 start({self.start})"
            )


@dataclass(frozen=True)
class PronunciationHint:
    token_text: str
    reading_key: str
    language: str
    char_start: int
    char_end: int
    source: str = "pronunciation_frontend"

    def __post_init__(self) -> None:
        _ensure_non_negative_int("PronunciationHint.char_start", self.char_start)
        if int(self.char_end) <= int(self.char_start):
            raise ValueError(
                f"PronunciationHint 非法：char_end({self.char_end}) 必须大于 char_start({self.char_start})"
            )


@dataclass(frozen=True)
class FastHook:
    hook_text: str
    start: float
    end: float
    confidence: float
    source_chunk_id: str
    source_chunk_index: int
    token_type: str = "word"

    def __post_init__(self) -> None:
        _ensure_time_span("FastHook", self.start, self.end)
        _ensure_non_negative_int("FastHook.source_chunk_index", self.source_chunk_index)


@dataclass(frozen=True)
class PreparedSlowText:
    window_text: SlowWindowTextPackage
    token_units: tuple[PreparedTokenUnit, ...]
    punctuation_evidences: tuple[PunctuationEvidence, ...]
    protected_units: tuple[ProtectedUnit, ...]
    language_runs: tuple[LanguageRun, ...]
    pronunciation_hints: tuple[PronunciationHint, ...]


@dataclass(frozen=True)
class AlignmentPreparationCompat:
    time_base: TimeBasePackage
    text_truth: TextTruthPackage
    protected_spans: tuple[ProtectedSpan, ...]
    language_runs: LanguageRunPackage
    pronunciation: PronunciationPackage
    pronunciation_report: dict[str, Any]
    chunk_window: ChunkWindow


@dataclass(frozen=True)
class CanonicalToken:
    token_id: str
    text: str
    normalized_text: str
    char_start: int
    char_end: int
    language: str = "auto"
    source_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    source_chunk_indices: tuple[int, ...] = field(default_factory=tuple)
    is_protected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.token_id:
            raise ValueError("CanonicalToken.token_id 不能为空")
        if not self.text:
            raise ValueError("CanonicalToken.text 不能为空")
        if not self.normalized_text:
            raise ValueError("CanonicalToken.normalized_text 不能为空")
        _ensure_non_negative_int("CanonicalToken.char_start", self.char_start)
        if int(self.char_end) <= int(self.char_start):
            raise ValueError("CanonicalToken.char_end 必须大于 char_start")
        if len(self.source_chunk_ids) != len(self.source_chunk_indices):
            raise ValueError("CanonicalToken.source_chunk_ids/source_chunk_indices 长度必须一致")


@dataclass(frozen=True)
class CanonicalSequence:
    original_text: str
    normalized_text: str
    tokens: tuple[CanonicalToken, ...]
    protected_spans: tuple[ProtectedSpan, ...]
    language_runs: tuple[LanguageRun, ...]
    frontend_version: str
    language_hint: str = "auto"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.frontend_version:
            raise ValueError("CanonicalSequence.frontend_version 不能为空")


@dataclass(frozen=True)
class PronunciationVariant:
    reading_key: str
    weight: float = 1.0
    source: str = "pronunciation_frontend"

    def __post_init__(self) -> None:
        if not self.reading_key:
            raise ValueError("PronunciationVariant.reading_key 不能为空")
        if float(self.weight) <= 0.0:
            raise ValueError("PronunciationVariant.weight 必须大于 0")


@dataclass(frozen=True)
class PronunciationTokenNode:
    node_id: str
    token_index: int
    token_text: str
    language: str
    variants: tuple[PronunciationVariant, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("PronunciationTokenNode.node_id 不能为空")
        _ensure_non_negative_int("PronunciationTokenNode.token_index", self.token_index)
        if not self.token_text:
            raise ValueError("PronunciationTokenNode.token_text 不能为空")


@dataclass(frozen=True)
class PronunciationStateNode:
    node_id: str
    token_index: int
    reading_key: str
    language: str
    state_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("PronunciationStateNode.node_id 不能为空")
        _ensure_non_negative_int("PronunciationStateNode.token_index", self.token_index)
        _ensure_non_negative_int("PronunciationStateNode.state_index", self.state_index)
        if not self.reading_key:
            raise ValueError("PronunciationStateNode.reading_key 不能为空")


@dataclass(frozen=True)
class PronunciationEdge:
    from_node_id: str
    to_node_id: str
    edge_kind: str = "next"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.from_node_id or not self.to_node_id:
            raise ValueError("PronunciationEdge.from_node_id/to_node_id 不能为空")


@dataclass(frozen=True)
class PronunciationGraph:
    token_nodes: tuple[PronunciationTokenNode, ...]
    state_nodes: tuple[PronunciationStateNode, ...]
    edges: tuple[PronunciationEdge, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparationScope:
    window_id: str
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    absolute_time_range: tuple[float, float]

    def __post_init__(self) -> None:
        if not self.window_id:
            raise ValueError("PreparationScope.window_id 不能为空")
        if len(self.source_chunk_ids) != len(self.source_chunk_indices):
            raise ValueError("PreparationScope.source_chunk_ids/source_chunk_indices 长度必须一致")
        if not self.source_chunk_ids:
            raise ValueError("PreparationScope.source_chunk_ids 不能为空")
        _ensure_time_span(
            "PreparationScope.absolute_time_range",
            self.absolute_time_range[0],
            self.absolute_time_range[1],
        )


@dataclass(frozen=True)
class PreparationProvenance:
    text_source: str
    observation_source: str
    stable_fact_sources: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text_source:
            raise ValueError("PreparationProvenance.text_source 不能为空")
        if not self.observation_source:
            raise ValueError("PreparationProvenance.observation_source 不能为空")


@dataclass(frozen=True)
class ExternalStableFacts:
    punctuation_facts: tuple[Any, ...] = field(default_factory=tuple)
    speaker_turns: tuple[Any, ...] = field(default_factory=tuple)
    pause_facts: tuple[Any, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparationBundle:
    window_id: str
    owner_chunk_id: str
    owner_chunk_index: int
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    slow_text: PreparedSlowText
    fast_hooks: tuple[FastHook, ...]
    coverage: WindowCoverage
    compat: AlignmentPreparationCompat
    canonical_sequence: CanonicalSequence
    pronunciation_graph: PronunciationGraph
    acoustic_observation_pack: AcousticObservationPack
    scope: PreparationScope
    provenance: PreparationProvenance
    external_stable_facts: ExternalStableFacts
    report: "PreparationReport"
    debug_refs: dict[str, str] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _ensure_non_negative_int(
            "PreparationBundle.owner_chunk_index",
            self.owner_chunk_index,
        )
        if len(self.source_chunk_ids) != len(self.source_chunk_indices):
            raise ValueError(
                "PreparationBundle.source_chunk_ids/source_chunk_indices 长度必须一致"
            )
        if not self.source_chunk_ids:
            raise ValueError("PreparationBundle.source_chunk_ids 不能为空")
        if self.owner_chunk_id not in self.source_chunk_ids:
            raise ValueError("PreparationBundle.owner_chunk_id 必须属于 source_chunk_ids")
        if self.owner_chunk_index not in self.source_chunk_indices:
            raise ValueError(
                "PreparationBundle.owner_chunk_index 必须属于 source_chunk_indices"
            )
        if self.scope.window_id != self.window_id:
            raise ValueError("PreparationBundle.scope.window_id 必须与 window_id 一致")
        if self.scope.source_chunk_ids != self.source_chunk_ids:
            raise ValueError("PreparationBundle.scope.source_chunk_ids 必须与 preparation 一致")
        if self.scope.source_chunk_indices != self.source_chunk_indices:
            raise ValueError(
                "PreparationBundle.scope.source_chunk_indices 必须与 preparation 一致"
            )
        if self.report.summary.layer != "preparation":
            raise ValueError("PreparationBundle.report.summary.layer 必须为 preparation")


@dataclass(frozen=True)
class PreparationReport:
    summary: LayerSummary
    canonical_version: str = ""
    observation_capability: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
