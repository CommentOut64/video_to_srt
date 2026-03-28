"""AlignmentPreparation window-first 契约定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.timeanchored_alignment.chunk_projector import ChunkWindow
from app.services.timeanchored_alignment.contracts import (
    CONTRACT_VERSION,
    LanguageRun,
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
class AlignmentPreparationPackage:
    window_id: str
    owner_chunk_id: str
    owner_chunk_index: int
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    slow_text: PreparedSlowText
    fast_hooks: tuple[FastHook, ...]
    coverage: WindowCoverage
    compat: AlignmentPreparationCompat
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _ensure_non_negative_int(
            "AlignmentPreparationPackage.owner_chunk_index",
            self.owner_chunk_index,
        )
        if len(self.source_chunk_ids) != len(self.source_chunk_indices):
            raise ValueError(
                "AlignmentPreparationPackage.source_chunk_ids/source_chunk_indices 长度必须一致"
            )
        if not self.source_chunk_ids:
            raise ValueError("AlignmentPreparationPackage.source_chunk_ids 不能为空")
        if self.owner_chunk_id not in self.source_chunk_ids:
            raise ValueError("AlignmentPreparationPackage.owner_chunk_id 必须属于 source_chunk_ids")
        if self.owner_chunk_index not in self.source_chunk_indices:
            raise ValueError(
                "AlignmentPreparationPackage.owner_chunk_index 必须属于 source_chunk_indices"
            )
