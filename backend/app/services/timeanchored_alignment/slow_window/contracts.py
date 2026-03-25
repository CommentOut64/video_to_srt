"""Slow window window-first 契约定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_ALLOWED_WINDOW_MODES = frozenset({"bootstrap", "steady", "drain"})
_ALLOWED_BINDING_ROLES = frozenset({"owner", "core", "left_guard", "right_guard"})
_ALLOWED_MIX_STATES = frozenset({"single_language", "dominant_mixed", "true_mixed"})


def _ensure_non_negative(name: str, value: float) -> float:
    normalized = float(value)
    if normalized < 0.0:
        raise ValueError(f"{name} 必须 >= 0，当前为 {value}")
    return normalized


def _ensure_probability(name: str, value: float) -> float:
    normalized = float(value)
    if normalized < 0.0 or normalized > 1.0:
        raise ValueError(f"{name} 必须位于 [0, 1]，当前为 {value}")
    return normalized


def _ensure_time_span(name: str, start: float, end: float) -> None:
    if float(end) < float(start):
        raise ValueError(f"{name} 非法：end({end}) < start({start})")


@dataclass(frozen=True)
class SlowWindowIngressUnit:
    semantic_chunk_id: str
    text: str
    sentences: tuple[Any, ...]
    punctuation_decision: Any | None
    audio_range: tuple[float, float]
    language: str
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    speaker_id: str
    turn_id: str | None
    word_timestamps: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    arrived_at: float = 0.0

    def __post_init__(self) -> None:
        _ensure_time_span("SlowWindowIngressUnit.audio_range", self.audio_range[0], self.audio_range[1])


@dataclass(frozen=True)
class WindowSourceUnit:
    unit_id: str
    semantic_chunk_id: str
    text: str
    audio_start: float
    audio_end: float
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    speaker_id: str
    turn_id: str | None
    language: str
    arrived_at: float

    def __post_init__(self) -> None:
        _ensure_time_span("WindowSourceUnit", self.audio_start, self.audio_end)


@dataclass(frozen=True)
class WindowChunkBinding:
    chunk_id: str
    chunk_index: int
    chunk_start: float
    chunk_end: float
    overlap_ratio: float
    role: str
    is_owner: bool

    def __post_init__(self) -> None:
        _ensure_time_span("WindowChunkBinding", self.chunk_start, self.chunk_end)
        _ensure_probability("WindowChunkBinding.overlap_ratio", self.overlap_ratio)
        if self.role not in _ALLOWED_BINDING_ROLES:
            raise ValueError(f"WindowChunkBinding.role 不支持: {self.role}")


@dataclass(frozen=True)
class WindowCoverage:
    core_segments: tuple[tuple[float, float], ...]
    left_guard_sec: float
    right_guard_sec: float
    chunk_bindings: tuple[WindowChunkBinding, ...]

    def __post_init__(self) -> None:
        _ensure_non_negative("WindowCoverage.left_guard_sec", self.left_guard_sec)
        _ensure_non_negative("WindowCoverage.right_guard_sec", self.right_guard_sec)
        for idx, segment in enumerate(self.core_segments):
            _ensure_time_span(f"WindowCoverage.core_segments[{idx}]", segment[0], segment[1])


@dataclass(frozen=True)
class DialogueShapeSnapshot:
    shape: str
    speaker_count: int
    dominant_speaker_id: str | None
    dominant_speaker_ratio: float
    speaker_switch_count: int
    speaker_switch_density: float
    turn_count: int
    avg_turn_duration_sec: float

    def __post_init__(self) -> None:
        _ensure_probability("DialogueShapeSnapshot.dominant_speaker_ratio", self.dominant_speaker_ratio)
        _ensure_non_negative("DialogueShapeSnapshot.speaker_switch_density", self.speaker_switch_density)
        _ensure_non_negative("DialogueShapeSnapshot.avg_turn_duration_sec", self.avg_turn_duration_sec)


@dataclass(frozen=True)
class WindowLanguageProfile:
    primary_language: str
    language_mix_state: str
    decision_domains: tuple[str, ...]
    should_bypass_whisper: bool

    def __post_init__(self) -> None:
        if self.language_mix_state not in _ALLOWED_MIX_STATES:
            raise ValueError(f"WindowLanguageProfile.language_mix_state 不支持: {self.language_mix_state}")


@dataclass(frozen=True)
class PromptSeed:
    text: str
    keywords: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class WindowBatchHint:
    duration_bucket: str
    token_estimate: int
    acoustic_density_hint: str
    queue_priority: int

    def __post_init__(self) -> None:
        if int(self.token_estimate) < 0:
            raise ValueError("WindowBatchHint.token_estimate 必须 >= 0")


@dataclass(frozen=True)
class ReadySlowWindow:
    window_id: str
    owner_chunk_id: str
    owner_chunk_index: int
    window_mode: str
    flush_reason: str
    audio_segments: tuple[tuple[float, float], ...]
    coverage: WindowCoverage
    source_semantic_chunk_ids: tuple[str, ...]
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    source_units: tuple[WindowSourceUnit, ...]
    dialogue_shape: DialogueShapeSnapshot
    language_profile: WindowLanguageProfile
    prompt_seed: PromptSeed
    batch_hint: WindowBatchHint
    created_at: float

    def __post_init__(self) -> None:
        if self.window_mode not in _ALLOWED_WINDOW_MODES:
            raise ValueError(f"ReadySlowWindow.window_mode 不支持: {self.window_mode}")
        for idx, segment in enumerate(self.audio_segments):
            _ensure_time_span(f"ReadySlowWindow.audio_segments[{idx}]", segment[0], segment[1])
        if len(self.source_chunk_ids) != len(self.source_chunk_indices):
            raise ValueError("ReadySlowWindow.source_chunk_ids/source_chunk_indices 长度必须一致")
        if not self.source_chunk_ids:
            raise ValueError("ReadySlowWindow.source_chunk_ids 不能为空")
        if self.owner_chunk_id not in self.source_chunk_ids:
            raise ValueError("ReadySlowWindow.owner_chunk_id 必须属于 source_chunk_ids")
        if self.owner_chunk_index not in self.source_chunk_indices:
            raise ValueError("ReadySlowWindow.owner_chunk_index 必须属于 source_chunk_indices")
