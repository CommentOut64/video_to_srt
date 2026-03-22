"""慢流窗口组装器。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from app.services.punctuation.semantic_buffer import SemanticChunk
from app.services.timeanchored_alignment.contracts import (
    SlowInferenceWindow,
    SlowInferenceWindowEnvelope,
)
from app.services.timeanchored_alignment.hint_builder import HintBuilder
from app.services.timeanchored_alignment.window_language_classifier import (
    WindowLanguageClassifier,
)


@dataclass(frozen=True)
class SlowWindowAssemblerConfig:
    """组窗参数。"""

    first_window_target_sec: float = 5.0
    steady_window_target_sec: float = 12.0
    long_pause_cut_sec: float = 1.8
    tail_idle_sec: float = 1.0


@dataclass
class _PendingWindow:
    chunks: List[SemanticChunk] = field(default_factory=list)
    speaker_id: str = "unknown"
    turn_ids: List[str] = field(default_factory=list)
    source_contexts: List[str] = field(default_factory=list)
    last_arrived_at: float = 0.0
    last_audio_end: float = 0.0
    last_language: str = ""


class SlowWindowAssembler:
    """基于 flush 策略输出 SlowInferenceWindowEnvelope。"""

    def __init__(
        self,
        *,
        config: Optional[SlowWindowAssemblerConfig] = None,
        classifier: Optional[WindowLanguageClassifier] = None,
        hint_builder: Optional[HintBuilder] = None,
    ) -> None:
        self.config = config or SlowWindowAssemblerConfig()
        self.classifier = classifier or WindowLanguageClassifier()
        self.hint_builder = hint_builder or HintBuilder()
        self._pending = _PendingWindow()
        self._window_counter = 0

    def add_chunk(
        self,
        chunk: SemanticChunk,
        *,
        speaker_id: Optional[str],
        turn_id: Optional[str],
        slow_language_hint: Optional[str] = None,
        now: Optional[float] = None,
    ) -> list[SlowInferenceWindowEnvelope]:
        if chunk is None:
            return []

        now_ts = float(now) if now is not None else time.time()
        current_speaker = str(speaker_id or chunk.speaker_id or "unknown")
        current_language = self._normalize_language(getattr(chunk, "language", ""))
        outputs: list[SlowInferenceWindowEnvelope] = []

        if self._has_pending():
            pause_gap = max(0.0, float(chunk.audio_range[0]) - self._pending.last_audio_end)
            if current_speaker != self._pending.speaker_id:
                outputs.append(self._flush(reason="speaker_change", slow_language_hint=slow_language_hint))
            elif pause_gap >= self.config.long_pause_cut_sec:
                outputs.append(self._flush(reason="long_pause_cut", slow_language_hint=slow_language_hint))
            elif self._is_language_changed(current_language):
                outputs.append(self._flush(reason="language_change", slow_language_hint=slow_language_hint))

        self._append_chunk(
            chunk,
            speaker_id=current_speaker,
            turn_id=turn_id,
            arrived_at=now_ts,
            language=current_language,
        )
        if self._pending_duration() >= self._target_duration_sec():
            outputs.append(self._flush(reason="target_duration", slow_language_hint=slow_language_hint))
        return outputs

    def flush(
        self,
        reason: str = "eof_flush",
        *,
        slow_language_hint: Optional[str] = None,
    ) -> Optional[SlowInferenceWindowEnvelope]:
        if not self._has_pending():
            return None
        return self._flush(reason=reason, slow_language_hint=slow_language_hint)

    def flush_idle(
        self,
        *,
        now: Optional[float] = None,
        slow_language_hint: Optional[str] = None,
    ) -> Optional[SlowInferenceWindowEnvelope]:
        if not self._has_pending():
            return None
        now_ts = float(now) if now is not None else time.time()
        idle_sec = max(0.0, now_ts - self._pending.last_arrived_at)
        if idle_sec < self.config.tail_idle_sec:
            return None
        return self._flush(reason="idle_tail", slow_language_hint=slow_language_hint)

    @staticmethod
    def can_enter_main_chain(envelope: SlowInferenceWindowEnvelope) -> bool:
        metadata = envelope.window.metadata if envelope and envelope.window else {}
        return not bool(metadata.get("is_mixed_window", False))

    @staticmethod
    def to_ready_window_record(envelope: SlowInferenceWindowEnvelope) -> dict:
        """冻结 ReadyWindowQueue 口径所需最小元数据。"""
        window = envelope.window
        metadata = dict(window.metadata or {})
        return {
            "window_id": window.window_id,
            "start": float(window.start),
            "end": float(window.end),
            "primary_language": metadata.get("primary_language", window.language),
            "is_mixed_window": bool(metadata.get("is_mixed_window", False)),
            "flush_reason": metadata.get("flush_reason", ""),
            "ready_queue_key": metadata.get("ready_queue_key", ""),
            "ready_queue_priority": int(metadata.get("ready_queue_priority", 0)),
            "target_duration_sec": float(metadata.get("target_duration_sec", 0.0)),
        }

    def _flush(self, *, reason: str, slow_language_hint: Optional[str]) -> SlowInferenceWindowEnvelope:
        self._window_counter += 1
        window_mode = "bootstrap" if self._window_counter == 1 else "steady"
        target_duration = (
            self.config.first_window_target_sec
            if window_mode == "bootstrap"
            else self.config.steady_window_target_sec
        )
        start = min(chunk.audio_range[0] for chunk in self._pending.chunks)
        end = max(chunk.audio_range[1] for chunk in self._pending.chunks)
        decision = self.classifier.classify(
            self._pending.chunks,
            slow_language_hint=slow_language_hint,
        )
        hints = self.hint_builder.build_hints(
            self._pending.chunks,
            primary_language=decision.primary_language,
        )
        chunk_indices = self._collect_chunk_indices(self._pending.source_contexts)
        route = "fallback" if decision.is_mixed_window else "main_chain"
        metadata = {
            "flush_reason": str(reason),
            "speaker_id": self._pending.speaker_id,
            "turn_ids": tuple(self._pending.turn_ids),
            "source_chunk_ids": tuple(self._pending.source_contexts),
            "primary_language": decision.primary_language,
            "is_mixed_window": bool(decision.is_mixed_window),
            "decision_domains": tuple(decision.decision_domains),
            "route": route,
            "window_mode": window_mode,
            "ready_queue_key": f"{window_mode}:{decision.primary_language}",
            "ready_queue_priority": 0 if window_mode == "bootstrap" else 1,
            "target_duration_sec": float(target_duration),
            "audio_segments": tuple((float(chunk.audio_range[0]), float(chunk.audio_range[1])) for chunk in self._pending.chunks),
        }
        window = SlowInferenceWindow(
            window_id=f"sw-{self._window_counter:06d}",
            start=float(start),
            end=float(end),
            language=decision.primary_language,
            chunk_indices=chunk_indices,
            hints=hints,
            window_time_base=None,
            metadata=metadata,
        )
        envelope = SlowInferenceWindowEnvelope(
            window=window,
            aggregated_time_base=None,
            source_contexts=tuple(self._pending.source_contexts),
        )
        self._pending = _PendingWindow()
        return envelope

    def _append_chunk(
        self,
        chunk: SemanticChunk,
        *,
        speaker_id: str,
        turn_id: Optional[str],
        arrived_at: float,
        language: str,
    ) -> None:
        if not self._has_pending():
            self._pending.speaker_id = speaker_id
        self._pending.chunks.append(chunk)
        self._pending.last_arrived_at = arrived_at
        self._pending.last_audio_end = max(float(chunk.audio_range[1]), self._pending.last_audio_end)
        if language:
            self._pending.last_language = language
        source_ids = list(getattr(chunk, "source_chunks", []) or [])
        self._pending.source_contexts.extend(source_ids)
        if turn_id:
            self._pending.turn_ids.append(str(turn_id))

    def _has_pending(self) -> bool:
        return bool(self._pending.chunks)

    def _pending_duration(self) -> float:
        if not self._pending.chunks:
            return 0.0
        start = min(chunk.audio_range[0] for chunk in self._pending.chunks)
        end = max(chunk.audio_range[1] for chunk in self._pending.chunks)
        return max(float(end) - float(start), 0.0)

    def _target_duration_sec(self) -> float:
        if self._window_counter == 0:
            return float(self.config.first_window_target_sec)
        return float(self.config.steady_window_target_sec)

    def _is_language_changed(self, current_language: str) -> bool:
        if not current_language or not self._pending.last_language:
            return False
        return current_language != self._pending.last_language

    @staticmethod
    def _collect_chunk_indices(source_contexts: Sequence[str]) -> Tuple[int, ...]:
        indices: list[int] = []
        for item in source_contexts:
            text = str(item or "")
            if "-" not in text:
                continue
            candidate = text.rsplit("-", 1)[-1]
            if candidate.isdigit():
                indices.append(int(candidate))
        return tuple(indices)

    @staticmethod
    def _normalize_language(language: str) -> str:
        normalized = str(language or "").strip().lower()
        if normalized in {"zh", "ja", "en", "mixed"}:
            return normalized
        return ""
