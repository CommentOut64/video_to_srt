"""Slow window builder facade。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from app.services.timeanchored_alignment.slow_window.batch_hint_estimator import BatchHintEstimator
from app.services.timeanchored_alignment.slow_window.contracts import (
    ReadySlowWindow,
    SlowWindowIngressUnit,
    WindowSourceUnit,
)
from app.services.timeanchored_alignment.slow_window.coverage_planner import CoveragePlanner
from app.services.timeanchored_alignment.slow_window.cut_candidate_tracker import CutCandidateTracker
from app.services.timeanchored_alignment.slow_window.dialogue_shape_classifier import DialogueShapeClassifier
from app.services.timeanchored_alignment.slow_window.language_hint_resolver import LanguageHintResolver
from app.services.timeanchored_alignment.slow_window.packing_policy import PackingPolicy
from app.services.timeanchored_alignment.slow_window.prompt_seed_builder import PromptSeedBuilder
from app.services.timeanchored_alignment.slow_window.ready_window_queue import ReadyWindowQueue
from app.services.timeanchored_alignment.slow_window.window_accumulator import (
    PendingSlowWindowState,
    SlowWindowAccumulator,
)


@dataclass(frozen=True)
class SlowWindowBuilderConfig:
    first_window_target_sec: float = 5.0
    steady_window_target_sec: float = 12.0
    steady_two_party_target_sec: float = 9.0
    steady_fragmented_target_sec: float = 5.0
    hard_max_window_sec: float = 16.0
    long_pause_cut_sec: float = 1.8
    tail_idle_sec: float = 1.0
    ready_queue_low_watermark: int = 0
    ready_queue_target_depth: int = 1
    ready_queue_high_watermark: int = 2


class SlowWindowBuilder:
    """Window-first slow window builder。"""

    def __init__(
        self,
        *,
        config: Optional[SlowWindowBuilderConfig] = None,
    ) -> None:
        self.config = config or SlowWindowBuilderConfig()
        self._accumulator = SlowWindowAccumulator()
        self._dialogue_shape_classifier = DialogueShapeClassifier()
        self._cut_candidate_tracker = CutCandidateTracker(long_pause_sec=self.config.long_pause_cut_sec)
        self._packing_policy = PackingPolicy(
            first_window_target_sec=self.config.first_window_target_sec,
            steady_window_target_sec=self.config.steady_window_target_sec,
            steady_two_party_target_sec=self.config.steady_two_party_target_sec,
            steady_fragmented_target_sec=self.config.steady_fragmented_target_sec,
            hard_max_window_sec=self.config.hard_max_window_sec,
            ready_queue_low_watermark=self.config.ready_queue_low_watermark,
            ready_queue_target_depth=self.config.ready_queue_target_depth,
            ready_queue_high_watermark=self.config.ready_queue_high_watermark,
        )
        self._coverage_planner = CoveragePlanner()
        self._language_hint_resolver = LanguageHintResolver()
        self._prompt_seed_builder = PromptSeedBuilder()
        self._batch_hint_estimator = BatchHintEstimator()
        self._ready_window_queue = ReadyWindowQueue()
        self._pending = PendingSlowWindowState()
        self._window_counter = 0

    @property
    def pending_candidate_cut_reasons(self) -> tuple[str, ...]:
        return tuple(self._pending.candidate_cut_reasons)

    def add_chunk(
        self,
        ingress: SlowWindowIngressUnit,
        *,
        ready_queue_depth: Optional[int] = None,
    ) -> list[ReadySlowWindow]:
        previous = self._pending.ingress_units[-1] if self._pending.ingress_units else None
        for reason in self._cut_candidate_tracker.evaluate(previous, ingress):
            if reason not in self._pending.candidate_cut_reasons:
                self._pending.candidate_cut_reasons.append(reason)
        self._accumulator.append(self._pending, ingress)
        ready_snapshot = self._ready_window_queue.observe(ready_queue_depth)
        dialogue_shape = self._dialogue_shape_classifier.classify(tuple(self._pending.ingress_units))

        decision = self._packing_policy.evaluate(
            window_count=self._window_counter,
            duration_sec=self._pending_duration_sec(),
            dialogue_shape=dialogue_shape.shape,
            has_candidate_cut=bool(self._pending.candidate_cut_reasons),
            ready_queue_depth=ready_snapshot.depth,
        )
        if not decision.should_emit:
            return []
        return [self._emit_ready_window(reason=decision.reason, dialogue_shape=dialogue_shape)]

    def flush(self, reason: str = "eof_flush") -> ReadySlowWindow | None:
        if not self._pending.ingress_units:
            return None
        return self._emit_ready_window(reason=reason)

    def flush_idle(self, *, now: Optional[float] = None) -> ReadySlowWindow | None:
        if not self._pending.ingress_units:
            return None
        current = float(now if now is not None else time.time())
        if current - float(self._pending.last_arrived_at) < self.config.tail_idle_sec:
            return None
        return self._emit_ready_window(reason="idle_tail")

    def mark_ready_window_dequeued(self) -> None:
        self._ready_window_queue.pop()

    def _emit_ready_window(
        self,
        *,
        reason: str,
        dialogue_shape: Optional[object] = None,
    ) -> ReadySlowWindow:
        ingress_units = tuple(self._pending.ingress_units)
        self._window_counter += 1
        target = (
            self.config.first_window_target_sec
            if self._window_counter == 1
            else self.config.steady_window_target_sec
        )
        duration = self._pending_duration_sec()
        if reason in {"eof_flush", "idle_tail"} and duration < target:
            window_mode = "drain"
        else:
            window_mode = "bootstrap" if self._window_counter == 1 else "steady"

        source_units = tuple(self._to_source_unit(index, unit) for index, unit in enumerate(ingress_units))
        coverage = self._coverage_planner.build(source_units)
        owner_binding = next(binding for binding in coverage.chunk_bindings if binding.is_owner)
        if dialogue_shape is None:
            dialogue_shape = self._dialogue_shape_classifier.classify(ingress_units)
        language_profile = self._language_hint_resolver.resolve(source_units)
        prompt_seed = self._prompt_seed_builder.build(source_units)
        batch_hint = self._batch_hint_estimator.build(
            source_units=source_units,
            duration_sec=duration,
            window_mode=window_mode,
            ready_queue_depth=self._ready_window_queue.depth,
        )
        self._ready_window_queue.push()

        ready_window = ReadySlowWindow(
            window_id=f"sw-{self._window_counter:06d}",
            owner_chunk_id=owner_binding.chunk_id,
            owner_chunk_index=owner_binding.chunk_index,
            window_mode=window_mode,
            flush_reason=reason,
            audio_segments=tuple(unit.audio_range for unit in ingress_units),
            coverage=coverage,
            source_semantic_chunk_ids=tuple(unit.semantic_chunk_id for unit in ingress_units),
            source_chunk_ids=tuple(binding.chunk_id for binding in coverage.chunk_bindings),
            source_chunk_indices=tuple(binding.chunk_index for binding in coverage.chunk_bindings),
            source_units=source_units,
            dialogue_shape=dialogue_shape,
            language_profile=language_profile,
            prompt_seed=prompt_seed,
            batch_hint=batch_hint,
            created_at=time.time(),
        )
        self._pending.clear()
        return ready_window

    def _pending_duration_sec(self) -> float:
        if not self._pending.ingress_units:
            return 0.0
        start = min(unit.audio_range[0] for unit in self._pending.ingress_units)
        end = max(unit.audio_range[1] for unit in self._pending.ingress_units)
        return max(float(end) - float(start), 0.0)

    @staticmethod
    def _to_source_unit(position: int, unit: SlowWindowIngressUnit) -> WindowSourceUnit:
        return WindowSourceUnit(
            unit_id=f"{unit.semantic_chunk_id or 'semantic'}:{position}",
            semantic_chunk_id=unit.semantic_chunk_id,
            text=unit.text,
            audio_start=float(unit.audio_range[0]),
            audio_end=float(unit.audio_range[1]),
            source_chunk_ids=unit.source_chunk_ids,
            source_chunk_indices=unit.source_chunk_indices,
            speaker_id=unit.speaker_id,
            turn_id=unit.turn_id,
            language=unit.language,
            arrived_at=float(unit.arrived_at),
        )
