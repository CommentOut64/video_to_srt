"""Slow window packing 策略。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PackingDecision:
    should_emit: bool
    reason: str
    target_sec: float


class PackingPolicy:
    """根据窗口时长决定是否出窗。"""

    def __init__(
        self,
        *,
        first_window_target_sec: float,
        steady_window_target_sec: float,
        steady_two_party_target_sec: float,
        steady_fragmented_target_sec: float,
        hard_max_window_sec: float,
        ready_queue_low_watermark: int,
        ready_queue_target_depth: int,
        ready_queue_high_watermark: int,
    ) -> None:
        self._first_window_target_sec = float(first_window_target_sec)
        self._steady_window_target_sec = float(steady_window_target_sec)
        self._steady_two_party_target_sec = float(steady_two_party_target_sec)
        self._steady_fragmented_target_sec = float(steady_fragmented_target_sec)
        self._hard_max_window_sec = float(hard_max_window_sec)
        self._ready_queue_low_watermark = max(0, int(ready_queue_low_watermark))
        self._ready_queue_target_depth = max(self._ready_queue_low_watermark, int(ready_queue_target_depth))
        self._ready_queue_high_watermark = max(self._ready_queue_target_depth, int(ready_queue_high_watermark))

    def evaluate(
        self,
        *,
        window_count: int,
        duration_sec: float,
        dialogue_shape: str,
        has_candidate_cut: bool,
        ready_queue_depth: int,
    ) -> PackingDecision:
        target = self._resolve_target(window_count=window_count, dialogue_shape=dialogue_shape)
        if duration_sec >= self._hard_max_window_sec:
            return PackingDecision(should_emit=True, reason="hard_max", target_sec=target)
        if duration_sec >= target:
            return PackingDecision(should_emit=True, reason="target_duration", target_sec=target)
        if not has_candidate_cut or ready_queue_depth <= self._ready_queue_low_watermark:
            return PackingDecision(should_emit=False, reason="hold", target_sec=target)

        pressure_target = target
        reason = "hold"
        if ready_queue_depth >= self._ready_queue_high_watermark:
            pressure_target = max(self._resolve_floor_target(dialogue_shape), target - 2.0)
            reason = "backpressure"
        elif ready_queue_depth >= self._ready_queue_target_depth:
            pressure_target = max(self._resolve_floor_target(dialogue_shape), target - 1.0)
            reason = "candidate_cut_ready"

        if duration_sec >= pressure_target:
            return PackingDecision(should_emit=True, reason=reason, target_sec=pressure_target)
        return PackingDecision(should_emit=False, reason="hold", target_sec=target)

    def _resolve_target(self, *, window_count: int, dialogue_shape: str) -> float:
        if window_count == 0:
            return self._first_window_target_sec
        if dialogue_shape in {"dominant_plus_backchannel", "two_party_stable"}:
            return self._steady_two_party_target_sec
        if dialogue_shape == "ping_pong_fragmented":
            return self._steady_fragmented_target_sec
        return self._steady_window_target_sec

    def _resolve_floor_target(self, dialogue_shape: str) -> float:
        if dialogue_shape == "ping_pong_fragmented":
            return self._steady_fragmented_target_sec
        if dialogue_shape in {"dominant_plus_backchannel", "two_party_stable"}:
            return self._steady_two_party_target_sec
        return max(self._first_window_target_sec, self._steady_window_target_sec - 2.0)
