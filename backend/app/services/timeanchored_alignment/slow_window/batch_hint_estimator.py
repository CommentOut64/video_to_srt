"""批处理提示估计。"""

from __future__ import annotations

from app.services.timeanchored_alignment.slow_window.contracts import WindowBatchHint, WindowSourceUnit


class BatchHintEstimator:
    """估算调度所需的批处理提示。"""

    def build(
        self,
        *,
        source_units: tuple[WindowSourceUnit, ...],
        duration_sec: float,
        window_mode: str,
        ready_queue_depth: int,
    ) -> WindowBatchHint:
        if duration_sec < 6.0:
            duration_bucket = "short"
        elif duration_sec < 10.0:
            duration_bucket = "medium"
        else:
            duration_bucket = "long"
        token_estimate = sum(len(unit.text.strip()) for unit in source_units)
        base_priority = 0 if window_mode == "bootstrap" else 1
        pressure_bonus = min(max(0, int(ready_queue_depth)), 2)
        return WindowBatchHint(
            duration_bucket=duration_bucket,
            token_estimate=token_estimate,
            acoustic_density_hint="normal",
            queue_priority=base_priority + pressure_bonus,
        )
