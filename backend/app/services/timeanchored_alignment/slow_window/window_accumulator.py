"""Slow window 累积状态。"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.timeanchored_alignment.slow_window.contracts import SlowWindowIngressUnit


@dataclass
class PendingSlowWindowState:
    ingress_units: list[SlowWindowIngressUnit] = field(default_factory=list)
    candidate_cut_reasons: list[str] = field(default_factory=list)
    last_arrived_at: float = 0.0

    def clear(self) -> None:
        self.ingress_units.clear()
        self.candidate_cut_reasons.clear()
        self.last_arrived_at = 0.0


class SlowWindowAccumulator:
    """管理当前 pending window。"""

    def append(self, pending: PendingSlowWindowState, ingress: SlowWindowIngressUnit) -> None:
        pending.ingress_units.append(ingress)
        pending.last_arrived_at = float(ingress.arrived_at)
