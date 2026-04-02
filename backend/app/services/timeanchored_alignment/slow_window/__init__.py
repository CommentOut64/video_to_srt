"""Slow window window-first 主链契约与服务。"""

from app.services.timeanchored_alignment.slow_window.builder import (
    SlowWindowBuilder,
    SlowWindowBuilderConfig,
)
from app.services.timeanchored_alignment.slow_window.contracts import (
    DialogueShapeSnapshot,
    PromptSeed,
    ReadySlowWindow,
    SlowWindow,
    SlowWindowIngressUnit,
    WindowBatchHint,
    WindowChunkBinding,
    WindowCoverage,
    WindowLanguageProfile,
    WindowSourceUnit,
)
from app.services.timeanchored_alignment.slow_window.ingress_adapter import SlowWindowIngressAdapter

__all__ = [
    "DialogueShapeSnapshot",
    "PromptSeed",
    "ReadySlowWindow",
    "SlowWindow",
    "SlowWindowBuilder",
    "SlowWindowBuilderConfig",
    "SlowWindowIngressAdapter",
    "SlowWindowIngressUnit",
    "WindowBatchHint",
    "WindowChunkBinding",
    "WindowCoverage",
    "WindowLanguageProfile",
    "WindowSourceUnit",
]
