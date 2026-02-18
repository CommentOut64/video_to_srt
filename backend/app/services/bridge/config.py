"""Bridge 配置（Phase 3 精简版）。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BridgeConfig:
    """Bridge 运行参数配置（仅保留空闲 flush 相关字段）。"""

    is_enable_dynamic_batch: bool = False
    dynamic_batch_max_wait_ms: int = 300
    queue_low_watermark: int = 2
    queue_high_watermark: int = 6
    force_flush_on_gpu_idle: bool = True
