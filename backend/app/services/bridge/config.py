"""
Bridge 配置。
V3.2.0+dev.20260201.04
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BridgeConfig:
    """Bridge 运行参数配置。"""

    min_batch_duration: float = 20.0
    max_batch_duration: float = 30.0
    min_batch_sentences: int = 6
    max_batch_sentences: int = 10
    starvation_timeout: float = 5.0
    overlap_duration: float = 1.5
    queue_maxsize: int = 50
    backpressure_timeout: float = 5.0
    is_enable_frame_drop: bool = False
    long_pause_threshold: float = 2.0
    is_enable_dynamic_thresholds: bool = True
    dynamic_sentence_step: int = 1
    dynamic_duration_step: float = 2.0
