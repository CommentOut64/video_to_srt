"""
模型常驻与显存驱逐策略

负责根据模型使用频率、体积、任务队列状态等因素计算驱逐顺序。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class QueueSnapshot:
    """任务队列快照（用于显存策略判断）。"""

    running_job_id: Optional[str]
    queued_job_ids: List[str]
    pending_jobs: int
    is_idle: bool
    idle_seconds: float


@dataclass
class ModelResidencyEntry:
    """模型缓存条目快照。"""

    plan_key: str
    model_id: str
    vram_mb: int
    loaded_at: float
    last_used_at: float
    use_count: int
    evict_priority: int
    is_keep_resident: bool
    is_force_resident: bool

    @property
    def idle_seconds(self) -> float:
        """计算模型空闲时间（秒）。"""
        return max(0.0, time.time() - self.last_used_at)


@dataclass
class ResidencyPolicyConfig:
    """显存策略配置。"""

    idle_unload_after_sec: float = 300.0
    min_models_when_idle: int = 0
    recency_window_sec: float = 1800.0
    frequency_weight: float = 0.4
    recency_weight: float = 0.4
    size_weight: float = 0.2
    priority_weight: float = 0.2
    hot_use_count_threshold: int = 3
    hot_recency_window_sec: float = 600.0
    hot_bonus: float = 0.3
    dynamic_vram_ratio: float = 0.9
    dynamic_vram_safety_mb: int = 200


@dataclass
class EvictionPlan:
    """驱逐计划输出。"""

    keys: List[str]
    reason: str


class ModelResidencyPolicy:
    """
    模型常驻策略引擎。

    设计模式：策略模式，用于将显存驱逐规则与 ModelManagerV2 解耦。
    """

    def __init__(self, config: Optional[ResidencyPolicyConfig] = None) -> None:
        self._config = config or ResidencyPolicyConfig()

    @property
    def config(self) -> ResidencyPolicyConfig:
        """获取策略配置。"""
        return self._config

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelResidencyPolicy":
        """从 YAML 加载策略配置。"""
        if not path.exists():
            return cls()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("加载显存策略配置失败，回退默认配置: %s", exc)
            return cls()

        policy = data.get("policy", {}) if isinstance(data, dict) else {}
        config = ResidencyPolicyConfig(
            idle_unload_after_sec=float(policy.get("idle_unload_after_sec", 300.0)),
            min_models_when_idle=int(policy.get("min_models_when_idle", 0)),
            recency_window_sec=float(policy.get("recency_window_sec", 1800.0)),
            frequency_weight=float(policy.get("frequency_weight", 0.4)),
            recency_weight=float(policy.get("recency_weight", 0.4)),
            size_weight=float(policy.get("size_weight", 0.2)),
            priority_weight=float(policy.get("priority_weight", 0.2)),
            hot_use_count_threshold=int(policy.get("hot_use_count_threshold", 3)),
            hot_recency_window_sec=float(policy.get("hot_recency_window_sec", 600.0)),
            hot_bonus=float(policy.get("hot_bonus", 0.3)),
            dynamic_vram_ratio=float(policy.get("dynamic_vram_ratio", 0.9)),
            dynamic_vram_safety_mb=int(policy.get("dynamic_vram_safety_mb", 200)),
        )
        return cls(config=config)

    def resolve_max_models(self, base_max_models: int, snapshot: Optional[QueueSnapshot]) -> int:
        """根据队列状态调整最大模型数。"""
        max_models = max(1, base_max_models)
        if snapshot and snapshot.is_idle and snapshot.idle_seconds >= self._config.idle_unload_after_sec:
            max_models = max(self._config.min_models_when_idle, 0)
        return max_models

    def select_evictions(
        self,
        entries: List[ModelResidencyEntry],
        required_vram_mb: int,
        max_models: int,
        available_vram_mb: int,
        current_vram_mb: int,
        snapshot: Optional[QueueSnapshot],
        incoming_models: int = 1,
    ) -> EvictionPlan:
        """生成驱逐计划。"""
        if not entries:
            return EvictionPlan(keys=[], reason="none")

        evict_keys: List[str] = []
        freed_vram = 0
        total_models = len(entries)
        max_models = max(max_models, 0)
        target_models = max_models if max_models > 0 else 0

        candidates = [
            entry for entry in entries
            if not entry.is_force_resident and not entry.is_keep_resident
        ]

        max_use_count = max((entry.use_count for entry in candidates), default=0)
        max_vram = max((entry.vram_mb for entry in candidates), default=0)
        max_priority = max((entry.evict_priority for entry in candidates), default=0)

        def needs_eviction() -> bool:
            projected_models = total_models - len(evict_keys) + incoming_models
            projected_vram = current_vram_mb - freed_vram + max(required_vram_mb, 0)
            over_models = projected_models > target_models
            over_vram = projected_vram > available_vram_mb
            return over_models or over_vram

        while needs_eviction() and candidates:
            candidates.sort(
                key=lambda entry: self._score_entry(
                    entry,
                    max_use_count=max_use_count,
                    max_vram=max_vram,
                    max_priority=max_priority,
                )
            )
            candidate = candidates.pop(0)
            evict_keys.append(candidate.plan_key)
            freed_vram += candidate.vram_mb

        reason = "budget"
        if snapshot and snapshot.is_idle and snapshot.idle_seconds >= self._config.idle_unload_after_sec:
            reason = "idle"
        elif total_models + incoming_models > target_models:
            reason = "capacity"

        return EvictionPlan(keys=evict_keys, reason=reason)

    def _score_entry(
        self,
        entry: ModelResidencyEntry,
        max_use_count: int,
        max_vram: int,
        max_priority: int,
    ) -> float:
        """计算条目保留分数（越低越容易被驱逐）。"""
        if entry.is_force_resident or entry.is_keep_resident:
            return float("inf")

        freq_norm = entry.use_count / max_use_count if max_use_count > 0 else 0.0
        recency_norm = 1.0 - min(entry.idle_seconds / self._config.recency_window_sec, 1.0)
        size_norm = entry.vram_mb / max_vram if max_vram > 0 else 0.0
        priority_norm = entry.evict_priority / max_priority if max_priority > 0 else 0.0

        score = (
            self._config.frequency_weight * freq_norm
            + self._config.recency_weight * recency_norm
            - self._config.size_weight * size_norm
            + self._config.priority_weight * priority_norm
        )

        if self._is_hot(entry):
            score += self._config.hot_bonus

        return score

    def _is_hot(self, entry: ModelResidencyEntry) -> bool:
        """判断模型是否为高频模型（优先保留）。"""
        return (
            entry.use_count >= self._config.hot_use_count_threshold
            and entry.idle_seconds <= self._config.hot_recency_window_sec
        )
