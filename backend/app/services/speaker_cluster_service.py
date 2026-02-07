"""
SpeakerClusterService - 在线说话人聚类服务

V3.2.0+dev.20260207.01: 最小可行版本
- 余弦相似度 + 最近质心在线聚类
- 防抖机制：连续2个chunk确认切换 + 最小turn时长门槛
- 仅做"是否换人"判定，不做复杂 diarization
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SpeakerClusterConfig:
    """说话人聚类配置"""
    # 余弦相似度阈值：低于此值视为新说话人
    similarity_threshold: float = 0.75
    # 防抖：需要连续多少个 chunk 判定为新说话人才确认切换
    debounce_count: int = 2
    # 最小 turn 时长（秒）：当前说话人持续时间不足此值时不切换
    min_turn_duration: float = 1.5
    # 最小 turn chunk 数：当前说话人持续 chunk 数不足此值时不切换
    min_turn_chunks: int = 3
    # 是否启用聚类（默认开启）
    enabled: bool = True


@dataclass
class SpeakerCentroid:
    """说话人质心"""
    speaker_id: str
    embedding: np.ndarray  # 质心向量（累积平均）
    sample_count: int = 1  # 样本数量（用于增量更新质心）


@dataclass
class ClusterState:
    """聚类状态（用于防抖）"""
    current_speaker_id: Optional[str] = None
    current_turn_start: float = 0.0  # 当前 turn 起始时间
    current_turn_chunks: int = 0  # 当前 turn 的 chunk 数
    pending_speaker_id: Optional[str] = None  # 待确认的新说话人
    pending_count: int = 0  # 待确认计数


@dataclass
class ClusterResult:
    """聚类结果"""
    speaker_id: str
    is_speaker_changed: bool  # 是否确认切换（经过防抖）
    similarity: float  # 与匹配质心的相似度
    is_new_speaker: bool  # 是否为新发现的说话人


class SpeakerClusterService:
    """
    在线说话人聚类服务

    核心逻辑：
    1. 对每个 chunk 的 speaker_embedding 计算与所有质心的余弦相似度
    2. 若最高相似度 >= 阈值，归入该说话人；否则创建新说话人
    3. 防抖机制：连续 N 个 chunk 判定为同一新说话人才确认切换
    4. 最小 turn 门槛：当前说话人持续时间/chunk数不足时不切换
    """

    def __init__(
        self,
        config: Optional[SpeakerClusterConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config or SpeakerClusterConfig()
        self._logger = logger or logging.getLogger(__name__)
        self._centroids: Dict[str, SpeakerCentroid] = {}
        self._state = ClusterState()
        self._speaker_counter = 0

    def reset(self) -> None:
        """重置聚类状态（新任务时调用）"""
        self._centroids.clear()
        self._state = ClusterState()
        self._speaker_counter = 0
        self._logger.debug("SpeakerClusterService 状态已重置")

    @property
    def config(self) -> SpeakerClusterConfig:
        return self._config

    @property
    def speaker_count(self) -> int:
        """当前识别的说话人数量"""
        return len(self._centroids)

    def cluster(
        self,
        embedding: List[float],
        chunk_start: float,
        chunk_end: float,
    ) -> ClusterResult:
        """
        对单个 chunk 进行聚类

        Args:
            embedding: 192维声纹向量
            chunk_start: chunk 起始时间（秒）
            chunk_end: chunk 结束时间（秒）

        Returns:
            ClusterResult: 聚类结果（包含是否确认切换）
        """
        if not self._config.enabled:
            # 禁用时返回默认说话人
            return ClusterResult(
                speaker_id="speaker_0",
                is_speaker_changed=False,
                similarity=1.0,
                is_new_speaker=False,
            )

        emb_array = np.array(embedding, dtype=np.float32)

        # 计算与所有质心的相似度
        best_speaker_id, best_similarity = self._find_nearest_centroid(emb_array)

        # 判断是否为新说话人
        is_new = best_similarity < self._config.similarity_threshold

        # 调试日志：观察相似度分布
        if best_speaker_id is not None:
            self._logger.debug(
                "聚类判定: best_sim=%.3f, threshold=%.3f, is_new=%s, speakers=%d",
                best_similarity, self._config.similarity_threshold, is_new, len(self._centroids)
            )

        if is_new:
            # 创建新说话人
            raw_speaker_id = self._create_new_speaker(emb_array)
        else:
            # 归入已有说话人，更新质心
            raw_speaker_id = best_speaker_id
            self._update_centroid(raw_speaker_id, emb_array)

        # 应用防抖逻辑
        final_speaker_id, is_changed = self._apply_debounce(
            raw_speaker_id=raw_speaker_id,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )

        return ClusterResult(
            speaker_id=final_speaker_id,
            is_speaker_changed=is_changed,
            similarity=best_similarity if not is_new else 0.0,
            is_new_speaker=is_new,
        )

    def _find_nearest_centroid(
        self, embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """找到最近的质心"""
        if not self._centroids:
            return None, 0.0

        best_id = None
        best_sim = -1.0

        for speaker_id, centroid in self._centroids.items():
            sim = self._cosine_similarity(embedding, centroid.embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = speaker_id

        return best_id, best_sim

    def _create_new_speaker(self, embedding: np.ndarray) -> str:
        """创建新说话人"""
        speaker_id = f"speaker_{self._speaker_counter}"
        self._speaker_counter += 1

        self._centroids[speaker_id] = SpeakerCentroid(
            speaker_id=speaker_id,
            embedding=embedding.copy(),
            sample_count=1,
        )

        self._logger.debug(
            "创建新说话人: %s (当前共 %d 人)",
            speaker_id, len(self._centroids)
        )
        return speaker_id

    def _update_centroid(self, speaker_id: str, embedding: np.ndarray) -> None:
        """增量更新质心（在线平均）"""
        if speaker_id not in self._centroids:
            return

        centroid = self._centroids[speaker_id]
        n = centroid.sample_count
        # 增量平均：new_mean = old_mean + (new_sample - old_mean) / (n + 1)
        centroid.embedding = centroid.embedding + (embedding - centroid.embedding) / (n + 1)
        centroid.sample_count = n + 1

    def _apply_debounce(
        self,
        raw_speaker_id: str,
        chunk_start: float,
        chunk_end: float,
    ) -> Tuple[str, bool]:
        """
        应用防抖逻辑

        Returns:
            (final_speaker_id, is_speaker_changed)
        """
        state = self._state

        # 首次调用，初始化状态
        if state.current_speaker_id is None:
            state.current_speaker_id = raw_speaker_id
            state.current_turn_start = chunk_start
            state.current_turn_chunks = 1
            state.pending_speaker_id = None
            state.pending_count = 0
            return raw_speaker_id, False

        # 与当前说话人相同
        if raw_speaker_id == state.current_speaker_id:
            state.current_turn_chunks += 1
            state.pending_speaker_id = None
            state.pending_count = 0
            return raw_speaker_id, False

        # 检查最小 turn 门槛
        turn_duration = chunk_start - state.current_turn_start
        if (turn_duration < self._config.min_turn_duration or
            state.current_turn_chunks < self._config.min_turn_chunks):
            # 当前 turn 太短，不允许切换，保持当前说话人
            self._logger.debug(
                "防抖：turn 太短 (%.2fs, %d chunks)，保持 %s",
                turn_duration, state.current_turn_chunks, state.current_speaker_id
            )
            return state.current_speaker_id, False

        # 检查是否与待确认说话人相同
        if raw_speaker_id == state.pending_speaker_id:
            state.pending_count += 1
        else:
            # 新的待确认说话人
            state.pending_speaker_id = raw_speaker_id
            state.pending_count = 1

        # 检查是否达到防抖阈值
        if state.pending_count >= self._config.debounce_count:
            # 确认切换
            old_speaker = state.current_speaker_id
            state.current_speaker_id = raw_speaker_id
            state.current_turn_start = chunk_start
            state.current_turn_chunks = 1
            state.pending_speaker_id = None
            state.pending_count = 0

            self._logger.debug(
                "说话人切换确认: %s -> %s",
                old_speaker, raw_speaker_id
            )
            return raw_speaker_id, True

        # 未达到防抖阈值，保持当前说话人
        self._logger.debug(
            "防抖中: pending=%s, count=%d/%d",
            state.pending_speaker_id,
            state.pending_count,
            self._config.debounce_count,
        )
        return state.current_speaker_id, False

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# 单例模式
_cluster_service: Optional[SpeakerClusterService] = None


def get_speaker_cluster_service(
    config: Optional[SpeakerClusterConfig] = None,
    logger: Optional[logging.Logger] = None,
) -> SpeakerClusterService:
    """获取说话人聚类服务单例"""
    global _cluster_service
    if _cluster_service is None:
        _cluster_service = SpeakerClusterService(config=config, logger=logger)
    return _cluster_service


def reset_speaker_cluster_service() -> None:
    """重置说话人聚类服务（测试用）"""
    global _cluster_service
    if _cluster_service is not None:
        _cluster_service.reset()
    _cluster_service = None
