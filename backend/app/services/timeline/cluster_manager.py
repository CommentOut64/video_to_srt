"""
Timeline 聚类管理器（Phase 2）。

设计模式：State + Strategy。
原因：用显式状态机管理 speaker 生命周期（candidate/confirmed/merged），
并把相似度策略收敛在同一服务，避免业务层散落阈值判断。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from app.models.speaker_timeline_models import SpeakerProfile


SpeakerStatus = Literal["candidate", "confirmed", "merged", "unknown"]


@dataclass(frozen=True)
class ClusterManagerConfig:
    """聚类管理配置。"""

    similarity_threshold: float = 0.68
    merge_similarity_threshold: float = 0.88
    min_support_turns: int = 2
    min_total_duration: float = 2.0
    min_chunk_duration: float = 0.30
    min_quality_score: float = 0.35
    merge_every_n_turns: int = 6


@dataclass(frozen=True)
class ClusterAssignment:
    """单个片段的聚类判定结果。"""

    speaker_id: str
    status: SpeakerStatus
    similarity: float
    is_new_speaker: bool
    is_low_quality_blocked: bool


@dataclass
class _SpeakerState:
    """内部 speaker 状态。"""

    speaker_id: str
    centroid: np.ndarray
    sample_count: int
    support_turns: int
    total_duration: float
    status: SpeakerStatus
    merged_into: Optional[str] = None


class ClusterManager:
    """说话人聚类状态管理器。"""

    def __init__(
        self,
        config: Optional[ClusterManagerConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or ClusterManagerConfig()
        self.logger = logger or logging.getLogger(__name__)
        self._speaker_states: dict[str, _SpeakerState] = {}
        self._speaker_remap: dict[str, str] = {}
        self._speaker_counter = 0
        self._assignment_count = 0

    def reset(self) -> None:
        """重置状态。"""
        self._speaker_states.clear()
        self._speaker_remap.clear()
        self._speaker_counter = 0
        self._assignment_count = 0

    def assign_chunk(
        self,
        embedding: list[float],
        *,
        duration: float,
        quality_score: float = 1.0,
        is_overlap: bool = False,
    ) -> ClusterAssignment:
        """为单个片段分配 speaker。"""
        if not embedding:
            return ClusterAssignment(
                speaker_id="unknown",
                status="unknown",
                similarity=0.0,
                is_new_speaker=False,
                is_low_quality_blocked=True,
            )

        emb = self._normalize(np.asarray(embedding, dtype=np.float32))
        best_speaker_id, best_similarity = self._find_best_match(emb)

        is_low_quality = (
            is_overlap
            or duration < self.config.min_chunk_duration
            or quality_score < self.config.min_quality_score
        )

        is_can_create_new = not is_low_quality
        is_need_new = (
            best_speaker_id is None
            or best_similarity < self.config.similarity_threshold
        )

        if is_need_new and is_can_create_new:
            speaker_id = self._create_candidate_speaker(embedding=emb, duration=duration)
            state = self._speaker_states[speaker_id]
            is_new_speaker = True
            status: SpeakerStatus = state.status
            similarity = 0.0
        elif best_speaker_id is not None:
            speaker_id = self.resolve_speaker_id(best_speaker_id)
            self._update_speaker(speaker_id, emb, duration)
            state = self._speaker_states[speaker_id]
            is_new_speaker = False
            status = state.status
            similarity = max(0.0, float(best_similarity))
        else:
            return ClusterAssignment(
                speaker_id="unknown",
                status="unknown",
                similarity=0.0,
                is_new_speaker=False,
                is_low_quality_blocked=True,
            )

        self._maybe_promote(state)
        status = state.status
        self._assignment_count += 1
        self._maybe_merge_confirmed()

        return ClusterAssignment(
            speaker_id=speaker_id,
            status=status,
            similarity=similarity,
            is_new_speaker=is_new_speaker,
            is_low_quality_blocked=is_low_quality and is_need_new,
        )

    def resolve_speaker_id(self, speaker_id: str) -> str:
        """将被合并的 speaker_id 解析为当前有效 id。"""
        current = speaker_id
        while current in self._speaker_remap:
            current = self._speaker_remap[current]
        return current

    def get_status(self, speaker_id: str) -> SpeakerStatus:
        """读取 speaker 当前状态。"""
        resolved = self.resolve_speaker_id(speaker_id)
        state = self._speaker_states.get(resolved)
        if state is None:
            return "unknown"
        return state.status

    def build_profiles(self) -> list[SpeakerProfile]:
        """导出 SpeakerProfile 列表（仅 active speaker）。"""
        profiles: list[SpeakerProfile] = []
        for state in self._speaker_states.values():
            if state.status == "merged":
                continue
            quality_score = self._estimate_quality_score(state)
            profiles.append(
                SpeakerProfile(
                    speaker_id=state.speaker_id,
                    embedding_centroid=state.centroid.astype(np.float32).tolist(),
                    sample_count=state.sample_count,
                    quality_score=quality_score,
                )
            )
        profiles.sort(key=lambda item: item.speaker_id)
        return profiles

    def to_dict(self) -> dict[str, object]:
        """导出聚类状态快照（用于断点恢复）。"""
        states_payload: list[dict[str, object]] = []
        for state in sorted(self._speaker_states.values(), key=lambda item: item.speaker_id):
            states_payload.append(
                {
                    "speaker_id": state.speaker_id,
                    "centroid": state.centroid.astype(np.float32).tolist(),
                    "sample_count": int(state.sample_count),
                    "support_turns": int(state.support_turns),
                    "total_duration": float(state.total_duration),
                    "status": str(state.status),
                    "merged_into": state.merged_into,
                }
            )

        return {
            "speaker_states": states_payload,
            "speaker_remap": dict(self._speaker_remap),
            "speaker_counter": int(self._speaker_counter),
            "assignment_count": int(self._assignment_count),
        }

    def load_from_dict(self, payload: dict[str, object]) -> bool:
        """从聚类状态快照恢复。"""
        try:
            if not isinstance(payload, dict):
                return False

            raw_states = payload.get("speaker_states", [])
            if not isinstance(raw_states, list):
                return False

            restored_states: dict[str, _SpeakerState] = {}
            for raw_state in raw_states:
                if not isinstance(raw_state, dict):
                    return False

                speaker_id = str(raw_state.get("speaker_id") or "").strip()
                if not speaker_id:
                    return False

                centroid_values = raw_state.get("centroid", [])
                if not isinstance(centroid_values, list) or not centroid_values:
                    return False

                centroid = self._normalize(np.asarray(centroid_values, dtype=np.float32))
                status_value = str(raw_state.get("status") or "candidate")
                if status_value not in {"candidate", "confirmed", "merged", "unknown"}:
                    status_value = "candidate"

                restored_states[speaker_id] = _SpeakerState(
                    speaker_id=speaker_id,
                    centroid=centroid,
                    sample_count=max(1, int(raw_state.get("sample_count", 1))),
                    support_turns=max(1, int(raw_state.get("support_turns", 1))),
                    total_duration=max(0.0, float(raw_state.get("total_duration", 0.0))),
                    status=status_value,  # type: ignore[arg-type]
                    merged_into=(
                        str(raw_state.get("merged_into"))
                        if raw_state.get("merged_into") is not None
                        else None
                    ),
                )

            raw_remap = payload.get("speaker_remap", {})
            if not isinstance(raw_remap, dict):
                return False
            restored_remap = {str(key): str(value) for key, value in raw_remap.items()}

            self._speaker_states = restored_states
            self._speaker_remap = restored_remap

            max_counter = 0
            for speaker_id in restored_states:
                if speaker_id.startswith("spk_"):
                    suffix = speaker_id.replace("spk_", "", 1)
                    if suffix.isdigit():
                        max_counter = max(max_counter, int(suffix) + 1)

            payload_counter = int(payload.get("speaker_counter", 0))
            self._speaker_counter = max(max_counter, payload_counter)
            self._assignment_count = max(0, int(payload.get("assignment_count", 0)))
            return True
        except (TypeError, ValueError) as exc:
            # Why: 断点恢复使用外部持久化数据，必须显式记录反序列化失败原因。
            self.logger.warning("ClusterManager 状态恢复失败: %s", exc)
            return False

    def _create_candidate_speaker(self, embedding: np.ndarray, duration: float) -> str:
        speaker_id = f"spk_{self._speaker_counter:03d}"
        self._speaker_counter += 1
        self._speaker_states[speaker_id] = _SpeakerState(
            speaker_id=speaker_id,
            centroid=embedding.copy(),
            sample_count=1,
            support_turns=1,
            total_duration=max(0.0, float(duration)),
            status="candidate",
        )
        return speaker_id

    def _update_speaker(self, speaker_id: str, embedding: np.ndarray, duration: float) -> None:
        state = self._speaker_states[speaker_id]
        sample_count = max(1, state.sample_count)
        state.centroid = self._normalize(
            state.centroid + (embedding - state.centroid) / (sample_count + 1)
        )
        state.sample_count = sample_count + 1
        state.support_turns += 1
        state.total_duration += max(0.0, float(duration))

    def _maybe_promote(self, state: _SpeakerState) -> None:
        if state.status != "candidate":
            return
        is_support_ready = state.support_turns >= self.config.min_support_turns
        is_duration_ready = state.total_duration >= self.config.min_total_duration
        if is_support_ready and is_duration_ready:
            state.status = "confirmed"

    def _maybe_merge_confirmed(self) -> None:
        merge_every = max(1, self.config.merge_every_n_turns)
        if self._assignment_count % merge_every != 0:
            return

        confirmed = [
            state
            for state in self._speaker_states.values()
            if state.status == "confirmed"
        ]
        if len(confirmed) < 2:
            return

        best_pair: tuple[_SpeakerState, _SpeakerState] | None = None
        best_similarity = -1.0
        for left_idx in range(len(confirmed)):
            for right_idx in range(left_idx + 1, len(confirmed)):
                left = confirmed[left_idx]
                right = confirmed[right_idx]
                similarity = self._cosine_similarity(left.centroid, right.centroid)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pair = (left, right)

        if best_pair is None:
            return
        if best_similarity < self.config.merge_similarity_threshold:
            return

        left, right = best_pair
        target, source = self._select_merge_target(left, right)
        self._merge_into(target=target, source=source)
        self.logger.info(
            "ClusterManager 增量合并: %s <- %s similarity=%.4f",
            target.speaker_id,
            source.speaker_id,
            best_similarity,
        )

    def _select_merge_target(
        self,
        left: _SpeakerState,
        right: _SpeakerState,
    ) -> tuple[_SpeakerState, _SpeakerState]:
        if left.sample_count >= right.sample_count:
            return left, right
        return right, left

    def _merge_into(self, *, target: _SpeakerState, source: _SpeakerState) -> None:
        total_samples = target.sample_count + source.sample_count
        if total_samples <= 0:
            return

        target.centroid = self._normalize(
            (target.centroid * target.sample_count + source.centroid * source.sample_count)
            / total_samples
        )
        target.sample_count = total_samples
        target.support_turns += source.support_turns
        target.total_duration += source.total_duration
        target.status = "confirmed"

        source.status = "merged"
        source.merged_into = target.speaker_id
        self._speaker_remap[source.speaker_id] = target.speaker_id

    def _find_best_match(self, embedding: np.ndarray) -> tuple[Optional[str], float]:
        best_speaker_id: Optional[str] = None
        best_similarity = -1.0
        for state in self._speaker_states.values():
            if state.status == "merged":
                continue
            similarity = self._cosine_similarity(embedding, state.centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker_id = state.speaker_id
        return best_speaker_id, best_similarity

    @staticmethod
    def _estimate_quality_score(state: _SpeakerState) -> float:
        sample_term = min(1.0, state.sample_count / 8.0)
        duration_term = min(1.0, state.total_duration / 12.0)
        return float(max(0.0, min(1.0, 0.5 * sample_term + 0.5 * duration_term)))

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            return embedding
        return embedding / norm

    @staticmethod
    def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
        left_norm = np.linalg.norm(left)
        right_norm = np.linalg.norm(right)
        if left_norm < 1e-8 or right_norm < 1e-8:
            return 0.0
        return float(np.dot(left, right) / (left_norm * right_norm))
