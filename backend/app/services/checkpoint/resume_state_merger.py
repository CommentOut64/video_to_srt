"""
恢复状态双源合并器。

设计模式：策略模式（Strategy Pattern）
原因：将“控制面优先”和“数据面优先”的合并规则集中在独立模块，
避免恢复路径在编排器内不断膨胀。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.core.logging import resolve_loguru_logger
from app.services.checkpoint.runtime_checkpoint_models import RuntimeCheckpointSnapshot
from app.services.checkpoint.runtime_checkpoint_service import RuntimeCheckpointService
from app.services.job.checkpoint_manager import CheckpointManagerV37


@dataclass
class MergeDecision:
    """单个字段的合并决策记录。"""

    field_name: str
    source: str
    reason: str
    value_summary: str


@dataclass
class MergedResumeState:
    """合并后的恢复状态。"""

    # 预处理边界
    vad_completed: bool = False
    spectral_triage_completed: bool = False
    separation_completed: bool = False
    langid_completed: bool = False

    # 转录细粒度状态
    fast_processed_indices: Set[int] = field(default_factory=set)
    slow_processed_indices: Set[int] = field(default_factory=set)
    finalized_indices: Set[int] = field(default_factory=set)
    previous_whisper_text: str = ""
    sentences_snapshot: List[Dict[str, Any]] = field(default_factory=list)
    sentence_count: int = 0
    chunk_sentences_map: Dict[Any, List[int]] = field(default_factory=dict)

    # 审计信息
    merge_decisions: List[MergeDecision] = field(default_factory=list)
    runtime_state_available: bool = False
    checkpoint_json_available: bool = False


class ResumeStateMerger:
    """
    双源恢复合并器。

    规则：
    1. 预处理完成位优先取 runtime_state.db
    2. 转录细粒度字段优先取 checkpoint.json
    3. 冲突保留诊断，不做激进跳跃
    """

    _PREPROCESS_COMMIT_VAD = {
        "vad_chunk",
        "triage_chunk",
        "separation_chunk",
        "langid_chunk",
        "speaker_chunk",
    }
    _PREPROCESS_COMMIT_TRIAGE = {
        "triage_chunk",
        "separation_chunk",
        "langid_chunk",
        "speaker_chunk",
    }
    _PREPROCESS_COMMIT_SEPARATION = {
        "separation_chunk",
        "langid_chunk",
        "speaker_chunk",
    }
    _PREPROCESS_COMMIT_LANGID = {
        "langid_chunk",
        "speaker_chunk",
    }

    def __init__(self, job_dir: Path):
        self._job_dir = Path(job_dir)
        self._logger = resolve_loguru_logger(
            None,
            __name__,
            job_id=self._job_dir.name,
            layer="检查点层",
            processor_name="resume_state_merger",
        )

    def merge(self) -> MergedResumeState:
        """执行双源合并。"""
        result = MergedResumeState()

        runtime_snapshot = self._load_runtime_state()
        checkpoint_data = self._load_checkpoint()

        result.runtime_state_available = runtime_snapshot is not None
        result.checkpoint_json_available = checkpoint_data is not None

        if runtime_snapshot is None and checkpoint_data is None:
            self._logger.info("恢复合并: runtime_state 与 checkpoint 均不可用，按全新任务执行")
            return result

        self._merge_preprocessing(
            result=result,
            runtime=runtime_snapshot,
            checkpoint=checkpoint_data,
        )
        self._merge_transcription(
            result=result,
            runtime=runtime_snapshot,
            checkpoint=checkpoint_data,
        )

        self._logger.info(
            f"恢复合并完成: runtime={result.runtime_state_available} "
            f"checkpoint={result.checkpoint_json_available} "
            f"decisions={len(result.merge_decisions)}"
        )
        return result

    def _load_runtime_state(self) -> Optional[RuntimeCheckpointSnapshot]:
        """加载 runtime_state 快照。"""
        try:
            service = RuntimeCheckpointService(job_dir=self._job_dir)
            if service.has_runtime_state():
                return service.load_snapshot()
        except Exception as exc:
            self._logger.warning(f"读取 runtime_state 失败: {exc}")
        return None

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """加载 checkpoint.json。"""
        try:
            manager = CheckpointManagerV37(self._job_dir, logger=self._logger)
            checkpoint = manager.load_checkpoint()
            if checkpoint is None:
                return None
            if isinstance(checkpoint, dict):
                return checkpoint
            if hasattr(checkpoint, "to_dict"):
                checkpoint_dict = checkpoint.to_dict()
                return checkpoint_dict if isinstance(checkpoint_dict, dict) else None
            return None
        except Exception as exc:
            self._logger.warning(f"读取 checkpoint 失败: {exc}")
            return None

    def _merge_preprocessing(
        self,
        *,
        result: MergedResumeState,
        runtime: Optional[RuntimeCheckpointSnapshot],
        checkpoint: Optional[Dict[str, Any]],
    ) -> None:
        """合并预处理边界。"""
        if runtime and runtime.last_unit_commits:
            preprocess_commit = runtime.last_unit_commits.get("preprocess", "")
            result.vad_completed = preprocess_commit in self._PREPROCESS_COMMIT_VAD
            result.spectral_triage_completed = preprocess_commit in self._PREPROCESS_COMMIT_TRIAGE
            result.separation_completed = preprocess_commit in self._PREPROCESS_COMMIT_SEPARATION
            result.langid_completed = preprocess_commit in self._PREPROCESS_COMMIT_LANGID
            result.merge_decisions.append(
                MergeDecision(
                    field_name="preprocessing",
                    source="runtime_state",
                    reason="runtime_state 提供事务化单元提交边界",
                    value_summary=f"preprocess_commit={preprocess_commit or 'none'}",
                )
            )
            return

        if checkpoint:
            preprocessing = checkpoint.get("preprocessing", {}) or {}
            spectral_triage = preprocessing.get("spectral_triage", {}) or {}
            separation = preprocessing.get("separation", {}) or {}
            result.vad_completed = bool(preprocessing.get("vad_completed", False))
            result.spectral_triage_completed = bool(
                preprocessing.get("spectral_triage_completed", spectral_triage.get("completed", False))
            )
            result.separation_completed = bool(
                preprocessing.get("separation_completed", separation.get("completed", False))
            )
            result.langid_completed = bool(preprocessing.get("langid_completed", False))
            result.merge_decisions.append(
                MergeDecision(
                    field_name="preprocessing",
                    source="checkpoint_json",
                    reason="runtime_state 不可用，降级使用 checkpoint 预处理快照",
                    value_summary=(
                        f"vad={result.vad_completed}, triage={result.spectral_triage_completed}, "
                        f"sep={result.separation_completed}, langid={result.langid_completed}"
                    ),
                )
            )

    def _merge_transcription(
        self,
        *,
        result: MergedResumeState,
        runtime: Optional[RuntimeCheckpointSnapshot],
        checkpoint: Optional[Dict[str, Any]],
    ) -> None:
        """合并转录细粒度状态。"""
        if checkpoint is None:
            result.merge_decisions.append(
                MergeDecision(
                    field_name="transcription",
                    source="default",
                    reason="checkpoint 不可用，转录恢复为空",
                    value_summary="fast=0,slow=0,finalized=0,sentences=0",
                )
            )
            return

        transcription = checkpoint.get("transcription", {}) or {}
        fast_worker = transcription.get("fast_worker", {}) or {}
        slow_worker = transcription.get("slow_worker", {}) or {}
        alignment = transcription.get("alignment", {}) or {}
        slow_context = slow_worker.get("context_state", {}) or {}

        result.fast_processed_indices = self._normalize_index_set(
            transcription.get("fast_processed_indices")
            or fast_worker.get("processed_indices")
            or transcription.get("processed_indices")
            or []
        )
        result.slow_processed_indices = self._normalize_index_set(
            transcription.get("slow_processed_indices")
            or slow_worker.get("processed_indices")
            or []
        )
        result.finalized_indices = self._normalize_index_set(
            transcription.get("finalized_indices")
            or alignment.get("finalized_indices")
            or []
        )
        result.previous_whisper_text = str(
            transcription.get("previous_whisper_text")
            or slow_context.get("previous_whisper_text")
            or ""
        )
        result.sentences_snapshot = self._normalize_sentences_snapshot(
            transcription.get("sentences_snapshot", [])
        )
        result.sentence_count = self._normalize_int(
            transcription.get("sentence_count", len(result.sentences_snapshot))
        )
        result.chunk_sentences_map = self._normalize_chunk_sentences_map(
            transcription.get("chunk_sentences_map", {}) or {}
        )
        if result.sentence_count <= 0:
            result.sentence_count = len(result.sentences_snapshot)

        result.merge_decisions.append(
            MergeDecision(
                field_name="transcription",
                source="checkpoint_json",
                reason="checkpoint 提供转录细粒度恢复数据",
                value_summary=(
                    f"fast={len(result.fast_processed_indices)}, "
                    f"slow={len(result.slow_processed_indices)}, "
                    f"finalized={len(result.finalized_indices)}, "
                    f"sentences={len(result.sentences_snapshot)}"
                ),
            )
        )

        if runtime and runtime.transcription_hint:
            runtime_hint_max = self._extract_runtime_hint_max_index(runtime.transcription_hint)
            checkpoint_max = max(result.finalized_indices) if result.finalized_indices else -1
            if runtime_hint_max > checkpoint_max:
                result.merge_decisions.append(
                    MergeDecision(
                        field_name="transcription_conflict",
                        source="runtime_state",
                        reason="runtime_hint 提示提交点更新，但保留 checkpoint 细粒度数据",
                        value_summary=f"runtime_hint_max={runtime_hint_max}, checkpoint_max={checkpoint_max}",
                    )
                )

    @staticmethod
    def _normalize_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _normalize_index_set(raw_values: Any) -> Set[int]:
        indices: Set[int] = set()
        if not isinstance(raw_values, (list, tuple, set)):
            return indices
        for raw_value in raw_values:
            try:
                indices.add(int(raw_value))
            except (TypeError, ValueError):
                continue
        return indices

    @staticmethod
    def _normalize_sentences_snapshot(raw_snapshot: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_snapshot, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in raw_snapshot:
            if isinstance(item, dict):
                normalized.append(item)
        return normalized

    @staticmethod
    def _normalize_chunk_sentences_map(raw_map: Any) -> Dict[Any, List[int]]:
        if not isinstance(raw_map, dict):
            return {}
        normalized: Dict[Any, List[int]] = {}
        for raw_key, raw_value in raw_map.items():
            key: Any = raw_key
            if isinstance(raw_key, str) and raw_key.lstrip("-").isdigit():
                try:
                    key = int(raw_key)
                except (TypeError, ValueError):
                    key = raw_key
            if not isinstance(raw_value, list):
                continue
            normalized[key] = []
            for sentence_index in raw_value:
                try:
                    normalized[key].append(int(sentence_index))
                except (TypeError, ValueError):
                    continue
        return normalized

    @staticmethod
    def _extract_runtime_hint_max_index(hint: Dict[str, Any]) -> int:
        candidate_fields = (
            "last_committed_chunk_index",
            "last_chunk_index",
            "finalized_max_index",
        )
        for field_name in candidate_fields:
            if field_name in hint:
                try:
                    return int(hint[field_name])
                except (TypeError, ValueError):
                    continue
        return -1
