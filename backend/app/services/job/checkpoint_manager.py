"""
CheckpointManager - 断点续传管理器

Phase 4 实现 - 2025-12-11
V3.1.0 扩展 - 2025-12-21

核心职责：
1. 保存和加载检查点（checkpoint.json）
2. 原子性写入（临时文件 + 重命名）
3. 检查点验证和恢复
4. 检查点清理
5. [V3.1.0] 支持新的统一检查点格式，包含预处理、转录、输出状态

从 transcription_service.py 拆分出来，专注于断点续传功能。
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass, field
from datetime import datetime


# V3.1.0 检查点版本
CHECKPOINT_VERSION = "3.1.0"


@dataclass
class CheckpointData:
    """检查点数据结构（兼容旧版本）"""
    job_id: str
    phase: str
    total_chunks: int
    processed_chunks: int
    processed_indices: list
    language: Optional[str] = None
    original_settings: Optional[Dict[str, Any]] = None
    # 可扩展字段
    extra_data: Optional[Dict[str, Any]] = None


@dataclass
class PreprocessingState:
    """V3.7 预处理阶段状态"""
    audio_extracted: bool = False
    audio_path: Optional[str] = None
    audio_duration_sec: float = 0.0
    sample_rate: int = 16000

    vad_completed: bool = False
    total_chunks: int = 0
    chunks_metadata: List[Dict[str, Any]] = field(default_factory=list)

    # 频谱分诊状态
    spectral_triage_completed: bool = False
    diagnosed_indices: List[int] = field(default_factory=list)
    need_separation_count: int = 0

    # 人声分离状态
    separation_mode: str = "on_demand"  # global / on_demand
    separation_completed: bool = False
    separated_indices: List[int] = field(default_factory=list)
    global_separation_done: bool = False


@dataclass
class TranscriptionState:
    """V3.7 转录阶段状态"""
    # FastWorker (SenseVoice)
    fast_processed_indices: List[int] = field(default_factory=list)
    fast_completed_count: int = 0
    fuse_upgraded_indices: List[int] = field(default_factory=list)

    # SlowWorker (Whisper) - 关键：包含上文状态
    slow_processed_indices: List[int] = field(default_factory=list)
    slow_completed_count: int = 0
    previous_whisper_text: str = ""  # Whisper 上文状态（恢复必需）
    last_processed_index: int = -1

    # AlignmentWorker
    finalized_indices: List[int] = field(default_factory=list)
    alignment_completed_count: int = 0
    alignment_levels: Dict[int, str] = field(default_factory=dict)

    # V3.1.0: 字幕快照（断点续传核心）
    # 保存已生成的所有字幕句子，确保恢复时不丢失
    sentences_snapshot: List[Dict[str, Any]] = field(default_factory=list)
    sentence_count: int = 0  # 全局句子计数器
    chunk_sentences_map: Dict[int, List[int]] = field(default_factory=dict)  # chunk_index -> [sentence_indices]


@dataclass
class ControlState:
    """V3.7 控制状态"""
    paused: bool = False
    canceled: bool = False
    pending_pause: bool = False
    pending_cancel: bool = False
    pause_reason: Optional[str] = None


@dataclass
class CheckpointV37:
    """
    V3.7 统一检查点格式

    支持完整的断点续传，包括：
    - 预处理阶段状态
    - 转录阶段状态（含 Whisper 上文）
    - 输出状态
    - 控制状态
    """
    version: str = CHECKPOINT_VERSION
    job_id: str = ""
    created_at: str = ""
    updated_at: str = ""

    phase: str = "pending"  # pending/preprocessing/transcribe/finalize/complete
    phase_status: str = "pending"  # pending/in_progress/completed/failed

    preprocessing: PreprocessingState = field(default_factory=PreprocessingState)
    transcription: TranscriptionState = field(default_factory=TranscriptionState)

    # 输出状态
    srt_generated: bool = False
    srt_path: Optional[str] = None

    # 控制状态
    control: ControlState = field(default_factory=ControlState)

    # 原始设置（用于兼容性检查）
    original_settings: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        now = datetime.utcnow().isoformat() + "Z"
        return {
            "version": self.version,
            "job_id": self.job_id,
            "created_at": self.created_at or now,
            "updated_at": now,
            "phase": self.phase,
            "phase_status": self.phase_status,
            "preprocessing": {
                "audio_extracted": self.preprocessing.audio_extracted,
                "audio_path": self.preprocessing.audio_path,
                "audio_duration_sec": self.preprocessing.audio_duration_sec,
                "sample_rate": self.preprocessing.sample_rate,
                "vad_completed": self.preprocessing.vad_completed,
                "total_chunks": self.preprocessing.total_chunks,
                "chunks_metadata": self.preprocessing.chunks_metadata,
                "spectral_triage": {
                    "completed": self.preprocessing.spectral_triage_completed,
                    "diagnosed_indices": self.preprocessing.diagnosed_indices,
                    "need_separation_count": self.preprocessing.need_separation_count
                },
                "separation": {
                    "mode": self.preprocessing.separation_mode,
                    "completed": self.preprocessing.separation_completed,
                    "separated_indices": self.preprocessing.separated_indices,
                    "global_separation_done": self.preprocessing.global_separation_done
                }
            },
            "transcription": {
                "fast_worker": {
                    "processed_indices": self.transcription.fast_processed_indices,
                    "completed_count": self.transcription.fast_completed_count,
                    "fuse_upgraded_indices": self.transcription.fuse_upgraded_indices
                },
                "slow_worker": {
                    "processed_indices": self.transcription.slow_processed_indices,
                    "completed_count": self.transcription.slow_completed_count,
                    "context_state": {
                        "previous_whisper_text": self.transcription.previous_whisper_text,
                        "last_processed_index": self.transcription.last_processed_index
                    }
                },
                "alignment": {
                    "finalized_indices": self.transcription.finalized_indices,
                    "completed_count": self.transcription.alignment_completed_count,
                    "alignment_levels": self.transcription.alignment_levels
                },
                # V3.1.0: 字幕快照
                "sentences_snapshot": self.transcription.sentences_snapshot,
                "sentence_count": self.transcription.sentence_count,
                "chunk_sentences_map": self.transcription.chunk_sentences_map
            },
            "output": {
                "srt_generated": self.srt_generated,
                "srt_path": self.srt_path
            },
            "control": {
                "paused": self.control.paused,
                "canceled": self.control.canceled,
                "pending_pause": self.control.pending_pause,
                "pending_cancel": self.control.pending_cancel,
                "pause_reason": self.control.pause_reason
            },
            "original_settings": self.original_settings
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointV37":
        """从字典创建"""
        checkpoint = cls()
        checkpoint.version = data.get("version", CHECKPOINT_VERSION)
        checkpoint.job_id = data.get("job_id", "")
        checkpoint.created_at = data.get("created_at", "")
        checkpoint.updated_at = data.get("updated_at", "")
        checkpoint.phase = data.get("phase", "pending")
        checkpoint.phase_status = data.get("phase_status", "pending")

        # 预处理状态
        prep = data.get("preprocessing", {})
        checkpoint.preprocessing.audio_extracted = prep.get("audio_extracted", False)
        checkpoint.preprocessing.audio_path = prep.get("audio_path")
        checkpoint.preprocessing.audio_duration_sec = prep.get("audio_duration_sec", 0.0)
        checkpoint.preprocessing.sample_rate = prep.get("sample_rate", 16000)
        checkpoint.preprocessing.vad_completed = prep.get("vad_completed", False)
        checkpoint.preprocessing.total_chunks = prep.get("total_chunks", 0)
        checkpoint.preprocessing.chunks_metadata = prep.get("chunks_metadata", [])

        spectral = prep.get("spectral_triage", {})
        checkpoint.preprocessing.spectral_triage_completed = spectral.get("completed", False)
        checkpoint.preprocessing.diagnosed_indices = spectral.get("diagnosed_indices", [])
        checkpoint.preprocessing.need_separation_count = spectral.get("need_separation_count", 0)

        separation = prep.get("separation", {})
        checkpoint.preprocessing.separation_mode = separation.get("mode", "on_demand")
        checkpoint.preprocessing.separation_completed = separation.get("completed", False)
        checkpoint.preprocessing.separated_indices = separation.get("separated_indices", [])
        checkpoint.preprocessing.global_separation_done = separation.get("global_separation_done", False)

        # 转录状态
        trans = data.get("transcription", {})
        fast = trans.get("fast_worker", {})
        checkpoint.transcription.fast_processed_indices = fast.get("processed_indices", [])
        checkpoint.transcription.fast_completed_count = fast.get("completed_count", 0)
        checkpoint.transcription.fuse_upgraded_indices = fast.get("fuse_upgraded_indices", [])

        slow = trans.get("slow_worker", {})
        checkpoint.transcription.slow_processed_indices = slow.get("processed_indices", [])
        checkpoint.transcription.slow_completed_count = slow.get("completed_count", 0)
        context = slow.get("context_state", {})
        checkpoint.transcription.previous_whisper_text = context.get("previous_whisper_text", "")
        checkpoint.transcription.last_processed_index = context.get("last_processed_index", -1)

        align = trans.get("alignment", {})
        checkpoint.transcription.finalized_indices = align.get("finalized_indices", [])
        checkpoint.transcription.alignment_completed_count = align.get("completed_count", 0)
        checkpoint.transcription.alignment_levels = align.get("alignment_levels", {})

        # V3.1.0: 字幕快照
        checkpoint.transcription.sentences_snapshot = trans.get("sentences_snapshot", [])
        checkpoint.transcription.sentence_count = trans.get("sentence_count", 0)
        # chunk_sentences_map 的键是 int，JSON 反序列化后变成 str，需要转换
        raw_chunk_map = trans.get("chunk_sentences_map", {})
        checkpoint.transcription.chunk_sentences_map = {
            int(k): v for k, v in raw_chunk_map.items()
        } if raw_chunk_map else {}

        # 输出状态
        output = data.get("output", {})
        checkpoint.srt_generated = output.get("srt_generated", False)
        checkpoint.srt_path = output.get("srt_path")

        # 控制状态
        control = data.get("control", {})
        checkpoint.control.paused = control.get("paused", False)
        checkpoint.control.canceled = control.get("canceled", False)
        checkpoint.control.pending_pause = control.get("pending_pause", False)
        checkpoint.control.pending_cancel = control.get("pending_cancel", False)
        checkpoint.control.pause_reason = control.get("pause_reason")

        checkpoint.original_settings = data.get("original_settings")

        return checkpoint


class CheckpointManager:
    """
    断点续传管理器

    负责检查点的保存、加载、验证和清理。
    使用原子性写入策略确保数据完整性。
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化断点续传管理器

        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)

    def save_checkpoint(
        self,
        job_dir: Path,
        data: Dict[str, Any],
        original_settings: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        原子性保存检查点

        使用"写临时文件 -> 重命名"策略，确保文件要么完整写入，要么保持原样。

        Args:
            job_dir: 任务目录
            data: 检查点数据
            original_settings: 原始设置（用于校验参数兼容性）

        Returns:
            bool: 保存是否成功
        """
        # 添加原始设置到 checkpoint（用于校验参数兼容性）
        if original_settings:
            data["original_settings"] = original_settings

        checkpoint_path = job_dir / "checkpoint.json"
        temp_path = checkpoint_path.with_suffix(".tmp")

        try:
            # 1. 写入临时文件
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 2. 原子替换（Windows/Linux/macOS 均支持）
            # 如果程序在这里崩溃，checkpoint.json 依然是旧版本，不会损坏
            os.replace(temp_path, checkpoint_path)

            self.logger.debug(f"检查点已保存: {checkpoint_path}")
            return True

        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}", exc_info=True)
            # 清理临时文件
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            return False

    def load_checkpoint(self, job_dir: Path) -> Optional[Dict[str, Any]]:
        """
        加载检查点

        Args:
            job_dir: 任务目录

        Returns:
            Optional[Dict[str, Any]]: 检查点数据，不存在或损坏则返回 None
        """
        checkpoint_path = job_dir / "checkpoint.json"

        if not checkpoint_path.exists():
            self.logger.debug(f"检查点不存在: {checkpoint_path}")
            return None

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.logger.debug(f"检查点已加载: {checkpoint_path}")
                return data

        except (json.JSONDecodeError, OSError) as e:
            self.logger.warning(f"检查点文件损坏，将重新开始任务: {checkpoint_path} - {e}")
            return None

    def checkpoint_exists(self, job_dir: Path) -> bool:
        """
        检查检查点是否存在

        Args:
            job_dir: 任务目录

        Returns:
            bool: 检查点是否存在
        """
        checkpoint_path = job_dir / "checkpoint.json"
        return checkpoint_path.exists()

    def delete_checkpoint(self, job_dir: Path) -> bool:
        """
        删除检查点

        Args:
            job_dir: 任务目录

        Returns:
            bool: 删除是否成功
        """
        checkpoint_path = job_dir / "checkpoint.json"

        if not checkpoint_path.exists():
            return True

        try:
            checkpoint_path.unlink()
            self.logger.debug(f"检查点已删除: {checkpoint_path}")
            return True

        except Exception as e:
            self.logger.error(f"删除检查点失败: {e}", exc_info=True)
            return False

    def validate_checkpoint(
        self,
        checkpoint_data: Dict[str, Any],
        current_settings: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        验证检查点数据的完整性和兼容性

        Args:
            checkpoint_data: 检查点数据
            current_settings: 当前设置（用于兼容性检查）

        Returns:
            tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        # 检查必需字段
        required_fields = ['job_id', 'phase', 'total_chunks', 'processed_chunks']
        for field in required_fields:
            if field not in checkpoint_data:
                return False, f"缺少必需字段: {field}"

        # 检查数据类型
        if not isinstance(checkpoint_data['total_chunks'], int):
            return False, "total_chunks 必须是整数"

        if not isinstance(checkpoint_data['processed_chunks'], int):
            return False, "processed_chunks 必须是整数"

        # 检查数据合理性
        if checkpoint_data['processed_chunks'] > checkpoint_data['total_chunks']:
            return False, "processed_chunks 不能大于 total_chunks"

        # 检查设置兼容性（如果提供了当前设置）
        if current_settings and 'original_settings' in checkpoint_data:
            original = checkpoint_data['original_settings']
            # 检查关键设置是否一致
            critical_settings = ['model', 'device', 'word_timestamps']
            for setting in critical_settings:
                if setting in original and setting in current_settings:
                    if original[setting] != current_settings[setting]:
                        return False, f"设置不兼容: {setting} 已更改"

        return True, None

    def get_checkpoint_info(self, job_dir: Path) -> Optional[Dict[str, Any]]:
        """
        获取检查点的摘要信息（不加载完整数据）

        Args:
            job_dir: 任务目录

        Returns:
            Optional[Dict[str, Any]]: 检查点摘要信息
        """
        checkpoint_data = self.load_checkpoint(job_dir)

        if not checkpoint_data:
            return None

        return {
            'job_id': checkpoint_data.get('job_id'),
            'phase': checkpoint_data.get('phase'),
            'total_chunks': checkpoint_data.get('total_chunks', 0),
            'processed_chunks': checkpoint_data.get('processed_chunks', 0),
            'progress': (
                checkpoint_data.get('processed_chunks', 0) /
                checkpoint_data.get('total_chunks', 1) * 100
                if checkpoint_data.get('total_chunks', 0) > 0 else 0
            ),
            'language': checkpoint_data.get('language'),
            'has_original_settings': 'original_settings' in checkpoint_data
        }

    def merge_checkpoint_data(
        self,
        old_data: Dict[str, Any],
        new_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        合并检查点数据（用于增量更新）

        Args:
            old_data: 旧的检查点数据
            new_data: 新的检查点数据

        Returns:
            Dict[str, Any]: 合并后的检查点数据
        """
        merged = old_data.copy()
        merged.update(new_data)

        # 合并 processed_indices（去重并排序）
        if 'processed_indices' in old_data and 'processed_indices' in new_data:
            old_indices = set(old_data['processed_indices'])
            new_indices = set(new_data['processed_indices'])
            merged['processed_indices'] = sorted(old_indices | new_indices)

        return merged

    # ==================== V3.7 新增方法 ====================

    def save_checkpoint_v37(
        self,
        job_dir: Path,
        checkpoint: CheckpointV37
    ) -> bool:
        """
        保存 V3.7 格式检查点

        Args:
            job_dir: 任务目录
            checkpoint: V3.7 检查点对象

        Returns:
            bool: 保存是否成功
        """
        return self.save_checkpoint(job_dir, checkpoint.to_dict())

    def load_checkpoint_v37(self, job_dir: Path) -> Optional[CheckpointV37]:
        """
        加载 V3.7 格式检查点

        自动处理版本兼容性。

        Args:
            job_dir: 任务目录

        Returns:
            Optional[CheckpointV37]: 检查点对象，不存在或损坏则返回 None
        """
        data = self.load_checkpoint(job_dir)

        if not data:
            return None

        # 检查版本
        version = data.get("version", "")
        if not version.startswith("3.7"):
            # 尝试从旧格式迁移
            self.logger.debug(f"检测到旧版本检查点 ({version})，尝试迁移到 V3.7")
            data = self._migrate_to_v37(data)

        return CheckpointV37.from_dict(data)

    def _migrate_to_v37(self, old_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        将旧版本检查点迁移到 V3.7 格式

        Args:
            old_data: 旧版本检查点数据

        Returns:
            Dict[str, Any]: V3.7 格式数据
        """
        now = datetime.utcnow().isoformat() + "Z"

        # 提取旧版本数据
        job_id = old_data.get("job_id", "")
        phase = old_data.get("phase", "pending")
        total_chunks = old_data.get("total_chunks", 0)
        processed_chunks = old_data.get("processed_chunks", 0)
        processed_indices = old_data.get("processed_indices", [])

        # 构建 V3.7 格式
        new_data = {
            "version": CHECKPOINT_VERSION,
            "job_id": job_id,
            "created_at": old_data.get("created_at", now),
            "updated_at": now,
            "phase": phase,
            "phase_status": "in_progress" if processed_chunks < total_chunks else "completed",
            "preprocessing": {
                "audio_extracted": True,
                "audio_path": None,
                "audio_duration_sec": 0.0,
                "sample_rate": 16000,
                "vad_completed": True,
                "total_chunks": total_chunks,
                "chunks_metadata": [],
                "spectral_triage": {
                    "completed": True,
                    "diagnosed_indices": list(range(total_chunks)),
                    "need_separation_count": 0
                },
                "separation": {
                    "mode": "on_demand",
                    "completed": True,
                    "separated_indices": [],
                    "global_separation_done": False
                }
            },
            "transcription": {
                "fast_worker": {
                    "processed_indices": processed_indices,
                    "completed_count": processed_chunks,
                    "fuse_upgraded_indices": []
                },
                "slow_worker": {
                    "processed_indices": processed_indices,
                    "completed_count": processed_chunks,
                    "context_state": {
                        "previous_whisper_text": "",
                        "last_processed_index": max(processed_indices) if processed_indices else -1
                    }
                },
                "alignment": {
                    "finalized_indices": processed_indices,
                    "completed_count": processed_chunks,
                    "alignment_levels": {}
                }
            },
            "output": {
                "srt_generated": False,
                "srt_path": None
            },
            "control": {
                "paused": False,
                "canceled": False,
                "pending_pause": False,
                "pending_cancel": False,
                "pause_reason": None
            },
            "original_settings": old_data.get("original_settings")
        }

        self.logger.debug(f"检查点迁移完成: {job_id}")
        return new_data

    def create_checkpoint_v37(self, job_id: str) -> CheckpointV37:
        """
        创建新的 V3.7 检查点

        Args:
            job_id: 任务ID

        Returns:
            CheckpointV37: 新检查点对象
        """
        now = datetime.utcnow().isoformat() + "Z"
        checkpoint = CheckpointV37()
        checkpoint.job_id = job_id
        checkpoint.created_at = now
        return checkpoint

    def update_preprocessing_state(
        self,
        job_dir: Path,
        audio_extracted: Optional[bool] = None,
        audio_path: Optional[str] = None,
        audio_duration_sec: Optional[float] = None,
        vad_completed: Optional[bool] = None,
        total_chunks: Optional[int] = None,
        chunks_metadata: Optional[List[Dict]] = None,
        spectral_completed: Optional[bool] = None,
        diagnosed_indices: Optional[List[int]] = None,
        separation_completed: Optional[bool] = None,
        separated_indices: Optional[List[int]] = None
    ) -> bool:
        """
        增量更新预处理状态

        只更新提供的字段。

        Args:
            job_dir: 任务目录
            其他参数: 要更新的字段

        Returns:
            bool: 更新是否成功
        """
        checkpoint = self.load_checkpoint_v37(job_dir)
        if not checkpoint:
            self.logger.warning(f"无法加载检查点进行更新: {job_dir}")
            return False

        # 更新预处理状态
        if audio_extracted is not None:
            checkpoint.preprocessing.audio_extracted = audio_extracted
        if audio_path is not None:
            checkpoint.preprocessing.audio_path = audio_path
        if audio_duration_sec is not None:
            checkpoint.preprocessing.audio_duration_sec = audio_duration_sec
        if vad_completed is not None:
            checkpoint.preprocessing.vad_completed = vad_completed
        if total_chunks is not None:
            checkpoint.preprocessing.total_chunks = total_chunks
        if chunks_metadata is not None:
            checkpoint.preprocessing.chunks_metadata = chunks_metadata
        if spectral_completed is not None:
            checkpoint.preprocessing.spectral_triage_completed = spectral_completed
        if diagnosed_indices is not None:
            checkpoint.preprocessing.diagnosed_indices = diagnosed_indices
        if separation_completed is not None:
            checkpoint.preprocessing.separation_completed = separation_completed
        if separated_indices is not None:
            checkpoint.preprocessing.separated_indices = separated_indices

        return self.save_checkpoint_v37(job_dir, checkpoint)

    def update_transcription_state(
        self,
        job_dir: Path,
        fast_processed_indices: Optional[List[int]] = None,
        slow_processed_indices: Optional[List[int]] = None,
        previous_whisper_text: Optional[str] = None,
        last_processed_index: Optional[int] = None,
        finalized_indices: Optional[List[int]] = None,
        alignment_level: Optional[Dict[int, str]] = None
    ) -> bool:
        """
        增量更新转录状态

        Args:
            job_dir: 任务目录
            fast_processed_indices: FastWorker 已处理索引
            slow_processed_indices: SlowWorker 已处理索引
            previous_whisper_text: Whisper 上文状态（关键）
            last_processed_index: 最后处理的索引
            finalized_indices: 已定稿索引
            alignment_level: 对齐级别映射

        Returns:
            bool: 更新是否成功
        """
        checkpoint = self.load_checkpoint_v37(job_dir)
        if not checkpoint:
            self.logger.warning(f"无法加载检查点进行更新: {job_dir}")
            return False

        # 更新 FastWorker 状态
        if fast_processed_indices is not None:
            checkpoint.transcription.fast_processed_indices = fast_processed_indices
            checkpoint.transcription.fast_completed_count = len(fast_processed_indices)

        # 更新 SlowWorker 状态（关键）
        if slow_processed_indices is not None:
            checkpoint.transcription.slow_processed_indices = slow_processed_indices
            checkpoint.transcription.slow_completed_count = len(slow_processed_indices)
        if previous_whisper_text is not None:
            checkpoint.transcription.previous_whisper_text = previous_whisper_text
        if last_processed_index is not None:
            checkpoint.transcription.last_processed_index = last_processed_index

        # 更新 Alignment 状态
        if finalized_indices is not None:
            checkpoint.transcription.finalized_indices = finalized_indices
            checkpoint.transcription.alignment_completed_count = len(finalized_indices)
        if alignment_level is not None:
            checkpoint.transcription.alignment_levels.update(alignment_level)

        return self.save_checkpoint_v37(job_dir, checkpoint)

    def get_resume_point(self, job_dir: Path) -> Dict[str, Any]:
        """
        获取恢复点信息

        分析检查点，返回应该从哪里恢复。

        Args:
            job_dir: 任务目录

        Returns:
            Dict[str, Any]: 恢复点信息
                - phase: 应恢复的阶段
                - start_index: 起始索引
                - context: 上下文数据（如 Whisper 上文）
        """
        checkpoint = self.load_checkpoint_v37(job_dir)

        if not checkpoint:
            return {
                "phase": "start",
                "start_index": 0,
                "context": None,
                "message": "无检查点，从头开始"
            }

        prep = checkpoint.preprocessing
        trans = checkpoint.transcription

        # 1. 检查音频提取
        if not prep.audio_extracted:
            return {
                "phase": "extract",
                "start_index": 0,
                "context": None,
                "message": "从音频提取阶段恢复"
            }

        # 2. 检查 VAD 切分
        if not prep.vad_completed:
            return {
                "phase": "vad",
                "start_index": 0,
                "context": None,
                "message": "从 VAD 切分阶段恢复"
            }

        # 3. 检查频谱分诊
        if not prep.spectral_triage_completed:
            diagnosed = set(prep.diagnosed_indices)
            return {
                "phase": "spectral_triage",
                "start_index": len(diagnosed),
                "skip_indices": diagnosed,
                "context": None,
                "message": f"从频谱分诊恢复，已诊断 {len(diagnosed)} 个"
            }

        # 4. 检查人声分离
        if not prep.separation_completed:
            separated = set(prep.separated_indices)
            return {
                "phase": "separation",
                "start_index": len(separated),
                "skip_indices": separated,
                "context": None,
                "message": f"从人声分离恢复，已分离 {len(separated)} 个"
            }

        total = prep.total_chunks

        # 5. 检查 FastWorker
        fast_done = set(trans.fast_processed_indices)
        if len(fast_done) < total:
            return {
                "phase": "fast_worker",
                "start_index": len(fast_done),
                "skip_indices": fast_done,
                "context": None,
                "message": f"从 FastWorker 恢复，已处理 {len(fast_done)}/{total}"
            }

        # 6. 检查 SlowWorker（关键：恢复上文）
        slow_done = set(trans.slow_processed_indices)
        if len(slow_done) < total:
            return {
                "phase": "slow_worker",
                "start_index": len(slow_done),
                "skip_indices": slow_done,
                "context": {
                    "previous_whisper_text": trans.previous_whisper_text,
                    "last_processed_index": trans.last_processed_index
                },
                "message": f"从 SlowWorker 恢复，已处理 {len(slow_done)}/{total}，上文已恢复"
            }

        # 7. 检查对齐
        aligned = set(trans.finalized_indices)
        if len(aligned) < total:
            return {
                "phase": "alignment",
                "start_index": len(aligned),
                "skip_indices": aligned,
                "context": None,
                "message": f"从对齐阶段恢复，已对齐 {len(aligned)}/{total}"
            }

        # 8. 检查 SRT 生成
        if not checkpoint.srt_generated:
            return {
                "phase": "srt",
                "start_index": 0,
                "context": None,
                "message": "从 SRT 生成阶段恢复"
            }

        return {
            "phase": "complete",
            "start_index": 0,
            "context": None,
            "message": "任务已完成"
        }

    def validate_checkpoint_v37(
        self,
        checkpoint: CheckpointV37,
        current_settings: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, Optional[str]]:
        """
        验证 V3.7 检查点的完整性和兼容性

        Args:
            checkpoint: 检查点对象
            current_settings: 当前设置

        Returns:
            tuple[bool, Optional[str]]: (是否有效, 错误信息)
        """
        # 检查版本
        if not checkpoint.version.startswith("3.7"):
            return False, f"不支持的检查点版本: {checkpoint.version}"

        # 检查 job_id
        if not checkpoint.job_id:
            return False, "缺少 job_id"

        # 检查设置兼容性
        if current_settings and checkpoint.original_settings:
            critical_settings = ['model', 'device', 'word_timestamps']
            for setting in critical_settings:
                orig = checkpoint.original_settings.get(setting)
                curr = current_settings.get(setting)
                if orig is not None and curr is not None and orig != curr:
                    return False, f"设置不兼容: {setting} 从 {orig} 变为 {curr}"

        return True, None


# 便捷函数
def get_checkpoint_manager(logger: Optional[logging.Logger] = None) -> CheckpointManager:
    """
    获取 CheckpointManager 实例

    Args:
        logger: 日志记录器

    Returns:
        CheckpointManager 实例
    """
    return CheckpointManager(logger=logger)


def create_checkpoint_v37(job_id: str) -> CheckpointV37:
    """
    创建新的 V3.7 检查点

    Args:
        job_id: 任务ID

    Returns:
        CheckpointV37 实例
    """
    now = datetime.utcnow().isoformat() + "Z"
    checkpoint = CheckpointV37()
    checkpoint.job_id = job_id
    checkpoint.created_at = now
    return checkpoint


class CheckpointManagerV37:
    """
    V3.7 检查点管理器（简化包装）

    提供更便捷的 API 用于流水线中的检查点保存和加载。
    与 CancellationToken 的 check_and_save 方法配合使用。
    """

    def __init__(self, job_dir: Path, logger: Optional[logging.Logger] = None):
        """
        初始化 V3.7 检查点管理器

        Args:
            job_dir: 任务目录
            logger: 日志记录器
        """
        self.job_dir = Path(job_dir) if isinstance(job_dir, str) else job_dir
        self.logger = logger or logging.getLogger(__name__)
        self._manager = CheckpointManager(logger=self.logger)
        self._checkpoint: Optional[CheckpointV37] = None

    def load_checkpoint(self) -> Optional[CheckpointV37]:
        """
        加载检查点

        Returns:
            Optional[CheckpointV37]: 检查点对象，不存在则返回 None
        """
        self._checkpoint = self._manager.load_checkpoint_v37(self.job_dir)
        return self._checkpoint

    def save_checkpoint(self, checkpoint_data: Dict[str, Any]) -> bool:
        """
        保存检查点数据（增量更新）

        支持以下字段结构：
        - preprocessing: 预处理状态
        - transcription: 转录状态
        - output: 输出状态
        - control: 控制状态

        Args:
            checkpoint_data: 要保存的检查点数据

        Returns:
            bool: 保存是否成功
        """
        # 加载现有检查点或创建新的
        if self._checkpoint is None:
            self._checkpoint = self._manager.load_checkpoint_v37(self.job_dir)
            if self._checkpoint is None:
                self._checkpoint = create_checkpoint_v37("")
                # 尝试从 job_dir 推断 job_id
                self._checkpoint.job_id = self.job_dir.name

        # 更新预处理状态
        if "preprocessing" in checkpoint_data:
            prep_data = checkpoint_data["preprocessing"]
            prep = self._checkpoint.preprocessing

            # 音频提取
            if "audio_extracted" in prep_data:
                prep.audio_extracted = prep_data["audio_extracted"]
            if "audio_path" in prep_data:
                prep.audio_path = prep_data["audio_path"]
            if "vad_completed" in prep_data:
                prep.vad_completed = prep_data["vad_completed"]
            if "total_chunks" in prep_data:
                prep.total_chunks = prep_data["total_chunks"]
            # V3.1.0: 保存 chunks_metadata（用于跳过 VAD 恢复）
            if "chunks_metadata" in prep_data:
                prep.chunks_metadata = prep_data["chunks_metadata"]

        # 更新频谱分诊状态
        if "spectral_triage" in checkpoint_data:
            triage_data = checkpoint_data["spectral_triage"]
            prep = self._checkpoint.preprocessing

            if "completed" in triage_data:
                prep.spectral_triage_completed = triage_data["completed"]
            if "diagnosed_indices" in triage_data:
                prep.diagnosed_indices = triage_data["diagnosed_indices"]
            if "need_separation_count" in triage_data:
                prep.need_separation_count = triage_data["need_separation_count"]

        # 更新分离状态
        if "separation" in checkpoint_data:
            sep_data = checkpoint_data["separation"]
            prep = self._checkpoint.preprocessing

            if "mode" in sep_data:
                prep.separation_mode = sep_data["mode"]
            if "completed" in sep_data:
                prep.separation_completed = sep_data["completed"]
            if "separated_indices" in sep_data:
                prep.separated_indices = sep_data["separated_indices"]
            if "global_separation_done" in sep_data:
                prep.global_separation_done = sep_data["global_separation_done"]

        # 更新转录状态
        if "transcription" in checkpoint_data:
            trans_data = checkpoint_data["transcription"]
            trans = self._checkpoint.transcription
            prep = self._checkpoint.preprocessing

            # 如果有转录数据，意味着预处理已完成
            # 确保预处理状态被正确设置（用于恢复时的阶段判断）
            if not prep.audio_extracted:
                prep.audio_extracted = True
            if not prep.vad_completed:
                prep.vad_completed = True
            if not prep.spectral_triage_completed:
                prep.spectral_triage_completed = True
            if not prep.separation_completed:
                prep.separation_completed = True

            # 从转录数据推断 total_chunks
            if prep.total_chunks == 0:
                if "total_chunks" in trans_data:
                    prep.total_chunks = trans_data["total_chunks"]

            # 通用字段
            if "mode" in trans_data:
                pass  # 仅记录模式，不影响状态
            if "processed_indices" in trans_data:
                trans.fast_processed_indices = trans_data["processed_indices"]
                trans.fast_completed_count = len(trans_data["processed_indices"])
            if "processed_count" in trans_data:
                trans.fast_completed_count = trans_data["processed_count"]
            if "total_chunks" in trans_data:
                pass  # 已在预处理中记录

            # FastWorker 字段
            if "fast_processed_indices" in trans_data:
                trans.fast_processed_indices = trans_data["fast_processed_indices"]
                trans.fast_completed_count = len(trans_data["fast_processed_indices"])
            if "fast_processed_count" in trans_data:
                trans.fast_completed_count = trans_data["fast_processed_count"]

            # SlowWorker 字段（关键：previous_whisper_text）
            if "slow_processed_count" in trans_data:
                trans.slow_completed_count = trans_data["slow_processed_count"]
            if "slow_processed_indices" in trans_data:
                trans.slow_processed_indices = trans_data["slow_processed_indices"]  # V3.1.0
            if "previous_whisper_text" in trans_data:
                trans.previous_whisper_text = trans_data["previous_whisper_text"]
            if "last_slow_chunk_index" in trans_data:
                trans.last_processed_index = trans_data["last_slow_chunk_index"]

            # AlignmentWorker 字段
            if "align_processed_count" in trans_data:
                trans.alignment_completed_count = trans_data["align_processed_count"]
            if "last_align_chunk_index" in trans_data:
                pass  # 可选记录
            if "completed_chunks" in trans_data:
                trans.alignment_completed_count = trans_data["completed_chunks"]

            # V3.1.0: 字幕快照字段（实时持久化核心）
            if "sentences_snapshot" in trans_data:
                trans.sentences_snapshot = trans_data["sentences_snapshot"]
            if "sentence_count" in trans_data:
                trans.sentence_count = trans_data["sentence_count"]
            if "chunk_sentences_map" in trans_data:
                # JSON 键是 str，需要转换为 int
                raw_map = trans_data["chunk_sentences_map"]
                trans.chunk_sentences_map = {
                    int(k): v for k, v in raw_map.items()
                } if raw_map else {}
            # 支持 finalized_indices 字段（用于恢复点计算）
            if "finalized_indices" in trans_data:
                trans.finalized_indices = trans_data["finalized_indices"]
                trans.alignment_completed_count = len(trans_data["finalized_indices"])

        # 更新控制状态
        if "control" in checkpoint_data:
            ctrl_data = checkpoint_data["control"]
            ctrl = self._checkpoint.control

            if "paused" in ctrl_data:
                ctrl.paused = ctrl_data["paused"]
            if "canceled" in ctrl_data:
                ctrl.canceled = ctrl_data["canceled"]

        # 更新阶段
        self._checkpoint.phase = self._determine_current_phase()
        self._checkpoint.phase_status = "in_progress"

        # 保存
        return self._manager.save_checkpoint_v37(self.job_dir, self._checkpoint)

    def _determine_current_phase(self) -> str:
        """根据当前状态确定阶段"""
        if self._checkpoint is None:
            return "pending"

        prep = self._checkpoint.preprocessing
        trans = self._checkpoint.transcription

        if not prep.audio_extracted:
            return "extract"
        if not prep.vad_completed:
            return "vad"
        if not prep.spectral_triage_completed:
            return "spectral_triage"
        if not prep.separation_completed:
            return "separation"
        if trans.fast_completed_count < prep.total_chunks:
            return "fast_worker"
        if trans.slow_completed_count < prep.total_chunks:
            return "slow_worker"
        if trans.alignment_completed_count < prep.total_chunks:
            return "alignment"
        if not self._checkpoint.srt_generated:
            return "srt"
        return "complete"

    def delete_checkpoint(self) -> bool:
        """删除检查点"""
        self._checkpoint = None
        return self._manager.delete_checkpoint(self.job_dir)

    def get_current_phase(self) -> str:
        """获取当前阶段"""
        if self._checkpoint:
            return self._checkpoint.phase
        return "pending"

    @property
    def current_phase(self) -> str:
        """当前阶段属性"""
        return self.get_current_phase()
