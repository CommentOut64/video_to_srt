"""
对齐日志数据模型

用于记录对齐阶段的详细对齐过程
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class AlignmentStage(Enum):
    """对齐阶段"""
    EARLY_INTERCEPTION = "early_interception"  # 早期拦截
    DUAL_MODAL = "dual_modal"                  # 双模态对齐
    WHISPER_PSEUDO = "whisper_pseudo"          # Whisper 伪对齐
    SENSEVOICE_ONLY = "sensevoice_only"        # SenseVoice 草稿


class AlignmentResult(Enum):
    """对齐结果"""
    SUCCESS = "success"          # 成功
    FAILED = "failed"            # 失败
    SKIPPED = "skipped"          # 跳过
    FALLBACK = "fallback"        # 降级


@dataclass
class StageLog:
    """单个阶段的日志"""
    stage: str                           # 阶段名称
    result: str                          # 结果（success/failed/skipped/fallback）
    reason: Optional[str] = None         # 原因说明
    details: Dict[str, Any] = field(default_factory=dict)  # 详细信息
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class InputData:
    """输入数据信息"""
    whisper_text: str                    # Whisper 文本
    whisper_text_length: int             # Whisper 文本长度
    whisper_confidence: float            # Whisper 置信度
    whisper_language: str                # Whisper 检测到的语言
    sensevoice_text: str                 # SenseVoice 文本
    sensevoice_text_length: int          # SenseVoice 文本长度
    sensevoice_words_count: int          # SenseVoice 字级时间戳数量


@dataclass
class OutputData:
    """输出数据信息"""
    final_alignment_level: str           # 最终使用的对齐级别
    sentences_count: int                 # 句子数量
    total_duration: float                # 总时长（秒）
    sentences: List[Dict[str, Any]]      # 句子列表（简化版）


@dataclass
class AlignmentLog:
    """
    对齐日志（单个 Chunk）

    记录完整的对齐过程，包括：
    - Chunk 基本信息
    - 输入数据
    - 各阶段尝试过程
    - 输出数据
    """
    # Chunk 基本信息
    chunk_index: int                     # Chunk 索引
    chunk_start: float                   # Chunk 开始时间（秒）
    chunk_end: float                     # Chunk 结束时间（秒）
    chunk_duration: float                # Chunk 时长（秒）

    # 输入数据
    input_data: InputData

    # 阶段日志
    stages: List[StageLog] = field(default_factory=list)

    # 输出数据
    output_data: Optional[OutputData] = None

    # 元信息
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_stage(self, stage: AlignmentStage, result: AlignmentResult,
                  reason: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """添加阶段日志"""
        stage_log = StageLog(
            stage=stage.value,
            result=result.value,
            reason=reason,
            details=details or {}
        )
        self.stages.append(stage_log)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 JSON 序列化）"""
        return {
            "chunk_index": self.chunk_index,
            "chunk_start": self.chunk_start,
            "chunk_end": self.chunk_end,
            "chunk_duration": self.chunk_duration,
            "input_data": asdict(self.input_data),
            "stages": [asdict(stage) for stage in self.stages],
            "output_data": asdict(self.output_data) if self.output_data else None,
            "timestamp": self.timestamp
        }


@dataclass
class AlignmentLogSummary:
    """
    对齐日志汇总（整个任务）

    包含所有 Chunk 的对齐日志和统计信息
    """
    job_id: str
    total_chunks: int
    chunks: List[AlignmentLog] = field(default_factory=list)

    # 统计信息
    stats: Dict[str, Any] = field(default_factory=dict)

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_chunk_log(self, log: AlignmentLog):
        """添加 Chunk 日志"""
        self.chunks.append(log)

    def calculate_stats(self):
        """计算统计信息"""
        if not self.chunks:
            return

        # 统计各对齐级别的使用次数
        level_counts = {}
        for chunk_log in self.chunks:
            if chunk_log.output_data:
                level = chunk_log.output_data.final_alignment_level
                level_counts[level] = level_counts.get(level, 0) + 1

        # 统计各阶段的成功/失败次数
        stage_stats = {}
        for chunk_log in self.chunks:
            for stage in chunk_log.stages:
                stage_name = stage.stage
                if stage_name not in stage_stats:
                    stage_stats[stage_name] = {"success": 0, "failed": 0, "skipped": 0, "fallback": 0}
                stage_stats[stage_name][stage.result] = stage_stats[stage_name].get(stage.result, 0) + 1

        # 计算总时长
        total_duration = sum(chunk_log.chunk_duration for chunk_log in self.chunks)

        # 计算总句子数
        total_sentences = sum(
            chunk_log.output_data.sentences_count
            for chunk_log in self.chunks
            if chunk_log.output_data
        )

        self.stats = {
            "total_chunks": len(self.chunks),
            "total_duration": total_duration,
            "total_sentences": total_sentences,
            "alignment_level_distribution": level_counts,
            "stage_statistics": stage_stats
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 JSON 序列化）"""
        return {
            "job_id": self.job_id,
            "total_chunks": self.total_chunks,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "stats": self.stats,
            "timestamp": self.timestamp
        }
