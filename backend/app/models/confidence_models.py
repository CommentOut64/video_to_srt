"""
置信度相关数据模型

用于双流对齐架构中的置信度追踪和对齐结果表示
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from app.services.alignment.gap_resolver import GapResolution


class AlignmentStatus(Enum):
    """对齐状态"""
    MATCHED = "matched"           # 完美匹配
    SUBSTITUTED = "substituted"   # Whisper 纠错替换
    INSERTED = "inserted"         # SenseVoice 漏字，Whisper 插入
    DELETED = "deleted"           # SenseVoice 幻觉，已删除
    PSEUDO = "pseudo"             # 伪对齐（均匀分布）


class ConfidenceLevel(Enum):
    """置信度等级"""
    HIGH = "high"         # >= 0.8
    MEDIUM = "medium"     # 0.6 - 0.8
    LOW = "low"           # 0.4 - 0.6
    VERY_LOW = "very_low" # < 0.4

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """根据分数获取置信度等级"""
        if score >= 0.8:
            return cls.HIGH
        elif score >= 0.6:
            return cls.MEDIUM
        elif score >= 0.4:
            return cls.LOW
        else:
            return cls.VERY_LOW


@dataclass
class AlignedWord:
    """对齐后的字级时间戳（双流对齐版）"""
    word: str
    start: float
    end: float

    # 置信度信息
    sv_confidence: float = 1.0        # SenseVoice 原始置信度
    whisper_confidence: float = 1.0   # Whisper 置信度（如有）
    final_confidence: float = 1.0     # 最终综合置信度

    # 对齐状态
    alignment_status: AlignmentStatus = field(default=AlignmentStatus.MATCHED)
    is_pseudo: bool = False           # 是否为伪对齐生成

    # 来源追踪
    sv_original: Optional[str] = None      # SenseVoice 原始文本
    whisper_original: Optional[str] = None # Whisper 原始文本

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "sv_confidence": self.sv_confidence,
            "whisper_confidence": self.whisper_confidence,
            "final_confidence": self.final_confidence,
            "alignment_status": self.alignment_status.value,
            "is_pseudo": self.is_pseudo,
            "sv_original": self.sv_original,
            "whisper_original": self.whisper_original
        }

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """获取置信度等级"""
        return ConfidenceLevel.from_score(self.final_confidence)

    @property
    def duration(self) -> float:
        """获取持续时间"""
        return self.end - self.start


@dataclass
class AlignedSubtitle:
    """对齐后的字幕段（双流对齐版）"""
    text: str                                    # 最终文本
    start: float                                 # 开始时间
    end: float                                   # 结束时间
    words: List[AlignedWord] = field(default_factory=list)

    # 置信度信息
    avg_confidence: float = 1.0                  # 平均置信度
    min_confidence: float = 1.0                  # 最低置信度

    # 对齐质量
    alignment_score: float = 1.0                 # 对齐质量分数 (0-1)
    matched_ratio: float = 1.0                   # 匹配比例
    coverage: float = 1.0                        # 对齐覆盖率
    gap_ratio: float = 0.0                       # Gap 比例
    gap_resolution: Optional["GapResolution"] = None  # Gap 处理策略
    gap_positions: List[int] = field(default_factory=list)  # Gap 聚集位置

    # 来源信息
    sv_text: Optional[str] = None                # SenseVoice 原始文本
    whisper_text: Optional[str] = None           # Whisper 原始文本

    # VAD 信息
    vad_start: Optional[float] = None            # VAD 检测的开始时间
    vad_end: Optional[float] = None              # VAD 检测的结束时间

    # 状态标记
    is_draft: bool = False                       # 是否为草稿（快流）
    is_finalized: bool = False                   # 是否已定稿（慢流完成）
    needs_review: bool = False                   # 是否需要人工审核

    # 语义分组（继承自 SentenceSegment）
    group_id: Optional[str] = None
    is_soft_break: bool = False
    group_position: Optional[str] = None

    def compute_statistics(self):
        """计算统计信息"""
        if not self.words:
            return

        confidences = [w.final_confidence for w in self.words]
        self.avg_confidence = sum(confidences) / len(confidences)
        self.min_confidence = min(confidences)

        # 计算匹配比例
        matched_count = sum(1 for w in self.words
                          if w.alignment_status == AlignmentStatus.MATCHED)
        self.matched_ratio = matched_count / len(self.words)

        # 判断是否需要审核
        self.needs_review = (
            self.min_confidence < 0.4 or
            self.matched_ratio < 0.5
        )

    def to_dict(self) -> Dict:
        """转换为字典格式（用于 SSE 推送）"""
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "words": [w.to_dict() for w in self.words],
            "avg_confidence": self.avg_confidence,
            "min_confidence": self.min_confidence,
            "alignment_score": self.alignment_score,
            "matched_ratio": self.matched_ratio,
            "coverage": self.coverage,
            "gap_ratio": self.gap_ratio,
            "gap_resolution": self.gap_resolution.value if self.gap_resolution else None,
            "gap_positions": self.gap_positions,
            "sv_text": self.sv_text,
            "whisper_text": self.whisper_text,
            "vad_start": self.vad_start,
            "vad_end": self.vad_end,
            "is_draft": self.is_draft,
            "is_finalized": self.is_finalized,
            "needs_review": self.needs_review,
            "group_id": self.group_id,
            "is_soft_break": self.is_soft_break,
            "group_position": self.group_position
        }

    def to_srt_entry(self, index: int) -> str:
        """转换为 SRT 格式条目"""
        def format_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        return f"{index}\n{format_time(self.start)} --> {format_time(self.end)}\n{self.text}\n"

    @property
    def duration(self) -> float:
        """获取持续时间"""
        return self.end - self.start

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """获取置信度等级"""
        return ConfidenceLevel.from_score(self.avg_confidence)


@dataclass
class AlignmentResult:
    """对齐算法输出结果"""
    subtitles: List[AlignedSubtitle] = field(default_factory=list)

    # 整体统计
    total_duration: float = 0.0
    total_words: int = 0
    avg_alignment_score: float = 0.0

    # 对齐统计
    matched_count: int = 0
    substituted_count: int = 0
    inserted_count: int = 0
    deleted_count: int = 0

    # 质量指标
    overall_confidence: float = 0.0
    low_confidence_count: int = 0
    needs_review_count: int = 0

    def add_subtitle(self, subtitle: AlignedSubtitle):
        """添加字幕段"""
        self.subtitles.append(subtitle)
        self._update_statistics(subtitle)

    def _update_statistics(self, subtitle: AlignedSubtitle):
        """更新统计信息"""
        self.total_words += len(subtitle.words)

        for word in subtitle.words:
            if word.alignment_status == AlignmentStatus.MATCHED:
                self.matched_count += 1
            elif word.alignment_status == AlignmentStatus.SUBSTITUTED:
                self.substituted_count += 1
            elif word.alignment_status == AlignmentStatus.INSERTED:
                self.inserted_count += 1
            elif word.alignment_status == AlignmentStatus.DELETED:
                self.deleted_count += 1

        if subtitle.needs_review:
            self.needs_review_count += 1

        if subtitle.avg_confidence < 0.6:
            self.low_confidence_count += 1

    def finalize(self):
        """完成统计计算"""
        if not self.subtitles:
            return

        # 计算总时长
        self.total_duration = max(s.end for s in self.subtitles) if self.subtitles else 0.0

        # 计算平均对齐分数
        scores = [s.alignment_score for s in self.subtitles]
        self.avg_alignment_score = sum(scores) / len(scores) if scores else 0.0

        # 计算整体置信度
        confidences = [s.avg_confidence for s in self.subtitles]
        self.overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "subtitles": [s.to_dict() for s in self.subtitles],
            "statistics": {
                "total_duration": self.total_duration,
                "total_words": self.total_words,
                "avg_alignment_score": self.avg_alignment_score,
                "matched_count": self.matched_count,
                "substituted_count": self.substituted_count,
                "inserted_count": self.inserted_count,
                "deleted_count": self.deleted_count,
                "overall_confidence": self.overall_confidence,
                "low_confidence_count": self.low_confidence_count,
                "needs_review_count": self.needs_review_count
            }
        }

    def to_srt(self) -> str:
        """转换为完整 SRT 格式"""
        entries = []
        for i, subtitle in enumerate(self.subtitles, 1):
            entries.append(subtitle.to_srt_entry(i))
        return "\n".join(entries)
