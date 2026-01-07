"""
智能进度追踪系统

核心特性：
1. 根据预设配置动态调整各阶段权重
2. 支持流式更新（SV、Whisper、LLM 各阶段独立更新）
3. 统一 SSE 事件 tag 设计
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class ProcessPhase(Enum):
    """处理阶段枚举"""
    PENDING = "pending"
    EXTRACT = "extract"
    BGM_DETECT = "bgm_detect"
    SPECTRUM_ANALYSIS = "spectrum_analysis"  # 频谱分诊
    DEMUCS = "demucs"
    VAD = "vad"
    SENSEVOICE = "sensevoice"
    WHISPER_PATCH = "whisper"
    LLM_PROOF = "llm_proof"
    LLM_TRANS = "llm_trans"
    SRT = "srt"
    COMPLETE = "complete"


@dataclass
class PhaseProgress:
    """阶段进度"""
    phase: ProcessPhase
    weight: float
    total_items: int = 0
    completed_items: int = 0
    is_active: bool = False
    message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def get_progress_ratio(self) -> float:
        """获取阶段进度比例"""
        if self.total_items <= 0:
            return 0.0
        return min(1.0, self.completed_items / self.total_items)

    def get_elapsed_time(self) -> float:
        """获取已用时间"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


@dataclass
class PresetWeights:
    """预设权重配置"""
    extract: float = 5
    bgm_detect: float = 2
    spectrum_analysis: float = 3
    demucs: float = 8
    vad: float = 5
    sensevoice: float = 40
    whisper: float = 0
    llm_proof: float = 0
    llm_trans: float = 0
    srt: float = 10

    @classmethod
    def from_preset(cls, preset_id: str) -> 'PresetWeights':
        """根据预设ID生成权重配置"""
        presets = {
            # 极速模式：仅 SenseVoice
            'default': cls(sensevoice=50, srt=10),

            # 智能补刀：SV + Whisper 局部
            'preset1': cls(sensevoice=35, whisper=20, srt=10),

            # 轻度校对：智能补刀 + LLM 按需校对
            'preset2': cls(sensevoice=30, whisper=15, llm_proof=15, srt=10),

            # 深度校对：智能补刀 + LLM 全文精修
            'preset3': cls(sensevoice=25, whisper=15, llm_proof=25, srt=10),

            # 校对+翻译：深度校对 + 全文翻译
            'preset4': cls(sensevoice=20, whisper=10, llm_proof=20, llm_trans=15, srt=10),

            # 校对+重点翻译：深度校对 + 部分翻译
            'preset5': cls(sensevoice=22, whisper=12, llm_proof=20, llm_trans=8, srt=10),
        }
        return presets.get(preset_id, cls())

    def total_weight(self) -> float:
        """计算总权重"""
        return (
            self.extract + self.bgm_detect + self.spectrum_analysis +
            self.demucs + self.vad + self.sensevoice + self.whisper +
            self.llm_proof + self.llm_trans + self.srt
        )

    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            "extract": self.extract,
            "bgm_detect": self.bgm_detect,
            "spectrum_analysis": self.spectrum_analysis,
            "demucs": self.demucs,
            "vad": self.vad,
            "sensevoice": self.sensevoice,
            "whisper": self.whisper,
            "llm_proof": self.llm_proof,
            "llm_trans": self.llm_trans,
            "srt": self.srt
        }


class ProgressTracker:
    """智能进度追踪器"""

    def __init__(self, job_id: str, preset_id: str = 'default'):
        self.job_id = job_id
        self.preset_id = preset_id
        self.weights = PresetWeights.from_preset(preset_id)
        self.total_weight = self.weights.total_weight()
        self.phases: Dict[ProcessPhase, PhaseProgress] = {}
        self._init_phases()
        self.completed_weight = 0.0
        self.current_phase = ProcessPhase.PENDING
        self.start_time = time.time()
        self.error_message: Optional[str] = None

    def _init_phases(self):
        """初始化各阶段"""
        weight_map = {
            ProcessPhase.EXTRACT: self.weights.extract,
            ProcessPhase.BGM_DETECT: self.weights.bgm_detect,
            ProcessPhase.SPECTRUM_ANALYSIS: self.weights.spectrum_analysis,
            ProcessPhase.DEMUCS: self.weights.demucs,
            ProcessPhase.VAD: self.weights.vad,
            ProcessPhase.SENSEVOICE: self.weights.sensevoice,
            ProcessPhase.WHISPER_PATCH: self.weights.whisper,
            ProcessPhase.LLM_PROOF: self.weights.llm_proof,
            ProcessPhase.LLM_TRANS: self.weights.llm_trans,
            ProcessPhase.SRT: self.weights.srt,
        }
        for phase, weight in weight_map.items():
            self.phases[phase] = PhaseProgress(phase=phase, weight=weight)

    def start_phase(self, phase: ProcessPhase, total_items: int = 1, message: str = ""):
        """开始某个阶段"""
        if phase in self.phases:
            pp = self.phases[phase]
            pp.is_active = True
            pp.total_items = total_items
            pp.completed_items = 0
            pp.message = message
            pp.start_time = time.time()
            self.current_phase = phase
            logger.info(f"[{self.job_id}] 开始阶段: {phase.value} (共 {total_items} 项)")

    def resume_phase(self, phase: ProcessPhase, total_items: int, completed_items: int, message: str = ""):
        """
        V3.1.0: 恢复某个阶段（用于断点续传）

        与 start_phase 的区别：
        - start_phase: 从 0 开始，completed_items = 0
        - resume_phase: 从已完成数量开始，completed_items = 已完成数量

        Args:
            phase: 阶段枚举
            total_items: 总项目数
            completed_items: 已完成项目数
            message: 进度消息
        """
        if phase in self.phases:
            pp = self.phases[phase]
            pp.is_active = True
            pp.total_items = total_items
            pp.completed_items = completed_items  # 保留已完成数量
            pp.message = message
            pp.start_time = time.time()
            self.current_phase = phase
            logger.info(
                f"[{self.job_id}] 恢复阶段: {phase.value} "
                f"(已完成 {completed_items}/{total_items} 项)"
            )

    def update_phase(
        self,
        phase: ProcessPhase,
        completed: int = None,
        increment: int = None,
        message: str = None
    ):
        """更新阶段进度"""
        if phase not in self.phases:
            return

        pp = self.phases[phase]

        if completed is not None:
            pp.completed_items = completed
        elif increment is not None:
            pp.completed_items += increment

        if message is not None:
            pp.message = message

        logger.debug(
            f"[{self.job_id}] 更新阶段: {phase.value} "
            f"({pp.completed_items}/{pp.total_items})"
        )

    def complete_phase(self, phase: ProcessPhase, message: str = None):
        """完成某个阶段"""
        if phase in self.phases:
            pp = self.phases[phase]
            pp.is_active = False
            pp.completed_items = pp.total_items
            pp.end_time = time.time()
            if message:
                pp.message = message
            self.completed_weight += pp.weight
            logger.info(
                f"[{self.job_id}] 完成阶段: {phase.value} "
                f"(耗时 {pp.get_elapsed_time():.1f}s)"
            )

    def skip_phase(self, phase: ProcessPhase, reason: str = ""):
        """跳过某个阶段"""
        if phase in self.phases:
            pp = self.phases[phase]
            pp.is_active = False
            pp.completed_items = pp.total_items
            pp.message = f"跳过: {reason}" if reason else "跳过"
            self.completed_weight += pp.weight
            logger.info(f"[{self.job_id}] 跳过阶段: {phase.value} ({reason})")

    def set_error(self, error_message: str):
        """设置错误信息"""
        self.error_message = error_message
        logger.error(f"[{self.job_id}] 错误: {error_message}")

    def get_overall_progress(self) -> float:
        """获取总体进度百分比"""
        progress = self.completed_weight

        # 加上当前阶段的部分进度
        if self.current_phase in self.phases:
            pp = self.phases[self.current_phase]
            if pp.is_active and pp.total_items > 0:
                phase_progress = pp.completed_items / pp.total_items
                progress += pp.weight * phase_progress

        return round((progress / self.total_weight) * 100, 1)

    def get_elapsed_time(self) -> float:
        """获取总耗时"""
        return time.time() - self.start_time

    def to_sse_data(self) -> dict:
        """生成 SSE 推送数据"""
        current_pp = self.phases.get(self.current_phase)
        return {
            "job_id": self.job_id,
            "phase": self.current_phase.value,
            "percent": self.get_overall_progress(),
            "message": current_pp.message if current_pp else "",
            "elapsed_time": round(self.get_elapsed_time(), 1),
            "error": self.error_message
        }

    def to_dict(self) -> dict:
        """转换为完整字典"""
        return {
            "job_id": self.job_id,
            "preset_id": self.preset_id,
            "current_phase": self.current_phase.value,
            "overall_progress": self.get_overall_progress(),
            "elapsed_time": round(self.get_elapsed_time(), 1),
            "error": self.error_message,
            "phases": {
                phase.value: {
                    "weight": pp.weight,
                    "total_items": pp.total_items,
                    "completed_items": pp.completed_items,
                    "is_active": pp.is_active,
                    "message": pp.message,
                    "elapsed_time": round(pp.get_elapsed_time(), 1)
                }
                for phase, pp in self.phases.items()
            }
        }


# 单例工厂
_tracker_instances: Dict[str, ProgressTracker] = {}


def get_progress_tracker(job_id: str, preset_id: str = None) -> ProgressTracker:
    """获取或创建进度追踪器"""
    global _tracker_instances
    if job_id not in _tracker_instances:
        if preset_id is None:
            preset_id = 'default'
        _tracker_instances[job_id] = ProgressTracker(job_id, preset_id)
    return _tracker_instances[job_id]


def remove_progress_tracker(job_id: str):
    """移除进度追踪器"""
    global _tracker_instances
    if job_id in _tracker_instances:
        del _tracker_instances[job_id]
        logger.debug(f"移除进度追踪器: {job_id}")


def get_all_trackers() -> Dict[str, ProgressTracker]:
    """获取所有进度追踪器"""
    return _tracker_instances.copy()
