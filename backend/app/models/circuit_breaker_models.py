"""
熔断机制数据模型（v2.1 概念重构版）

熔断 = 升级分离模型（转录过程中）
后处理增强 = Whisper复核 / LLM校对 / 翻译（转录完成后）
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict
import numpy as np


class DiagnosisResult(Enum):
    """分诊结果类型"""
    CLEAN = "clean"           # 纯净人声，无需分离
    MUSIC = "music"           # 检测到音乐
    NOISE = "noise"           # 检测到噪音
    MIXED = "mixed"           # 混合情况


@dataclass
class SpectrumFeatures:
    """
    频谱特征

    V3.1.1+dev.20260108.01: 新增 SNR/C50/spectral_contrast 字段，支持 Brouhaha 分诊
    """
    # 基础特征
    zcr: float = 0.0                    # 过零率 (Zero Crossing Rate)
    zcr_variance: float = 0.0           # ZCR方差

    # 频谱特征
    spectral_centroid: float = 0.0      # 谱质心 (Hz)
    spectral_bandwidth: float = 0.0     # 谱带宽
    spectral_flatness: float = 0.0      # 频谱平坦度
    spectral_rolloff: float = 0.0       # 频谱滚降点
    spectral_contrast: float = 0.0      # 频谱对比度 (dB)，人声 > 15dB，噪音 < 12dB

    # 谐波特征
    harmonic_ratio: float = 0.0         # 谐波比 (Harmonic-to-Noise Ratio)

    # 能量特征
    rms_energy: float = 0.0             # RMS能量
    energy_variance: float = 0.0        # 能量方差
    high_freq_ratio: float = 0.0        # 高频能量占比 (4kHz以上)

    # 节奏特征
    onset_strength: float = 0.0         # 节拍强度
    tempo: float = 0.0                  # 估计BPM

    # Brouhaha 检测特征 (V3.1.1+dev.20260108.01)
    snr: float = 0.0                    # 信噪比 (dB)，-10 ~ 50
    c50: float = 0.0                    # 清晰度指数 (dB)，-20 ~ 30


@dataclass
class SpectrumDiagnosis:
    """
    频谱分诊结果

    V3.1.1+dev.20260108.01: 新增 SNR/C50 级别和分诊层级字段，支持三层决策
    """
    chunk_index: int                           # Chunk索引
    diagnosis: DiagnosisResult                 # 分诊结果
    need_separation: bool                      # 是否需要分离

    # 评分
    music_score: float = 0.0                   # 音乐得分 (0-1)
    noise_score: float = 0.0                   # 噪音得分 (0-1)
    clean_score: float = 0.0                   # 纯净度得分 (0-1)

    # 推荐的分离模型
    recommended_model: Optional[str] = None    # None / "htdemucs" / "mdx_extra"

    # 原始特征（用于调试）
    features: SpectrumFeatures = field(default_factory=SpectrumFeatures)

    # 决策原因
    reason: str = ""

    # Brouhaha 分诊扩展字段 (V3.1.1+dev.20260108.01)
    snr: float = 0.0                           # 信噪比 (dB)
    c50: float = 0.0                           # 清晰度指数 (dB)
    snr_level: str = ""                        # SNR 级别: "high" / "warn" / "low"
    c50_level: str = ""                        # C50 级别: "good" / "warn" / "bad"
    triage_layer: int = 0                      # 决策层级: 1 / 2 / 3，0 表示未使用 SNR 策略


class SeparationLevel(Enum):
    """人声分离级别"""
    NONE = "none"              # 未分离
    HTDEMUCS = "htdemucs"      # 轻度分离
    MDX_EXTRA = "mdx_extra"    # 重度分离（最高级别）

    def can_upgrade(self) -> bool:
        """是否可以升级到更高级别"""
        return self != SeparationLevel.MDX_EXTRA

    def next_level(self) -> Optional['SeparationLevel']:
        """
        获取下一个分离级别

        升级路径：NONE → HTDEMUCS → MDX_EXTRA

        Returns:
            下一个级别，如果已是最高级别则返回None
        """
        if self == SeparationLevel.NONE:
            return SeparationLevel.HTDEMUCS
        elif self == SeparationLevel.HTDEMUCS:
            return SeparationLevel.MDX_EXTRA
        else:
            return None


class FuseAction(Enum):
    """熔断动作（仅升级分离相关）"""
    ACCEPT = "accept"                           # 接受结果，继续流程
    UPGRADE_SEPARATION = "upgrade_separation"   # 升级分离模型，回溯重做


@dataclass
class ChunkProcessState:
    """
    Chunk 处理状态（核心：保留原始音频引用）

    每个 VAD Chunk 在处理过程中需要维护的状态

    音频引用说明：
    - original_audio 和 current_audio 都是 Chunk 片段（已从完整音频中截取）
    - start_time/end_time 是该 Chunk 在原始音频中的绝对时间点（秒）
    - sample_rate 是音频采样率（默认 16000Hz，与 Whisper/SenseVoice 一致）
    """
    chunk_index: int
    start_time: float                          # Chunk 在原始音频中的起始时间（秒）
    end_time: float                            # Chunk 在原始音频中的结束时间（秒）

    # ========== 音频引用（关键！）==========
    original_audio: np.ndarray = None          # 原始音频片段（分离前），用于熔断回溯
    current_audio: np.ndarray = None           # 当前使用的音频片段（可能已分离）
    sample_rate: int = 16000                   # 采样率（默认 16kHz）

    # ========== 分离状态 ==========
    separation_level: SeparationLevel = SeparationLevel.NONE
    separation_model_used: Optional[str] = None  # 实际使用的模型名

    # ========== 熔断状态 ==========
    fuse_retry_count: int = 0                  # 熔断重试次数
    max_fuse_retry: int = 1                    # 最大重试次数（止损点）

    # ========== 转录结果 ==========
    transcription_confidence: float = 0.0
    event_tag: Optional[str] = None            # SenseVoice 检测到的事件标签（BGM/Noise等）

    def can_upgrade_separation(self) -> bool:
        """是否可以升级分离模型"""
        if self.fuse_retry_count >= self.max_fuse_retry:
            return False
        if self.separation_level == SeparationLevel.MDX_EXTRA:
            return False  # 已是最高级别
        return True

    def get_next_separation_level(self) -> Optional[SeparationLevel]:
        """获取下一个分离级别"""
        if self.separation_level == SeparationLevel.NONE:
            return SeparationLevel.HTDEMUCS
        elif self.separation_level == SeparationLevel.HTDEMUCS:
            return SeparationLevel.MDX_EXTRA
        return None


@dataclass
class FuseDecision:
    """熔断决策结果"""
    action: FuseAction
    target_level: Optional[SeparationLevel] = None
    reason: str = ""
