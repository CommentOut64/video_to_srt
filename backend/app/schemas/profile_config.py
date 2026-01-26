"""
Profile 配置数据类 - V3.2.0+dev.20260125.06

统一的流水线配置，包含转录模式、VAD 配置、ASR 引擎等。
"""
from dataclasses import dataclass
from typing import Optional

from app.core.asr.engine import ASREngine
from app.core.thresholds import ThresholdConfig
from app.services.audio.vad_service import VADConfig


@dataclass
class ProfileConfig:
    """Profile 配置 - 统一的流水线配置"""
    # 转录流水线模式: sensevoice_only/sv_whisper_patch/sv_whisper_dual
    transcription_profile: str
    # VAD 策略: sensevoice/whisper
    vad_profile: str
    # VAD 配置
    vad_config: VADConfig
    # 草稿引擎（SenseVoice）
    draft_engine: Optional[ASREngine]
    # 补刀引擎（Whisper）
    patch_engine: Optional[ASREngine]
    # 补刀触发阈值配置
    patching_threshold: ThresholdConfig
