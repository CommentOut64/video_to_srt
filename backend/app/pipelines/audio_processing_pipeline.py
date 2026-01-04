"""
AudioProcessingPipeline - 音频前处理流水线

Phase 2 实现 - 2025-12-10

完整的音频前处理流程：
1. 音频提取 & 降采样（16kHz）
2. 显存策略检查
3. Demucs 整轨分离（可选，根据显存）
4. VAD 语音检测和切分
5. 生成 AudioChunk 队列
"""

import logging
import tempfile
from typing import List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.services.audio.chunk_engine import ChunkEngine, AudioChunk
from app.services.audio.vad_service import VADService, VADConfig
from app.services.demucs_service import DemucsService
from app.services.monitoring.hardware_monitor import HardwareMonitor
from app.core.resource_manager import ResourceManager


class SeparationStrategy(Enum):
    """人声分离策略"""
    NONE = "none"              # 不分离
    FULL_TRACK = "full_track"  # 整轨分离
    LARGE_CHUNK = "large_chunk"  # 大块切分（5分钟）


@dataclass
class AudioProcessingConfig:
    """
    音频处理配置

    控制音频前处理流水线的各项参数。
    """
    # VAD 配置
    vad_config: VADConfig = None

    # Demucs 配置
    enable_demucs: bool = True          # 是否启用 Demucs
    demucs_model: str = "htdemucs"      # Demucs 模型名称
    auto_strategy: bool = True          # 是否自动选择分离策略

    # 显存阈值（MB）
    vram_threshold_full: int = 6000     # 整轨分离最低显存
    vram_threshold_chunk: int = 4000    # 大块切分最低显存

    # 音频参数
    target_sample_rate: int = 16000     # 目标采样率

    def __post_init__(self):
        """初始化后处理"""
        if self.vad_config is None:
            self.vad_config = VADConfig()


@dataclass
class AudioProcessingResult:
    """
    音频处理结果

    包含处理后的 Chunk 列表和相关元数据。
    """
    chunks: List[AudioChunk]            # Chunk 列表
    full_audio: np.ndarray              # 完整音频数组
    sample_rate: int                    # 采样率
    total_duration: float               # 总时长（秒）
    separation_strategy: SeparationStrategy  # 使用的分离策略
    vram_used_mb: int                   # 使用的显存（MB）

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "chunk_count": len(self.chunks),
            "total_duration": self.total_duration,
            "sample_rate": self.sample_rate,
            "separation_strategy": self.separation_strategy.value,
            "vram_used_mb": self.vram_used_mb,
            "chunks": [chunk.to_dict() for chunk in self.chunks]
        }


class AudioProcessingPipeline:
    """
    音频前处理流水线

    完整的音频前处理流程，包括音频提取、人声分离、VAD 切分等。
    支持显存自适应策略。
    """

    def __init__(
        self,
        chunk_engine: Optional[ChunkEngine] = None,
        hardware_monitor: Optional[HardwareMonitor] = None,
        resource_manager: Optional[ResourceManager] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化音频处理流水线

        Args:
            chunk_engine: 音频切分引擎
            hardware_monitor: 硬件监控器
            resource_manager: 资源管理器
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.chunk_engine = chunk_engine or ChunkEngine(logger=self.logger)
        self.hardware_monitor = hardware_monitor or HardwareMonitor()
        self.resource_manager = resource_manager

    async def process(
        self,
        video_path: str,
        config: Optional[AudioProcessingConfig] = None,
        progress_callback: Optional[callable] = None
    ) -> AudioProcessingResult:
        """
        处理视频/音频文件，返回 Chunk 列表

        完整流程：
        1. 提取音频并降采样到 16kHz
        2. 检查显存并决定分离策略
        3. 执行 Demucs 分离（如果需要）
        4. VAD 切分
        5. 生成 AudioChunk 队列

        Args:
            video_path: 视频/音频文件路径
            config: 音频处理配置
            progress_callback: 进度回调 callback(progress: float, message: str)

        Returns:
            AudioProcessingResult: 处理结果
        """
        config = config or AudioProcessingConfig()

        self.logger.info(f"开始音频前处理流水线: {video_path}")

        # 1. 提取音频
        if progress_callback:
            progress_callback(0.05, "提取音频...")

        audio_path = await self._extract_audio(video_path, config.target_sample_rate)
        self.logger.info(f"音频提取完成: {audio_path}")

        # 2. 显存策略检查
        if progress_callback:
            progress_callback(0.1, "检查显存...")

        separation_strategy, vram_mb = self._determine_separation_strategy(config)
        self.logger.info(f"分离策略: {separation_strategy.value}, 可用显存: {vram_mb}MB")

        # 3. 执行音频处理（Demucs + VAD）
        if separation_strategy == SeparationStrategy.NONE:
            # 不分离
            chunks, full_audio, sr = self.chunk_engine.process_audio(
                audio_path,
                enable_demucs=False,
                vad_config=config.vad_config,
                progress_callback=progress_callback
            )
        elif separation_strategy == SeparationStrategy.FULL_TRACK:
            # 整轨分离
            chunks, full_audio, sr = self.chunk_engine.process_audio(
                audio_path,
                enable_demucs=True,
                demucs_model=config.demucs_model,
                vad_config=config.vad_config,
                progress_callback=progress_callback
            )
        else:
            # 大块切分（暂时使用整轨分离的轻量模型）
            chunks, full_audio, sr = self.chunk_engine.process_audio(
                audio_path,
                enable_demucs=True,
                demucs_model="htdemucs",  # 使用快速模型
                vad_config=config.vad_config,
                progress_callback=progress_callback
            )

        # 4. 生成结果
        total_duration = len(full_audio) / sr

        result = AudioProcessingResult(
            chunks=chunks,
            full_audio=full_audio,
            sample_rate=sr,
            total_duration=total_duration,
            separation_strategy=separation_strategy,
            vram_used_mb=vram_mb
        )

        self.logger.info(f"音频前处理完成: {len(chunks)} 个 Chunk, 总时长 {total_duration:.2f}s")

        if progress_callback:
            progress_callback(1.0, "处理完成")

        return result

    async def _extract_audio(self, video_path: str, target_sr: int = 16000) -> str:
        """
        从视频中提取音频并降采样

        使用 FFmpeg 提取音频，降采样到 16kHz 单声道。

        Args:
            video_path: 视频文件路径
            target_sr: 目标采样率

        Returns:
            str: 提取的音频文件路径
        """
        import subprocess
        from app.core.config import config

        # 生成临时音频文件路径
        temp_dir = Path(tempfile.gettempdir())
        audio_path = temp_dir / f"{Path(video_path).stem}_audio.wav"

        # FFmpeg 命令
        ffmpeg_cmd = str(config.FFMPEG_EXE) if config.FFMPEG_EXE.exists() else "ffmpeg"
        cmd = [
            ffmpeg_cmd, '-y', '-i', video_path,
            '-vn',                    # 仅音频
            '-ac', '1',               # 单声道
            '-ar', str(target_sr),    # 采样率
            '-acodec', 'pcm_s16le',   # PCM 编码
            str(audio_path)
        ]

        try:
            self.logger.info(f"提取音频: {video_path} -> {audio_path}")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return str(audio_path)
        except subprocess.CalledProcessError as e:
            # V3.1.0+dev.20260104.01: 修复 Windows 编码问题
            self.logger.error(f"音频提取失败: {e.stderr.decode('utf-8', errors='replace')}")
            raise RuntimeError(f"音频提取失败: {e}")

    def _determine_separation_strategy(
        self,
        config: AudioProcessingConfig
    ) -> Tuple[SeparationStrategy, int]:
        """
        根据显存大小决定分离策略

        策略：
        - VRAM > 6GB: 整轨分离
        - VRAM 4-6GB: 大块切分（5分钟）
        - VRAM < 4GB 或无 GPU: 不分离

        Args:
            config: 音频处理配置

        Returns:
            Tuple[SeparationStrategy, int]: 分离策略和可用显存（MB）
        """
        if not config.enable_demucs:
            return SeparationStrategy.NONE, 0

        # 获取可用显存
        if self.hardware_monitor.has_gpu:
            status = self.hardware_monitor.get_current_status()
            if status.gpu:
                vram_mb = status.gpu.total_memory_mb - status.gpu.used_memory_mb
            else:
                vram_mb = 0
        else:
            vram_mb = 0

        # 如果有资源管理器，使用其显存信息
        if self.resource_manager:
            vram_mb = self.resource_manager.get_available_vram()

        # 决策
        if not config.auto_strategy:
            # 手动模式：始终使用整轨分离
            return SeparationStrategy.FULL_TRACK, vram_mb

        if vram_mb >= config.vram_threshold_full:
            return SeparationStrategy.FULL_TRACK, vram_mb
        elif vram_mb >= config.vram_threshold_chunk:
            return SeparationStrategy.LARGE_CHUNK, vram_mb
        else:
            return SeparationStrategy.NONE, vram_mb

    def get_status(self) -> dict:
        """
        获取流水线状态

        Returns:
            dict: 状态信息
        """
        hw_status = self.hardware_monitor.get_current_status()

        return {
            "has_gpu": self.hardware_monitor.has_gpu,
            "gpu_count": self.hardware_monitor.gpu_count,
            "available_vram_mb": hw_status.gpu.total_memory_mb - hw_status.gpu.used_memory_mb if hw_status.gpu else 0,
            "cpu_usage": hw_status.cpu.usage_percent,
            "memory_usage": hw_status.memory.usage_percent
        }


# 便捷函数
def get_audio_processing_pipeline(
    chunk_engine: Optional[ChunkEngine] = None,
    hardware_monitor: Optional[HardwareMonitor] = None,
    resource_manager: Optional[ResourceManager] = None,
    logger: Optional[logging.Logger] = None
) -> AudioProcessingPipeline:
    """
    获取音频处理流水线实例

    Args:
        chunk_engine: 音频切分引擎
        hardware_monitor: 硬件监控器
        resource_manager: 资源管理器
        logger: 日志记录器

    Returns:
        AudioProcessingPipeline 实例
    """
    return AudioProcessingPipeline(
        chunk_engine=chunk_engine,
        hardware_monitor=hardware_monitor,
        resource_manager=resource_manager,
        logger=logger
    )
