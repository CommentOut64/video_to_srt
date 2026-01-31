"""
ChunkEngine - 音频切分引擎

Phase 2 实现 - 2025-12-10

负责将音频文件切分为适合推理的 Chunk 片段。
集成 Demucs 人声分离和 VAD 语音检测功能。
"""

import logging
import warnings
from typing import Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import numpy as np
# 抑制 librosa 的 PySoundFile/audioread 警告
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="PySoundFile failed")
    warnings.filterwarnings("ignore", message="audioread")
    import librosa

from app.services.audio.vad_service import VADService, VADConfig, VADMethod
from app.services.demucs_service import DemucsService, BGMLevel
from app.models.circuit_breaker_models import SeparationLevel, SpectrumDiagnosis


@dataclass
class AudioChunk:
    """
    音频片段数据结构 - 扩展版

    表示一个经过 VAD 切分的音频片段，包含时间信息和音频数据。
    支持频谱分诊、人声分离和熔断回溯功能。
    """
    # 基础字段
    index: int                          # 片段索引
    start: float                        # 起始时间（秒）
    end: float                          # 结束时间（秒）
    audio: np.ndarray                   # 音频数组（单声道，16kHz）
    sample_rate: int = 16000            # 采样率

    # 分离状态字段（原有）
    is_separated: bool = False          # 是否已进行人声分离
    separation_model: Optional[str] = None  # 使用的分离模型

    # 频谱分诊相关字段（新增）
    needs_separation: bool = False      # 是否需要人声分离
    recommended_model: Optional[str] = None  # 推荐模型 (htdemucs/mdx_extra)
    spectrum_diagnosis: Optional[SpectrumDiagnosis] = None  # 完整的分诊结果

    # 分离级别相关字段（新增）
    separation_level: SeparationLevel = SeparationLevel.NONE  # 分离级别
    original_audio: Optional[np.ndarray] = None  # 原始音频（用于熔断回溯）

    # 熔断回溯相关字段（新增）
    fuse_retry_count: int = 0           # 熔断重试次数
    last_confidence: float = 1.0        # 上次转录的置信度

    # V3.2.0+dev.20260127.02: LangID 语言检测字段
    language: Optional[str] = None      # 语言标签（ISO 639-3，低置信度可标记为 auto）
    language_confidence: Optional[float] = None  # 语言检测置信度 [0, 1]

    # V3.2.0+dev.20260127.06: 声纹向量字段（用于后续说话人聚类）
    speaker_embedding: Optional[List[float]] = None  # 192 维声纹向量

    @property
    def duration(self) -> float:
        """片段时长（秒）"""
        return self.end - self.start

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "is_separated": self.is_separated,
            "separation_model": self.separation_model,
            "audio_shape": self.audio.shape if self.audio is not None else None,
            # 新增字段
            "needs_separation": self.needs_separation,
            "recommended_model": self.recommended_model,
            "separation_level": self.separation_level.value if self.separation_level else None,
            "fuse_retry_count": self.fuse_retry_count,
            "last_confidence": self.last_confidence,
            "language": self.language,
            "language_confidence": self.language_confidence,
            # 避免日志/序列化膨胀，仅记录声纹维度
            "speaker_embedding_dim": len(self.speaker_embedding) if self.speaker_embedding else None,
        }


class ChunkEngine:
    """
    音频切分引擎

    负责将音频文件切分为适合推理的 Chunk 片段。
    支持可选的 Demucs 人声分离和 VAD 语音检测。
    """

    def __init__(
        self,
        vad_service: Optional[VADService] = None,
        demucs_service: Optional[DemucsService] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化音频切分引擎

        Args:
            vad_service: VAD 服务实例，如果为 None 则创建新的
            demucs_service: Demucs 服务实例，如果为 None 则创建新的
            logger: 日志记录器，如果为 None 则创建新的
        """
        self.logger = logger or logging.getLogger(__name__)
        self.vad_service = vad_service or VADService(logger=self.logger)
        self.demucs_service = demucs_service or DemucsService()

    def process_audio(
        self,
        audio_path: str,
        enable_demucs: bool = False,
        demucs_model: Optional[str] = None,
        vad_config: Optional[VADConfig] = None,
        progress_callback: Optional[callable] = None,
        diagnostic_service: Optional[Any] = None  # V3.2.0+dev.20260131: 诊断服务
    ) -> Tuple[List[AudioChunk], np.ndarray, int]:
        """
        处理音频文件，返回切分后的 Chunk 列表

        处理流程：
        1. 加载音频并降采样到 16kHz
        2. （可选）Demucs 整轨人声分离
        3. VAD 语音检测和切分
        4. 生成 AudioChunk 列表

        Args:
            audio_path: 音频文件路径
            enable_demucs: 是否启用 Demucs 人声分离
            demucs_model: Demucs 模型名称（可选）
            vad_config: VAD 配置（可选，使用默认配置）
            progress_callback: 进度回调 callback(progress: float, message: str)
            diagnostic_service: 诊断服务实例（可选，V3.2.0+dev.20260131）

        Returns:
            Tuple[List[AudioChunk], np.ndarray, int]:
                - Chunk 列表
                - 完整音频数组
                - 采样率
        """
        self.logger.info(f"开始处理音频: {audio_path}")

        # 1. 加载音频
        if progress_callback:
            progress_callback(0.1, "加载音频...")

        audio_array, sr = self._load_audio(audio_path)
        self.logger.info(f"音频加载完成: 时长 {len(audio_array)/sr:.2f}s, 采样率 {sr}Hz")

        # 2. 可选的 Demucs 人声分离
        separated_audio = audio_array
        separation_model_used = None

        if enable_demucs:
            if progress_callback:
                progress_callback(0.2, "Demucs 人声分离中...")

            self.logger.info("开始 Demucs 整轨人声分离...")
            separated_path = self._separate_vocals(
                audio_path,
                model=demucs_model,
                progress_callback=progress_callback
            )

            # 重新加载分离后的音频
            separated_audio, _ = self._load_audio(separated_path)
            separation_model_used = demucs_model or self.demucs_service.config.model_name
            self.logger.info(f"Demucs 分离完成，使用模型: {separation_model_used}")

            # 立即卸载 Demucs 模型释放显存
            self.logger.info("释放 Demucs 显存...")
            self.demucs_service.unload_model()

        # 3. VAD 语音检测和切分
        if progress_callback:
            progress_callback(0.6, "VAD 语音检测中...")

        if vad_config is None:
            from app.services.runtime_param_resolver import build_vad_config

            vad_config = build_vad_config()
        segments = self._detect_speech_segments(separated_audio, sr, vad_config, diagnostic_service)

        self.logger.info(f"VAD 检测完成: {len(segments)} 个语音段")

        # 4. 生成 AudioChunk 列表
        if progress_callback:
            progress_callback(0.8, "生成 Chunk 列表...")

        # V3.1.1+dev.20260107.04: 传递原始音频用于能量对比
        chunks = self._create_chunks(
            separated_audio,
            sr,
            segments,
            is_separated=enable_demucs,
            separation_model=separation_model_used,
            original_audio=audio_array if enable_demucs else None
        )

        if progress_callback:
            progress_callback(1.0, "处理完成")

        self.logger.info(f"音频处理完成: {len(chunks)} 个 Chunk")
        return chunks, audio_array, sr

    def _load_audio(self, audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        加载音频文件并降采样到目标采样率

        Args:
            audio_path: 音频文件路径
            target_sr: 目标采样率（默认 16kHz）

        Returns:
            Tuple[np.ndarray, int]: 音频数组和采样率
        """
        try:
            # 使用 librosa 加载音频（自动降采样到 16kHz，单声道）
            audio_array, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            return audio_array, sr
        except Exception as e:
            self.logger.error(f"加载音频失败: {e}")
            raise

    def _separate_vocals(
        self,
        audio_path: str,
        model: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> str:
        """
        使用 Demucs 进行整轨人声分离

        Args:
            audio_path: 音频文件路径
            model: Demucs 模型名称（可选）
            progress_callback: 进度回调

        Returns:
            str: 分离后的人声文件路径
        """
        if model:
            self.demucs_service.set_model(model)

        return self.demucs_service.separate_vocals(
            audio_path,
            progress_callback=progress_callback
        )

    def _detect_speech_segments(
        self,
        audio_array: np.ndarray,
        sr: int,
        vad_config: VADConfig,
        diagnostic_service: Optional[Any] = None  # V3.2.0+dev.20260131: 诊断服务
    ) -> List[dict]:
        """
        使用 VAD 检测语音段

        Args:
            audio_array: 音频数组
            sr: 采样率
            vad_config: VAD 配置
            diagnostic_service: 诊断服务实例（可选）

        Returns:
            List[dict]: 语音段元数据列表
        """
        return self.vad_service.detect_speech_segments(
            audio_array, sr, vad_config, diagnostic_service=diagnostic_service
        )

    def _create_chunks(
        self,
        audio_array: np.ndarray,
        sr: int,
        segments: List[dict],
        is_separated: bool = False,
        separation_model: Optional[str] = None,
        original_audio: Optional[np.ndarray] = None
    ) -> List[AudioChunk]:
        """
        根据 VAD 分段结果创建 AudioChunk 列表

        Args:
            audio_array: 完整音频数组（可能是分离后的）
            sr: 采样率
            segments: VAD 分段元数据列表
            is_separated: 是否已进行人声分离
            separation_model: 使用的分离模型
            original_audio: 原始音频数组（用于能量对比和回退）

        Returns:
            List[AudioChunk]: Chunk 列表
        """
        chunks = []

        for seg in segments:
            start_sec = seg["start"]
            end_sec = seg["end"]

            # 提取音频片段
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            chunk_audio = audio_array[start_sample:end_sample]

            # V3.1.1+dev.20260107.04: 提取原始音频片段（用于能量对比）
            original_chunk = None
            if original_audio is not None:
                original_chunk = original_audio[start_sample:end_sample]

                # 能量检测：如果分离后能量显著低于原始，自动回退
                sep_rms = np.sqrt(np.mean(chunk_audio**2))
                orig_rms = np.sqrt(np.mean(original_chunk**2))

                # 阈值：分离后 RMS < 原始 RMS * 20% 且原始 RMS > 1e-3
                if sep_rms < orig_rms * 0.2 and orig_rms > 1e-3:
                    self.logger.warning(
                        f"Chunk {seg['index']}: Demucs 分离失败，能量过低 "
                        f"(sep_rms={sep_rms:.6f}, orig_rms={orig_rms:.6f}, "
                        f"ratio={sep_rms/orig_rms*100:.1f}%)，回退到原始音频"
                    )
                    chunk_audio = original_chunk  # 回退到原始音频

            # 创建 AudioChunk
            chunk = AudioChunk(
                index=seg["index"],
                start=start_sec,
                end=end_sec,
                audio=chunk_audio,
                sample_rate=sr,
                is_separated=is_separated,
                separation_model=separation_model,
                original_audio=original_chunk
            )

            chunks.append(chunk)

        return chunks

    def process_audio_with_adaptive_separation(
        self,
        audio_path: str,
        vram_mb: int,
        vad_config: Optional[VADConfig] = None,
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[AudioChunk], np.ndarray, int]:
        """
        根据显存大小自适应选择分离策略

        显存策略：
        - VRAM > 6GB: Demucs 整轨分离
        - VRAM < 6GB: 跳过 Demucs 或使用大块切分（5分钟）
        - No GPU: 跳过 Demucs

        Args:
            audio_path: 音频文件路径
            vram_mb: 可用显存（MB）
            vad_config: VAD 配置（可选）
            progress_callback: 进度回调

        Returns:
            Tuple[List[AudioChunk], np.ndarray, int]: Chunk 列表、完整音频、采样率
        """
        self.logger.info(f"自适应分离策略: 可用显存 {vram_mb}MB")

        # 显存策略决策
        from app.services.runtime_param_resolver import get_demucs_runtime_params
        runtime_demucs = get_demucs_runtime_params()
        runtime_model = runtime_demucs.get("model_name") or "htdemucs"

        if vram_mb >= 6000:
            # 高显存：整轨分离
            self.logger.info("显存充足，使用 Demucs 整轨分离")
            return self.process_audio(
                audio_path,
                enable_demucs=True,
                demucs_model=runtime_model,
                vad_config=vad_config,
                progress_callback=progress_callback
            )
        elif vram_mb >= 4000:
            # 中等显存：使用轻量模型
            self.logger.info("显存中等，使用轻量 Demucs 模型")
            return self.process_audio(
                audio_path,
                enable_demucs=True,
                demucs_model=runtime_model,
                vad_config=vad_config,
                progress_callback=progress_callback
            )
        else:
            # 低显存或无 GPU：跳过 Demucs
            self.logger.info("显存不足，跳过 Demucs 分离")
            return self.process_audio(
                audio_path,
                enable_demucs=False,
                vad_config=vad_config,
                progress_callback=progress_callback
            )


# 便捷函数
def get_chunk_engine(
    vad_service: Optional[VADService] = None,
    demucs_service: Optional[DemucsService] = None,
    logger: Optional[logging.Logger] = None
) -> ChunkEngine:
    """
    获取 ChunkEngine 实例

    Args:
        vad_service: VAD 服务实例
        demucs_service: Demucs 服务实例
        logger: 日志记录器

    Returns:
        ChunkEngine 实例
    """
    return ChunkEngine(
        vad_service=vad_service,
        demucs_service=demucs_service,
        logger=logger
    )
