"""
VAD (Voice Activity Detection) 服务

从 transcription_service.py 中提取的 VAD 功能，封装为独立服务。
支持 Silero VAD（默认）和 Pyannote VAD（可选）。

Phase 2 实现 - 2025-12-10
"""

import os
import logging
import tempfile
from enum import Enum
from dataclasses import dataclass
from typing import Any, List, Dict, Optional
from pathlib import Path

import numpy as np
import torch

from app.services.pyannote_compat import load_pyannote_pipeline


class VADMethod(Enum):
    """
    VAD模型选择枚举
    用于选择语音活动检测（Voice Activity Detection）模型
    """
    SILERO = "silero"      # 默认，无需认证，速度快
    PYANNOTE = "pyannote"  # 可选，需要HF Token，精度更高


class OptimizedOnnxWrapper:
    """
    优化版 Silero VAD ONNX Wrapper

    相比原版 OnnxWrapper 的改进：
    1. 使用硬件能力提供者智能计算线程数（而非硬编码为 1）
    2. 支持 Intel 混合架构 P-Core 亲和性绑定
    3. 启用 ONNX Runtime CPU 优化选项
    """

    def __init__(self, path: str, logger: Optional[logging.Logger] = None):
        """
        初始化优化版 ONNX Wrapper

        Args:
            path: ONNX 模型文件路径
            logger: 日志记录器
        """
        import onnxruntime as ort

        self.logger = logger or logging.getLogger(__name__)
        self._np = __import__('numpy')

        # 使用硬件能力提供者获取优化的 SessionOptions
        sess_options = self._create_optimized_session_options()

        # 创建 ONNX 推理会话（强制使用 CPU，确保 P-Core 亲和性生效）
        self.session = ort.InferenceSession(
            path,
            providers=['CPUExecutionProvider'],
            sess_options=sess_options
        )

        # 设置 P-Core 亲和性（在 Intel 混合架构上）
        self._setup_pcore_affinity()

        self.reset_states()
        if '16k' in path:
            self.sample_rates = [16000]
        else:
            self.sample_rates = [8000, 16000]

    def _create_optimized_session_options(self):
        """创建优化的 ONNX SessionOptions"""
        try:
            from app.services.hardware_profile_service import get_hardware_profile_provider

            provider = get_hardware_profile_provider()
            sess_options, optimal_threads, info = provider.build_onnx_session_options(
                usage_ratio=0.6
            )

            self.logger.info(
                f"Silero VAD ONNX 优化: {optimal_threads} 线程 "
                f"({info.get('strategy', 'unknown')})"
            )

            return sess_options

        except Exception as e:
            # 回退：使用默认配置
            import onnxruntime as ort
            self.logger.warning(f"硬件能力提供者不可用，使用默认配置: {e}")

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4  # 回退默认值
            sess_options.inter_op_num_threads = 1
            return sess_options

    def _setup_pcore_affinity(self):
        """
        设置 P-Core 亲和性（仅 Intel 混合架构）

        在 Intel 12代+ 混合架构 CPU 上，将当前线程绑定到 P-Core，
        避免 ONNX 推理任务被调度到 E-Core 导致性能下降。
        """
        try:
            from app.services.hardware_profile_service import get_hardware_profile_provider

            provider = get_hardware_profile_provider()
            provider.apply_pcore_affinity(context="Silero VAD P-Core")

        except Exception as e:
            # 亲和性设置失败不是致命错误
            self.logger.debug(f"P-Core 亲和性设置失败（非致命）: {e}")

    def _validate_input(self, x, sr: int):
        """验证输入"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

        if sr != 16000 and (sr % 16000 == 0):
            step = sr // 16000
            x = x[:, ::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)")
        if sr / x.shape[1] > 31.25:
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def reset_states(self, batch_size=1):
        """重置状态"""
        self._state = torch.zeros((2, batch_size, 128)).float()
        self._context = torch.zeros(0)
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sr: int):
        """执行 VAD 推理"""
        x, sr = self._validate_input(x, sr)
        num_samples = 512 if sr == 16000 else 256

        if x.shape[-1] != num_samples:
            raise ValueError(
                f"Provided number of samples is {x.shape[-1]} "
                f"(Supported values: 256 for 8000 sample rate, 512 for 16000)"
            )

        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if self._last_sr and self._last_sr != sr:
            self.reset_states(batch_size)
        if self._last_batch_size and self._last_batch_size != batch_size:
            self.reset_states(batch_size)

        if not len(self._context):
            self._context = torch.zeros(batch_size, context_size)

        x = torch.cat([self._context, x], dim=1)
        if sr in [8000, 16000]:
            ort_inputs = {
                'input': x.numpy(),
                'state': self._state.numpy(),
                'sr': self._np.array(sr, dtype='int64')
            }
            ort_outs = self.session.run(None, ort_inputs)
            out, state = ort_outs
            self._state = torch.from_numpy(state)
        else:
            raise ValueError()

        self._context = x[..., -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size

        out = torch.from_numpy(out)
        return out

    def audio_forward(self, x, sr: int):
        """处理完整音频"""
        outs = []
        x, sr = self._validate_input(x, sr)
        self.reset_states()
        num_samples = 512 if sr == 16000 else 256

        if x.shape[1] % num_samples:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = torch.nn.functional.pad(x, (0, pad_num), 'constant', value=0.0)

        for i in range(0, x.shape[1], num_samples):
            wavs_batch = x[:, i:i + num_samples]
            out_chunk = self.__call__(wavs_batch, sr)
            outs.append(out_chunk)

        stacked = torch.cat(outs, dim=1)
        return stacked.cpu()


@dataclass
class VADConfig:
    """
    VAD配置数据类
    用于配置语音活动检测的参数

    参数说明：
    - onset (0.0-1.0)：语音开始阈值，越高越严格，推荐0.5-0.6以避免截断语音起始
    - offset (0.0-1.0)：语音结束阈值，通常为onset的80%左右
    - min_speech_duration_ms：最小语音段长度，避免误检碎片音（推荐300-500ms）
    - min_silence_duration_ms：最小静音长度，越长越能过滤背景音乐（推荐300-500ms）

    Post-VAD合并参数（2025-12-17 迁移自旧架构 _merge_vad_segments）：
    - merge_max_gap：允许合并的最大静音间隔，超过1.5秒通常意味着换气或换话题
    - merge_max_duration：合并后的最大时长，12秒是甜蜜点，避免Whisper幻觉和对齐算法爆炸
    - merge_min_fragment：短于此时长的片段强制尝试合并（碎片保护）

    Smart Accumulation 参数（V3.1.0 - 2025-12-19）：
    - smart_target_duration：软上限，达到后寻找断点截断（默认12.0s）
    - smart_max_duration：硬上限，绝对不能超（默认30.0s，Whisper物理限制）
    - smart_min_gap_to_split：达到软上限后，多大的间隔算"合适的断点"（默认0.3s）

    修改历史：
    - 2025-12: onset 从 0.7 降低至 0.5，offset 从 0.5 降低至 0.4
      原因：避免语音起始被截断，提高时间戳准确性
    - 2025-12-17: 新增 Post-VAD 合并参数，迁移自旧架构
      max_duration 从 25.0 改为 12.0，max_gap 从 1.0 改为 1.5
    - 2025-12-19: V3.1.0 引入 Smart Accumulation 智能累积算法
      替换贪婪合并，在源头利用VAD断点精准控制Chunk时长
    """
    method: VADMethod = VADMethod.SILERO  # 默认使用Silero
    hf_token: Optional[str] = None         # Pyannote需要的HF Token
    onset: float = 0.4                     # 语音开始阈值（恢复旧值，保持向后兼容）
    offset: float = 0.4                    # 语音结束阈值（恢复旧值）
    chunk_size: int = 30                   # 最大段长（秒）
    min_speech_duration_ms: int = 250      # 最小语音段长度（恢复旧值400ms）
    min_silence_duration_ms: int = 400     # 最小静音长度（恢复旧值400ms）
    speech_pad_ms: int = 300
    # Post-VAD 合并参数
    merge_max_gap: float = 1.0             # 允许合并的最大静音间隔（秒），恢复旧值1.0s
    merge_max_duration: float = 12.0       # 合并后的最大时长（秒），恢复旧值25.0s
    merge_min_fragment: float = 1.0        # 短于此时长的片段强制尝试合并（碎片保护）
    # Smart Accumulation 参数（V3.1.0）
    smart_target_duration: float = 12.0    # 软上限（甜蜜点）
    smart_max_duration: float = 30.0       # 硬上限（Whisper物理限制）
    smart_min_gap_to_split: float = 0.3    # 达到软上限后的最小断点间隔

    def validate(self) -> bool:
        """验证配置有效性"""
        if self.method == VADMethod.PYANNOTE and not self.hf_token:
            return False  # Pyannote需要Token
        if not (0.0 <= self.onset <= 1.0) or not (0.0 <= self.offset <= 1.0):
            return False  # 阈值必须在0-1之间
        return True


class VADService:
    """
    VAD 服务类

    提供语音活动检测功能，支持多种 VAD 模型。
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化 VAD 服务

        Args:
            logger: 日志记录器，如果为 None 则创建新的
        """
        self.logger = logger or logging.getLogger(__name__)

    def detect_speech_segments(
        self,
        audio_array: np.ndarray,
        sr: int,
        config: VADConfig,
        enable_merge: bool = True,
        diagnostic_service: Optional[Any] = None  # V3.2.0+dev.20260131: 诊断服务
    ) -> List[Dict]:
        """
        检测语音段

        Args:
            audio_array: 音频数组
            sr: 采样率
            config: VAD 配置
            enable_merge: 是否启用 Post-VAD 智能合并（默认启用）
            diagnostic_service: 诊断服务实例（可选，用于记录诊断信息）

        Returns:
            List[Dict]: 分段元数据列表，每个元素包含:
                - index: 段索引
                - start: 起始时间（秒）
                - end: 结束时间（秒）
                - mode: 处理模式（"memory"）

        Raises:
            ValueError: 配置无效时抛出
        """
        if not config.validate():
            raise ValueError("VAD 配置无效")

        try:
            if config.method == VADMethod.SILERO:
                segments = self._vad_silero(audio_array, sr, config)
            elif config.method == VADMethod.PYANNOTE:
                segments = self._vad_pyannote(audio_array, sr, config)
            else:
                raise ValueError(f"不支持的 VAD 方法: {config.method}")
        except Exception as e:
            self.logger.error(f"VAD 检测失败: {e}")
            self.logger.warning("降级到能量检测分段")
            segments = self._energy_based_split(audio_array, sr, config.chunk_size)

        # Post-VAD 智能合并（默认启用）
        if enable_merge and len(segments) > 1:
            segments = self.merge_vad_segments(segments, config)

        # V3.2.0+dev.20260131: 记录诊断信息
        if diagnostic_service is not None:
            try:
                # 记录 VAD 切分结果
                diagnostic_service.record_vad_segments(segments)
                
                # 计算并记录每段边界的 RMS
                rms_window_samples = int(0.05 * sr)  # 50ms 窗口
                for i, seg in enumerate(segments):
                    start_sample = int(seg['start'] * sr)
                    end_sample = int(seg['end'] * sr)
                    
                    # 计算起始点 RMS
                    start_window = audio_array[start_sample:start_sample + rms_window_samples]
                    rms_start = float(np.sqrt(np.mean(start_window ** 2))) if len(start_window) > 0 else 0.0
                    
                    # 计算结束点 RMS
                    end_window = audio_array[max(0, end_sample - rms_window_samples):end_sample]
                    rms_end = float(np.sqrt(np.mean(end_window ** 2))) if len(end_window) > 0 else 0.0
                    
                    diagnostic_service.record_vad_rms(i, rms_start, rms_end)
            except Exception as e:
                self.logger.warning(f"[诊断] 记录 VAD 诊断信息失败: {e}")

        return segments

    def _vad_silero(
        self,
        audio_array: np.ndarray,
        sr: int,
        vad_config: VADConfig
    ) -> List[Dict]:
        """
        Silero VAD分段（使用内置ONNX模型，无需下载）

        优点：
        - 使用项目内置ONNX模型，无需网络下载
        - 使用 onnxruntime 推理，跨平台兼容性好
        - 速度快，内存占用低（~2MB）

        Args:
            audio_array: 音频数组
            sr: 采样率
            vad_config: VAD配置

        Returns:
            List[Dict]: 分段元数据列表
        """
        self.logger.info("加载Silero VAD模型（优化版 ONNX）...")

        # 使用 silero-vad 库的 get_speech_timestamps 函数
        from silero_vad import get_speech_timestamps

        # 通过 ModelManagerV2 统一获取模型路径，回退到内置模型
        try:
            from app.services.model_manager_v2 import get_model_manager_v2

            manager = get_model_manager_v2()
            vad_dir = Path(manager.ensure_available("silero-vad"))
            builtin_model_path = vad_dir / "silero_vad.onnx"
            self.logger.info(f"Silero VAD 使用 ModelManagerV2 模型: {builtin_model_path}")
        except Exception as exc:
            builtin_model_path = Path(__file__).parent.parent.parent / "assets" / "silero" / "silero_vad.onnx"
            self.logger.warning(
                "ModelManagerV2 获取 Silero VAD 失败，回退内置路径: %s (reason=%s)",
                builtin_model_path,
                exc,
            )

        if not builtin_model_path.exists():
            raise FileNotFoundError(
                f"Silero VAD 模型不存在: {builtin_model_path}\n"
                "请确保项目完整，或重新从源码仓库获取"
            )

        # 使用优化版 OnnxWrapper（多线程 + P-Core 亲和性）
        model = OptimizedOnnxWrapper(str(builtin_model_path), logger=self.logger)

        # 转换为torch tensor（silero-vad 需要）
        audio_tensor = torch.from_numpy(audio_array)

        # 获取语音时间戳
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=sr,
            threshold=vad_config.onset,                    # 检测阈值（从config读取，默认0.4）
            min_speech_duration_ms=vad_config.min_speech_duration_ms,   # 最小语音段长度（默认250ms）
            min_silence_duration_ms=vad_config.min_silence_duration_ms, # 最小静音长度（默认400ms）
            speech_pad_ms=vad_config.speech_pad_ms,        # 语音段前后padding（默认300ms）
            return_seconds=False  # 返回采样点而非秒数
        )

        # VAD detection complete

        # V3.1.0: 预处理 - 强制拆分超长原始片段
        # 使用软上限（12秒）作为拆分阈值，确保Smart Accumulation有足够的断点可用
        # 如果VAD原始片段本身就有20秒，Smart Accumulation无法在12秒处截断
        speech_timestamps = self._break_long_segments(
            speech_timestamps,
            audio_array,
            sr,
            max_duration=vad_config.smart_target_duration  # 使用软上限12秒
        )

        # V3.2.0+dev.20260131d: 在 Smart Accumulation 之前先修复高 RMS 边界
        # 原因: Smart Accumulation 会调整边界,导致原始高 RMS 信息丢失
        # 必须在 Silero VAD 原始输出上修复,才能捕获到真实的高能量误判
        speech_timestamps = self._fix_high_rms_boundaries_raw(
            speech_timestamps, audio_array, sr
        )

        # V3.2.0+dev.20260131: 重构 Smart Accumulation 智能累积合并
        # 新增"回溯找最佳断点"逻辑，避免错过更早的理想断点
        # V3.2.0+dev.20260131b: 传入音频数据用于 RMS 验证，防止在高能量点切分
        segments_metadata = self._smart_accumulation(
            speech_timestamps, sr, vad_config, audio_array
        )

        # 如果没有检测到任何语音段，按固定时长分段
        if len(segments_metadata) == 0:
            self.logger.warning("VAD未检测到语音，使用固定时长分段")
            return self._energy_based_split(audio_array, sr, vad_config.chunk_size)

        # V3.1.0: 统计 Smart Accumulation 效果
        durations = [seg['end'] - seg['start'] for seg in segments_metadata]
        avg_duration = sum(durations) / len(durations) if durations else 0
        max_seg_duration = max(durations) if durations else 0
        TARGET = vad_config.smart_target_duration

        self.logger.info(
            f"Silero VAD 检测完成: {len(segments_metadata)} 个语音段 "
            f"(avg={avg_duration:.2f}s, max={max_seg_duration:.2f}s, target={TARGET}s)"
        )
        return segments_metadata

    def _vad_pyannote(
        self,
        audio_array: np.ndarray,
        sr: int,
        vad_config: VADConfig
    ) -> List[Dict]:
        """
        Pyannote VAD分段（高精度方案，需要HF Token）

        优点：
        - 精度更高
        - 支持更复杂的语音活动检测

        注意：
        - 需要HuggingFace Token
        - 首次使用需要接受模型使用协议

        Args:
            audio_array: 音频数组
            sr: 采样率
            vad_config: VAD配置

        Returns:
            List[Dict]: 分段元数据列表

        Raises:
            ValueError: 未配置HF Token时抛出
        """
        if not vad_config.hf_token:
            raise ValueError("Pyannote VAD需要HuggingFace Token，请在设置中配置")

        self.logger.info("加载Pyannote VAD模型（需要HF Token）...")

        try:
            pipeline = load_pyannote_pipeline(
                "pyannote/voice-activity-detection",
                token=vad_config.hf_token,
                logger=self.logger,
            )
        except ImportError:
            raise RuntimeError("Pyannote未安装，请使用Silero VAD或安装pyannote-audio")

        # 准备输入（Pyannote需要特定格式）
        # 创建临时文件用于Pyannote处理
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio_array, sr)

        try:
            # 执行VAD
            vad_result = pipeline(temp_path)

            # V3.1.0: 收集原始时间戳（秒 -> 采样点）
            raw_timestamps = []
            for speech in vad_result.get_timeline().support():
                raw_timestamps.append({
                    'start': int(speech.start * sr),
                    'end': int(speech.end * sr)
                })

            # V3.1.0: 预处理 - 强制拆分超长原始片段
            # 使用软上限（12秒）作为拆分阈值，确保Smart Accumulation有足够的断点可用
            raw_timestamps = self._break_long_segments(
                raw_timestamps,
                audio_array,
                sr,
                max_duration=vad_config.smart_target_duration  # 使用软上限12秒
            )

            # V3.2.0+dev.20260131d: 在 Smart Accumulation 之前先修复高 RMS 边界
            raw_timestamps = self._fix_high_rms_boundaries_raw(
                raw_timestamps, audio_array, sr
            )

            # V3.2.0+dev.20260131: 使用统一的 Smart Accumulation 方法
            # V3.2.0+dev.20260131b: 传入音频数据用于 RMS 验证
            segments_metadata = self._smart_accumulation(
                raw_timestamps, sr, vad_config, audio_array
            )

            self.logger.info(f"Pyannote VAD 检测完成: {len(segments_metadata)} 个语音段")
            return segments_metadata

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _smart_accumulation(
        self,
        speech_timestamps: List[Dict],
        sr: int,
        vad_config: VADConfig,
        audio_array: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        V3.2.0+dev.20260131: 重构 Smart Accumulation 智能累积合并
        V3.2.0+dev.20260131b: 新增 RMS 验证，拒绝在高能量点切分

        相比旧版的改进：
        1. 添加"回溯找最佳断点"逻辑：在接近软上限时，回溯查找最佳断点
        2. 记录断点历史：跟踪每个潜在断点的 gap 大小和边界 RMS
        3. 标记 _smart_accumulated：防止 Post-VAD 合并重复处理
        4. RMS 验证：拒绝在高能量（正在说话）位置切分

        算法核心：
        - 软上限（TARGET）：Whisper 最佳处理时长（默认 12s）
        - 硬上限（MAX）：Whisper 物理限制（默认 30s）
        - 最小断点间隔（MIN_GAP）：用于判断"合适的断点"（默认 0.3s）
        - RMS 阈值：边界 RMS 超过此值视为不良断点（默认 0.1）

        断点选择策略：
        1. 硬上限：绝对不能超过 MAX
        2. 预警阈值：当接近 TARGET 时（80%），开始记录断点
        3. 最佳断点：在 TARGET 到 MAX*1.25 区间，选择 gap 最大且 RMS 最低的断点

        Args:
            speech_timestamps: VAD 原始输出的时间戳列表（采样点单位）
            sr: 采样率
            vad_config: VAD 配置
            audio_array: 音频数据（用于 RMS 验证，可选）

        Returns:
            List[Dict]: 智能累积后的分段元数据列表
        """
        TARGET = vad_config.smart_target_duration    # 默认 12.0s
        MAX = vad_config.smart_max_duration          # 默认 30.0s
        MIN_GAP = vad_config.smart_min_gap_to_split  # 默认 0.3s
        
        # V3.2.0+dev.20260131b: RMS 验证参数
        RMS_THRESHOLD = 0.1     # RMS 超过此值视为"正在说话"，不应在此处切分
        RMS_PENALTY = 0.5       # 高 RMS 断点的惩罚系数（降低其得分）
        RMS_WINDOW_MS = 50      # RMS 计算窗口（毫秒）
        rms_window_samples = int(RMS_WINDOW_MS / 1000 * sr)
        
        def compute_boundary_rms(end_sec: Optional[float], next_start_sec: float) -> float:
            """计算断点边界的 RMS 值（取前段结尾和后段开头的最大值）"""
            if audio_array is None or end_sec is None:
                return 0.0
            try:
                # 前段结尾 RMS
                end_sample = int(end_sec * sr)
                end_window = audio_array[max(0, end_sample - rms_window_samples):end_sample]
                rms_end = float(np.sqrt(np.mean(end_window ** 2))) if len(end_window) > 0 else 0.0
                
                # 后段开头 RMS
                start_sample = int(next_start_sec * sr)
                start_window = audio_array[start_sample:start_sample + rms_window_samples]
                rms_start = float(np.sqrt(np.mean(start_window ** 2))) if len(start_window) > 0 else 0.0
                
                return max(rms_end, rms_start)
            except Exception:
                return 0.0
        
        def compute_breakpoint_score(gap: float, rms: float) -> float:
            """计算断点质量得分：gap 越大越好，RMS 越低越好"""
            # 基础得分 = gap 大小
            score = gap
            # 如果 RMS 过高，施加惩罚
            if rms > RMS_THRESHOLD:
                score *= RMS_PENALTY
            return score
        
        # 预警阈值：从 TARGET 的 80% 开始记录断点（例如 9.6s）
        EARLY_WARNING_RATIO = 0.8
        early_warning_threshold = TARGET * EARLY_WARNING_RATIO

        segments_metadata = []
        
        # 当前累积段的信息
        current_start = None
        current_end = None
        
        # V3.2.0+dev.20260131b: 断点历史记录
        # 格式: [(切分位置(秒), gap大小(秒), 切分后的当前段时长, 边界RMS, 下一段起始时间)]
        breakpoint_history = []

        for ts in speech_timestamps:
            start_sec = ts['start'] / sr
            end_sec = ts['end'] / sr

            if current_start is None:
                # 第一个片段，直接初始化
                current_start = start_sec
                current_end = end_sec
                continue

            # 计算关键指标
            gap = start_sec - current_end                # VAD断点间隔
            combined_duration = end_sec - current_start  # 合并后的总时长
            current_duration = current_end - current_start  # 当前段时长

            # ========== 策略1: 硬上限检查 ==========
            # 绝对不能超30秒（Whisper物理限制）
            if combined_duration > MAX:
                segments_metadata.append({
                    "index": len(segments_metadata),
                    "start": current_start,
                    "end": current_end,
                    "mode": "memory",
                    "_smart_accumulated": True  # 标记：已经过 Smart Accumulation
                })
                self.logger.debug(
                    f"Smart Accumulation: 硬上限截断 [{current_start:.2f}s - {current_end:.2f}s] "
                    f"({current_duration:.2f}s), gap={gap:.2f}s"
                )
                current_start = start_sec
                current_end = end_sec
                breakpoint_history.clear()
                continue

            # ========== 策略2: 记录断点历史 ==========
            # 当当前段接近软上限时，开始记录潜在断点
            if current_duration >= early_warning_threshold and gap >= MIN_GAP:
                # V3.2.0+dev.20260131b: 计算边界 RMS 并记录
                boundary_rms = compute_boundary_rms(current_end, start_sec)
                score = compute_breakpoint_score(gap, boundary_rms)
                breakpoint_history.append((current_end, gap, current_duration, boundary_rms, start_sec, score))

            # ========== 策略3: 软上限触发 - 回溯找最佳断点 ==========
            # 当合并后超过软上限的 1.25 倍（15秒），触发截断决策
            if combined_duration > TARGET * 1.25:
                # V3.2.0+dev.20260131b: 选择得分最高的断点（考虑 gap 和 RMS）
                if breakpoint_history:
                    # 按得分排序，选择最佳断点
                    best_breakpoint = max(breakpoint_history, key=lambda x: x[5])  # x[5] = score
                    split_end, best_gap, split_duration, split_rms, next_start, _ = best_breakpoint
                    
                    self.logger.debug(
                        f"Smart Accumulation: 回溯最佳断点 [{current_start:.2f}s - {split_end:.2f}s] "
                        f"({split_duration:.2f}s), gap={best_gap:.2f}s, rms={split_rms:.4f}, "
                        f"candidates={len(breakpoint_history)}"
                    )
                    
                    segments_metadata.append({
                        "index": len(segments_metadata),
                        "start": current_start,
                        "end": split_end,
                        "mode": "memory",
                        "_smart_accumulated": True
                    })
                    
                    # 重新计算：从最佳断点之后的片段重新累积
                    # V3.2.0+dev.20260131b: 使用记录的下一段起始时间
                    current_start = next_start
                    current_end = end_sec
                    breakpoint_history.clear()
                else:
                    # 没有记录到好的断点，直接在当前位置截断
                    segments_metadata.append({
                        "index": len(segments_metadata),
                        "start": current_start,
                        "end": current_end,
                        "mode": "memory",
                        "_smart_accumulated": True
                    })
                    self.logger.debug(
                        f"Smart Accumulation: 软上限强制截断 [{current_start:.2f}s - {current_end:.2f}s] "
                        f"({current_duration:.2f}s), 无可用断点"
                    )
                    current_start = start_sec
                    current_end = end_sec
                    breakpoint_history.clear()
                continue

            # ========== 策略4: 软上限 + 良好断点 ==========
            # 当前段已达软上限，且当前 gap 足够大
            if current_duration >= TARGET and gap >= MIN_GAP:
                # V3.2.0+dev.20260131b: 计算当前断点的 RMS 和得分
                current_rms = compute_boundary_rms(current_end, start_sec)
                current_score = compute_breakpoint_score(gap, current_rms)
                
                # 检查是否有更好的历史断点
                if breakpoint_history:
                    best_history_score = max(bp[5] for bp in breakpoint_history)
                    # 如果当前得分是最好的，或者差距不大，就在当前位置断开
                    if current_score >= best_history_score * 0.8:
                        # 当前位置是好的断点
                        segments_metadata.append({
                            "index": len(segments_metadata),
                            "start": current_start,
                            "end": current_end,
                            "mode": "memory",
                            "_smart_accumulated": True
                        })
                        self.logger.debug(
                            f"Smart Accumulation: 软上限断点截断 [{current_start:.2f}s - {current_end:.2f}s] "
                            f"({current_duration:.2f}s), gap={gap:.2f}s, rms={current_rms:.4f}"
                        )
                        current_start = start_sec
                        current_end = end_sec
                        breakpoint_history.clear()
                        continue
                    else:
                        # V3.2.0+dev.20260131b: 历史断点更好，使用历史断点
                        best_breakpoint = max(breakpoint_history, key=lambda x: x[5])
                        split_end, best_gap, split_duration, split_rms, next_start, _ = best_breakpoint
                        
                        segments_metadata.append({
                            "index": len(segments_metadata),
                            "start": current_start,
                            "end": split_end,
                            "mode": "memory",
                            "_smart_accumulated": True
                        })
                        self.logger.debug(
                            f"Smart Accumulation: 使用更优历史断点 [{current_start:.2f}s - {split_end:.2f}s] "
                            f"({split_duration:.2f}s), gap={best_gap:.2f}s, rms={split_rms:.4f} "
                            f"(current_rms={current_rms:.4f} too high)"
                        )
                        current_start = next_start
                        current_end = end_sec
                        breakpoint_history.clear()
                        continue
                else:
                    # 没有历史记录，检查当前 RMS 是否可接受
                    if current_rms <= RMS_THRESHOLD:
                        # 当前 RMS 可接受
                        segments_metadata.append({
                            "index": len(segments_metadata),
                            "start": current_start,
                            "end": current_end,
                            "mode": "memory",
                            "_smart_accumulated": True
                        })
                        self.logger.debug(
                            f"Smart Accumulation: 软上限断点截断 [{current_start:.2f}s - {current_end:.2f}s] "
                            f"({current_duration:.2f}s), gap={gap:.2f}s, rms={current_rms:.4f}"
                        )
                        current_start = start_sec
                        current_end = end_sec
                        breakpoint_history.clear()
                        continue
                    else:
                        # V3.2.0+dev.20260131b: 当前 RMS 过高，记录但不立即切分
                        # 继续累积，希望后面有更好的断点
                        self.logger.debug(
                            f"Smart Accumulation: 跳过高RMS断点 at {current_end:.2f}s, "
                            f"gap={gap:.2f}s, rms={current_rms:.4f} > {RMS_THRESHOLD}"
                        )
                        breakpoint_history.append((current_end, gap, current_duration, current_rms, start_sec, current_score))

            # ========== 继续累积 ==========
            current_end = end_sec

        # 保存最后一段
        if current_start is not None:
            segments_metadata.append({
                "index": len(segments_metadata),
                "start": current_start,
                "end": current_end,
                "mode": "memory",
                "_smart_accumulated": True
            })

        return segments_metadata

    def _compute_rms_at_boundary(
        self,
        audio_array: np.ndarray,
        position_samples: int,
        sr: int,
        window_seconds: float,
        boundary_type: str  # 'start' or 'end'
    ) -> float:
        """
        计算边界位置的 RMS 能量值
        
        Args:
            audio_array: 音频数据
            position_samples: 边界位置 (采样点)
            sr: 采样率
            window_seconds: RMS 计算窗口 (秒)
            boundary_type: 'start' (向后) 或 'end' (向前)
            
        Returns:
            RMS 值
        """
        window_samples = int(window_seconds * sr)
        
        if boundary_type == 'end':
            # 段结束:向前取窗口
            start = max(0, position_samples - window_samples)
            end = position_samples
        else:  # 'start'
            # 段开始:向后取窗口
            start = position_samples
            end = min(len(audio_array), position_samples + window_samples)
        
        if start >= end:
            return 0.0
            
        window_audio = audio_array[start:end]
        if len(window_audio) == 0:
            return 0.0
            
        return float(np.sqrt(np.mean(window_audio.astype(np.float32) ** 2)))

    def _fix_high_rms_boundaries_raw(
        self,
        speech_timestamps: List[Dict],
        audio_array: np.ndarray,
        sr: int
    ) -> List[Dict]:
        """
        V3.2.0+dev.20260131d: 修复 Silero VAD 原始输出中的高 RMS 边界
        
        在 Smart Accumulation 之前处理,因为:
        1. Silero VAD 在活跃语音中误判静音 (RMS=0.13-0.15)
        2. Smart Accumulation 会调整边界,丢失原始 RMS 信息
        3. 必须在原始输出上修复才能捕获真实的高能量误判
        
        Args:
            speech_timestamps: Silero VAD 原始输出 [{'start': int, 'end': int}, ...]
            audio_array: 完整音频数据 (16kHz)
            sr: 采样率 (必须是16000)
            
        Returns:
            修复后的 speech_timestamps
        """
        if len(speech_timestamps) <= 1:
            return speech_timestamps
            
        RMS_THRESHOLD = 0.1  # 高能量阈值
        MAX_GAP_FOR_MERGE = 0.5  # 最大合并间隙 (秒)
        RMS_WINDOW = 0.05  # RMS 计算窗口 (50ms)
        
        self.logger.info(
            f"[RMS边界修复-原始] 开始处理 {len(speech_timestamps)} 个 Silero VAD 原始段, "
            f"audio_array.shape={audio_array.shape}, sr={sr}"
        )
        
        result = []
        i = 0
        merged_count = 0
        
        while i < len(speech_timestamps):
            current = speech_timestamps[i]
            
            # 尝试与下一个段合并
            if i + 1 < len(speech_timestamps):
                next_seg = speech_timestamps[i + 1]
                
                # 计算间隙
                gap_samples = next_seg['start'] - current['end']
                gap_seconds = gap_samples / sr
                
                # 计算边界 RMS
                rms_end = self._compute_rms_at_boundary(
                    audio_array, current['end'], sr, RMS_WINDOW, 'end'
                )
                rms_start = self._compute_rms_at_boundary(
                    audio_array, next_seg['start'], sr, RMS_WINDOW, 'start'
                )
                
                self.logger.debug(
                    f"[RMS边界检查-原始] Chunk {i} -> {i+1}: "
                    f"gap={gap_seconds:.3f}s, rms_end={rms_end:.4f}, rms_start={rms_start:.4f}"
                )
                
                # 判断是否需要合并
                should_merge = (
                    gap_seconds <= MAX_GAP_FOR_MERGE and
                    (rms_end > RMS_THRESHOLD or rms_start > RMS_THRESHOLD)
                )
                
                if should_merge:
                    # 合并
                    merged = {
                        'start': current['start'],
                        'end': next_seg['end']
                    }
                    result.append(merged)
                    merged_count += 1
                    
                    self.logger.info(
                        f"[RMS边界修复-原始] 合并了 Chunk {i} 和 {i+1}: "
                        f"[{current['start']/sr:.2f}s - {current['end']/sr:.2f}s] + "
                        f"[{next_seg['start']/sr:.2f}s - {next_seg['end']/sr:.2f}s] -> "
                        f"[{merged['start']/sr:.2f}s - {merged['end']/sr:.2f}s], "
                        f"RMS_end={rms_end:.4f}, RMS_start={rms_start:.4f}, gap={gap_seconds:.3f}s"
                    )
                    
                    i += 2  # 跳过已合并的两个段
                    continue
            
            # 不合并,保留原段
            result.append(current)
            i += 1
        
        self.logger.info(
            f"[RMS边界修复-原始] 完成: {len(speech_timestamps)} -> {len(result)} 段 "
            f"(合并了 {merged_count} 对)"
        )
        
        return result

    def _fix_high_rms_boundaries(
        self,
        segments: List[Dict],
        audio_array: np.ndarray,
        sr: int,
        vad_config: VADConfig
    ) -> List[Dict]:
        """
        V3.2.0+dev.20260131c: RMS 边界修复
        
        解决 Silero VAD 在活跃语音中误判静音的问题。
        当两个相邻段的边界 RMS 都很高时，将它们合并。
        
        典型案例："附近品川" / "医院进行抢救" 
        - Silero 在"品川"和"医院"之间检测到了静音
        - 但实际 RMS 表明那里仍有语音
        - 需要将这两段合并
        
        Args:
            segments: 经过 Smart Accumulation 的分段列表
            audio_array: 音频数据
            sr: 采样率
            vad_config: VAD 配置
            
        Returns:
            修复后的分段列表
        """
        if len(segments) <= 1:
            return segments
        
        self.logger.info(f"[RMS边界修复] 开始处理 {len(segments)} 个分段, audio_array.shape={audio_array.shape}, sr={sr}")
        
        # RMS 阈值：边界 RMS 超过此值视为"高能量边界"
        RMS_THRESHOLD = 0.1
        # 合并后的最大时长限制
        MAX_MERGED_DURATION = vad_config.smart_max_duration  # 30s
        # RMS 计算窗口
        RMS_WINDOW_MS = 50
        rms_window_samples = int(RMS_WINDOW_MS / 1000 * sr)
        
        def compute_rms(sample_pos: int) -> float:
            """计算指定位置的 RMS"""
            try:
                window = audio_array[max(0, sample_pos - rms_window_samples // 2):
                                    sample_pos + rms_window_samples // 2]
                return float(np.sqrt(np.mean(window ** 2))) if len(window) > 0 else 0.0
            except Exception:
                return 0.0
        
        merged = []
        current = segments[0].copy()
        merge_count = 0
        
        for next_seg in segments[1:]:
            gap = next_seg['start'] - current['end']
            merged_duration = next_seg['end'] - current['start']
            
            # 计算边界 RMS
            end_sample = int(current['end'] * sr)
            start_sample = int(next_seg['start'] * sr)
            rms_at_boundary_end = compute_rms(end_sample)
            rms_at_boundary_start = compute_rms(start_sample)
            
            # 调试日志：显示每对边界的检查结果
            self.logger.debug(
                f"[RMS边界检查] Chunk {current.get('index', '?')} -> {next_seg.get('index', '?')}: "
                f"gap={gap:.3f}s, rms_end={rms_at_boundary_end:.4f}, rms_start={rms_at_boundary_start:.4f}, "
                f"merged_dur={merged_duration:.2f}s"
            )
            
            # 合并条件：
            # 1. 边界 RMS 都很高（都超过阈值）
            # 2. 间隔较小（< 0.5s）
            # 3. 合并后不超过硬上限
            should_merge = (
                rms_at_boundary_end > RMS_THRESHOLD and
                rms_at_boundary_start > RMS_THRESHOLD and
                gap < 0.5 and
                merged_duration <= MAX_MERGED_DURATION
            )
            
            if should_merge:
                # 合并：扩展当前段的结束时间
                self.logger.debug(
                    f"RMS边界修复: 合并 [{current['start']:.2f}s-{current['end']:.2f}s] + "
                    f"[{next_seg['start']:.2f}s-{next_seg['end']:.2f}s] "
                    f"(rms_end={rms_at_boundary_end:.4f}, rms_start={rms_at_boundary_start:.4f})"
                )
                current['end'] = next_seg['end']
                merge_count += 1
            else:
                # 不合并：保存当前段，开始新段
                merged.append(current)
                current = next_seg.copy()
        
        # 保存最后一段
        merged.append(current)
        
        # 重新编号
        for i, seg in enumerate(merged):
            seg['index'] = i
        
        if merge_count > 0:
            self.logger.info(
                f"RMS边界修复: 合并了 {merge_count} 对高能量边界段 "
                f"({len(segments)} -> {len(merged)} 段)"
            )
        
        return merged

    def _energy_based_split(
        self,
        audio_array: np.ndarray,
        sr: int,
        chunk_size: int = 30
    ) -> List[Dict]:
        """
        基于能量的简易分段（降级方案）

        当VAD模型加载失败时使用，按固定时长分段。
        会尝试在静音处分割以避免切断语音。

        Args:
            audio_array: 音频数组
            sr: 采样率
            chunk_size: 每段最大长度（秒）

        Returns:
            List[Dict]: 分段元数据列表
        """
        self.logger.warning("使用能量检测降级分段（固定时长）")

        total_duration = len(audio_array) / sr
        segments_metadata = []
        pos = 0.0

        while pos < total_duration:
            # 计算理想结束位置
            ideal_end = min(pos + chunk_size, total_duration)

            # 尝试在静音处分割（在理想结束点前后1秒范围内寻找）
            if ideal_end < total_duration:
                search_start = max(pos, ideal_end - 1.0)
                search_end = min(total_duration, ideal_end + 1.0)

                # 计算搜索范围内的能量
                start_sample = int(search_start * sr)
                end_sample = int(search_end * sr)
                search_audio = audio_array[start_sample:end_sample]

                if len(search_audio) > 0:
                    # 计算短时能量（每100ms一个窗口）
                    window_size = int(0.1 * sr)
                    energies = []
                    for i in range(0, len(search_audio) - window_size, window_size):
                        window = search_audio[i:i + window_size]
                        energy = np.sum(window ** 2)
                        energies.append((i, energy))

                    if energies:
                        # 找到能量最低的点
                        min_energy_idx = min(energies, key=lambda x: x[1])[0]
                        actual_end = search_start + (min_energy_idx / sr)
                        # 确保分段至少有1秒
                        if actual_end - pos >= 1.0:
                            ideal_end = actual_end

            segments_metadata.append({
                "index": len(segments_metadata),
                "start": pos,
                "end": ideal_end,
                "mode": "memory"
            })
            pos = ideal_end

        self.logger.info(f"能量检测分段完成: {len(segments_metadata)}段")
        return segments_metadata

    def merge_vad_segments(
        self,
        segments: List[Dict],
        config: VADConfig
    ) -> List[Dict]:
        """
        Post-VAD 智能合并层

        迁移自旧架构 transcription_service.py:_merge_vad_segments (2025-12-17)
        V3.2.0+dev.20260131: 修复与 Smart Accumulation 的冲突

        策略：宁可错合（依赖 SentenceSplitter 分句），不可错分（导致 ASR 丢失上下文）。

        合并条件1：基础合并 - 间隔小且总长不超标
        合并条件2：碎片保护 - 当前段极短（可能是被切断的单词），强制合并

        V3.2.0+dev.20260131 修复：
        - 检测 _smart_accumulated 标记，跳过已经过 Smart Accumulation 的片段
        - Smart Accumulation 已经做了最优断点决策，不应被二次合并破坏

        Args:
            segments: VAD 切分后的原始片段列表 [{start, end, index, mode}, ...]
            config: VAD 配置，包含合并参数

        Returns:
            合并后的片段列表
        """
        if not segments:
            return []

        # V3.2.0+dev.20260131: 检测是否已经过 Smart Accumulation
        # 如果所有片段都标记了 _smart_accumulated，则跳过合并
        smart_accumulated_count = sum(
            1 for seg in segments if seg.get('_smart_accumulated', False)
        )
        if smart_accumulated_count == len(segments):
            self.logger.debug(
                f"Post-VAD 合并: 跳过（所有 {len(segments)} 段已经过 Smart Accumulation）"
            )
            # 清理内部标记并重新编号
            for i, seg in enumerate(segments):
                seg['index'] = i
                seg.pop('_smart_accumulated', None)
            return segments

        max_gap = config.merge_max_gap            # 默认 1.0s
        max_duration = config.merge_max_duration  # 默认 12.0s
        min_fragment = config.merge_min_fragment  # 默认 1.0s

        merged = []
        current = segments[0].copy()

        for next_seg in segments[1:]:
            # V3.2.0+dev.20260131: 如果当前段已经过 Smart Accumulation，不合并
            if current.get('_smart_accumulated', False):
                merged.append(current)
                current = next_seg.copy()
                continue

            gap = next_seg['start'] - current['end']
            current_duration = current['end'] - current['start']
            combined_duration = next_seg['end'] - current['start']

            should_merge = False

            # 条件 1: 基础合并（间隔小且总长不超标）
            if gap <= max_gap and combined_duration <= max_duration:
                should_merge = True

            # 条件 2: 碎片保护（当前段极短，可能是被切断的单词）
            # 例如: "It's" (0.5s) ... [gap 1.5s] ... "only..."
            elif current_duration < min_fragment and combined_duration <= max_duration:
                # 限制 gap 不超过 3s，避免引入过长静音
                if gap < 3.0:
                    self.logger.debug(
                        f"碎片强制合并: fragment={current_duration:.2f}s, gap={gap:.2f}s"
                    )
                    should_merge = True

            if should_merge:
                current['end'] = next_seg['end']
                # V3.2.0+dev.20260131: 如果合并了 Smart Accumulated 段，继承标记
                if next_seg.get('_smart_accumulated', False):
                    current['_smart_accumulated'] = True
            else:
                merged.append(current)
                current = next_seg.copy()

        merged.append(current)

        # 重新编号并清理内部标记
        for i, seg in enumerate(merged):
            seg['index'] = i
            seg.pop('_smart_accumulated', None)  # V3.2.0+dev.20260131: 清理内部标记

        self.logger.info(
            f"Post-VAD 智能合并: 原始 {len(segments)} -> 合并后 {len(merged)} 段 "
            f"(max_gap={max_gap}s, max_dur={max_duration}s)"
        )
        return merged

    def _break_long_segments(
        self,
        timestamps: List[Dict],
        audio_array: np.ndarray,
        sr: int,
        max_duration: float
    ) -> List[Dict]:
        """
        V3.1.0: 强制拆分超长片段
        V3.2.0+dev.20260131: 修复能量检测方法，使用 RMS 窗口替代单点采样

        在获取到VAD原始输出后，检查每个片段是否超过max_duration。
        如果超过，在片段中间（60%-95%区域）寻找能量最低点进行拆分。

        Args:
            timestamps: VAD原始输出的时间戳列表（采样点单位）
            audio_array: 完整音频数组
            sr: 采样率
            max_duration: 最大允许时长（秒）

        Returns:
            拆分后的时间戳列表
        """
        refined_timestamps = []
        max_samples = int(max_duration * sr)

        # V3.2.0+dev.20260131: RMS 窗口参数
        rms_window_ms = 50  # 50ms RMS 窗口
        rms_window_samples = int(rms_window_ms / 1000 * sr)
        rms_hop_samples = rms_window_samples // 2  # 50% 重叠
        # 拆分点前后的微小间隙（防止被 Smart Accumulation 重新合并）
        split_gap_ms = 30  # 30ms 间隙
        split_gap_samples = int(split_gap_ms / 1000 * sr)
        
        # V3.2.0+dev.20260131b: RMS 阈值 - 超过此值视为"正在说话"，不应切分
        RMS_THRESHOLD_FOR_SPLIT = 0.08  # 比 Smart Accumulation 更严格

        for ts in timestamps:
            start = ts['start']
            end = ts['end']
            duration_samples = end - start

            if duration_samples <= max_samples:
                # 正常长度，直接保留
                refined_timestamps.append(ts)
                continue

            # 处理超长片段
            self.logger.debug(
                f"发现超长 VAD 片段: {duration_samples/sr:.2f}s > {max_duration}s，执行强制拆分"
            )

            # 使用while循环处理超长片段（可能需要切成多段）
            current_seg_start = start
            remaining_end = end

            while (remaining_end - current_seg_start) > max_samples:
                # V3.2.0+dev.20260131b: 多阶段搜索策略
                # 阶段1: 在 60%-95% 区域找低 RMS 点
                # 阶段2: 如果失败，扩大到 40%-95% 区域
                # 阶段3: 如果仍然失败，接受硬上限内的超长段
                
                split_point = None
                min_rms = float('inf')
                
                for search_phase, (min_ratio, max_ratio) in enumerate([
                    (0.6, 0.95),   # 阶段1: 60%-95%
                    (0.4, 0.95),   # 阶段2: 40%-95%
                    (0.2, 0.95),   # 阶段3: 20%-95%
                ], 1):
                    search_min_offset = int(max_samples * min_ratio)
                    search_max_offset = int(max_samples * max_ratio)
                    
                    search_start = current_seg_start + search_min_offset
                    search_end = min(current_seg_start + search_max_offset, remaining_end)

                    # V3.2.0+dev.20260131: 使用 RMS 窗口能量检测
                    if search_end > search_start + rms_window_samples:
                        segment_view = audio_array[search_start:search_end]

                        # 计算滑动窗口 RMS 能量
                        rms_values = []
                        for i in range(0, len(segment_view) - rms_window_samples, rms_hop_samples):
                            window = segment_view[i:i + rms_window_samples]
                            rms = np.sqrt(np.mean(window ** 2))
                            center_idx = i + rms_window_samples // 2
                            rms_values.append((center_idx, rms))

                        if rms_values:
                            # 找 RMS 最低的点
                            candidate_idx, candidate_rms = min(rms_values, key=lambda x: x[1])
                            
                            # V3.2.0+dev.20260131b: 检查 RMS 是否足够低
                            if candidate_rms < RMS_THRESHOLD_FOR_SPLIT:
                                split_point = search_start + candidate_idx
                                min_rms = candidate_rms
                                self.logger.debug(
                                    f"  阶段{search_phase}: 找到合适切分点 RMS={min_rms:.4f} at {split_point/sr:.2f}s"
                                )
                                break
                            elif candidate_rms < min_rms:
                                # 记录目前为止的最佳候选
                                split_point = search_start + candidate_idx
                                min_rms = candidate_rms
                                self.logger.debug(
                                    f"  阶段{search_phase}: RMS={candidate_rms:.4f} 仍然过高 (>{RMS_THRESHOLD_FOR_SPLIT})，继续搜索"
                                )
                
                # 检查是否找到了可接受的切分点
                if min_rms >= RMS_THRESHOLD_FOR_SPLIT:
                    # V3.2.0+dev.20260131b: 无法找到好的切分点
                    remaining_duration = (remaining_end - current_seg_start) / sr
                    hard_max = 30.0  # Whisper 硬上限
                    
                    if remaining_duration <= hard_max:
                        # 剩余段在硬上限内，直接保留整段（不再强制拆分）
                        self.logger.warning(
                            f"  无法找到低RMS切分点 (min_rms={min_rms:.4f})，保留超长段 "
                            f"({remaining_duration:.2f}s <= {hard_max}s)"
                        )
                        refined_timestamps.append({'start': current_seg_start, 'end': remaining_end})
                        break
                    else:
                        # 超过硬上限，必须切分
                        if split_point is None:
                            # 没有找到任何候选点，硬切
                            split_point = current_seg_start + max_samples
                        self.logger.warning(
                            f"  必须切分（超硬上限），使用次优点 RMS={min_rms:.4f} at {split_point/sr:.2f}s"
                        )

                # 此时 split_point 保证不为 None（要么找到了好点，要么用了硬切点）
                assert split_point is not None, "split_point should not be None at this point"
                
                # V3.2.0+dev.20260131: 添加前半段（结束点略微提前，创建间隙）
                seg_end = max(current_seg_start + 1, split_point - split_gap_samples)
                refined_timestamps.append({'start': current_seg_start, 'end': seg_end})
                self.logger.debug(
                    f"  强制拆分: [{current_seg_start/sr:.2f}s - {seg_end/sr:.2f}s] "
                    f"({(seg_end - current_seg_start)/sr:.2f}s)"
                )

                # V3.2.0+dev.20260131: 更新后半段起点（略微延后，创建间隙）
                current_seg_start = min(split_point + split_gap_samples, remaining_end)

            # 添加最后剩余的一段
            if current_seg_start < remaining_end:
                refined_timestamps.append({'start': current_seg_start, 'end': remaining_end})
                self.logger.debug(
                    f"  强制拆分: [{current_seg_start/sr:.2f}s - {remaining_end/sr:.2f}s] "
                    f"({(remaining_end - current_seg_start)/sr:.2f}s)"
                )

        if len(refined_timestamps) > len(timestamps):
            self.logger.info(
                f"强制拆分完成: {len(timestamps)} 个原始片段 -> {len(refined_timestamps)} 个片段"
            )

        return refined_timestamps


# 便捷函数
def get_vad_service(logger: Optional[logging.Logger] = None) -> VADService:
    """
    获取 VAD 服务实例

    Args:
        logger: 日志记录器

    Returns:
        VADService 实例
    """
    return VADService(logger=logger)
