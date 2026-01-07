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
from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import torch


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
    1. 使用 cpu_optimizer 智能计算线程数（而非硬编码为 1）
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

        # 使用 cpu_optimizer 获取优化的 SessionOptions
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
            from app.utils.cpu_optimizer import ONNXThreadOptimizer

            # 使用 cpu_optimizer 计算最优线程数
            optimal_threads, info = ONNXThreadOptimizer.calculate_optimal_threads(
                usage_ratio=0.6  # 使用 60% 核心
            )

            # 获取配置好的 SessionOptions
            sess_options = ONNXThreadOptimizer.get_onnx_session_options(
                optimal_threads=optimal_threads
            )

            self.logger.info(
                f"Silero VAD ONNX 优化: {optimal_threads} 线程 "
                f"({info.get('strategy', 'unknown')})"
            )

            return sess_options

        except Exception as e:
            # 回退：使用默认配置
            import onnxruntime as ort
            self.logger.warning(f"cpu_optimizer 不可用，使用默认配置: {e}")

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
            from app.utils.cpu_optimizer import CPUArchitectureDetector
            import psutil

            detector = CPUArchitectureDetector()
            cpu_name = None

            # 尝试获取 CPU 名称
            try:
                from app.services.hardware_service import get_hardware_detector
                hw = get_hardware_detector().detect()
                cpu_name = hw.cpu_name
            except:
                pass

            # 检测是否为 Intel 混合架构
            if not detector.is_intel_hybrid_architecture(cpu_name):
                self.logger.debug("非 Intel 混合架构，跳过 P-Core 亲和性设置")
                return

            # 获取 P-Core 数量
            physical_cores = detector.get_physical_cores()
            p_cores = detector.detect_intel_p_cores(cpu_name, physical_cores)

            # P-Core 通常是前 N 个逻辑核心（每个 P-Core 有 2 个超线程）
            # 例如 i7-12700: P-Core 0-7 对应逻辑核心 0-15
            p_core_logical_ids = list(range(p_cores * 2))

            # 设置当前进程的 CPU 亲和性
            process = psutil.Process()
            process.cpu_affinity(p_core_logical_ids)

            self.logger.info(
                f"Silero VAD P-Core 亲和性: 绑定到核心 {p_core_logical_ids[:4]}... "
                f"(共 {len(p_core_logical_ids)} 个逻辑核心)"
            )

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
        enable_merge: bool = True
    ) -> List[Dict]:
        """
        检测语音段

        Args:
            audio_array: 音频数组
            sr: 采样率
            config: VAD 配置
            enable_merge: 是否启用 Post-VAD 智能合并（默认启用）

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

        # 使用项目内置的 ONNX 模型
        builtin_model_path = Path(__file__).parent.parent.parent / "assets" / "silero" / "silero_vad.onnx"

        if not builtin_model_path.exists():
            raise FileNotFoundError(
                f"内置Silero VAD模型不存在: {builtin_model_path}\n"
                "请确保项目完整，或重新从源码仓库获取"
            )

        self.logger.info(f"使用内置模型: {builtin_model_path}")

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

        # V3.1.0: Smart Accumulation 智能累积合并
        # 软上限（甜蜜点）和硬上限（物理极限）双重标准
        TARGET = vad_config.smart_target_duration    # 默认 12.0s
        MAX = vad_config.smart_max_duration          # 默认 30.0s
        MIN_GAP = vad_config.smart_min_gap_to_split  # 默认 0.3s

        segments_metadata = []
        current_start = None
        current_end = None

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

            # 硬上限：绝对不能超30秒（Whisper物理限制）
            if combined_duration > MAX:
                segments_metadata.append({
                    "index": len(segments_metadata),
                    "start": current_start,
                    "end": current_end,
                    "mode": "memory"
                })
                self.logger.debug(
                    f"Smart Accumulation: 硬上限截断 [{current_start:.2f}s - {current_end:.2f}s] "
                    f"({current_duration:.2f}s), gap={gap:.2f}s"
                )
                current_start = start_sec
                current_end = end_sec

            # 软上限策略1：合并后显著超过目标（>15秒），立即截断
            elif combined_duration > TARGET * 1.25:  # 15秒
                segments_metadata.append({
                    "index": len(segments_metadata),
                    "start": current_start,
                    "end": current_end,
                    "mode": "memory"
                })
                self.logger.debug(
                    f"Smart Accumulation: 软上限预判截断 [{current_start:.2f}s - {current_end:.2f}s] "
                    f"({current_duration:.2f}s), 合并后会超过{TARGET * 1.25:.1f}s, gap={gap:.2f}s"
                )
                current_start = start_sec
                current_end = end_sec

            # 软上限策略2：当前段已达12秒，且有合适的断点（≥0.3s）
            elif current_duration >= TARGET and gap >= MIN_GAP:
                segments_metadata.append({
                    "index": len(segments_metadata),
                    "start": current_start,
                    "end": current_end,
                    "mode": "memory"
                })
                self.logger.debug(
                    f"Smart Accumulation: 软上限断点截断 [{current_start:.2f}s - {current_end:.2f}s] "
                    f"({current_duration:.2f}s), gap={gap:.2f}s"
                )
                current_start = start_sec
                current_end = end_sec

            # 继续累积：允许12-15秒的合理合并（如果gap很小）
            else:
                current_end = end_sec

        # 保存最后一段
        if current_start is not None:
            segments_metadata.append({
                "index": len(segments_metadata),
                "start": current_start,
                "end": current_end,
                "mode": "memory"
            })

        # 如果没有检测到任何语音段，按固定时长分段
        if len(segments_metadata) == 0:
            self.logger.warning("VAD未检测到语音，使用固定时长分段")
            return self._energy_based_split(audio_array, sr, vad_config.chunk_size)

        # V3.1.0: 统计 Smart Accumulation 效果
        durations = [seg['end'] - seg['start'] for seg in segments_metadata]
        avg_duration = sum(durations) / len(durations) if durations else 0
        max_duration = max(durations) if durations else 0

        self.logger.info(
            f"Silero VAD 检测完成: {len(segments_metadata)} 个语音段 "
            f"(avg={avg_duration:.2f}s, max={max_duration:.2f}s, target={TARGET}s)"
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
            from pyannote.audio import Pipeline
        except ImportError:
            raise RuntimeError("Pyannote未安装，请使用Silero VAD或安装pyannote-audio")

        # 初始化Pyannote VAD Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=vad_config.hf_token
        )

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

            # V3.1.0: Smart Accumulation 智能累积合并
            TARGET = vad_config.smart_target_duration
            MAX = vad_config.smart_max_duration
            MIN_GAP = vad_config.smart_min_gap_to_split

            segments_metadata = []
            current_start = None
            current_end = None

            for ts in raw_timestamps:
                start_sec = ts['start'] / sr
                end_sec = ts['end'] / sr

                if current_start is None:
                    current_start = start_sec
                    current_end = end_sec
                    continue

                gap = start_sec - current_end
                combined_duration = end_sec - current_start
                current_duration = current_end - current_start

                # 硬上限：绝对不能超30秒
                if combined_duration > MAX:
                    segments_metadata.append({
                        "index": len(segments_metadata),
                        "start": current_start,
                        "end": current_end,
                        "mode": "memory"
                    })
                    current_start = start_sec
                    current_end = end_sec

                # 软上限策略1：合并后显著超过目标（>15秒），立即截断
                elif combined_duration > TARGET * 1.25:  # 15秒
                    segments_metadata.append({
                        "index": len(segments_metadata),
                        "start": current_start,
                        "end": current_end,
                        "mode": "memory"
                    })
                    current_start = start_sec
                    current_end = end_sec

                # 软上限策略2：当前段已达12秒，且有合适的断点（≥0.3s）
                elif current_duration >= TARGET and gap >= MIN_GAP:
                    segments_metadata.append({
                        "index": len(segments_metadata),
                        "start": current_start,
                        "end": current_end,
                        "mode": "memory"
                    })
                    current_start = start_sec
                    current_end = end_sec

                # 继续累积：允许12-15秒的合理合并（如果gap很小）
                else:
                    current_end = end_sec

            # 保存最后一段
            if current_start is not None:
                segments_metadata.append({
                    "index": len(segments_metadata),
                    "start": current_start,
                    "end": current_end,
                    "mode": "memory"
                })

            self.logger.info(f"Pyannote VAD 检测完成: {len(segments_metadata)} 个语音段")
            return segments_metadata

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)

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

        策略：宁可错合（依赖 SentenceSplitter 分句），不可错分（导致 ASR 丢失上下文）。

        合并条件1：基础合并 - 间隔小且总长不超标
        合并条件2：碎片保护 - 当前段极短（可能是被切断的单词），强制合并

        Args:
            segments: VAD 切分后的原始片段列表 [{start, end, index, mode}, ...]
            config: VAD 配置，包含合并参数

        Returns:
            合并后的片段列表
        """
        if not segments:
            return []

        max_gap = config.merge_max_gap            # 默认 1.5s
        max_duration = config.merge_max_duration  # 默认 12.0s
        min_fragment = config.merge_min_fragment  # 默认 1.0s

        merged = []
        current = segments[0].copy()

        for next_seg in segments[1:]:
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
            else:
                merged.append(current)
                current = next_seg.copy()

        merged.append(current)

        # 重新编号
        for i, seg in enumerate(merged):
            seg['index'] = i

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
                # 目标：在60%-95%位置找到切分点（避免切在开头或结尾）
                search_min_offset = int(max_samples * 0.6)  # 18s
                search_max_offset = int(max_samples * 0.95)  # 28.5s

                search_start = current_seg_start + search_min_offset
                search_end = min(current_seg_start + search_max_offset, remaining_end)

                # 在搜索区间内找能量最低点（最接近静音）
                if search_end > search_start:
                    segment_view = audio_array[search_start:search_end]
                    # 找绝对值最小的点（最接近静音）
                    min_idx = np.argmin(np.abs(segment_view))
                    split_point = search_start + min_idx
                else:
                    # 极端情况：硬切
                    split_point = current_seg_start + max_samples

                # 添加前半段
                refined_timestamps.append({'start': current_seg_start, 'end': split_point})
                self.logger.debug(
                    f"  强制拆分: [{current_seg_start/sr:.2f}s - {split_point/sr:.2f}s] "
                    f"({(split_point - current_seg_start)/sr:.2f}s)"
                )

                # 更新后半段起点
                current_seg_start = split_point

            # 添加最后剩余的一段
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
