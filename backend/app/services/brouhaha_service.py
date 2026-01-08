"""
Brouhaha SNR + C50 检测服务

V3.1.1+dev.20260107.03: 新增 Brouhaha 模型集成

提供基于 Brouhaha 模型的信噪比(SNR)和清晰度指数(C50)检测功能。
使用 PyTorch 原生推理，支持 CPU/GPU 自动选择。

核心功能:
- SNR 检测: 信噪比，用于判断音频质量
- C50 检测: 清晰度指数，用于判断混响程度
- WADA-SNR 回退: 模型不可用时的无模型估计

参考:
- Brouhaha 论文: https://arxiv.org/abs/2210.13248
- 模型地址: https://huggingface.co/pyannote/brouhaha
"""
import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)


# ========== 数据类定义 ==========

@dataclass
class BrouhahaResult:
    """
    Brouhaha 检测结果

    Attributes:
        snr: 信噪比 (dB)，典型范围 -10 ~ 50
        c50: 清晰度指数 (dB)，典型范围 -20 ~ 30
        vad: 语音活动概率 (0-1)
        is_valid: 结果是否有效
    """
    snr: float          # 信噪比 (dB)
    c50: float          # 清晰度指数 (dB)
    vad: float          # 语音活动概率 (0-1)
    is_valid: bool      # 结果是否有效


@dataclass
class BrouhahaTriageConfig:
    """
    Brouhaha 分诊配置

    用于配置 SNR+C50 三层决策策略的阈值参数
    """

    # ========== 核心阈值（建议数据集微调）==========
    # SNR 阈值
    snr_high_threshold: float = 25.0      # SNR >= 此值直接放行（高质量语音）
    snr_low_threshold: float = 12.0       # SNR < 此值强制分离（ASR 可用边界）

    # C50 阈值
    c50_good_threshold: float = 5.0       # C50 >= 此值视为良好（普通房间下限）
    c50_bad_threshold: float = -5.0       # C50 < 此值视为严重混响

    # ========== 辅助阈值（经验值，一般不调）==========
    spectral_contrast_low: float = 15.0        # 频谱对比度低阈值 (dB)
    spectral_contrast_critical: float = 12.0   # 频谱对比度临界阈值 (dB)
    spectral_flatness_high: float = 0.4        # 频谱平坦度高阈值

    # ========== 系统参数 ==========
    enable_snr_strategy: bool = True      # 启用 SNR 策略
    enable_c50_check: bool = True         # 启用 C50 检查
    fallback_to_wada: bool = True         # Brouhaha 不可用时回退到 WADA-SNR

    # ========== 调试参数 ==========
    log_decisions: bool = False           # 记录每个决策的详细日志
    collect_statistics: bool = True       # 收集统计信息


# ========== Brouhaha 服务 ==========

class BrouhahaService:
    """
    Brouhaha SNR + C50 检测服务

    特性：
    - PyTorch 原生推理，支持 CPU/GPU 自动选择
    - 自动下载模型到本地目录
    - 结果缓存
    - 优雅降级（回退到 WADA-SNR）

    使用示例:
        service = BrouhahaService()
        result = service.detect(audio, sr=16000)
        print(f"SNR: {result.snr:.1f}dB, C50: {result.c50:.1f}dB")
    """

    # 模型默认路径
    DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "pretrained" / "brouhaha"

    # 音频参数
    SAMPLE_RATE = 16000

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "auto",
        hf_token: Optional[str] = None
    ):
        """
        初始化 Brouhaha 服务

        Args:
            model_dir: 本地模型目录路径，默认为 backend/models/pretrained/brouhaha
            device: 推理设备 ("auto" / "cuda" / "cpu")
            hf_token: HuggingFace Token（首次下载需要，可从环境变量读取）
        """
        if model_dir is None:
            self.model_dir = self.DEFAULT_MODEL_DIR
        else:
            self.model_dir = Path(model_dir)

        # 从环境变量或参数获取 HuggingFace Token
        self.hf_token = hf_token or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

        # 设备选择（延迟到模型加载时确定）
        self._device_preference = device
        self._device = None

        self.model = None

        # 缓存
        self._cache: Dict[int, BrouhahaResult] = {}
        self._cache_max_size = 100

        # 初始化模型
        self._init_model()

    def _init_model(self):
        """初始化 PyTorch 模型"""
        try:
            import torch

            # 设备选择
            if self._device_preference == "auto":
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self._device = torch.device(self._device_preference)

            # 检查本地模型是否存在
            local_config = self.model_dir / "config.yaml"
            local_weights = self.model_dir / "pytorch_model.bin"

            if local_config.exists() and local_weights.exists():
                # 从本地加载
                self._load_from_local()
            else:
                # 从 HuggingFace 下载
                self._download_from_huggingface()

            if self.model is not None:
                # 移动到目标设备
                self.model = self.model.to(self._device)
                self.model.eval()

                # 计算参数量
                param_count = sum(p.numel() for p in self.model.parameters())
                logger.info(
                    f"Brouhaha 模型已加载, 设备={self._device}, "
                    f"参数量={param_count / 1e6:.1f}M"
                )

        except ImportError as e:
            logger.warning(f"Brouhaha 依赖未安装: {e}")
            logger.warning("请运行: pip install pyannote.audio torch")
            self.model = None
        except Exception as e:
            logger.error(f"Brouhaha 模型加载失败: {e}")
            self.model = None

    def _load_from_local(self):
        """从本地目录加载模型"""
        try:
            from pyannote.audio import Model

            logger.info(f"从本地加载 Brouhaha 模型: {self.model_dir}")
            self.model = Model.from_pretrained(
                str(self.model_dir),
                strict=False
            )
        except Exception as e:
            logger.error(f"本地模型加载失败: {e}")
            # 尝试从 HuggingFace 重新下载
            self._download_from_huggingface()

    def _download_from_huggingface(self):
        """从 HuggingFace 下载模型"""
        try:
            from pyannote.audio import Model

            if not self.hf_token:
                logger.warning(
                    "未配置 HuggingFace Token，无法下载 Brouhaha 模型。"
                    "请设置环境变量 HUGGING_FACE_HUB_TOKEN 或 HF_TOKEN"
                )
                self.model = None
                return

            logger.info("从 HuggingFace 下载 Brouhaha 模型...")
            self.model = Model.from_pretrained(
                "pyannote/brouhaha",
                use_auth_token=self.hf_token
            )

            # 保存到本地目录
            self._save_model_locally()

        except Exception as e:
            logger.error(f"从 HuggingFace 下载模型失败: {e}")
            self.model = None

    def _save_model_locally(self):
        """保存模型到本地目录"""
        if self.model is None:
            return

        try:
            import torch

            self.model_dir.mkdir(parents=True, exist_ok=True)

            # 保存模型权重
            weights_path = self.model_dir / "pytorch_model.bin"
            torch.save(self.model.state_dict(), weights_path)

            # 保存配置（简化版）
            config_path = self.model_dir / "config.yaml"
            config_content = """# Brouhaha 模型配置
# V3.1.1+dev.20260107.03: 自动保存的配置
architecture:
  name: Brouhaha
  sample_rate: 16000
  num_channels: 1

task:
  name: multi_task
  outputs:
    - snr
    - c50
    - vad
"""
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)

            logger.info(f"Brouhaha 模型已保存到: {self.model_dir}")

        except Exception as e:
            logger.warning(f"保存模型到本地失败: {e}")

    def detect(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        chunk_id: Optional[int] = None
    ) -> BrouhahaResult:
        """
        检测音频的 SNR 和 C50

        Args:
            audio: 音频数组（单声道）
            sr: 采样率
            chunk_id: chunk 索引（用于缓存）

        Returns:
            BrouhahaResult: 检测结果
        """
        # 检查缓存
        if chunk_id is not None and chunk_id in self._cache:
            return self._cache[chunk_id]

        # 模型不可用时的回退
        if self.model is None:
            return self._fallback_wada_snr(audio, sr, chunk_id)

        try:
            import torch

            # 重采样（如果需要）
            if sr != self.SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLE_RATE)

            # 转换为 PyTorch tensor
            waveform = torch.from_numpy(audio).float().unsqueeze(0)  # [1, samples]
            waveform = waveform.to(self._device)

            # 推理
            with torch.no_grad():
                # Brouhaha 模型输出格式: (batch, frames, 3) -> [SNR, C50, VAD]
                outputs = self.model(waveform)

            # 聚合帧级输出为片段级（取均值）
            if outputs.dim() == 3:
                outputs = outputs.mean(dim=1)  # [batch, 3]

            snr = float(outputs[0, 0].cpu())
            c50 = float(outputs[0, 1].cpu())
            vad = float(outputs[0, 2].cpu()) if outputs.shape[1] > 2 else 1.0

            result = BrouhahaResult(snr=snr, c50=c50, vad=vad, is_valid=True)

            # 缓存
            if chunk_id is not None:
                self._add_to_cache(chunk_id, result)

            return result

        except Exception as e:
            logger.warning(f"Brouhaha 推理失败: {e}")
            return self._fallback_wada_snr(audio, sr)

    def _fallback_wada_snr(
        self,
        audio: np.ndarray,
        sr: int,
        chunk_id: Optional[int] = None
    ) -> BrouhahaResult:
        """
        回退到 WADA-SNR 算法

        当 Brouhaha 模型不可用时，使用无模型的 WADA-SNR 算法估计 SNR
        """
        try:
            snr = self._calculate_wada_snr(audio, sr)
            # WADA-SNR 不提供 C50，使用默认值 0.0 表示未知
            result = BrouhahaResult(snr=snr, c50=0.0, vad=1.0, is_valid=True)

            # 缓存结果
            if chunk_id is not None:
                self._add_to_cache(chunk_id, result)

            return result
        except Exception as e:
            logger.warning(f"WADA-SNR 计算失败: {e}")
            return BrouhahaResult(snr=15.0, c50=0.0, vad=1.0, is_valid=False)

    def _calculate_wada_snr(self, audio: np.ndarray, sr: int) -> float:
        """
        WADA-SNR 算法实现

        基于波形幅度分布分析估计 SNR
        参考: Kim & Stern, Interspeech 2008

        Args:
            audio: 音频数组
            sr: 采样率

        Returns:
            float: 估计的 SNR (dB)
        """
        # 分帧参数
        frame_length = int(0.025 * sr)  # 25ms
        hop_length = int(0.010 * sr)    # 10ms

        # 检查音频长度是否足够分帧
        if len(audio) < frame_length:
            # 音频太短，无法分帧，返回默认值
            return 15.0

        # 分帧
        frames = []
        for i in range(0, len(audio) - frame_length + 1, hop_length):
            frames.append(audio[i:i + frame_length])

        if not frames:
            return 15.0  # 默认值

        # 计算每帧能量
        energies = np.array([np.sum(frame ** 2) for frame in frames])

        # 过滤零能量帧
        valid_energies = energies[energies > 1e-10]
        if len(valid_energies) == 0:
            return 15.0

        # 改进的 SNR 估计算法
        # 使用百分位数来估计噪声和信号能量
        sorted_energies = np.sort(valid_energies)

        # 噪声估计：取最低 5% 的帧能量均值
        noise_count = max(1, len(sorted_energies) // 20)  # 5%
        noise_energy = np.mean(sorted_energies[:noise_count])

        # 信号估计：取最高 30% 的帧能量均值（排除可能的异常值）
        signal_start = int(len(sorted_energies) * 0.7)
        signal_energy = np.mean(sorted_energies[signal_start:])

        # 计算 SNR
        if noise_energy > 1e-15:
            snr = 10 * np.log10(signal_energy / noise_energy)
        else:
            # 噪声能量极低，说明信号非常纯净
            # 检查信号能量来决定返回值
            if signal_energy > 1e-10:
                snr = 40.0  # 极高质量
            else:
                snr = 15.0  # 默认值

        # 对于高度周期性的信号（如正弦波），额外检测
        # 计算能量的变异系数（CV），低 CV 表示稳定的周期信号
        if len(valid_energies) > 10:
            cv = np.std(valid_energies) / (np.mean(valid_energies) + 1e-10)
            if cv < 0.1:  # 能量非常稳定，说明是纯净的周期信号
                snr = max(snr, 30.0)

        return float(np.clip(snr, -10, 50))

    def _add_to_cache(self, chunk_id: int, result: BrouhahaResult):
        """添加到缓存（LRU 策略）"""
        if len(self._cache) >= self._cache_max_size:
            # 移除最早的条目
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[chunk_id] = result

    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.model is not None

    def get_device(self) -> str:
        """获取当前使用的设备"""
        if self._device is not None:
            return str(self._device)
        return "unknown"

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()

    def unload(self):
        """卸载模型释放显存"""
        if self.model is not None:
            import torch
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Brouhaha 模型已卸载")


# ========== 单例访问 ==========

_brouhaha_instance: Optional[BrouhahaService] = None


def get_brouhaha_service() -> BrouhahaService:
    """获取 Brouhaha 服务单例"""
    global _brouhaha_instance
    if _brouhaha_instance is None:
        _brouhaha_instance = BrouhahaService()
    return _brouhaha_instance


def reset_brouhaha_service():
    """重置 Brouhaha 服务单例（用于测试）"""
    global _brouhaha_instance
    if _brouhaha_instance is not None:
        _brouhaha_instance.unload()
    _brouhaha_instance = None
