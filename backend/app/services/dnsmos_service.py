"""
DNSMOS 音频质量评估服务。

职责：
- 使用 ONNX 模型输出 SIG/BAK/OVRL/P808；
- 默认使用 CPU 推理，避免占用 GPU 主链资源；
- 为音频预检主链提供统一 detect 接口。
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from app.services.runtime_param_resolver import get_runtime_group

logger = logging.getLogger(__name__)


@dataclass
class DNSMOSResult:
    """DNSMOS 单段检测结果。"""

    sig: float
    bak: float
    ovrl: float
    p808: float
    is_valid: bool


class DNSMOSService:
    """DNSMOS 模型服务（单模型会话 + 校准 + 分段聚合）。"""

    SAMPLE_RATE = 16000
    INPUT_LENGTH_SEC = 9.01
    HOP_LENGTH_SEC = 1.0
    P808_TRIM_SAMPLES = 160
    DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "pretrained" / "dnsmos"

    def __init__(self, model_dir: Optional[str] = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else self._resolve_model_dir()
        self._session_main = None
        self._session_p808 = None
        self._device = "cpu"
        self._cache: Dict[int, DNSMOSResult] = {}
        self._cache_max_size = 256
        self._model_hash = ""
        self._librosa = None
        self._init_sessions()

    def _resolve_model_dir(self) -> Path:
        """优先通过模型管理器获取路径，失败时回退默认目录。"""
        try:
            from app.services.model_manager_v2 import get_model_manager_v2

            manager = get_model_manager_v2()
            resolved = Path(manager.ensure_available("dnsmos-quality"))
            logger.info("DNSMOS 使用 ModelManagerV2 路径: %s", resolved)
            return resolved
        except Exception as exc:
            logger.warning("ModelManagerV2 获取 DNSMOS 模型失败，回退默认路径: %s", exc)
            return self.DEFAULT_MODEL_DIR

    def _ensure_librosa(self):
        if self._librosa is None:
            try:
                import librosa  # type: ignore

                # 某些损坏环境会出现“空命名空间包”，需要显式校验关键 API。
                if not hasattr(librosa, "feature") or not hasattr(librosa, "resample"):
                    raise ImportError("librosa 缺少 feature/resample 接口")
                self._librosa = librosa
            except Exception as exc:
                logger.warning("librosa 不可用，DNSMOS 将使用回退实现: %s", exc)
                self._librosa = False
        if self._librosa is False:
            return None
        return self._librosa

    def _init_sessions(self) -> None:
        """加载 ONNX 会话。默认 CPU，按运行参数可选 CUDA。"""
        try:
            import onnxruntime as ort
        except Exception as exc:
            logger.warning("DNSMOS 初始化失败：onnxruntime 不可用: %s", exc)
            self._session_main = None
            self._session_p808 = None
            return

        main_path = self.model_dir / "sig_bak_ovr.onnx"
        p808_path = self.model_dir / "model_v8.onnx"
        if not main_path.exists() or not p808_path.exists():
            logger.warning("DNSMOS 模型文件缺失: main=%s p808=%s", main_path, p808_path)
            self._session_main = None
            self._session_p808 = None
            return

        runtime = get_runtime_group("dnsmos")
        requested_device = str(runtime.get("device", "cpu") or "cpu").lower()
        providers = ["CPUExecutionProvider"]
        provider_options = None
        if requested_device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            provider_options = [{}, {}]
            self._device = "cuda"
        else:
            self._device = "cpu"

        try:
            options = ort.SessionOptions()
            onnx_intra_threads = runtime.get("onnx_intra_threads")
            onnx_inter_threads = runtime.get("onnx_inter_threads")
            if isinstance(onnx_intra_threads, int) and onnx_intra_threads > 0:
                options.intra_op_num_threads = onnx_intra_threads
            if isinstance(onnx_inter_threads, int) and onnx_inter_threads > 0:
                options.inter_op_num_threads = onnx_inter_threads

            if provider_options is not None:
                self._session_main = ort.InferenceSession(
                    str(main_path),
                    sess_options=options,
                    providers=providers,
                    provider_options=provider_options,
                )
                self._session_p808 = ort.InferenceSession(
                    str(p808_path),
                    sess_options=options,
                    providers=providers,
                    provider_options=provider_options,
                )
            else:
                self._session_main = ort.InferenceSession(
                    str(main_path),
                    sess_options=options,
                    providers=providers,
                )
                self._session_p808 = ort.InferenceSession(
                    str(p808_path),
                    sess_options=options,
                    providers=providers,
                )
            self._model_hash = self._build_model_hash(main_path, p808_path)
            logger.info("DNSMOS 模型加载成功: device=%s", self._device)
        except Exception as exc:
            logger.warning("DNSMOS 会话初始化失败: %s", exc)
            self._session_main = None
            self._session_p808 = None

    @staticmethod
    def _build_model_hash(main_path: Path, p808_path: Path) -> str:
        """构建模型版本哈希，供缓存键使用。"""
        hasher = hashlib.sha1()
        for path in (main_path, p808_path):
            stat = path.stat()
            hasher.update(str(path.name).encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(int(stat.st_mtime)).encode("utf-8"))
        return hasher.hexdigest()[:16]

    @staticmethod
    def _fit_melspec_frames(mel_spec: np.ndarray, target_frames: int = 900) -> np.ndarray:
        """将 mel 帧数对齐到模型输入要求。"""
        frames, bins = mel_spec.shape
        if bins != 120:
            raise ValueError(f"P808 mel 维度错误: {mel_spec.shape}")
        if frames == target_frames:
            return mel_spec
        if frames > target_frames:
            return mel_spec[:target_frames, :]
        pad = np.zeros((target_frames - frames, bins), dtype=mel_spec.dtype)
        return np.concatenate([mel_spec, pad], axis=0)

    @staticmethod
    def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
        """Hz -> Mel。"""
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
        """Mel -> Hz。"""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    @classmethod
    def _build_mel_filterbank(
        cls,
        sr: int,
        n_fft: int,
        n_mels: int,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
    ) -> np.ndarray:
        """构建 Mel 滤波器组（兼容 librosa 口径的近似实现）。"""
        if fmax is None:
            fmax = float(sr) / 2.0
        mel_min = cls._hz_to_mel(np.array([fmin], dtype=np.float32))[0]
        mel_max = cls._hz_to_mel(np.array([fmax], dtype=np.float32))[0]
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float32)
        hz_points = cls._mel_to_hz(mel_points)
        bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        max_bin = n_fft // 2
        bins = np.clip(bins, 0, max_bin)

        fb = np.zeros((n_mels, max_bin + 1), dtype=np.float32)
        for m in range(1, n_mels + 1):
            left = int(bins[m - 1])
            center = int(bins[m])
            right = int(bins[m + 1])
            if center <= left:
                center = min(left + 1, max_bin)
            if right <= center:
                right = min(center + 1, max_bin)
            if center > left:
                fb[m - 1, left:center] = (np.arange(left, center) - left) / max(1, center - left)
            if right > center:
                fb[m - 1, center:right] = (right - np.arange(center, right)) / max(1, right - center)
        return fb

    @classmethod
    def _audio_melspec_scipy(cls, audio: np.ndarray) -> np.ndarray:
        """纯 scipy/numpy 回退实现。"""
        from scipy.signal import stft

        _, _, zxx = stft(
            audio,
            fs=16000,
            nperseg=321,
            noverlap=321 - 160,
            nfft=321,
            boundary="zeros",
            padded=True,
        )
        power = np.abs(zxx) ** 2  # [freq, time]
        mel_fb = cls._build_mel_filterbank(sr=16000, n_fft=321, n_mels=120)
        mel_power = mel_fb @ power  # [120, time]
        mel_power = np.maximum(mel_power, 1e-10)
        mel_db = 10.0 * np.log10(mel_power)
        mel_np = mel_db.T.astype(np.float32)  # [time, 120]
        mel_np = (mel_np + 40.0) / 40.0
        return cls._fit_melspec_frames(mel_np)

    def _audio_melspec(self, audio: np.ndarray) -> np.ndarray:
        """提取 P808 输入特征（log-mel）。"""
        librosa = self._ensure_librosa()
        if librosa is not None:
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=16000,
                n_fft=321,
                hop_length=160,
                n_mels=120,
            )
            mel_spec = mel_spec.T
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
            return self._fit_melspec_frames(mel_spec.astype(np.float32))

        # 回退实现：基于 torchaudio 计算 log-mel，避免 librosa 缺失导致主链中断。
        try:
            import torch
            import torchaudio

            waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=321,
                hop_length=160,
                n_mels=120,
                center=True,
                power=2.0,
            )
            mel = transform(waveform).squeeze(0)  # [120, T]
            mel = torch.clamp(mel, min=1e-10)
            mel_db = 10.0 * torch.log10(mel)
            mel_np = mel_db.transpose(0, 1).cpu().numpy()  # [T, 120]
            mel_np = (mel_np + 40.0) / 40.0
            return self._fit_melspec_frames(mel_np.astype(np.float32))
        except Exception:
            pass

        try:
            return self._audio_melspec_scipy(audio)
        except Exception as exc:
            raise RuntimeError(
                f"P808 mel 特征提取失败（librosa/torchaudio/scipy 均不可用）: {exc}"
            ) from exc

    @staticmethod
    def _apply_polyfit(sig: float, bak: float, ovrl: float, is_personalized: bool) -> Tuple[float, float, float]:
        """应用 DNSMOS 官方多项式校准。"""
        if is_personalized:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return float(p_sig(sig)), float(p_bak(bak)), float(p_ovr(ovrl))

    def is_available(self) -> bool:
        """模型是否可用。"""
        return self._session_main is not None and self._session_p808 is not None

    def get_model_hash(self) -> str:
        """返回模型哈希，用于缓存键版本化。"""
        return self._model_hash or "dnsmos-unavailable"

    def _resample_to_16k(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr == self.SAMPLE_RATE:
            return audio.astype(np.float32)
        librosa = self._ensure_librosa()
        if librosa is not None:
            return librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=self.SAMPLE_RATE).astype(np.float32)

        # 回退优先使用 scipy，高质量且无额外模型依赖。
        try:
            from scipy.signal import resample_poly

            gcd = math.gcd(sr, self.SAMPLE_RATE)
            up = self.SAMPLE_RATE // gcd
            down = sr // gcd
            return resample_poly(audio.astype(np.float32), up, down).astype(np.float32)
        except Exception:
            # 最后回退：线性插值，保证流程可运行。
            src = audio.astype(np.float32)
            src_len = src.shape[-1]
            if src_len <= 1:
                return np.zeros(1, dtype=np.float32)
            target_len = max(1, int(round(src_len * self.SAMPLE_RATE / sr)))
            old_idx = np.linspace(0.0, 1.0, num=src_len, endpoint=False)
            new_idx = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
            return np.interp(new_idx, old_idx, src).astype(np.float32)

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if audio.size == 0:
            return np.zeros(1, dtype=np.float32)
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio = audio / peak
        return audio.astype(np.float32)

    def detect(self, audio: np.ndarray, sr: int = 16000, chunk_id: Optional[int] = None) -> DNSMOSResult:
        """
        执行 DNSMOS 检测。

        返回值均为校准后的分数（1~5 量级）。
        """
        if chunk_id is not None and chunk_id in self._cache:
            return self._cache[chunk_id]

        if not self.is_available():
            return DNSMOSResult(sig=0.0, bak=0.0, ovrl=0.0, p808=0.0, is_valid=False)

        runtime = get_runtime_group("dnsmos")
        is_personalized = bool(runtime.get("personalized", False))
        normalized = self._normalize_audio(audio)
        audio_16k = self._resample_to_16k(normalized, sr)

        len_samples = int(self.INPUT_LENGTH_SEC * self.SAMPLE_RATE)
        while audio_16k.shape[-1] < len_samples:
            audio_16k = np.concatenate([audio_16k, audio_16k], axis=-1)

        hop_len = int(self.HOP_LENGTH_SEC * self.SAMPLE_RATE)
        max_start = max(0, audio_16k.shape[-1] - len_samples)
        num_hops = max(1, int(np.floor(max_start / max(1, hop_len))) + 1)

        p808_scores: list[float] = []
        sig_scores: list[float] = []
        bak_scores: list[float] = []
        ovrl_scores: list[float] = []

        for idx in range(num_hops):
            start = min(idx * hop_len, max_start)
            end = start + len_samples
            audio_seg = audio_16k[start:end]
            if audio_seg.shape[-1] != len_samples:
                continue

            try:
                input_features = np.array(audio_seg.reshape(1, -1), dtype=np.float32)
                mel_source = audio_seg[:-self.P808_TRIM_SAMPLES]
                p808_input = np.array(self._audio_melspec(mel_source), dtype=np.float32).reshape(1, 900, 120)

                main_raw = self._session_main.run(None, {"input_1": input_features})[0][0]
                p808_raw = self._session_p808.run(None, {"input_1": p808_input})[0][0]
                p808 = float(p808_raw[0])
                sig_raw = float(main_raw[0])
                bak_raw = float(main_raw[1])
                ovrl_raw = float(main_raw[2])
            except Exception as exc:
                logger.warning("DNSMOS 单段推理失败，已跳过该窗口: %s", exc)
                continue

            sig, bak, ovrl = self._apply_polyfit(sig_raw, bak_raw, ovrl_raw, is_personalized=is_personalized)
            p808_scores.append(p808)
            sig_scores.append(sig)
            bak_scores.append(bak)
            ovrl_scores.append(ovrl)

        if not sig_scores:
            return DNSMOSResult(sig=0.0, bak=0.0, ovrl=0.0, p808=0.0, is_valid=False)

        result = DNSMOSResult(
            sig=float(np.mean(sig_scores)),
            bak=float(np.mean(bak_scores)),
            ovrl=float(np.mean(ovrl_scores)),
            p808=float(np.mean(p808_scores)),
            is_valid=True,
        )
        if chunk_id is not None:
            if len(self._cache) >= self._cache_max_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[chunk_id] = result
        return result


_dnsmos_instance: Optional[DNSMOSService] = None


def get_dnsmos_service() -> DNSMOSService:
    """获取 DNSMOS 服务单例。"""
    global _dnsmos_instance
    if _dnsmos_instance is None:
        _dnsmos_instance = DNSMOSService()
    return _dnsmos_instance


def reset_dnsmos_service() -> None:
    """重置 DNSMOS 服务单例（测试用）。"""
    global _dnsmos_instance
    _dnsmos_instance = None
