"""
音频预检分类器（DNSMOS 主链）。

主决策：
1. DNSMOS 硬分离/硬放行；
2. 灰区走 YAMNet-Lite；
3. 兜底不可用时保守分离。
"""

from __future__ import annotations

import hashlib
import logging
from importlib import util
from typing import List, Optional, Tuple

import numpy as np

from app.core.spectrum_thresholds import SpectrumThresholds
from app.models.circuit_breaker_models import DiagnosisResult, SpectrumDiagnosis, SpectrumFeatures
from app.services.runtime_param_resolver import build_spectrum_thresholds, get_dnsmos_runtime_params

logger = logging.getLogger(__name__)


def _is_dnsmos_runtime_available() -> bool:
    """检查 DNSMOS 运行时依赖是否可用。"""
    return util.find_spec("onnxruntime") is not None


def _get_dnsmos_service():
    """懒加载 DNSMOS 服务。"""
    try:
        from app.services.dnsmos_service import get_dnsmos_service

        return get_dnsmos_service()
    except Exception as exc:
        logger.warning("DNSMOS 服务加载失败: %s", exc)
        return None


def _get_yamnet_classifier():
    """懒加载 YAMNet 分类器。"""
    try:
        from app.services.yamnet_classifier import get_yamnet_classifier

        return get_yamnet_classifier()
    except Exception as exc:
        logger.warning("YAMNet 分类器加载失败: %s", exc)
        return None


class AudioSpectrumClassifier:
    """音频预检分类器。"""

    def __init__(
        self,
        thresholds: Optional[SpectrumThresholds] = None,
        use_yamnet: Optional[bool] = None,
        use_dnsmos_strategy: Optional[bool] = None,
    ) -> None:
        self.thresholds = thresholds or build_spectrum_thresholds()
        self._use_yamnet = True if use_yamnet is None else bool(use_yamnet)
        self._use_dnsmos_strategy = True if use_dnsmos_strategy is None else bool(use_dnsmos_strategy)
        self._yamnet = None
        self._dnsmos = None
        self._librosa = None

    def apply_runtime_params(
        self,
        thresholds: Optional[SpectrumThresholds] = None,
        use_yamnet: Optional[bool] = None,
        use_dnsmos_strategy: Optional[bool] = None,
    ) -> None:
        """应用运行参数（阈值与开关）。"""
        if thresholds is not None:
            self.thresholds = thresholds
        if use_yamnet is not None and bool(use_yamnet) != self._use_yamnet:
            self._use_yamnet = bool(use_yamnet)
            if not self._use_yamnet:
                self._yamnet = None
        if use_dnsmos_strategy is not None and bool(use_dnsmos_strategy) != self._use_dnsmos_strategy:
            self._use_dnsmos_strategy = bool(use_dnsmos_strategy)
            if not self._use_dnsmos_strategy:
                self._dnsmos = None

    def _get_yamnet(self):
        if self._yamnet is None and self._use_yamnet:
            self._yamnet = _get_yamnet_classifier()
        return self._yamnet

    def _get_dnsmos(self):
        if not self._use_dnsmos_strategy:
            return None
        if not _is_dnsmos_runtime_available():
            return None
        if self._dnsmos is None:
            self._dnsmos = _get_dnsmos_service()
        return self._dnsmos

    @staticmethod
    def _get_dnsmos_thresholds() -> dict:
        runtime = get_dnsmos_runtime_params()
        return {
            "ovrl_sep_hard": float(runtime.get("ovrl_sep_hard", 2.0)),
            "sig_sep_hard": float(runtime.get("sig_sep_hard", 2.1)),
            "p808_sep_hard": float(runtime.get("p808_sep_hard", 2.0)),
            "ovrl_pass_hard": float(runtime.get("ovrl_pass_hard", 3.1)),
            "sig_pass_hard": float(runtime.get("sig_pass_hard", 3.2)),
            "bak_pass_hard": float(runtime.get("bak_pass_hard", 3.0)),
            "p808_pass_soft": float(runtime.get("p808_pass_soft", 2.9)),
        }

    def get_dnsmos_threshold_profile_hash(self) -> str:
        """返回 DNSMOS 阈值配置哈希。"""
        payload = self._get_dnsmos_thresholds()
        serialized = "|".join(f"{key}={payload[key]:.6f}" for key in sorted(payload))
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]

    def get_dnsmos_model_hash(self) -> str:
        """返回 DNSMOS 模型哈希。"""
        dnsmos = self._get_dnsmos()
        if dnsmos is None:
            return "dnsmos-unavailable"
        try:
            return str(dnsmos.get_model_hash())
        except Exception:
            return "dnsmos-unavailable"

    def diagnose_chunk(self, audio: np.ndarray, chunk_index: int, sr: int = 16000) -> SpectrumDiagnosis:
        """对单个 Chunk 执行音频预检。"""
        duration_sec = len(audio) / sr if sr > 0 else 0.0
        if duration_sec < 0.5:
            return SpectrumDiagnosis(
                chunk_index=chunk_index,
                diagnosis=DiagnosisResult.CLEAN,
                need_separation=False,
                clean_score=1.0,
                reason=f"极短片段({duration_sec:.2f}s)，跳过音频预检",
                decision_source="short_clip_pass",
                decision_margin=0.0,
            )

        dnsmos = self._get_dnsmos()
        if dnsmos is not None and dnsmos.is_available():
            return self._diagnose_with_dnsmos(audio=audio, chunk_index=chunk_index, sr=sr, dnsmos=dnsmos)

        logger.debug("Chunk %s: DNSMOS 不可用，进入兼容降级链", chunk_index)
        return self._diagnose_with_compat_fallback(audio=audio, chunk_index=chunk_index, sr=sr)

    def _diagnose_with_dnsmos(self, audio: np.ndarray, chunk_index: int, sr: int, dnsmos) -> SpectrumDiagnosis:
        thresholds = self._get_dnsmos_thresholds()
        result = dnsmos.detect(audio, sr=sr, chunk_id=chunk_index)
        if not result.is_valid:
            return self._diagnose_with_compat_fallback(audio=audio, chunk_index=chunk_index, sr=sr)

        # D1: 硬分离
        sep_margins = []
        if result.ovrl <= thresholds["ovrl_sep_hard"]:
            sep_margins.append(thresholds["ovrl_sep_hard"] - result.ovrl)
        if result.sig <= thresholds["sig_sep_hard"]:
            sep_margins.append(thresholds["sig_sep_hard"] - result.sig)
        if result.p808 <= thresholds["p808_sep_hard"]:
            sep_margins.append(thresholds["p808_sep_hard"] - result.p808)
        if sep_margins:
            return self._build_dnsmos_diagnosis(
                chunk_index=chunk_index,
                result=result,
                is_need_separation=True,
                reason=(
                    "DNSMOS 硬分离: "
                    f"sig={result.sig:.2f}, bak={result.bak:.2f}, ovrl={result.ovrl:.2f}, p808={result.p808:.2f}"
                ),
                decision_source="dnsmos_hard_separate",
                decision_margin=max(sep_margins),
                triage_layer=1,
            )

        # D2: 硬放行
        is_hard_pass = (
            result.ovrl >= thresholds["ovrl_pass_hard"]
            and result.sig >= thresholds["sig_pass_hard"]
            and result.bak >= thresholds["bak_pass_hard"]
            and result.p808 >= thresholds["p808_pass_soft"]
        )
        if is_hard_pass:
            margin = min(
                result.ovrl - thresholds["ovrl_pass_hard"],
                result.sig - thresholds["sig_pass_hard"],
                result.bak - thresholds["bak_pass_hard"],
                result.p808 - thresholds["p808_pass_soft"],
            )
            return self._build_dnsmos_diagnosis(
                chunk_index=chunk_index,
                result=result,
                is_need_separation=False,
                reason=(
                    "DNSMOS 硬放行: "
                    f"sig={result.sig:.2f}, bak={result.bak:.2f}, ovrl={result.ovrl:.2f}, p808={result.p808:.2f}"
                ),
                decision_source="dnsmos_hard_pass",
                decision_margin=margin,
                triage_layer=1,
            )

        # D3: 灰区进入 YAMNet-Lite
        yamnet = self._get_yamnet()
        if yamnet is not None and yamnet.is_available():
            yamnet_result = yamnet.classify_chunk(audio, chunk_id=chunk_index)
            is_need_separation = bool(yamnet_result.is_music)
            return self._build_dnsmos_diagnosis(
                chunk_index=chunk_index,
                result=result,
                is_need_separation=is_need_separation,
                reason=(
                    "DNSMOS 灰区 -> YAMNet-Lite: "
                    f"music={yamnet_result.music_score:.2f}, speech={yamnet_result.speech_score:.2f}, "
                    f"tags={yamnet_result.tags}"
                ),
                decision_source="dnsmos_yamnet_fallback",
                decision_margin=abs(yamnet_result.speech_score - yamnet_result.music_score),
                triage_layer=2,
            )

        # DNSMOS 灰区且兜底不可用：保守分离
        return self._build_dnsmos_diagnosis(
            chunk_index=chunk_index,
            result=result,
            is_need_separation=True,
            reason="DNSMOS 灰区且 YAMNet 不可用，保守分离",
            decision_source="compat_fallback",
            decision_margin=0.0,
            triage_layer=2,
        )

    def _build_dnsmos_diagnosis(
        self,
        chunk_index: int,
        result,
        is_need_separation: bool,
        reason: str,
        decision_source: str,
        decision_margin: float,
        triage_layer: int,
    ) -> SpectrumDiagnosis:
        return SpectrumDiagnosis(
            chunk_index=chunk_index,
            diagnosis=DiagnosisResult.NOISE if is_need_separation else DiagnosisResult.CLEAN,
            need_separation=is_need_separation,
            music_score=0.0,
            noise_score=0.0,
            clean_score=0.0 if is_need_separation else 1.0,
            recommended_model="htdemucs" if is_need_separation else None,
            reason=reason,
            sig=float(result.sig),
            bak=float(result.bak),
            ovrl=float(result.ovrl),
            p808=float(result.p808),
            decision_source=decision_source,
            decision_margin=float(max(0.0, decision_margin)),
            triage_layer=triage_layer,
        )

    def _diagnose_with_compat_fallback(self, audio: np.ndarray, chunk_index: int, sr: int) -> SpectrumDiagnosis:
        """兼容降级链：YAMNet-Lite -> 规则法 -> 保守分离。"""
        yamnet = self._get_yamnet()
        if yamnet is not None and yamnet.is_available():
            yamnet_result = yamnet.classify_chunk(audio, chunk_id=chunk_index)
            is_need_separation = bool(yamnet_result.is_music)
            return SpectrumDiagnosis(
                chunk_index=chunk_index,
                diagnosis=DiagnosisResult.NOISE if is_need_separation else DiagnosisResult.CLEAN,
                need_separation=is_need_separation,
                music_score=float(yamnet_result.music_score),
                clean_score=float(yamnet_result.speech_score),
                recommended_model="htdemucs" if is_need_separation else None,
                reason=f"DNSMOS 不可用，YAMNet-Lite 兼容判定: tags={yamnet_result.tags}",
                decision_source="compat_fallback",
                decision_margin=abs(float(yamnet_result.speech_score) - float(yamnet_result.music_score)),
                triage_layer=2,
            )

        # 最终兜底：仍保留规则法，保证最低可用性
        rules_diag = self._diagnose_with_rules(audio=audio, chunk_index=chunk_index, sr=sr)
        rules_diag.decision_source = "compat_fallback"
        return rules_diag

    def _ensure_librosa(self):
        if self._librosa is None:
            import librosa

            self._librosa = librosa
        return self._librosa

    def extract_features(self, audio: np.ndarray, sr: int = 16000) -> SpectrumFeatures:
        """提取规则兜底需要的频谱特征。"""
        librosa = self._ensure_librosa()
        features = SpectrumFeatures()
        if len(audio) < sr * 0.1:
            return features

        try:
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.zcr = float(np.mean(zcr))
            features.zcr_variance = float(np.var(zcr))

            stft = np.abs(librosa.stft(audio))
            cent = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
            features.spectral_centroid = float(np.mean(cent))
            bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
            features.spectral_bandwidth = float(np.mean(bandwidth))
            flatness = librosa.feature.spectral_flatness(S=stft)[0]
            features.spectral_flatness = float(np.mean(flatness))
            rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)[0]
            features.spectral_rolloff = float(np.mean(rolloff))
            contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
            features.spectral_contrast = float(np.mean(contrast))

            harmonic, _ = librosa.effects.hpss(audio)
            h_energy = np.sum(harmonic ** 2)
            total_energy = np.sum(audio ** 2)
            features.harmonic_ratio = float(h_energy / (total_energy + 1e-10))

            rms = librosa.feature.rms(y=audio)[0]
            features.rms_energy = float(np.mean(rms))
            features.energy_variance = float(np.var(rms))

            freq_bins = librosa.fft_frequencies(sr=sr)
            high_freq_idx = freq_bins >= 4000
            if np.any(high_freq_idx):
                high_freq_energy = np.sum(stft[high_freq_idx, :] ** 2)
                total_spectral_energy = np.sum(stft ** 2)
                features.high_freq_ratio = float(high_freq_energy / (total_spectral_energy + 1e-10))

            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            features.onset_strength = float(np.mean(onset_env))
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            features.tempo = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0])
        except Exception as exc:
            logger.debug("规则特征提取失败: %s", exc)

        return features

    def _calculate_music_score(self, features: SpectrumFeatures) -> float:
        th = self.thresholds
        score = 0.0
        if features.harmonic_ratio >= th.harmonic_ratio_music:
            score += 0.35
        elif features.harmonic_ratio >= th.harmonic_ratio_music * 0.7:
            score += 0.2
        if th.spectral_centroid_music_low <= features.spectral_centroid <= th.spectral_centroid_music_high:
            score += 0.25
        if features.energy_variance >= th.energy_variance_music:
            score += 0.2
        if features.onset_strength >= th.onset_strength_music:
            score += 0.2
        return min(score, 1.0)

    def _calculate_noise_score(self, features: SpectrumFeatures) -> float:
        th = self.thresholds
        score = 0.0
        if features.zcr >= th.zcr_noise_high:
            score += 0.3
            if features.zcr_variance <= th.zcr_variance_noise:
                score += 0.15
        if features.high_freq_ratio >= th.high_freq_ratio_noise:
            score += 0.25
        if features.spectral_flatness >= th.spectral_flatness_noise:
            score += 0.2
        if features.harmonic_ratio < 0.3:
            score += 0.1
        return min(score, 1.0)

    def _diagnose_with_rules(self, audio: np.ndarray, chunk_index: int, sr: int = 16000) -> SpectrumDiagnosis:
        """规则法兜底（仅兼容应急）。"""
        features = self.extract_features(audio, sr)
        music_score = self._calculate_music_score(features)
        noise_score = self._calculate_noise_score(features)
        clean_score = 1.0 - max(music_score, noise_score)
        th = self.thresholds

        diagnosis = DiagnosisResult.CLEAN
        is_need_separation = False
        recommended_model = None
        reason = "规则兜底：纯净人声"
        if music_score >= th.music_score_threshold:
            diagnosis = DiagnosisResult.MUSIC
            is_need_separation = True
            recommended_model = "htdemucs"
            reason = f"规则兜底：检测到音乐 (score={music_score:.2f})"
        elif noise_score >= th.noise_score_threshold:
            diagnosis = DiagnosisResult.NOISE
            is_need_separation = True
            recommended_model = "htdemucs"
            reason = f"规则兜底：检测到噪音 (score={noise_score:.2f})"

        return SpectrumDiagnosis(
            chunk_index=chunk_index,
            diagnosis=diagnosis,
            need_separation=is_need_separation,
            music_score=music_score,
            noise_score=noise_score,
            clean_score=clean_score,
            recommended_model=recommended_model,
            features=features,
            reason=reason,
            decision_source="compat_fallback",
            decision_margin=max(abs(clean_score - th.clean_score_threshold), 0.0),
        )

    def diagnose_chunks(
        self,
        chunks: List[Tuple[np.ndarray, float, float]],
        sr: int = 16000,
    ) -> List[SpectrumDiagnosis]:
        """批量音频预检。"""
        results: List[SpectrumDiagnosis] = []
        for idx, (audio, _start, _end) in enumerate(chunks):
            results.append(self.diagnose_chunk(audio=audio, chunk_index=idx, sr=sr))
        return results

    def quick_global_diagnosis(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        sample_duration: float = 10.0,
    ) -> Tuple[str, float]:
        """全局快速预判（保留旧接口，供兼容路径使用）。"""
        duration_sec = len(audio) / sr
        if duration_sec < sample_duration * 2:
            return "unknown", 0.0

        sample_positions = [0.15, 0.50, 0.85]
        music_scores: List[float] = []
        noise_scores: List[float] = []
        for pos in sample_positions:
            start_time = duration_sec * pos
            if start_time + sample_duration > duration_sec:
                start_time = duration_sec - sample_duration
            start_sample = int(max(0, start_time) * sr)
            end_sample = int((max(0, start_time) + sample_duration) * sr)
            chunk = audio[start_sample:end_sample]
            features = self.extract_features(chunk, sr)
            music_scores.append(self._calculate_music_score(features))
            noise_scores.append(self._calculate_noise_score(features))

        avg_music = float(np.mean(music_scores))
        max_music = float(np.max(music_scores))
        max_noise = float(np.max(noise_scores))
        th = self.thresholds
        if max_music >= th.heavy_bgm_threshold:
            return "heavy", avg_music
        if max_music >= th.light_bgm_threshold or max_noise >= th.noise_score_threshold:
            return "light", avg_music
        return "none", avg_music


_classifier_instance: Optional[AudioSpectrumClassifier] = None


def get_spectrum_classifier(
    use_dnsmos_strategy: Optional[bool] = None,
    use_snr_strategy: Optional[bool] = None,
) -> AudioSpectrumClassifier:
    """
    获取音频预检分类器单例。

    兼容说明：
    - 旧参数 `use_snr_strategy` 仍可传入，会映射到 `use_dnsmos_strategy`。
    """
    global _classifier_instance
    thresholds = build_spectrum_thresholds()

    requested_dnsmos = use_dnsmos_strategy
    if requested_dnsmos is None and use_snr_strategy is not None:
        requested_dnsmos = bool(use_snr_strategy)
    if requested_dnsmos is None:
        requested_dnsmos = True

    effective_dnsmos = bool(requested_dnsmos)
    if effective_dnsmos and not _is_dnsmos_runtime_available():
        logger.warning("DNSMOS 运行时不可用（onnxruntime 缺失），自动回退兼容链路")
        effective_dnsmos = False

    if _classifier_instance is None:
        _classifier_instance = AudioSpectrumClassifier(
            thresholds=thresholds,
            use_yamnet=True,
            use_dnsmos_strategy=effective_dnsmos,
        )
    else:
        _classifier_instance.apply_runtime_params(
            thresholds=thresholds,
            use_yamnet=True,
            use_dnsmos_strategy=effective_dnsmos,
        )
    return _classifier_instance
