"""
频谱指纹分诊台

对每个VAD Chunk进行频谱分析，决定是否需要人声分离。
在VAD切分之后、转录之前执行。

V2 更新 (2025-12-21):
- 新增 YAMNet 探针模式语义级分类器，替代基于规则的频谱分析
- 优先使用 YAMNet，回退到规则方法
- 解决人声被误判为音乐的问题 (谐波比阈值过低)

V3.1.1+dev.20260108.01:
- 新增 Brouhaha SNR+C50 三层决策策略
- Layer 1: SNR+C50 快速筛选（60-70% 直接决策）
- Layer 2: 频谱特征补充判断（10-20%）
- Layer 3: YAMNet 语义分类（10-20% 兜底）
"""
import numpy as np
import logging
from importlib import metadata, util
from typing import List, Tuple, Optional

from app.models.circuit_breaker_models import (
    SpectrumFeatures, SpectrumDiagnosis, DiagnosisResult
)
from app.core.spectrum_thresholds import SpectrumThresholds
from app.services.runtime_param_resolver import build_spectrum_thresholds

logger = logging.getLogger(__name__)


def _is_snr_strategy_runtime_available() -> bool:
    """
    判断 SNR+C50 策略运行时是否可用。

    Why:
    - 临时下线 Brouhaha 后，默认必须走兜底分诊；
    - 即使外部误传 use_snr_strategy=True，也要自动降级，避免触发导入/推理异常。
    """
    if util.find_spec("brouhaha") is None:
        return False

    try:
        pyannote_version = metadata.version("pyannote-audio")
    except metadata.PackageNotFoundError:
        pyannote_version = ""
    except Exception:
        pyannote_version = ""

    if pyannote_version.startswith("4."):
        return False

    return True


# ========== Brouhaha 集成 (V3.1.1+dev.20260108.01) ==========

def _get_brouhaha_service():
    """懒加载 Brouhaha 服务（避免循环导入）"""
    try:
        from app.services.brouhaha_service import get_brouhaha_service
        return get_brouhaha_service()
    except Exception as e:
        logger.warning(f"Brouhaha 服务加载失败: {e}")
        return None


# ========== YAMNet 集成 ==========

def _get_yamnet_classifier():
    """懒加载 YAMNet 分类器（避免循环导入）"""
    try:
        from app.services.yamnet_classifier import get_yamnet_classifier
        return get_yamnet_classifier()
    except Exception as e:
        logger.warning(f"YAMNet 分类器加载失败: {e}")
        return None


class AudioSpectrumClassifier:
    """
    音频频谱分诊器

    支持三种模式：
    1. SNR+C50 策略（V3.1.1+）：使用 Brouhaha 模型的三层决策，最高效
    2. YAMNet 探针模式（默认回退）：使用预训练模型进行语义级分类
    3. 规则模式（最终回退）：基于频谱特征的规则判断
    """

    def __init__(
        self,
        thresholds: Optional[SpectrumThresholds] = None,
        use_yamnet: Optional[bool] = None,
        use_snr_strategy: Optional[bool] = None  # V3.1.1+dev.20260108.01: 默认启用
    ):
        """
        初始化分诊器

        Args:
            thresholds: 频谱阈值配置
            use_yamnet: 是否使用 YAMNet 语义分类器（默认 True）
            use_snr_strategy: 是否使用 SNR+C50 三层决策策略（默认 False）
        """
        self.thresholds = thresholds or build_spectrum_thresholds()
        self._librosa = None  # 懒加载
        self._use_yamnet = True if use_yamnet is None else use_yamnet
        self._yamnet = None  # 懒加载
        self._use_snr_strategy = True if use_snr_strategy is None else use_snr_strategy
        self._brouhaha = None  # 懒加载

    def apply_runtime_params(
        self,
        thresholds: Optional[SpectrumThresholds] = None,
        use_yamnet: Optional[bool] = None,
        use_snr_strategy: Optional[bool] = None,
    ) -> None:
        """应用运行参数（动态更新阈值与开关）。"""
        if thresholds is not None:
            self.thresholds = thresholds

        if use_yamnet is not None and use_yamnet != self._use_yamnet:
            self._use_yamnet = use_yamnet
            if not use_yamnet:
                self._yamnet = None

        if use_snr_strategy is not None and use_snr_strategy != self._use_snr_strategy:
            self._use_snr_strategy = use_snr_strategy
            if not use_snr_strategy:
                self._brouhaha = None

    def _get_yamnet(self):
        """获取 YAMNet 分类器实例"""
        if self._yamnet is None and self._use_yamnet:
            self._yamnet = _get_yamnet_classifier()
        return self._yamnet

    def _get_brouhaha(self):
        """
        获取 Brouhaha 服务实例 (V3.1.1+dev.20260108.01)

        V3.1.1+dev.20260108.06: 修复卸载后未重新加载的问题
        每次检查模型是否可用，如果不可用则尝试重新加载
        """
        if not self._use_snr_strategy:
            return None
        if not _is_snr_strategy_runtime_available():
            return None

        # 如果没有实例或实例不可用，尝试获取/重新加载
        if self._brouhaha is None or not self._brouhaha.is_available():
            self._brouhaha = _get_brouhaha_service()

            # 如果获取后仍不可用，尝试重新初始化模型
            if not self._brouhaha.is_available():
                try:
                    self._brouhaha._init_model()
                except Exception as e:
                    logger.warning(f"Brouhaha 重新加载失败: {e}")

        return self._brouhaha if self._brouhaha.is_available() else None

    def _ensure_librosa(self):
        """确保 librosa 已加载"""
        if self._librosa is None:
            import librosa
            self._librosa = librosa
        return self._librosa

    def extract_features(self, audio: np.ndarray, sr: int = 16000) -> SpectrumFeatures:
        """
        提取频谱特征

        Args:
            audio: 音频数组 (单声道)
            sr: 采样率

        Returns:
            SpectrumFeatures: 提取的特征
        """
        librosa = self._ensure_librosa()

        features = SpectrumFeatures()

        # 确保音频有效
        if len(audio) < sr * 0.1:  # 至少0.1秒
            return features

        try:
            # 1. 过零率 (ZCR)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features.zcr = float(np.mean(zcr))
            features.zcr_variance = float(np.var(zcr))

            # 2. 频谱特征
            # 短时傅里叶变换
            stft = np.abs(librosa.stft(audio))

            # 谱质心
            cent = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
            features.spectral_centroid = float(np.mean(cent))

            # 谱带宽
            bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
            features.spectral_bandwidth = float(np.mean(bandwidth))

            # 频谱平坦度
            flatness = librosa.feature.spectral_flatness(S=stft)[0]
            features.spectral_flatness = float(np.mean(flatness))

            # 频谱滚降点 (85%能量点)
            rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr)[0]
            features.spectral_rolloff = float(np.mean(rolloff))

            # 频谱对比度 (V3.1.1+dev.20260108.01: 用于 Layer 2 决策)
            # spectral_contrast 返回 [n_bands, frames]，取所有频段的平均值
            contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
            features.spectral_contrast = float(np.mean(contrast))

            # 3. 谐波比 (简化计算)
            harmonic, percussive = librosa.effects.hpss(audio)
            h_energy = np.sum(harmonic ** 2)
            total_energy = np.sum(audio ** 2)
            features.harmonic_ratio = float(h_energy / (total_energy + 1e-10))

            # 4. 能量特征
            rms = librosa.feature.rms(y=audio)[0]
            features.rms_energy = float(np.mean(rms))
            features.energy_variance = float(np.var(rms))

            # 高频能量占比 (4kHz以上)
            freq_bins = librosa.fft_frequencies(sr=sr)
            high_freq_idx = freq_bins >= 4000
            if np.any(high_freq_idx):
                high_freq_energy = np.sum(stft[high_freq_idx, :] ** 2)
                total_spectral_energy = np.sum(stft ** 2)
                features.high_freq_ratio = float(high_freq_energy / (total_spectral_energy + 1e-10))

            # 5. 节奏特征
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            features.onset_strength = float(np.mean(onset_env))

            # 估计BPM
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            features.tempo = float(tempo) if isinstance(tempo, (int, float)) else float(tempo[0])

        except Exception as e:
            logger.warning(f"特征提取失败: {e}")

        return features

    def diagnose_chunk(
        self,
        audio: np.ndarray,
        chunk_index: int,
        sr: int = 16000
    ) -> SpectrumDiagnosis:
        """
        对单个Chunk进行频谱分诊

        V3.1.1+dev.20260108.01: 新增三层决策策略
        - 若启用 use_snr_strategy 且 Brouhaha 可用，使用三层决策
        - 否则回退到 YAMNet 或规则方法

        Args:
            audio: 音频数组
            chunk_index: Chunk索引
            sr: 采样率

        Returns:
            SpectrumDiagnosis: 分诊结果
        """
        duration_sec = len(audio) / sr

        # 极短片段（< 0.5秒）直接返回 CLEAN，样本量不足无法可靠分析
        if duration_sec < 0.5:
            logger.debug(f"Chunk {chunk_index}: 极短片段({duration_sec:.2f}s)，跳过分诊")
            return SpectrumDiagnosis(
                chunk_index=chunk_index,
                diagnosis=DiagnosisResult.CLEAN,
                need_separation=False,
                music_score=0.0,
                noise_score=0.0,
                clean_score=1.0,
                recommended_model=None,
                features=SpectrumFeatures(),
                reason=f"极短片段({duration_sec:.2f}s)，跳过分诊"
            )

        # V3.1.1+dev.20260108.01: 尝试使用 SNR+C50 三层决策策略
        if self._use_snr_strategy:
            brouhaha = self._get_brouhaha()
            if brouhaha is not None:
                return self._diagnose_with_snr_c50_strategy(audio, chunk_index, sr, brouhaha)
            else:
                logger.debug(f"Chunk {chunk_index}: Brouhaha 不可用，回退到 YAMNet/规则方法")

        # 尝试使用 YAMNet 语义分类器
        yamnet = self._get_yamnet()
        if yamnet is not None and yamnet.is_available():
            return self._diagnose_with_yamnet(audio, chunk_index, sr, yamnet)

        # 回退到规则方法
        return self._diagnose_with_rules(audio, chunk_index, sr)

    def _diagnose_with_yamnet(
        self,
        audio: np.ndarray,
        chunk_index: int,
        sr: int,
        yamnet
    ) -> SpectrumDiagnosis:
        """
        使用 YAMNet 进行语义级分诊

        Args:
            audio: 音频数组
            chunk_index: Chunk索引
            sr: 采样率
            yamnet: YAMNet 分类器实例

        Returns:
            SpectrumDiagnosis: 分诊结果
        """
        th = self.thresholds

        # YAMNet 分类
        result = yamnet.classify_chunk(audio, chunk_id=chunk_index)

        # 转换为 SpectrumDiagnosis 格式
        if result.is_music:
            diagnosis = DiagnosisResult.MUSIC
            need_separation = True
            # 统一使用 htdemucs（shift=1 模式）
            recommended_model = "htdemucs"
            reason = f"[YAMNet] 检测到音乐 (score={result.music_score:.2f})"
        else:
            diagnosis = DiagnosisResult.CLEAN
            need_separation = False
            recommended_model = None
            reason = f"[YAMNet] {', '.join(result.tags)} (speech={result.speech_score:.2f})"

        # 日志记录
        logger.debug(
            f"Chunk {chunk_index} [YAMNet]: is_music={result.is_music}, "
            f"music={result.music_score:.3f}, speech={result.speech_score:.3f}, "
            f"tags={result.tags}"
        )

        return SpectrumDiagnosis(
            chunk_index=chunk_index,
            diagnosis=diagnosis,
            need_separation=need_separation,
            music_score=result.music_score,
            noise_score=0.0,  # YAMNet 不单独计算噪音分数
            clean_score=result.speech_score,  # 用 speech_score 作为 clean_score
            recommended_model=recommended_model,
            features=SpectrumFeatures(),  # YAMNet 模式不提取传统特征
            reason=reason
        )

    def _diagnose_with_rules(
        self,
        audio: np.ndarray,
        chunk_index: int,
        sr: int = 16000
    ) -> SpectrumDiagnosis:
        """
        使用规则方法进行频谱分诊（回退方法）

        Args:
            audio: 音频数组
            chunk_index: Chunk索引
            sr: 采样率

        Returns:
            SpectrumDiagnosis: 分诊结果
        """
        th = self.thresholds
        duration_sec = len(audio) / sr

        # 短 Chunk 保守策略：降低敏感度
        short_chunk_threshold = 2.0
        is_short_chunk = duration_sec < short_chunk_threshold

        features = self.extract_features(audio, sr)

        # 计算各项得分
        music_score = self._calculate_music_score(features)
        noise_score = self._calculate_noise_score(features)
        clean_score = 1.0 - max(music_score, noise_score)

        # 短 Chunk 敏感度调整：提高阈值 30%
        if is_short_chunk:
            effective_music_threshold = th.music_score_threshold * 1.3
            effective_noise_threshold = th.noise_score_threshold * 1.3
            effective_mixed_threshold = 0.2 * 1.3
            logger.debug(
                f"Chunk {chunk_index}: 短片段({duration_sec:.2f}s)，"
                f"提高阈值 music>{effective_music_threshold:.2f}, noise>{effective_noise_threshold:.2f}"
            )
        else:
            effective_music_threshold = th.music_score_threshold
            effective_noise_threshold = th.noise_score_threshold
            effective_mixed_threshold = 0.2

        # 综合判定
        diagnosis = DiagnosisResult.CLEAN
        need_separation = False
        recommended_model = None
        reason = "纯净人声"

        if is_short_chunk:
            reason = f"纯净人声 (短片段{duration_sec:.1f}s)"

        if music_score >= effective_music_threshold:
            diagnosis = DiagnosisResult.MUSIC
            need_separation = True
            # 统一使用 htdemucs（shift=1 模式）
            recommended_model = "htdemucs"
            reason = f"检测到音乐 (score={music_score:.2f})"

        elif noise_score >= effective_noise_threshold:
            diagnosis = DiagnosisResult.NOISE
            need_separation = True
            recommended_model = "htdemucs"
            reason = f"检测到噪音 (score={noise_score:.2f})"

        elif music_score > effective_mixed_threshold and noise_score > effective_mixed_threshold:
            diagnosis = DiagnosisResult.MIXED
            need_separation = True
            recommended_model = "htdemucs"
            reason = f"混合噪音 (music={music_score:.2f}, noise={noise_score:.2f})"

        return SpectrumDiagnosis(
            chunk_index=chunk_index,
            diagnosis=diagnosis,
            need_separation=need_separation,
            music_score=music_score,
            noise_score=noise_score,
            clean_score=clean_score,
            recommended_model=recommended_model,
            features=features,
            reason=reason
        )

    def _calculate_music_score(self, f: SpectrumFeatures) -> float:
        """计算音乐得分"""
        th = self.thresholds
        score = 0.0

        # 谐波比高 → 音乐
        if f.harmonic_ratio >= th.harmonic_ratio_music:
            score += 0.35
        elif f.harmonic_ratio >= th.harmonic_ratio_music * 0.7:
            score += 0.2

        # 谱质心在音乐范围内
        if th.spectral_centroid_music_low <= f.spectral_centroid <= th.spectral_centroid_music_high:
            score += 0.25

        # 能量有节奏性波动
        if f.energy_variance >= th.energy_variance_music:
            score += 0.2

        # 有明显节拍
        if f.onset_strength >= th.onset_strength_music:
            score += 0.2

        return min(score, 1.0)

    def _calculate_noise_score(self, f: SpectrumFeatures) -> float:
        """计算噪音得分"""
        th = self.thresholds
        score = 0.0

        # 过零率高 → 噪音
        if f.zcr >= th.zcr_noise_high:
            score += 0.3
            # ZCR方差小说明是稳态噪音（如白噪声）
            if f.zcr_variance <= th.zcr_variance_noise:
                score += 0.15

        # 高频能量占比高 → 噪音
        if f.high_freq_ratio >= th.high_freq_ratio_noise:
            score += 0.25

        # 频谱平坦 → 噪音
        if f.spectral_flatness >= th.spectral_flatness_noise:
            score += 0.2

        # 谐波比低 → 噪音
        if f.harmonic_ratio < 0.3:
            score += 0.1

        return min(score, 1.0)

    # ========== V3.1.1+dev.20260108.01: SNR+C50 三层决策策略 ==========

    def _diagnose_with_snr_c50_strategy(
        self,
        audio: np.ndarray,
        chunk_index: int,
        sr: int,
        brouhaha
    ) -> SpectrumDiagnosis:
        """
        使用 Brouhaha SNR+C50 三层决策策略进行分诊

        三层决策流程：
        - Layer 1: SNR+C50 快速筛选（60-70% 直接决策）
        - Layer 2: 频谱特征补充判断（10-20%）
        - Layer 3: YAMNet 语义分类（10-20% 兜底）

        Args:
            audio: 音频数组
            chunk_index: Chunk索引
            sr: 采样率
            brouhaha: Brouhaha 服务实例

        Returns:
            SpectrumDiagnosis: 分诊结果
        """
        th = self.thresholds

        # 获取 Brouhaha SNR/C50 检测结果
        brouhaha_result = brouhaha.detect(audio, sr, chunk_id=chunk_index)
        snr = brouhaha_result.snr
        c50 = brouhaha_result.c50

        # 确定 SNR/C50 级别
        snr_level = self._classify_snr_level(snr)
        c50_level = self._classify_c50_level(c50)

        # Layer 1: SNR+C50 快速筛选
        layer1_result = self._layer1_decision(snr, c50, snr_level, c50_level, chunk_index)
        if layer1_result is not None:
            need_separation, reason = layer1_result
            return self._build_snr_diagnosis(
                chunk_index=chunk_index,
                snr=snr,
                c50=c50,
                snr_level=snr_level,
                c50_level=c50_level,
                triage_layer=1,
                need_separation=need_separation,
                reason=reason
            )

        # Layer 2: 频谱特征补充判断
        features = self.extract_features(audio, sr)
        features.snr = snr
        features.c50 = c50

        layer2_result = self._layer2_decision(features, snr_level, c50_level, chunk_index)
        if layer2_result is not None:
            need_separation, reason = layer2_result
            return self._build_snr_diagnosis(
                chunk_index=chunk_index,
                snr=snr,
                c50=c50,
                snr_level=snr_level,
                c50_level=c50_level,
                triage_layer=2,
                need_separation=need_separation,
                reason=reason,
                features=features
            )

        # Layer 3: YAMNet 语义分类兜底 (V3.1.1+dev.20260108.07: 针对 CTC 模型强化)
        yamnet = self._get_yamnet()
        if yamnet is not None and yamnet.is_available():
            yamnet_result = yamnet.classify_chunk(audio, chunk_id=chunk_index)

            # V3.1.1+dev.20260108.07: SNR 二次检查（针对 CTC 模型的保守策略）
            # 对于 25-30dB 的灰色地带，即使 YAMNet 判定为人声，也要求更高的置信度
            if snr < 30.0 and not yamnet_result.is_music:
                # SNR 偏低，需要更高的 YAMNet 置信度才能放行
                if yamnet_result.speech_score < 0.95:
                    need_separation = True
                    reason = f"[L3] SNR 偏低({snr:.1f}dB) + YAMNet 置信度不足(speech={yamnet_result.speech_score:.2f})，保守分离"
                else:
                    need_separation = False
                    reason = f"[L3] SNR 偏低({snr:.1f}dB) 但 YAMNet 高置信度(speech={yamnet_result.speech_score:.2f})，放行"
            else:
                # 正常判断
                need_separation = yamnet_result.is_music
                if need_separation:
                    reason = f"[L3] YAMNet 检测到音乐 (music={yamnet_result.music_score:.2f})"
                else:
                    reason = f"[L3] YAMNet 判定为人声 (speech={yamnet_result.speech_score:.2f})"
        else:
            # 最终回退：使用规则方法判断
            music_score = self._calculate_music_score(features)
            noise_score = self._calculate_noise_score(features)
            need_separation = music_score >= th.music_score_threshold or noise_score >= th.noise_score_threshold
            reason = f"[L3] 规则判断 (music={music_score:.2f}, noise={noise_score:.2f})"

        return self._build_snr_diagnosis(
            chunk_index=chunk_index,
            snr=snr,
            c50=c50,
            snr_level=snr_level,
            c50_level=c50_level,
            triage_layer=3,
            need_separation=need_separation,
            reason=reason,
            features=features
        )

    def _classify_snr_level(self, snr: float) -> str:
        """
        分类 SNR 级别

        Returns:
            "high" / "warn" / "low"
        """
        th = self.thresholds
        if snr >= th.snr_high_threshold:
            return "high"
        elif snr >= th.snr_low_threshold:
            return "warn"
        else:
            return "low"

    def _classify_c50_level(self, c50: float) -> str:
        """
        分类 C50 级别

        Returns:
            "good" / "warn" / "bad"
        """
        th = self.thresholds
        if c50 >= th.c50_good_threshold:
            return "good"
        elif c50 >= th.c50_bad_threshold:
            return "warn"
        else:
            return "bad"

    def _layer1_decision(
        self,
        snr: float,
        c50: float,
        snr_level: str,
        c50_level: str,
        chunk_index: int
    ) -> Optional[Tuple[bool, str]]:
        """
        Layer 1: SNR+C50 快速筛选

        决策矩阵：
        - SNR >= 25dB 且 C50 >= 5dB → 直接放行
        - SNR < 12dB 或 C50 < -5dB → 强制分离
        - 其他 → 进入 Layer 2

        Returns:
            None 表示需要进入下一层，否则返回 (need_separation, reason)
        """
        # 高质量：直接放行
        if snr_level == "high" and c50_level == "good":
            reason = f"[L1] 高SNR({snr:.1f}dB)+良好C50({c50:.1f}dB)，纯净人声"
            logger.debug(f"Chunk {chunk_index}: {reason}")
            return (False, reason)

        # 低质量：强制分离
        if snr_level == "low":
            reason = f"[L1] 低SNR({snr:.1f}dB)，需要分离"
            logger.debug(f"Chunk {chunk_index}: {reason}")
            return (True, reason)

        if c50_level == "bad":
            reason = f"[L1] 严重混响C50({c50:.1f}dB)，需要分离"
            logger.debug(f"Chunk {chunk_index}: {reason}")
            return (True, reason)

        # 警戒区，进入 Layer 2
        return None

    def _layer2_decision(
        self,
        features: SpectrumFeatures,
        snr_level: str,
        c50_level: str,
        chunk_index: int
    ) -> Optional[Tuple[bool, str]]:
        """
        Layer 2: 频谱特征补充判断

        决策规则：
        - 频谱对比度 < 12dB → 分离
        - 对比度 < 15dB 且 平坦度 > 0.4 → 分离
        - 其他 → 进入 Layer 3

        Returns:
            None 表示需要进入下一层，否则返回 (need_separation, reason)
        """
        th = self.thresholds
        contrast = features.spectral_contrast
        flatness = features.spectral_flatness

        # 极低对比度：强制分离
        if contrast < th.spectral_contrast_critical:
            reason = f"[L2] 极低对比度({contrast:.1f}dB)，需要分离"
            logger.debug(f"Chunk {chunk_index}: {reason}")
            return (True, reason)

        # 低对比度 + 高平坦度：分离
        if contrast < th.spectral_contrast_low and flatness > th.spectral_flatness_high:
            reason = f"[L2] 低对比度({contrast:.1f}dB)+高平坦度({flatness:.2f})，需要分离"
            logger.debug(f"Chunk {chunk_index}: {reason}")
            return (True, reason)

        # 高 SNR 但 C50 警戒：检查是否能直接放行
        if snr_level == "high" and c50_level == "warn":
            # 高 SNR + 中等 C50 + 良好对比度 = 放行
            if contrast >= th.spectral_contrast_low:
                reason = f"[L2] 高SNR+中C50+良好对比度({contrast:.1f}dB)，放行"
                logger.debug(f"Chunk {chunk_index}: {reason}")
                return (False, reason)

        # 进入 Layer 3
        return None

    def _build_snr_diagnosis(
        self,
        chunk_index: int,
        snr: float,
        c50: float,
        snr_level: str,
        c50_level: str,
        triage_layer: int,
        need_separation: bool,
        reason: str,
        features: SpectrumFeatures = None
    ) -> SpectrumDiagnosis:
        """
        构建 SNR 策略的分诊结果

        Args:
            chunk_index: Chunk 索引
            snr: 信噪比 (dB)
            c50: 清晰度指数 (dB)
            snr_level: SNR 级别
            c50_level: C50 级别
            triage_layer: 决策层级 (1/2/3)
            need_separation: 是否需要分离
            reason: 决策原因
            features: 频谱特征（可选）

        Returns:
            SpectrumDiagnosis: 分诊结果
        """
        if need_separation:
            diagnosis = DiagnosisResult.NOISE  # 默认标记为噪音
            recommended_model = "htdemucs"
        else:
            diagnosis = DiagnosisResult.CLEAN
            recommended_model = None

        return SpectrumDiagnosis(
            chunk_index=chunk_index,
            diagnosis=diagnosis,
            need_separation=need_separation,
            music_score=0.0,
            noise_score=0.0,
            clean_score=1.0 if not need_separation else 0.0,
            recommended_model=recommended_model,
            features=features or SpectrumFeatures(snr=snr, c50=c50),
            reason=reason,
            snr=snr,
            c50=c50,
            snr_level=snr_level,
            c50_level=c50_level,
            triage_layer=triage_layer
        )

    def diagnose_chunks(
        self,
        chunks: List[Tuple[np.ndarray, float, float]],
        sr: int = 16000
    ) -> List[SpectrumDiagnosis]:
        """
        批量分诊多个Chunk

        Args:
            chunks: [(audio_array, start_time, end_time), ...]
            sr: 采样率

        Returns:
            List[SpectrumDiagnosis]: 分诊结果列表
        """
        results = []
        for i, (audio, start, end) in enumerate(chunks):
            diag = self.diagnose_chunk(audio, i, sr)
            logger.debug(
                f"Chunk {i} [{start:.1f}s-{end:.1f}s]: "
                f"{diag.diagnosis.value}, need_sep={diag.need_separation}, "
                f"model={diag.recommended_model}"
            )
            results.append(diag)

        # 统计日志
        need_sep_count = sum(1 for d in results if d.need_separation)
        logger.info(
            f"频谱分诊完成: {len(results)} chunks, "
            f"{need_sep_count} 需要分离 ({need_sep_count/len(results)*100:.1f}%)"
        )

        return results

    def quick_global_diagnosis(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        sample_duration: float = 10.0
    ) -> Tuple[str, float]:
        """
        快速全局预判（替代旧的 detect_background_music_level）

        采用分位数采样策略，快速判断整体音频的 BGM 情况。
        这是一个轻量级预判，不需要运行 Demucs。

        采样策略：取音频时长的 15%、50%、85% 处各截取 sample_duration 秒

        Args:
            audio: 完整音频数组 (samples,) 单声道
            sr: 采样率（默认 16000）
            sample_duration: 每个采样点的时长（秒，默认 10）

        Returns:
            Tuple[str, float]: (建议级别 "none"/"light"/"heavy", 平均音乐得分)
        """
        duration_sec = len(audio) / sr

        # 音频太短，无法可靠预判
        if duration_sec < sample_duration * 2:
            logger.warning(f"音频太短({duration_sec:.1f}s)，跳过全局预判")
            return "unknown", 0.0

        # 分位数采样位置
        sample_positions = [0.15, 0.50, 0.85]
        music_scores = []
        noise_scores = []

        for pos in sample_positions:
            start_time = duration_sec * pos
            # 确保不超出边界
            if start_time + sample_duration > duration_sec:
                start_time = duration_sec - sample_duration
            if start_time < 0:
                start_time = 0

            # 截取采样片段
            start_sample = int(start_time * sr)
            end_sample = int((start_time + sample_duration) * sr)
            chunk = audio[start_sample:end_sample]

            # 提取特征并计算得分
            features = self.extract_features(chunk, sr)
            music_score = self._calculate_music_score(features)
            noise_score = self._calculate_noise_score(features)

            music_scores.append(music_score)
            noise_scores.append(noise_score)

            logger.debug(
                f"全局采样 {pos*100:.0f}% ({start_time:.1f}s): "
                f"music={music_score:.2f}, noise={noise_score:.2f}"
            )

        # 使用最大值判断（保守策略：只要有一处 BGM 很重，就视为 heavy）
        avg_music = sum(music_scores) / len(music_scores)
        max_music = max(music_scores)
        max_noise = max(noise_scores)

        logger.info(
            f"全局预判完成: music_scores={[f'{s:.2f}' for s in music_scores]}, "
            f"avg={avg_music:.2f}, max={max_music:.2f}"
        )

        # 决策逻辑
        th = self.thresholds
        if max_music >= th.heavy_bgm_threshold:  # 默认 0.6
            return "heavy", avg_music
        elif max_music >= th.light_bgm_threshold:  # 默认 0.35
            return "light", avg_music
        elif max_noise >= th.noise_score_threshold:  # 默认 0.45
            return "light", avg_music  # 有噪音也建议轻度处理
        else:
            return "none", avg_music


# ========== 单例访问 ==========

_classifier_instance = None


def get_spectrum_classifier(use_snr_strategy: Optional[bool] = None) -> AudioSpectrumClassifier:
    """
    获取频谱分诊器单例

    V3.2.0+dev.20260212.01: 临时下线 Brouhaha，默认关闭 SNR+C50 三层决策策略

    Args:
        use_snr_strategy: 是否使用 SNR+C50 策略（默认 False）

    Returns:
        AudioSpectrumClassifier: 分诊器单例实例
    """
    global _classifier_instance
    thresholds = build_spectrum_thresholds()
    requested_snr = False if use_snr_strategy is None else bool(use_snr_strategy)
    if requested_snr and not _is_snr_strategy_runtime_available():
        logger.warning("SNR+C50 策略运行时不可用（brouhaha 缺失或 pyannote>=4），自动回退兜底分诊")
        effective_snr = False
    else:
        effective_snr = requested_snr
    effective_yamnet = True
    if _classifier_instance is None:
        _classifier_instance = AudioSpectrumClassifier(
            thresholds=thresholds,
            use_yamnet=effective_yamnet,
            use_snr_strategy=effective_snr,
        )
    else:
        _classifier_instance.apply_runtime_params(
            thresholds=thresholds,
            use_yamnet=effective_yamnet,
            use_snr_strategy=effective_snr,
        )
    return _classifier_instance
