"""
频谱分诊阈值配置

基于 librosa 提取的频谱特征进行分类判断

V3.1.1+dev.20260108.01: 新增 SNR/C50 分诊阈值，支持三层决策策略
"""
from dataclasses import dataclass


@dataclass
class SpectrumThresholds:
    """
    频谱分诊阈值

    V3.1.1+dev.20260108.01: 新增 Brouhaha SNR+C50 三层决策阈值
    """

    # ========== 音乐检测阈值 ==========
    # 谐波比：音乐通常有明确的谐波结构
    harmonic_ratio_music: float = 0.6        # 谐波比高于此值可能有音乐

    # 谱质心：音乐频段丰富，质心偏低
    spectral_centroid_music_low: float = 1500   # Hz，低于此值偏向音乐
    spectral_centroid_music_high: float = 4000  # Hz，高于此值偏向噪音

    # 能量方差：音乐有节奏性能量波动
    energy_variance_music: float = 0.25      # 方差高于此值有节奏性

    # 节拍强度：音乐有明显节拍
    onset_strength_music: float = 0.3        # 节拍强度高于此值

    # ========== 噪音检测阈值 ==========
    # 过零率：噪音（尤其是白噪/风声）ZCR极高
    zcr_noise_high: float = 0.15             # ZCR高于此值可能是噪音
    zcr_variance_noise: float = 0.02         # ZCR方差小说明是稳态噪音

    # 高频能量占比：噪音高频能量占比高
    high_freq_ratio_noise: float = 0.4       # 4kHz以上能量占比超过此值

    # 频谱平坦度：噪音频谱接近平坦
    spectral_flatness_noise: float = 0.5     # 平坦度高于此值偏向噪音

    # ========== 综合判定阈值 ==========
    music_score_threshold: float = 0.35      # 音乐得分超过此值需要分离
    noise_score_threshold: float = 0.45      # 噪音得分超过此值需要分离
    clean_score_threshold: float = 0.7       # 纯净度高于此值跳过分离

    # ========== 分离模型选择阈值 ==========
    heavy_bgm_threshold: float = 0.6         # 重度BGM，使用 mdx_extra
    light_bgm_threshold: float = 0.35        # 轻度BGM，使用 htdemucs

    # ========== Brouhaha SNR+C50 分诊阈值 (V3.1.1+dev.20260108.01) ==========
    # SNR 阈值（信噪比）
    snr_high_threshold: float = 25.0         # SNR >= 此值直接放行（高质量语音）
    snr_low_threshold: float = 12.0          # SNR < 此值强制分离（ASR 可用边界）

    # C50 阈值（清晰度指数/混响）
    c50_good_threshold: float = 5.0          # C50 >= 此值视为良好（普通房间下限）
    c50_bad_threshold: float = -5.0          # C50 < 此值视为严重混响

    # 频谱对比度阈值（Layer 2 决策）
    spectral_contrast_low: float = 15.0      # 频谱对比度低阈值 (dB)，低于此值警戒
    spectral_contrast_critical: float = 12.0 # 频谱对比度临界阈值 (dB)，低于此值分离
    spectral_flatness_high: float = 0.4      # 频谱平坦度高阈值（用于 Layer 2）


# 默认配置实例
DEFAULT_SPECTRUM_THRESHOLDS = SpectrumThresholds()
