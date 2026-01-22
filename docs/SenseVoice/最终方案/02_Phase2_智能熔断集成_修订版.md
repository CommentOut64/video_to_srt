# Phase 2: 智能熔断集成（修订版 v2.1）

> 目标：实现频谱指纹分诊台、熔断决策器、动态权重计算和组合方案矩阵
>
> 工期：2-3天
>
> 版本更新：整合 [06_转录层深度优化_时空解耦架构](./06_转录层深度优化_时空解耦架构.md) 设计

---

## ⚠️ 重要修订

### v2.1 新增（VAD优先 + 频谱分诊）

- ✅ **核心变更**：处理流程调整为 `音频提取 → VAD → 频谱分诊 → 按需分离 → 转录`
- ✅ **新增**：频谱指纹分诊台 `audio_spectrum_classifier.py`（Chunk级别分诊）
- ✅ **新增**：频谱特征提取（ZCR、Spectral Centroid、Harmonic Ratio、Energy Variance）
- ✅ **新增**：分诊阈值配置 `SpectrumThresholds`
- ✅ **新增**：分诊结果数据模型 `SpectrumDiagnosis`

### v2.0 新增（时空解耦架构）

- ✅ **修改**：熔断决策增加 `WHISPER_TEXT_ONLY` 动作（仅取文本，使用伪对齐）
- ✅ **修改**：熔断决策增加 `LLM_ARBITRATE` 动作（LLM 仲裁）
- ✅ **新增**：组合方案矩阵 `solution_matrix.py`
- ✅ **新增**：前端预设方案体系

### v1.0 基础修订

- ✅ **保留**：所有新文件创建（无重复）
- ✅ **修改**：动态权重计算方法集成到现有 `config.py`
- ✅ **无冲突**：本阶段文件均为新建，无需复用现有服务

---

## 一、任务清单

| 任务 | 文件 | 优先级 | 状态 |
|------|------|--------|------|
| **频谱指纹分诊台（v2.1核心）** | `audio_spectrum_classifier.py` | **P0** | **新建** |
| **分诊阈值配置** | `spectrum_thresholds.py` | **P0** | **新建** |
| 频谱分诊数据模型 | `circuit_breaker_models.py` | P0 | 新建 |
| **熔断决策器（时空解耦扩展）** | `fuse_breaker.py` | **P0** | **修改** |
| **组合方案矩阵** | `solution_matrix.py` | **P1** | **新建** |
| 动态权重计算 | `config.py` (修改) | P1 | 修改 |

---

## 二、核心概念

### 2.1 VAD优先原则（v2.1核心变更）

**为什么必须先 VAD 再频谱分诊？**

1. **效率（Efficiency）**：
   - 2小时电影可能只有40分钟有人说话
   - **先VAD**：只对40分钟有效片段进行频谱分析和分离
   - **后VAD**：必须对2小时音频跑Demucs，极大浪费

2. **显存安全（OOM Protection）**：
   - VAD将长音频切分为15-30s的Chunk
   - Demucs对短Chunk处理显存可控
   - 避免对超长音频一次性分离导致OOM

3. **精细控制**：
   - 每个Chunk独立判断是否需要分离
   - 纯净片段直接跳过分离，节省时间

```
旧流程: 音频提取 → BGM检测(全局) → [Demucs全局分离] → VAD → 转录
新流程: 音频提取 → VAD → 频谱分诊(Chunk级) → [按需Chunk分离] → 转录
```

### 2.2 熔断机制设计原则

1. **熔断升级优先于 ASR 复核**
   - 在噪音环境下，优先升级分离模型去除噪音
   - 已升级后才允许 Whisper 复核

2. **时空解耦原则（新增）**
   - Whisper 复核**仅取文本**，时间戳由 SenseVoice 确定
   - 默认使用伪对齐

3. **止损点机制**
   - `max_retry_count = 1`，防止无限循环
   - 超过重试次数后接受低质量结果

---

## 三、频谱指纹分诊台（v2.1核心新增）

### 3.1 分诊原理

频谱分析可以作为第一道防线，可靠地区分以下音频类型：

| 音频类型 | 频谱特征 |
|----------|----------|
| **纯净人声** | 基频集中在85-300Hz，谐波结构清晰，能量方差小 |
| **音乐** | 明确的谐波结构，能量波动有节律，频段丰富 |
| **高频噪音** | 过零率(ZCR)极高且方差小，高频能量占比过大 |
| **脉冲噪音** | 短时能量突增 |

### 3.2 分诊阈值配置

**路径**: `backend/app/core/spectrum_thresholds.py`

```python
"""
频谱分诊阈值配置

基于 librosa 提取的频谱特征进行分类判断
"""
from dataclasses import dataclass


@dataclass
class SpectrumThresholds:
    """频谱分诊阈值"""

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


# 默认配置实例
DEFAULT_SPECTRUM_THRESHOLDS = SpectrumThresholds()
```

### 3.3 分诊数据模型

**路径**: `backend/app/models/circuit_breaker_models.py`（扩展）

```python
from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum


class DiagnosisResult(Enum):
    """分诊结果类型"""
    CLEAN = "clean"           # 纯净人声，无需分离
    MUSIC = "music"           # 检测到音乐
    NOISE = "noise"           # 检测到噪音
    MIXED = "mixed"           # 混合情况


@dataclass
class SpectrumFeatures:
    """频谱特征"""
    # 基础特征
    zcr: float = 0.0                    # 过零率 (Zero Crossing Rate)
    zcr_variance: float = 0.0           # ZCR方差

    # 频谱特征
    spectral_centroid: float = 0.0      # 谱质心 (Hz)
    spectral_bandwidth: float = 0.0     # 谱带宽
    spectral_flatness: float = 0.0      # 频谱平坦度
    spectral_rolloff: float = 0.0       # 频谱滚降点

    # 谐波特征
    harmonic_ratio: float = 0.0         # 谐波比 (Harmonic-to-Noise Ratio)

    # 能量特征
    rms_energy: float = 0.0             # RMS能量
    energy_variance: float = 0.0        # 能量方差
    high_freq_ratio: float = 0.0        # 高频能量占比 (4kHz以上)

    # 节奏特征
    onset_strength: float = 0.0         # 节拍强度
    tempo: float = 0.0                  # 估计BPM


@dataclass
class SpectrumDiagnosis:
    """频谱分诊结果"""
    chunk_index: int                           # Chunk索引
    diagnosis: DiagnosisResult                 # 分诊结果
    need_separation: bool                      # 是否需要分离

    # 评分
    music_score: float = 0.0                   # 音乐得分 (0-1)
    noise_score: float = 0.0                   # 噪音得分 (0-1)
    clean_score: float = 0.0                   # 纯净度得分 (0-1)

    # 推荐的分离模型
    recommended_model: Optional[str] = None    # None / "htdemucs" / "mdx_extra"

    # 原始特征（用于调试）
    features: SpectrumFeatures = field(default_factory=SpectrumFeatures)

    # 决策原因
    reason: str = ""
```

### 3.4 频谱分诊器实现

**路径**: `backend/app/services/audio_spectrum_classifier.py`

```python
"""
频谱指纹分诊台

对每个VAD Chunk进行频谱分析，决定是否需要人声分离。
在VAD切分之后、转录之前执行。
"""
import numpy as np
import logging
from typing import List, Tuple
from dataclasses import dataclass

from ..models.circuit_breaker_models import (
    SpectrumFeatures, SpectrumDiagnosis, DiagnosisResult
)
from ..core.spectrum_thresholds import SpectrumThresholds, DEFAULT_SPECTRUM_THRESHOLDS

logger = logging.getLogger(__name__)


class AudioSpectrumClassifier:
    """音频频谱分诊器"""

    def __init__(self, thresholds: SpectrumThresholds = None):
        self.thresholds = thresholds or DEFAULT_SPECTRUM_THRESHOLDS
        self._librosa = None  # 懒加载

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

        Args:
            audio: 音频数组
            chunk_index: Chunk索引
            sr: 采样率

        Returns:
            SpectrumDiagnosis: 分诊结果
        """
        features = self.extract_features(audio, sr)
        th = self.thresholds

        # 计算各项得分
        music_score = self._calculate_music_score(features)
        noise_score = self._calculate_noise_score(features)
        clean_score = 1.0 - max(music_score, noise_score)

        # 综合判定
        diagnosis = DiagnosisResult.CLEAN
        need_separation = False
        recommended_model = None
        reason = "纯净人声"

        if music_score >= th.music_score_threshold:
            diagnosis = DiagnosisResult.MUSIC
            need_separation = True
            reason = f"检测到音乐 (score={music_score:.2f})"

            # 选择分离模型
            if music_score >= th.heavy_bgm_threshold:
                recommended_model = "mdx_extra"
                reason += " [重度BGM]"
            else:
                recommended_model = "htdemucs"
                reason += " [轻度BGM]"

        elif noise_score >= th.noise_score_threshold:
            diagnosis = DiagnosisResult.NOISE
            need_separation = True
            recommended_model = "htdemucs"
            reason = f"检测到噪音 (score={noise_score:.2f})"

        elif music_score > 0.2 and noise_score > 0.2:
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


# ========== 单例访问 ==========

_classifier_instance = None


def get_spectrum_classifier() -> AudioSpectrumClassifier:
    """获取频谱分诊器单例"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = AudioSpectrumClassifier()
    return _classifier_instance
```

---

## 四、熔断数据模型（v2.1重构）

### 4.1 概念澄清

> **重要**：熔断和后处理增强是两个不同的概念
>
> - **熔断**：转录过程中检测到BGM/Noise标签+低置信度，回溯升级分离模型，属于**实时纠错**
> - **Whisper复核**：转录完成后根据用户配置执行，属于**后处理增强**，不是熔断

### 4.2 数据模型定义

**路径**: `backend/app/models/circuit_breaker_models.py`

```python
"""
熔断机制数据模型（v2.1 概念重构版）

熔断 = 升级分离模型（转录过程中）
后处理增强 = Whisper复核 / LLM校对 / 翻译（转录完成后）
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class SeparationLevel(Enum):
    """人声分离级别"""
    NONE = "none"              # 未分离
    HTDEMUCS = "htdemucs"      # 轻度分离
    MDX_EXTRA = "mdx_extra"    # 重度分离（最高级别）


class FuseAction(Enum):
    """熔断动作（仅升级分离相关）"""
    ACCEPT = "accept"                           # 接受结果，继续流程
    UPGRADE_SEPARATION = "upgrade_separation"   # 升级分离模型，回溯重做


@dataclass
class ChunkProcessState:
    """
    Chunk 处理状态（核心：保留原始音频引用）

    每个 VAD Chunk 在处理过程中需要维护的状态
    """
    chunk_index: int
    start_time: float
    end_time: float

    # ========== 音频引用（关键！）==========
    original_audio: np.ndarray = None          # 原始音频（分离前），用于熔断回溯
    current_audio: np.ndarray = None           # 当前使用的音频（可能已分离）

    # ========== 分离状态 ==========
    separation_level: SeparationLevel = SeparationLevel.NONE
    separation_model_used: Optional[str] = None  # 实际使用的模型名

    # ========== 熔断状态 ==========
    fuse_retry_count: int = 0                  # 熔断重试次数
    max_fuse_retry: int = 1                    # 最大重试次数（止损点）

    # ========== 转录结果 ==========
    transcription_confidence: float = 0.0
    event_tag: Optional[str] = None            # SenseVoice 检测到的事件标签（BGM/Noise等）

    def can_upgrade_separation(self) -> bool:
        """是否可以升级分离模型"""
        if self.fuse_retry_count >= self.max_fuse_retry:
            return False
        if self.separation_level == SeparationLevel.MDX_EXTRA:
            return False  # 已是最高级别
        return True

    def get_next_separation_level(self) -> Optional[SeparationLevel]:
        """获取下一个分离级别"""
        if self.separation_level == SeparationLevel.NONE:
            return SeparationLevel.HTDEMUCS
        elif self.separation_level == SeparationLevel.HTDEMUCS:
            return SeparationLevel.MDX_EXTRA
        return None


@dataclass
class FuseDecision:
    """熔断决策结果"""
    action: FuseAction
    reason: str
    next_separation_level: Optional[SeparationLevel] = None
```

---

## 五、熔断决策器（v2.1重构）

> **注意**：熔断决策器**仅负责升级分离模型**的决策，Whisper复核属于后处理增强，在Phase 3中处理。

**路径**: `backend/app/services/fuse_breaker.py`

```python
"""
熔断决策器（v2.1 概念重构版）

职责：判断是否需要升级分离模型
触发条件：检测到 BGM/Noise 标签 + 低置信度
动作：回溯到原始音频，使用升级的模型重新分离

注意：Whisper复核不在此处理，那是后处理增强阶段的事
"""
import logging
from typing import Optional
from ..models.circuit_breaker_models import (
    FuseAction, FuseDecision, ChunkProcessState, SeparationLevel
)

logger = logging.getLogger(__name__)


class FuseBreaker:
    """
    熔断决策器（v2.1）

    仅负责判断是否需要升级分离模型
    """

    def __init__(
        self,
        fuse_confidence_threshold: float = 0.5,  # 低于此值且有BGM标签才考虑熔断
        bgm_tags: tuple = ('BGM', 'Noise', 'Music', 'Applause')  # 触发熔断的事件标签
    ):
        self.fuse_confidence_threshold = fuse_confidence_threshold
        self.bgm_tags = bgm_tags

    def should_fuse(
        self,
        chunk_state: ChunkProcessState,
        confidence: float,
        event_tag: Optional[str]
    ) -> FuseDecision:
        """
        判断是否需要熔断（升级分离模型）

        Args:
            chunk_state: Chunk 处理状态
            confidence: SenseVoice 转录置信度
            event_tag: SenseVoice 检测到的事件标签

        Returns:
            FuseDecision: 熔断决策
        """
        # 1. 置信度足够高，不需要熔断
        if confidence >= self.fuse_confidence_threshold:
            return FuseDecision(
                action=FuseAction.ACCEPT,
                reason=f"置信度 {confidence:.2f} >= {self.fuse_confidence_threshold}"
            )

        # 2. 没有 BGM/Noise 标签，不需要熔断（可能只是说话不清晰）
        if event_tag not in self.bgm_tags:
            return FuseDecision(
                action=FuseAction.ACCEPT,
                reason=f"无BGM/Noise标签，不触发熔断（置信度低可由后处理增强补救）"
            )

        # 3. 检查是否可以升级分离
        if not chunk_state.can_upgrade_separation():
            return FuseDecision(
                action=FuseAction.ACCEPT,
                reason=f"无法升级分离（已达止损点或最高级别），接受当前结果"
            )

        # 4. 触发熔断：升级分离模型
        next_level = chunk_state.get_next_separation_level()
        return FuseDecision(
            action=FuseAction.UPGRADE_SEPARATION,
            reason=f"检测到 {event_tag} + 低置信度 {confidence:.2f}，升级分离模型",
            next_separation_level=next_level
        )


def execute_fuse_upgrade(
    chunk_state: ChunkProcessState,
    next_level: SeparationLevel,
    demucs_service
) -> ChunkProcessState:
    """
    执行熔断升级：使用原始音频重新分离

    Args:
        chunk_state: Chunk 处理状态
        next_level: 目标分离级别
        demucs_service: Demucs 服务实例

    Returns:
        ChunkProcessState: 更新后的状态
    """
    logger.info(
        f"Chunk {chunk_state.chunk_index} 熔断升级: "
        f"{chunk_state.separation_level.value} → {next_level.value}"
    )

    # 关键：使用原始音频（分离前）进行重新分离
    original_audio = chunk_state.original_audio
    if original_audio is None:
        logger.error("熔断失败：原始音频引用丢失")
        return chunk_state

    # 选择模型
    model_name = "htdemucs" if next_level == SeparationLevel.HTDEMUCS else "mdx_extra"

    # 执行分离
    separated_audio = demucs_service.separate_chunk(
        audio=original_audio,
        model=model_name
    )

    # 更新状态
    chunk_state.current_audio = separated_audio
    chunk_state.separation_level = next_level
    chunk_state.separation_model_used = model_name
    chunk_state.fuse_retry_count += 1

    logger.info(f"Chunk {chunk_state.chunk_index} 熔断升级完成，重试次数: {chunk_state.fuse_retry_count}")

    return chunk_state


# ========== 单例访问 ==========

_fuse_breaker_instance = None


def get_fuse_breaker() -> FuseBreaker:
    """获取熔断决策器单例"""
    global _fuse_breaker_instance
    if _fuse_breaker_instance is None:
        _fuse_breaker_instance = FuseBreaker()
    return _fuse_breaker_instance
```

---

## 六、组合方案矩阵（新增）

**路径**: `backend/app/services/solution_matrix.py`

```python
"""
组合方案矩阵 (Solution Matrix)

提供高度模块化的配置，允许在"速度、成本、质量"之间自由组合。
前端预设对应后端具体配置。
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EnhancementMode(Enum):
    """增强模式"""
    OFF = "off"                      # A1: SenseVoice Only
    SMART_PATCH = "smart_patch"      # B1: SenseVoice + Whisper Partial（推荐）
    DEEP_LISTEN = "deep_listen"      # C1: SenseVoice + Whisper Full


class ProofreadMode(Enum):
    """校对模式"""
    OFF = "off"
    SPARSE = "sparse"                # P1: 按需修复（推荐）
    FULL = "full"                    # P2: 全文精修


class TranslateMode(Enum):
    """翻译模式"""
    OFF = "off"
    FULL = "full"                    # T1: 全量翻译
    PARTIAL = "partial"              # T2: 部分翻译


@dataclass
class SolutionConfig:
    """方案配置"""
    preset_id: str = "default"
    enhancement: EnhancementMode = EnhancementMode.OFF
    proofread: ProofreadMode = ProofreadMode.OFF
    translate: TranslateMode = TranslateMode.OFF
    target_language: Optional[str] = None

    # 高级选项
    confidence_threshold: float = 0.6

    @classmethod
    def from_preset(cls, preset_id: str) -> 'SolutionConfig':
        """根据预设ID创建配置"""
        presets = {
            'default': cls(
                preset_id='default',
                enhancement=EnhancementMode.OFF,
                proofread=ProofreadMode.OFF,
                translate=TranslateMode.OFF
            ),
            'preset1': cls(
                preset_id='preset1',
                enhancement=EnhancementMode.SMART_PATCH,
                proofread=ProofreadMode.OFF,
                translate=TranslateMode.OFF
            ),
            'preset2': cls(
                preset_id='preset2',
                enhancement=EnhancementMode.SMART_PATCH,
                proofread=ProofreadMode.SPARSE,
                translate=TranslateMode.OFF
            ),
            'preset3': cls(
                preset_id='preset3',
                enhancement=EnhancementMode.SMART_PATCH,
                proofread=ProofreadMode.FULL,
                translate=TranslateMode.OFF
            ),
            'preset4': cls(
                preset_id='preset4',
                enhancement=EnhancementMode.SMART_PATCH,
                proofread=ProofreadMode.FULL,
                translate=TranslateMode.FULL,
                target_language='en'
            ),
            'preset5': cls(
                preset_id='preset5',
                enhancement=EnhancementMode.SMART_PATCH,
                proofread=ProofreadMode.FULL,
                translate=TranslateMode.PARTIAL
            ),
        }
        return presets.get(preset_id, cls())


# ========== 方案矩阵描述 ==========

SOLUTION_MATRIX = {
    # ========== 基础层 ==========
    "A1": {
        "name": "SenseVoice Only",
        "flow": ["sensevoice"],
        "scenario": "实时预览、即时日志",
        "note": "极速。原汁原味"
    },
    "B1": {
        "name": "SenseVoice + Whisper Partial",
        "flow": ["sensevoice", "whisper_partial", "pseudo_align"],
        "scenario": "嘈杂、多BGM环境",
        "note": "默认推荐。仅对低置信度片段进行Whisper重听，伪对齐"
    },
    # ========== 语义层（可叠加） ==========
    "P1": {
        "name": "LLM Partial Proof",
        "flow": ["llm_sparse_proof"],
        "scenario": "个人笔记、日常Vlog",
        "note": "性价比之王。按需稀疏校对，节省90%+Token"
    },
    "P2": {
        "name": "LLM Full Proof",
        "flow": ["llm_full_proof"],
        "scenario": "正式出版、文稿整理",
        "note": "高质量。全量滑动窗口，润色口语、修正逻辑"
    },

    # ========== 翻译层 ==========
    "T1": {
        "name": "LLM Full Trans",
        "flow": ["llm_full_translate"],
        "scenario": "跨语言内容",
        "note": "传统的全量翻译"
    },
    "T2": {
        "name": "LLM Partial Trans",
        "flow": ["llm_partial_translate"],
        "scenario": "教学重点标注",
        "note": "仅翻译用户指定的重点段落"
    }
}


# ========== 前端预设方案 ==========

FRONTEND_PRESETS = [
    {
        "id": "default",
        "name": "SenseVoice Only",
        "description": "极速模式，仅使用 SenseVoice 转录",
        "timeMultiplier": 0.1
    },
    {
        "id": "preset1",
        "name": "智能复核",
        "description": "SV + Whisper 局部复核，平衡速度与质量",
        "timeMultiplier": 0.15
    },
    {
        "id": "preset2",
        "name": "轻度校对",
        "description": "智能复核 + LLM 按需校对问题片段",
        "timeMultiplier": 0.2
    },
    {
        "id": "preset3",
        "name": "深度校对",
        "description": "智能复核 + LLM 全文精修润色",
        "timeMultiplier": 0.3
    },
    {
        "id": "preset4",
        "name": "校对+翻译",
        "description": "深度校对 + 全文翻译（同步处理）",
        "timeMultiplier": 0.5
    },
    {
        "id": "preset5",
        "name": "校对+重点翻译",
        "description": "深度校对 + 仅翻译标记的重点段落",
        "timeMultiplier": 0.35
    }
]
```

---

## 七、动态权重计算

**修改文件**: `backend/app/core/config.py`

```python
def calculate_dynamic_weights(
    self,
    engine: str,
    total_segments: int,
    segments_to_separate: int,
    segments_to_retry: int
) -> Dict[str, int]:
    """
    根据引擎和实际场景动态计算权重

    Args:
        engine: 'faster_whisper' | 'sensevoice'
        total_segments: 总片段数
        segments_to_separate: 需要分离的片段数
        segments_to_retry: 需要复核的片段数

    Returns:
        动态权重字典
    """
    if engine == 'faster_whisper':
        return self.PHASE_WEIGHTS.copy()

    base_weights = self.PHASE_WEIGHTS.copy()

    if total_segments > 0:
        sep_ratio = segments_to_separate / total_segments
        retry_ratio = segments_to_retry / total_segments

        base_weights['demucs_global'] = int(15 * sep_ratio)
        base_weights['transcribe'] = 70
        base_weights['align'] = 0

        if 'retry' not in base_weights:
            base_weights['retry'] = 0
        base_weights['retry'] = int(15 * retry_ratio)

        used = sum(base_weights.values())
        if used < 100:
            base_weights['transcribe'] += (100 - used)

    return base_weights
```

---

## 八、验收标准

### 基础能力

- [ ] 频谱检测器可正确识别纯人声和BGM片段
- [ ] 熔断决策器遵循"升级优先于复核"原则
- [ ] 动态权重计算符合预期

### 频谱指纹分诊台（v2.1核心新增）

- [ ] `AudioSpectrumClassifier` 可正确提取频谱特征
- [ ] 音乐检测：高谐波比、质心在1500-4000Hz、有节奏性能量波动
- [ ] 噪音检测：高ZCR、高频能量占比高、频谱平坦
- [ ] 分诊结果正确区分 CLEAN/MUSIC/NOISE/MIXED
- [ ] 分离模型选择正确：轻度BGM用htdemucs，重度BGM用mdx_extra

### 时空解耦架构

- [ ] 熔断决策器支持 `WHISPER_TEXT_ONLY` 动作
- [ ] 熔断决策器支持 `LLM_ARBITRATE` 动作
- [ ] 组合方案矩阵正确配置 6 个预设
- [ ] `SolutionConfig.from_preset()` 正确返回配置

---

## 九、注意事项

1. **止损点机制**（P0 优先级）：
   - `max_retry_count = 1`，防止无限重试
   - 超过重试次数后会接受低质量结果

2. **时空解耦原则**：
   - Whisper 复核时**仅取文本**，弃用其时间戳
   - 默认使用伪对齐

3. **组合方案矩阵**：
   - 前端预设对应后端 `SolutionConfig`
   - 高级用户可自定义组合

---

## 十、下一步

完成 Phase 2（修订版 v2.1）后，进入 [Phase 3: 转录服务重构（修订版）](./03_Phase3_转录服务重构_修订版.md)
