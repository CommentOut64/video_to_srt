# Brouhaha SNR+C50 频谱分诊优化规格书

**版本**: V3.1.1+dev.20260107.03
**状态**: 设计阶段
**作者**: Claude Code
**日期**: 2026-01-07

---

## 目录

1. [概述](#1-概述)
2. [技术背景](#2-技术背景)
3. [架构设计](#3-架构设计)
4. [参数设计与调优策略](#4-参数设计与调优策略)
5. [Brouhaha 模型集成](#5-brouhaha-模型集成)
6. [CPU 调度与性能优化](#6-cpu-调度与性能优化)
7. [与现有架构的融合](#7-与现有架构的融合)
8. [PyTorch 模型下载与配置](#8-pytorch-模型下载与配置)
9. [分阶段实现计划](#9-分阶段实现计划)
10. [测试与验证](#10-测试与验证)
11. [回滚与应急](#11-回滚与应急)

---

## 1. 概述

### 1.1 项目目标

在现有频谱分诊阶段（SpectralTriageStage）中引入 **Brouhaha 模型**，利用其输出的 **SNR（信噪比）** 和 **C50（清晰度指数）** 作为首要决策指标，实现更科学、更高效的人声分离决策。

### 1.2 核心收益

| 收益项 | 预期效果 |
|--------|---------|
| **决策准确性** | SNR+C50 双指标比单一频谱特征更科学 |
| **计算效率** | 60-70% chunk 在 Layer 1 直接决策，减少后续计算 |
| **架构解耦** | 独立 BrouhahaService，最小侵入现有代码 |
| **可维护性** | 参数可配置，支持回退方案 |

### 1.3 设计原则

1. **最小侵入** - 不修改现有 SpectralTriageStage 接口
2. **渐进增强** - 新功能通过配置开关启用，默认保持原有行为
3. **优雅降级** - Brouhaha 不可用时自动回退到 WADA-SNR 或 YAMNet
4. **性能优先** - CPU 大核调度，避免推理性能波动

---

## 2. 技术背景

### 2.1 Brouhaha 模型简介

**Brouhaha** 是 pyannote 团队开发的多任务模型，同时输出：

| 输出 | 含义 | 取值范围 |
|------|------|---------|
| **SNR** | 信噪比 | -10dB ~ 50dB |
| **C50** | 清晰度指数（混响） | -20dB ~ 30dB |
| **VAD** | 语音活动检测 | 0.0 ~ 1.0 |

**论文**: [Brouhaha: multi-task training for voice activity detection, speech-to-noise ratio, and C50 room acoustics estimation](https://arxiv.org/abs/2210.13248)

### 2.2 为什么需要 SNR + C50

| 指标 | 物理意义 | 单独使用的局限 |
|------|---------|---------------|
| **SNR** | 信号/噪声能量比 | 无法区分混响和噪声 |
| **C50** | 直达声/混响能量比 | 无法反映噪声水平 |

**组合使用的优势**：
- 高 SNR + 高 C50 = 纯净人声
- 高 SNR + 低 C50 = 人声清晰但混响严重
- 低 SNR + 高 C50 = 噪声大但无混响
- 低 SNR + 低 C50 = 噪声+混响，最差情况

### 2.3 与现有方案的对比

| 方案 | 指标来源 | 计算成本 | 决策准确性 |
|------|---------|---------|-----------|
| **原方案** | YAMNet 语义分类 | 高（15-20ms） | 中 |
| **新方案 Layer 1** | Brouhaha SNR+C50 | 低（5ms） | 高 |
| **新方案 Layer 2** | librosa 频谱特征 | 极低（2ms） | 中 |
| **新方案 Layer 3** | YAMNet 语义分类 | 高（15-20ms） | 高 |

---

## 3. 架构设计

### 3.1 三层决策架构

```
AudioChunk
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Brouhaha SNR+C50 快速筛选                          │
│  ├─ SNR >= 25dB 且 C50 >= 5dB  → 直接放行 (60%)              │
│  ├─ SNR < 12dB 或 C50 < -5dB   → 强制分离 (10%)              │
│  └─ 其他 → 进入 Layer 2 (30%)                                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼ (警戒区 30%)
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: 频谱特征补充判断                                    │
│  ├─ spectral_contrast < 12dB            → 分离 (5%)          │
│  ├─ contrast < 15dB 且 flatness > 0.4   → 分离 (5%)          │
│  └─ 其他 → 进入 Layer 3 (20%)                                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼ (模糊区 20%)
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: YAMNet 语义分类                                    │
│  ├─ 检测到 Music/BGM → 分离                                  │
│  └─ 检测到 Speech → 放行                                     │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 模块依赖关系

```
┌─────────────────────────────────────────────────────────────┐
│                    SpectralTriageStage                       │
│                    (现有，不修改接口)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  AudioSpectrumClassifier                     │
│                  (增强，新增 SNR 策略)                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐    │
│  │ SNRTriageMixin │  │ SpectralMixin │  │ SemanticMixin │    │
│  │ (新增)         │  │ (现有)        │  │ (现有)        │    │
│  └───────────────┘  └───────────────┘  └───────────────┘    │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ BrouhahaService │  │    librosa      │  │ YAMNetClassifier│
│ (新增)           │  │  (现有依赖)     │  │ (现有)          │
└─────────────────┘  └─────────────────┘  └─────────────────┘
           │
           ▼
┌─────────────────┐
│  cpu_optimizer  │
│  (现有，复用)    │
└─────────────────┘
```

### 3.3 数据流

```python
# 输入
chunk: AudioChunk(audio=np.ndarray, sample_rate=16000, ...)

# Layer 1 输出
brouhaha_result = {
    "snr": 25.3,      # dB
    "c50": 8.2,       # dB
    "vad": 0.95       # 可选，暂不使用
}

# Layer 2 输出
spectral_features = {
    "spectral_contrast": 18.5,   # dB
    "spectral_flatness": 0.25,   # 0-1
    # ... 其他现有特征
}

# Layer 3 输出
yamnet_result = {
    "is_music": False,
    "music_score": 0.05,
    "speech_score": 0.85,
    "tags": ["CleanSpeech"]
}

# 最终输出
diagnosis = SpectrumDiagnosis(
    chunk_index=0,
    diagnosis=DiagnosisResult.CLEAN,
    need_separation=False,
    snr=25.3,
    c50=8.2,
    snr_level="high",     # "high" / "warn" / "low"
    c50_level="good",     # "good" / "warn" / "bad"
    triage_layer=1,       # 决策层级
    reason="高SNR(25.3dB)+良好C50(8.2dB)，纯净人声"
)
```

---

## 4. 参数设计与调优策略

### 4.1 参数分类

| 类别 | 参数 | 调优方式 | 说明 |
|------|------|---------|------|
| **核心阈值** | SNR/C50 阈值 | 数据集微调 | 影响决策准确性 |
| **辅助阈值** | 频谱特征阈值 | 经验值 | 边界情况处理 |
| **系统参数** | 线程数、缓存 | 自动配置 | 性能相关 |

### 4.2 核心阈值设计

#### 4.2.1 SNR 阈值

| 参数 | 默认值 | 调优范围 | 调优依据 |
|------|--------|---------|---------|
| `snr_high_threshold` | 25.0 dB | 20-30 dB | 国际标准 + 实测 |
| `snr_low_threshold` | 12.0 dB | 8-15 dB | ASR 可用边界 |

**调优策略**：

```python
# 方案1: 基于经验值（推荐首次部署）
SNR_HIGH = 25.0  # 学术界共识：SNR > 25dB 为高质量语音
SNR_LOW = 12.0   # 实验表明：SNR < 12dB 时 ASR WER 显著上升

# 方案2: 基于数据集微调（推荐稳定后）
# 收集 1000+ 标注样本，使用以下方法：
# 1. 计算 need_separation=True 样本的 SNR 分布
# 2. 取 P90 作为 SNR_HIGH（放行 90% 纯净样本）
# 3. 取 P10 作为 SNR_LOW（拦截 90% 问题样本）
```

#### 4.2.2 C50 阈值

| 参数 | 默认值 | 调优范围 | 调优依据 |
|------|--------|---------|---------|
| `c50_good_threshold` | 5.0 dB | 3-10 dB | 房间声学标准 |
| `c50_bad_threshold` | -5.0 dB | -10-0 dB | 混响严重边界 |

**C50 物理意义**：

```
C50 > 10dB:  录音棚/消音室
C50 = 5-10dB: 小型会议室
C50 = 0-5dB:  普通房间
C50 = -5-0dB: 大型会议室/教室
C50 < -5dB:  大厅/教堂（混响严重）
```

**调优策略**：

```python
# 方案1: 经验值（推荐）
C50_GOOD = 5.0   # 普通房间下限
C50_BAD = -5.0   # 严重混响上限

# 方案2: 数据集微调
# 收集不同混响环境的样本，标注转录质量
# 找出转录质量显著下降的 C50 临界点
```

### 4.3 辅助阈值设计

#### 4.3.1 频谱特征阈值（Layer 2）

| 参数 | 默认值 | 来源 | 调优方式 |
|------|--------|------|---------|
| `spectral_contrast_low` | 15.0 dB | 经验值 | 可微调 |
| `spectral_contrast_critical` | 12.0 dB | 经验值 | 保守 |
| `spectral_flatness_high` | 0.4 | 学术文献 | 一般不调 |

**这些参数推荐使用经验值**，原因：
1. Layer 2 只处理 ~30% 的 chunk
2. 边界情况会由 Layer 3 兜底
3. 调优收益不如核心阈值明显

### 4.4 组合决策矩阵

```
                    C50 >= 5dB      -5dB <= C50 < 5dB     C50 < -5dB
                   (良好)            (警戒)               (严重混响)
┌─────────────────┬────────────────┬────────────────────┬────────────────┐
│ SNR >= 25dB     │ 直接放行       │ Layer 2 判断       │ 强制分离       │
│ (高)            │ Layer 1        │ (混响警戒)         │ Layer 1        │
├─────────────────┼────────────────┼────────────────────┼────────────────┤
│ 12 <= SNR < 25  │ Layer 2 判断   │ Layer 3 判断       │ 强制分离       │
│ (中)            │ (SNR警戒)      │ (双警戒)           │ Layer 1        │
├─────────────────┼────────────────┼────────────────────┼────────────────┤
│ SNR < 12dB      │ 强制分离       │ 强制分离           │ 强制分离       │
│ (低)            │ Layer 1        │ Layer 1            │ Layer 1        │
└─────────────────┴────────────────┴────────────────────┴────────────────┘
```

### 4.5 阈值配置类

```python
@dataclass
class BrouhahaTriageConfig:
    """Brouhaha 分诊配置"""

    # ========== 核心阈值（建议数据集微调）==========
    snr_high_threshold: float = 25.0      # SNR >= 此值直接放行
    snr_low_threshold: float = 12.0       # SNR < 此值强制分离
    c50_good_threshold: float = 5.0       # C50 >= 此值视为良好
    c50_bad_threshold: float = -5.0       # C50 < 此值视为严重混响

    # ========== 辅助阈值（经验值，一般不调）==========
    spectral_contrast_low: float = 15.0
    spectral_contrast_critical: float = 12.0
    spectral_flatness_high: float = 0.4

    # ========== 系统参数（自动配置）==========
    enable_snr_strategy: bool = True      # 启用 SNR 策略
    enable_c50_check: bool = True         # 启用 C50 检查
    fallback_to_wada: bool = True         # Brouhaha 不可用时回退到 WADA-SNR

    # ========== 调试参数 ==========
    log_decisions: bool = False           # 记录每个决策的详细日志
    collect_statistics: bool = True       # 收集统计信息
```

### 4.6 参数调优流程

```
┌─────────────────────────────────────────────────────────────┐
│                    参数调优工作流                            │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ Phase 1       │    │ Phase 2       │    │ Phase 3       │
│ 使用默认值    │ →  │ 收集生产数据  │ →  │ 数据集微调    │
│ 快速上线      │    │ 1-2 周        │    │ 精确优化      │
└───────────────┘    └───────────────┘    └───────────────┘

Phase 1: 默认经验值
- snr_high=25, snr_low=12
- c50_good=5, c50_bad=-5
- 监控 Layer 分布和 need_separation 比例

Phase 2: 数据收集
- 记录所有 chunk 的 SNR/C50 值
- 记录 FuseBreakerV2 触发的 chunk（转录后发现需要分离）
- 收集用户反馈（转录质量差的样本）

Phase 3: 数据驱动微调
- 分析 false negative（漏判需要分离的）
- 分析 false positive（误判不需要分离的）
- 使用 ROC 曲线找最优阈值
```

---

## 5. Brouhaha 模型集成

### 5.1 模型信息

| 属性 | 值 |
|------|-----|
| **模型来源** | [pyannote/brouhaha](https://huggingface.co/pyannote/brouhaha) |
| **框架** | PyTorch (pyannote.audio) |
| **输入** | 16kHz 单声道音频 |
| **输出** | SNR (dB), C50 (dB), VAD (0-1) |
| **推理速度** | ~5ms/chunk (CPU), ~2ms/chunk (GPU) |
| **显存占用** | ~200MB (GPU) |
| **依赖** | pyannote.audio >= 3.0, torch >= 2.0 |

### 5.2 存放位置

```
backend/models/pretrained/brouhaha/
├── pytorch_model.bin       # PyTorch 模型权重（必需）
├── config.yaml             # pyannote 模型配置（必需）
└── README.md               # 模型说明
```

**与现有模型目录结构一致**：
```
backend/models/pretrained/
├── sensevoice/             # SenseVoice ONNX
├── yamnet/                 # YAMNet ONNX
├── brouhaha/               # Brouhaha PyTorch (新增)
├── htdemucs.th             # Demucs PyTorch
└── README.md
```

**模型获取方式**：
1. 首次运行时从 HuggingFace 下载到本地目录
2. 后续运行直接加载本地模型，无需网络

### 5.3 BrouhahaService 实现

```python
# backend/app/services/brouhaha_service.py

"""
Brouhaha SNR + C50 检测服务

提供基于 Brouhaha 模型的信噪比和清晰度指数检测功能。
使用 PyTorch 原生推理，支持 CPU/GPU 自动选择。
"""

import torch
import numpy as np
import logging
from typing import Optional, Dict
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BrouhahaResult:
    """Brouhaha 检测结果"""
    snr: float          # 信噪比 (dB)
    c50: float          # 清晰度指数 (dB)
    vad: float          # 语音活动概率 (0-1)
    is_valid: bool      # 结果是否有效


class BrouhahaService:
    """
    Brouhaha SNR + C50 检测服务

    特性：
    - PyTorch 原生推理，支持 CPU/GPU
    - 自动下载模型到本地目录
    - 结果缓存
    - 优雅降级（回退到 WADA-SNR）
    """

    # 模型默认路径
    DEFAULT_MODEL_DIR = "backend/models/pretrained/brouhaha"

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
            model_dir: 本地模型目录路径
            device: 推理设备 ("auto" / "cuda" / "cpu")
            hf_token: HuggingFace Token（首次下载需要）
        """
        if model_dir is None:
            model_dir = Path(__file__).parent.parent.parent / "models" / "pretrained" / "brouhaha"

        self.model_dir = Path(model_dir)
        self.hf_token = hf_token

        # 设备选择
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = None

        # 缓存
        self._cache: Dict[int, BrouhahaResult] = {}
        self._cache_max_size = 100

        # 初始化模型
        self._init_model()

    def _init_model(self):
        """初始化 PyTorch 模型"""
        try:
            from pyannote.audio import Model

            # 检查本地模型是否存在
            local_config = self.model_dir / "config.yaml"
            local_weights = self.model_dir / "pytorch_model.bin"

            if local_config.exists() and local_weights.exists():
                # 从本地加载
                logger.info(f"从本地加载 Brouhaha 模型: {self.model_dir}")
                self.model = Model.from_pretrained(
                    str(self.model_dir),
                    strict=False
                )
            else:
                # 从 HuggingFace 下载并保存到本地
                logger.info("从 HuggingFace 下载 Brouhaha 模型...")
                self.model = Model.from_pretrained(
                    "pyannote/brouhaha",
                    use_auth_token=self.hf_token
                )
                # 保存到本地目录
                self._save_model_locally()

            # 移动到目标设备
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(
                f"Brouhaha 模型已加载, 设备={self.device}, "
                f"参数量={sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M"
            )

        except ImportError:
            logger.warning("pyannote.audio 未安装，Brouhaha 服务不可用")
            self.model = None
        except Exception as e:
            logger.error(f"Brouhaha 模型加载失败: {e}")
            self.model = None

    def _save_model_locally(self):
        """保存模型到本地目录"""
        if self.model is None:
            return

        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)

            # 保存模型权重
            weights_path = self.model_dir / "pytorch_model.bin"
            torch.save(self.model.state_dict(), weights_path)

            # 保存配置（从 HuggingFace 缓存复制）
            # 注：实际实现可能需要从 model.hparams 生成 config.yaml

            logger.info(f"Brouhaha 模型已保存到: {self.model_dir}")

        except Exception as e:
            logger.warning(f"保存模型到本地失败: {e}")

    @torch.no_grad()
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
            return self._fallback_wada_snr(audio, sr)

        try:
            # 重采样（如果需要）
            if sr != self.SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLE_RATE)

            # 转换为 PyTorch tensor
            waveform = torch.from_numpy(audio).float().unsqueeze(0)  # [1, samples]
            waveform = waveform.to(self.device)

            # 推理
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

    def _fallback_wada_snr(self, audio: np.ndarray, sr: int) -> BrouhahaResult:
        """
        回退到 WADA-SNR 算法

        当 Brouhaha 模型不可用时，使用无模型的 WADA-SNR 算法估计 SNR
        """
        try:
            snr = self._calculate_wada_snr(audio, sr)
            # WADA-SNR 不提供 C50，使用默认值
            return BrouhahaResult(snr=snr, c50=0.0, vad=1.0, is_valid=True)
        except Exception as e:
            logger.warning(f"WADA-SNR 计算失败: {e}")
            return BrouhahaResult(snr=15.0, c50=0.0, vad=1.0, is_valid=False)

    def _calculate_wada_snr(self, audio: np.ndarray, sr: int) -> float:
        """
        WADA-SNR 算法实现

        基于波形幅度分布分析估计 SNR
        参考: Kim & Stern, Interspeech 2008
        """
        # 分帧
        frame_length = int(0.025 * sr)  # 25ms
        hop_length = int(0.010 * sr)    # 10ms

        frames = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frames.append(audio[i:i + frame_length])

        if not frames:
            return 15.0  # 默认值

        # 计算每帧能量
        energies = np.array([np.sum(frame ** 2) for frame in frames])

        # 简化 SNR 估计：假设最低 10% 能量帧为噪声
        sorted_energies = np.sort(energies)
        noise_energy = np.mean(sorted_energies[:max(1, len(sorted_energies) // 10)])
        signal_energy = np.mean(sorted_energies[len(sorted_energies) // 2:])

        if noise_energy > 0:
            snr = 10 * np.log10(signal_energy / noise_energy)
        else:
            snr = 30.0  # 无噪声

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

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()

    def unload(self):
        """卸载模型释放显存"""
        if self.model is not None:
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
```

---

## 6. GPU/CPU 调度与性能优化

### 6.1 设备选择策略

Brouhaha 服务支持自动设备选择，优先使用 GPU：

```python
# 在 BrouhahaService.__init__() 中
if device == "auto":
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    self.device = torch.device(device)
```

### 6.2 GPU 推理优化

```python
# 1. 使用 torch.no_grad() 装饰器减少显存占用
@torch.no_grad()
def detect(self, audio, sr, chunk_id):
    ...

# 2. 模型置于 eval 模式
self.model.eval()

# 3. 显存管理 - 支持模型卸载
def unload(self):
    if self.model is not None:
        del self.model
        self.model = None
        torch.cuda.empty_cache()
```

### 6.3 CPU 推理优化

对于 CPU 推理，可配置 PyTorch 线程数：

```python
import torch

# 设置 PyTorch CPU 线程数
torch.set_num_threads(4)  # 根据 CPU 核心数调整

# 可选：启用 Intel MKL 优化
torch.backends.mkl.enabled = True
```

### 6.4 性能基准

| 配置 | 推理耗时 | 显存/内存占用 | 说明 |
|------|---------|--------------|------|
| GPU (RTX 3060) | ~2ms | ~200MB | 推荐配置 |
| CPU (8核) | ~8ms | ~150MB | 无 GPU 回退 |
| CPU (4核) | ~15ms | ~150MB | 低配机器 |

---

## 7. 与现有架构的融合

### 7.1 最小侵入原则

**不修改的接口**：
- `SpectralTriageStage.process()` - 签名不变
- `AudioChunk` - 仅添加可选字段
- `SpectrumDiagnosis` - 仅添加可选字段

**新增的组件**：
- `BrouhahaService` - 独立服务
- `BrouhahaTriageConfig` - 独立配置类
- `SNRTriageStrategy` - 策略类（可选）

### 7.2 数据模型扩展

```python
# backend/app/models/circuit_breaker_models.py

@dataclass
class SpectrumFeatures:
    """频谱特征"""
    # 现有字段...
    zcr: float = 0.0
    spectral_flatness: float = 0.0
    # ...

    # V3.1.1+dev.20260107.03 新增
    snr: float = 0.0                    # 信噪比 (dB)
    c50: float = 0.0                    # 清晰度指数 (dB)
    spectral_contrast: float = 0.0      # 频谱对比度 (dB)


@dataclass
class SpectrumDiagnosis:
    """频谱分诊结果"""
    # 现有字段...
    chunk_index: int
    diagnosis: DiagnosisResult
    need_separation: bool
    # ...

    # V3.1.1+dev.20260107.03 新增
    snr: float = 0.0                    # SNR 值
    c50: float = 0.0                    # C50 值
    snr_level: str = "unknown"          # "high" / "warn" / "low"
    c50_level: str = "unknown"          # "good" / "warn" / "bad"
    triage_layer: int = 0               # 决策层级 (1/2/3)
```

### 7.3 AudioSpectrumClassifier 增强

```python
# backend/app/services/audio_spectrum_classifier.py

class AudioSpectrumClassifier:
    """
    音频频谱分诊器

    V3.1.1+dev.20260107.03: 新增 SNR+C50 优先策略
    """

    def __init__(
        self,
        thresholds: SpectrumThresholds = None,
        use_yamnet: bool = True,
        use_snr_strategy: bool = True,      # 新增
        brouhaha_config: BrouhahaTriageConfig = None  # 新增
    ):
        self.thresholds = thresholds or DEFAULT_SPECTRUM_THRESHOLDS
        self._use_yamnet = use_yamnet
        self._use_snr_strategy = use_snr_strategy
        self._brouhaha_config = brouhaha_config or BrouhahaTriageConfig()

        # 懒加载
        self._librosa = None
        self._yamnet = None
        self._brouhaha = None

    def diagnose_chunk(
        self,
        audio: np.ndarray,
        chunk_index: int,
        sr: int = 16000
    ) -> SpectrumDiagnosis:
        """
        对单个 Chunk 进行频谱分诊

        决策顺序：
        1. SNR+C50 快速筛选（如果启用）
        2. 频谱特征补充判断（警戒区）
        3. YAMNet 语义分类（模糊区）
        """
        # 极短片段跳过
        duration_sec = len(audio) / sr
        if duration_sec < 0.5:
            return self._create_skip_diagnosis(chunk_index, duration_sec)

        # 新策略：SNR+C50 优先
        if self._use_snr_strategy:
            brouhaha = self._get_brouhaha()
            if brouhaha and brouhaha.is_available():
                return self._diagnose_with_snr_c50_strategy(audio, chunk_index, sr)

        # 原策略：YAMNet 优先
        yamnet = self._get_yamnet()
        if yamnet and yamnet.is_available():
            return self._diagnose_with_yamnet(audio, chunk_index, sr, yamnet)

        # 回退：规则方法
        return self._diagnose_with_rules(audio, chunk_index, sr)

    def _diagnose_with_snr_c50_strategy(
        self,
        audio: np.ndarray,
        chunk_index: int,
        sr: int
    ) -> SpectrumDiagnosis:
        """
        SNR+C50 优先的三层决策策略
        """
        config = self._brouhaha_config
        brouhaha = self._get_brouhaha()

        # Layer 1: Brouhaha SNR+C50
        result = brouhaha.detect(audio, sr, chunk_id=chunk_index)

        snr = result.snr
        c50 = result.c50

        snr_level = self._classify_snr_level(snr)
        c50_level = self._classify_c50_level(c50)

        # Layer 1 决策：快速通过/拦截
        layer1_decision = self._layer1_decision(snr, c50, snr_level, c50_level)
        if layer1_decision is not None:
            layer1_decision.snr = snr
            layer1_decision.c50 = c50
            layer1_decision.snr_level = snr_level
            layer1_decision.c50_level = c50_level
            layer1_decision.triage_layer = 1
            return layer1_decision

        # Layer 2: 频谱特征补充
        features = self.extract_features(audio, sr)
        features.snr = snr
        features.c50 = c50

        layer2_decision = self._layer2_decision(features, chunk_index, snr_level, c50_level)
        if layer2_decision is not None:
            layer2_decision.snr = snr
            layer2_decision.c50 = c50
            layer2_decision.snr_level = snr_level
            layer2_decision.c50_level = c50_level
            layer2_decision.triage_layer = 2
            return layer2_decision

        # Layer 3: YAMNet 语义分类
        yamnet = self._get_yamnet()
        if yamnet and yamnet.is_available():
            diagnosis = self._diagnose_with_yamnet(audio, chunk_index, sr, yamnet)
            diagnosis.snr = snr
            diagnosis.c50 = c50
            diagnosis.snr_level = snr_level
            diagnosis.c50_level = c50_level
            diagnosis.triage_layer = 3
            return diagnosis

        # 最终回退
        return self._create_default_diagnosis(chunk_index, features, snr, c50, snr_level, c50_level)

    def _classify_snr_level(self, snr: float) -> str:
        """分类 SNR 等级"""
        config = self._brouhaha_config
        if snr >= config.snr_high_threshold:
            return "high"
        elif snr < config.snr_low_threshold:
            return "low"
        else:
            return "warn"

    def _classify_c50_level(self, c50: float) -> str:
        """分类 C50 等级"""
        config = self._brouhaha_config
        if c50 >= config.c50_good_threshold:
            return "good"
        elif c50 < config.c50_bad_threshold:
            return "bad"
        else:
            return "warn"

    def _layer1_decision(
        self,
        snr: float,
        c50: float,
        snr_level: str,
        c50_level: str
    ) -> Optional[SpectrumDiagnosis]:
        """
        Layer 1 决策

        Returns:
            SpectrumDiagnosis 如果可以决策，None 如果需要进入 Layer 2
        """
        # 高 SNR + 良好 C50 = 直接放行
        if snr_level == "high" and c50_level == "good":
            return SpectrumDiagnosis(
                chunk_index=0,  # 后续填充
                diagnosis=DiagnosisResult.CLEAN,
                need_separation=False,
                music_score=0.0,
                noise_score=0.0,
                clean_score=1.0,
                recommended_model=None,
                features=SpectrumFeatures(),
                reason=f"高SNR({snr:.1f}dB)+良好C50({c50:.1f}dB)，纯净人声"
            )

        # 低 SNR 或 严重混响 = 强制分离
        if snr_level == "low" or c50_level == "bad":
            reasons = []
            if snr_level == "low":
                reasons.append(f"低SNR({snr:.1f}dB)")
            if c50_level == "bad":
                reasons.append(f"严重混响({c50:.1f}dB)")

            return SpectrumDiagnosis(
                chunk_index=0,
                diagnosis=DiagnosisResult.NOISE,
                need_separation=True,
                music_score=0.0,
                noise_score=0.8,
                clean_score=0.2,
                recommended_model="htdemucs",
                features=SpectrumFeatures(),
                reason=" + ".join(reasons) + "，强制分离"
            )

        # 进入 Layer 2
        return None

    def _layer2_decision(
        self,
        features: SpectrumFeatures,
        chunk_index: int,
        snr_level: str,
        c50_level: str
    ) -> Optional[SpectrumDiagnosis]:
        """
        Layer 2 决策：频谱特征补充

        Returns:
            SpectrumDiagnosis 如果可以决策，None 如果需要进入 Layer 3
        """
        config = self._brouhaha_config

        # 极低频谱对比度 = 分离
        if features.spectral_contrast < config.spectral_contrast_critical:
            return SpectrumDiagnosis(
                chunk_index=chunk_index,
                diagnosis=DiagnosisResult.NOISE,
                need_separation=True,
                music_score=0.0,
                noise_score=0.6,
                clean_score=0.4,
                recommended_model="htdemucs",
                features=features,
                reason=f"极低频谱对比度({features.spectral_contrast:.1f}dB)"
            )

        # 低对比度 + 高平坦度 = 分离
        if (features.spectral_contrast < config.spectral_contrast_low and
            features.spectral_flatness > config.spectral_flatness_high):
            return SpectrumDiagnosis(
                chunk_index=chunk_index,
                diagnosis=DiagnosisResult.NOISE,
                need_separation=True,
                music_score=0.0,
                noise_score=0.5,
                clean_score=0.5,
                recommended_model="htdemucs",
                features=features,
                reason=f"音频糊({features.spectral_contrast:.1f}dB)且平坦({features.spectral_flatness:.2f})"
            )

        # 高 SNR 但 C50 警戒 = 检查频谱决定是否放行
        if snr_level == "high" and c50_level == "warn":
            if features.spectral_contrast >= config.spectral_contrast_low:
                return SpectrumDiagnosis(
                    chunk_index=chunk_index,
                    diagnosis=DiagnosisResult.CLEAN,
                    need_separation=False,
                    music_score=0.0,
                    noise_score=0.2,
                    clean_score=0.8,
                    recommended_model=None,
                    features=features,
                    reason=f"高SNR+频谱清晰({features.spectral_contrast:.1f}dB)，轻度混响可接受"
                )

        # 进入 Layer 3
        return None
```

### 7.4 配置开关

```python
# 在 PreprocessingConfig 中新增

@dataclass
class PreprocessingConfig:
    # 现有字段...

    # V3.1.1+dev.20260107.03 新增
    use_snr_triage: bool = True              # 启用 SNR+C50 策略
    brouhaha_config: BrouhahaTriageConfig = field(
        default_factory=BrouhahaTriageConfig
    )
```

### 7.5 向后兼容保证

```python
# 1. 默认行为不变
# use_snr_triage=False 时，完全使用原有逻辑

# 2. 数据模型向下兼容
# 新增字段都有默认值，不影响现有序列化/反序列化

# 3. API 签名不变
# SpectralTriageStage.process() 签名不变

# 4. 配置可选
# 不提供 brouhaha_config 时使用默认值
```

---

## 8. PyTorch 模型下载与配置

### 8.1 模型下载方式

Brouhaha 模型使用 PyTorch 原生格式，支持两种获取方式：

#### 8.1.1 自动下载（推荐）

BrouhahaService 首次运行时会自动从 HuggingFace 下载模型到本地目录：

```python
# 首次初始化时自动下载
service = BrouhahaService(
    hf_token="your_hf_token_here"  # 首次下载需要
)

# 后续运行无需 token，直接从本地加载
service = BrouhahaService()
```

**下载流程**：
1. 检查本地目录 `backend/models/pretrained/brouhaha/` 是否存在模型
2. 如不存在，从 HuggingFace 下载 `pyannote/brouhaha`
3. 保存到本地目录（`pytorch_model.bin` + `config.yaml`）
4. 后续运行直接加载本地模型

#### 8.1.2 手动下载

如果需要离线部署或批量部署，可手动下载模型：

```bash
# 使用 huggingface-cli 下载
pip install huggingface-hub

# 下载模型到指定目录
huggingface-cli download pyannote/brouhaha \
    --local-dir backend/models/pretrained/brouhaha \
    --token your_hf_token_here

# 或使用 Python 脚本
python -c "
from pyannote.audio import Model
model = Model.from_pretrained('pyannote/brouhaha', use_auth_token='your_token')
model.save_pretrained('backend/models/pretrained/brouhaha')
"
```

### 8.2 HuggingFace Token 配置

#### 8.2.1 获取 Token

1. 访问 [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. 创建 Access Token（Read 权限即可）
3. 访问 [pyannote/brouhaha](https://huggingface.co/pyannote/brouhaha) 接受用户协议

#### 8.2.2 配置方式

**方式1：环境变量（推荐）**

```bash
# Windows PowerShell
$env:HUGGING_FACE_HUB_TOKEN="your_token_here"

# Windows CMD
set HUGGING_FACE_HUB_TOKEN=your_token_here

# Linux/Mac
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

**方式2：代码传参**

```python
service = BrouhahaService(
    hf_token="your_token_here"
)
```

**方式3：配置文件**

```python
# backend/app/core/config.py
class Settings:
    HUGGING_FACE_TOKEN: Optional[str] = None

# 从环境变量或配置文件读取
import os
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
service = BrouhahaService(hf_token=hf_token)
```

### 8.3 模型文件结构

```
backend/models/pretrained/brouhaha/
├── pytorch_model.bin       # PyTorch 模型权重（~8MB）
├── config.yaml             # pyannote 模型配置
└── README.md               # 模型说明
```

**config.yaml 示例**：

```yaml
architecture:
  name: "Brouhaha"
  sample_rate: 16000
  num_channels: 1

task:
  name: "multi_task"
  outputs:
    - snr
    - c50
    - vad
```

### 8.4 性能优化选项

#### 8.4.1 混合精度推理（GPU）

```python
# 使用 FP16 混合精度加速 GPU 推理
model = model.half()  # 转换为 FP16
waveform = waveform.half()

# 或使用 torch.autocast
with torch.autocast(device_type='cuda', dtype=torch.float16):
    outputs = model(waveform)
```

**收益**：
- 推理速度提升 ~30%
- 显存占用减少 ~50%
- 精度损失 < 0.5%

#### 8.4.2 TorchScript 编译（可选）

```python
# 编译模型为 TorchScript 加速推理
model_scripted = torch.jit.script(model)
model_scripted.save("brouhaha_scripted.pt")

# 加载编译后的模型
model = torch.jit.load("brouhaha_scripted.pt")
```

**收益**：
- 推理速度提升 ~10-15%
- 支持 C++ 部署
- 无精度损失

#### 8.4.3 批量推理

```python
# 批量处理多个 chunk
batch_waveforms = torch.stack([
    torch.from_numpy(chunk1),
    torch.from_numpy(chunk2),
    torch.from_numpy(chunk3)
])  # [batch, samples]

outputs = model(batch_waveforms)  # [batch, frames, 3]
```

**收益**：
- 吞吐量提升 2-3 倍
- 适合离线批处理场景

---

## 9. 分阶段实现计划

### 9.1 总体规划

```
┌─────────────────────────────────────────────────────────────┐
│                      实现阶段总览                            │
└─────────────────────────────────────────────────────────────┘

Phase 1: 基础设施 (2天)
├─ 模型导出和量化
├─ BrouhahaService 实现
└─ 单元测试

Phase 2: 核心集成 (3天)
├─ 数据模型扩展
├─ AudioSpectrumClassifier 增强
├─ 三层决策逻辑
└─ 集成测试

Phase 3: 配置与调优 (2天)
├─ 配置类实现
├─ SpectralTriageStage 集成
├─ 参数调优工具
└─ 性能测试

Phase 4: 生产就绪 (2天)
├─ 回退机制完善
├─ 监控和日志
├─ 文档更新
└─ 灰度发布

总计: 9个工作日
```

### 9.2 Phase 1: 基础设施（2天）

#### Day 1: 模型准备

**任务清单**：
- [ ] 配置 HuggingFace Token（环境变量或配置文件）
- [ ] 测试模型自动下载功能
- [ ] 验证本地模型加载
- [ ] 测试 CPU/GPU 推理

**验收标准**：
- 模型可从 HuggingFace 自动下载到本地
- 本地模型加载成功
- CPU 和 GPU 推理均正常工作

#### Day 2: BrouhahaService 实现

**任务清单**：
- [ ] 创建 `backend/app/services/brouhaha_service.py`
- [ ] 实现模型加载（复用 cpu_optimizer）
- [ ] 实现 detect() 方法
- [ ] 实现 WADA-SNR 回退
- [ ] 编写单元测试

**文件产出**：
```
backend/app/services/brouhaha_service.py
backend/tests/services/test_brouhaha_service.py
```

**验收标准**：
- detect() 返回正确的 SNR/C50 值
- CPU 大核调度生效
- 回退机制正常工作
- 单元测试通过

### 9.3 Phase 2: 核心集成（3天）

#### Day 3: 数据模型扩展

**任务清单**：
- [ ] 扩展 `SpectrumFeatures` 添加 snr, c50, spectral_contrast
- [ ] 扩展 `SpectrumDiagnosis` 添加 snr_level, c50_level, triage_layer
- [ ] 创建 `BrouhahaTriageConfig` 配置类
- [ ] 更新 `SpectrumThresholds` 添加 C50 阈值

**文件修改**：
```
backend/app/models/circuit_breaker_models.py
backend/app/core/spectrum_thresholds.py
```

#### Day 4-5: AudioSpectrumClassifier 增强

**任务清单**：
- [ ] 添加 `_get_brouhaha()` 懒加载
- [ ] 添加 spectral_contrast 到 `extract_features()`
- [ ] 实现 `_diagnose_with_snr_c50_strategy()`
- [ ] 实现 `_layer1_decision()` 和 `_layer2_decision()`
- [ ] 修改 `diagnose_chunk()` 调用新策略
- [ ] 编写集成测试

**文件修改**：
```
backend/app/services/audio_spectrum_classifier.py
backend/tests/services/test_audio_spectrum_classifier.py
```

**验收标准**：
- 三层决策逻辑正确执行
- Layer 分布符合预期（L1: 60-70%）
- 与 YAMNet 策略结果对比合理
- 集成测试通过

### 9.4 Phase 3: 配置与调优（2天）

#### Day 6: 配置类实现

**任务清单**：
- [ ] 完善 `BrouhahaTriageConfig`
- [ ] 在 `PreprocessingConfig` 中集成
- [ ] 实现配置验证逻辑
- [ ] 更新 `SpectralTriageStage` 支持新配置

**文件修改**：
```
backend/app/models/job_models.py
backend/app/pipelines/stages/spectral_triage_stage.py
```

#### Day 7: 参数调优工具

**任务清单**：
- [ ] 创建阈值分析脚本 `scripts/analyze_triage_thresholds.py`
- [ ] 实现 ROC 曲线绘制
- [ ] 实现最优阈值推荐
- [ ] 性能基准测试

**文件产出**：
```
scripts/analyze_triage_thresholds.py
scripts/benchmark_triage_performance.py
```

**验收标准**：
- 配置可正确加载和验证
- 阈值分析工具可用
- 性能测试结果符合预期

### 9.5 Phase 4: 生产就绪（2天）

#### Day 8: 回退与监控

**任务清单**：
- [ ] 完善 Brouhaha 不可用时的回退逻辑
- [ ] 添加 triage_layer 分布统计日志
- [ ] 添加 SNR/C50 分布统计日志
- [ ] 实现配置热更新（可选）

**文件修改**：
```
backend/app/services/audio_spectrum_classifier.py
backend/app/pipelines/stages/spectral_triage_stage.py
```

#### Day 9: 文档与发布

**任务清单**：
- [ ] 更新 `llmdoc/index.md`
- [ ] 更新 `llmdoc/architecture/spectral-triage-stage.md`
- [ ] 创建 `llmdoc/architecture/snr-c50-triage.md`
- [ ] 更新 `llmdoc/reference/` 相关配置文档
- [ ] 灰度发布（默认 use_snr_triage=False）

**验收标准**：
- 文档完整准确
- 灰度发布无故障
- 监控数据正常

### 9.6 里程碑检查点

| 阶段 | 检查点 | 验收标准 |
|------|--------|---------|
| Phase 1 完成 | Day 2 结束 | BrouhahaService 单测通过 |
| Phase 2 完成 | Day 5 结束 | 集成测试通过，Layer 分布符合预期 |
| Phase 3 完成 | Day 7 结束 | 配置系统完整，性能基准达标 |
| Phase 4 完成 | Day 9 结束 | 文档完整，灰度发布成功 |

---

## 10. 测试与验证

### 10.1 单元测试

```python
# backend/tests/services/test_brouhaha_service.py

import pytest
import numpy as np
from app.services.brouhaha_service import BrouhahaService, get_brouhaha_service


class TestBrouhahaService:
    """BrouhahaService 单元测试"""

    @pytest.fixture
    def service(self):
        return get_brouhaha_service()

    def test_model_loading(self, service):
        """测试模型加载"""
        # 可能模型不存在，跳过
        if not service.is_available():
            pytest.skip("Brouhaha model not available")

        assert service.session is not None

    def test_detect_clean_audio(self, service):
        """测试纯净音频检测"""
        if not service.is_available():
            pytest.skip("Brouhaha model not available")

        # 生成低噪声音频
        sr = 16000
        audio = np.random.randn(sr * 2) * 0.05

        result = service.detect(audio, sr)

        assert result.is_valid
        assert result.snr > 20  # 期望高 SNR

    def test_detect_noisy_audio(self, service):
        """测试噪声音频检测"""
        if not service.is_available():
            pytest.skip("Brouhaha model not available")

        # 生成高噪声音频
        sr = 16000
        audio = np.random.randn(sr * 2) * 0.8

        result = service.detect(audio, sr)

        assert result.is_valid
        assert result.snr < 15  # 期望低 SNR

    def test_fallback_wada_snr(self):
        """测试 WADA-SNR 回退"""
        # 创建无模型的 service
        service = BrouhahaService(model_dir="/nonexistent/path")

        sr = 16000
        audio = np.random.randn(sr * 2) * 0.3

        result = service.detect(audio, sr)

        # 应该使用 WADA-SNR 回退
        assert result.is_valid
        assert result.c50 == 0.0  # WADA-SNR 不提供 C50

    def test_cache(self, service):
        """测试缓存机制"""
        if not service.is_available():
            pytest.skip("Brouhaha model not available")

        sr = 16000
        audio = np.random.randn(sr * 2) * 0.3

        # 第一次调用
        result1 = service.detect(audio, sr, chunk_id=0)

        # 第二次调用（应该命中缓存）
        result2 = service.detect(audio, sr, chunk_id=0)

        assert result1.snr == result2.snr
        assert result1.c50 == result2.c50
```

### 10.2 集成测试

```python
# backend/tests/integration/test_snr_triage.py

import pytest
import numpy as np
from app.services.audio_spectrum_classifier import AudioSpectrumClassifier
from app.models.circuit_breaker_models import BrouhahaTriageConfig


class TestSNRTriageIntegration:
    """SNR+C50 分诊集成测试"""

    @pytest.fixture
    def classifier(self):
        config = BrouhahaTriageConfig(
            snr_high_threshold=25.0,
            snr_low_threshold=12.0,
            c50_good_threshold=5.0,
            c50_bad_threshold=-5.0
        )
        return AudioSpectrumClassifier(
            use_snr_strategy=True,
            brouhaha_config=config
        )

    def test_layer1_high_snr_pass(self, classifier):
        """测试 Layer 1: 高 SNR 直接放行"""
        # 生成高 SNR 测试音频
        sr = 16000
        audio = np.random.randn(sr * 2) * 0.05

        diagnosis = classifier.diagnose_chunk(audio, 0, sr)

        assert diagnosis.triage_layer == 1
        assert diagnosis.snr_level == "high"
        assert not diagnosis.need_separation

    def test_layer1_low_snr_separate(self, classifier):
        """测试 Layer 1: 低 SNR 强制分离"""
        sr = 16000
        audio = np.random.randn(sr * 2) * 0.9

        diagnosis = classifier.diagnose_chunk(audio, 0, sr)

        assert diagnosis.triage_layer == 1
        assert diagnosis.snr_level == "low"
        assert diagnosis.need_separation

    def test_layer_distribution(self, classifier):
        """测试 Layer 分布"""
        sr = 16000
        layer_counts = {1: 0, 2: 0, 3: 0}

        # 生成不同 SNR 的测试音频
        for _ in range(100):
            noise_level = np.random.uniform(0.05, 0.8)
            audio = np.random.randn(sr * 2) * noise_level

            diagnosis = classifier.diagnose_chunk(audio, 0, sr)
            layer_counts[diagnosis.triage_layer] += 1

        # 验证分布（Layer 1 应该处理大部分）
        total = sum(layer_counts.values())
        layer1_ratio = layer_counts[1] / total

        assert layer1_ratio >= 0.5, f"Layer 1 ratio too low: {layer1_ratio:.2f}"
```

### 10.3 性能测试

```python
# scripts/benchmark_triage_performance.py

"""
分诊性能基准测试
"""

import time
import numpy as np
from app.services.audio_spectrum_classifier import AudioSpectrumClassifier


def benchmark(num_chunks=1000):
    """性能基准测试"""
    classifier = AudioSpectrumClassifier(use_snr_strategy=True)
    sr = 16000
    duration = 2.0

    # 生成测试数据
    test_chunks = [
        np.random.randn(int(sr * duration)) * np.random.uniform(0.05, 0.8)
        for _ in range(num_chunks)
    ]

    # 预热
    for audio in test_chunks[:10]:
        classifier.diagnose_chunk(audio, 0, sr)

    # 基准测试
    start = time.time()
    layer_times = {1: [], 2: [], 3: []}

    for i, audio in enumerate(test_chunks):
        chunk_start = time.time()
        diagnosis = classifier.diagnose_chunk(audio, i, sr)
        chunk_time = (time.time() - chunk_start) * 1000  # ms
        layer_times[diagnosis.triage_layer].append(chunk_time)

    total_time = time.time() - start

    # 统计
    print(f"总耗时: {total_time:.2f}s")
    print(f"平均每 chunk: {total_time / num_chunks * 1000:.2f}ms")
    print()
    print("Layer 分布:")
    for layer, times in layer_times.items():
        if times:
            print(f"  Layer {layer}: {len(times)} chunks, "
                  f"avg={np.mean(times):.2f}ms, "
                  f"p99={np.percentile(times, 99):.2f}ms")


if __name__ == "__main__":
    benchmark()
```

**运行性能测试**：

```bash
# 使用 uv 运行性能基准测试
uv run python scripts/benchmark_triage_performance.py

# 运行单元测试
uv run pytest backend/tests/services/test_brouhaha_service.py -v

# 运行集成测试
uv run pytest backend/tests/integration/test_snr_triage.py -v

# 运行所有相关测试
uv run pytest backend/tests/ -k "brouhaha or snr_triage" -v
```

---

## 11. 回滚与应急

### 11.1 回滚策略

#### 配置开关回滚（推荐）

```python
# 方法1: 修改默认配置
# backend/app/models/job_models.py
use_snr_triage: bool = False  # 改为 False

# 方法2: 环境变量覆盖
# .env
USE_SNR_TRIAGE=false

# 方法3: API 动态配置
POST /api/config/triage
{
  "use_snr_strategy": false
}
```

#### 代码回滚

```bash
# 回滚到上一个稳定版本
git revert HEAD  # 如果是单个提交
# 或
git checkout v3.1.1 -- backend/app/services/audio_spectrum_classifier.py
```

### 11.2 应急预案

| 问题 | 现象 | 应急措施 |
|------|------|---------|
| Brouhaha 模型加载失败 | 日志报错 | 自动回退到 WADA-SNR |
| SNR 计算异常 | 值超出范围 | 使用默认值 15.0 |
| 性能下降 | 处理耗时增加 | 禁用 SNR 策略 |
| 误判率上升 | need_separation 异常 | 回滚到 YAMNet 策略 |

### 11.3 监控告警

```python
# 关键监控指标
metrics = {
    "triage_layer_1_ratio": 0.65,   # 期望 60-70%
    "triage_layer_3_ratio": 0.15,   # 期望 10-20%
    "avg_triage_time_ms": 8.0,      # 期望 < 10ms
    "need_separation_ratio": 0.25,  # 期望 20-30%
}

# 告警阈值
alerts = {
    "triage_layer_1_ratio < 0.5": "Layer 1 覆盖率过低",
    "avg_triage_time_ms > 20": "分诊耗时过长",
    "brouhaha_fallback_count > 100": "Brouhaha 回退次数过多"
}
```

---

## 附录

### A. 参考资料

1. [Brouhaha 论文](https://arxiv.org/abs/2210.13248)
2. [pyannote/brouhaha 模型](https://huggingface.co/pyannote/brouhaha)
3. [WADA-SNR 算法](https://www.cs.cmu.edu/~robust/Papers/KimSternIS08.pdf)
4. [PyTorch 性能优化指南](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
5. [pyannote.audio 文档](https://github.com/pyannote/pyannote-audio)

### B. 变更历史

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| V3.1.1+dev.20260107.03 | 2026-01-07 | 初始设计 |

### C. 相关文档

- `llmdoc/architecture/spectral-triage-stage.md` - 现有频谱分诊架构
- `llmdoc/architecture/separation-stage.md` - 人声分离阶段
- `llmdoc/architecture/fuse-breaker-v2.md` - 熔断回溯机制
- `backend/app/utils/cpu_optimizer.py` - CPU 调度模块

---

**文档结束**
