# DNSMOS 替代 Brouhaha 频谱分诊方案

> **Type**: Architecture Replacement Plan
> **Status**: Draft
> **Version**: V3.1.2+dev.20260212.01
> **Author**: wgh
> **Date**: 2026-02-12

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [DNSMOS 官方使用说明](#2-dnsmos-官方使用说明)
3. [当前 Brouhaha 系统全景](#3-当前-brouhaha-系统全景)
4. [替代方案设计](#4-替代方案设计)
5. [接入方案（逐文件改造清单）](#5-接入方案逐文件改造清单)
6. [阈值校准与自动调参](#6-阈值校准与自动调参)
7. [风险与回退策略](#7-风险与回退策略)
8. [验收标准](#8-验收标准)

---

## 1. 背景与动机

### 1.1 Brouhaha 当前不可用

经实测确认，Brouhaha 模型链路已完全失效：

| 问题 | 详情 |
|------|------|
| 模型文件是 Git LFS 指针 | `pytorch_model.bin` 仅 136 bytes，实际应为 ~47MB |
| pyannote 与 torchaudio 不兼容 | `module 'torchaudio' has no attribute 'AudioMetaData'` |
| 系统实际运行 WADA-SNR 回退 | 回退算法的 SNR 标度与 Brouhaha 阈值不匹配，决策几乎随机 |

WADA-SNR 回退的灾难性效果：

| 场景 | WADA-SNR 估计 | 真实情况 | 套用 Brouhaha 阈值的决策 |
|------|--------------|---------|------------------------|
| 纯净语音 | 5.2 dB | 高质量 | 强制分离（错误） |
| 重噪声 | 30.0 dB | 低质量 | 放行（错误） |

**结论**：当前 Layer 1（SNR+C50 策略）实质处于失效状态，系统完全依赖 YAMNet/规则兜底。

### 1.2 为什么选择 DNSMOS

| 维度 | Brouhaha（当前） | DNSMOS（替代） |
|------|-----------------|---------------|
| 评估视角 | 物理测量（SNR dB, C50 dB） | 主观感知（MOS 1-5 分） |
| 评分空间 | 2D: SNR x C50 | 3D: SIG x BAK x OVRL + P808 |
| 模型格式 | PyTorch（pyannote 生态） | ONNX（独立推理） |
| 模型大小 | ~47 MB | ~1.35 MB（两个模型合计） |
| 推理设备 | GPU 优先（~200MB 显存） | CPU 即可（~44ms/chunk） |
| 依赖 | pyannote.audio + brouhaha-vad + PyTorch | onnxruntime + librosa（项目已有） |
| 维护风险 | pyannote 生态不稳定 | ONNX 无版本依赖问题 |
| 当前状态 | **不可用** | 可用，模型已在本地 |

---

## 2. DNSMOS 官方使用说明

### 2.1 项目来源

DNSMOS（Deep Noise Suppression Mean Opinion Score）是 Microsoft 在 ICASSP 2023 DNS Challenge 5 中发布的**非侵入式语音质量评估模型**。基于 ITU-T P.808/P.835 标准，不需要干净参考音频，直接对含噪语音进行质量打分。

- 论文: [DNSMOS](https://arxiv.org/pdf/2010.15258.pdf)、[DNSMOS P.835](https://arxiv.org/pdf/2110.01763.pdf)
- 仓库: `tmp/DNS-Challenge-master/DNSMOS/`
- 许可: MIT License

### 2.2 模型文件清单

```
tmp/DNS-Challenge-master/DNSMOS/
├── DNSMOS/
│   ├── sig_bak_ovr.onnx   (1130.8 KB)  <- 主模型：输出 SIG/BAK/OVRL
│   ├── model_v8.onnx       (219.6 KB)  <- P808 模型：输出 P808 MOS
│   ├── bak_ovr.onnx         (725.0 KB)  <- 拆分模型（本项目不使用）
│   └── sig.onnx             (724.8 KB)  <- 拆分模型（本项目不使用）
├── pDNSMOS/
│   └── sig_bak_ovr.onnx   (1130.8 KB)  <- 个性化版本（检测干扰人声用）
└── dnsmos_local.py                      <- 官方推理脚本
```

本项目只使用 `DNSMOS/sig_bak_ovr.onnx`（主模型）和 `DNSMOS/model_v8.onnx`（P808 模型），合计 ~1.35 MB。

### 2.3 三维评分体系

模型输出三个独立维度，MOS 分范围约 1.0 ~ 5.0：

| 维度 | 全称 | 物理意义 | 高分含义 | 低分含义 |
|------|------|----------|---------|---------|
| **SIG** | Signal Quality | 语音信号本身的质量 | 语音清晰完整 | 语音被破坏/失真 |
| **BAK** | Background Noise | 背景噪声的安静程度 | 背景安静 | 背景嘈杂 |
| **OVRL** | Overall Quality | 综合主观质量 | 整体听感好 | 整体听感差 |

P808 模型额外输出一个 **P808 MOS** 单维综合分。

各场景下的评分表现（合成音频测试）：

| 场景 | SIG | BAK | OVRL | 预期分诊决策 |
|------|-----|-----|------|-------------|
| 纯净语音 | 3.28 | 3.12 | 2.66 | PASS（放行） |
| 轻 BGM (x0.3) | 3.41 | 3.38 | 2.83 | PASS（ASR 可处理） |
| 中 BGM (x0.7) | 2.69 | 3.04 | 2.23 | 灰色地带 -> YAMNet |
| 重白噪声 (x0.5) | 1.27 | 1.09 | 1.05 | SEPARATE（分离） |
| 纯噪声 | 1.05 | 1.05 | 0.97 | SEPARATE（分离） |

关键发现：
- **OVRL 是最佳单一判别指标**：动态范围最大，对各种干扰都有响应
- **BAK 对 BGM 不敏感**：BGM 不被视为传统"噪声"，这在 ASR 场景下反而正确（轻 BGM 不影响 ASR）
- **SIG 对混响敏感**：混响和干扰人声都会拉低 SIG

### 2.4 推理流程

```
音频输入 (np.ndarray, sr)
    |
    v
重采样到 16kHz
    |
    v
填充/截取到 9.01s (144160 samples)
    |                              |
    v                              v
主模型推理                      Mel 频谱图提取
sig_bak_ovr.onnx               n_mels=120, hop=160
input: [1, 144160] float32     input: [1, 900, 120] float32
    |                              |
    v                              v
output: [SIG_raw, BAK_raw,     output: P808_MOS
         OVRL_raw]
    |
    v
多项式校准 (poly1d)
    |
    v
最终输出: {SIG, BAK, OVRL, P808}
```

**输入要求**：
- 采样率：16000 Hz
- 长度：固定 9.01 秒（144160 采样点）
- 短于 9.01 秒：循环自我复制填充
- 长于 9.01 秒：1 秒步长滑窗，取所有窗口均值

**多项式校准**（将原始输出校准到 ITU-T P.808 MOS 标度）：

```python
# 标准 DNSMOS 校准
p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439])
p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])
```

### 2.5 依赖

| 依赖 | 版本 | 项目是否已有 |
|------|------|------------|
| onnxruntime | >=1.13.1 | 已有（SenseVoice/YAMNet 使用） |
| librosa | >=0.8.1 | 已有（频谱特征提取使用） |
| numpy | >=1.22.4 | 已有 |

**零新增依赖**。

### 2.6 性能指标

| 指标 | 数值 |
|------|------|
| 单 chunk 推理（CPU） | ~44 ms |
| 模型加载时间 | ~200 ms |
| 内存占用 | ~50 MB |
| 批量推理 | 支持（输入 shape `[N, 144160]`） |

---

## 3. 当前 Brouhaha 系统全景

### 3.1 涉及的全部文件

#### 核心实现（需要替换/重写）

| 文件 | 当前职责 | 替换动作 |
|------|---------|---------|
| `backend/app/services/brouhaha_service.py` | Brouhaha PyTorch 推理 + WADA-SNR 回退 | **重写为 `dnsmos_service.py`** |
| `backend/app/services/pyannote_compat.py` | pyannote 3.x/4.x 兼容层 | **删除**（DNSMOS 不依赖 pyannote） |
| `backend/app/services/audio_spectrum_classifier.py` | 频谱分诊主入口，三层决策 | **修改**：替换 Layer 1 决策逻辑 |
| `backend/app/core/spectrum_thresholds.py` | SNR/C50 阈值配置 | **修改**：替换为 SIG/BAK/OVRL 阈值 |

#### 数据模型（需要修改）

| 文件 | 当前结构 | 修改内容 |
|------|---------|---------|
| `backend/app/models/circuit_breaker_models.py` | `SpectrumFeatures`（含 snr/c50）、`SpectrumDiagnosis`（含 snr_level/c50_level/triage_layer） | 替换 snr/c50 字段为 sig/bak/ovrl |
| `backend/app/services/brouhaha_service.py` | `BrouhahaResult`（snr/c50/vad）、`BrouhahaTriageConfig` | 替换为 `DNSMOSResult`（sig/bak/ovrl/p808） |

#### 管道集成（需要适配）

| 文件 | 当前用法 | 修改内容 |
|------|---------|---------|
| `backend/app/pipelines/stages/spectral_triage_stage.py` | 调用 classifier，保存 triage_log（含 snr/c50） | 修改日志字段名 |
| `backend/app/pipelines/stages/separation_stage.py` | 读取 `chunk.needs_separation` | **不变**（分诊接口不变） |
| `backend/app/pipelines/preprocessing_pipeline.py` | 传递 `use_snr_triage` 配置 | 修改配置名为 `use_dnsmos_triage` |

#### 配置文件

| 文件 | 修改内容 |
|------|---------|
| `backend/app/config/models.yaml` | 添加 DNSMOS 模型注册 |
| `model_runtime_config.json` | 添加 DNSMOS 阈值参数 |
| `backend/app/models/job_models.py` | `use_snr_triage` -> `use_dnsmos_triage` |

#### 模型文件

| 操作 | 路径 |
|------|------|
| 新增 | `backend/models/pretrained/dnsmos/sig_bak_ovr.onnx` (~1.1MB) |
| 新增 | `backend/models/pretrained/dnsmos/model_v8.onnx` (~220KB) |
| 保留（可选） | `backend/models/pretrained/dnsmos/p_sig_bak_ovr.onnx` (~1.1MB, pDNSMOS) |
| 删除 | `backend/models/pretrained/brouhaha/` 整个目录 |

#### 测试文件

| 文件 | 动作 |
|------|------|
| `backend/tests/test_brouhaha_service.py` | 重写为 `test_dnsmos_service.py` |
| `backend/tests/test_snr_c50_triage_integration.py` | 重写为 `test_dnsmos_triage_integration.py` |

#### 可删除的文件（Brouhaha 专属依赖）

| 文件 | 原因 |
|------|------|
| `backend/app/services/pyannote_compat.py` | 仅 Brouhaha 使用 |
| `backend/app/services/brouhaha_service.py` | 整体替换 |
| `backend/models/pretrained/brouhaha/` | 模型文件 |

### 3.2 当前三层决策流程（将被简化）

```
当前 Brouhaha 三层:
  Layer 1: SNR+C50 快速筛选 (Brouhaha PyTorch) -> 60-70% 直接决策
  Layer 2: 频谱特征补充 (librosa)              -> 10-20%
  Layer 3: YAMNet 语义分类 (ONNX)              -> 10-20% 兜底
```

### 3.3 下游消费方式（不变）

分诊结果通过 `AudioChunk` 的以下字段传递给下游，**这些字段名和语义不变**：

```python
chunk.needs_separation: bool          # 是否需要分离
chunk.recommended_model: Optional[str] # "htdemucs" / None
chunk.spectrum_diagnosis: SpectrumDiagnosis  # 完整诊断信息
```

`SeparationStage` 仅读取 `needs_separation` 和 `recommended_model`，**完全不感知上游检测器是 Brouhaha 还是 DNSMOS**。

---

## 4. 替代方案设计

### 4.1 新架构：两层决策

```
新 DNSMOS 两层:
  Layer 0: DNSMOS SIG+BAK+OVRL 快速筛选 (ONNX CPU) -> 70-80% 直接决策
  Layer 1: YAMNet 语义分类 (ONNX CPU)               -> 20-30% 兜底
```

**为什么从三层简化为两层？**

DNSMOS 的三维评分空间（SIG x BAK x OVRL）比 Brouhaha 的二维（SNR x C50）信息量更大，可以直接覆盖原 Layer 1 + Layer 2 的功能：
- SIG 低 -> 涵盖了原 Layer 2 的"频谱对比度低"场景
- BAK 低 -> 涵盖了原 Layer 2 的"频谱平坦度高"场景
- 原 Layer 2 的 librosa 特征提取变为多余

### 4.2 决策矩阵

```
Layer 0: DNSMOS 快速筛选
├── PASS（放行）:     OVRL >= 2.5 AND BAK >= 2.8 AND SIG >= 3.0
├── SEPARATE（分离）: OVRL < 1.3 OR SIG < 1.5
├── BGM 嫌疑:         BAK >= 2.5 AND SIG < 2.8 -> 交给 Layer 1 (YAMNet)
└── 灰色地带:          其他情况 -> 交给 Layer 1 (YAMNet)

Layer 1: YAMNet 语义分类
├── 检测到 Music/BGM -> SEPARATE
└── Speech 高置信度 -> PASS
```

**阈值说明（初始值，需要通过自动调参校准）**：

| 参数 | 初始值 | 搜索范围 | 物理意义 |
|------|--------|---------|---------|
| `ovrl_pass_threshold` | 2.5 | 2.0 ~ 3.5 | OVRL 放行下限 |
| `ovrl_separate_threshold` | 1.3 | 1.0 ~ 2.0 | OVRL 分离上限 |
| `sig_pass_threshold` | 3.0 | 2.5 ~ 4.0 | SIG 放行下限 |
| `sig_separate_threshold` | 1.5 | 1.0 ~ 2.5 | SIG 分离上限 |
| `bak_pass_threshold` | 2.8 | 2.0 ~ 3.5 | BAK 放行下限 |
| `bak_bgm_suspect_threshold` | 2.5 | 2.0 ~ 3.0 | BAK 高 + SIG 低 = BGM 嫌疑 |

### 4.3 核心数据结构

```python
@dataclass
class DNSMOSResult:
    """DNSMOS 检测结果"""
    sig: float          # 语音质量 (1-5)
    bak: float          # 背景安静度 (1-5)
    ovrl: float         # 综合质量 (1-5)
    p808: float         # P.808 MOS 分 (1-5)
    is_valid: bool      # 结果是否有效
```

对应替换 `BrouhahaResult`（snr/c50/vad）。

### 4.4 新的 SpectrumDiagnosis 字段

```python
@dataclass
class SpectrumDiagnosis:
    # ... 保留不变的字段 ...
    chunk_index: int
    diagnosis: DiagnosisResult
    need_separation: bool
    music_score: float
    noise_score: float
    clean_score: float
    recommended_model: Optional[str]
    features: SpectrumFeatures
    reason: str

    # --- 替换后的字段 ---
    # 旧: snr, c50, snr_level, c50_level, triage_layer
    # 新:
    sig: float = 0.0            # DNSMOS 语音质量 (1-5)
    bak: float = 0.0            # DNSMOS 背景安静度 (1-5)
    ovrl: float = 0.0           # DNSMOS 综合质量 (1-5)
    quality_level: str = ""     # "good" / "warn" / "bad"
    triage_layer: int = 0       # 0=DNSMOS, 1=YAMNet
```

### 4.5 新的 SpectrumFeatures 字段

```python
@dataclass
class SpectrumFeatures:
    # ... 保留原有 librosa 特征字段 ...

    # --- 替换后的字段 ---
    # 旧: snr: float, c50: float
    # 新:
    sig: float = 0.0            # DNSMOS SIG
    bak: float = 0.0            # DNSMOS BAK
    ovrl: float = 0.0           # DNSMOS OVRL
```

### 4.6 新的 SpectrumThresholds 字段

```python
@dataclass
class SpectrumThresholds:
    # ... 保留 YAMNet/规则方法的阈值 ...

    # --- 替换后的字段 ---
    # 旧: snr_high_threshold, snr_low_threshold, c50_good_threshold, c50_bad_threshold,
    #     spectral_contrast_low, spectral_contrast_critical, spectral_flatness_high
    # 新:
    ovrl_pass_threshold: float = 2.5
    ovrl_separate_threshold: float = 1.3
    sig_pass_threshold: float = 3.0
    sig_separate_threshold: float = 1.5
    bak_pass_threshold: float = 2.8
    bak_bgm_suspect_threshold: float = 2.5
```

---

## 5. 接入方案（逐文件改造清单）

### 阶段 1: 新建 DNSMOS 服务

#### 5.1 新建 `backend/app/services/dnsmos_service.py`

替代 `brouhaha_service.py`，核心实现：

```python
class DNSMOSService:
    """
    DNSMOS 语音质量评估服务

    基于 Microsoft DNSMOS ONNX 模型，提供 SIG/BAK/OVRL/P808 评分。
    替代原 BrouhahaService 的 SNR/C50 检测。
    """
    SAMPLE_RATE = 16000
    INPUT_LENGTH = 9.01
    LEN_SAMPLES = int(INPUT_LENGTH * SAMPLE_RATE)  # 144160

    def __init__(self, model_dir: Optional[str] = None):
        # 模型路径: backend/models/pretrained/dnsmos/
        # 加载两个 ONNX 模型: sig_bak_ovr.onnx + model_v8.onnx
        # 初始化多项式校准系数

    def detect(self, audio: np.ndarray, sr: int = 16000,
               chunk_id: Optional[int] = None) -> DNSMOSResult:
        # 1. 重采样到 16kHz
        # 2. 填充/截取到 144160 samples
        # 3. 主模型推理 -> SIG_raw, BAK_raw, OVRL_raw
        # 4. P808 模型推理 -> P808_MOS
        # 5. 多项式校准
        # 6. 返回 DNSMOSResult

    def is_available(self) -> bool:
        return self._primary_sess is not None

    def unload(self):
        # 释放 ONNX session
```

**与 BrouhahaService 的接口对照**：

| BrouhahaService 方法 | DNSMOSService 方法 | 变化 |
|---------------------|-------------------|------|
| `detect(audio, sr, chunk_id) -> BrouhahaResult` | `detect(audio, sr, chunk_id) -> DNSMOSResult` | 返回类型变化 |
| `is_available() -> bool` | `is_available() -> bool` | 不变 |
| `unload()` | `unload()` | 不变 |
| `clear_cache()` | `clear_cache()` | 不变 |
| `get_device() -> str` | 删除 | ONNX 无需指定设备 |

#### 5.2 复制模型文件

```
源: tmp/DNS-Challenge-master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
    tmp/DNS-Challenge-master/DNSMOS/DNSMOS/model_v8.onnx
目标: backend/models/pretrained/dnsmos/sig_bak_ovr.onnx
      backend/models/pretrained/dnsmos/model_v8.onnx
```

### 阶段 2: 修改数据模型

#### 5.3 修改 `backend/app/models/circuit_breaker_models.py`

```diff
 @dataclass
 class SpectrumFeatures:
-    snr: float = 0.0
-    c50: float = 0.0
+    sig: float = 0.0            # DNSMOS SIG (1-5)
+    bak: float = 0.0            # DNSMOS BAK (1-5)
+    ovrl: float = 0.0           # DNSMOS OVRL (1-5)

 @dataclass
 class SpectrumDiagnosis:
-    snr: float = 0.0
-    c50: float = 0.0
-    snr_level: str = ""
-    c50_level: str = ""
-    triage_layer: int = 0
+    sig: float = 0.0
+    bak: float = 0.0
+    ovrl: float = 0.0
+    quality_level: str = ""     # "good" / "warn" / "bad"
+    triage_layer: int = 0       # 0=DNSMOS, 1=YAMNet
```

#### 5.4 修改 `backend/app/core/spectrum_thresholds.py`

```diff
 @dataclass
 class SpectrumThresholds:
-    snr_high_threshold: float = 40.0
-    snr_low_threshold: float = 25.0
-    c50_good_threshold: float = 13.70
-    c50_bad_threshold: float = -12.33
-    spectral_contrast_low: float = 19.84
-    spectral_contrast_critical: float = 13.59
-    spectral_flatness_high: float = 0.29
+    # DNSMOS 分诊阈值
+    ovrl_pass_threshold: float = 2.5
+    ovrl_separate_threshold: float = 1.3
+    sig_pass_threshold: float = 3.0
+    sig_separate_threshold: float = 1.5
+    bak_pass_threshold: float = 2.8
+    bak_bgm_suspect_threshold: float = 2.5
```

### 阶段 3: 修改分诊核心逻辑

#### 5.5 修改 `backend/app/services/audio_spectrum_classifier.py`

**核心改动**：

1. `_get_brouhaha()` -> `_get_dnsmos()`：懒加载 DNSMOSService
2. `_diagnose_with_snr_c50_strategy()` -> `_diagnose_with_dnsmos_strategy()`：重写决策逻辑
3. 删除 `_layer1_decision()`、`_layer2_decision()`（三层合并为两层）
4. 新增 `_dnsmos_layer0_decision()`：DNSMOS 快速筛选
5. `_build_snr_diagnosis()` -> `_build_dnsmos_diagnosis()`：替换字段名
6. `_classify_snr_level()` / `_classify_c50_level()` -> `_classify_quality_level()`
7. `_is_snr_strategy_runtime_available()` -> `_is_dnsmos_runtime_available()`：检查 onnxruntime

新的决策逻辑骨架：

```python
def _diagnose_with_dnsmos_strategy(self, audio, chunk_index, sr, dnsmos):
    # 获取 DNSMOS 评分
    result = dnsmos.detect(audio, sr, chunk_id=chunk_index)
    sig, bak, ovrl = result.sig, result.bak, result.ovrl
    quality_level = self._classify_quality_level(ovrl, sig, bak)

    # Layer 0: DNSMOS 快速筛选
    layer0 = self._dnsmos_layer0_decision(sig, bak, ovrl, quality_level, chunk_index)
    if layer0 is not None:
        need_separation, reason = layer0
        return self._build_dnsmos_diagnosis(
            chunk_index, sig, bak, ovrl, quality_level,
            triage_layer=0, need_separation=need_separation, reason=reason
        )

    # Layer 1: YAMNet 语义分类兜底
    yamnet = self._get_yamnet()
    # ... 与现有 Layer 3 逻辑相同 ...
```

#### 5.6 修改 `backend/app/pipelines/stages/spectral_triage_stage.py`

改动较小：
1. `use_snr_triage` -> `use_dnsmos_triage`（参数名）
2. triage_log 中 `snr`/`c50`/`snr_level`/`c50_level` -> `sig`/`bak`/`ovrl`/`quality_level`
3. `_save_triage_log()` 中的统计字段同步更新

#### 5.7 修改 `backend/app/models/job_models.py`

```diff
 @dataclass
 class PreprocessingConfig:
-    use_snr_triage: bool = False
+    use_dnsmos_triage: bool = True  # 默认启用 DNSMOS 分诊
```

### 阶段 4: 配置与注册

#### 5.8 修改 `backend/app/config/models.yaml`

添加 DNSMOS 模型注册：

```yaml
dnsmos-quality:
  type: onnx
  path: models/pretrained/dnsmos
  files:
    - sig_bak_ovr.onnx
    - model_v8.onnx
  description: "DNSMOS 语音质量评估（SIG/BAK/OVRL）"
```

#### 5.9 修改 `model_runtime_config.json`

添加 DNSMOS 阈值配置节：

```json
{
  "dnsmos_triage": {
    "ovrl_pass_threshold": 2.5,
    "ovrl_separate_threshold": 1.3,
    "sig_pass_threshold": 3.0,
    "sig_separate_threshold": 1.5,
    "bak_pass_threshold": 2.8,
    "bak_bgm_suspect_threshold": 2.5
  }
}
```

### 阶段 5: 清理

#### 5.10 删除 Brouhaha 专属文件

| 操作 | 文件 |
|------|------|
| 删除 | `backend/app/services/brouhaha_service.py` |
| 删除 | `backend/app/services/pyannote_compat.py` |
| 删除 | `backend/models/pretrained/brouhaha/` |
| 删除 | `backend/tests/test_brouhaha_service.py` |
| 删除 | `backend/tests/test_snr_c50_triage_integration.py` |

#### 5.11 清理依赖

从 `pyproject.toml` 中移除：

```diff
- "pyannote.audio>=3.3.0,<4.0.0"
- "brouhaha @ https://github.com/marianne-m/brouhaha-vad/archive/main.zip"
```

#### 5.12 更新 llmdoc 文档

| 文档 | 操作 |
|------|------|
| `llmdoc/architecture/brouhaha-snr-c50-triage.md` | **重写为** `dnsmos-triage.md` |
| `llmdoc/architecture/spectral-triage-stage.md` | 更新 Layer 描述 |

### 阶段 6: 测试

#### 5.13 新建测试

| 文件 | 覆盖范围 |
|------|---------|
| `backend/tests/test_dnsmos_service.py` | 模型加载、单 chunk 推理、填充逻辑、缓存、降级 |
| `backend/tests/test_dnsmos_triage_integration.py` | 两层决策覆盖、边界值、YAMNet 兜底 |

---

## 6. 阈值校准与自动调参

### 6.1 初始阈值的来源

第 4.2 节中的阈值是基于合成音频测试的初始估计值。正式上线前**必须**通过自动调参框架校准。

### 6.2 自动调参框架适配

现有 Optuna 调参框架（`F:\brouhaha-auto-tuner-project\`）约 70% 可直接复用：

| 模块 | 复用性 | 说明 |
|------|--------|------|
| TTS 生成器 (`data/tts_generator.py`) | 100% 复用 | Edge-TTS 生成中英日三语纯净人声 |
| 噪声加载器 (`data/noise_streamer.py`) | 100% 复用 | HuggingFace 流式加载 MUSAN 噪声 |
| 音频混合器 (`data/audio_mixer.py`) | 100% 复用 | 按精确 SNR 混合人声与噪声 |
| MUSAN 数据集 | 100% 复用 | 2016 个 wav 文件 |
| Optuna 引擎 | 90% 复用 | Study 创建 + 优化循环 |
| 检测器 | **重写** | BrouhahaDetector -> DNSMOSDetector (~30 行) |
| 决策逻辑 | **重写** | SNR+C50 三层 -> SIG+BAK+OVRL 两层 (~50 行) |
| 目标函数 | **重写** | SNR GT -> ASR CER GT (~80 行) |
| 搜索空间 | **重写** | 7 参数 -> 6 参数 (~20 行) |

### 6.3 目标函数的关键改进：ASR CER 作为 Ground Truth

原框架的致命问题（循环论证）：
```
SNR -> Brouhaha 检测出 SNR -> 与 SNR 阈值比较 -> 优化 SNR 阈值
```
本质上是用 SNR 优化 SNR，结果只是拟合了 Brouhaha 的输出分布。

新框架的正确做法：
```
混合音频 -> DNSMOS 评分 -> 决策判断 -> 对比 ASR CER Ground Truth -> 计算惩罚
```

流程：
1. 纯净语音 + 噪声 -> 混合音频
2. 对纯净语音运行 ASR -> 参考文本
3. 对混合音频运行 ASR -> 识别文本
4. 计算 CER（Character Error Rate）
5. CER > 阈值 -> ground_truth = SEPARATE
6. CER <= 阈值 -> ground_truth = PASS
7. 与 DNSMOS 决策比较 -> 漏报惩罚 10x / 误报惩罚 1x

### 6.4 新搜索空间

```python
search_space = {
    "ovrl_pass":     (2.0, 3.5),
    "ovrl_separate": (1.0, 2.0),
    "sig_pass":      (2.5, 4.0),
    "sig_separate":  (1.0, 2.5),
    "bak_pass":      (2.0, 3.5),
    "bak_separate":  (1.0, 2.0),
}
```

---

## 7. 风险与回退策略

### 7.1 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| DNSMOS 阈值不准导致误判 | 中 | 中 | 初始阈值保守（宁可多分离），后续通过 Optuna 校准 |
| DNSMOS 对某些噪声类型不敏感 | 低 | 中 | YAMNet Layer 1 兜底 |
| ONNX Runtime 推理异常 | 极低 | 高 | 项目已大量使用 ORT，稳定性已验证 |
| CPU 44ms/chunk 太慢 | 低 | 低 | 仍远快于 Demucs 分离（秒级），且支持批量推理 |

### 7.2 回退策略

如果 DNSMOS 模型不可用（文件缺失/ORT 异常），回退路径：

```
DNSMOSService 不可用
    -> AudioSpectrumClassifier 检测到 dnsmos.is_available() == False
    -> 自动跳过 Layer 0
    -> 直接进入 YAMNet 分诊（与当前系统行为一致）
```

这意味着最坏情况下，新系统的表现**不会差于**当前系统（因为当前 Brouhaha 本身就已失效，系统已在用 YAMNet 兜底）。

---

## 8. 验收标准

### 8.1 功能验收

- [ ] `DNSMOSService` 能正确加载两个 ONNX 模型并推理
- [ ] 纯净语音的 OVRL >= 2.5，判定为 PASS
- [ ] 重噪声的 OVRL < 1.3，判定为 SEPARATE
- [ ] 灰色地带正确下沉到 YAMNet
- [ ] `triage_log.json` 包含 sig/bak/ovrl/quality_level 字段
- [ ] 下游 `SeparationStage` 行为不变

### 8.2 性能验收

- [ ] 单 chunk 推理 < 100ms（CPU）
- [ ] 模型加载 < 500ms
- [ ] 内存增量 < 100MB

### 8.3 兼容性验收

- [ ] `use_dnsmos_triage=False` 时完全回退到 YAMNet/规则
- [ ] DNSMOS 模型文件缺失时优雅降级
- [ ] 所有现有预设（快速/标准/精细）正常工作
- [ ] pyannote 相关依赖成功移除

### 8.4 文档验收

- [ ] `llmdoc/architecture/dnsmos-triage.md` 已就位
- [ ] `llmdoc/architecture/spectral-triage-stage.md` 已更新
- [ ] 本文档状态更新为 Active
