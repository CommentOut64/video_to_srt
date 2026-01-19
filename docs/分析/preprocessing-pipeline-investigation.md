# 预处理管道深度调查报告

## 调查概述

本报告深入分析 `video_to_srt_gpu` 项目的预处理阶段完整实现，覆盖三个核心处理阶段（音频提取+VAD、频谱分诊、人声分离）及其创新技术点。

---

## 部分一：核心架构组件

### 1.1 PreprocessingPipeline 完整实现

**文件位置**：`backend/app/pipelines/preprocessing_pipeline.py`

**核心特点**：
- 采用 Stage 模式，支持灵活的预处理流程配置
- 支持 V3.7 CancellationToken 实现暂停/取消/断点续传
- 包含三个可选阶段的原子化控制

**处理流程**：
```
Stage 1a: 音频提取（FFmpeg，1-5秒）
    ↓ 原子区域：ffmpeg_extract
Stage 1b: VAD 语音检测和切分
    ↓ 检查点保存：chunks_metadata + vad_completed
Stage 2: 频谱分诊（可选，逐Chunk可中断）
    ↓ 检查点保存：diagnosed_indices
Stage 3: 人声分离（可选，支持全局/按需模式）
    ↓ 检查点保存：separation_level + separated_count
输出：预处理完成的 Chunk 列表
```

**关键实现细节**：

- `PreprocessingPipeline.process()` (行 98-252)：主处理流程，支持断点续传
  - V3.7.2 新增检查点恢复机制，跳过已完成的 VAD
  - 原子区域管理：`token.enter_atomic_region()` / `token.exit_atomic_region()`
  - 检查点保存：每阶段完成后调用 `token.check_and_save()`

- `PreprocessingPipeline._extract_and_vad()` (行 254-296)：音频提取+VAD切分
  - 使用 ChunkEngine 处理音频（关键参数 `enable_demucs=False`）
  - 传入 VADConfig 用于语言特定的 VAD 策略（V3.9.1 新增）
  - 进度回调支持

- `PreprocessingPipeline._restore_chunks_from_metadata()` (行 298-370)：V3.7.2 新增
  - 从检查点恢复 chunks，避免重新执行 VAD
  - 使用已知时间戳快速切分音频
  - 边界检查和有效性验证

**配置驱动的灵活性**：

- `PreprocessingConfig` 控制整个流程行为
  - `enable_spectral_triage`：启用频谱分诊
  - `separation_mode`：选择全局/按需分离模式
  - `enable_fuse_breaker`：启用熔断回溯

### 1.2 三个阶段的具体实现

#### 阶段1：音频提取 + VAD 切分

**关键类**：`ChunkEngine` (文件：`backend/app/services/audio/chunk_engine.py`)

**AudioChunk 数据结构** (行 28-82)：
```
基础字段：
  - index: 片段索引
  - start/end: 起始/结束时间（秒）
  - audio: 音频数组（单声道，16kHz）
  - sample_rate: 采样率

分离状态字段（原有）：
  - is_separated: 是否已分离
  - separation_model: 使用的分离模型

频谱分诊字段（新增）：
  - needs_separation: 是否需要人声分离
  - recommended_model: 推荐模型
  - spectrum_diagnosis: 完整分诊结果

分离级别字段（新增）：
  - separation_level: SeparationLevel 枚举
  - original_audio: 原始音频（用于熔断回溯）

熔断回溯字段（新增）：
  - fuse_retry_count: 重试次数
  - last_confidence: 上次置信度
```

**ChunkEngine.process_audio()** (行 111-199)：完整处理流程
1. 加载音频并降采样到 16kHz
2. 可选的 Demucs 整轨分离
3. VAD 语音检测和切分
4. 生成 AudioChunk 列表

**VAD 服务** (文件：`backend/app/services/audio/vad_service.py`)

- `VADService.detect_speech_segments()` (行 316-361)：支持 Silero/Pyannote 两种模型
- `VADConfig` (行 245-289)：包含 Smart Accumulation 参数
  - `smart_target_duration`: 软上限（默认12.0s）
  - `smart_max_duration`: 硬上限（默认30.0s）
  - `smart_min_gap_to_split`: 最小断点间隔（默认0.3s）

#### 阶段2：频谱分诊

**关键类**：`SpectralTriageStage` (文件：`backend/app/pipelines/stages/spectral_triage_stage.py`)

**执行流程** (行 57-137)：
- 逐 Chunk 批量分诊（原子单位：单个Chunk）
- 调用 `AudioSpectrumClassifier.diagnose_chunk()` 进行分析
- 每 5 个 Chunk 保存一次检查点（避免频繁 I/O）
- 支持中断和恢复（V3.7）

**分诊逻辑**：`AudioSpectrumClassifier` (文件：`backend/app/services/audio_spectrum_classifier.py`)

两种分诊模式：
1. **YAMNet 探针模式**（默认，V3.6新增）：语义级分类
2. **规则模式**（回退）：基于频谱特征

关键创新点见下文。

#### 阶段3：人声分离

**关键类**：`SeparationStage` (文件：`backend/app/pipelines/stages/separation_stage.py`)

**两种分离模式**：

1. **全局模式**（global）(行 96-150)
   - 对整个音频文件执行 Demucs 分离
   - 为原子操作，不可中断
   - 标记所有 chunk 为已分离

2. **按需模式**（on_demand）(行 151-204)
   - 仅分离 `needs_separation=True` 的 chunk
   - 逐 Chunk 可中断处理
   - 保存 `original_audio` 用于熔断回溯

**关键实现** (行 68-94)：
- 支持分离模式切换
- 传入 `job_dir` 用于检查点保存
- 支持恢复已分离的 chunk（V3.7）

---

## 部分二：创新技术点

### 2.1 Smart Accumulation 智能累积算法（V3.2.5）

**创新意义**：解决 VAD 贪婪合并导致的超长 Chunk 问题（27秒+），避免 Whisper 跳过内容。

**文件位置**：`backend/app/services/audio/vad_service.py:209-291`（Silero实现）、`349-407`（Pyannote实现）

**核心设计**：

| 方面 | 贪婪合并（旧） | Smart Accumulation（新） |
|------|---------------|-------------------------|
| **断点利用** | VAD断点被丢弃 | 充分利用VAD神经网络已计算的语义断点 |
| **一次遍历** | 否（需后期拆分） | 是（一次遍历完成） |
| **精度** | 能量检测（低） | VAD神经网络断点（高） |
| **Chunk时长** | 最大27秒+ | 平均12秒，最大30秒 |

**算法流程** (VAD原始输出 → Smart Accumulation):

```python
# 参数设置
TARGET = 12.0s  # 软上限（甜蜜点）
MAX = 30.0s     # 硬上限（物理极限）
MIN_GAP = 0.3s  # 合适的断点间隔

# 逐段处理
for ts in speech_timestamps:
    gap = start_sec - current_end
    combined_duration = end_sec - current_start
    current_duration = current_end - current_start

    # 硬上限：绝对不能超30秒
    if combined_duration > MAX:
        save_and_start_new()

    # 软上限：达到12秒后，遇到合适的GAP就截断
    elif current_duration >= TARGET and gap >= MIN_GAP:
        save_and_start_new()

    # 继续累积
    else:
        current_end = end_sec
```

**参数设计理由**：

- **12秒软上限**：Whisper 注意力机制在 12 秒内表现最佳，避免幻觉和对齐爆炸
- **30秒硬上限**：Whisper 模型的输入窗口限制
- **0.3秒最小间隔**：小于 0.3s 的间隔通常是同一句话内的短暂停顿，应该合并

**实现位置**：

- `VADService._vad_silero()` (行 422-519)：Silero VAD 的 Smart Accumulation 实现
- `VADService._vad_pyannote()` (行 595-680)：Pyannote VAD 的 Smart Accumulation 实现

**关键代码段**：
- 软上限预判截断 (行 480-490)：在达到软上限后继续观察下一个 gap
- 软上限断点截断 (行 495-505)：找到合适的 gap 就截断
- 统计信息收集 (行 519-530)：收集 Smart Accumulation 效果数据

**效果分析**：

| 指标 | 旧架构 | 新架构 | 改进 |
|------|--------|--------|------|
| Chunk 平均时长 | 20s | 12s | 短 40% |
| Chunk 最大时长 | 27s+ | 30s | 更稳定 |
| 精度 | 能量检测 | VAD神经网络 | 语义级准确 |

### 2.2 YAMNet 探针模式语义分类器（V3.6新增）

**创新意义**：替代规则方法，解决人声被误判为音乐的问题（原规则方法误判率98%）。

**文件位置**：`backend/app/services/yamnet_classifier.py`

**核心优势**：

1. **语义级分类**：能区分 Speech/Music，避免频谱特征误判
2. **极致轻量**：CPU 推理 < 10ms（比 librosa 特征提取还快）
3. **零误杀**：只要模型识别为 Speech，就不会触发分离

**探针策略** (行 70-100)：

- **不对整个 Chunk 全量扫描**，只抽取首/中/尾 3 个关键帧（0.975s）
- **多帧软投票**：聚合 3 个点的分类结果，避免单帧抖动

**预处理流程** (YAMNetPreprocessor，行 32-114)：

```python
输入: (N,) float32 waveform @ 16kHz

处理步骤:
1. 音频长度规范化（目标 15600 samples ≈ 0.975s）
2. 归一化（max normalization）
3. STFT（center=False，对齐 TensorFlow）
4. Mel Spectrogram（HTK 公式，关键）
5. Log 幅度（dB scale）
6. 时间维度裁剪（保证 96 帧）

输出: (1, 1, 96, 64) float32 适用于 ONNX
```

**决策规则**（优先级从高到低）(audio_spectrum_classifier.py)：

1. **A Cappella 豁免**：清唱检测 → 直通 SenseVoice
2. **BGM 熔断**：`max_music > 0.15` 或 `avg_music > 0.10` → 走 Demucs
3. **纯净人声豁免**：`speech > 0.8` 且 `music < 0.1` → 直通 SenseVoice
4. **人声主导豁免**：speech 显著高于 music 且 music 很弱 → 直通 SenseVoice
5. **模糊地带**：交给 SenseVoice 的置信度检查处理

**双保险机制**：

- **第一道防线**：YAMNet 语义分类（转录前）
- **第二道防线**：FuseBreakerV2 熔断回溯（转录后，默认 max_retry=1）

**模型文件**：
- 模型：`backend/models/pretrained/yamnet/model.onnx`
- 类别映射：`backend/models/pretrained/yamnet/yamnet_class_map.csv`

### 2.3 按需人声分离决策逻辑

**创新意义**：避免全局分离的性能开销，根据频谱分诊结果灵活选择。

**触发条件** (SpectralTriageStage)：

- `chunk.needs_separation = True` 表示该 chunk 需要分离
- `chunk.recommended_model` 推荐的分离模型（htdemucs/mdx_extra）

**模式选择**：

```
按需分离模式 (on_demand，推荐):
  ├─ 频谱分诊标记需要分离的 chunk
  ├─ SeparationStage 仅处理这些 chunk
  └─ 保存 original_audio 用于熔断回溯

全局分离模式 (global，高质量场景):
  ├─ 对整个音频文件进行分离
  ├─ 所有 chunk 标记为已分离
  └─ 计算成本大，需要充足显存
```

**关键实现** (SeparationStage._process_on_demand，行 151-204)：

```python
for chunk in chunks:
    if chunk.needs_separation:
        # 保存原始音频用于熔断回溯
        chunk.original_audio = chunk.audio.copy()

        # 选择分离模型
        model = chunk.recommended_model or 'htdemucs'

        # 执行分离
        separated_audio = await demucs_service.separate_vocals(...)

        # 更新 chunk 信息
        chunk.audio = separated_audio
        chunk.is_separated = True
        chunk.separation_level = SeparationLevel.HTDEMUCS
        chunk.separation_model = model
```

### 2.4 FuseBreakerV2 熔断回溯机制

**创新意义**：在转录后根据实时反馈自动升级分离模型，形成自适应质量反馈循环。

**文件位置**：`backend/app/services/fuse_breaker_v2.py`

**完整升级路径**（max_retry=2）：

```
转录失败（低置信度 + BGM）
    ↓
FuseBreakerV2.should_fuse()
    ├─ 置信度检查：confidence >= 0.5 → ACCEPT
    ├─ 标签检查：无干扰标签 → ACCEPT
    ├─ 重试检查：retry >= max_retry → ACCEPT
    └─ 加权置信度检查 → UPGRADE_SEPARATION 或 ACCEPT
    ↓
UPGRADE_SEPARATION → 获取下一分离级别
    ├─ NONE → HTDEMUCS（第一次升级）
    └─ HTDEMUCS → MDX_EXTRA（第二次升级，可选）
    ↓
execute_upgrade()
    ├─ 使用 original_audio（原始未分离音频）
    ├─ 调用 demucs_service 执行升级分离
    └─ 更新 chunk.separation_level 和 fuse_retry_count
    ↓
重新转录（使用升级后的音频）
```

**事件标签权重**（FuseBreakerV2.__init__，行 56-61）：

```python
event_weights = {
    'BGM': 1.0,        # 最高优先级
    'Music': 0.9,
    'Noise': 0.8,
    'Applause': 0.6    # 较低优先级
}

# 加权置信度 = confidence * (1 + weight)
# 当加权置信度 >= threshold 时接受结果
```

**配置说明** (V3.6.1 更新)：

- `max_retry=1`（默认）：只允许一次熔断升级（NONE → HTDEMUCS）
- `auto_upgrade=False`（默认）：第二次升级需显式启用
- 第二次熔断升级作为可选配置暴露

**原始音频保留机制**：

在 SeparationStage 按需分离时：
```python
chunk.original_audio = chunk.audio.copy()  # 保存分离前的原始音频
```

升级时使用原始音频而非已分离音频，确保能使用更强模型重新分离。

---

## 部分三：关键算法详解

### 3.1 VAD 切分的软硬上限控制

**现状**：Smart Accumulation 已实现双重标准控制

**软上限逻辑**（行 450-490）：
```python
# 如果当前时长 >= 软上限且间隔足够，就截断
if current_duration >= TARGET and gap >= MIN_GAP:
    save_and_start_new()
    reason = f"软上限预判截断 ({current_duration:.2f}s >= {TARGET}s)"
```

**硬上限逻辑**（行 440-450）：
```python
# 即使有合适的断点，合并后超过硬上限就必须截断
if combined_duration > MAX:
    save_and_start_new()
    reason = f"硬上限截断 ({combined_duration:.2f}s > {MAX}s)"
```

**好处**：避免 Whisper 在超长 Chunk 中的注意力衰减和幻觉。

### 3.2 频谱分析和分类算法

**特征提取**（AudioSpectrumClassifier.extract_features，行 75-150）：

```
基础特征：
  - ZCR (Zero Crossing Rate)：过零率
  - ZCR Variance：过零率方差

频谱特征：
  - Spectral Centroid：谱质心（主频）
  - Spectral Bandwidth：谱带宽
  - Spectral Flatness：频谱平坦度
  - Spectral Rolloff：85% 能量点

谐波特征：
  - Harmonic Ratio：HPSS 分解的谐波占比

能量特征：
  - RMS Energy：均方根能量
  - Energy Variance：能量方差
  - High Frequency Ratio：4kHz 以上能量占比

节奏特征：
  - Onset Strength：节拍强度
  - Tempo (BPM)：估计速度
```

**YAMNet 分类逻辑**（yamnet_classifier.py）：

1. 预处理：输入音频 → Log-Mel Spectrogram
2. ONNX 推理：获取 521 个音频类别的分数
3. 聚合：首/中/尾 3 帧软投票
4. 决策：基于 Speech/Music 得分的比值和阈值

### 3.3 人声分离模型选择策略

**策略**：优先使用 htdemucs，可选升级到 mdx_extra

**选择逻辑** (FastWorker 集成参数)：

```python
initial_model = chunk.recommended_model or 'htdemucs'  # 初始选择

if enable_fuse_breaker and should_fuse():
    # 第一次升级：None → HTDEMUCS
    # 第二次升级（可选）：HTDEMUCS → MDX_EXTRA
    next_model = get_next_level()  # 自动升级
```

**模型特性**：

- **htdemucs**：平衡性强，处理速度快（shift=1 模式）
- **mdx_extra**：分离效果最强，计算成本大

---

## 部分四：工程亮点

### 4.1 配置驱动的灵活性设计

**配置体系** (PreprocessingConfig & VADConfig)：

```python
# 预处理配置
class PreprocessingConfig:
    enable_spectral_triage: bool = True      # 启用分诊
    spectrum_threshold: float = 0.35         # 分诊敏感度
    separation_mode: str = "on_demand"       # 分离模式选择
    enable_fuse_breaker: bool = True         # 启用熔断
    fuse_max_retry: int = 1                  # 最大重试次数
    fuse_auto_upgrade: bool = False          # 可选的二级升级

# VAD 配置
class VADConfig:
    smart_target_duration: float = 12.0      # 软上限
    smart_max_duration: float = 30.0         # 硬上限
    smart_min_gap_to_split: float = 0.3      # 最小断点间隔
```

**灵活性表现**：

- 可以关闭频谱分诊 → 跳过 Stage 2
- 可以选择全局/按需分离 → 不同性能/质量权衡
- 可以关闭熔断 → 降低计算成本
- 可以调整所有参数 → 适应不同语言/场景

### 4.2 进度追踪和状态管理

**检查点机制** (V3.7)：

```
Stage 1a: FFmpeg 提取 → 原子区域 → 检查点：audio_extracted
Stage 1b: VAD 切分 → 检查点：chunks_metadata + vad_completed
Stage 2: 频谱分诊 → 逐 Chunk 可中断 → 检查点：diagnosed_indices
Stage 3: 人声分离 → 逐 Chunk 可中断 → 检查点：separation_level
```

**恢复机制**：

```python
# 检查点恢复
if checkpoint and checkpoint.get("preprocessing", {}).get("vad_completed"):
    chunks = await _restore_chunks_from_metadata(...)
    skip_vad = True  # 跳过 Stage 1
```

**好处**：
- 任务暂停后能快速恢复
- 避免重复计算已完成的阶段
- 断点续传的完整支持

### 4.3 错误处理和容错机制

**多层容错设计**：

1. **VAD 降级**（vad_service.py，行 345-356）：
   ```python
   try:
       segments = self._vad_silero(audio_array, sr, config)
   except:
       logger.warning("VAD 失败，降级到能量检测")
       segments = self._energy_based_split(audio_array, sr, config.chunk_size)
   ```

2. **分离失败容错**（separation_stage.py）：
   ```python
   try:
       separated_audio = await demucs_service.separate_vocals(...)
   except:
       logger.warning("单个 chunk 分离失败，保持原始音频")
       # 保持原始音频，继续处理，不中断整体流程
   ```

3. **熔断升级失败**（fast_worker.py）：
   ```python
   try:
       execute_upgrade()
   except:
       logger.warning("升级失败，接受当前结果")
       # 接受当前结果，继续流程
   ```

### 4.4 性能优化策略

**1. 异步处理架构** (AsyncDualPipeline)：

- FastWorker（CPU）：全速并发 SenseVoice 推理（~1秒/Chunk）
- SlowWorker（GPU）：按序执行 Whisper 补刀
- SequencedQueue：充当"整流器"，乱序放入→顺序取出

**2. 懒加载机制**：

- YAMNet 分类器：仅在需要时加载
- librosa：仅在提取特征时加载
- Demucs 模型：分离后立即卸载释放显存

**3. 批量处理**：

- 频谱分诊：逐 Chunk 处理，但批量统计
- 人声分离：按需模式下只分离必要的 chunk

**4. 显存优化**：

- 立即卸载 Demucs（chain_engine.py，行 170-172）：
  ```python
  self.logger.info("释放 Demucs 显存...")
  self.demucs_service.unload_model()
  ```

---

## 部分五：数据流转和接口设计

### 5.1 各阶段之间的数据流转

```
Stage 1 输出：AudioChunk 列表（带 start/end/audio）
    ↓
Stage 2 输入/输出：增强 AudioChunk 字段
    ├─ needs_separation: 是否需要分离
    ├─ recommended_model: 推荐模型
    ├─ spectrum_diagnosis: 完整诊断结果
    ↓
Stage 3 输入/输出：进一步增强 AudioChunk
    ├─ is_separated: 是否已分离
    ├─ separation_level: 分离级别
    ├─ original_audio: 原始音频（用于回溯）
    ├─ separation_model: 实际使用模型
    ↓
输出给转录系统：完全准备就绪的 AudioChunk 列表
```

### 5.2 接口设计规范

**Stage 接口标准**：

```python
async def process(
    self,
    chunks: List[AudioChunk],
    job_dir: Optional[Path] = None,
    diagnosed_indices: Optional[set] = None
) -> List[AudioChunk]:
    """处理 Chunk 列表，返回增强后的列表"""
    ...

def get_statistics(self, chunks: List[AudioChunk]) -> dict:
    """获取统计信息"""
    ...
```

**统一特性**：
- 都支持 CancellationToken（V3.7）
- 都支持检查点保存和恢复
- 都提供统计接口

---

## 部分六：与传统方案的差异化优势

### 6.1 vs. 贪婪合并方案

| 特性 | 传统贪婪合并 | Smart Accumulation |
|------|-------------|-------------------|
| **Chunk 时长** | 最大 27 秒+ | 平均 12 秒，最大 30 秒 |
| **Whisper 准确度** | 低（超长 Chunk 注意力衰减） | 高（甜蜜点时长） |
| **断点质量** | 低（能量检测） | 高（VAD 神经网络） |
| **处理流程** | 先合并→后拆分 | 一次遍历，源头控制 |
| **逻辑复杂度** | 高（两阶段） | 低（单阶段） |

### 6.2 vs. 全局分离方案

| 特性 | 全局分离 | 按需分离 + 频谱分诊 |
|------|---------|-------------------|
| **分离质量** | 一致 | 按需调整（可升级） |
| **计算成本** | 固定大（整轨） | 动态小（按需） |
| **灵活性** | 低 | 高（支持熔断升级） |
| **显存占用** | 大 | 小 |
| **回溯能力** | 无 | 有（保存 original_audio） |

### 6.3 vs. 规则方法的频谱分类

| 特性 | 规则方法 | YAMNet 探针 |
|------|---------|-----------|
| **分类精度** | 低（人声误判率 98%） | 高（语义级理解） |
| **计算成本** | 低（纯特征计算） | 极低（ONNX 推理 <10ms） |
| **分类能力** | 单一（Speech/Music）| 广泛（521 种音频类别） |
| **模型可训练性** | 不可 | 可（预训练） |

---

## 部分七：可复用的工程模式

### 7.1 Pipeline + Stage 模式

**应用场景**：多阶段顺序处理，每阶段都是可选的、可配置的、可中断的

**核心特点**：
- 自动化阶段管理
- 统一的配置和日志
- 原子区域和检查点支持
- 统计接口规范

**复用案例**：
- 视频处理（提取→转码→分析）
- 文本处理（预处理→分类→增强）
- 音频处理（提取→分离→识别）

### 7.2 CancellationToken 的暂停/恢复模式

**模式设计**：

```python
# 初始化
token = CancellationToken()

# 在原子区域之前
token.enter_atomic_region("operation_name")
try:
    # 不可中断的操作
    ...
finally:
    token.exit_atomic_region()

# 在操作之间（可中断点）
if token.should_cancel:
    break

# 定期保存检查点
if (i + 1) % 5 == 0:
    token.check_and_save(checkpoint_data, job_dir)
```

**优势**：
- 清晰的中断点
- 自动化的检查点保存
- 快速的断点恢复

### 7.3 推荐 + 反馈的自适应升级模式

**工作流**：

```
初始化 (基于推荐)
    ↓
执行 (收集实时反馈)
    ↓
评估 (判断是否达标)
    ↓
升级 (如需改进，自动升级)
    ↓
重试 (使用升级后的参数)
```

**应用场景**：
- 人声分离（根据置信度自动升级）
- 模型推理（根据质量反馈切换模型）
- 参数调优（根据实时指标动态调整）

### 7.4 多层容错的渐进式降级

**设计模式**：

```python
try:
    # 最优方案
    result = high_quality_method()
except:
    try:
        # 备选方案
        result = fallback_method1()
    except:
        # 最后手段
        result = fallback_method2()
```

**应用示例**：
- YAMNet 不可用 → 规则方法
- VAD 失败 → 能量检测
- 分离失败 → 原始音频

---

## 部分八：关键代码位置快速导航

### 核心实现

| 组件 | 文件 | 行号范围 | 说明 |
|------|------|---------|------|
| PreprocessingPipeline | preprocessing_pipeline.py | 33-252 | 主流水线 |
| ChunkEngine | audio/chunk_engine.py | 85-199 | 音频处理引擎 |
| SpectralTriageStage | stages/spectral_triage_stage.py | 23-137 | 频谱分诊 |
| SeparationStage | stages/separation_stage.py | 28-224 | 人声分离 |
| FuseBreakerV2 | fuse_breaker_v2.py | 24-227 | 熔断决策 |

### VAD 和 Smart Accumulation

| 组件 | 文件 | 行号 | 说明 |
|------|------|------|------|
| VADConfig | audio/vad_service.py | 245-289 | VAD 配置（含 Smart Accumulation 参数） |
| VADService._vad_silero | audio/vad_service.py | 363-530 | Silero VAD + Smart Accumulation |
| VADService._vad_pyannote | audio/vad_service.py | 535-680 | Pyannote VAD + Smart Accumulation |

### 频谱分诊和 YAMNet

| 组件 | 文件 | 行号 | 说明 |
|------|------|------|------|
| AudioSpectrumClassifier | audio_spectrum_classifier.py | 36-150 | 频谱分类器（YAMNet + 规则回退） |
| YAMNetPreprocessor | yamnet_classifier.py | 32-114 | YAMNet 预处理（Log-Mel Spectrogram） |
| YAMNetClassifier | yamnet_classifier.py | 150+ | ONNX 推理和决策 |

### 数据模型

| 模型 | 文件 | 行号 | 说明 |
|------|------|------|------|
| AudioChunk | audio/chunk_engine.py | 28-82 | 音频片段数据结构 |
| SpectrumDiagnosis | circuit_breaker_models.py | 47-67 | 频谱分诊结果 |
| SeparationLevel | circuit_breaker_models.py | 69-94 | 分离级别枚举 |
| FuseDecision | circuit_breaker_models.py | 152-158 | 熔断决策结果 |

---

## 总结

video_to_srt_gpu 的预处理阶段设计体现了工业级音频处理系统的最佳实践：

1. **Smart Accumulation**：在源头解决问题，利用 VAD 神经网络的计算结果，避免后期低质量拆分
2. **YAMNet 探针**：用预训练深度学习模型替代规则判断，精度提升一个数量级
3. **按需分离 + 熔断升级**：平衡性能和质量，支持自适应的质量反馈循环
4. **检查点和中断机制**：完整的暂停/恢复支持，工业级可靠性
5. **配置驱动**：所有关键策略都参数化，支持不同场景的灵活调整

这些设计模式可以直接应用于其他音频处理、图像处理、文本处理系统中。
