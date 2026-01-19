# 完整系统架构设计文档

**文档版本**: 1.0
**创建日期**: 2025-12-26
**涵盖范围**: V3.8.0 完整系统架构
**受众**: LLM 代理、开发人员、技术评审

---

## 目录

1. [预处理阶段](#1-预处理阶段)
2. [快慢双流架构](#2-快慢双流架构)
3. [智能补刀和熔断](#3-智能补刀和熔断)
4. [对齐和分句](#4-对齐和分句)
5. [后端推送 (SSE)](#5-后端推送-sse)
6. [前端实现](#6-前端实现)
7. [完整数据流图](#7-完整数据流图)
8. [关键设计决策](#8-关键设计决策)

---

## 1. 预处理阶段

### 1.1 Identity

音频处理流水线的第一阶段，包括音频提取、频谱分诊、人声分离、VAD切分和智能累积。输出经过优化的 AudioChunk 列表供转录流水线使用。

### 1.2 Core Components

- `backend/app/pipelines/audio_processing_pipeline.py` (AudioProcessingPipeline): 主控制器，编排各子阶段
- `backend/app/services/audio/chunk_engine.py` (AudioChunk): 音频片段数据结构（9 个新字段）
- `backend/app/services/audio/vad_service.py` (VADService, VADConfig): VAD 语音检测和切分，包含 Smart Accumulation（V3.2.5）
- `backend/app/pipelines/stages/spectral_triage_stage.py` (SpectralTriageStage): 频谱分诊阶段
- `backend/app/services/audio_spectrum_classifier.py` (AudioSpectrumClassifier): 频谱分析，支持 YAMNet 语义分类
- `backend/app/services/yamnet_classifier.py` (YAMNetClassifier): YAMNet 探针模式分类器（V3.6）
- `backend/app/pipelines/stages/separation_stage.py` (SeparationStage): 人声分离阶段
- `backend/app/services/demucs_service.py` (DemucsService): Demucs 人声分离服务

### 1.3 Execution Flow (LLM Retrieval Map)

```
源音频文件
  ↓
[阶段1] 音频提取 & 降采样
  - 使用 librosa 或 pydub 提取音频
  - 转换为 16kHz 单声道
  - 生成 np.ndarray (full_audio)
  ↓
[阶段2] 显存策略检查（可选）
  - 查询 NVIDIA GPU 可用显存
  - 决定整轨分离 vs 大块切分 vs 不分离
  ↓
[阶段3] 人声分离（可选）
  - 全局模式: 整轨分离，更新全局音频
  - 按需模式: 根据频谱分诊结果，仅对需要的 chunk 分离
  - 保存 chunk.original_audio 用于熔断回溯
  ↓
[阶段4] VAD 切分 & Smart Accumulation
  - Silero VAD / Pyannote: 检测语音活动
  - Smart Accumulation（V3.2.5）: 利用 VAD 精确断点
    * 软上限: 12 秒（甜蜜点）
    * 硬上限: 30 秒（物理极限）
    * 最小间隔: 0.3 秒
  - 输出: 平均 12s、最大 30s 的 Chunk 列表
  ↓
[阶段5] 频谱分诊（可选）
  - 批量分析 Chunk 频谱特征
  - YAMNet 探针模式: 首/中/尾 3 点采样，语义分类 Speech vs Music
  - 为每个 chunk 标记:
    * needs_separation: bool
    * recommended_model: 'htdemucs' 或 None
    * spectrum_diagnosis: SpectrumDiagnosis 对象
  ↓
最终 AudioChunk 列表（带分离标记）
```

### 1.4 关键字段传递

**AudioChunk 扩展字段（9 个新字段，参考 reference/data-models-expansion.md）**:

| 字段名 | 类型 | 来源 | 用途 |
|--------|------|------|------|
| `needs_separation` | bool | 频谱分诊 | 是否需要人声分离 |
| `recommended_model` | str | 频谱分诊 | 推荐分离模型（'htdemucs'） |
| `spectrum_diagnosis` | obj | 频谱分诊 | 频谱分析诊断结果 |
| `original_audio` | ndarray | 人声分离 | 原始音频（用于熔断回溯） |
| `is_separated` | bool | 人声分离 | 是否已分离 |
| `separation_level` | enum | 人声分离/熔断 | 分离级别（NONE/HTDEMUCS/MDX_EXTRA） |
| `fuse_retry_count` | int | 熔断 | 熔断重试次数 |
| `last_confidence` | float | 转录 | SenseVoice 最后一次置信度 |
| `fuse_action` | enum | 熔断 | 熔断决策结果 |

### 1.5 YAMNet 频谱分诊流程（V3.6）

```
AudioChunk 列表（来自 VAD 切分）
  ↓
SpectralTriageStage.process()
  ↓
  ├─ [YAMNet 可用] → YAMNetClassifier.classify_chunk()
  │   ├─ 探针模式: 首（0s）/中（duration/2）/尾（duration）3 点采样
  │   ├─ 软投票: max_music、avg_music、speech 得分
  │   └─ 决策规则（优先级从高到低）:
  │       1. A Cappella 豁免: 清唱检测 → 直通 SenseVoice
  │       2. BGM 熔断: max_music > 0.15 或 avg_music > 0.10 → 走 Demucs
  │       3. 纯净人声豁免: speech > 0.8 且 music < 0.1 → 直通 SenseVoice
  │       4. 人声主导豁免: speech 显著高于 music → 直通 SenseVoice
  │       5. 模糊地带: 交给 SenseVoice 置信度处理
  │
  └─ [YAMNet 不可用] → 规则方法（已弃用，保留向后兼容）

为每个 chunk 标记: needs_separation, recommended_model
```

### 1.6 Smart Accumulation 算法（V3.2.5）

**核心问题**: 贪婪合并导致 27 秒+ 超长 Chunk，Whisper 在超长 Chunk 中跳过内容。

**解决方案**: 在 VAD 原始输出遍历时，利用精确的语义断点进行智能累积。

```python
TARGET = 12.0s  # 软上限（甜蜜点）
MAX = 30.0s     # 硬上限（物理极限）
MIN_GAP = 0.3s  # 合适的断点间隔

for ts in speech_timestamps:
    gap = start_sec - current_end
    combined_duration = end_sec - current_start
    current_duration = current_end - current_start

    # 硬上限: 绝对不能超 30 秒
    if combined_duration > MAX:
        save_and_start_new()

    # 软上限: 达到 12 秒后，遇到合适的 GAP 就截断
    elif current_duration >= TARGET and gap >= MIN_GAP:
        save_and_start_new()

    # 继续累积
    else:
        current_end = end_sec
```

**参数设计**:
- **12 秒**: Whisper 注意力机制最佳表现点；WhisperX 对齐算法不会"爆炸"；显存占用线性增长
- **30 秒**: Whisper 模型的输入窗口限制
- **0.3 秒**: 小于 0.3s 的间隔通常是同一句话内的短暂停顿

### 1.7 人声分离两种模式

**全局模式（global）**:
- 对完整音频文件进行整轨分离
- 质量最高但计算成本大
- 适合高显存场景

**按需模式（on_demand）**:
- 根据频谱分诊结果，仅对 `needs_separation=True` 的 chunk 分离
- 灵活高效，适合资源受限场景
- 保存 `original_audio` 用于熔断回溯

---

## 2. 快慢双流架构

### 2.1 Identity

三级异步流水线，实现乱序执行、顺序提交。CPU 层（FastWorker）并发处理音频 Chunk，GPU 层（SlowWorker）顺序处理确保 Whisper 上下文连贯性。支持极速模式（sensevoice_only）跳过 Whisper 直接输出定稿。

### 2.2 Core Components

- `backend/app/pipelines/async_dual_pipeline.py` (AsyncDualPipeline): 三级流水线控制器
- `backend/app/pipelines/workers/fast_worker.py` (FastWorker): CPU 层快速推理
- `backend/app/pipelines/workers/slow_worker.py` (SlowWorker): GPU 层顺序处理
- `backend/app/utils/sequenced_queue.py` (SequencedAsyncQueue): 智能序列化队列
- `backend/app/services/gpu_coordinator.py` (GPUCoordinator): GPU 资源协调器

### 2.3 Execution Flow (LLM Retrieval Map)

#### 2.3.1 V3.5 极速模式（sensevoice_only）

```
VAD 切分后的 AudioChunk 列表
  ↓
AsyncDualPipeline.__init__()
  - transcription_profile == "sensevoice_only"
  - is_sensevoice_only = True
  - FastWorker: is_final_output = True
  - 跳过 SlowWorker 和 AlignmentWorker 创建
  ↓
_run_sensevoice_only()
  - 顺序遍历 Chunk 列表（不使用异步队列）
  - 每个 Chunk:
      1. FastWorker.process(chunk)
      2. add_finalized_sentences()  # 推送定稿，不推送草稿
      3. 完成
  ↓
最终输出（无 Whisper 延迟）
```

#### 2.3.2 智能补刀模式（sv_whisper_patch，V3.10）

```
VAD 切分后的 AudioChunk 列表
  ↓
AsyncDualPipeline.__init__()
  - transcription_profile == "sv_whisper_patch"
  - is_patching_mode = True
  - 创建 FastWorker（is_final_output=False，输出草稿）
  - 创建 SlowWorker（is_patching_mode=True）
  - 创建 AlignmentWorker
  ↓
_run_full_pipeline()
  - FastWorker 循环（CPU 并发）
      + 并发读取 Chunk 列表
      + 执行 SenseVoice 推理（~1 秒/Chunk）
      + 分句和推送草稿 SSE 事件
      + 结果放入 SequencedQueue
  ↓
  - SequencedQueue（中间层）
      + 缓冲乱序的处理结果
      + 按 chunk_index 顺序推送到内部队列
      + put() 内置背压（当 buffer + inner_queue >= max_size 时阻塞）
  ↓
  - SlowWorker 循环（GPU 顺序）
      + 从 SequencedQueue 按序取出 Chunk
      + 拼接长音频（WhisperBufferPool）
      + 调用 Whisper 推理（GPU 独占）
      + 仲裁机制（V3.8）: Whisper 二次听诊低置信度句子
      + 结果放入 Queue2
  ↓
  - AlignmentWorker 循环（CPU）
      + 对齐文本和时间戳
      + 推送最终字幕 SSE 事件
  ↓
最终输出
```

#### 2.3.3 双流并行机制

```
FastWorker（56 个任务并发）
  Chunk 0: 0.1s (快速) → 立即完成 → Queue1
  Chunk 1: 5.2s (复杂) → 后台处理 → Queue1
  Chunk 2: 0.3s (快速) → 立即完成 → Queue1
  ...
  ↓ 乱序缓冲在 SequencedQueue

SequencedQueue（V3.2.3 精确唤醒）
  - 接收乱序的处理结果
  - 缓冲在内部 buffer（最多 10 个）
  - 按 chunk_index 推送到 inner_queue
  - 背压: 当 buffer + inner_queue >= max_size 时 put() 阻塞
  - V3.2.4 修复: 在 _try_flush() 中 await asyncio.sleep(0)，让出控制权
  ↓

SlowWorker（GPU 顺序，同步消费）
  Chunk 0: Whisper 推理（GPU）→ Queue2
  Chunk 1: Whisper 推理（GPU）→ Queue2
  Chunk 2: Whisper 推理（GPU）→ Queue2
  ...
  ↓ 严格按时间戳排序

AlignmentWorker
  Chunk 0: 对齐 → 最终字幕
  Chunk 1: 对齐 → 最终字幕
  ...
```

### 2.4 SequencedQueue 序列化机制（V3.2.3 + V3.2.4）

```python
class SequencedAsyncQueue:
    """智能序列化队列"""

    async def put(self, item: ProcessingContext):
        """
        乱序放入（V3.2.1 背压逻辑）
        """
        # 放入缓冲区
        buffer[item.chunk_index] = item

        # 内置背压: 当 buffer + inner_queue >= max_buffer_size 时阻塞
        while len(buffer) + self.inner_queue.qsize() >= max_buffer_size:
            await self.condition.wait()

        # 尝试推送到内队列
        await self._try_flush()

    async def get(self) -> ProcessingContext:
        """
        顺序取出（V3.2.3 精确唤醒）
        """
        # 从内队列获取已排序项
        item = await self.inner_queue.get()

        # 每消费一个，只唤醒一个等待的 put()（V3.2.3）
        self.condition.notify()

        return item

    async def _try_flush(self):
        """
        推送排序好的项到内队列（V3.2.4）
        """
        while next_expected_index in buffer:
            item = buffer.pop(next_expected_index)
            self.inner_queue.put_nowait(item)
            next_expected_index += 1

            # V3.2.4: 让出控制权，防止 SlowWorker 饿死
            await asyncio.sleep(0)
```

### 2.5 FastWorker 推理流程

```
process(chunk: AudioChunk)
  ↓
[1] SenseVoice 推理
    - executor.run_inference(chunk.audio)
    - 返回: 词序列 + 时间戳 + 置信度 + 事件标签
  ↓
[2] CTC 重叠去重（V3.8）
    - 移除 "W" + "Would" → "WWouldn't" 这类重复
    - 三重条件判断: 前缀 + 时间间隔 <= 0.1s + 词长 <= 2
  ↓
[3] 分句（两层策略）
    - Layer 1: SentenceSplitter（主要依赖 VAD 停顿）
      * 使用动态停顿阈值（V3.2.0 优化）
      * 75 百分位数替代中位数
      * 软上限 12s、硬上限 20s
    - Layer 2: SemanticGrouper（物理约束）
      * 时间间隔 >= 2s → 分组边界
      * 单组最大时长 10s
      * 最多合并 5 个句子
  ↓
[4] 熔断回溯（可选，enable_fuse_breaker=True）
    while True:
      - should_fuse() 判断是否需要升级
      - 若 ACCEPT → 退出循环
      - 若 UPGRADE_SEPARATION → 执行升级，继续循环
      - 若升级失败 → 接受当前结果，退出循环
  ↓
[5] 推送到 SSE
    - is_final_output=True: add_finalized_sentences()（定稿）
    - is_final_output=False: add_draft_sentences()（草稿）
  ↓
填充 context.sv_result
返回给流水线
```

### 2.6 SlowWorker 推理流程

```
process(context: ProcessingContext)
  ↓
[1] 组织音频上下文
    - WhisperBufferPool: 拼接当前 chunk 和前几个 chunk 的音频
    - 保持 Whisper 的上文连贯性
  ↓
[2] Whisper 推理
    - faster_whisper.transcribe()
    - 返回: 完整转录 + 分段 + 置信度
  ↓
[3] 仲裁机制（V3.8）
    - 识别低置信度句子（< 0.4）
    - 对疑似垃圾句子进行 Whisper 二次听诊
    - 决策: 删除 / 替换 / 保留
  ↓
[4] 字级单字符强制补刀（V3.5）
    - 检查单字符实词（"I", "A" 等）
    - 置信度 < 0.9 → 触发补刀
  ↓
[5] 词级对齐
    - 使用 faster-whisper 的对齐结果
    - 生成字级时间戳
  ↓
填充 context.whisper_result
返回给流水线
```

### 2.7 关键设计决策

**为什么需要乱序执行？**
1. CPU 层并发: FastWorker 处理时间差异大（简单 0.1s，复杂 5s+），串行处理浪费资源
2. GPU 层顺序: Whisper 需要上下文连贯性，必须按时间戳顺序处理
3. SequencedQueue: 在两层之间充当"整流器"，解耦并发和顺序

**为什么采用 V3.2.4 修复？**
- **问题**: `_try_flush()` 是同步循环，无 `await`，占用事件循环 → SlowWorker 饿死
- **解决**: 在 `_try_flush()` 中每次推送后加 `await asyncio.sleep(0)`，让出控制权
- **效果**: 确保 SlowWorker 能在 FastWorker 处理期间及时获得调度，实现真正的双流并行

**为什么使用 Condition 而非 Event？（V3.2.3）**
- **Event.set()**: 唤醒所有等待者（惊群效应）
- **Condition.notify()**: 只唤醒一个等待者（精确流量控制）
- **效果**: 实现真正的背压 - 每消费一个 Chunk，只允许一个新 Chunk 进入

---

## 3. 智能补刀和熔断

### 3.1 Identity

SenseVoice 转录后的质量监控和自动补救机制。包括四种触发条件的 Whisper 补刀判决、FuseBreakerV2 熔断决策、加权置信度计算和自动升级策略。

### 3.2 Core Components

- `backend/app/core/thresholds.py` (ThresholdConfig, needs_whisper_patch): 补刀触发条件定义
- `backend/app/services/fuse_breaker_v2.py` (FuseBreakerV2): 熔断决策器核心类
- `backend/app/models/circuit_breaker_models.py` (FuseAction, FuseDecision, SeparationLevel): 决策数据模型
- `backend/app/services/audio/chunk_engine.py` (AudioChunk): 包含熔断相关字段
- `backend/app/services/demucs_service.py` (DemucsService): 分离升级服务
- `backend/app/services/transcription_service.py` (_post_process_enhancement): 补刀判决集成

### 3.3 Execution Flow (LLM Retrieval Map)

#### 3.3.1 四种 Whisper 补刀触发条件（V3.5 增强）

```
后处理增强流程
  ↓
for sentence in sentences:
    if needs_whisper_patch(sentence):
        触发 Whisper 补刀
  ↓
conditions:
  1. 置信度低: confidence < 0.6（原有逻辑）
  2. 短片段: duration < 1s AND text_length < 3（应对 CTC 限制）
  3. 字级检查: 检查单词级别的置信度
  4. 单字符强制: single_char_word AND confidence < 0.9
     - 优先于停用词列表
     - 因为单字符错误风险极高
```

#### 3.3.2 FuseBreakerV2 完整升级路径（max_retry=2）

```
FastWorker 熔断循环
  ↓
while True:
    [1] SenseVoice 推理
        chunk = perform_sensevoice_inference(chunk)
        confidence = chunk.last_confidence
        event_tags = chunk.event_tags
    ↓
    [2] 熔断决策（should_fuse()）
        检查1: confidence >= 0.5 → ACCEPT
        检查2: 无事件标签（BGM/Music/Noise/Applause） → ACCEPT
        检查3: fuse_retry_count >= max_retry → ACCEPT（放弃升级）
        检查4: 加权置信度 >= threshold → ACCEPT
        否则 → UPGRADE_SEPARATION
    ↓
    [3] 升级分离（execute_upgrade()）
        使用原始音频: chunk.original_audio
        升级路径:
          NONE → HTDEMUCS（第 1 次重试）
          HTDEMUCS → MDX_EXTRA（第 2 次重试，仅当 fuse_auto_upgrade=True）
        更新: chunk.separation_level, chunk.fuse_retry_count
    ↓
    [4] 重新转录
        使用升级后的分离音频重新调用 SenseVoice
        继续循环
    ↓
    [5] 退出循环（ACCEPT 或达到 max_retry）

最后接受的结果 → 加入字幕管理器
```

### 3.3.3 加权置信度计算

```python
# FuseBreakerV2.should_fuse() 逻辑
def should_fuse(self, chunk: AudioChunk, confidence: float, event_tags: list) -> FuseAction:
    """
    判断是否需要升级分离

    Args:
        confidence: SenseVoice 置信度（0-1）
        event_tags: 检测到的事件标签列表

    Returns:
        FuseAction: ACCEPT / UPGRADE_SEPARATION / ...
    """

    # 检查1: 置信度足够 → 直接接受
    if confidence >= 0.5:
        return FuseAction.ACCEPT

    # 检查2: 无背景干扰 → 直接接受
    if not has_background_tags(event_tags):
        return FuseAction.ACCEPT

    # 检查3: 重试次数已达上限 → 放弃升级
    if chunk.fuse_retry_count >= self.max_retry:
        return FuseAction.ACCEPT

    # 检查4: 计算加权置信度
    weighted_confidence = confidence
    for tag in event_tags:
        weight = tag_weights.get(tag, 0.0)  # BGM:1.0, Music:0.9, Noise:0.8, Applause:0.6
        weighted_confidence = confidence * (1 + weight)

    if weighted_confidence >= self.confidence_threshold:
        return FuseAction.ACCEPT

    # 需要升级
    return FuseAction.UPGRADE_SEPARATION
```

### 3.3.4 自动升级策略

```
升级决策（V3.6.1 默认配置）:

  max_retry = 1（默认）:
    第 1 次: NONE → HTDEMUCS
    第 2 次: 不再升级，接受结果

  max_retry = 2 && auto_upgrade = True:
    第 1 次: NONE → HTDEMUCS
    第 2 次: HTDEMUCS → MDX_EXTRA（跳级）

  max_retry = 2 && auto_upgrade = False（默认）:
    第 1 次: NONE → HTDEMUCS
    第 2 次: HTDEMUCS → HTDEMUCS（不升级，重复尝试）
```

### 3.3.5 事件标签权重表

| 标签 | 权重 | 严重性 | 处理优先级 |
|------|------|--------|-----------|
| BGM（背景音乐） | 1.0 | 最高 | 1 - 立即升级 |
| Music（音乐） | 0.9 | 高 | 2 - 优先升级 |
| Noise（噪音） | 0.8 | 中 | 3 - 适度升级 |
| Applause（掌声） | 0.6 | 低 | 4 - 必要时升级 |

### 3.4 设计决策

**为什么需要原始音频保留？**
- 在分离阶段保存 `chunk.original_audio`，为熔断回溯提供回退点
- 升级时使用原始音频而非已分离音频，确保能使用更强模型重新分离

**为什么 max_retry 默认为 1？**
- 兼顾性能和质量的平衡
- 避免过度升级导致处理时间过长
- 第二次升级作为可选配置暴露，需显式启用

**为什么分离失败时不中断流程？**
- 错误容错: 升级失败时接受当前结果，继续处理下一个 Chunk
- 保证系统可用性：单个 Chunk 的分离失败不应导致整体任务中断

---

## 4. 对齐和分句

### 4.1 Identity

文本和时间戳的精确对齐，以及多层分句策略。包括动态停顿阈值算法、两层分句策略、硬上限强制保留机制、CTC 重叠去重和 Whisper 仲裁机制。

### 4.2 Core Components

- `backend/app/services/sentence_splitter.py` (SentenceSplitter, _calculate_dynamic_pause_threshold): 分句器，Layer 1
- `backend/app/services/semantic_grouper.py` (SemanticGrouper): 语义分组器，Layer 2
- `backend/app/services/sensevoice_onnx_service.py` (CTCDecoder._remove_overlap_duplicates): CTC 重叠去重
- `backend/app/services/streaming_subtitle.py` (mark_for_deletion, remove_marked_sentences): 仲裁机制
- `backend/app/services/transcription_service.py` (_whisper_text_patch_with_arbitration): Whisper 仲裁
- `backend/app/services/word_timestamp_calculator.py` (WordTimestampCalculator): 字级时间戳计算

### 4.3 Execution Flow (LLM Retrieval Map)

#### 4.3.1 动态停顿阈值算法（V3.2.0 优化）

```
输入: 词序列 + 词间停顿时间列表
  ↓
_calculate_dynamic_pause_threshold()
  ↓
[1] 统计停顿时间
    - 收集所有词间间隔
    - 排序: [0.12s, 0.18s, 0.45s, 0.78s, 1.2s, ...]
  ↓
[2] 百分位数计算（V3.2.0 优化）
    - 原: median（50 百分位）
    - 新: 75 百分位数
    - 理由: 提高抗噪性，避免对短间隔过度敏感
  ↓
[3] 权重平衡（V3.2.0 优化）
    - 动态权重: percentile_75
    - 静态基准: 0.7s（提升自 0.5s）
    - 权重分配: 50% 动态 + 50% 静态（原为 70:30）
    - 动态阈值 = percentile_75 * 0.5 + 0.7 * 0.5
  ↓
[4] 乘数调整（V3.2.0 优化）
    - 原: pause_multiplier = 2.0
    - 新: pause_multiplier = 2.5
    - long_pause_threshold = 1.5s（原 1.0s）
  ↓
[5] 最小阈值保护（V3.2.0 新增）
    - min_pause_threshold = 0.5s（默认）
    - 防止极端情况下阈值过低
  ↓
输出: 自适应停顿阈值
```

#### 4.3.2 两层分句策略

```
Layer 1: SentenceSplitter（主要依赖 VAD 停顿）
  ↓
  [1] 遍历词序列
      for word in words:
          累积当前句子
          检查是否到达分句点
  ↓
  [2] 分句触发条件（优先级）:
      - 标点符号 (prefer_punctuation_break=False，不使用)
      - VAD 停顿 > 动态阈值（主要方法）
      - 时长超过硬上限 20s（保护机制，V3.2.2）
      - 到达 Chunk 末尾
  ↓
  [3] 硬上限强制保留（V3.2.2）
      if current_duration >= hard_limit_duration:
          force_create = True（强制创建句子，跳过 min_chars 检查）
  ↓
  输出: 句子列表（可能较碎片化）

  ↓

Layer 2: SemanticGrouper（物理约束）
  ↓
  [1] 语义分组规则:
      - 时间间隔 >= 2s → 分组边界
      - 单组最大时长 10s
      - 最多合并 5 个句子
  ↓
  [2] 合并逻辑:
      for sentence in sentences:
          if can_merge_with_next(sentence):
              merge
          else:
              new_group
  ↓
  输出: 优化后的句子列表（质量更好）
```

#### 4.3.3 CTC 重叠去重（V3.8）

```
CTC 解码完成 (sensevoice_onnx_service.py)
  ↓
_remove_overlap_duplicates()
  ↓
for i, word in enumerate(words):
    next_word = words[i+1]

    # 三重条件判断（需同时满足）:
    条件1: word 是 next_word 的前缀（忽略大小写）
    条件2: 时间间隔 <= 0.1s
    条件3: 当前词长度 <= 2（避免误删）

    if all(conditions):
        remove(word)  # 移除重复的前缀词
  ↓
示例:
  输入: ["W", "Would", "you", ...]
       时间: [0.1s, 0.15s, 0.5s, ...]
  输出: ["Would", "you", ...]  # "W" 被移除
```

#### 4.3.4 Whisper 仲裁机制（V3.8）

```
SlowWorker 完成 Whisper 推理
  ↓
后处理增强阶段
  ↓
[1] 识别嫌疑句子
    for sentence in sv_sentences:
        if confidence < 0.4:  # 垃圾嫌疑阈值
            mark_for_deletion(sentence)
  ↓
[2] 批量仲裁（_whisper_text_patch_with_arbitration）
    for suspect_sentence in suspects:
        whisper_result = whisper.transcribe(suspect_sentence.audio)

        [3] 判决逻辑 (三选一):
            a. Whisper confidence < 0.5 或文本为空
               → 删除（确实是垃圾）
            b. Whisper confidence >= 0.5 且有实质文本
               → 用 Whisper 结果替换
            c. 其他情况
               → 保留原句（特殊保护）
  ↓
[4] 物理删除
    remove_marked_sentences()  # 批量删除
  ↓
最终字幕列表
```

### 4.4 字级时间戳计算

```
来自 FastWorker 的句子:
  - 词序列: ["It's", "still", "only", "7:28", "pm"]
  - 词级时间戳: [(0.0-0.4), (0.45-0.8), (0.85-1.3), (1.35-2.0), (2.05-2.5)]

字级时间戳生成策略:
  [混合策略] 优先使用字级时间戳，回退字数比例
  ↓
  for word in words:
      if has_precise_timestamp:
          use_precise_timestamp
      else:
          calculate_by_proportion(
              word_start,
              word_duration,
              word_chars
          )
```

### 4.5 设计决策

**为什么 Layer 1 依赖 VAD 停顿而非标点？**
- 某些语言（中文）缺乏自然标点
- VAD 是神经网络计算的真实语音活动断点
- 避免对标点的过度依赖

**为什么 V3.2.2 需要硬上限强制保留？**
- VAD 切分可能产生超长 Chunk（极端情况）
- 硬上限（20 秒）作为最后防线
- 强制保留内容（跳过 min_chars 检查），避免尾部词汇丢失

**为什么需要 Whisper 仲裁？**
- SenseVoice 在低置信度（< 0.4）时可能产生幻觉（如"SRRCT"）
- 不能直接删除（可能误杀有效内容）
- 通过 Whisper 二次听诊准确区分垃圾 vs 含糊但有效的语音

---

## 5. 后端推送 (SSE)

### 5.1 Identity

Server-Sent Events 系统，使用命名空间统一事件类型。为前端提供实时的转录进度和阶段性字幕更新。

### 5.2 Core Components

- `backend/app/services/sse_service.py` (SSEManager): SSE 连接管理器
- `backend/app/services/job_queue_service.py` (QueueService): 队列事件广播
- `backend/app/services/transcription_service.py`: 转录进度事件发送
- `backend/app/services/streaming_subtitle.py` (StreamingSubtitleManager): 字幕流式传输
- `backend/app/services/progress_emitter.py` (ProgressEventEmitter): V3.7.1 统一进度发射器

### 5.3 Execution Flow (LLM Retrieval Map)

#### 5.3.1 事件发送端（后端）

```
转录流水线运行中
  ↓
[1] TranscriptionService 在关键节点调用 sse_manager.broadcast()
    - 事件类型使用命名空间前缀: progress.*, signal.*, subtitle.*
  ↓
[2] 进度事件发送
    ProgressEventEmitter（V3.7.1）
      - 统一进度发射器
      - 实时同步 job.progress 和推送 SSE
      - 计算快流、慢流、对齐、预处理的独立进度
      - 推送 progress.overall 事件（含 detail.fast, detail.slow, detail.align 等）
  ↓
[3] 字幕事件发送
    StreamingSubtitleManager
      - add_draft_sentences()：推送 subtitle.sv_sentence（草稿）
      - add_finalized_sentences()：推送 subtitle.finalized（定稿，V3.5）
      - update_sentence()：推送 subtitle.whisper_patch（补刀）
  ↓
[4] 信号事件发送
    JobQueueService
      - signal.job_start, signal.job_complete, signal.job_failed
      - signal.circuit_breaker, signal.model_upgrade
```

#### 5.3.2 V3.7.1 双流进度计算和推送

```
ProgressEventEmitter.update_overall_progress()
  ↓
[1] 计算各层进度（基于 Chunk 处理数）
    - preprocess: 音频提取 + 频谱分诊 + 人声分离 + VAD 切分
    - fast: FastWorker 处理完成的 Chunk 数 / 总数
    - slow: SlowWorker 处理完成的 Chunk 数 / 总数
    - align: AlignmentWorker 处理完成的 Chunk 数 / 总数
  ↓
[2] 推送 progress.overall 事件（V3.7.1）
    {
      "percent": 45.2,              # 总进度百分比
      "phase": "sensevoice",        # 当前主阶段
      "status": "processing",       # 状态
      "total": 20,                  # 总 Chunk 数
      "detail": {                   # V3.7.1 新增
        "preprocess": 100,          # 预处理进度 (%)
        "fast": 45,                 # FastWorker 进度 (%)
        "slow": 30,                 # SlowWorker 进度 (%)
        "align": 15                 # AlignmentWorker 进度 (%)
      }
    }
  ↓
[3] 前端消费（EditorView.vue:onProgress）
    - 检查 data.detail 是否存在
    - 提取 data.detail.fast 和 data.detail.slow
    - 更新双层进度条
    - 渲染 UI
```

#### 5.3.3 事件类型规范（命名空间化）

**进度事件（progress.*）**:
```
progress.overall       # 总体进度（V3.7.1+ 含 detail 字段）
progress.preprocess    # 预处理阶段
progress.extract       # 音频提取
progress.spectrum_analysis  # 频谱分析
progress.demucs        # 人声分离
progress.vad           # 语音活动检测
progress.sensevoice    # SenseVoice 推理
progress.whisper       # Whisper 补刀
progress.llm_proof     # LLM 校对
progress.llm_trans     # LLM 翻译
progress.srt           # SRT 生成
```

**信号事件（signal.*）**:
```
signal.job_start       # 任务开始
signal.job_complete    # 任务完成
signal.job_failed      # 任务失败
signal.job_paused      # 任务暂停（V3.7.2）
signal.pause_pending   # 暂停中（V3.7.2）
signal.job_canceled    # 任务取消
signal.job_resumed     # 任务恢复
signal.circuit_breaker # 熔断触发
signal.model_upgrade   # 模型升级
```

**字幕事件（subtitle.*）**:
```
subtitle.sv_sentence   # SenseVoice 句子生成（草稿）
subtitle.finalized     # 定稿字幕（V3.5）
subtitle.whisper_patch # Whisper 补刀完成
subtitle.llm_proof     # LLM 校对完成
subtitle.llm_trans     # LLM 翻译完成
subtitle.batch_update  # 批量更新
```

#### 5.3.4 SSE 事件数据格式

```javascript
// 进度事件
{
  "percent": 45.2,
  "phase": "sensevoice",
  "message": "正在进行 SenseVoice 推理...",
  "status": "processing"
}

// 进度事件（V3.7.1+ 双流进度）
{
  "percent": 45.2,
  "phase": "sensevoice",
  "status": "processing",
  "total": 20,
  "detail": {
    "preprocess": 100,
    "fast": 45,
    "slow": 30,
    "align": 15
  }
}

// 字幕事件
{
  "index": 0,
  "start": 1.23,
  "end": 3.45,
  "text": "转录文本",
  "confidence": 0.85,
  "warning_type": "low_transcription",
  "source": "sensevoice"
}

// 字幕定稿事件（V3.5）
{
  "index": 5,
  "sentences": [
    {
      "text": "This is a sentence",
      "start": 5.2,
      "end": 8.5,
      "confidence": 0.92,
      "is_finalized": true
    }
  ],
  "source": "sensevoice"
}
```

### 5.4 设计决策

**为什么使用命名空间？**
1. 避免冲突：不同模块的事件不会相互干扰
2. 清晰分类：前端可根据前缀快速过滤事件
3. 易于扩展：新功能可添加新的命名空间

**为什么 progress.overall 需要 detail 字段？**
- **准确性**: 后端基于实际 Chunk 处理数计算，而非字幕条数
- **独立性**: 快流和慢流处理进度独立，前端字幕数无法分别计算
- **实时性**: 后端推送真实进度，前端无需等待字幕完全生成

---

## 6. 前端实现

### 6.1 Identity

Vue 3 + Pinia 前端应用，接收 SSE 事件，实时渲染双模态字幕（草稿/定稿）、双流进度条和置信度警告。

### 6.2 Core Components

- `frontend/src/composables/useSseManager.js` (SSEManager): SSE 事件接收和分发
- `frontend/src/views/EditorView.vue` (EditorView): 主编辑界面，处理 SSE 事件
- `frontend/src/components/editor/SubtitleList/index.vue` (SubtitleList): 字幕列表渲染
- `frontend/src/stores/projectStore.js` (projectStore): Pinia 状态管理
- `frontend/src/services/api/transcriptionApi.js` (transcriptionApi): 转录 API 客户端

### 6.3 Execution Flow (LLM Retrieval Map)

#### 6.3.1 SSE 事件接收和分发

```
useSseManager 设置连接
  ↓
EventSource('/api/stream/{job_id}')
  ↓
监听事件类型:
  ├─ 'progress.*' → handleProgress()
  ├─ 'signal.*' → handleSignal()
  └─ 'subtitle.*' → handleSubtitle()
  ↓
[1] handleProgress()
    - 检查 event.detail 是否存在（V3.7.2）
    - 若存在: 调用 updateDualStreamProgressFromSSE()
    - 若不存在: 用传统逻辑计算进度
    - 更新 projectStore.progressTracker
  ↓
[2] handleSignal()
    - signal.job_complete → 任务完成
    - signal.job_failed → 显示错误提示
    - signal.job_paused → 更新 UI 状态
  ↓
[3] handleSubtitle()
    - subtitle.sv_sentence → 添加草稿字幕
    - subtitle.whisper_patch → 更新字幕（补刀）
    - subtitle.finalized → 标记为定稿
    - 在 projectStore.sentences 中保存
```

#### 6.3.2 双流进度条显示（V3.7.2）

```
EditorView.vue:onProgress()
  ↓
if (data.detail) {
    // V3.7.2 修复: 使用后端推送的真实进度
    updateDualStreamProgressFromSSE({
        fastProgress: data.detail.fast,
        slowProgress: data.detail.slow,
        alignProgress: data.detail.align
    })
} else {
    // 降级: 用字幕数计算进度（旧逻辑）
    calculateProgressFromSubtitles()
}
  ↓
UI 渲染:
┌─────────────────────────────┐
│ 总体进度: [=================] 45% │
├─────────────────────────────┤
│ 快流(SenseVoice): [==========] 45% │
│ 慢流(Whisper):     [=======] 30%    │
│ 对齐:               [==] 15%       │
└─────────────────────────────┘
```

#### 6.3.3 双模态字幕渲染（草稿/定稿）

```
subtitle.sv_sentence 事件到达
  ↓
[1] 检查是否为草稿或定稿
    if (sentence.is_finalized):
        status = "finalized"（定稿，显示为绿色）
    else:
        status = "draft"（草稿，显示为灰色）
  ↓
[2] 字幕列表项渲染（SubtitleItem.vue）
    <div :class="['subtitle-item', status]">
        <span class="text">{{ sentence.text }}</span>
        <span class="confidence" v-if="status === 'draft'">
            ⚠ {{ confidence }}
        </span>
    </div>
  ↓
[3] 样式差异
    .subtitle-item.draft {
        background: #f0f0f0;
        opacity: 0.7;
    }

    .subtitle-item.finalized {
        background: #ffffff;
        opacity: 1.0;
        border-left: 3px solid #67c23a;  # 绿色确认
    }
```

#### 6.3.4 置信度可视化

```
字幕项显示置信度警告
  ↓
if (confidence < 0.6):
    显示 ⚠ 低置信度警告
    warning_type = "low_transcription"
  ↓
if (confidence < 0.4):
    显示 ❌ 极低置信度
    warning_type = "garbage_suspect"
    hint: "可能被 Whisper 仲裁删除"
  ↓
UI 效果:
┌──────────────────────────────────┐
│ It's still only 7:28 pm  ⚠ 0.58  │
│ There's plenty of time   ✓ 0.95  │
│ to go and hunt evil      ⚠ 0.45  │
└──────────────────────────────────┘
```

#### 6.3.5 实时字幕同步

```
播放视频中
  ↓
currentTime 更新
  ↓
[1] 查找当前时间对应的字幕
    for sentence in projectStore.sentences:
        if sentence.start <= currentTime <= sentence.end:
            highlight(sentence)
  ↓
[2] 自动滚动字幕列表
    scrollIntoView(highlightedSentence)
  ↓
[3] 波形图高亮显示
    WaveformTimeline 更新 Region 高亮
```

### 6.4 关键前端修复（V3.7.2）

**问题**: 前端未使用后端推送的 `progress.overall` 事件中的 `detail` 字段，导致双流进度条卡住。

**修复方案**:
```javascript
// EditorView.vue
async onProgress(data) {
    if (data.detail) {
        // V3.7.2: 新增方法，从 SSE 事件中提取真实进度
        this.projectStore.updateDualStreamProgressFromSSE(data.detail)
    }
}

// projectStore.js
updateDualStreamProgressFromSSE(detail) {
    this.progressTracker.sensevoice = detail.fast
    this.progressTracker.whisper = detail.slow
    this.progressTracker.alignment = detail.align
    this.progressTracker.overall = detail.preprocess  // 预处理进度
}
```

### 6.5 设计决策

**为什么需要双模态字幕？**
- 草稿: FastWorker 立即输出，给用户快速反馈
- 定稿: Whisper 补刀后，最终输出，可直接使用

**为什么前端需要处理 detail 字段？**
- 后端在多个地方（FastWorker、SlowWorker、AlignmentWorker）推送进度
- 前端需要区分这些独立的进度流
- SSE 事件中的 detail 字段是唯一可靠的真实数据源

---

## 7. 完整数据流图

### 7.1 端到端数据流

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                          完整转录数据流                                    ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│ 用户上传视频                                                             │
│ ↓                                                                        │
│ [前端] TranscriptionAPI.start()                                          │
│ → POST /api/transcribe { video_path, preset_config }                    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ [后端] 预处理阶段（AudioProcessingPipeline）                             │
│                                                                          │
│ 1. 音频提取 → full_audio (16kHz, mono)                                  │
│ 2. 显存策略检查 → 决定分离策略                                          │
│ 3. 人声分离（可选）→ separated_audio                                    │
│ 4. VAD 切分 → speech_timestamps                                         │
│ 5. Smart Accumulation → AudioChunk[] (avg=12s, max=30s)               │
│ 6. 频谱分诊（可选）→ chunk.needs_separation, chunk.recommended_model  │
│                                                                          │
│ 输出: AudioChunk[] (包含 9 个新字段)                                    │
│                                                                          │
│ SSE事件: progress.preprocess, progress.extract, ...                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ [后端] 转录阶段（AsyncDualPipeline）                                    │
│                                                                          │
│ ┌─────────────────────────────────────────────────────────────────┐    │
│ │ FastWorker（CPU并发，~1s/Chunk）                               │    │
│ │  ↓                                                              │    │
│ │  SenseVoice ONNX 推理                                          │    │
│ │  + CTC 重叠去重                                               │    │
│ │  + 两层分句（VAD停顿 + 物理约束）                              │    │
│ │  + 可选熔断回溯（enable_fuse_breaker）                         │    │
│ │  + 推送草稿字幕 SSE                                           │    │
│ │  ↓                                                              │    │
│ │  SequencedQueue (内置背压)                                      │    │
│ │                                                                │    │
│ └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│ ┌─────────────────────────────────────────────────────────────────┐    │
│ │ SlowWorker（GPU顺序，Whisper）                                  │    │
│ │  ↓                                                              │    │
│ │  Whisper 推理                                                   │    │
│ │  + Whisper 仲裁机制（低置信度二次听诊）                         │    │
│ │  + 字级单字符强制补刀                                          │    │
│ │  + 推送补刀字幕 SSE                                           │    │
│ │  ↓                                                              │    │
│ │  Queue2                                                        │    │
│ │                                                                │    │
│ └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│ ┌─────────────────────────────────────────────────────────────────┐    │
│ │ AlignmentWorker（CPU）                                          │    │
│ │  ↓                                                              │    │
│ │  文本和时间戳对齐                                              │    │
│ │  + 字级时间戳计算                                              │    │
│ │  + 推送最终字幕 SSE                                           │    │
│ │  ↓                                                              │    │
│ │  最终字幕 (SentenceSegment[])                                  │    │
│ │                                                                │    │
│ └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│ SSE事件流:                                                              │
│ - progress.sensevoice (FastWorker 进度)                               │
│ - subtitle.sv_sentence (草稿字幕)                                       │
│ - progress.whisper (SlowWorker 进度)                                  │
│ - subtitle.whisper_patch (补刀字幕)                                     │
│ - progress.alignment (AlignmentWorker 进度)                            │
│ - progress.overall (统一进度，含 detail 字段)                          │
│ - signal.job_complete (任务完成)                                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ [前端] 实时字幕渲染（EditorView.vue）                                    │
│                                                                          │
│ SSE 事件处理:                                                           │
│ - progress 事件                                                         │
│   → 更新双流进度条（fast/slow/align）                                 │
│   → 检查 data.detail 字段（V3.7.2）                                   │
│                                                                          │
│ - subtitle.sv_sentence                                                  │
│   → 添加草稿字幕到列表（灰色、低透明度）                              │
│   → 显示置信度和警告图标                                              │
│                                                                          │
│ - subtitle.whisper_patch                                                │
│   → 更新对应索引的字幕文本（草稿→定稿）                              │
│   → 改变样式为绿色确认                                                │
│   → 移除警告图标                                                      │
│                                                                          │
│ UI 显示:                                                                │
│ ┌──────────────────────────────────────────┐                           │
│ │ 进度: [████████░░░░░] 45%                │                           │
│ │ ├ 预处理: [██████████░] 95%              │                           │
│ │ ├ SenseVoice: [████████░░] 45%          │                           │
│ │ ├ Whisper: [██████░░░░░] 30%            │                           │
│ │ └ 对齐: [███░░░░░░░░░] 15%              │                           │
│ └──────────────────────────────────────────┘                           │
│                                                                          │
│ 字幕列表:                                                               │
│ [灰色] It's still only 7:28 pm ⚠ 0.58                                 │
│ [绿色] There's plenty of time ✓ 0.95                                  │
│ [灰色] to go and hunt evil ⚠ 0.45                                     │
│                                                                          │
│ 波形图:                                                                 │
│ - 显示音频波形                                                          │
│ - 高亮当前播放位置                                                      │
│ - 显示字幕区间 Regions                                                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ [后端/前端] 保存和导出                                                   │
│                                                                          │
│ 1. 用户编辑字幕（可选）                                                │
│ 2. 导出 SRT 格式                                                        │
│ 3. 保存到文件系统                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 关键字段在各层的流转

```
AudioChunk 字段流转图
═════════════════════

预处理阶段:
  AudioChunk (初始)
    ├─ audio: ndarray
    ├─ start, end, duration
    └─ chunk_index
         ↓
  频谱分诊阶段:
    AudioChunk (频谱标记)
    ├─ needs_separation: bool
    ├─ recommended_model: str
    └─ spectrum_diagnosis: obj
         ↓
  人声分离阶段:
    AudioChunk (分离标记)
    ├─ original_audio: ndarray (保存原始)
    ├─ is_separated: bool
    ├─ separation_level: enum (NONE/HTDEMUCS/MDX_EXTRA)
    └─ audio: ndarray (已分离)
         ↓
  FastWorker (熔断标记):
    AudioChunk (熔断标记)
    ├─ fuse_retry_count: int
    ├─ last_confidence: float
    └─ fuse_action: enum
         ↓
  SlowWorker & AlignmentWorker:
    ProcessingContext
    ├─ chunk: AudioChunk (完整元数据)
    ├─ sv_result: SentenceSegment[]
    └─ whisper_result: SentenceSegment[]

字幕流转图
═════════

FastWorker 输出:
  SentenceSegment
    ├─ text: str
    ├─ start, end: float
    ├─ confidence: float (SenseVoice)
    ├─ is_draft: True
    ├─ source: TextSource.SENSEVOICE
    └─ words: WordTimestamp[]
         ↓
  SSE 事件: subtitle.sv_sentence
         ↓
  前端显示: 灰色草稿

SlowWorker 输出:
  SentenceSegment
    ├─ text: str (Whisper 结果，可能与 SenseVoice 不同)
    ├─ start, end: float (精确时间戳)
    ├─ confidence: float (Whisper)
    ├─ is_draft: False
    ├─ source: TextSource.WHISPER
    └─ words: WordTimestamp[]
         ↓
  SSE 事件: subtitle.whisper_patch
         ↓
  前端更新: 同索引字幕变为绿色定稿

AlignmentWorker 输出:
  最终字幕列表（已对齐）
         ↓
  导出 SRT 格式
         ↓
  用户使用
```

---

## 8. 关键设计决策

### 8.1 架构决策总结

| 决策 | 原因 | 影响 |
|------|------|------|
| **Smart Accumulation** | 利用 VAD 精确断点在源头控制 Chunk 时长 | 避免 27s+ 超长 Chunk 导致的内容丢失 |
| **乱序执行，顺序提交** | CPU 并发，GPU 顺序，平衡吞吐和准确性 | 整体性能提升 3-5 倍，无上下文丢失 |
| **SequencedQueue 背压** | 内置背压防止内存溢出，Condition 精确唤醒 | 消除惊群效应，实现真正的流量控制 |
| **YAMNet 频谱分诊** | 神经网络语义分类代替规则阈值 | 避免 98% 人声误判为音乐 |
| **FuseBreakerV2 升级路径** | max_retry=2 允许完整升级（NONE→HTDEMUCS→MDX_EXTRA） | 处理极难识别的音频场景 |
| **硬上限强制保留** | hard_limit_duration=20s，force_create=True | 避免尾部词汇丢失 |
| **Whisper 仲裁机制** | 二次听诊低置信度句子，区分幻觉vs含糊语音 | 平衡删除垃圾和保留有效内容 |
| **双模态字幕** | 草稿（快）+ 定稿（准），分离显示 | 提升用户体验，提供快速反馈 |
| **V3.7.1 双流进度** | SSE 中推送 detail.fast/slow/align | 前端可独立跟踪各层进度，无卡顿 |

### 8.2 性能优化决策

| 优化点 | 技术方案 | 数据 |
|--------|---------|------|
| **CPU 并发** | Semaphore 限制，背压控制 | FastWorker 吞吐 +300% |
| **GPU 效率** | Whisper 预加载，掩盖 27s 加载延迟 | GPU 闲置时间 从 27s → 0s |
| **显存管理** | 转码前自动卸载推理模型 | 释放 2-5GB 显存 |
| **Chunk 大小** | Smart Accumulation 从 27s → 12s | 内容丢失率 99% → 0% |
| **VAD 计算** | 一次遍历实现累积+分诊 | CPU 时间 降低 40% |

### 8.3 可靠性决策

| 决策 | 风险 | 缓解 |
|------|------|------|
| **熔断升级** | 升级分离消耗时间 | max_retry=1 默认，auto_upgrade=False |
| **Whisper 仲裁删除** | 误删有效内容 | 二次听诊，低阈值（< 0.4） |
| **Chunk 失败跳号** | 部分内容丢失 | 失败 Chunk 保留原始 SenseVoice 结果 |
| **CTC 去重** | 误删合理重复词 | 三重条件判断，词长限制 |
| **暂停/恢复** | 状态不一致 | Checkpoint 保存，Token 覆盖完整（V3.7） |

---

## 附录：术语表

| 术语 | 定义 |
|------|------|
| **AudioChunk** | 音频片段，典型时长 12 秒，包含元数据 |
| **SenseVoice** | CPU ONNX 推理，速度快（~1s），质量可接受 |
| **Whisper** | GPU 推理，速度慢（~5-10s），质量最高 |
| **FastWorker** | CPU 层 Worker，并发处理，输出草稿 |
| **SlowWorker** | GPU 层 Worker，顺序处理，输出定稿 |
| **SequencedQueue** | 乱序输入、顺序输出的队列，实现背压 |
| **Smart Accumulation** | VAD 原始输出的智能累积，控制 Chunk 时长 |
| **YAMNet** | 音频事件分类模型，用于频谱分诊 |
| **FuseBreakerV2** | 熔断决策器，自动升级分离模型 |
| **CTC 重叠去重** | 去除 "W" + "Would" → "WWouldn't" 的重复 |
| **Whisper 仲裁** | 对低置信度句子的二次听诊和判决 |
| **SSE** | Server-Sent Events，实时推送事件 |
| **ProgressEventEmitter** | V3.7.1 统一进度发射器，同步 job.progress 和推送 SSE |
| **TextSource** | 字幕来源枚举（SENSEVOICE、WHISPER、LLM） |

---

**文档完成时间**: 2025-12-26
**覆盖版本**: V3.8.0
**主要模块**: 预处理、双流架构、补刀熔断、分句对齐、SSE 推送、前端渲染
**总代码行数引用**: 50+ 个核心文件，5000+ 行代码

1. **安装CUDA和cuDNN**
   - 下载并安装 [CUDA 11.8+](https://developer.nvidia.com/cuda-11-8-0-download-archive)
   - 下载并安装 [cuDNN 8](https://developer.nvidia.com/rdp/cudnn-archive)
   - 验证安装: `nvidia-smi` 和 `nvcc --version`
2. **运行启动脚本**
