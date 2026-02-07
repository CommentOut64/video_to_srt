# V3.2.2-V3.2.3 双流模式完整架构流程图

> **Type**: Architecture Flowchart
> **Status**: Final
> **Version**: V3.2.2+dev.20260128.07
> **Last Updated**: 2026-01-28

---

## 1. 总体架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           预处理阶段（已完成）                                │
│  VAD 切分 → 频谱分诊 → 人声分离 → AudioChunk 列表（带 LangID）                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AsyncDualPipeline 双流流水线                          │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    快流（FastWorker - CPU）                        │     │
│  │  SenseVoice 推理 → PunctuationService → SemanticBuffer            │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                      ↓                                      │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    Bridge 层（句子聚合与调度）                      │     │
│  │  句子队列 → 批次构建 → Prompt 构建 → 背压控制                       │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                      ↓                                      │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    慢流（SlowWorker - GPU）                        │     │
│  │  Whisper 推理（使用 Bridge Prompt）                                │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                      ↓                                      │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    对齐与仲裁阶段                                   │     │
│  │  PunctuationArbiter → DefaultAligner → 定稿推送                    │     │
│  └───────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           字幕输出与持久化                                    │
│  StreamingSubtitleManager → SSE 推送 → 前端实时显示                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 快流（FastWorker）详细流程

### 2.1 FastWorker 核心职责

**定位**：纯 ASR 推理 + 标点恢复（NLP）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FastWorker 处理流程                                 │
└─────────────────────────────────────────────────────────────────────────────┘

输入：AudioChunk (带 language 字段)
  │
  ├─ chunk.audio: np.ndarray (16kHz 单声道)
  ├─ chunk.language: str (来自 LangID，如 "zh", "en", "ja")
  ├─ chunk.start_time: float
  └─ chunk.duration: float

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: SenseVoice ONNX 推理（ASR）                                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 输入：chunk.audio                                                        │
│  • 引擎：SenseVoiceEngine (ONNX Runtime, CPU)                              │
│  • 输出：ASRResult                                                          │
│    ├─ text: str (无标点文本，如 "今天天气很好适合出去玩")                    │
│    ├─ words: List[WordTimestamp] (词级时间戳)                              │
│    ├─ confidence: float (句级置信度)                                        │
│    └─ metadata: Dict (包含 language)                                       │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: 标点恢复（NLP - 由流水线调用）                                      │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 服务：PunctuationService.restore()                                      │
│  • 输入：                                                                   │
│    ├─ text: sv_result.text                                                 │
│    ├─ language: chunk.language                                             │
│    └─ word_timestamps: sv_result.words                                     │
│                                                                             │
│  • 语言路由（通过 PunctuationRegistry）：                                   │
│    ├─ 中文 (zh) → ChinesePunctuationStrategy                               │
│    │   ├─ WeTextProcessing (ITN 归一化)                                    │
│    │   └─ CT-Transformer ONNX (标点预测)                                   │
│    ├─ 英文 (en) → EnglishPunctuationStrategy                               │
│    │   └─ Edge-Punct-Casing ONNX (带重叠窗口)                              │
│    ├─ 日文 (ja) → JapanesePunctuationStrategy                              │
│    │   └─ Char-BERT ONNX                                                   │
│    └─ 其他 → FallbackStrategy (返回无标点结果)                             │
│                                                                             │
│  • 输出：PunctuationResult                                                  │
│    ├─ text: str (带标点文本，如 "今天天气很好，适合出去玩。")                │
│    ├─ split_points: List[SplitPoint] (建议切分点)                          │
│    │   └─ SplitPoint(char_index, relative_time, punctuation, confidence)  │
│    ├─ punctuation_positions: List[PuncPosition] (标点位置列表)             │
│    ├─ confidence: float (标点整体置信度)                                    │
│    └─ model_id: str (使用的模型 ID)                                        │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: 草稿推送（由流水线调用）                                            │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 服务：DefaultSegmenter.segment()                                        │
│  • 输入：sv_result + punct_result                                          │
│  • 输出：List[SentenceSegment] (草稿句子)                                  │
│                                                                             │
│  • SSE 推送：subtitle.draft 事件                                           │
│    └─ 前端显示灰色草稿字幕                                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

输出：FastWorkerResult
  ├─ sv_result: ASRResult (SenseVoice 原始结果)
  ├─ punct_result: PunctuationResult (标点恢复结果)
  └─ chunk: AudioChunk (原始 Chunk)

  ↓ 进入 SemanticBuffer
```

---

## 3. SemanticBuffer 语义缓冲详细流程

### 3.1 SemanticBuffer 核心职责

**定位**：跨 Chunk 边界处理 + 语义级句子切分

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SemanticBuffer 处理流程                               │
└─────────────────────────────────────────────────────────────────────────────┘

输入：FastWorkerResult (来自 FastWorker)

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: 累积带标点文本                                                      │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 将 punct_result.text 添加到内部缓冲区                                    │
│  • 保留 split_points 用于边界判断                                           │
│  • 维护 pending_tail（待定区）                                              │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: 边界判断（基于标点类型）                                            │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 强标点（。！？）→ 立即切分，输出 SemanticChunk                           │
│  • 弱标点（，、；）→ 暂存到 pending_tail                                    │
│  • 无标点 → 继续累积                                                        │
│                                                                             │
│  • 强制切分条件：                                                           │
│    ├─ pending_tail 超过 max_pending_chars (100 字符)                       │
│    ├─ pending_tail 超过 max_pending_duration (3.0 秒)                      │
│    ├─ 累积时长超过 force_flush_duration (10.0 秒)                          │
│    ├─ 检测到说话人切换                                                      │
│    └─ 检测到 VAD 长停顿 (>2s)                                              │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: 跨 Chunk 边界处理示例                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Chunk 1: "今天天气很好"                                                    │
│    → pending_tail = "今天天气很好"                                          │
│                                                                             │
│  Chunk 2: "，适合出去玩。"                                                  │
│    → 合并: "今天天气很好，适合出去玩。"                                      │
│    → 遇到强标点 "。"，输出 SemanticChunk                                    │
│    → pending_tail = ""                                                      │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

输出：SemanticChunk
  ├─ chunk_id: str (唯一标识)
  ├─ text: str (带标点的完整句子)
  ├─ sentences: List[SentenceSegment] (已确定的句子)
  ├─ pending_tail: str (尾部待定区，传递给下一个 Chunk)
  ├─ audio_range: Tuple[float, float] (音频时间范围)
  ├─ language: str (语言标识)
  └─ source_chunks: List[str] (来源 AudioChunk ID 列表)

  ↓ 进入 Bridge 层
```

---

## 4. Bridge 层详细流程

### 4.1 Bridge 核心职责

**定位**：句子聚合 + 批次构建 + Whisper Prompt 生成 + 背压控制

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Bridge 层处理流程                                   │
└─────────────────────────────────────────────────────────────────────────────┘

输入：SemanticChunk (来自 SemanticBuffer)

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: 句子队列管理（SentenceQueue）                                      │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 数据结构：asyncio.Queue(maxsize=50) - 有界队列                          │
│  • 操作：push_segment(sentence: SentenceSegment)                           │
│                                                                             │
│  • 背压控制：                                                               │
│    ├─ 队列满载时阻塞 FastWorker（await queue.put() 超时）                  │
│    ├─ 触发 backpressure_event，通知上游暂停                                │
│    └─ SlowWorker 消费后释放空间，恢复 FastWorker                           │
│                                                                             │
│  • 丢帧策略（可选，直播场景）：                                             │
│    └─ enable_frame_drop=True 时，丢弃最旧的非边界句子                      │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: 批次触发条件判断                                                   │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 时间长度：累积音频时长达到 20-30s                                        │
│  • 句子数量：累积句子数达到 6-10 句                                         │
│  • 长停顿：检测到 VAD 停顿 >2s                                              │
│  • 说话人切换：检测到 speaker_id 变化                                       │
│  • 饥饿超时：5s 内无新数据强制 flush                                        │
│                                                                             │
│  • 动态阈值（V3.2.3 完整版）：                                              │
│    └─ 根据音频特征（语速、停顿分布）动态调整批次大小                        │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: 批次构建（BatchBuilder）                                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 聚合句子：将队列中的句子聚合为一个批次                                   │
│  • 音频拼接：使用 WhisperBufferPool 拼接音频片段                           │
│  • 重叠音频：保留 1-2s 重叠音频用于上下文连贯                               │
│                                                                             │
│  • 输出：BridgeBatch                                                        │
│    ├─ batch_id: str (批次唯一标识)                                         │
│    ├─ sentences: List[SentenceSegment] (句子列表)                          │
│    ├─ audio_segments: List[AudioSegment] (音频片段)                        │
│    ├─ total_duration: float (总时长)                                       │
│    ├─ language: str (主要语言)                                             │
│    ├─ speaker_id: Optional[str] (说话人 ID)                                │
│    ├─ prompt: str (预构建的 Whisper Prompt)                                │
│    ├─ overlap_audio: Optional[np.ndarray] (重叠音频)                       │
│    └─ flush_reason: str (触发原因)                                         │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 4: Whisper Prompt 构建（PromptBuilder）                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 策略：滑动窗口（取前一批次的最后 2 句）                                  │
│                                                                             │
│  • 示例：                                                                   │
│    前一批次最后 2 句: "今天天气很好。适合出去玩。"                          │
│    当前批次 Prompt: "今天天气很好。适合出去玩。"                            │
│                                                                             │
│  • 作用：                                                                   │
│    ├─ 提供上下文，提高 Whisper 识别准确率                                  │
│    ├─ 保持语义连贯性                                                       │
│    └─ 减少幻觉（Whisper 倾向于延续 Prompt 的风格）                         │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

输出：BridgeBatch
  ↓ 进入 SlowWorker
```

---

## 5. 慢流（SlowWorker）详细流程

### 5.1 SlowWorker 核心职责

**定位**：纯 ASR 推理（Whisper）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SlowWorker 处理流程                                   │
└─────────────────────────────────────────────────────────────────────────────┘

输入：BridgeBatch (来自 Bridge 层)

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: 音频拼接（WhisperBufferPool）                                      │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 拼接批次中的所有音频片段                                                 │
│  • 添加重叠音频（overlap_audio）用于上下文连贯                             │
│  • 输出：完整的音频数组 (np.ndarray)                                        │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: Whisper 推理（ASR）                                                │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 引擎：WhisperEngine (Faster-Whisper, GPU)                               │
│  • 输入：                                                                   │
│    ├─ audio: np.ndarray (拼接后的音频)                                     │
│    ├─ prompt: str (来自 Bridge 的预构建 Prompt)                            │
│    └─ language: str (批次主要语言)                                         │
│                                                                             │
│  • Whisper 推理特点：                                                       │
│    ├─ 自回归模型，逐词生成                                                  │
│    ├─ 内置标点能力（Whisper V3）                                           │
│    ├─ 使用 Prompt 提供上下文，减少幻觉                                     │
│    └─ 输出词级时间戳（精度高于 SenseVoice）                                │
│                                                                             │
│  • 输出：WhisperResult                                                      │
│    ├─ text: str (带标点文本，如 "今天天气很好，适合出去玩。")               │
│    ├─ segments: List[Segment] (句子片段)                                   │
│    ├─ words: List[WordTimestamp] (词级时间戳)                              │
│    ├─ confidence: float (句级置信度)                                        │
│    └─ metadata: Dict (包含 language, prompt_used 等)                       │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: 结果封装                                                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 输出：SlowWorkerResult                                                   │
│    ├─ batch_id: str (对应的 BridgeBatch ID)                                │
│    ├─ whisper_result: WhisperResult (Whisper 推理结果)                     │
│    └─ source_sentences: List[SentenceSegment] (来自 FastWorker 的原始句子) │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

输出：SlowWorkerResult
  ↓ 进入对齐与仲裁阶段
```

**关键说明**：
- **SlowWorker 不做标点处理**：Whisper 输出已包含标点，但这是 Whisper 模型的副产品
- **SlowWorker 不做分句**：分句由 PunctuationArbiter 和 DefaultAligner 完成
- **SlowWorker 不做对齐**：对齐由 DefaultAligner 完成
- **SlowWorker 职责单一**：仅负责 Whisper ASR 推理

---

## 6. PunctuationArbiter 仲裁机制详细流程

### 6.1 PunctuationArbiter 核心职责

**定位**：融合快慢流标点结果 + 幻觉过滤 + 置信度仲裁（NLP）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PunctuationArbiter 仲裁流程                             │
└─────────────────────────────────────────────────────────────────────────────┘

输入：SlowWorkerResult (包含 FastWorker 和 Whisper 的结果)

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: 数据准备                                                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • FastWorker 结果（来自 source_sentences）：                               │
│    ├─ SenseVoice 文本（无标点）                                             │
│    ├─ CT-Transformer 标点位置（split_points）                              │
│    ├─ 词级时间戳（word_timestamps）                                        │
│    └─ VAD 停顿信息                                                          │
│                                                                             │
│  • Whisper 结果（来自 whisper_result）：                                   │
│    ├─ Whisper 文本（带标点）                                                │
│    ├─ Whisper 词级时间戳                                                    │
│    └─ Whisper 置信度                                                        │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: 锚点对齐（避免全文 DTW）                                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 锚点选择策略：                                                           │
│    ├─ 句首词（两者应该一致）                                                │
│    ├─ 高置信度词（SenseVoice 置信度 > 0.9）                                │
│    └─ 数字、专有名词（不易变形）                                            │
│                                                                             │
│  • 局部对齐：                                                               │
│    └─ 在锚点之间做局部对齐，比全文对齐快 10 倍以上                         │
│                                                                             │
│  • 输出：List[AnchorPoint]                                                  │
│    └─ AnchorPoint(sv_index, whisper_index, confidence)                     │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: 标点映射（Whisper → SenseVoice 时间轴）                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 通过锚点对齐，将 Whisper 的标点位置映射到 SenseVoice 时间轴              │
│  • 输出：List[MappedPunctuation]                                            │
│    └─ MappedPunctuation(char_index, punctuation, timestamp, confidence)    │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 4: 幻觉标点过滤（关键）                                               │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 过滤规则：                                                               │
│    ├─ 规则 1：句末标点必须在 VAD 停顿点附近（±0.3s）                       │
│    ├─ 规则 2：句末标点之间最小间隔 > 1.5s（避免连续句号）                  │
│    ├─ 规则 3：CT-Transformer 同位置预测 → 置信度 +1                        │
│    └─ 规则 4：Whisper 置信度 < 0.7 且 CT 无预测 → 丢弃                     │
│                                                                             │
│  • 仲裁决策矩阵：                                                           │
│    ┌──────────────┬─────────┬──────────┬──────────────────┐               │
│    │ CT-Transform │ Whisper │ VAD 停顿 │ 决策             │               │
│    ├──────────────┼─────────┼──────────┼──────────────────┤               │
│    │ 句末         │ 句末    │ 有       │ 高置信度采纳     │               │
│    │ 句末         │ 句末    │ 无       │ 采纳但标记语义   │               │
│    │ 无           │ 句末    │ 有       │ 采纳             │               │
│    │ 无           │ 句末    │ 无       │ 降级为逗号或丢弃 │               │
│    │ 句末         │ 无      │ 有       │ 采纳但标记声学   │               │
│    │ 无           │ 无      │ -        │ 不加标点         │               │
│    └──────────────┴─────────┴──────────┴──────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 5: 仲裁结果生成                                                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 输出：ArbitrationResult                                                  │
│    ├─ final_sentences: List[SentenceSegment] (仲裁后的句子列表)            │
│    ├─ confidence_score: float (仲裁置信度)                                  │
│    └─ arbitration_details: Dict (仲裁详情)                                 │
│                                                                             │
│  • 降级策略：                                                               │
│    └─ if confidence_score < 0.6:                                           │
│        → 回退到纯 SenseVoice 结果（保守策略）                               │
│    └─ else:                                                                │
│        → 采用仲裁结果                                                       │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

输出：ArbitrationResult
  ↓ 进入 DefaultAligner
```

---

## 7. DefaultAligner 对齐阶段详细流程

### 7.1 DefaultAligner 核心职责

**定位**：双流结果对齐 + 降级策略 + 定稿推送

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DefaultAligner 对齐流程                               │
└─────────────────────────────────────────────────────────────────────────────┘

输入：ArbitrationResult (来自 PunctuationArbiter)

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: 双流结果对齐                                                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 对齐策略：                                                               │
│    ├─ 优先使用 Whisper 文本（内容准确）                                    │
│    ├─ 使用 SenseVoice 时间戳（时间准确）                                   │
│    └─ 使用仲裁后的标点（置信度高）                                         │
│                                                                             │
│  • 降级策略：                                                               │
│    ├─ 仲裁置信度 < 0.6 → 回退到纯 SenseVoice 结果                          │
│    ├─ Whisper 幻觉检测 → 使用 SenseVoice 结果                              │
│    └─ 对齐失败 → 使用 DefaultSegmenter 分句                                │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: 最终分句                                                           │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 基于仲裁后的标点位置进行最终分句                                         │
│  • 输出：List[SentenceSegment] (定稿句子)                                  │
│    └─ SentenceSegment(text, start_time, end_time, confidence, words)       │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: 定稿推送（由流水线调用 StreamingSubtitleManager）                 │
│  ─────────────────────────────────────────────────────────────────────────  │
│  • 服务：StreamingSubtitleManager.add_finalized_sentences()                │
│  • SSE 推送：subtitle.finalized 事件                                       │
│    └─ 前端显示白色定稿字幕，覆盖草稿                                        │
│                                                                             │
│  • 字幕快照保存：                                                           │
│    └─ 保存到 Checkpoint，用于暂停恢复                                      │
└─────────────────────────────────────────────────────────────────────────────┘

  ↓

输出：List[SentenceSegment] (最终定稿)
  ↓ 完成
```

---

## 8. 数据流总结

### 8.1 完整数据流图

```
AudioChunk (带 LangID)
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ FastWorker (CPU)                                                            │
│   SenseVoice → ASRResult (无标点)                                           │
│        ↓                                                                    │
│   PunctuationService (按 LangID 路由)                                       │
│        ├─ 中文: WeTextProcessing + CT-Transformer                          │
│        ├─ 英文: Edge-Punct-Casing                                          │
│        └─ 日文: Char-BERT                                                  │
│        ↓                                                                    │
│   PunctuationResult (带标点 + split_points)                                │
│        ↓                                                                    │
│   DefaultSegmenter → 草稿句子 → SSE 推送                                   │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ SemanticBuffer                                                              │
│   累积带标点文本 → 边界判断 → SemanticChunk                                │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ Bridge 层                                                                   │
│   句子队列 → 批次构建 → Prompt 构建 → BridgeBatch                          │
│   (背压控制：队列满载时阻塞 FastWorker)                                     │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ SlowWorker (GPU)                                                            │
│   音频拼接 → Whisper 推理 (使用 Prompt) → WhisperResult (带标点)           │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PunctuationArbiter (NLP)                                                    │
│   锚点对齐 → 标点映射 → 幻觉过滤 → ArbitrationResult                       │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ DefaultAligner                                                              │
│   双流对齐 → 最终分句 → 定稿推送 → SSE 推送                                │
└─────────────────────────────────────────────────────────────────────────────┘
    ↓
StreamingSubtitleManager → 前端实时显示
```

### 8.2 ASR vs NLP 职责划分

| 组件 | 定位 | 输入 | 输出 | 职责 |
|------|------|------|------|------|
| **FastWorker** | ASR + NLP | AudioChunk | ASRResult + PunctuationResult | SenseVoice 推理 + 标点恢复 |
| **SemanticBuffer** | NLP | FastWorkerResult | SemanticChunk | 跨 Chunk 边界处理 + 语义切分 |
| **Bridge** | 调度 | SemanticChunk | BridgeBatch | 句子聚合 + 批次构建 + Prompt 生成 |
| **SlowWorker** | **纯 ASR** | BridgeBatch | WhisperResult | Whisper 推理（不做标点/分句/对齐） |
| **PunctuationArbiter** | **NLP** | SlowWorkerResult | ArbitrationResult | 标点仲裁 + 幻觉过滤 |
| **DefaultAligner** | NLP | ArbitrationResult | List[SentenceSegment] | 双流对齐 + 最终分句 |

**关键结论**：
- **慢流（SlowWorker）是纯 ASR**，不做任何 NLP 处理
- **NLP 任务分布在三个位置**：FastWorker（标点恢复）、PunctuationArbiter（仲裁）、DefaultAligner（对齐）

---

## 9. 关键数据结构定义

### 9.1 AudioChunk（输入）

```python
@dataclass
class AudioChunk:
    """音频切片（VAD 切分后）"""
    chunk_id: str                          # 唯一标识
    audio: np.ndarray                      # 音频数据 (16kHz 单声道)
    start_time: float                      # 起始时间（秒）
    duration: float                        # 时长（秒）
    language: str                          # 语言标识（来自 LangID，如 "zh", "en", "ja"）
    speaker_id: Optional[str]              # 说话人 ID（可选）
    separation_level: SeparationLevel      # 分离级别（NONE/HTDEMUCS/MDX_EXTRA）
```

### 9.2 ASRResult（SenseVoice 输出）

```python
@dataclass
class ASRResult:
    """ASR 推理结果（统一接口）"""
    text: str                              # 转录文本（无标点）
    segments: List[Segment]                # 句子片段
    words: List[WordTimestamp]             # 词级时间戳
    confidence: float                      # 句级置信度
    metadata: Dict[str, Any]               # 元数据（包含 language）
```

### 9.3 PunctuationResult（标点恢复输出）

```python
@dataclass
class PunctuationResult:
    """标点恢复结果"""
    text: str                              # 带标点文本
    split_points: List[SplitPoint]         # 建议切分点
    punctuation_positions: List[PuncPosition]  # 标点位置列表
    confidence: float                      # 整体置信度
    model_id: str                          # 使用的模型 ID
    processing_time_ms: float              # 处理耗时

@dataclass
class SplitPoint:
    """切分点"""
    char_index: int                        # 字符索引
    relative_time: float                   # 相对时间戳
    punctuation: str                       # 触发切分的标点
    confidence: float                      # 切分置信度
```

### 9.4 SemanticChunk（SemanticBuffer 输出）

```python
@dataclass
class SemanticChunk:
    """语义切分后的 Chunk"""
    chunk_id: str                          # 唯一标识
    text: str                              # 带标点文本
    sentences: List[SentenceSegment]       # 已确定的句子
    pending_tail: str                      # 尾部待定区
    audio_range: Tuple[float, float]       # 音频时间范围
    language: str                          # 语言标识
    source_chunks: List[str]               # 来源 AudioChunk ID 列表
```

### 9.5 BridgeBatch（Bridge 输出）

```python
@dataclass
class BridgeBatch:
    """Bridge 输出的批次"""
    batch_id: str                          # 批次唯一标识
    sentences: List[SentenceSegment]       # 句子列表
    audio_segments: List[AudioSegment]     # 音频片段
    total_duration: float                  # 总时长
    language: str                          # 主要语言
    speaker_id: Optional[str]              # 说话人 ID
    prompt: str                            # 预构建的 Whisper Prompt
    overlap_audio: Optional[np.ndarray]    # 1-2s 重叠音频
    flush_reason: str                      # 触发原因
```

### 9.6 WhisperResult（Whisper 输出）

```python
@dataclass
class WhisperResult:
    """Whisper 推理结果"""
    text: str                              # 带标点文本
    segments: List[Segment]                # 句子片段
    words: List[WordTimestamp]             # 词级时间戳
    confidence: float                      # 句级置信度
    metadata: Dict[str, Any]               # 元数据（包含 language, prompt_used）
```

### 9.7 ArbitrationResult（仲裁输出）

```python
@dataclass
class ArbitrationResult:
    """标点仲裁结果"""
    final_sentences: List[SentenceSegment] # 仲裁后的句子列表
    confidence_score: float                # 仲裁置信度
    arbitration_details: Dict[str, Any]    # 仲裁详情
        # 包含：
        # - anchor_points: List[AnchorPoint]
        # - filtered_punctuations: List[PuncPosition]
        # - fallback_reason: Optional[str]
```

---

## 10. 性能指标与瓶颈分析

### 10.1 各阶段性能预估

| 阶段 | 组件 | 设备 | 延迟 | 内存占用 | 瓶颈风险 |
|------|------|------|------|----------|----------|
| 快流推理 | SenseVoice ONNX | CPU | ~1s/Chunk | ~500MB | 低 |
| 标点恢复 | CT-Transformer ONNX | CPU | <20ms/句 | ~150MB | 低 |
| 语义缓冲 | SemanticBuffer | CPU | <5ms | ~10MB | 极低 |
| Bridge 聚合 | BridgeController | CPU | <10ms | ~50MB | 低 |
| 慢流推理 | Whisper GPU | GPU | 3-5s/批次 | ~2GB | 中 |
| 标点仲裁 | PunctuationArbiter | CPU | <50ms | ~20MB | 低 |
| 对齐阶段 | DefaultAligner | CPU | <30ms | ~10MB | 低 |

**总体影响**：
- 快流增加延迟：+2% (~20ms)
- 慢流增加延迟：+1% (~50ms)
- 内存增加：~200MB（标点模型）
- **不会成为流水线瓶颈**

### 10.2 背压控制关键参数

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| queue_maxsize | 50 | Bridge 队列最大容量 | CPU 强劲可增至 100 |
| backpressure_timeout | 5.0s | 背压等待超时 | 直播场景降至 2.0s |
| max_pending_chars | 100 | SemanticBuffer 待定区字符数 | 中文可降至 50 |
| max_pending_duration | 3.0s | SemanticBuffer 待定区时长 | 快速语音降至 2.0s |
| force_flush_duration | 10.0s | 强制刷新时长 | 直播场景降至 5.0s |

---

## 11. 关键设计决策

### 11.1 为什么 FastWorker 做标点恢复？

**决策**：标点恢复在 FastWorker 之后、流水线层调用

**理由**：
1. **职责单一**：FastWorker 仅负责 ASR 推理，标点恢复由流水线统一调度
2. **复用性**：标点服务可被其他组件复用（如后处理、编辑器）
3. **可测试性**：标点逻辑独立，便于单元测试和模型替换
4. **性能优化**：标点模型在 CPU 上推理，不占用 GPU 资源

### 11.2 为什么 SlowWorker 不做标点处理？

**决策**：SlowWorker 仅负责 Whisper ASR 推理

**理由**：
1. **Whisper 内置标点**：Whisper V3 自带标点能力，无需额外处理
2. **职责单一**：SlowWorker 专注于 ASR，标点仲裁由 PunctuationArbiter 完成
3. **架构清晰**：ASR 和 NLP 职责分离，便于维护和扩展
4. **性能考虑**：避免在 GPU 推理后再做 CPU 标点处理，减少延迟

### 11.3 为什么需要 Bridge 层？

**决策**：在快慢流之间引入 Bridge 层

**理由**：
1. **句子聚合**：将快流的碎片化句子聚合为批次，提高慢流效率
2. **Prompt 构建**：为 Whisper 提供上下文 Prompt，减少幻觉
3. **背压控制**：防止快流过快导致慢流积压，实现流量控制
4. **时序解耦**：快流和慢流独立运行，通过 Bridge 协调

### 11.4 为什么需要 PunctuationArbiter？

**决策**：引入标点仲裁机制

**理由**：
1. **三重保障**：FastWorker（CT-Transformer）+ Whisper + VAD 停顿，交叉验证
2. **幻觉过滤**：Whisper 可能产生幻觉标点，需要通过 VAD 和 CT-Transformer 过滤
3. **置信度融合**：综合快慢流的置信度，输出最可靠的标点结果
4. **降级策略**：低置信度时回退到 SenseVoice 结果，保证鲁棒性

---

## 12. 相关文档

### 12.1 架构文档

- [AsyncDualPipeline 乱序执行架构](../../llmdoc/architecture/async-dual-pipeline.md)
- [FastWorker 快流推理](../../llmdoc/architecture/fast-worker.md)
- [标点符号模型架构](../../llmdoc/architecture/punctuation-model-architecture.md)
- [V3.2.2-V3.2.3 实施计划](v3.2.2-3-implementation-plan.md)

### 12.2 参考文档

- [标点符号配置参考](../../llmdoc/reference/punctuation-config.md)
- [转录 API 参考](../../llmdoc/reference/transcription-api.md)

---

## 13. 总结

### 13.1 核心要点

1. **双流职责清晰**：
   - **FastWorker**：ASR (SenseVoice) + NLP (标点恢复)
   - **SlowWorker**：纯 ASR (Whisper)
   - **PunctuationArbiter**：NLP (标点仲裁)

2. **数据流单向**：
   ```
   AudioChunk → FastWorker → SemanticBuffer → Bridge → SlowWorker → Arbiter → Aligner → 定稿
   ```

3. **标点处理三层**：
   - **Layer 1**：FastWorker 的 CT-Transformer（快速、保守）
   - **Layer 2**：Whisper 内置标点（准确但可能幻觉）
   - **Layer 3**：PunctuationArbiter 仲裁（取交集、过滤幻觉）

4. **背压控制**：
   - Bridge 队列有界（maxsize=50）
   - 队列满载时阻塞 FastWorker
   - 实现快慢流流量平衡

5. **降级策略**：
   - 标点模型不可用 → DefaultSegmenter
   - 仲裁置信度低 → 纯 SenseVoice 结果
   - Whisper 幻觉检测 → SenseVoice 结果

### 13.2 性能预期

- **快流延迟增加**：+2% (~20ms)
- **慢流延迟增加**：+1% (~50ms)
- **内存增加**：~200MB（标点模型）
- **不会成为瓶颈**：标点模型在 CPU 上推理，不占用 GPU

### 13.3 实施优先级

**V3.2.2**（基础设施）：
- Phase A: 标点模型基础设施
- Phase B: FastWorker 标点集成
- Phase C: SemanticBuffer 语义缓冲
- Phase D: 测试验证 + Bridge MVP

**V3.2.3**（完整实现）：
- Phase E: Bridge 完整实现（背压控制）
- Phase F: SlowWorker 时序升级
- Phase G: PunctuationArbiter 仲裁机制
- Phase H: 端到端集成测试

---

**文档版本**：V3.2.2+dev.20260128.07
**最后更新**：2026-01-28
**作者**：Claude (wgh)

