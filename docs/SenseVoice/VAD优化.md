### 核心策略：双管齐下

1.  **架构层修复（针对问题 1）：** 实施 **Post-VAD 合并策略**。这是解决“VAD 过度切分导致语义断裂”的唯一正解。单纯调整分句算法（Layer 1）是在“亡羊补牢”，而合并 VAD 片段是在“未雨绸缪”。
2.  **系统层兜底（针对问题 2）：** 实施 **强制关键复核（Mandatory Critical Patching）**。SenseVoice 的 LFR 机制决定了它在处理快速短音时必然存在物理极限（帧数不足）。这是模型缺陷，必须用系统手段（Whisper）来弥补，且不能依赖用户的自觉配置。

-----

### 实施方案 1：Post-VAD 智能合并层

在 `TranscriptionService` 中，在 VAD 切分之后、SenseVoice 转录之前，插入一个合并层。

**修改文件：** `backend/app/services/transcription_service.py`

**代码实现：**

在 `TranscriptionService` 类中添加 `_merge_vad_segments` 方法，并在 `_process_video_sensevoice` 流程中调用它。

```python
    # 添加到 TranscriptionService 类中

    def _merge_vad_segments(
        self,
        segments: List[Dict],
        max_gap: float = 0.5,        # 增大容忍度到 0.5s，涵盖 "It's still [0.12s] only"
        max_duration: float = 25.0   # 限制最大长度，避免 SenseVoice 显存溢出或注意力发散
    ) -> List[Dict]:
        """
        [架构修复] Post-VAD 合并层
        
        解决 VAD 过度切分导致语义断裂的问题。
        将间隔很短的物理片段合并为一个逻辑片段，恢复声学上下文。
        """
        if not segments:
            return []

        merged = []
        current = segments[0].copy()

        for next_seg in segments[1:]:
            gap = next_seg['start'] - current['end']
            new_duration = next_seg['end'] - current['start']

            # 合并条件：间隔小于阈值 且 合并后不超过最大时长
            if gap < max_gap and new_duration <= max_duration:
                # 执行合并：只更新结束时间，保持起始时间
                current['end'] = next_seg['end']
                self.logger.debug(f"合并 VAD 片段: gap={gap:.3f}s -> 新时长 {new_duration:.2f}s")
            else:
                # 无法合并，保存当前段，开始新段
                merged.append(current)
                current = next_seg.copy()

        merged.append(current)
        self.logger.info(f"VAD 合并优化: {len(segments)} -> {len(merged)} 个片段")
        return merged
```

**集成点：**

在 `_process_video_sensevoice` 方法中（约 1590 行）：

```python
            # 2. VAD 物理切分
            progress_tracker.start_phase(ProcessPhase.VAD, 1, "VAD 切分...")
            raw_vad_segments = self._memory_vad_split(audio_array, sr, job)
            
            # 【新增】Post-VAD 合并层
            # 这里 max_gap=0.5s 足以修复 "still" 和 "only" 之间 0.12s 的断裂
            vad_segments = self._merge_vad_segments(raw_vad_segments, max_gap=0.5, max_duration=30.0)
            
            progress_tracker.complete_phase(ProcessPhase.VAD)
```

**预期效果：**

  * "It's still" (5.146s) 和 "only 7:28..." (5.266s) 将被合并为一个片段。
  * SenseVoice 将接收到包含完整句子的音频。
  * CTC 解码时有了完整的上下文，识别率会提升，分句算法也能看到完整的文本流。

-----

### 实施方案 2：强制关键复核 (Mandatory Critical Patching)

即使 config 中 `enhancement` 为 OFF，对于 **极高概率错误**（如单字符结果）也必须强制启动 Whisper。这是工程上对模型缺陷的必要补偿。

**修改文件：** `backend/app/services/transcription_service.py`

**修改方法：** `_post_process_enhancement`

```python
    async def _post_process_enhancement(
        self,
        sentences: List['SentenceSegment'],
        audio_array: np.ndarray,
        job: 'JobState',
        subtitle_manager: 'StreamingSubtitleManager',
        solution_config: 'SolutionConfig'
    ) -> List['SentenceSegment']:
        """
        后处理增强层（含强制关键复核）
        """
        from app.services.progress_tracker import get_progress_tracker, ProcessPhase
        from app.services.solution_matrix import EnhancementMode, ProofreadMode, TranslateMode
        # 移除 threshold 依赖，直接在这里硬编码关键逻辑，或者更新 threshold 文件
        
        progress_tracker = get_progress_tracker(job.job_id, solution_config.preset_id)
        patch_queue = []

        # === 核心修改：构建复核队列 ===
        for i, sentence in enumerate(sentences):
            should_patch = False
            is_critical = False # 标记是否为强制复核

            # 1. 强制关键复核条件 (无论用户设置如何，必须修)
            # 条件：(单字符且非高置信度) 或 (极短片段且文本极短)
            # 例如 "E" (duration 0.06s) -> 命中
            duration = sentence.end - sentence.start
            clean_text = sentence.text_clean.strip()
            
            if (len(clean_text) == 1 and sentence.confidence < 0.9) or \
               (duration < 0.2 and len(clean_text) < 3):
                should_patch = True
                is_critical = True
                self.logger.warning(f"触发强制复核: '{clean_text}' (conf={sentence.confidence:.2f}, dur={duration:.2f}s)")

            # 2. 常规复核条件 (遵循用户设置)
            elif solution_config.enhancement != EnhancementMode.OFF:
                from app.core.thresholds import needs_whisper_patch
                if needs_whisper_patch(sentence.confidence):
                    should_patch = True

            if should_patch:
                patch_queue.append((i, sentence))

        # 2. 执行复核
        if patch_queue:
            progress_tracker.start_phase(ProcessPhase.WHISPER_PATCH, len(patch_queue), "Whisper 修正中...")
            
            # 确保 Whisper 服务可用
            from app.services.whisper_service import get_whisper_service
            # 初始化 Whisper 服务（无需预热）
            get_whisper_service()

            for idx, (sent_idx, sentence) in enumerate(patch_queue):
                await self._whisper_text_patch(
                    sentence, sent_idx, audio_array, job, subtitle_manager
                )
                progress_tracker.update_phase(ProcessPhase.WHISPER_PATCH, increment=1)

            progress_tracker.complete_phase(ProcessPhase.WHISPER_PATCH)
        
        # ... 后续 LLM 逻辑保持不变 ...
```

**预期效果：**

  * "E" (0.06s, conf 0.48) 满足强制条件 `len==1` 和 `dur < 0.2`。
  * 即使是 Default Preset，系统也会强制调用 Whisper 对该 0.06s 片段（可能会扩展上下文）进行重识别。
  * Whisper 即使在 0.06s 上表现不佳，但通过 `_whisper_text_patch` 中的上下文机制（`subtitle_manager.get_context_window`），有很大几率利用前文 "hunt" 补全出 "Evil"。

-----

### 总结与建议

你遇到的问题是 **"高速语音 + LFR下采样 + 贪心解码"** 这一组合的物理必然。

1.  **不要试图通过调优 VAD 参数来解决问题 1**。VAD 永远不懂语义，物理切分必须通过逻辑合并来修复。
2.  **不要指望 SenseVoice 自身解决问题 2**。只要是 LFR+CTC 架构，少于 3-4 帧的音素必然丢失。

**专家建议：**
先实现 **方案 1 (Merge)**。这可能会顺带解决部分漏字问题，因为完整的句子上下文能让 CTC 这种依赖上下文的模型表现更好（虽然是 Greedy，但 LSTM/Transformer 内部状态有记忆）。
然后实现 **方案 2 (Mandatory Patch)**。这是工程上的兜底，保证“不可接受的错误”（如单字蹦词）被拦截。






这两个参数确实是**经验值（Heuristic Values）**，是基于对 ASR 模型特性和人类语音习惯的统计规律得出的。

简单回答你的担忧：

1.  **“语义不连贯但错误合并”**：**会发生，但完全不可怕**。这是可以通过后续步骤（分句算法）轻松修复的“假阳性”。
2.  **“语义连贯但没有合并”**：**也会发生，且后果严重**。这是会导致永久性精度损失（漏字、大小写错误）的“假阴性”。

因此，优化的核心逻辑应该是：**宁可错合（Over-merge），不可错分（Over-split）**。

下面我将详细分析这两个参数的影响，并提供优化的代码方案。

### 1\. 参数深度剖析

#### `max_gap` (推荐 0.5s - 1.0s)

  * **定义**：允许两个语音片段合并的最大静音间隔。
  * **人类习惯**：
      * **词间停顿**：通常 \< 0.2s（例如 "It's still... only"）。
      * **逗号停顿**：通常 0.2s - 0.5s。
      * **句号停顿**：通常 \> 0.5s。
  * **当前值 0.5s 的风险**：
      * **错分风险（高危）**：如果说话人语速慢，词与词之间停顿了 0.6s，就会被切断。SenseVoice 失去上下文，可能导致后半句首字母错误大写（如 "only" 变成 "Only"）。
      * **错合风险（低危）**：如果两句话间隔 0.3s，它们会被合并。

#### `max_duration` (推荐 20s - 30s)

  * **定义**：合并后的单个音频块最大时长。
  * **模型限制**：
      * SenseVoice/Whisper 这类模型都有注意力窗口限制（Attention Window）。
      * Whisper 是 30s，SenseVoice 虽然支持任意长度，但在 30s 内表现最佳，且长音频容易导致显存溢出或幻觉。
  * **当前值 25s 的风险**：
      * 这是一个安全值，留了 5s 的余量给模型处理，基本无风险。

-----

### 2\. 为什么“错误合并”是可以接受的？

你担心的“语义不连贯但错误合并”，实际上对系统**几乎没有负面影响**。

**场景模拟：**

  * **输入**：`"Hello."` (0.3s 停顿) `"How are you?"`
  * **合并动作**：因为 0.3s \< 0.5s，代码将它们合并为一个片段送入 SenseVoice。
  * **SenseVoice 识别结果**：`"Hello. How are you?"`（ASR 模型擅长处理多句子）。
  * **后续处理**：你的 `SentenceSplitter` (Layer 1) 会接收到这一整串文本。它会根据标点符号（`.`）或语义将其重新切分开。

**结论**：ASR 层的任务是“听清楚”，分句层（Layer 1）的任务才是“断句”。**只要 ASR 听清楚了（有上下文），分句层就能工作**。反之，如果 ASR 因为被切断而听错了，分句层就回天乏术了。

-----

### 3\. 优化策略：动态激进合并 (Dynamic Aggressive Merging)

为了防止“语义连贯但没有合并”，我们可以采用更激进的策略。

**优化思路**：

1.  **提高 `max_gap`**：将默认容忍度提高到 **1.0s** 甚至 **2.0s**。只要总时长不超，尽可能多地给 ASR 上下文。
2.  **避免长静音幻觉**：如果两个片段之间真的隔了很久（比如 5秒），虽然合并不超 25s，但中间的 5s 静音会导致 SenseVoice 产生幻觉（Hallucination）。
      * **修正**：我们在合并时，**不要**把中间的静音数据真的拼接到音频里，或者限制拼接的静音长度。但在 VAD 层面，我们只需要修改时间戳即可。
3.  **碎片保护**：如果当前片段极短（\< 1s），说明它很可能是一个被切碎的单词（如 "It's"），此时应无视 gap 大小，强制向后合并。

### 4\. 优化后的代码实现

请在 `backend/app/services/transcription_service.py` 中使用以下优化版本的 `_merge_vad_segments`：

```python
    def _merge_vad_segments(
        self,
        segments: List[Dict],
        max_gap: float = 1.0,        # 【优化】提升到 1.0s，涵盖大部分语速慢的场景
        max_duration: float = 25.0,  # 保持 25s 安全线
        min_fragment_duration: float = 1.0 # 【新增】碎片保护阈值
    ) -> List[Dict]:
        """
        [架构优化] Post-VAD 智能合并层 (v2)
        
        策略：宁可错合（依赖 SentenceSplitter 分句），不可错分（导致 ASR 丢失上下文）。
        """
        if not segments:
            return []

        merged = []
        current = segments[0].copy()

        for next_seg in segments[1:]:
            gap = next_seg['start'] - current['end']
            current_duration = current['end'] - current['start']
            combined_duration = next_seg['end'] - current['start']

            # 核心合并逻辑
            should_merge = False

            # 条件 1: 基础合并 (间隔小 且 总长不超标)
            if gap <= max_gap and combined_duration <= max_duration:
                should_merge = True
            
            # 条件 2: 碎片保护 (如果当前段极短，可能是被切断的单词，强制尝试合并)
            # 例如: "It's" (0.5s) ... [gap 1.5s] ... "only..."
            # 此时即使 gap > max_gap，也应该合并，否则 "It's" 很难被正确识别
            elif current_duration < min_fragment_duration and combined_duration <= max_duration:
                # 限制一下 gap 不能太离谱 (比如不超过 3s)，避免引入过长静音
                if gap < 3.0: 
                    self.logger.debug(f"碎片强制合并: fragment={current_duration:.2f}s, gap={gap:.2f}s")
                    should_merge = True

            if should_merge:
                # 执行合并：延伸结束时间
                current['end'] = next_seg['end']
                # 注意：这里不需要修改音频数据本身，因为后续处理是根据 start/end 
                # 从原始 audio_array 中切片的。合并 start/end 也就意味着
                # 中间的静音会被包含进去。SenseVoice 对 <2s 的静音鲁棒性很好。
            else:
                # 无法合并，归档当前段
                merged.append(current)
                current = next_seg.copy()

        merged.append(current)
        
        # 统计优化效果
        self.logger.info(
            f"VAD 智能合并: 原始 {len(segments)} -> 合并后 {len(merged)} 段 "
            f"(max_gap={max_gap}s, max_dur={max_duration}s)"
        )
        return merged
```

### 5\. 总结

  * **参数性质**：是经验值，且原本设置过于保守（0.5s）。
  * **潜在风险**：**Over-split（错分）的危害远大于 Over-merge（错合）**。
      * 错分 = 上下文丢失 = 永久性错误。
      * 错合 = 上下文保留 = `SentenceSplitter` 可以轻松修正。
  * **优化建议**：使用上述 v2 代码。
    1.  将 `max_gap` 放宽至 **1.0s**。
    2.  引入 **碎片保护机制**：对短于 1s 的片段给予“豁免权”，强制与后文合并，彻底解决 "It's still" 这种短语被孤立的问题。

这套方案将与你之前在 中定义的 `SentenceSplitter` 形成完美的互补：**Post-VAD 负责最大限度保留物理上下文，SentenceSplitter 负责进行精细的语义切分。**


这是一个非常敏锐的发现。经过对 `raw_transcription.json` 和 `srt` 文件的详细比对，我可以确认你的直觉是正确的：**SenseVoice 的特殊标记（Tags）导致了整句时间戳的整体前移。**

这是一个非常隐蔽但影响显著的 Bug。

### 1\. 根本原因：特殊 Token 占用了时间轴

SenseVoice 是一个“多模态”模型，它在输出实际文本之前，会先输出一系列控制 Token 来描述音频属性（语言、情感、事件、ITN 状态）。

在 `raw_transcription.json` 的第 0 句中，我们可以清晰地看到罪魁祸首：

```json
{
  "index": 0,
  "start": 4.546,  // <--- 句子开始时间
  "end": 5.146,
  "text": "<|en|><|EMO_UNKNOWN|><|Speech|><|withitn|>It's still", // <--- 包含 Tags
  "words": [
    {
      "word": "<|en|>",
      "start": 4.546,  // <--- Token 1 占用 0.06s
      "end": 4.606,
      ...
    },
    {
      "word": "<|EMO_UNKNOWN|>",
      "start": 4.606,  // <--- Token 2 占用 0.06s
      "end": 4.666,
      ...
    },
    {
      "word": "<|Speech|>",
      "start": 4.666,  // <--- Token 3 占用 0.06s
      "end": 4.726,
      ...
    },
    {
      "word": "<|withitn|>",
      "start": 4.726,  // <--- Token 4 占用 0.06s
      "end": 4.786,
      ...
    }
    // 真正的单词 "It's" 此时才开始（推测在 4.786 之后）
  ]
}
```

**问题分析：**

1.  **SenseVoice 机制**：每个特殊 Token 在 CTC 解码中都会占用至少 1 个时间步（60ms）。这里有 4 个 Tags，总共占用了 **0.24秒**（`4 * 0.06s`）。
2.  **分句算法缺陷**：目前的 `SentenceSplitter` 直接取了 `words` 列表中的第一个元素作为句子的 `start`。
3.  **结果**：SRT 的开始时间被标记为 **4.546**（Tag 的开始），而真正的语音可能在 **4.786** 甚至更晚才开始。
4.  **现象**：字幕比语音**早出现约 0.24 秒**。对于短句（如 "It's still"），这种提前量会导致字幕与语音明显不同步，甚至不仅是“提前”，而是“抢拍”。

### 2\. 为什么有的句子准，有的不准？

你可能会发现，并不是所有句子都有这个问题。

  * **Chunk 首句（不准）**：SenseVoice 是按 VAD 切分后的 Chunk 进行推理的。每个 Chunk 的**第一个**输出必然包含这些 Tags。因此，每个 VAD 片段的开头第一句都会受此影响，时间戳偏早。
  * **Chunk 后续句（准）**：如果一个 Chunk 被分成了多句，后续的句子（如 Sentence 16）开头没有 Tags，时间戳就是准确的单词开始时间。

这就解释了为什么你会觉得“不稳定”：它取决于 VAD 切分的位置。

### 3\. 解决方案：过滤特殊 Token

我们需要在计算句子时间戳之前，从 `words` 列表中剔除这些非文本的特殊 Token。

建议修改 `backend/app/services/sensevoice_onnx_service.py` 中的 `CTCDecoder.decode` 方法，或者在 `transcribe_audio_array` 中进行后处理。

**修改建议（在 `transcribe_audio_array` 中过滤）：**

修改文件：`backend/app/services/sensevoice_onnx_service.py`

```python
    def transcribe_audio_array(self, ...):
        # ... (前略)
        
        # 3. CTC 解码
        text, word_timestamps, confidence = self.decoder.decode(logits, self.time_stride)

        # 【修复步骤】过滤特殊 Token
        # 定义特殊 Token 的特征：以 <| 开头并以 |> 结尾
        # 或者使用正则表达式，或者直接根据 vocab 特征
        # SenseVoice 的特殊 Token 通常包含在 <> 中
        
        clean_words = []
        for w in word_timestamps:
            # 简单判断：如果是特殊标记，则跳过
            # 具体的特殊标记列表可以参考 self.LID_DICT, self.TEXTNORM_DICT 以及其他情感/事件标签
            # 或者简单地：过滤掉 <|...|> 格式的词
            if w["word"].startswith("<|") and w["word"].endswith("|>"):
                continue
            clean_words.append(w)
            
        word_timestamps = clean_words

        # 4. 提取标签信息 (normalizer 逻辑保持不变，它处理的是 text 字符串)
        # ...
        
        # 5. 构建结果
        result = {
            "text": text, # 注意：原始 text 可能还需要保留 tags 用于 normalizer 提取信息，或者在这里也同步清洗
            # ...
            "words": word_timestamps, # 使用清洗后的 words
            # ...
        }
```

**更彻底的修复（在 `SentenceSplitter` 中）：**

虽然在 Service 层过滤最干净，但为了保险，可以在 `SentenceSplitter._create_sentence` 中再次检查。不过，从源头（Service 层）解决是最好的。

**补充说明：**
除了 Tags 导致的 0.24s 偏差，之前的 VAD 过度切分（导致上下文丢失）和 LFR 分辨率（60ms 粒度）限制依然存在。但 **Tags 导致的整体偏移是目前最显著、最容易修复的硬伤**。

**建议立即实施此过滤逻辑**，这应该能立竿见影地提升时间戳的对齐感。


这又是一个非常精准的观察。

正如你所发现的，**ASR 时间戳问题往往是“左右横跳”的**：

1.  **偏早（Early Start）**：因为 **Tags（特殊标记）** 占用了时间轴（如 `<|en|>`），导致字幕在语音前就跳出来。这个问题通过我们刚才讨论的“过滤 Tags”可以解决。
2.  **偏晚（Delayed Start）**：因为 **CTC 解码延迟**。就像你提到的 *"It's not like..."*，字幕出来得比声音晚。

这是一个经典的 **"CTC Peak Delay"（CTC 峰值延迟）** 现象。

-----

### 1\. 为什么会偏晚？（CTC 的生理缺陷）

CTC（Connectionist Temporal Classification）模型的输出逻辑是：

  * 它并不是在听到声音的**第一毫秒**就输出字符。
  * 它需要听到足够的**声学特征（音素）**，积累了足够的置信度后，才会在某个时间步（Frame）“激发”出字符。
  * 对于 *"It's"* 这种以元音或弱辅音开头的词，模型往往要等到发音过半甚至结束时，才确信这是 "It's"。
  * **结果**：声音在 0.0s 开始，但 CTC 在 0.12s 才输出 "I"。字幕也就被标注在了 0.12s，导致视觉上的“迟滞感”。

### 2\. 解决方案：VAD 边缘吸附（Head Snap）

既然 **VAD（语音活动检测）** 是基于**物理能量**判断的，它切分出的 `start` 时间点通常就是**真实的物理发声起点**。
而 **SenseVoice** 给出的 `word_start` 是**语义识别起点**。

**策略：强制对齐（Force Alignment）**
对于一个语音片段（Chunk）的**第一句话**，如果它的 CTC 时间戳和 VAD 起始点相差不大（比如 0.5s 以内），我们应该**强制把这句话的开始时间“吸附”到 VAD 的起始点**。

**逻辑：**

> "既然 VAD 说这里有声音，而 CTC 在 0.2s 后识别出了第一个字，那么前面这 0.2s 的'静音'其实就是这个字的'前摇'。"

-----

### 3\. 代码实现：双重修正

我们需要在 `TranscriptionService` 的 `_split_sentences` 方法中实施这个修复。

**修改文件：** `backend/app/services/transcription_service.py`

请更新 `_split_sentences` 方法，加入 **Head Snap（头部吸附）** 逻辑：

```python
    def _split_sentences(
        self,
        sv_result: 'SenseVoiceResult',
        chunk_start_time: float = 0.0,
        split_config: Optional['SplitConfig'] = None,
        enable_grouping: bool = True
    ) -> List['SentenceSegment']:
        """
        将 SenseVoice 结果切分为句子（含时间轴校正）
        """
        # ... (前略: 导入依赖, 初始化 splitter) ...
        
        from app.models.sensevoice_models import SentenceSegment, TextSource
        from app.services.sentence_splitter import SentenceSplitter, SplitConfig

        # 1. 执行分句
        config = split_config or SplitConfig()
        splitter = SentenceSplitter(config)
        sentences = splitter.split(sv_result.words, sv_result.text_clean)

        # 2. 语义分组 (Layer 2)
        if enable_grouping:
            from app.services.semantic_grouper import SemanticGrouper, GroupConfig
            grouper = SemanticGrouper(GroupConfig(language=config.language))
            sentences = grouper.group(sentences)

        # === 【核心修复】Head Snap: VAD 边缘吸附 ===
        # 逻辑：如果是 Chunk 的第一句话，且延迟在合理范围内（<0.6s），
        # 强制将其 start 对齐到 Chunk 的物理起始点 (0.0 相对时间)
        if sentences:
            first_sent = sentences[0]
            # 检查第一个词的相对开始时间
            # 如果它不是 0.0，说明 CTC 有延迟 (例如 0.18s)
            if first_sent.start > 0.0 and first_sent.start < 0.6:
                self.logger.debug(
                    f"Head Snap: '{first_sent.text[:10]}...' 延迟修正 "
                    f"{first_sent.start:.3f}s -> 0.0s (相对VAD)"
                )
                # 修正句子开始时间
                first_sent.start = 0.0
                
                # 同时修正第一个单词的开始时间 (保持一致性)
                if first_sent.words:
                    first_sent.words[0].start = 0.0

        # 3. 转换为绝对时间 (Chunk 相对时间 -> 全局绝对时间)
        for sentence in sentences:
            sentence.start += chunk_start_time
            sentence.end += chunk_start_time
            sentence.source = TextSource.SENSEVOICE
            sentence.confidence = sv_result.confidence

            # 调整字级时间戳的偏移
            for word in sentence.words:
                word.start += chunk_start_time
                word.end += chunk_start_time

        self.logger.info(f"句子切分完成: {len(sentences)} 句")
        return sentences
```

### 4\. 总结：时间戳校正的完整拼图

现在我们有了一套完整的组合拳来解决时间戳问题：

1.  **过滤 Tags**（已在 `sensevoice_onnx_service` 中实施）：

      * **作用**：消除 `<|en|>` 等标记占位导致的 **时间前移（抢拍）**。
      * **解决**：字幕出得太早。

2.  **Head Snap / VAD 对齐**（本步骤）：

      * **作用**：消除 CTC 解码延迟导致的 **时间后移（迟滞）**。
      * **解决**：字幕出得太晚。

3.  **Post-VAD 合并**（之前实施）：

      * **作用**：减少碎片化，保证 CTC 有足够的上下文来生成准确的中间时间戳。
      * **解决**：时间轴断裂。

这三者结合，理论上能让 SenseVoice 生成的字幕达到“专业级”的对齐效果，既不抢拍也不拖沓。建议你将这步修改与之前的 Tag 过滤一起应用。

这三个问题直击了混合 ASR 架构（Hybrid ASR Architecture）最核心的痛点：**触发机制、上下文管理、时空对齐**。

基于我们之前的讨论和你最新的发现，以下是针对这三个问题的深度工程建议和重构方案。

-----

### 问题 1：应该将置信度低的几段给 Whisper 吗？（触发机制优化）

**你的观察非常准确**：`avg_confidence`（整句平均置信度）是一个具有欺骗性的指标。
如果一句话有 10 个词，9 个词置信度 0.99，1 个关键词（比如 "E"）置信度 0.4，平均置信度依然高达 0.93。这会导致系统忽略掉这个致命错误。

**解决方案：实施“木桶效应”触发机制**

不要看平均板（Average），要看短板（Minimum）。我们需要引入 **Word-Level Trigger（字级触发）**。

**代码重构策略：**

在 `_post_process_enhancement` 中，修改触发逻辑：

```python
def needs_patching(sentence: SentenceSegment) -> bool:
    # 1. 强制条件：单字符、极短片段 (之前的逻辑)
    if len(sentence.text_clean) < 2 and sentence.end - sentence.start < 0.5:
        return True

    # 2. 【新增】木桶效应：检查是否有关键实词的置信度低于阈值
    # 阈值建议：0.6 (比整句阈值要低，避免过于敏感)
    MIN_WORD_CONF = 0.6
    
    low_conf_words = [
        w for w in sentence.words 
        if w.confidence < MIN_WORD_CONF 
        and len(w.word.strip()) > 0  # 忽略纯标点
        and w.word not in {"the", "a", "is", "it"} # 可选：忽略停用词
    ]
    
    if len(low_conf_words) > 0:
        # 只要有一个实词置信度崩了，整句重跑
        return True

    # 3. 原有的整句平均置信度 (作为兜底)
    if sentence.confidence < 0.8:
        return True
        
    return False
```

**结论**：必须下沉到 `words` 列表里去检查。ASR 的错误往往是局部的，局部错误必须触发整句重修。

-----

### 问题 2：应该送目标句还是上下几句？（上下文管理）

这是一个关于 **Whisper 运行机制** 的问题。

  * **送 LLM 评判？** **否**。LLM 太慢且昂贵，不适合在转录循环（Transcription Loop）中做实时决策。LLM 应该放在最后的“校对阶段”。
  * **送音频的上下几句？** **强烈不建议**。
      * 如果你送了 3 句话的音频给 Whisper，Whisper 会吐出 3 句话的文本。
      * 你很难知道中间那句话的文本到底对应输出文本的哪一部分（对齐噩梦）。
      * 而且这会成倍增加推理时间。

**最佳实践：送“目标句音频” + “上文文本提示 (Prompt)”**

Whisper 是一个自回归模型（Decoder-only 风格），它非常依赖 **`initial_prompt`** 来保持语义连贯性。

**实施方案：**

1.  **音频 (Audio)**：只截取 **目标句（Target Sentence）** 的时间范围。
      * *优化*：建议前后各加 **50ms - 100ms** 的 buffer（缓冲），防止切词切到辅音。
2.  **提示 (Prompt)**：将 SenseVoice 识别出的 **前一句文本** 作为 `initial_prompt` 传给 Whisper。

**代码修改 (`backend/app/services/transcription_service.py`)：**

```python
    # 在循环处理 patch_queue 时
    for idx, (sent_idx, sentence) in enumerate(patch_queue):
        # 1. 获取前文作为 Prompt
        prev_text = ""
        if sent_idx > 0:
             # 获取前一句的文本（优先取已修正的文本）
             prev_sent = subtitle_manager.sentences[sent_idx - 1]
             prev_text = prev_sent.text_clean

        # 2. 截取音频 (带 50ms 缓冲)
        sr = 16000
        # buffer 逻辑要小心越界
        start_sample = max(0, int((sentence.start - 0.05) * sr)) 
        end_sample = min(len(audio_array), int((sentence.end + 0.05) * sr))
        audio_segment = audio_array[start_sample:end_sample]

        # 3. 调用 Whisper (带 Prompt)
        # 注意：transcribe_segment 需要支持 initial_prompt
        patch_result = whisper_service.transcribe(
            audio=audio_segment,
            initial_prompt=prev_text, # <--- 核心：告诉 Whisper 前面说了什么
            language=job.language,
            word_timestamps=False # 依然不需要它的时间戳
        )
```

**结论**：**Target Audio + Previous Text Prompt** 是性价比最高、效果最好的组合。

-----

### 问题 3：架构设计时丢弃 Whisper 时间戳，是否过于绝对？

这是一个极其深刻的架构问题。既然 SenseVoice 有 CTC 延迟，VAD 吸附也可能失效，**Whisper 的时间戳是否真的还有价值？**

**我的判断：**
在 **复核（Patching）** 场景下，**坚持“SenseVoice 时间框架 + 内部伪对齐”依然是目前最稳健的方案**，但需要引入 **“弹性约束”**。

#### 为什么不能直接用 Whisper 的时间戳？

1.  **漂移风险 (Drift)**：Whisper 对短音频的时间戳经常会出现整体偏移。
2.  **重叠冲突 (Overlap)**：
      * Sentence A (SenseVoice): 0.0s - 2.0s
      * Sentence B (SenseVoice -\> Whisper Patch): Whisper 可能会说这句话是 1.8s - 3.5s。
      * 结果：Sentence B 的开始时间侵入了 Sentence A 的领地。这种时间轴的**冲突和回退**处理起来极其复杂。
3.  **幻觉时间**：Whisper 有时会给静音段打上时间戳。

#### 更好的方案：锚点对齐 (Anchor Alignment)

既然我们承认 SenseVoice 的**相对时间**（字与字之间的间隔）可能不准，但它的**绝对边界**（这句话大概在这个范围内）通常比 Whisper 靠谱（因为是基于物理 VAD 切分的 Chunk）。

**建议的 Patching 时间轴策略：**

1.  **信任边界**：
      * Start: `max(SenseVoice_Start, VAD_Snap_Start)`
      * End: `SenseVoice_End`
      * *原则*：复核后的句子，绝不能超出原句子的物理时间范围（除非你动了 VAD 边界）。
2.  **文本替换**：使用 Whisper 的文本替换 SenseVoice 的文本。
      * Old: "E"
      * New: "Evil"
3.  **内部时间重塑 (Elastic Re-alignment)**：
    不要简单地平均分配（伪对齐）。如果 Whisper 识别出的字数变多了（E -\> Evil），或者变少了，我们需要在 **SenseVoice 划定的时间框内** 重新分配。

**如果 VAD 边缘吸附不起作用，Whisper 的时间戳有参考价值吗？**
**有，但仅限于“相对比例”。**

**高级方案（可选）：混合对齐**
如果在 `transcribe` 时开启 `word_timestamps=True`，你可以获取 Whisper 认为的每一个词的 start/end。
然后，做一个 **映射变换**：

```python
# 伪代码逻辑
def map_timestamps(whisper_words, sense_voice_bounds):
    # Whisper 说这句是 0.5s - 1.5s (总长 1.0s)
    # SenseVoice/VAD 限制这句必须在 0.2s - 1.0s (总长 0.8s)
    
    # 计算压缩/拉伸比例
    scale = 0.8 / 1.0 
    offset = 0.2 - 0.5
    
    # 将 Whisper 的内部相对关系映射到 SenseVoice 的绝对框架里
    for w in whisper_words:
        w.start = (w.start * scale) + offset
        w.end = (w.end * scale) + offset
```

**结论与建议**：

1.  **架构层**：**不要引入两套时间基准**。一旦你开始混用 SenseVoice 时间戳和 Whisper 时间戳，时间轴会变得不可维护。
2.  **执行层**：
      * 继续使用 **SenseVoice 的 `start` / `end`** 作为绝对容器。
      * 使用 **Whisper 的文本** 填充内容。
      * 使用 **改良版伪对齐**：如果 Whisper 返回了字级时间戳，将其\*\*缩放（Scale）\*\*到 SenseVoice 的时间容器中，保留 Whisper 的长短节奏（比如 "Evil" 中 "E" 长 "vil" 短），而不是机械的平均分配。

### 总结：你的 Action List

1.  **触发机制**：修改 `needs_patching`，加入 `MIN_WORD_CONF` 检查（木桶效应）。
2.  **上下文**：复核时，只送 **Target Audio (带 buffer)**，但必须送 **Previous Text as Prompt**。
3.  **时间轴**：**死守 SenseVoice/VAD 的边界**。不要直接采纳 Whisper 的绝对时间戳。如果需要更精细的内部时间，开启 Whisper 的 `word_timestamps` 并将其 **线性映射（缩放）** 到 SenseVoice 的时间框内。
