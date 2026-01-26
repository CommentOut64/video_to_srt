# VAD 优化开发计划

> 基于 VAD优化.md 中的讨论，结合当前代码实际情况，细化的分阶段开发文档。

## 代码调查结论

| 功能点 | 当前状态 | 关键文件 |
|--------|----------|----------|
| Post-VAD 合并 | ❌ 未实现 | `transcription_service.py` |
| 强制关键复核 | ⚠️ 部分实现（有基础复核，无强制逻辑） | `transcription_service.py`, `thresholds.py` |
| Tags 过滤 | ⚠️ text 已过滤，word_timestamps 未过滤 | `sensevoice_onnx_service.py` |
| VAD 边缘吸附 | ❌ 未实现 | `transcription_service.py` |
| Word-Level Trigger | ❌ 未实现（仅句级触发） | `thresholds.py` |
| 前端字级高亮 | ⚠️ 句级已实现，字级未实现 | `SubtitleList/index.vue` |
| Whisper 模型配置 | ✅ 默认 medium，已有模型管理器 | `whisper_service.py`, `model_manager_service.py` |

---

## 阶段一：Tags 过滤（优先级最高）

### 问题描述

SenseVoice 输出的特殊标记（`<|en|>`, `<|EMO_UNKNOWN|>`, `<|Speech|>`, `<|withitn|>`）占用了时间轴，导致字幕时间戳整体前移约 0.24 秒。

### 当前实现
- `text_normalizer.py:49` 使用正则 `r'<\|.*?\|>'` 过滤 `text` 字段
- **问题**：`word_timestamps` 列表中的特殊标记未被过滤

### 修改文件
`backend/app/services/sensevoice_onnx_service.py`

### 修改位置
`transcribe_audio_array` 方法（约 413-484 行）

### 具体修改

```python
# 在 CTC 解码后、构建结果前添加过滤逻辑
# 位置：约第 460 行附近

# 3. CTC 解码
text, word_timestamps, confidence = self.decoder.decode(logits, self.time_stride)

# 【新增】过滤特殊标记，修复时间戳前移问题
# 特殊标记格式：<|xxx|>
clean_word_timestamps = []
for w in word_timestamps:
    word = w.get("word", "")
    if word.startswith("<|") and word.endswith("|>"):
        # 跳过特殊标记，它们不应占用时间轴
        continue
    clean_word_timestamps.append(w)
word_timestamps = clean_word_timestamps
```

### 预期效果
- 消除 `<|en|>` 等标记占位导致的时间前移（约 0.24s）
- 字幕开始时间与实际语音同步

### 测试要点
- 对比过滤前后的 `raw_transcription.json`，确认首句时间戳不再提前

---

## 阶段二：Post-VAD 智能合并层

### 问题描述

VAD 过度切分导致语义断裂，例如 "It's still" 和 "only 7:28..." 被切成两个片段，SenseVoice 失去上下文，识别率下降。

### 当前实现

- VAD 切分后直接进入转录，无合并逻辑

### 修改文件
`backend/app/services/transcription_service.py`

### 修改位置
1. 新增 `_merge_vad_segments` 方法
2. 在 `_process_video_sensevoice` 中调用（约 4500 行附近）

### 具体修改

#### 2.1 新增合并方法（建议放在 `_memory_vad_split` 方法后）

```python
def _merge_vad_segments(
    self,
    segments: List[Dict],
    max_gap: float = 1.0,           # 最大允许间隔，涵盖慢语速场景
    max_duration: float = 25.0,      # 最大合并时长，避免显存溢出
    min_fragment_duration: float = 1.0  # 碎片保护阈值
) -> List[Dict]:
    """
    [架构优化] Post-VAD 智能合并层

    策略：宁可错合（依赖 SentenceSplitter 分句），不可错分（导致 ASR 丢失上下文）。

    Args:
        segments: VAD 切分后的原始片段列表 [{start, end}, ...]
        max_gap: 允许合并的最大静音间隔（秒）
        max_duration: 合并后的最大时长（秒）
        min_fragment_duration: 短于此时长的片段强制尝试合并

    Returns:
        合并后的片段列表
    """
    if not segments:
        return []

    merged = []
    current = segments[0].copy()

    for next_seg in segments[1:]:
        gap = next_seg['start'] - current['end']
        current_duration = current['end'] - current['start']
        combined_duration = next_seg['end'] - current['start']

        should_merge = False

        # 条件 1: 基础合并（间隔小且总长不超标）
        if gap <= max_gap and combined_duration <= max_duration:
            should_merge = True

        # 条件 2: 碎片保护（当前段极短，可能是被切断的单词）
        # 例如: "It's" (0.5s) ... [gap 1.5s] ... "only..."
        elif current_duration < min_fragment_duration and combined_duration <= max_duration:
            # 限制 gap 不超过 3s，避免引入过长静音
            if gap < 3.0:
                self.logger.debug(
                    f"碎片强制合并: fragment={current_duration:.2f}s, gap={gap:.2f}s"
                )
                should_merge = True

        if should_merge:
            current['end'] = next_seg['end']
        else:
            merged.append(current)
            current = next_seg.copy()

    merged.append(current)

    self.logger.info(
        f"VAD 智能合并: 原始 {len(segments)} -> 合并后 {len(merged)} 段 "
        f"(max_gap={max_gap}s, max_dur={max_duration}s)"
    )
    return merged
```

#### 2.2 集成到处理流程

在 `_process_video_sensevoice` 方法中，VAD 切分后调用：

```python
# 位置：约 4510 行附近，在 _memory_vad_split 调用之后

# 2. VAD 物理切分
progress_tracker.start_phase(ProcessPhase.VAD, 1, "VAD 切分...")
raw_vad_segments = self._memory_vad_split(audio_array, sr, job)

# 【新增】Post-VAD 智能合并层
# max_gap=1.0s 涵盖慢语速场景，碎片保护避免短片段孤立
vad_segments = self._merge_vad_segments(
    raw_vad_segments,
    max_gap=1.0,
    max_duration=25.0,
    min_fragment_duration=1.0
)

progress_tracker.complete_phase(ProcessPhase.VAD)
```

### 预期效果
- "It's still" 和 "only 7:28..." 合并为一个片段
- SenseVoice 获得完整上下文，识别率提升
- 分句算法能看到完整文本流

### 测试要点
- 检查日志中的合并统计信息
- 对比合并前后的转录结果准确性

---

## 阶段三：VAD 边缘吸附（Head Snap）

### 问题描述

CTC 解码延迟导致字幕开始时间偏晚，例如语音在 0.0s 开始，但 CTC 在 0.12s 才输出第一个字符。

### 当前实现

- 无边缘吸附逻辑

### 修改文件

`backend/app/services/transcription_service.py`

### 修改位置

`_split_sentences` 方法（约 3891-3940 行）

### 具体修改

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

    # === 【新增】VAD 边缘吸附 (Head Snap) ===
    # 逻辑：如果是 Chunk 的第一句话，且延迟在合理范围内（<0.6s），
    # 强制将其 start 对齐到 Chunk 的物理起始点 (0.0 相对时间)
    HEAD_SNAP_THRESHOLD = 0.6  # 最大允许吸附的延迟

    if sentences:
        first_sent = sentences[0]
        # 检查第一个词的相对开始时间（相对于 chunk）
        # 如果 > 0 且 < 阈值，说明 CTC 有延迟
        if 0 < first_sent.start < HEAD_SNAP_THRESHOLD:
            self.logger.debug(
                f"Head Snap: '{first_sent.text[:15]}...' 延迟修正 "
                f"{first_sent.start:.3f}s -> 0.0s (相对VAD)"
            )
            # 修正句子开始时间
            first_sent.start = 0.0
            # 同时修正第一个单词的开始时间（保持一致性）
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

### 预期效果
- 消除 CTC 解码延迟导致的字幕迟滞
- 字幕开始时间与 VAD 检测到的物理发声点对齐

### 测试要点
- 检查日志中的 Head Snap 修正记录
- 主观评估字幕是否与语音同步

---

## 阶段四：强制关键复核

### 问题描述
即使用户选择"极速"预设（enhancement=OFF），对于极高概率错误（如单字符结果 "E"）也应强制启动 Whisper 复核。

### 当前实现
- `thresholds.py` 已有三种触发条件：低置信度、短片段、单字符
- `_post_process_enhancement` 方法遵循用户设置，enhancement=OFF 时不复核

### 修改文件
1. `backend/app/core/thresholds.py` - 新增强制复核判断函数
2. `backend/app/services/transcription_service.py` - 修改后处理逻辑

### 修改位置
1. `thresholds.py` - 新增 `is_critical_patch_needed` 函数
2. `transcription_service.py:4380-4453` - 修改 `_post_process_enhancement`

### 具体修改

#### 4.1 新增强制复核判断函数

在 `backend/app/core/thresholds.py` 中添加：

```python
def is_critical_patch_needed(
    text: str,
    duration: float,
    confidence: float
) -> bool:
    """
    判断是否需要强制复核（无论用户设置如何）

    强制条件（二选一）：
    1. 单字符结果且置信度 < 0.9（极可能是 CTC 漏字）
    2. 极短片段（<0.2s）且文本极短（<3字符）

    Args:
        text: 清洗后的文本
        duration: 片段时长（秒）
        confidence: 置信度

    Returns:
        True 表示必须强制复核
    """
    clean_text = text.strip()

    # 条件 1：单字符 + 非高置信度
    if len(clean_text) == 1 and confidence < 0.9:
        return True

    # 条件 2：极短片段 + 极短文本
    if duration < 0.2 and len(clean_text) < 3:
        return True

    return False
```

#### 4.2 修改后处理增强方法

在 `transcription_service.py` 的 `_post_process_enhancement` 方法中：

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
    from app.services.solution_matrix import EnhancementMode
    from app.core.thresholds import needs_whisper_patch, is_critical_patch_needed

    progress_tracker = get_progress_tracker(job.job_id, solution_config.preset_id)
    patch_queue = []

    # === 构建复核队列 ===
    for i, sentence in enumerate(sentences):
        should_patch = False
        is_critical = False

        duration = sentence.end - sentence.start
        clean_text = sentence.text_clean.strip()

        # 1. 强制关键复核条件（无论用户设置如何，必须修）
        if is_critical_patch_needed(clean_text, duration, sentence.confidence):
            should_patch = True
            is_critical = True
            self.logger.warning(
                f"触发强制复核: '{clean_text}' "
                f"(conf={sentence.confidence:.2f}, dur={duration:.2f}s)"
            )

        # 2. 常规复核条件（遵循用户设置）
        elif solution_config.enhancement != EnhancementMode.OFF:
            text_length = len(clean_text)
            if needs_whisper_patch(sentence.confidence, duration, text_length):
                should_patch = True

        if should_patch:
            patch_queue.append((i, sentence, is_critical))

    # === 执行复核 ===
    if patch_queue:
        progress_tracker.start_phase(
            ProcessPhase.WHISPER_PATCH,
            len(patch_queue),
            "Whisper 修正中..."
        )

        # 确保 Whisper 服务可用
        from app.services.whisper_service import get_whisper_service
        whisper_service = get_whisper_service()

        for idx, (sent_idx, sentence, is_critical) in enumerate(patch_queue):
            # 获取前文作为 Prompt（提升复核准确性）
            prev_text = ""
            if sent_idx > 0 and sent_idx - 1 < len(subtitle_manager.sentences):
                prev_sent = subtitle_manager.sentences[sent_idx - 1]
                prev_text = prev_sent.text_clean

            await self._whisper_text_patch(
                sentence,
                sent_idx,
                audio_array,
                job,
                subtitle_manager,
                initial_prompt=prev_text  # 传递上下文
            )
            progress_tracker.update_phase(ProcessPhase.WHISPER_PATCH, increment=1)

        progress_tracker.complete_phase(ProcessPhase.WHISPER_PATCH)

    # ... 后续 LLM 校对逻辑保持不变 ...
    return sentences
```

#### 4.3 修改 Whisper 复核方法支持 initial_prompt

在 `_whisper_text_patch` 方法中添加 `initial_prompt` 参数：

```python
async def _whisper_text_patch(
    self,
    sentence: 'SentenceSegment',
    sent_idx: int,
    audio_array: np.ndarray,
    job: 'JobState',
    subtitle_manager: 'StreamingSubtitleManager',
    initial_prompt: str = ""  # 【新增】上下文提示
):
    """Whisper 复核（带上下文）"""
    # ... 现有逻辑 ...

    # 调用 Whisper 时传入 initial_prompt
    # 注意：需要确认 whisper_service.transcribe 支持此参数
```

### Whisper 模型配置说明

根据调查，当前系统已具备完整的模型管理能力：

- **默认模型**：`medium`（1500MB）- 已在 `whisper_service.py` 中配置
- **模型管理器**：`model_manager_service.py` 支持下载进度追踪
- **缓存目录**：`{项目根目录}/models/huggingface`
- **镜像源**：默认使用 `hf-mirror.com`

**无需额外修改**，Whisper 服务已正确接入模型管理器。

### 预期效果
- "E"（0.06s, conf 0.48）等极端情况被强制复核
- 即使用户选择"极速"预设，也能拦截不可接受的错误
- 复核时利用前文上下文，提高修正准确性

### 测试要点
- 使用"极速"预设处理包含快速语音的视频
- 检查单字符结果是否被强制复核

---

## 阶段五：Word-Level Trigger（字级触发）

### 问题描述
当前复核触发基于整句平均置信度，但如果一句话有 10 个词，9 个词置信度 0.99，1 个关键词置信度 0.4，平均置信度依然高达 0.93，导致漏补。

### 当前实现
- `needs_whisper_patch` 函数仅检查句级置信度

### 修改文件
`backend/app/core/thresholds.py`

### 具体修改

扩展 `needs_whisper_patch` 函数，增加字级检查：

```python
def needs_whisper_patch(
    confidence: float,
    duration: float = None,
    text_length: int = None,
    words: List[Dict] = None,  # 【新增】字级时间戳列表
    config: ThresholdConfig = None
) -> bool:
    """
    判断是否需要 Whisper 复核

    触发条件（任意满足即触发）：
    1. 句级置信度低于阈值
    2. 短片段（<1s）且字符数少（<3）
    3. 单字符结果（强制）
    4. 【新增】任意实词的置信度低于字级阈值（木桶效应）
    """
    if config is None:
        config = ThresholdConfig()

    # 条件 1：低置信度
    if confidence < config.whisper_patch_trigger_confidence:
        return True

    # 条件 2：短片段
    if duration is not None and text_length is not None:
        if duration < config.short_segment_duration and text_length < config.short_segment_chars:
            return True

    # 条件 3：单字符
    if text_length is not None and text_length == 1 and config.single_char_force_patch:
        return True

    # 条件 4：【新增】字级木桶效应
    # 检查是否有实词的置信度低于阈值
    if words:
        MIN_WORD_CONF = config.word_warning_confidence  # 使用已有的字级警告阈值 (0.5)
        # 停用词列表（这些词即使置信度低也不触发复核）
        STOP_WORDS = {"the", "a", "an", "is", "it", "to", "of", "and", "in", "on"}

        for w in words:
            word_text = w.get("word", "").strip().lower()
            word_conf = w.get("confidence", 1.0)

            # 跳过空字符、标点、停用词
            if not word_text or len(word_text) == 1 and not word_text.isalnum():
                continue
            if word_text in STOP_WORDS:
                continue

            # 任意实词置信度低于阈值，触发整句复核
            if word_conf < MIN_WORD_CONF:
                return True

    return False
```

### 调用方修改

在 `_post_process_enhancement` 中传入 words 参数：

```python
# 常规复核条件（遵循用户设置）
elif solution_config.enhancement != EnhancementMode.OFF:
    text_length = len(clean_text)
    # 传入字级时间戳列表
    words_data = [{"word": w.word, "confidence": w.confidence} for w in sentence.words]
    if needs_whisper_patch(
        sentence.confidence,
        duration,
        text_length,
        words=words_data  # 【新增】
    ):
        should_patch = True
```

### 预期效果
- 即使整句平均置信度高，只要有一个实词置信度崩了，也会触发复核
- 提高对局部错误的检测能力

### 测试要点
- 构造测试用例：大部分词高置信度，个别词低置信度
- 验证是否正确触发复核

---

## 阶段六：前端字级置信度高亮

### 问题描述
后端能产生字级置信度数据，但前端只实现了句级高亮，字级高亮未集成。

### 当前实现

- **后端**：`SentenceSegment.words` 包含每个词的 `confidence`
- **前端**：`SubtitleList/index.vue` 有句级警告样式，无字级

### 修改文件

1. `frontend/src/components/editor/SubtitleList/index.vue` - 添加字级渲染
2. `frontend/src/services/sseChannelManager.js` - 确保数据传递

### 具体修改

#### 6.1 修改字幕文本渲染逻辑

在 `SubtitleList/index.vue` 中，将纯文本渲染改为带标记的 HTML：

```vue
<template>
  <!-- 原有的纯文本显示 -->
  <!-- <span class="text">{{ subtitle.text }}</span> -->

  <!-- 【新增】支持字级高亮的渲染 -->
  <span class="text" v-html="renderTextWithHighlight(subtitle)"></span>
</template>

<script setup>
// 字级高亮渲染函数
function renderTextWithHighlight(subtitle) {
  // 如果没有字级数据，直接返回文本
  if (!subtitle.words || subtitle.words.length === 0) {
    return escapeHtml(subtitle.text);
  }

  const WARN_THRESHOLD = 0.5;    // 警告阈值
  const CRITICAL_THRESHOLD = 0.3; // 严重警告阈值

  let html = '';
  for (const word of subtitle.words) {
    const conf = word.confidence || 1.0;
    const text = escapeHtml(word.word);

    if (conf < CRITICAL_THRESHOLD) {
      html += `<span class="word-critical">${text}</span>`;
    } else if (conf < WARN_THRESHOLD) {
      html += `<span class="word-warning">${text}</span>`;
    } else {
      html += text;
    }
  }
  return html;
}

// HTML 转义函数
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
</script>

<style scoped>
/* 字级警告样式 */
.word-warning {
  background-color: rgba(255, 193, 7, 0.3);
  border-bottom: 2px solid var(--warning, #ffc107);
  padding: 0 2px;
  border-radius: 2px;
}

.word-critical {
  background-color: rgba(244, 67, 54, 0.3);
  border-bottom: 2px solid var(--error, #f44336);
  padding: 0 2px;
  border-radius: 2px;
}
</style>
```

#### 6.2 确保数据传递

在 SSE 事件处理中确保 `words` 数组被正确传递：

```javascript
// frontend/src/services/sseChannelManager.js
// 确认 subtitle.sv_sentence 事件中包含 words 数据

// 后端已在 streaming_subtitle.py 中传递完整的 sentence 对象
// 包含 words 列表，无需额外修改
```

#### 6.3 后端数据确认

确认 `backend/app/services/streaming_subtitle.py` 的 `_to_dict` 方法包含 words：

```python
def _to_dict(self, sentence: SentenceSegment) -> Dict:
    return {
        "index": sentence.index,
        "start": sentence.start,
        "end": sentence.end,
        "text": sentence.text,
        "text_clean": sentence.text_clean,
        "confidence": sentence.confidence,
        "warning_type": sentence.warning_type.value if sentence.warning_type else None,
        # 确保包含 words
        "words": [
            {
                "word": w.word,
                "start": w.start,
                "end": w.end,
                "confidence": w.confidence
            } for w in sentence.words
        ] if sentence.words else []
    }
```

### 预期效果
- 低置信度的词在前端以黄色/红色高亮显示
- 用户可直观看到哪些词可能识别有误

### 测试要点
- 检查 SSE 事件中是否包含 words 数据
- 检查前端是否正确渲染高亮样式

---

## 阶段七：四层分句系统审视

### 当前状态

| 层级 | 名称 | 状态 | 需要调整 |
|------|------|------|----------|
| Layer 1 | 基础分句 (SentenceSplitter) | ✅ 已实现 | 需适配 Tags 过滤后的数据 |
| Layer 2 | 语义分组 (SemanticGrouper) | ✅ 已实现 | 需适配 VAD 合并后的长片段 |
| Layer 3 | 流式合并 (StreamingSubtitleManager) | ✅ 已实现 | 无需修改 |
| Layer 4 | LLM 校对 | ❌ 未实现 | TODO |

### 需要审视的调整点

#### 7.1 Layer 1 适配

Tags 过滤后，`words` 列表中不再有特殊标记，分句算法应正常工作。

**验证点**：
- 确认 `_create_sentence` 方法处理的 words 不含 `<|...|>` 格式

#### 7.2 Layer 2 适配

Post-VAD 合并后，单个 Chunk 可能更长（最大 25s），语义分组需要处理更长的句子列表。

**潜在问题**：
- 合并后的长片段可能包含多个自然句，需要依赖 Layer 1 正确分割
- 时间重叠检测逻辑应正常工作

**验证点**：
- 检查长片段（>10s）的分句结果
- 确认语义分组不会错误合并不相关的句子

#### 7.3 与 VAD 边缘吸附的协调

Head Snap 在 `_split_sentences` 中执行，位于 Layer 1 和 Layer 2 之间。

**执行顺序**：
1. Layer 1 分句 → 2. Head Snap → 3. Layer 2 分组 → 4. 时间偏移转换

这个顺序是合理的，无需调整。

### 测试建议

1. **端到端测试**：使用包含快速语音、长静音、多语言的测试视频
2. **分层验证**：在每一层输出日志，验证数据流转正确性
3. **边界条件**：测试极短片段（<0.5s）、极长片段（>20s）的处理

---

## 实施顺序建议

```
阶段一 (Tags 过滤) ─────────────────────────────────────────┐
                                                            │
阶段二 (Post-VAD 合并) ─┬─ 阶段三 (VAD 边缘吸附) ──────────┤
                        │                                    │
                        └─ 阶段四 (强制关键复核) ──────────┤
                                      │                      │
                                      └─ 阶段五 (字级触发) ─┤
                                                            │
阶段六 (前端字级高亮) ─────────────────────────────────────┤
                                                            │
阶段七 (四层系统审视) ─────────────────────────────────────┘
```

**推荐顺序**：
1. 阶段一（Tags 过滤）- 立竿见影，修复时间戳前移
2. 阶段三（VAD 边缘吸附）- 修复时间戳后移
3. 阶段二（Post-VAD 合并）- 提升识别准确性
4. 阶段四（强制复核）+ 阶段五（字级触发）- 可并行开发
5. 阶段六（前端高亮）- 用户体验增强
6. 阶段七（系统审视）- 整体验证

---

## 风险评估

| 阶段 | 风险等级 | 风险描述 | 缓解措施 |
|------|----------|----------|----------|
| 一 | 低 | 简单过滤，影响范围小 | 充分测试不同语言 |
| 二 | 中 | 合并策略可能导致过长片段 | 设置 max_duration 上限 |
| 三 | 低 | 简单时间调整 | 设置吸附阈值上限 |
| 四 | 中 | Whisper 调用增加处理时间 | 仅对极端情况强制复核 |
| 五 | 中 | 字级检查增加计算开销 | 使用停用词过滤 |
| 六 | 低 | 前端样式修改 | 保持向后兼容 |
| 七 | 低 | 验证性工作 | 分层测试 |

---

## 文档更新清单

完成所有阶段后，需要更新以下文档：

1. `llmdoc/architecture/sensevoice-presets.md` - 更新预设行为说明
2. `llmdoc/reference/threshold-config.md` - 新增字级触发参数
3. `llmdoc/architecture/whisper-patch-enhancement.md` - 新增强制复核说明
4. `llmdoc/index.md` - 更新最后修改日期和内容

---

*文档生成时间：2025-12-08*
*基于：VAD优化.md 讨论 + 代码调查结果*
