# 对齐机制深度调查报告

> **调查日期**: 2025-12-26
> **调查范围**: 伪对齐算法、双模态对齐、WhisperX遗留、字级时间戳生成、创新点分析
> **项目版本**: V3.8+

---

## Code Sections (The Evidence)

### 1. 伪对齐算法核心实现

- `backend/app/services/pseudo_alignment.py` (PseudoAlignment): 伪对齐算法实现，将新文本均匀映射到SenseVoice确定的时间窗口内
  - `apply()` (L22-89): 核心对齐方法，支持中英文智能切分（英文按单词，中文按字符），生成标记为`is_pseudo=True`的字级时间戳
  - `apply_to_sentence()` (L92-119): 对句子应用伪对齐，保留原时间边界
  - `merge_words_to_sentence()` (L122-136): 将字级时间戳合并为句子文本

### 2. AlignmentWorker - 三级降级对齐策略

- `backend/app/pipelines/workers/alignment_worker.py` (AlignmentWorker): 对齐层Worker，实现三级降级策略
  - `AlignmentLevel` (L31-35): 对齐级别枚举 - DUAL_MODAL(黄金), WHISPER_PSEUDO(银标准), SENSEVOICE_ONLY(铜标准)
  - `process()` (L116-214): 对齐主流程，支持V3.10快速路径（Whisper跳过时直接用SenseVoice）
  - `_align_and_fallback()` (L216-481): 核心对齐+降级逻辑，包含四重检测机制
    - 检测0 (L246-268): Whisper单词数过少检测（幻觉拦截）
    - 检测1 (L275-301): Whisper长度暴涨检测（len(whisper) > 3*len(sv)+10 且置信度<0.5）
    - 检测2 (L304-321): Whisper过短检测（len(whisper) < len(sv)*0.65，V3.8.1从0.3提升到0.65）
    - Level 1 (L324-389): 双模态对齐（Needleman-Wunsch序列对齐）
    - Level 2 (L398-433): Whisper伪对齐（对齐失败时回退）
    - Level 3 (L438-481): SenseVoice草稿（最终兜底）
  - `_split_sentences_from_sv()` (L483-554): 从SenseVoice结果分句，V3.8修复无字级时间戳时的兜底逻辑

### 3. AlignmentService - Needleman-Wunsch序列对齐

- `backend/app/services/alignment/alignment_service.py` (AlignmentService): 双流对齐服务，使用Needleman-Wunsch算法
  - `AlignmentConfig` (L26-47): 对齐配置参数（match_score=2, mismatch_penalty=-1, gap_penalty=-2）
  - `align()` (L72-145): 双流对齐主入口，包含6个步骤
    1. 文本预处理 (L97-102)
    2. Needleman-Wunsch序列对齐 (L104-105)
    3. 生成对齐后的字级时间戳 (L107-114)
    4. 能量锚点校准（可选）(L116-122)
    5. VAD边界校准 (L124-129)
    6. 构建AlignedSubtitle (L131-145)
  - `_needleman_wunsch()` (L163-242): 全局序列对齐算法实现，返回对齐路径
  - `_generate_aligned_words()` (L302-379): 根据对齐路径生成字级时间戳，处理匹配/替换/插入三种情况
  - `_apply_energy_anchor()` (L438-483): 能量锚点校准，利用音频能量峰值重新定位词边界
  - `_apply_vad_calibration()` (L485-514): VAD边界校准，确保时间戳在语音范围内
  - `_compute_alignment_score()` (L563-588): 计算对齐质量分数（0.6*匹配率 + 0.4*平均置信度）

### 4. WhisperBufferPool - 长文本对齐到Chunk

- `backend/app/services/whisper_buffer_pool.py` (WhisperBufferPool): Whisper缓冲池，支持长音频拼接推理
  - `align_text_to_chunks()` (L262-320): 将Whisper长文本对齐到原始Chunk（V2: Needleman-Wunsch）
    - 收集Whisper词和SenseVoice词作为锚点
    - 使用Needleman-Wunsch进行全局序列对齐
    - 根据对齐结果分配Whisper词到对应Chunk
  - `_needleman_wunsch()` (L377-457): 序列对齐算法实现（与AlignmentService中的实现相同）
  - `_distribute_by_alignment()` (L476-547): 根据对齐路径分配词到Chunk
  - `_timestamp_based_alignment()` (L549-596): 时间戳对齐（回退方案）
  - `_fallback_alignment()` (L708-745): 兜底对齐（按文本长度比例分配）

### 5. 字级时间戳数据模型

- `backend/app/models/sensevoice_models.py` (WordTimestamp): 字级时间戳数据模型
  - `WordTimestamp` (L56-77): 字级时间戳结构，包含word, start, end, confidence, is_pseudo标记
  - `SentenceSegment` (L110-227): 句级字幕段，包含words数组、对齐质量字段（alignment_score, matched_ratio）

### 6. 字幕切分功能（前端）

- `frontend/src/stores/projectStore.js` (splitSubtitle): 字幕切分核心方法
  - `splitSubtitle()` (L343-422): 字幕切分主方法，支持时间点切分和光标位置切分
  - `_splitByTime()` (L428-482): 基于时间点切分（波形图模式），优先使用字级时间戳
    - 查找切分点所在的字 (L439-460)
    - 回退到字数比例估算 (L463-481)
  - `_splitByCursor()` (L488-557): 基于光标位置切分（文本编辑模式）
    - 根据文本切分点找字级时间戳边界 (L500-535)
    - 回退到字数比例估算时间 (L538-556)

### 7. 分句算法（语义完整性检查）

- `backend/app/services/sentence_splitter.py` (SentenceSplitter): 分句算法实现，支持多语言策略
  - `LanguageStrategy` (L33-74): 语言策略抽象基类
  - `ChineseStrategy` (L77-239): 中文语言策略，包含完整词排除列表和不完整结尾词集合
    - `COMPLETE_WORDS` (L81-92): 完整词排除列表（以"件"/"为"/"罐"等结尾但完整的词）
    - `INCOMPLETE_ENDINGS` (L94-175): 中文不完整结尾词集合（介词、连词、量词等）
  - `_smart_split_long_sentence()` (L912+): 智能长句拆分，优先在标点和停顿处切分

### 8. FastWorker - 快流推理（草稿生成）

- `backend/app/pipelines/workers/fast_worker.py` (FastWorker): CPU层快速推理，生成草稿字幕
  - `process()` (L165-186): 快流处理主入口，支持熔断和非熔断两种模式
  - `_split_sentences()` (L347+): 快流分句，使用FastWorker专用配置（动态停顿阈值，软上限5秒）

### 9. AsyncDualPipeline - 乱序执行架构

- `backend/app/pipelines/async_dual_pipeline.py` (AsyncDualPipeline): 三级异步流水线架构
  - `_fast_loop()`: FastWorker并发处理（CPU层）
  - `_slow_loop()`: SlowWorker顺序处理（GPU层，维持Whisper上下文）
  - `_align_loop()` (L657+): AlignmentWorker对齐层（CPU）
  - V3.5极速模式: 支持`sensevoice_only`配置，跳过Whisper直接输出定稿

### 10. WhisperX遗留检测

- 项目中已完全移除WhisperX依赖，无遗留代码
- 通过Grep搜索确认：无`import whisperx`或`from whisperx`语句
- 对齐模型管理已从model_config.py、model_routes.py中删除

---

## Report (The Answers)

### result

#### 1. 伪对齐算法（PseudoAlignment）

**核心原理**:
- **时间窗口不可变**: SenseVoice输出的`[start, end]`严格保留，作为绝对时间轴基准
- **均匀分布策略**: 新文本的每个token均匀分布在时间窗口内
  ```
  每个token的时长 = (end - start) / token_count
  token_i.timestamp = start + i * step
  ```
- **智能切分**: 英文按单词切分，中文按字符切分，支持混合文本
- **置信度标记**: 生成的时间戳标记为`is_pseudo=True`，用于区分真实对齐和估算对齐

**使用场景**:
1. Whisper补刀后替换文本时，重新生成字级时间戳
2. LLM校对/翻译后，将新文本映射到原时间窗口
3. AlignmentWorker Level 2降级策略中，作为双模态对齐失败的回退方案
4. 字幕切分时，当缺少字级时间戳时按比例估算时间

**调用链路**:
```
AlignmentWorker._align_and_fallback() (Level 2)
  → PseudoAlignment.apply()
  → 生成is_pseudo=True的WordTimestamp列表
  → SentenceSplitter.split() 重新分句
```

#### 2. WhisperX强制对齐遗留

**结论**: 项目中已**完全移除**WhisperX依赖，无遗留代码。

**证据**:
- 代码搜索: 无`import whisperx`或`from whisperx`语句
- 架构变化: 从单引擎（WhisperX）迁移到双引擎时空解耦（SenseVoice + Faster-Whisper）
- 组件删除:
  - `AlignModelInfo` 类已删除
  - `_align_all_results_batched()` 方法已删除
  - 对齐模型API端点已从`model_routes.py`中移除
  - 对齐模型配置已从`model_config.py`中删除

**迁移对比**:

| 方面 | WhisperX时代 | 当前架构 |
|------|--------------|---------|
| 时间戳来源 | WhisperX强制对齐模型 | SenseVoice ONNX推理 |
| 模型数量 | 2个（转录+对齐） | 1个（仅转录） |
| 对齐精度 | 字级对齐（真实） | 双模态对齐（Level 1）或伪对齐（Level 2） |
| 文本修正 | WhisperX输出即最终 | 多源修正（Whisper/LLM）+ 伪对齐 |
| 显存占用 | 更高 | 更低（少一个对齐模型） |

#### 3. SenseVoice与Whisper双流对齐机制

**核心理念**: "时空解耦" - SenseVoice是时间领主，Whisper是文本权威

**AlignmentWorker实现细节**:

1. **三级降级策略**（质量优先 → 速度兜底）:
   - **Level 1: 双模态对齐（黄金标准）** - 使用Needleman-Wunsch算法将Whisper文本对齐到SenseVoice时间轴
   - **Level 2: Whisper伪对齐（银标准）** - 双模态对齐失败时，使用伪对齐将Whisper文本均匀分布到Chunk时间范围
   - **Level 3: SenseVoice草稿（铜标准）** - 最终兜底，直接使用SenseVoice的原始输出

2. **早期拦截机制**（防止劣质对齐）:
   - **检测0**: Whisper单词数<2 → 幻觉拦截 → 直接降级到Level 3
   - **检测1**: Whisper长度暴涨（len(W) > 3*len(SV)+10 且置信度<0.5） → 降级到Level 3
   - **检测2**: Whisper过短（len(W) < len(SV)*0.65，V3.8.1阈值提升） → 降级到Level 3

3. **对齐策略**（Level 1详细流程）:
   ```
   Whisper文本 + SenseVoice词级时间戳
     ↓
   Needleman-Wunsch序列对齐
     ↓
   生成对齐后的字级时间戳（AlignedWord）
     ↓
   能量锚点校准（可选，利用音频能量峰值）
     ↓
   VAD边界校准（确保时间戳在语音范围内）
     ↓
   重新分句（基于Whisper的精准标点）
     ↓
   语义分组（使用SemanticGrouper）
     ↓
   推送定稿SSE事件
   ```

4. **降级决策逻辑**:
   - 对齐质量分数 < 阈值（默认0.3） → 降级到Level 2
   - Level 2失败 → 降级到Level 3
   - Level 3失败 → 使用任何可用文本创建兜底字幕

**时间戳权威来源**:
- **绝对权威**: SenseVoice的字级时间戳（在Level 1和Level 3中直接使用）
- **相对权威**: Whisper的内置时间戳（仅在Level 2伪对齐中使用，需偏移校准）
- **兜底策略**: 伪对齐均匀分布（当所有时间戳不可用时）

#### 4. 字级时间戳生成方法

**SenseVoice原生时间戳**（最精确）:
- 来源: `SenseVoiceONNXService` 的ONNX推理结果
- 字段: `words` 数组，每个元素包含`word, start, end, confidence`
- 特点: 基于CTC解码直接生成，精度最高
- 使用场景: Level 1双模态对齐、Level 3直接使用

**Whisper内置时间戳**（相对精确）:
- 来源: `faster_whisper.WhisperModel` 的推理结果
- 字段: `segments[].words[]`，包含`word, start, end, probability`
- 特点: 基于注意力机制生成，英文准确度高
- 使用场景: Level 2伪对齐的时间参考（需加上chunk_offset偏移）

**伪对齐生成时间戳**（估算）:
- 方法: `PseudoAlignment.apply()`
- 算法: 在给定时间窗口内均匀分布token
- 计算公式:
  ```python
  step = (end - start) / token_count
  token[i].start = start + i * step
  token[i].end = start + (i+1) * step
  ```
- 标记: `is_pseudo=True`
- 使用场景: Whisper补刀、LLM校对后的文本映射

**跨Chunk合并时的时间戳处理**:
- `WhisperBufferPool.align_text_to_chunks()`: 将Whisper长文本对齐回原始Chunk
  1. 收集Whisper的词级时间戳（相对于pool起始时间）
  2. 收集SenseVoice的词级时间戳（作为锚点）
  3. 使用Needleman-Wunsch对齐
  4. 根据对齐路径将Whisper词分配到对应Chunk
  5. 调整时间偏移（加上chunk.start）

**字幕切分时的时间戳重算**（前端）:
- 优先策略: 使用字级时间戳精确切分
  ```javascript
  // 查找切分点所在的字
  const splitIndex = words.findIndex(w => w.end >= splitTime);
  leftWords = words.slice(0, splitIndex);
  rightWords = words.slice(splitIndex);
  ```
- 回退策略: 按字数比例估算时间
  ```javascript
  const ratio = cursorPosition / text.length;
  const splitTime = start + (end - start) * ratio;
  ```

#### 5. 创新点和亮点

##### 5.1 三级降级对齐策略（业界首创）

**创新点**: 不同于传统的"对齐成功/失败"二元结果，项目实现了三级渐进式降级策略

- **Level 1 双模态对齐**: 质量最高，但计算开销大
- **Level 2 Whisper伪对齐**: 速度快，质量中等
- **Level 3 SenseVoice草稿**: 兜底保证可用性

**优势**:
- 质量与速度的自适应平衡
- 极端情况下仍能保证输出（不会出现字幕丢失）
- 降级决策基于明确的质量阈值（对齐分数、长度比例、置信度）

##### 5.2 早期拦截机制（防御性对齐）

**创新点**: 在执行昂贵的Needleman-Wunsch对齐之前，先进行四重轻量级检测

- **检测0**: 单词数过少检测（幻觉拦截）
- **检测1**: 长度暴涨检测（Whisper幻觉特征）
- **检测2**: 长度过短检测（Whisper漏识别）
- **特权放行**: 长度暴涨但置信度高时允许通过（可能是SenseVoice漏识别）

**优势**:
- 避免劣质对齐（垃圾进垃圾出）
- 性能优化（跳过无意义的对齐计算）
- 自适应调整（V3.8.1将阈值从0.3提升到0.65，减少误判）

##### 5.3 时空解耦架构（SenseVoice时间领主 + Whisper文本权威）

**创新点**: 不同于WhisperX和业界主流方案（Whisper文本+对齐模型时间戳），项目采用反向策略

- **传统方案**: Whisper文本 + WhisperX对齐模型时间戳 → 需要额外对齐模型，显存开销大
- **本项目**: SenseVoice时间戳 + Whisper文本 → 无需对齐模型，通过Needleman-Wunsch序列对齐映射

**优势**:
- 显存占用降低（少一个对齐模型）
- 灵活性提升（文本可由多个来源修正：Whisper、LLM）
- SenseVoice语义分句优势得到充分利用

##### 5.4 伪对齐通用化（时间戳修复中间件）

**创新点**: 伪对齐不仅用于Whisper补刀，还泛化到所有文本替换场景

**支持场景**:
1. Whisper补刀后文本映射
2. LLM校对/翻译后时间戳重建
3. 用户手动编辑后时间戳估算
4. 字幕切分时的回退方案

**优势**:
- 代码复用性高（单一职责原则）
- 扩展性强（新增文本处理层无需关心时间戳）
- 置信度标记清晰（is_pseudo标记便于后续优化）

##### 5.5 能量锚点校准（音频特征辅助）

**创新点**: 利用音频能量峰值微调词边界，提升对齐精度

**算法**:
1. 计算词时间范围内的音频能量
2. 定位能量峰值位置
3. 如果峰值偏离词中心>0.1秒，则调整词边界
4. 保持词时长不变，仅平移位置

**优势**:
- 物理特征校准（能量峰值对应发音中心）
- 视觉效果优化（字幕出现/消失与音频同步更自然）

##### 5.6 智能长句拆分（语义完整性保护）

**创新点**: 在物理停顿切分基础上，增加语义完整性检查

**策略**:
- **不完整结尾词检测**: 介词、连词、量词等不能作为句尾 → 延迟切分
- **续接词检测**: 下一句以"但是"/"所以"等开头 → 合并为一句
- **完整词排除列表**: "事件"/"案件"等虽以量词结尾但完整 → 允许切分

**优势**:
- 阅读体验优化（避免"这是一个"单独成句）
- 多语言支持（抽象LanguageStrategy，支持中英日）
- LLM校对友好（保证上下文完整）

##### 5.7 乱序执行、顺序提交流水线（V3.5架构）

**创新点**: FastWorker并发处理（CPU），SlowWorker顺序处理（GPU），AlignmentWorker对齐定稿

**架构**:
```
VAD切分 → [FastWorker 并发] → SequencedQueue → [SlowWorker 顺序] → Queue2 → [AlignmentWorker] → 定稿
```

**优势**:
- CPU/GPU资源利用率提升（FastWorker全速并发，SlowWorker按序处理）
- 简单Chunk秒级完成，复杂Chunk后台慢跑（不阻塞流水线）
- Whisper上下文连贯性保证（顺序处理维持时间轴）

##### 5.8 V3.10智能补刀-跳过机制

**创新点**: AlignmentWorker支持快速路径，当SlowWorker判断Whisper补刀无意义时直接跳过

**条件**:
- `ctx.whisper_skipped=True` → 直接使用SenseVoice草稿作为定稿
- 跳过昂贵的对齐计算

**优势**:
- 性能优化（避免无意义对齐）
- 质量保证（SenseVoice高质量时无需Whisper干扰）

---

### conclusions

1. **项目已完全移除WhisperX依赖**，从单引擎强制对齐迁移到双引擎时空解耦架构（SenseVoice时间领主 + Whisper文本权威）

2. **伪对齐算法是项目的核心创新**，提供通用化的时间戳修复能力，支持Whisper补刀、LLM校对、用户编辑等所有文本替换场景

3. **三级降级对齐策略是业界首创**，通过Level 1双模态对齐 → Level 2伪对齐 → Level 3草稿的渐进式降级，实现质量与速度的自适应平衡

4. **早期拦截机制大幅优化对齐质量**，通过四重检测在对齐前拦截劣质Whisper输出（幻觉、暴涨、过短），避免"垃圾进垃圾出"

5. **时间戳权威来源清晰**: SenseVoice原生时间戳（最精确） > Whisper内置时间戳（相对精确） > 伪对齐估算时间戳（兜底）

6. **字级时间戳生成贯穿全流程**: 从SenseVoice ONNX推理 → AlignmentWorker对齐 → 字幕切分 → 前端显示，所有阶段均保留和维护字级时间戳

7. **乱序执行架构（V3.5）解耦CPU/GPU资源竞争**，FastWorker并发+SlowWorker顺序，实现吞吐量最大化同时保证Whisper上下文连贯性

8. **能量锚点校准是独特优化**，利用音频物理特征（能量峰值）微调词边界，提升字幕视觉同步效果

9. **语义完整性保护优于主流方案**，通过不完整结尾词检测和续接词检测，避免机械分句破坏语义完整性

10. **V3.8.1持续优化**: Whisper过短检测阈值从0.3提升到0.65，减少误判，体现工程经验积累

---

### relations

#### 1. 对齐算法之间的关系

```
Needleman-Wunsch序列对齐 (AlignmentService)
    ↓ 用于
双模态对齐 (AlignmentWorker Level 1)
    ↓ 失败时降级到
Whisper伪对齐 (AlignmentWorker Level 2)
    ↓ 调用
PseudoAlignment.apply()
    ↓ 生成
is_pseudo=True的字级时间戳
```

#### 2. Worker之间的协作关系

```
FastWorker (CPU并发)
    ↓ 推送草稿到
StreamingSubtitleManager
    ↓ 同时提交到
SequencedQueue (序列化队列)
    ↓ 按序取出给
SlowWorker (GPU顺序)
    ↓ Whisper推理后提交到
Queue2
    ↓ 取出给
AlignmentWorker (CPU对齐)
    ↓ 对齐+分句后推送定稿到
StreamingSubtitleManager (替换草稿)
```

#### 3. 时间戳生成链路

```
SenseVoice ONNX推理
    ↓ 生成
原生字级时间戳 (words数组)
    ↓ 传递给
AlignmentService.align()
    ↓ 作为锚点与Whisper文本对齐
    ↓ 生成
AlignedWord列表
    ↓ 转换为
WordTimestamp列表
    ↓ 传递给
SentenceSplitter.split()
    ↓ 分句生成
SentenceSegment列表
    ↓ 推送到前端
    ↓ 用户切分时
    ↓ 再次使用字级时间戳精确切分
```

#### 4. 降级策略触发关系

```
AlignmentWorker._align_and_fallback()
    ↓ 早期拦截检测
    ├─ 检测0: 单词数<2 → 跳过对齐 → Level 3
    ├─ 检测1: 长度暴涨+低置信度 → 跳过对齐 → Level 3
    └─ 检测2: 长度过短(ratio<0.65) → 跳过对齐 → Level 3
    ↓ 通过检测后尝试
Level 1: 双模态对齐
    ↓ 失败(对齐分数<0.3)或异常 → 降级到
Level 2: Whisper伪对齐
    ↓ 失败或异常 → 降级到
Level 3: SenseVoice草稿
    ↓ 失败或异常 → 最终兜底
Level 4: 任何可用文本创建单句字幕
```

#### 5. 伪对齐的多场景应用关系

```
PseudoAlignment.apply() (核心方法)
    ↓ 被调用于
    ├─ AlignmentWorker Level 2 (Whisper补刀后)
    ├─ LLM校对后 (文本替换后)
    ├─ LLM翻译后 (生成目标语言时间戳)
    ├─ 字幕切分回退 (无字级时间戳时)
    └─ WhisperBufferPool._fallback_alignment() (兜底对齐)
```

#### 6. 配置与策略的关系

```
SentenceSplitter配置
    ├─ FastWorker: 软上限5秒 (快速分句，允许长句)
    ├─ AlignmentWorker: 软上限5秒 + 硬上限20秒 + 依赖标点
    └─ LanguageStrategy: 多语言策略 (中文/英文/日文)

AlignmentConfig
    ├─ match_score=2 (Needleman-Wunsch匹配得分)
    ├─ mismatch_penalty=-1
    ├─ gap_penalty=-2
    ├─ enable_energy_anchor=True (能量锚点校准)
    └─ enable_vad_calibration=True (VAD边界校准)
```

#### 7. SSE事件流关系

```
FastWorker处理完成
    ↓ 推送
subtitle.draft SSE事件 (草稿字幕)
    ↓ 前端显示淡色字幕
    ↓ 等待
AlignmentWorker处理完成
    ↓ 推送
subtitle.chunk_replace SSE事件 (定稿替换)
    ↓ 前端替换为正常字幕
```

#### 8. V3.10快速路径关系

```
SlowWorker.process()
    ↓ 判断是否需要Whisper补刀
    ├─ 需要 → ctx.whisper_skipped=False → 正常流程
    └─ 不需要 → ctx.whisper_skipped=True → 跳过推理
        ↓ 传递给
AlignmentWorker.process()
    ↓ 检测到whisper_skipped=True
    ↓ 直接使用SenseVoice草稿作为定稿
    ↓ 跳过对齐计算
```
