这是 **AnchorFlux v3.3.0 "Neural-Native" 架构** 的最终完整设计方案。

本方案彻底摒弃了 v3.2 中脆弱的“正则规则+时空硬映射”逻辑，转为以 **"模型级信号(Segmentation/LLM)"** 为决策核心，以 **"弹性对齐算法(Anchored NW)"** 为骨架的新一代架构。

---

# AnchorFlux v3.3.0 Architecture: Neural-Native

## 核心哲学

1. **Time Master (时钟源)**: **SenseVoice (快流)**。利用其极高的字级时间戳精度，作为所有字幕的物理时间基准。
2. **Content Master (内容源)**: **Whisper (慢流)**。利用其在大模型下优秀的文本连贯性和语义识别能力。
3. **Structure Master (结构源)**: **LLM (SLM)**。取代所有 regex/标点模型，负责语义分句、标点恢复和文本校对。

---

## Phase 1: 预处理与分诊 (Pre-L0)

**目标**：解决“多人对话切分”与“静音覆盖”的矛盾，确保送入 ASR 的是纯净的单人片段。

### 1.1 核心策略：VAD + Segmentation "Hard Mask" (硬掩码)

单纯的 Segmentation 模型倾向于平滑过渡（导致字幕拖尾进静音区），单纯的 VAD 分不清说话人。我们采用**“交集策略”**：

1. **VAD Stream (裁纸刀)**: 运行 Silero VAD，生成高精度的 `Speech_Mask` (0/1 序列)。
2. **Diarization Stream (染色剂)**: 运行 `pyannote/segmentation-3.1` (GPU)，生成 `Speaker_Tracks`。
* *显存控制*: 强制使用 **Sliding Window (滑窗)** 机制，显存占用恒定 <3GB，不随音频长度增加。
* *身份统一*: 对全长音频 Embedding 进行一次性 **Global Clustering**，确保第 1 分钟的张三和第 60 分钟的张三 ID 一致。


3. **Hard Mask Intersection (硬掩码运算)**:


* **逻辑**：只有当“Segmentation 认为是张三” **且** “VAD 认为有声音”时，才保留该片段。



### 1.2 输出产物

* **Input for L0**: `PreciseTimeline` (包含精确到毫秒的 `start`, `end`, `speaker_id`)。

---

## Phase 2: 桥接与双流执行 (L0 - L2)

**目标**：基于精确时间轴进行调度，杜绝多人语音混在一个 Batch 中。

* **L0 (Bridge)**:
* **Flush 规则**：不再依赖模糊的 Silence 阈值。严格依照 `PreciseTimeline` 中的 `Speaker Change` 事件触发 Flush。
* **短句粘连**：对于同一说话人的连续短句（间隙 < 0.5s），Bridge 负责合并为一个 Batch 发送，提供足够上下文。


* **L1 (Fast Worker)**: SenseVoice 运行，产出带精确时间戳的 **[字符骨架]**。
* **L2 (Slow Worker)**: Whisper 运行，产出高质量文本的 **[语义血肉]**。

---

## Phase 3: 弹性对齐层 (L4 - The Core)

**目标**：将 Whisper 的“肉”完美挂载到 SenseVoice 的“骨”上，解决多语言同音字导致的对齐断裂。

### 3.1 算法核心：Confidence-Anchored Hybrid-Scoring NW

采用改进的 **Needleman-Wunsch** 动态规划算法。

#### A. 锚点切分 (Confidence Anchoring)

为了防止长文本（>30s）对齐误差累积，利用 SenseVoice 的置信度信息：

1. **锚点选择**：筛选 SenseVoice 结果中 `confidence > 0.98` 且 `length > 1` 的词（如“人工智能”）。
2. **分段对齐**：以锚点为界，将长序列切分为多个微小的区间（3-5秒）。
3. **区间插值**：仅在锚点之间的“模糊区”运行 NW 算法。如果 Whisper 多出了字，则在两个锚点的时间差内进行线性插值。

#### B. 混合评分机制 (Hybrid Scoring)

解决“同音异形字”导致的 mismatch 问题。

**L4 对齐器打分逻辑 (`score_matrix`)**:

1. **Level 1: Hard Match (硬匹配)**
* `if char_A == char_B`: **Score +10**
* *适用*：绝大多数准确识别的情况。


2. **Level 2: Soft Match (软匹配 - 语言特化)**
* **中文 (CN)**:
* `if pinyin(char_A) == pinyin(char_B)`: **Score +8**
* *实现*：使用 `pypinyin.lazy_pinyin` (无声调)，解决“必须/必需”、“座/坐”问题。


* **英文 (EN)**:
* `if Levenshtein(word_A, word_B) < threshold`: **Score +8**
* *实现*：解决形态差异 (Run/Running)。


* **日文 (JP) - "地狱模式"特解**:
* **预处理**：将 Fast 流和 Slow 流文本**全量转换为罗马音 (Romaji)** (使用 `pykakasi`)。
* **Mapping**：建立 `Romaji_Index -> Original_Char_Index` 的映射表。
* **对齐**：在罗马音序列上跑 NW 算法。
* **回溯**：将对齐后的时间戳通过映射表贴回原始日文汉字/假名。




3. **Level 3: Mismatch (不匹配)**
* `else`: **Score -5** (丢弃 Fast 流该字符，或标记为 Slow 流新增字符)。



---

## Phase 4: 语义重构层 (L6 - The Brain)

**目标**：废弃规则脚本，使用 SLM 完成所有结构化任务。

### 4.1 模型选型

* **Model**: **Qwen2.5-3B-Instruct (Int4)** 或 **Llama-3.2-3B-Instruct**。
* **理由**：3B 模型在端侧 GPU 显存占用 <2GB，推理速度极快，且指令遵循能力远超 7B 以下旧模型。

### 4.2 One-Pass Prompt 策略

拒绝“流水线式”多次推理，一次 Prompt 解决所有问题：

> **System Prompt**:
> "你是一个专业的字幕编辑。请处理以下 ASR 识别文本：
> 1. **纠错**：去除重复词、口吃和无意义语气词。
> 2. **标点**：根据语义补充逗号、问号、感叹号（严禁使用句号）。
> 3. **分行**：按语义逻辑换行，每行一句话。
> 4. **严禁**：不要修改原意，不要输出任何解释性文字。"
> 
> 

* **输入**: L4 输出的已对齐长文本（无标点/标点混乱）。
* **输出**: 标准化、分好行的字幕块。
* **时间戳回填**: L6 将 L4 的字级时间戳映射回 LLM 输出的新文本序列（通过简单的字符匹配）。

### 4.3 组件去留

* **Regex Splitter**: **删除**。
* **L3 标点模型**: **删除/降级**。如果为了极致速度，可保留作为降级方案；但在 v3.3 标准模式下，由 LLM 接管。

---

## Phase 5: 导出与渲染 (L7 - The Stylist)

**目标**：分离“数据存储规范”与“用户视觉偏好”。

### 5.1 核心原则

* **中间层 (Store)**: 永远存储 **全标点、标准语法** 的文本。确保数据有最大的语义价值（便于未来翻译/搜索）。
* **导出层 (Export)**: 根据用户选择的 **Style Profile** 动态渲染。

### 5.2 Style Profiles (样式预设)

在生成 `.srt` / `.ass` 时应用：

1. **Standard (Default for EN)**: 原样输出，保留所有标点。
2. **Cinematic (Default for CN/JP)**:
* **CN**: `,` `、` -> `(空格)`; `。` -> `(空)`; 保留 `？` `！`。
* **JP**: 去除所有 `、` `。`，替换为全角空格。


3. **Strict**: 强制语法检查（用于听障字幕 CC）。

---

## 总结：AnchorFlux v3.3 升级路线图

| 模块 | v3.2 现状 (旧) | **v3.3 目标 (新)** | 关键技术点 |
| --- | --- | --- | --- |
| **Pre-L0** | Silero VAD | **VAD + Pyannote Segmentation (Hard Mask)** | Intersection Logic, Global Clustering |
| **L0 Bridge** | VAD Gap Flush | **Speaker Change Flush** | PreciseTimeline 驱动 |
| **L4 对齐** | 简单映射 | **Anchored Hybrid-Scoring NW** | 锚点切分, 拼音/罗马音软匹配 |
| **L6 处理** | Regex + Bert | **SLM (3B Int4) One-Pass** | 语义分行, 标点恢复, 纠错 |
| **L7 导出** | 硬编码格式 | **Dynamic Style Profiles** | 渲染时去标点 (电影感适配) |

此方案解决了你面临的“多说话人切分不清”、“时间戳漂移”、“规则维护成本高”三大核心痛点，是面向 2026 年技术标准的工程最优解。