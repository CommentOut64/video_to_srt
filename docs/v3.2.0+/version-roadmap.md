# V3.2 系列内部版本规划

> 基线从 `v3.2.0` 起算。该系列全部为内部版本，只需保证后端/流水线自洽，无需考虑用户端额外适配。每个版本都是一个能力闭环：完成后即可独立测试、可灰度、可回滚。

---

## v3.2.0 – 架构基线与 Worker 容器化

### 目标
- 完成 ASR 抽象接口、ModelManager V2，以及 SenseVoice / Whisper / DualStream / Dummy 引擎适配。
- FastWorker / SlowWorker 重构为“模型容器”，彻底移除内部的分句与对齐逻辑。
- 将旧版分句（CTC/基于 words 的 heuristics）与对齐（现有伪对齐器）抽离成独立服务，后续即可渐进替换实现。

### 统一模型管理系统（ModelManager V2）
- **统一注册表 + 类型适配**：以 `ModelSpec` 描述所有模型（ONNX、PyTorch、Faster-Whisper 乃至未来新引擎），抽象 `ModelLoader` 接口，按框架实现子类（OnnxLoader、TorchLoader、ExternalServiceLoader），可随时扩展。
- **下载 / 校验流水线**：优先读取打包模型；若缺失则触发 Downloader（支持断点续传 + 哈希校验 + 多源镜像）。对“随发布打包”的模型仅执行校验，不重复下载。
- **生命周期/资源治理**：
  - 统一 `load_model / warmup / unload / reload` 流程，所有 Worker/Engine 只能通过 ModelManager 获取实例。
  - 集成显存预算池与 CPU 调度器：加载时评估 GPU/CPU 占用，不足则自动降级或驱逐低优先级模型；暴露线程/设备分配接口给 SenseVoice/Whisper。
  - 实时监控内存/显存/句柄，提供 Prometheus exporter；触发阈值时记录日志并降级。
- **配置与参数暴露**：集中维护模型的输入/输出维度、可调参数、默认语言、支持特性等，供前端设置面板和 API 查询；所有参数变更通过 Manager 分发，避免散落各处。
- **可观测性**：记录下载/加载/热身耗时，统计使用频率，提供指令查询当前模型状态/设备占用。

### 主要工作
1. **ASR 接口与模型管理**
   - `backend/app/core/asr/*`: ASREngine 抽象、数据模型、能力枚举、置信度/时间戳工具。
   - `backend/app/services/model_manager_v2.py`: 模型注册、加载、显存预算、单例访问。
   - 引擎适配器：SenseVoiceEngine、WhisperEngine、DualStreamEngine、DummyEngine；均注册到 `ASREngineFactory`。
2. **Worker 容器化**
   - `FastWorker` 仅负责：拉取音频 chunk、调用 draft 引擎、输出草稿（含 words、metadata）。
   - `SlowWorker` 仅负责：接收句子/片段、调用 patch 引擎、输出补刀结果。
   - 所有分句、对齐、熔断判断从 worker 中剥离。
3. **分句/对齐服务（旧逻辑版）**
   - `DefaultSegmenter`：封装原 CTC/word-based 分句算法，对外暴露 `segment(words, text, metadata)` 接口。
   - `DefaultAligner`：封装原伪对齐逻辑，支持 SenseVoice 词级时间戳与 Whisper 伪对齐。
   - Worker 通过接口调用，不再直接操纵 `words`。

### 测试/验收
- DummyEngine 覆盖 Worker/Pipeline 的单元与集成测试，保证无模型环境下也能跑 CI。
- 端到端对比 `v3.1.2`：字幕质量差异 <2%、性能差异 <10%。
- 特性开关 `USE_NEW_ASR_ENGINE` 可回滚；旧配置自动映射到新工厂。

---

## v3.2.1 – Chunk 级语言检测基础设施

### 目标
- 在前处理阶段（人声分离后）引入 `speechbrain/lang-id-voxlingua107-ecapa`，为 **每个 Chunk** 打上语言标签。
- 把语言信息写入 `SubtitleChunk`、`SentenceSegment`、ASR metadata，成为后续标点模型、Bridge、置信度高亮的“路由开关”。

### 主要工作
1. **LangID 管道**
   - 新建 `LanguageDetector` 服务：接受 PCM/np.ndarray，输出 {chunk_id: lang, confidence}。
   - 与 ModelManager V2 集成，支持 CPU/GPU 推理；提供并发控制与缓存。
2. **数据流改造**
   - FastWorker 调用分句前必须携带 `chunk.language`。
   - `ASRResult.metadata.language` 作为默认语言；若 SenseVoice 自带标签，则两者对齐并记录差异。
   - 增加调试指标：语言置信度、覆盖率、不一致统计。
3. **测试**
   - 搭建 ≥5 种语言基准集（zh/en/ja/ko/yue），测 Chunk 级准确率 ≥95%。
   - 压测并确认 LangID 不会拖慢前处理（吞吐下降 <5%）。

---

## v3.2.2 – 语言驱动的标点分句与对齐重构

### 目标
- 基于 LangID 选择合适的“标点恢复小模型”，实现“去标点 → 模型添加 → 按标点分句”的新链路。
- 让对齐服务适配新的句子边界，并彻底与 worker 解耦。

### 主要工作
1. **PunctuationSegmenter**
   - 输入：SenseVoice 原始文本 / words + 语言标签。
   - 流程：去除模型原始标点 → 选择语言特定标点模型 → 恢复标点 → 根据标点/语言规则切分句子。
   - 需支持 fallback：标点模型不可用时退回 `DefaultSegmenter`。
2. **AlignmentService 改造**
   - 以 Segmenter 输出的句子为单元，再结合词级时间戳生成最终字幕。
   - 兼容多语言、句内混合语言（借助 metadata + 字符分类兜底）。
3. **测试**
   - 多语言回归：句边界差异率 ≤2%，对齐误差 ≤旧版。
   - Fallback 覆盖：标点模型缺失/超时时仍能输出结果。

---

## v3.2.3 – Bridge 层与慢流时序升级

### 目标
- 引入 `Bridge` 层，在快流与慢流之间建立缓冲/调度/上下文管理，改变“慢流按 Chunk 输入”的旧模式。
- 让慢流基于 **快流分句后的句子集合（约 30s）** 输入 Whisper，缓解语义切断和幻觉问题。

### 主要工作
1. **Bridge 设计**
   - 句子队列：快流输出的句子即时入队，Bridge 按时间长度/句数聚合（默认 20–30s）。
   - 饿死保护：若慢流长时间未收到数据，Bridge 强制 flush。
   - 提示词构建：使用滑动窗口（含语言、置信度）自动生成 Whisper prompt。
2. **SlowWorker 重构**
   - 改为消费 Bridge 批次，逐批调用 Whisper。
   - 结果回写 Bridge，再由 Subtitle 管理器落地；快慢流相互兜底（幻觉检测 + 回退策略）。
3. **测试**
   - DummyEngine 模拟不同置信度/延迟，验证 Bridge 队列、提示词、兜底逻辑。
   - 实际 1–2 分钟音频 smoke test：端到端延迟 ≤ 旧版；Whisper 幻觉率显著下降。

---

## v3.2.4 – 多语言置信度高亮与前端联动

### 目标
- 利用 LangID + 新分句结果，统一不同语言的字/词级置信度渲染，并完善“用户编辑后高亮即时消失”的体验。

### 主要工作
1. **后端 token 处理**
   - `_merge_tokens_to_words`（或新 processor）根据语言选择拆分策略，并在词结构中写入 `language`。
   - 句内多语言使用“语言优先 + 字符分类兜底”策略，确保不会整句高亮。
2. **前端改造**
   - `renderTextWithHighlight` 根据 `word.language` 决定空格/排版；统一 warning/critical 判定。
   - `projectStore.updateSubtitle` 对比文本差异 → 映射到 word 区间 → 局部清理高亮，其他不受影响；编辑态依旧不显示高亮。
3. **测试**
   - 多语言 e2e：中/英/粤/日样例、句内混合、高亮清除场景。
   - 前端模拟数据单测 + 手动验收，确保预览/编辑模式切换正确。

---

## v3.2.5（可选后续）

- 若需继续迭代，可在此版本引入第三方 ASR 引擎、完善 ModelManager 的显存调度、开放引擎列表 API 等；与当前需求无直接关系，后续另行排期。

---

## 互斥与依赖总结
- `v3.2.0` 完成后，Fast/Slow Worker 不再持有分句/对齐逻辑，否则无法在后续版本中无痛替换实现。
- `v3.2.1` 的 LangID 是 `v3.2.2` 标点分句、`v3.2.3` Bridge 提示词、`v3.2.4` 置信度高亮的前提。
- 每版上线前需完成对应单元 / 集成 / 回归测试，并确保 `USE_NEW_ASR_ENGINE` 等特性开关可回滚。
