• 总体评估

  - 双模态对齐（Whisper 主文本 + SenseVoice 时间锚点）用现有 VAD 切块→双模型推理→序列对齐的流程，技术路径与 WhisperX/
    业内对齐系统一致，可行性高，难点主要在高质量对齐算法和任务调度实现而非算法研究（docs/新架构阐述.md:21, docs/新架构
    阐述.md:95）。
  - 该方案天然满足“完全本地运行”的约束：两端模型均已有本地实现，且 SenseVoice ONNX + Faster-Whisper 的依赖堆栈已存在于项
    目中，只需在控制层序列化调度（docs/新架构阐述.md:71, docs/新架构阐述.md:215）。

  准确度与质量

  - Whisper Large/Medium 提供远高于 SenseVoice 的文本语义、标点和自纠能力，若全部句子都走 Whisper，则理论准确度可以逼近
    直接跑 Faster-Whisper 的水平（实际英文 WER 通常 5% 左右，远优于当前 SenseVoice-only 的 15%+），而 SenseVoice 仅负责
    时间戳，避免其“置信度高但错误”问题（docs/新架构阐述.md:25, docs/新架构阐述.md:111）。
  - 对齐算法只需保证词边界合理即可实现 90% 以上的时间映射，difflib/DTW 已可打底，但要处理 Whisper 与 SenseVoice 在数字、
    缩写、噪音处的分词差异，建议引入额外的预处理（统一大小写、去除标点、建立同义词替换表）以及锚点置信度权重，避免错配导
    致的错位字幕（docs/新架构阐述.md:103, docs/新架构阐述.md:111）。

  速度/性能

  - 计算量几乎等同“Whisper 全量 + 极轻量 SenseVoice”串行；以 3060 级显卡估算 1 分钟音频需要 10–20 秒，SenseVoice 仅占 1–
    2 秒与 0.1 秒融合处理，整体速度约为当前 SenseVoice 流水线的 10–20 倍，但与直接跑 Whisper 接近且更加稳定（docs/新架构
    阐述.md:133, docs/新架构阐述.md:137）。
  - 新方案在感知体验上反而更好：SenseVoice 快流草稿 + Whisper 慢流回写可保持“近实时预览”并在 1–2 秒内纠正，这与现有 SSE
    推模式兼容，属于 UX 提升而非牺牲（docs/新架构阐述.md:151, docs/新架构阐述.md:199）。

  可复用模块

  - 现有的任务创建、SSE 基础设施、前端任务/频道管理、Silero VAD、Demucs 人声分离、SenseVoice
    ONNX 模块，以及 StreamingSubtitleManager/SSEManager 的事件管线都可直接保留，仅需扩展事件类
    型（llmdoc/agent/video_to_srt_gpu_complete_pipeline_architecture_report.md:46, llmdoc/agent/
    video_to_srt_gpu_complete_pipeline_architecture_report.md:96）。
  - 预设体系和阈值模块可部分复用：仍可用 SolutionConfig 管理“极速（SenseVoice 草稿）”“精校（双模对齐）”等模式，只需重
    写“增强模式”的含义，把 smart_patch/deep_listen 改为“是否执行 Whisper 最终化/是否追加 LLM 校对”等新的维度（llmdoc/
    agent/video_to_srt_gpu_complete_pipeline_architecture_report.md:69）。

  需重做或新增的关键环节

  - TranscriptionService._process_video_sensevoice 及 _transcribe_chunk_with_fusing / _post_process_enhancement
    必须重构为“串行双模型 + 对齐”的主干，历史的复核触发和仲裁逻辑可以删除或转化为“草稿→成品”状态
    机（llmdoc/agent/video_to_srt_gpu_complete_pipeline_architecture_report.md:50, llmdoc/agent/
    video_to_srt_gpu_complete_pipeline_architecture_report.md:60）。
  - WhisperService 不再是条件调用，需要支持全量 chunk 级推理、可选 GPU/CPU 分时、int8/float16 自动切换等；
    SenseVoiceONNXService 要支持快速返回词级时间并暴露刑 words 数据供对齐函数使用。
  - 新增“锚点对齐”模块（纯 Python/NumPy），需包含：文本正则化、锚点识别、Gap 填充、孤词插值、长度差异裁剪等步骤，并在失
    败时自动回退到 Whisper 时间戳或降级的声学估计。
  - SSE 管线增加 subtitle.draft/subtitle.overwrite、前端列表支持同 ID 覆盖、任务频道对“草稿/定稿”进行视觉区分（docs/新架
    构阐述.md:278, docs/新架构阐述.md:333）。
  - 旧的 needs_whisper_patch、_whisper_text_patch_with_arbitration、伪对齐与复核事件需要整体淘汰，以防逻辑冲突。相应数据
    库/状态字段若绑定旧模式，要同步迁移。

  优化与补充建议

  1. 多档模型与推理策略：保留“极速”预设使用 SenseVoice+LLM 简校（原链路），同时新增“高精”预设走双模对齐，允许用户按算力
     和时效在 UI 中切换；这让已有代码被最大化复用。
  2. 对齐质量监控：实现 WER/对齐成功率的离线统计与阈值报警，一旦 difflib 匹配失败可自动改用 Whisper 原时间或延长
     SenseVoice 片段，避免极端情况下字幕错位。
  3. 缓存与增量：由于 Whisper 为全部瓶颈，可在任务目录缓存 chunk 的 mel/features，失败重跑时直接复用；若用户二次修改字
     幕，可基于对齐结果做增量生成而无需重复 SenseVoice。
  4. LLM 后处理整合：双模对齐完成后仍可保持原有 LLM 校对/翻译层，只需调整输入结构（词级数组而非 SenseVoice
     SentenceSegment），这样高阶校对能力得以保留。

  综上，双模态对齐方案在本地环境中完全可实现，并能显著提升准确率与用户体验；代价是需要重写转录主流程与事件协议，并接受整
  体速度等同 Whisper 的事实。通过保留现有 VAD、任务管理、SSE 系统与 SenseVoice 模块、对齐失败回退策略、缓存优化等手段，
  可以在可控工作量内完成 v3 重构。下一步建议优先实现对齐原型（Python 函数 + 单元测试）并在 CLI 下跑基准，随后再逐层替换
  控制逻辑。