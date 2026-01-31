# 快流词级/字级合并与置信度调整设计

> Type: Architecture | Status: Active

## 1. Summary

* **Goal**: 重构快流 token 合并与置信度计算，使中日文等无空格语言保持字级精度，同时为未来语言扩展提供可插拔策略，并将内部决策置信度与前端展示置信度解耦。

## 2. Diagram

* SenseVoice CTC tokens -> TokenNormalizer(清洗标签/标点分离) -> TokenMergeStrategy(语言/脚本策略) -> WordTimestamp(confidence_raw/confidence_display_raw) -> SentenceConfidence(双轨聚合) -> ConfidenceMapper(display_confidence) -> SemanticBuffer/StreamingSubtitle

## 3. Key Components

* `backend/app/services/sensevoice_onnx_service.py`: 输出 `raw_tokens` + `words`，将“标点时间戳并入相邻词”的处理前移到合并阶段，避免下游重复清洗并确保统一入口。
* `backend/app/services/token_merge_service.py`(新增): 合并策略注册与路由；优先使用 token 边界（空格/▁），若无边界则使用 Unicode 脚本策略（Han/Kana/Hangul 默认不跨字合并，Latin/Number 连续合并），并支持语言级可插拔分词器（可选）。
* `backend/app/services/token_merge_service.py`: **新增 SenseVoice 显示置信度校准**，将 CTC 原始置信度压缩到显示口径区间（当前 0.10~0.60），用于 `confidence_display_raw`，避免映射后“全 98%”饱和。
* `backend/app/core/asr/models.py`: ASR 统一模型补齐 `confidence_display_raw`/`token_type`/`raw_tokens` 字段，确保快流链路可传递。
* `backend/app/models/sensevoice_models.py`: 扩展 `WordTimestamp`（新增 `confidence_raw`、`confidence_display_raw`、`token_type`）；扩展 `SenseVoiceResult`（新增 `raw_tokens`）；扩展 `SentenceSegment`（新增 `confidence_display_raw`，`display_confidence` 由映射产生）。
* `backend/app/services/segmentation/sentence_splitter.py`: 构建句子时使用词级时间戳（word_timestamps）不丢失字级精度；句级 `confidence` 使用严格口径（如 min/低分加权），句级 `confidence_display_raw` 使用平滑口径（如按时长加权均值）。
* `backend/app/core/confidence_mapper.py`: `display_confidence` 仅基于 `confidence_display_raw` 映射，保持内部 `confidence` 用于补刀/熔断等决策逻辑。
* `backend/app/pipelines/workers/fast_worker.py`: 不再二次合并标点时间戳，仅透传 `raw_tokens/words` 供标点与语义缓冲使用。
* `backend/app/services/punctuation/semantic_buffer.py`: 若检测到 word_timestamps 粒度不足（如仅1词覆盖全段），可回退到 raw_tokens 或脚本切分后的伪 word 列表以保证时间轴单调与断句精度。
* `backend/app/services/punctuation/semantic_buffer.py`: 语义缓冲切分句子时继承词级时间戳与置信度，避免“全段98%+无高亮”的前端表现。
* `backend/app/services/punctuation/semantic_buffer.py`: 语义缓冲输出的词级时间戳统一转换为 `WordTimestamp`，确保 `SentenceSegment.to_dict()` 序列化不再触发 dict 类型错误。
* `backend/app/models/sensevoice_models.py`: `SentenceSegment.to_dict()` 对词级列表做容错，避免意外 dict 渗入导致任务中断，同时记录告警用于追踪根因。
* `backend/app/services/streaming_subtitle.py`: 前端展示使用 `display_confidence`，词级高亮使用 `confidence_display_raw`；内部决策仍使用 `confidence_raw` 或句级 `confidence`。
* `backend/tests/conftest.py`: 测试导入路径校准，确保 venv 下 pytest 可直接 `import app.*`。
