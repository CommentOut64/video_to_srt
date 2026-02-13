# Pyannote 软切重构与 Phase 5 收尾方案（v3.2.3）

> Type: Architecture | Status: Draft
> Version: V3.2.0+dev.20260213.01
> Scope: 仅覆盖说话人切分与 L5/L6 后处理，不改动 ASR 核心识别模型
> Non-Goal: 不追求一次性达到 100% 完美切分；不再继续堆叠硬边界补丁

## 1. Summary

* **Goal**: 把 Pyannote 从“硬切命令源”改为“说话人证据源”，在不破坏分层的前提下，显著降低漏切，并压制英文首词/单词句错切。
* **核心判断**:
  - 当前主要矛盾不是“缺少补丁”，而是“边界使用方式错误”：把 diarization/segmentation 边界直接当切句指令，导致时间基误差被 L6 放大。
  - 软切的关键不是弱化说话人优先级，而是把“说话人变化必须切”改造成“说话人变化必须在局部窗口内找到可落点锚点切”。
* **切分优先级（新口径）**:
  - `P0 强约束`: 禁止句中硬切；禁止由单一抖动边界直接切句。
  - `P1 主目标`: 说话人变化信号（必须触发“待切窗口”）。
  - `P2 落点锚点`: 停顿与词边界（优先）> 句末标点（次优）> 语义边界（仅同分候选时决策，不新增链路依赖）。
  - `P3 兜底`: 若窗口内无高质量锚点，延迟到最近可接受停顿处切分，并记录风险标签，不做句中强插切。
* **三步实施 + 收尾**:
  - 第一步（减法）: 下线硬切补丁链路，恢复 L5/L6 简洁职责。
  - 第二步（设计）: 建立“证据轨 + 锚点选择”的软切架构与契约。
  - 第三步（实施）: 分批落地、灰度评估、参数收敛。
  - 收尾: 完成 Phase 5（模式策略与无 GPU 回退）并并入主流程。

## 2. Diagram

* `Pyannote Diarization(exclusive)` -> `词级 overlap 归因(主) + nearest 归因(仅兜底)` -> `SpeakerEvidenceTrack` -> `L6 候选边界评分` -> `锚点选择(停顿/标点/语义)` -> `句段输出`
* `说话人变化` -> `生成待切窗口` -> `窗口内找锚点` -> `命中则切` -> `未命中则延迟到最近可接受停顿并标记 risk=deferred_cut`
* `ModePolicy.dual_full` -> 全量软切
* `ModePolicy.smart_patch` -> 仅高置信 speaker-change 生效，低置信退化为停顿+文本
* `ModePolicy.fast_only` -> 禁用 Pyannote，走停顿/文本切分（无 GPU 默认）

## 3. Key Components

* `docs/v3.2.3/说话人切分-破坏式重构开发文档.md`: 作为母文档保留；本方案定义其在“切分策略”维度的替换路径，尤其接续 `11.6 Phase 5`。
* `backend/app/services/timeline/diarization_service.py`: 释放 Pyannote 能力。
  - 默认取消固定 `num_speakers=3`，改为“已知人数才传 num_speakers，未知时优先 min/max 或自动估计”。
  - diarization 结果优先使用 `exclusive` 轨道做词级归因，避免重叠段直接制造冲突切口。
  - 保存每词归因证据分（overlap 时长占比、次优 speaker 差值），供下游软切评分使用。
* `backend/app/services/timeline/speaker_timeline_service.py`: 时间线只产“事实”，不产切句指令。
  - 输出 turn 仍保留，但 turn-change 不再直接映射为强制切点。
  - 额外输出 `SpeakerEvidenceTrack`（时间窗级变化证据），替代 `forced_boundary_times` 角色。
* `backend/app/services/alignment/types.py`: 契约重构。
  - 新增 `SpeakerEvidencePoint`（建议字段：`time_sec`, `left_speaker`, `right_speaker`, `change_score`, `evidence_source`, `window_start`, `window_end`）。
  - `L6Input` 删除/废弃 `forced_boundary_times` 的主流程依赖，改传 `speaker_evidence_points`。
* `backend/app/pipelines/async_dual_pipeline.py`: 链路收口。
  - 删除 L5.5->L6 的“硬边界点下传”主路径。
  - 改为“词级身份 + 证据点”下传，L6 统一做软切判定。
  - 保留统计但更名为软切统计，避免历史指标误导。
* `backend/app/services/segmentation/boundary_processor.py`: 第一步清理重点。
  - 下线或禁用以下硬切补丁分支：`shift_fallback`、`force_on_turn_overlap_without_pause`、`force_on_turn_change_same_speaker`、`forced_boundary_times` 直推。
  - 仅保留“词级身份修正 + 证据生成”职责，不再直接改写句段边界。
* `backend/app/services/segmentation/segmentation_processor.py`: 软切核心落点。
  - 新增“待切窗口”机制：speaker-change 只触发窗口，不直接切。
  - 在窗口内执行锚点选择：优先停顿词边界，其次句末标点，再次语义边界同分裁决。
  - 强约束：禁止句中切；若仅能切出 1 词短句，默认并入下一句并打标 `singleton_merged`。
  - 对所有未命中窗口输出 `deferred_cut`，用于漏切追踪，不再静默失败。
* `backend/app/services/text_pipeline_config.py` 与 `backend/app/services/model_runtime_config_service.py`: 参数减法。
  - 清理 L5.5/L6 历史补丁参数，减少互相打架的阈值。
  - 新增少量核心参数：`speaker_change_min_score`、`soft_cut_window_sec`、`anchor_pause_min_sec`、`singleton_merge_max_duration_sec`。
* `backend/app/pipelines/mode_policy.py`: Phase 5 收尾实现点。
  - `dual_full`: 完整软切。
  - `smart_patch`: 仅高置信说话人变化参与软切，其余退化。
  - `fast_only`: 无 GPU 强制路径，禁用 diarization/segmentation，保持可用性优先。
* `backend/tests/test_l55_boundary_processor.py`、`backend/tests/test_layer_processors_l5_l6_l7.py`、`backend/tests/test_async_dual_pipeline_timeline_priority.py`: 测试重构。
  - 删除针对硬边界补丁细节的脆弱断言，改为“行为验收”断言。
  - 新增文本域评估回归：漏切、错切、单词句、句中误切四类指标。
* `backend/scripts/eval_l55_forced_points_text_domain.py`（建议重命名）: 评估脚本升级为软切通用评估脚本。
  - 指标统一为文本域（适配 GT 无精确时间戳的现状）。
  - 建议输出：`gt_cut_hit_rate`、`extra_cut_count_in_gt_scope`、`singleton_sentence_rate`、`mid_sentence_wrong_split_rate`。
* `docs/v3.2.3/pyannote-软切重构与Phase5收尾方案.md`（本文）: 执行路线与通过标准。
  - Step 1 清理通过标准：
    - 主链路不再使用 `forced_boundary_times` 触发切分；
    - L6 仅接收证据点与常规停顿/文本信息；
    - 相关单测通过，且历史任务回放不崩溃。
  - Step 2 设计通过标准：
    - 证据点契约稳定；
    - 软切窗口与锚点选择流程跑通；
    - 无“句中硬切”回归。
  - Step 3 实施通过标准（文本域）：
    - `gt_cut_hit_rate` 相比当前基线提升至少 `+15%`；
    - `extra_cut_count_in_gt_scope` 不高于基线 `+10%`；
    - `singleton_sentence_rate` 降至 `< 2%`；
    - `mid_sentence_wrong_split_rate` 连续 2 个任务集下降。
  - Phase 5 收尾通过标准：
    - `dual_full/smart_patch/fast_only` 三模式均可独立跑通；
    - 无 GPU 时自动进入 `fast_only`，且流程可完成；
    - 暂停/恢复在 superblock 边界无状态损坏。
* `外部参考（用于约束方案，不直接复制实现）`:
  - Pyannote 官方 `diarization + STT merge`：先重叠归因，无法重叠时最近邻兜底。
  - Pyannote 官方 `speaker configuration`：未知人数场景优先自动估计或 min/max，避免错误固定人数。
  - WhisperX `assign_word_speakers`：以 overlap 面积主导词级 speaker 绑定，`fill_nearest` 仅兜底，不直接驱动切句。

## 4. 与母文档逐章处置清单（沿用/修改/删除）

### 4.1 可直接沿用（保持原设计，不改语义）

以下章节在软切方案下仍成立，可直接沿用：

* `1.2 设计原则（强约束）`：简洁优先、边界清晰、分阶段可回退继续有效。
* `2.2 总体流程`：`Preprocess -> Fast -> S-Epoch -> Bridge -> W-Epoch -> Finalize` 主链路不变。
* `2.3 GPU 调度策略`：`S-Epoch/W-Epoch` 串行占用 GPU 的约束不变。
* `3.1 ~ 3.7 领域边界`：各域职责划分保持不变。
* `4.1 AudioChunk 去 speaker 字段 + chunk_id`：契约方向正确，继续沿用。
* `4.2 SpeakerTimeline/SpeakerProfile/SpeakerTurn`：核心时间线契约继续沿用。
* `4.3 TurnGroup 契约`：作为慢流最小处理单元继续沿用。
* `4.4/4.5/4.6 speaker_store + API + sqlite 基础模块`：持久化与前端可编辑链路继续沿用。
* `5.1/5.2/5.3/5.4 ModePolicy 三模式与无 GPU 回退`：模式框架继续沿用。
* `6 S-Epoch/W-Epoch 协同时序`：启动与增量定稿时序继续沿用。
* `8 聚类准确性改造`：candidate/confirmed/merge 三态方向继续沿用。
* `10.1 新增文件（除软切新增模型外）`：已落地的契约/基础设施文件继续沿用。
* `10.4 断点与缓存统一重构蓝图`：双层断点方向继续沿用。
* `11.1 ~ 11.4`：Phase 0~3 的工程结果继续沿用。
* `15 执行清单`：按 phase 验收、打 tag、文档同步流程继续沿用。

### 4.2 需要修改（保留章节，但替换内容）

以下章节保留编号与目标，但需要按软切方案重写：

* `1.1 总目标`：
  - 新增“硬边界误用导致句中错切/首词错切”作为核心问题；
  - 目标从“加强硬切”改为“speaker 证据驱动软切”。
* `2.1 核心思想`：
  - 从“SpeakerTimeline 单一真相源”升级为“SpeakerTimeline + SpeakerEvidenceTrack 双真相源”；
  - Timeline 负责身份事实，Evidence 负责切分候选证据。
* `5.5 Pyannote segmentation 纳入统一模型管理`：
  - 增补 `num_speakers/min_speakers/max_speakers` 使用策略；
  - 明确“未知人数不固定 num_speakers”。
* `7 抗幻觉组批与 flush 机制`：
  - 保留“单 speaker 组批”约束；
  - 把 `speaker_change` flush 触发源从“原始 turn-change”改为“稳定 speaker 变化信号”；
  - 明确“同 speaker turn-change 不触发 flush”。
* `9.1 L5 输入升级`：
  - 从“speaker/turn 注入”升级为“speaker/turn + evidence 注入”；
  - 新增 `SpeakerEvidencePoint` 契约。
* `9.2 L6 切分升级`：
  - 由“speaker+turn 双硬边界”改为“待切窗口 + 锚点选择”的软切；
  - 强约束为“禁止句中硬切，单词句默认回并”。
* `9.3 L7 输出字段扩展`：
  - 保留 `speaker_id/turn_id/speaker_color_key`；
  - 新增 `split_reason/split_risk`（如 `deferred_cut/singleton_merged`）用于可观测。
* `9.4 L7 与声纹库闭环`：
  - 继续写 `subtitle_speaker_links`；
  - 增加软切证据追溯字段（便于回放“为何切/为何未切”）。
* `10.2 重写文件（职责描述）`：
  - `segmentation_processor.py` 的目标描述改为“软切评分与锚点决策”，删除“双硬边界”字样；
  - `bridge_controller.py` 的目标描述改为“消费稳定 speaker 变化信号”。
* `11.5 Phase 4`：
  - 目标改为“L2-L7 soft-cut 收口”；
  - 重点从 hard-boundary 改为 evidence-driven soft-cut。
* `11.6 Phase 5`：
  - 继续保留模式与无 GPU 回退；
  - 增加三模式下软切策略差异验收。
* `12 测试与验收标准`：
  - 从“是否跨 turn”改为文本域指标：
    - `gt_cut_hit_rate`
    - `extra_cut_count_in_gt_scope`
    - `singleton_sentence_rate`
    - `mid_sentence_wrong_split_rate`
* `13 配置项规划`：
  - 删除硬边界补丁参数族；
  - 收敛为少量软切核心参数：`speaker_change_min_score`、`soft_cut_window_sec`、`anchor_pause_min_sec`、`singleton_merge_max_duration_sec`。
* `14 Pyannote 模型占位`：
  - 保留占位思路；
  - 增补 diarization 社区版与 exclusive 轨道使用约束。
* `16 结论`：
  - 结论表述改为“以软切替代硬切补丁堆叠，降低漏切与错切耦合”。

### 4.3 直接删除（从母文档中移除，不再保留）

以下条目与软切目标冲突，需直接删除：

* `9.2 L6 切分升级` 下的硬切表述：
  - “仍按 speaker run 强切分”
  - “增加按 turn_id 的硬边界”
* `10.2 重写文件` 中 `segmentation_processor.py` 的“speaker+turn 双硬边界”目标描述。
* `12.1 功能验收` 中“L6 切分不跨 turn”这条刚性约束。
  - 原因：软切允许同 speaker 相邻 turn 在无可靠锚点时合并，避免句中误断。
* 所有“以 pyannote 边界点直接触发断句”的实现要求（若在实施细节中出现，统一删除）。
* 所有“依赖单次边界吸附失败后强制 shift 落刀”的设计要求（转为证据降权，不再强切）。

### 4.4 章节落地规则（执行时必须遵守）

* 修改母文档时采用“整段替换”，禁止在旧硬切段落后追加“补丁说明”。
* `11.5` 与 `11.6` 必须在同一轮文档更新中同步，避免 Phase 断层。
* 文档状态必须反映当前代码真相：已删除的硬切逻辑不得继续出现在“目标态”描述里。
