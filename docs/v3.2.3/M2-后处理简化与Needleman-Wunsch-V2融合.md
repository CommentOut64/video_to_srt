# M2 后处理简化与 Needleman-Wunsch V2 融合

> Type: Architecture | Status: Active
> Version: V3.2.0+dev.20260215.28
> Scope: 后处理主链简化、NW V2 改造、时间轴映射契约补齐

## 1. Summary

* **Goal**: 把当前复杂后处理链路收敛为四模块主链，并在同一阶段完成 NW V2 升级，统一“切分决策”和“对齐质量”基础能力。
* **关键结果**:
  - 七层行为收口到四模块契约。
  - 双 NW 实现收敛为单内核，支持置信度与先验加权。
  - 明确 pyannote 帧时间到快流词级时间轴的映射契约。

## 2. Diagram

* `FactBuilder` -> `EvidenceFusion` -> `SoftCutDecisionEngine` -> `OutputTrace`
* `fast words + slow words + speaker facts` -> `NW V2 alignment` -> `词级统一时基`
* `pyannote frame time` -> `候选边界` -> `锚点吸附` -> `最终切分时间`

## 3. Key Components

* `backend/app/services/alignment/alignment_service.py`: NW 主实现（已接入 V2 内核开关）。
* `backend/app/services/whisper_buffer_pool.py`: 第二入口 NW 调用方（已收敛到同一内核）。
* `backend/app/services/textflow/decision_layer.py`: 裁决层执行器（四层统一入口）。
* `backend/app/services/timeline/segmentation_service.py`: pyannote 帧时间源。
* `backend/app/services/alignment/types.py`: 四模块契约对象定义。

## 4. 后处理简化方案（7 -> 4）

### 4.1 四模块职责

1. `FactBuilder`：只生产事实，不做判断。  
输入：快慢流词、turn、基础时间戳。  
输出：`AlignedFacts`。

2. `EvidenceFusion`：把多源证据统一成可评分结构。  
输入：`AlignedFacts` + pyannote/speaker/pause/punctuation/semantic 信号。  
输出：`FusedEvidence`。

3. `SoftCutDecisionEngine`：唯一裁决层。  
输入：`AlignedFacts` + `FusedEvidence`。  
输出：`CutPlan`（含 deferred 队列）。

4. `OutputTrace`：输出句段与追踪信息。  
输入：`CutPlan`。  
输出：`FinalSegments` + `trace`。

### 4.2 与旧七层的映射原则

1. 旧层内部算法可暂时保留，但只能通过四模块契约连接。
2. 删除硬边界补丁直推链路，不允许旁路落刀。
3. 输出必须包含 `split_reason` 与 `split_risk`。

## 5. 快慢流“受影响窗口重算”规则

### 5.1 触发条件

1. 决策依赖的快流草稿被慢流覆盖。
2. 决策带 `deferred` 标签，且新锚点进入窗口。
3. speaker_change 分级变化（例如 mid 升级 high）。

### 5.2 参考伪代码

```python
def find_affected_windows(existing_decisions, slow_result):
    affected = []
    for decision in existing_decisions:
        if decision.depends_on_fast_draft and slow_result.covers(decision.time_range):
            affected.append(decision.window_id)
        if decision.risk == "deferred":
            new_anchors = slow_result.get_anchors_in_range(decision.time_range)
            if new_anchors:
                affected.append(decision.window_id)
    return sorted(set(affected))
```

## 6. Needleman-Wunsch V2（与 M2 同步交付）

### 6.1 现状问题

1. 当前 NW 主要使用固定分值（`match/mismatch/gap`），对词置信度不敏感。
2. 置信度多数在后验评分中使用，未进入 DP 路径决策。
3. 存在两套 NW 实现，维护与行为一致性成本高。

### 6.2 V2 改造目标

1. 单内核：`alignment_service` 与 `whisper_buffer_pool` 共用同一 NW V2。
2. 加权打分：将词级置信度纳入 `match/mismatch/gap`。
3. 先验接口：预留 LLM 语义先验矩阵输入。

### 6.3 建议打分公式

```python
match_score = base_match + alpha * conf_pair + beta * prior_pair
mismatch_penalty = base_mismatch * (0.5 + conf_pair)
delete_penalty = base_gap * (0.5 + conf_whisper_i)
insert_penalty = base_gap * (0.5 + conf_sv_j)
```

其中：

1. `conf_pair`：候选匹配对的综合置信度。
2. `prior_pair`：先验矩阵给出的语义配对概率。
3. `base_gap` 为负值，置信度越高，跳过代价越大。

### 6.4 LLM 预留接口

```python
class AlignmentPriorProvider(Protocol):
    def prior_matrix(
        self,
        whisper_tokens: list[str],
        sv_tokens: list[str],
    ) -> list[list[float]]:
        ...
```

默认实现返回零矩阵；M3 再引入语义先验实现。

## 7. Pyannote 帧时间到快流时间轴映射契约（补齐缺口）

### 7.1 现状事实

`segmentation_service` 当前使用：

```python
time_sec = frame_start + idx * frame_step
```

这是 pyannote 自身帧时间，不等于“快流可落刀时间”。

### 7.2 目标契约

1. 输入：`pyannote_frame_time`、`fast_words`、`pause_anchors`、`punctuation_anchors`。
2. 输出：`mapped_cut_time`、`mapping_quality`、`mapping_reason`。

### 7.3 映射流程

1. 生成候选边界：`t_raw = frame_start + idx * frame_step`。
2. 时间基校正：若属于块内时间，先加 `block_start_offset` 转为全局时间。
3. 锚点吸附：
  - 一级：窗口内最近停顿词边界（优先）。
  - 二级：语义锚点（M3 接入后启用）。
  - 三级：句末标点锚点。
4. 超窗处理：若无可接受锚点，创建 deferred，不直接硬切。
5. 追踪落盘：记录 `raw_time`、`snapped_time`、`delta_ms`、`anchor_type`、`risk`。

### 7.4 参数建议

1. `anchor_snap_tolerance_sec=0.22`
2. `timebase_offset_strategy=block_global`
3. `unresolved_frame_policy=defer_then_force`

## 8. 兼容与灰度策略

1. 新四模块与旧逻辑并行跑一段时间，进行结果比对。
2. NW V2 可配置开关，先对实验任务集启用。
3. 若指标不达标，可回退：
  - 后处理回退到 M1 兼容链路。
  - NW 回退到固定分值版本。

## 9. 验收标准

1. 四模块主链可独立跑通。
2. NW V2 单元测试覆盖匹配、错配、gap、先验注入场景。
3. pyannote 映射契约可追溯，无无因切分。
4. 指标目标：
  - `gt_cut_hit_rate` 提升。
  - `extra_cut_count_in_gt_scope` 不显著上升。
  - 对齐异常 gap 率下降。

## 10. 分阶段实现计划（完整）

### 10.1 目标口径与命名

1. 本计划采用四层中文命名：`集合层` -> `评分层` -> `裁决层` -> `输出层`。
2. 对外以四层命名对齐认知；对内允许复用现有 `L1-L7` Processor，只做调用关系收口。
3. 所有阶段必须满足三条硬约束：
  - 不阻塞首条字幕可见性。
  - 保持可灰度、可回退。
  - 结果可追溯（必须有 `split_reason/split_risk`）。

### 10.2 阶段 0：基线冻结与开关准备（0.5 周）

1. 目标：冻结评测口径和回退路径，避免后续“无基线优化”。
2. 主要改动：
  - 固化评测集与指标脚本（切分、对齐、时延）。
  - 增加 M2 专用运行开关：`m2.enable`、`m2.nw_v2.enable`、`m2.time_mapping.enable`。
  - 建立旧路径与新路径双跑采样记录。
3. 代码落点：
  - `backend/app/services/text_pipeline_config.py`
  - `backend/app/pipelines/async_dual_pipeline.py`
4. 退出门槛：
  - 基线报告可复现（同输入同指标结果稳定）。
  - 开关可单独控制四模块与 NW V2。
5. 回滚策略：关闭 `m2.enable`，全量回退现网路径。

#### 10.2.1 阶段 0 当前实现状态（代码已落地）

1. 已新增运行参数组 `runtime.m2`：
  - `enable`（默认 `false`）
  - `nw_v2.enable`（默认 `false`）
  - `time_mapping.enable`（默认 `false`）
  - `shadow.sample_rate`（默认 `0.1`）
  - `shadow.provider_class`（默认空字符串）
2. 已在 `TextPipelineConfig` 接入 `M2StageConfig`，统一从运行参数读取 `m2` 开关。
3. 已在 `AsyncDualPipeline` 接入阶段0观测：
  - 读取并缓存 `m2` 开关；
  - 按 `shadow.sample_rate` 做稳定哈希采样；
  - 输出 `debug/m2_stage0_shadow_samples.jsonl`；
  - 记录 `active/legacy/shadow`（若存在）对齐指标与 `selected_variant/reason`。
4. 当前行为保证：仅写观测，不改变 L4-L6 结果裁决路径。

### 10.3 阶段 1：契约收口（1 周）

1. 目标：先统一数据契约，再迁移逻辑，避免跨阶段字段漂移。
2. 主要改动：
  - 在类型层补齐四层契约对象：`AlignedFacts`、`FusedEvidence`、`CutPlan`、`OutputTrace`。
  - 扩展 `L6/L7` 输入输出，强制透传 `split_reason/split_risk/window_id`。
  - 明确 `pyannote_frame_time -> mapped_cut_time` 的入参与出参字段。
3. 代码落点：
  - `backend/app/services/alignment/types.py`
  - `backend/app/services/segmentation/soft_cut/types.py`
  - `backend/app/services/streaming/output_processor.py`
4. 退出门槛：
  - 现有流程零行为变化（仅结构变更）。
  - 新字段在调试文件可见且不为空。
5. 回滚策略：契约字段保留为可选，旧字段优先读取。

#### 10.3.1 阶段 1 当前实现状态（代码已落地）

1. 已在 `alignment/types.py` 新增契约对象：
  - `AlignedFacts`
  - `FusedEvidence`
  - `OutputTrace`
2. 已扩展 `L6/L7` 输入输出契约：
  - `L6Input` 新增 `aligned_facts/fused_evidence`（可选，兼容模式）
  - `L6Output` 新增 `output_traces`
  - `L7Input/L7Output` 新增 `output_traces` 透传
3. 已扩展 `SentenceSegment` 字段并对外序列化：
  - `split_reason/split_risk/window_id`
  - `pyannote_frame_time/mapped_cut_time/mapping_quality/mapping_reason`
4. 已在 `SegmentationProcessor` 完成句级 trace 构建与透传：
  - CutPlan 路径：按 `decision(window_id/reason/risk)` + 词边界映射生成 `OutputTrace`
  - 默认路径：生成 `default_splitter` trace，保证字段非空
5. 已在 `OutputProcessor` 与 `AsyncDualPipeline` 完成 L7 透传与调试输出接入：
  - `output_payload.output_trace` 可直接观测
  - `layer_trace_full` 句级输出新增 split/mapping 字段
6. 当前行为保证：不改变切分算法与选路，仅新增契约和可观测字段。

### 10.4 阶段 2：NW V2 单内核改造（1 周）

1. 目标：合并双 NW 实现，完成可加权 DP 核心。
2. 主要改动：
  - 抽取统一内核（建议 `alignment/nw_v2_core.py`），支持 `match/mismatch/gap` 动态加权。
  - `alignment_service` 与 `whisper_buffer_pool` 改为同内核调用。
  - 预留 `AlignmentPriorProvider` 注入点，默认零矩阵。
3. 代码落点：
  - `backend/app/services/alignment/alignment_service.py`
  - `backend/app/services/whisper_buffer_pool.py`
4. 测试与验收：
  - 单测覆盖：匹配、错配、插删、先验注入、置信度极值。
  - 一致性验证：双入口同样本路径一致率达到门槛。
5. 回滚策略：保留 `nw_v1` 分支，`m2.nw_v2.enable=false` 即回退。

#### 10.4.1 阶段 2 当前实现状态（代码已落地）

1. 已新增统一内核 `backend/app/services/alignment/nw_v2_core.py`：
  - `NeedlemanWunschV2Core`
  - `NeedlemanWunschScoreConfig`
  - `AlignmentPriorProvider`（含 `ZeroPriorProvider` 默认实现）
2. 已将 `alignment_service` 的 NW 逻辑改为调用统一内核：
  - 保留 `_needleman_wunsch` 外部签名（兼容）
  - 支持 `seq1/seq2` 置信度输入
  - 支持先验矩阵接口（默认零矩阵）
3. 已将 `whisper_buffer_pool` 的 NW 逻辑改为调用同一内核：
  - 删除独立 DP 评分实现
  - 使用 Whisper 词概率与 SenseVoice 词置信度参与打分（在启用 V2 时生效）
4. 开关与回退：
  - `AlignmentProcessor` 按 `m2.nw_v2.enable` 注入 `AlignmentConfig.is_enable_nw_v2`
  - `WhisperBufferAligner` 默认读取运行参数 `m2.nw_v2.enable`（也可通过 `WhisperBufferConfig.nw_v2_enable` 显式覆盖）
  - 关闭 `m2.nw_v2.enable` 时回退固定分值口径（V1 行为）
5. 当前行为保证：
  - 入口调用链不变；
  - 仅收敛内核与评分策略，不改 L4-L6 编排流程。

#### 10.4.2 阶段 2 验收补充（测试已补齐）

1. 已在 `backend/tests/unit/services/alignment/test_alignment_service_tokenize.py` 补充阶段2验收测试，覆盖：
  - 匹配与错配路径（`match/mismatch`）。
  - 插入与删除路径（`gap insert/delete`）。
  - 先验注入改变歧义路径（`prior` 生效）。
  - 置信度极值归一化与 gap 惩罚边界（`None/-1/2` 归一化到 `0~1`）。
  - 双入口一致性（`AlignmentService` 与 `WhisperBufferAligner` 同样本路径一致）。
2. 本次阶段2回归结果：
  - `backend/tests/unit/services/alignment/test_alignment_service_tokenize.py` + `test_alignment_processor.py`：`12 passed`。
  - `backend/tests/test_async_dual_pipeline_alignment_stage.py` + `test_layer_processors_l5_l6_l7.py` + `test_soft_cut_phase_d_integration.py` + `test_soft_cut_phase_d_pipeline_integration.py`：`40 passed`。
3. 当前结论：阶段2“单内核 + 回退开关 + 验收测试”已闭环，可进入阶段3（集合层）实施。

### 10.5 阶段 3：集合层落地（1 周）

1. 目标：把“事实收集”从编排层抽离，形成单一事实入口。
2. 主要改动：
  - 新增 `FactBuilder`（集合层），统一聚合快慢流词、turn、时间基与质量字段。
  - 接入 pyannote 时间映射契约：`raw_time/snapped_time/delta_ms/mapping_reason`。
  - 仅生产事实，不做切分裁决。
3. 代码落点：
  - `backend/app/pipelines/async_dual_pipeline.py`
  - `backend/app/services/timeline/segmentation_service.py`
  - `backend/app/services/alignment/types.py`
4. 退出门槛：
  - 每个 chunk 都能产出可用 `AlignedFacts`。
  - `mapping_quality` 分布可观测，无大量 `unknown`。
5. 回滚策略：保持旧词流拼接逻辑并行，切换到旧事实源。

#### 10.5.1 阶段 3 当前实现状态（代码已落地）

1. 已新增集合层构建器 `backend/app/services/alignment/fact_builder.py`：
  - `FactBuilder` / `FactBuilderConfig`
  - 统一构建 `AlignedFacts`（词、turn、快流边界、时间映射）
2. 已扩展 `AlignedFacts` 契约（`backend/app/services/alignment/types.py`）：
  - 新增 `time_axis_version`
  - 新增 `time_mappings`（`raw_time/snapped_time/delta_ms/mapping_reason/mapping_quality`）
3. 已在 `AsyncDualPipeline` 接入阶段3主链：
  - `legacy/experiment` 双路径在进入 L6 前统一构建 `aligned_facts`
  - `sensevoice_only` 快路径同样构建并下传 `aligned_facts`
  - `L6Input.aligned_facts` 已接入，不改变现有切分裁决逻辑
4. 已在 `timeline/segmentation_service.py` 增加映射函数：
  - `map_frame_times_to_word_boundaries(...)`
  - 支持 `m2.time_mapping.enable` 开关（关闭时输出 `disabled` 映射）
5. 可观测性补齐：
  - `SegmentationProcessor.segmentation_report.aligned_facts_stats` 输出事实统计
  - `layer_trace_full.l5.aligned_facts` 可直接观测映射质量分布
6. 回归结果：
  - `backend/tests/test_layer_processors_l5_l6_l7.py` + `test_soft_cut_phase_d_integration.py` + `test_soft_cut_phase_d_pipeline_integration.py` + `test_async_dual_pipeline_alignment_stage.py`：`40 passed`
  - `backend/tests/unit/services/alignment/test_alignment_service_tokenize.py` + `test_alignment_processor.py`：`12 passed`
7. 当前结论：阶段3“集合层单入口 + 时间映射契约 + 主链透传”已完成，可进入阶段4（评分层）实施。

### 10.6 阶段 4：评分层落地（1 周）

1. 目标：统一证据融合口径，替换 pipeline 中散落规则。
2. 主要改动：
  - `EvidenceBuilder` 只负责事实转证据与窗口构建。
  - `EvidenceFusion` 统一锚点评分和冲突合并，输出 `FusedEvidence`。
  - 把当前 pipeline 内 speaker-change 与锚点拼装逻辑迁移到服务层。
3. 代码落点：
  - `backend/app/services/segmentation/soft_cut/evidence_builder.py`
  - `backend/app/services/segmentation/soft_cut/evidence_fusion.py`
  - `backend/app/pipelines/async_dual_pipeline.py`
4. 退出门槛：
  - 合证输出字段完整，窗口数与锚点数稳定可解释。
  - 与旧逻辑双跑时，异常切分不高于基线。
5. 回滚策略：`soft_cut.plan_provider=m1_internal` 回退到 M1 生产器。

#### 10.6.1 阶段 4 当前实现状态（代码已落地）

1. 已在 `AsyncDualPipeline` 新增评分层统一入口：
  - `backend/app/pipelines/async_dual_pipeline.py` 新增 `_build_fused_evidence_for_l6(...)`
  - 统一输出 `FusedEvidence`，收口 speaker-change 与各类锚点拼装
2. 已将 `EvidenceFusion` 扩展为契约输出端：
  - `backend/app/services/segmentation/soft_cut/evidence_fusion.py` 新增 `to_fused_evidence(...)`
  - 由合证服务统一封装 `FusedEvidence`，避免 pipeline 直接拼契约对象
3. 主链透传已接入：
  - `L6Input.fused_evidence` 在 `legacy/experiment/sensevoice_only` 三路径均已注入
  - 不改变现有 `CutPlan` 决策链，仅新增证据契约与观测字段
4. 可观测性补齐：
  - `SegmentationProcessor.segmentation_report.fused_evidence_stats` 输出合证统计
  - `layer_trace_full.l5.fused_evidence` 可直接观测证据与报告
5. 回归结果：
  - `backend/tests/test_soft_cut_plan_provider.py` + `test_soft_cut_phase_d_integration.py` + `test_soft_cut_phase_d_pipeline_integration.py` + `test_layer_processors_l5_l6_l7.py` + `test_async_dual_pipeline_alignment_stage.py`：`43 passed`
  - `backend/tests/unit/services/alignment/test_alignment_service_tokenize.py` + `test_alignment_processor.py`：`12 passed`
6. 当前结论：阶段4“评分层单入口 + FusedEvidence 统一透传 + 报告可观测”已完成，可进入阶段5（裁决层）实施。

### 10.7 阶段 5：裁决层落地（1 周）

1. 目标：实现唯一裁决层，统一 deferred 生命周期。
2. 主要改动：
  - `SoftCutDecisionEngine` 作为唯一落刀决策器输出 `CutPlan`。
  - 实现“受影响窗口重算”与跨 chunk deferred 状态维护。
  - 禁止旁路落刀，`SegmentationProcessor` 仅消费 `CutPlan`。
3. 代码落点：
  - `backend/app/services/segmentation/soft_cut/decision_engine.py`
  - `backend/app/services/segmentation/segmentation_processor.py`
  - `backend/app/pipelines/async_dual_pipeline.py`
4. 退出门槛：
  - `deferred` 终态闭环（pending/resolved/forced/expired）完整。
  - 句中误切率不升高，漏切率下降。
5. 回滚策略：`cut_plan=None` 退回 `FinalSplitter` 默认路径。

#### 10.7.1 阶段 5 当前实现状态（代码已落地）

1. 已在 `AsyncDualPipeline._build_soft_cut_plan_for_l6_m1(...)` 接入“受影响窗口重算”：
  - 基于 `aligned_facts/fused_evidence` 与 pending deferred，识别受影响窗口（`new_anchor/slow_covered/level_upgraded`）。
  - 将受影响 deferred 转换为重算窗口并并入当前轮 `SoftCutDecisionEngine` 统一裁决。
2. 已完成 deferred 生命周期闭环：
  - 新增 deferred 元数据（`window_start/window_end/trigger_level/depends_on_fast_draft`）。
  - 当前链路可形成完整终态：`pending/resolved/forced/expired`。
  - `is_last_chunk=true` 时，剩余 `pending` 自动转 `expired`，避免悬挂状态。
3. 已补齐观测字段与统计：
  - `CutPlan.generation_report` 新增 `affected_window_count/affected_window_ids/affected_reason_by_window`。
  - 新增 `deferred_state_stats` 与阶段5统计（重算窗口数、受影响 resolved/expired 数）。
  - `SegmentationProcessor.soft_cut_stats` 增加 `deferred_state_stats` 透传。
4. 行为边界保持不变：
  - `SegmentationProcessor` 仍仅消费 `CutPlan`，无旁路落刀。
  - 保留旧回退路径（`cut_plan=None` -> `FinalSplitter` 默认切分）。
5. 阶段5本地回归结果：
  - `test_soft_cut_decision_engine.py`
  - `test_soft_cut_plan_provider.py`
  - `test_soft_cut_phase_d_pipeline_integration.py`
  - `test_soft_cut_phase_d_integration.py`
  - `test_layer_processors_l5_l6_l7.py`
  - `test_async_dual_pipeline_alignment_stage.py`
  - `unit/services/alignment/test_alignment_service_tokenize.py`
  - `unit/services/alignment/test_alignment_processor.py`
  - 以上共 `62 passed`。

### 10.8 阶段 6：输出层收口（0.5 周）

1. 目标：输出层仅负责分发与追踪，不再混入文本逻辑。
2. 主要改动：
  - 统一 `OutputTrace` 输出结构（句段 + trace + transport_meta）。
  - `L7` 落盘与 SSE 同步透传 `split_reason/split_risk/window_id`。
  - 清理历史重复输出分支，保留单一出稿入口。
3. 代码落点：
  - `backend/app/services/streaming/output_processor.py`
  - `backend/app/pipelines/async_dual_pipeline.py`
4. 退出门槛：
  - 输出链无文本改写副作用。
  - 任一句段可反查证据与落刀原因。
5. 回滚策略：恢复旧输出 payload 结构映射，兼容读取新字段。

#### 10.8.1 阶段 6 当前实现状态（代码已落地）

1. 已完成 L7 单入口收口：
  - `AsyncDualPipeline` 新增统一分发函数 `_emit_l7_output(...)`。
  - 快路（`whisper_skipped`）与主路（L4-L6 常规路径）统一通过该入口构造 `L7Input` 并调用 `OutputProcessor`。
2. 已完成统一输出契约（句段 + trace + transport_meta）：
  - `OutputProcessor.output_payload` 新增 `sentence_segments` 序列化输出。
  - `transport_meta` 新增通道状态：
    - `subtitle_channel.status/replaced_sentence_count`
    - `speaker_store_channel.status/upsert_count`
  - 保留既有 `output_trace`，与句段一一对应可追溯。
3. 已明确“输出层不改文本逻辑”边界：
  - 输出层仅做分发、trace 贴附与序列化，不引入额外文本切分/改写策略。
  - 文本切分仍由 L6 `CutPlan` / `FinalSplitter` 决定，L7 仅透传。
4. 已补齐阶段6测试覆盖：
  - `test_l7_processor_outputs_sentence_segments_and_transport_meta`
  - `test_l7_processor_marks_speaker_store_channel_failed`
5. 阶段6回归结果：
  - 相关测试与阶段5回归集均通过（本次回归总计 `64 passed`）。

### 10.9 阶段 7：灰度放量与发布闸门（0.5-1 周）

1. 目标：在可控风险下完成从实验到默认启用。
2. 灰度步骤：
  - Shadow：新旧双跑，仅记日志不改线上结果。
  - Canary：小流量启用新结果，持续对比关键指标。
  - Ramp：分批扩大到全量，保留一键回退。
3. 发布闸门：
  - `gt_cut_hit_rate` 提升且 `extra_cut_count_in_gt_scope` 不显著上升。
  - 对齐异常 `gap_ratio` 下降。
  - TTFS、吞吐、GPU 峰值不劣于基线门槛。
4. 回滚策略：
  - 关闭 `m2.enable`：回退 M1 兼容后处理链路。
  - 关闭 `m2.nw_v2.enable`：回退固定分值 NW。

### 10.10 里程碑交付清单（DoD）

1. 代码：四层主链可单独开关，双 NW 已收敛为单内核。
2. 测试：单测、连接测试、回放测试全部通过。
3. 文档：`docs` 与 `llmdoc` 的架构口径一致，移除过时描述。
4. 运维：灰度看板、报警阈值、回滚手册齐备。

### 10.11 阶段 8：venv 真实 GPU 集成测试收口（已完成）

1. 目标：在项目 `.venv` 环境下，验证“新四层链路已生效 + 真实流水线可运行”。
2. 主要改动：
  - `ci_tests/integration/harness/test_runner.py`
    - `use_real_engines=true` 时接入 `ASREngineFactory` 创建真实引擎：
      - 草稿引擎：`sensevoice`
      - 补刀引擎：`whisper(device=\"cuda\")`（非 `sensevoice_only`）
    - 真实引擎模式下关闭 `SemanticBuffer`，强制进入对齐后处理链路，确保覆盖集合/评分/裁决/输出层。
  - `ci_tests/integration/test_local_pipeline.py`
    - `test_real_full_pipeline` 取消强制 `skip`，改为可执行真实用例。
    - 场景切换为 `sv_whisper_dual`，并新增层级执行证据断言：
      - `fact_word_count`（集合层）
      - `evidence_pause_anchor_count`（评分层）
      - `split_decision_count`（裁决层）
      - `l7_error_count`（输出层）
3. venv 执行命令：
  - `$env:PYTHONPATH='.'; .\\.venv\\Scripts\\python.exe -m pytest -o addopts= ci_tests/integration/test_local_pipeline.py::TestRealVideoIntegration::test_real_full_pipeline --test-video \"F:/video_to_srt_gpu/input/test_video_H265.mp4\" --real-engines -q -rs`
4. 验证结果：
  - `1 passed`（真实 GPU 用例已执行，不再 skip）。
  - 日志可见 `集合层对齐完成`、`评分层注入完成`、`裁决层切分完成`、`输出层完成`，与新四层口径一致。

### 10.12 阶段 C：四层目录统一入口（已完成）

1. 目标：在不破坏现有调用链的前提下，把主流水线依赖收口到四层目录，减少“同职跨目录导入”扩散。
2. 主要改动：
  - 新增四层统一入口目录：`backend/app/services/textflow/`
    - `collection_layer.py`
    - `scoring_layer.py`
    - `decision_layer.py`
    - `output_layer.py`
  - `AsyncDualPipeline` 主链导入已切换到四层入口：
    - `CollectionAlignmentProcessor/CollectionFactBuilder`
    - `ScoringSemanticInjectionProcessor/ScoringEvidenceFusion`
    - `DecisionSegmentationProcessor`
    - `OutputLayerProcessor`
  - 旧模块保留为兼容实现，不做破坏性删除。
3. 质量守卫升级：
  - `ci_tests/quality/check_textflow_layer_guards.py` 新增“旧入口导入冻结”规则；
  - 禁止新增以下旧入口直连导入：
    - `alignment.alignment_processor`
    - `alignment.fact_builder`
    - `punctuation.semantic_injection_processor`
    - `segmentation.segmentation_processor`
    - `streaming.output_processor`
4. 当前结论：
  - 四层目录入口已成为主编排路径；
  - 通过“兼容保留 + 新增冻结”实现平滑迁移，可继续推进后续物理合并与旧文件清理。

### 10.13 阶段 D：物理合并第一批（已完成）

1. 目标：将四层中体量可控的实现迁移到 `textflow` 目录，旧路径降级为兼容桥接文件。
2. 主要改动：
  - `backend/app/services/textflow/collection_layer.py`
    - 内置 `CollectionAlignmentProcessor` 与 `CollectionFactBuilder` 真实实现。
  - `backend/app/services/textflow/scoring_layer.py`
    - 内置 `ScoringSemanticInjectionProcessor` 真实实现；
    - 保留 `ScoringEvidenceFusion` 合证入口。
  - `backend/app/services/textflow/output_layer.py`
    - 内置 `OutputLayerProcessor` 真实实现。
  - 旧路径改为兼容桥接：
    - `backend/app/services/alignment/alignment_processor.py`
    - `backend/app/services/alignment/fact_builder.py`
    - `backend/app/services/punctuation/semantic_injection_processor.py`
    - `backend/app/services/streaming/output_processor.py`
3. 冻结守卫与基线：
  - `frozen_legacy_postprocess_imports.txt` 已收敛为 1 条（仅保留 `decision_layer` 对 `SegmentationProcessor` 的过渡导入）。
4. 当前结论：
  - 主编排、真实实现、兼容桥接三者已闭环；
  - 后续仅剩 `DecisionSegmentationProcessor` 的大文件（`segmentation_processor.py`）物理迁移与拆分。

### 10.14 阶段 D：物理合并第二批（已完成）

1. 目标：完成裁决层实现迁移，清零旧后处理入口导入。
2. 主要改动：
  - `backend/app/services/textflow/decision_layer.py`
    - 迁入 `SegmentationProcessor` 真实实现；
    - 新增别名 `DecisionSegmentationProcessor = SegmentationProcessor`。
  - `backend/app/services/segmentation/segmentation_processor.py`
    - 改为兼容桥接，转发到 `textflow.decision_layer`。
3. 冻结守卫补强：
  - `ci_tests/quality/check_textflow_layer_guards.py` 允许“空基线文件”（用于收敛到 0 条违规）。
  - `frozen_legacy_postprocess_imports.txt` 已收敛为 0 条。
4. 当前结论：
  - 四层真实实现已全部迁入 `textflow` 目录；
  - 旧路径仅保留兼容桥接文件，不再有旧入口直连导入。

### 10.15 阶段 E：稳定 API 固化与防回流（已完成）

1. 目标：固定 `textflow` 包级 API，避免调用方直接依赖层文件路径。
2. 主要改动：
  - `backend/app/services/textflow/__init__.py`
    - 新增稳定短名导出：
      - `AlignmentProcessor`
      - `FactBuilder` / `FactBuilderConfig`
      - `SemanticInjectionProcessor`
      - `SegmentationProcessor`
      - `OutputProcessor`
  - 调用方统一改为包级导入：
    - `backend/app/pipelines/async_dual_pipeline.py`
    - 旧桥接文件（alignment/punctuation/segmentation/streaming）全部改为 `from app.services.textflow import ...`
3. 质量守卫升级：
  - `ci_tests/quality/check_textflow_layer_guards.py` 新增规则：
    - 冻结新增 `from app.services.textflow.<layer_module> import ...` 直连导入
    - 统一要求调用方改用 `from app.services.textflow import ...`
4. 风险修复：
  - 修复 `textflow` 与 `alignment.__init__` 的循环依赖；
  - `alignment.__init__.py` 改为对 `AlignmentProcessor/FactBuilder` 使用延迟导出（`__getattr__`）。
5. 当前结论：
  - 包级 API 已稳定；
  - 旧入口导入为 0，层文件直连导入仅保留 `textflow/__init__.py` 内部聚合行为。

### 10.16 阶段 F：兼容桥接文件删除（已完成）

1. 目标：删除过渡期兼容桥接文件，消除“名义迁移但保留旧壳”的维护成本。
2. 删除文件：
  - `backend/app/services/alignment/alignment_processor.py`
  - `backend/app/services/alignment/fact_builder.py`
  - `backend/app/services/punctuation/semantic_injection_processor.py`
  - `backend/app/services/segmentation/segmentation_processor.py`
  - `backend/app/services/streaming/output_processor.py`
3. 同步改造：
  - 测试导入统一改为 `from app.services.textflow import ...`：
    - `backend/tests/test_l1_l4_flow_edges.py`
    - `backend/tests/test_async_pipeline_sensevoice_normalize.py`
    - `backend/tests/test_layer_processors_l5_l6_l7.py`
    - `backend/tests/test_soft_cut_phase_d_integration.py`
    - `backend/tests/unit/services/alignment/test_alignment_processor.py`
4. 兼容策略：
  - `backend/app/services/alignment/__init__.py` 保留延迟导出（`__getattr__`），继续支持包级兼容访问。
  - 推荐入口仍为 `app.services.textflow` 包级 API。
5. 验证结果：
  - 运行时引用扫描：`NO_RUNTIME_REFERENCES`。
  - 质量守卫：通过。
  - 回归测试：`44 passed`。

### 10.17 阶段 G：历史别名导出清理（已完成）

1. 目标：移除 `alignment` 包级历史别名导出，彻底收敛到 `textflow` 单入口。
2. 主要改动：
  - `backend/app/services/alignment/__init__.py`
    - 移除 `AlignmentProcessor/FactBuilder/FactBuilderConfig` 导出；
    - 删除对应 `__getattr__` 延迟兼容逻辑。
3. 守卫升级：
  - `ci_tests/quality/check_textflow_layer_guards.py` 新增规则：
    - 冻结新增 `from app.services.alignment import AlignmentProcessor/FactBuilder/FactBuilderConfig`。
  - 新增基线：`frozen_alignment_legacy_exports_imports.txt`（当前为 0 条）。
4. 当前结论：
  - 后处理入口已完整收敛到 `app.services.textflow`；
  - `alignment` 包仅保留算法与工具服务导出，不再承载后处理编排入口职责。

### 10.18 阶段A补充：统一切分优先级引擎（新增实施文档）

1. 目标：为 `speaker/pause/punctuation/semantic/llm` 提供统一优先级协议与 profile 化调权能力，支持“提权标点 -> LLM 提权 -> 标点可移除”的平滑迁移。
2. 详细实施方案：`docs/v3.2.3/M2-阶段A-统一切分优先级引擎实施文档.md`。
3. 落地约束：
  - `speaker` 保持软边界（高权重，不做硬切）；
  - 任何来源的优先级调整必须通过配置完成，不允许新增分支硬编码。
