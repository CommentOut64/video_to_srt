# Timeanchored Window Truth / Owner Truth 分离设计

**Status:** Proposed

**Date:** 2026-04-06

**Owner:** wgh / Codex

## 1. 背景

过去几轮修复已经证明：当前英文 timeanchored 问题不是单点阈值、单个 prompt、或单个贪心参数的问题，而是同一份“整窗 slow 文本”在系统里承担了三个互相冲突的职责：

1. 作为窗口级语义参考文本
2. 作为 owner chunk 的 canonical 文本真相
3. 作为 southbound 输出提交的默认真源

这三种职责一旦混在一份 payload 里，多 chunk window 就会出现结构性串扰：

1. 前一个 owner window 先吃进后一个 chunk 的文本
2. 后一个时间段被提前消耗，形成大空洞
3. canonical token 数显著高于 observation slice 数，导致强行压缩
4. 输出层把“整窗 carrier”当成 owner 的 replace truth，进一步扩大错误影响范围

这也是为什么最近的补丁式修复会出现“越修越差”的现象：它们只在局部抑制症状，但没有把“窗口真相”和“owner 可提交真相”拆开。

## 2. 直接证据

当前代码中，问题链条至少有四个正式落点：

### 2.1 Selection 只产出单一 `SelectedTextTruth`

`backend/app/services/timeanchored_alignment/selection/service.py`

当前 `SelectedTextTruth` 只有一份 `text + source_chunk_ids`，而 `source_chunk_ids` 默认取 `ready_window.source_chunk_ids`。这意味着一旦选择层接受的是整窗 slow text，下游很难再区分：

1. 这是一份窗口级辅助文本
2. 还是一份 owner 可提交文本

### 2.2 AlignmentStage 仍会把 preparation 输入提升成整窗文本

`backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`

`_resolve_preparation_text_inputs()` 目前在多 source chunk window 下，会把 `ready_window.source_units` 拼成整窗文本并提升给 preparation。这样即使选择层产物本来较短，进入 canonical 前也会再次扩大成整窗 truth。

### 2.3 Preparation 默认把 `SelectedTextTruth` 当 canonical 真源

`backend/app/services/timeanchored_alignment/preparation/assembler.py`

Preparation 当前没有“window truth / owner truth”区分，因此 canonical sequence、provenance、compat text truth 都会沿着这份单一 truth 继续传播。

### 2.4 Output projection 默认用整窗 `source_chunk_ids` 做 replace scope

`backend/app/services/timeanchored_alignment/output_projection/output_projector.py`

当前 `OutputProjectionInput.source_chunk_ids` 既表达“窗口覆盖范围”，也被当成 southbound `replace_scope_chunk_ids` 的默认值。Streaming subtitle 层收到 `projection_mode=window_group` 时，会直接清理整个 replace scope。这意味着：

1. 只要前一个 owner window 的 canonical 文本吃进了后续内容
2. 输出层就会把整块 scope 一并清掉并改写
3. 错误会从文本层扩散到提交层

## 3. 设计目标

本次重构只解决当前最小闭环中的结构性职责混淆，目标如下：

1. 明确区分 `window_text_truth` 与 `owner_text_truth`
2. 只有 `owner_text_truth` 可以进入 canonical sequence、alignment decoder、decision、final commit
3. `window_text_truth` 只保留给窗口级语义/标点/诊断/后续可扩展辅助能力
4. southbound replace scope 必须以 owner commit scope 为准，不能再默认为整窗 source scope
5. 修改范围只限于 `selection -> preparation -> output projection`，不重写 decoder / decision 主体

## 4. 非目标

本方案不做以下事情：

1. 不重写 Viterbi decoder、lattice scorer、planner 主算法
2. 不修改 Prompt 生成策略
3. 不引入新的整窗 fallback / safe_window_fallback / partial_commit 架构
4. 不在本轮直接承诺端到端所有历史坏例子一次性全清零

本轮的目标是先建立“真相分层”和“提交边界分层”，让后续所有修复有稳定落点。

## 5. 根因归纳

### 5.1 真相层与提交层共用一个 payload

`SelectedTextTruth` 目前既描述“系统选择了哪份文本”，又默认代表“允许提交哪段文本”。这是职责错误。

### 5.2 `source_chunk_ids` 语义被混用

当前系统里同一个 `source_chunk_ids` 在不同层被当成：

1. 窗口覆盖范围
2. canonical 来源范围
3. replace scope

这三者本应是不同概念：

1. 窗口覆盖范围可以是多 chunk
2. owner canonical 可能只对应其中一部分
3. replace scope 必须严格等于本次提交要替换的那部分

### 5.3 preparation promotion 放大了错误

即使选择层本身没完全失败，`_resolve_preparation_text_inputs()` 也会把真相重新放大成整窗文本，导致后续 decoder 和 decision 看到的 canonical text 比 owner 真正该提交的文本更长。

### 5.4 output projection 把“窗口 carrier”当成“owner 提交真源”

当前 projector 没有正式 commit scope，只能退化成“整窗进、整窗清、整窗替换”。这对 window-first 内部链路是方便的，但对 owner-first 提交语义是错误的。

## 6. 目标态概览

目标态链路调整为：

`selection(window truth + owner truth)`

`-> preparation(owner truth 做 canonical, window truth 留作辅助元数据)`

`-> decoder / decision(只消费 owner canonical)`

`-> output projection(窗口元数据 + owner replace scope 分离)`

也就是说：

1. 窗口 still window-first
2. canonical / commit 必须 owner-first

## 7. 新契约设计

### 7.1 `SelectedTextTruth` 改为“双真相”结构

建议将 `SelectedTextTruth` 扩展为以下正式语义：

1. `owner_text`
   - 本次 owner chunk 允许进入 canonical / decision / commit 的文本
2. `owner_source_chunk_ids`
   - 本次 owner truth 对应的正式 commit scope
3. `window_text`
   - 当前窗口被选择后的整窗文本视图
4. `window_source_chunk_ids`
   - 当前窗口覆盖范围
5. `text_source`
   - `fast / slow / mixed`
6. `language_hint`
7. `quality / rejection_reasons / metadata`

为了减少改动面，允许保留兼容属性：

1. `.text` 映射到 `owner_text`
2. `.source_chunk_ids` 映射到 `owner_source_chunk_ids`
3. 新字段通过正式属性暴露，不靠 metadata 偷渡

### 7.2 Selection 层只负责生成双真相，不再直接决定 preparation promotion

Selection 的职责调整为：

1. 生成 `window_text_truth`
   - 直接基于被选中的 `chosen_track`
2. 生成 `owner_text_truth`
   - 多 source chunk window 下，不能再把 `owner_chunk_id` 机械等同于 commit scope
   - 需要按 owner source units 生成 owner commit-eligible 文本
   - 当 owner source units 本身覆盖整个 window（例如单 turn 跨多个 source chunk）时，允许 `owner_text_truth == window_text_truth`

建议 owner truth 的导出规则是确定性的，而不是继续追加启发式补丁：

1. 先收集 `ReadySlowWindow.source_units` 中属于 owner commit scope 的 unit
   - 优先按 owner seed unit 的 `turn_id` 扩展
   - 若窗口本身是单 turn / 单 speaker，则允许整窗复用
   - 只有在无法推出更大稳定 scope 时，才回退到 owner seed chunk
2. 形成 `owner_source_text`
3. 若 `window_text` 能稳定裁剪出 owner 段，则优先保留 slow/chosen wording
4. 若不能稳定裁剪，则回退到 `owner_source_text`

这个规则的本质是：

1. slow/chosen text 仍然优先
2. 但提交边界必须受 owner source scope 约束
3. 当整窗文本不能证明自己只属于 owner 时，不允许直接进入 canonical

### 7.3 AlignmentStage 不再做整窗 promotion

`_resolve_preparation_text_inputs()` 需要从“提升为整窗文本”改为“标准化双真相输入”：

1. 不再根据 `ready_window.source_units` 长度把 canonical 输入提升成整窗 text
2. 如果 `ctx.selected_text_truth` 缺失或不完整，只补齐 owner/window 双真相的正式字段
3. preparation 输入必须保持 selection 已决定的 owner commit scope，不得再扩大

### 7.4 Preparation 只允许 owner truth 进入 canonical

Preparation 的正式约束改为：

1. canonical sequence 的文本来源 = `SelectedTextTruth.owner_text`
2. provenance 中的 commit scope = `SelectedTextTruth.owner_source_chunk_ids`
3. window scope 仍保留在 `PreparationBundle.source_chunk_ids`
4. `window_text_truth` 只允许留在 metadata / debug / punctuation remap 辅助输入中

建议在 `PreparationBundle` 或 `PreparationScope` 中新增正式 commit scope 字段，而不是继续把它藏在 provenance metadata 里。推荐最小改动方案：

1. `PreparationScope`
   - 继续表达 window scope
2. 新增 `commit_scope_chunk_ids / commit_scope_chunk_indices`
   - 正式表达本次 southbound replace scope

这样 downstream 不需要再从 `SelectedTextTruth` 回溯 commit scope。

### 7.5 OutputProjectionInput 显式区分 window scope 与 replace scope

`OutputProjectionInput` 至少应拆出两组字段：

1. `source_chunk_ids / source_chunk_indices`
   - 当前窗口覆盖范围，仅用于 window metadata / overlap context
2. `replace_scope_chunk_ids / replace_scope_chunk_indices`
   - 本次 owner commit 实际要替换的 southbound scope

projector 的默认行为调整为：

1. `projection_meta.source_chunk_ids = window scope`
2. `projection_meta.replace_scope_chunk_ids = commit scope`
3. `SentenceRecord.replace_scope_chunk_ids` 默认回填 commit scope，而不是整窗 source scope

### 7.6 southbound dispatch 仍保留 `window_group`，但 cleanup 范围改为 owner scope

`projection_mode=window_group` 可以保留，因为这只是 southbound transport 形状。

真正需要改的是 cleanup 语义：

1. 仍允许目标 chunk_id 使用 `ow-<window_id>`
2. 但 `StreamingSubtitleManager.replace_chunk_scope()` 必须只清理 `replace_scope_chunk_ids`
3. 不再允许 projector 把整窗 `source_chunk_ids` 默认冒充成 replace scope

这意味着：

1. window metadata 仍是 window-first
2. cleanup / commit 变成 owner-first

## 8. 关键不变量

本次重构完成后，系统必须满足以下不变量：

### 8.1 真相不变量

1. 多 source chunk window 下，`SelectedTextTruth.owner_text` 不得默认等于整窗 `window_text`
2. `SelectedTextTruth.owner_source_chunk_ids` 必须是非空集合
3. `owner_source_chunk_ids` 必须是 `window_source_chunk_ids` 的子集

### 8.2 canonical 不变量

1. `PreparationBundle.canonical_sequence` 只允许来自 owner truth
2. `canonical_token_count` 的增长不能再由整窗 promotion 驱动
3. preparation 不允许重新扩大 commit scope

### 8.3 projection 不变量

1. `projection_meta.source_chunk_ids` 与 `projection_meta.replace_scope_chunk_ids` 必须可区分
2. `replace_scope_chunk_ids` 不得默认回退为整窗 source scope
3. southbound cleanup 只能按 commit scope 清理

### 8.4 行为不变量

1. 前一个 owner window 不能再提前吃掉后一个 chunk 的提交权
2. 多 chunk window 的窗口级上下文仍可用于 punctuation / diagnostics / auxiliary remap
3. 若 owner truth 无法稳定提取，应退回 owner source text，而不是整窗放大

## 9. 对现有模块的影响

### 9.1 `selection/service.py`

需要新增：

1. owner/window 双真相构造逻辑
2. owner scope 解析逻辑
3. owner text 提取/裁剪逻辑

### 9.2 `contracts.py`

需要把 `SelectedTextTruth` 升级为正式双真相契约，并保留兼容属性。

### 9.3 `alignment_stage_service.py`

需要删除或重写：

1. `_resolve_preparation_text_inputs()` 中的整窗 promotion 语义

需要新增：

1. 双真相标准化
2. trace 中 owner/window truth 的分开输出

### 9.4 `preparation/contracts.py` 与 `preparation/assembler.py`

需要引入正式 commit scope，并把 canonical 真源切到 owner truth。

### 9.5 `output_projection/output_projector.py`

需要将 commit scope 从 window scope 中拆出，southbound 默认 replace scope 改为 owner commit scope。

## 10. 测试策略

### 10.1 单元测试

至少要覆盖以下三类行为：

1. `SelectionService`
   - 多 source chunk window 下，选择层产出 `window_text_truth` 和 `owner_text_truth`
   - owner truth 只能包含 owner source units 对应的文本
   - 若 owner source units 覆盖整窗，则允许 owner truth 与 window truth 相等
2. `AlignmentPreparationAssembler`
   - canonical sequence 基于 owner truth
   - window truth 只保留为辅助元数据
3. `OutputProjector`
   - `replace_scope_chunk_ids` 默认取 commit scope
   - `source_chunk_ids` 仅保留 window metadata

### 10.2 集成测试

至少要覆盖：

1. `test_no_inline_preparation_left.py`
   - preparation 不再做整窗 promotion
2. `test_output_integration.py`
   - window_group transport 下 cleanup scope 正确
3. `test_timeanchored_main_chain_routing.py`
   - projected chunk count 与 route 仍成立

### 10.3 端到端观察指标

实现后建议继续对 `test_en_1` 样本复跑，并重点看：

1. `05_selection.output.json`
   - owner/window truth 是否已分离
2. `10_preparation.input.json`
   - selected_text_len 是否不再被整窗 promotion 放大
3. `11_preparation.output.json`
   - `canonical_token_count - observation_slice_count` 是否收敛
4. 最终 SRT
   - 5 秒 / 10 秒级大空洞是否消失

## 11. 风险与缓解

### 11.1 风险：owner truth 过窄，导致漏提交通道

如果 owner text 提取过于保守，可能出现：

1. 文本变短
2. punctuation 映射变弱
3. 单个 owner window 提交句子数下降

缓解方式：

1. 先用 deterministic owner extraction
2. 保留 `window_text_truth` 供 punctuation remap 与 debug 对照
3. 用 targeted tests 锁住“owner 不串窗”和“不会空提交”两个边界

### 11.2 风险：scope 调整影响 southbound 清理语义

如果 `replace_scope_chunk_ids` 改错，可能出现：

1. 老句子残留
2. 邻近 owner 句子被误清

缓解方式：

1. output projection 合同测试必须新增 window scope / replace scope 分离断言
2. output integration 测试必须覆盖 overlapping scope 场景

### 11.3 风险：兼容属性不足导致隐式回归

`SelectedTextTruth` 被多个测试和 trace 直接序列化读取，改动时必须保留兼容字段，避免一次性炸掉整片测试夹具。

## 12. 最小闭环边界

本次方案严格限制在以下最小闭环：

1. `backend/app/services/timeanchored_alignment/contracts.py`
2. `backend/app/services/timeanchored_alignment/selection/contracts.py`
3. `backend/app/services/timeanchored_alignment/selection/service.py`
4. `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
5. `backend/app/services/timeanchored_alignment/preparation/contracts.py`
6. `backend/app/services/timeanchored_alignment/preparation/assembler.py`
7. `backend/app/services/timeanchored_alignment/output_projection/output_projector.py`
8. 对应单元/集成测试

不扩散到：

1. decoder 主算法
2. decision planner 主算法
3. prompt / bridge / slow window builder

## 13. 结论

当前问题的本质不是“Whisper 识别错了”，也不是“贪心参数太激进”，而是系统把“窗口辅助真相”和“owner 提交真相”混成了一份合同。

因此，彻底方案必须先完成两件事：

1. `SelectedTextTruth` 正式拆成 `window truth + owner truth`
2. `OutputProjectionInput` 正式拆成 `window scope + replace scope`

只有把这两个分层立住，后续 decoder / decision / punctuation 的任何修复才不会再被 preparation promotion 和 southbound whole-window cleanup 抵消掉。
