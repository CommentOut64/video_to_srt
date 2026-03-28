# Timeanchored Anchor Densification Mainline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不引入重型算法路由的前提下，重构 timeanchored 前半链，使其以高可信锚点为核心，使用稀疏 DP 选择主锚链、在 gap 内局部 DP/NW 增密可信锚点、按层级熔断 fail-closed，并最终显著减少重复字幕、时间戳混乱、speaker 漏切与 unresolved token 强行塞入。

**Architecture:** 新主线采用“严格 seed 骨架锚 -> block-level sparse DP -> anchor islands -> gap 内局部 DP/NW -> 2~3 轮 bounded densification -> 极小残余 gap 插值 -> 层级熔断与合法性门禁”的结构。`seed` 概念保留，但收缩为受限、高精度的起始锚发现；`Window Complexity Analyzer` 不再承担多路算法路由，只保留轻量观测和 rescue gate 角色。算法改造之外，必须同步修复 speaker 真值桥接和多 chunk window 的原子提交，否则即使前半链改好，仍会继续落地坏字幕。

**Tech Stack:** Python 3.10+、`backend/app/services/timeanchored_alignment/anchor_mount/*`、`dual_pipeline/alignment_stage_service.py`、pytest、Windows PowerShell、timeanchored debug trace JSON

---

## 0. ROI Gate（本次选择 B：局部重构）

当前问题已明确跨越 `seed candidate -> local block -> chain solve -> temporal envelope -> fallback/fuse -> stage commit` 六个强关联节点，无法用单点补丁稳定修复；继续补丁只会增加特殊分支，并把低可信锚点继续送进插值链，放大时间漂移与重复提交风险。选择 **B（局部重构）**：只重构 timeanchored 前半链与其必要的提交闭环，范围限制在 `AnchorMount + stage 路由/提交 + speaker bridge` 这个最小闭环内，不扩散到无关的 ASR / Decision 大重写。

## 1. 本计划相对前一版设计的修正

本计划以本文件为准，并修正此前讨论中的两点过度设计：

1. **不做重型 Window Complexity Analyzer 路由**
   - 不维护多套主算法，不把 window 先分类再切不同主链。
   - 只保留轻量指标，用于 observability、rescue 触发和 hard fuse。
2. **不把插值视为错误本身**
   - gap 插值仍保留。
   - 但插值只能发生在“左右边界 anchor 足够可信”的极小残余 gap 上。
   - 不再让插值承担“补整段失败时间轴”的职责。

## 2. 当前已确认的关键判断

### 2.1 当前坏窗不是“天然缺锚”，而是“有 hook 但锚链利用失败”

已确认的坏窗中，token 与 hook 数量接近 1:1：

- `chunk_0011`: `61 token / 60 hook`
- `chunk_0015`: `40 token / 39 hook`
- `chunk_0020`: `46 token / 43 hook`

但 `chunk_0020` 最终仅 `15 anchored / 28 inferred / 3 unresolved`。这说明核心矛盾不是“完全没锚”，而是“候选噪声、链选择与 gap 处理方式让大量潜在锚点失效”。

### 2.2 `seed` 思想不能直接删除

最初设计文档已明确 `seed` 要处理三类粒度关系：

- `1 slot -> 1 hook`
- `N slot -> 1 lexical merge`
- `1 slot -> N hook merge`

现有测试也证明中英日都不能假设粒度完全一致：

- 英文 contraction/normalization
- 中文 merge/split
- 日文 script-aware normalization
- 明确存在 `1 token -> N hooks` 的测试基线

因此本计划保留 `seed` 概念，但要求把当前 `SeedDiscovery` 改为严格受限版本。

### 2.3 当前最危险的设计缺陷是“低可信锚点 + 默认插值”

当前链路在以下两处叠加出坏时间轴：

1. `SeedDiscovery` / `LocalExtension` / `ChainSolver` 选出来的锚链本身可信度不足
2. `TemporalEnvelopeBuilder` 把 gap 默认转换为 inferred/unresolved timing，并继续向后提交

因此真正要重建的不是“有没有插值”，而是“锚点可信度体系 + 只有在可信边界内才允许插值”。

## 3. 成功标准

### 3.1 功能标准

1. 多 chunk window 不再因降级/回退出现重复字幕残留。
2. 坏时间轴不会再以“正常字幕”身份提交。
3. speaker 高置信切换能稳定进入前半链约束或 Decision 输入，不再因旧字段桥接失效而漏切。
4. unresolved token 不再大段直接硬塞；最终插值只用于小残差 gap。

### 3.2 算法标准

1. 主锚链由全局单调、总分最优的 sparse DP 选出，而非贪心 suffix reseed。
2. gap rescue 必须以局部 DP/NW 为主，允许 2~3 轮 bounded densification。
3. 任何 promoted anchor 都必须通过严格 trust 评估。
4. gap rescue 失败时必须熔断，不能继续默认插值。

### 3.3 观测标准

新增并稳定产出以下指标：

- `primary_anchor_count`
- `promoted_anchor_count`
- `promoted_anchor_precision_proxy`
- `dense_round_count`
- `gap_rescue_attempt_count`
- `gap_rescue_success_count`
- `gap_fuse_retry_count`
- `gap_fuse_hard_count`
- `anchor_trust_reject_count`
- `interpolation_residual_token_count`
- `timeline_validity`

## 4. 非目标

1. 不追求在首版中引入多套 window 级算法主线。
2. 不在首版中重写 `Decision` 主内核。
3. 不承诺任意极差音频都能完全恢复高质量时间轴。
4. 不在首版中实现复杂的全量自动参数学习或自适应阈值系统。

## 5. File Map

### 5.1 Create

- `backend/app/services/timeanchored_alignment/anchor_mount/anchor_trust_evaluator.py`
  - 统一计算 `seed/block/promoted anchor` 的可信度，给出 `trust tier / reject reason`
- `backend/app/services/timeanchored_alignment/anchor_mount/gap_rescue_aligner.py`
  - 在 anchor gap 内做局部 DP/NW，对齐 slow token span 与 fast hook span
- `backend/app/services/timeanchored_alignment/anchor_mount/densification_promoter.py`
  - 把 gap rescue 产生的高可信匹配升级成 `secondary anchors`
- `backend/app/services/timeanchored_alignment/anchor_mount/window_alignment_state.py`
  - 定义核心算法唯一状态对象，串联 `seed -> chain -> gap rescue -> densification -> fuse -> validity`
- `backend/app/services/timeanchored_alignment/anchor_mount/timeline_validity_validator.py`
  - 对最终时间轴做合法性校验，输出 `valid / repairable / quarantined / fatal`
- `backend/tests/timeanchored_alignment/test_anchor_trust_evaluator.py`
- `backend/tests/timeanchored_alignment/test_anchor_disambiguation.py`
- `backend/tests/timeanchored_alignment/test_gap_rescue_aligner.py`
- `backend/tests/timeanchored_alignment/test_densification_promoter.py`
- `backend/tests/timeanchored_alignment/test_timeline_validity_validator.py`
- `backend/tests/timeanchored_alignment/test_sparse_dp_chain_solver.py`
- `backend/tests/timeanchored_alignment/test_window_alignment_state.py`
- `docs/superpowers/specs/2026-03-28-timeanchored-hook-token-grain-audit.md`
  - 首轮多语言粒度审计结论文档

### 5.2 Modify

- `backend/app/services/timeanchored_alignment/anchor_mount/contracts.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/seed_discovery.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/low_complexity_mask.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/local_extension.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/chain_solver.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/temporal_envelope_builder.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/fallback_gate.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/service.py`
- `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- `backend/app/services/textflow/decision_ingress_adapter.py`
- `backend/tests/timeanchored_alignment/test_anchor_mount_seed_discovery.py`
- `backend/tests/timeanchored_alignment/test_anchor_mount_service.py`
- `backend/tests/timeanchored_alignment/test_anchor_mount_punctuation_and_envelopes.py`
- `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`
- `backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py`
- `backend/tests/timeanchored_alignment/test_alignment_stage_trace_payloads.py`
- `backend/tests/textflow/test_decision_layer_ingress_context.py`

### 5.3 Optional Modify（仅在审计确认必要时）

- `backend/app/services/timeanchored_alignment/preparation/hook_selector.py`
- `backend/app/services/timeanchored_alignment/window_time_base_assembler.py`

只有当多语言审计确认 `hook` 顺序与时间顺序存在系统性偏差时，才纳入首版改动。

## 6. 核心设计细节

### 6.1 不做重型 Window Complexity Analyzer 路由

首版只保留一个轻量 analyzer，职责缩减为：

1. 输出观测指标
2. 决定是否触发 gap rescue
3. 决定是否进入 hard fuse / safe fallback

它不负责：

1. 选择多套主算法
2. 在 window 级别切换多条完全不同的主链
3. 维护复杂策略矩阵

推荐仅保留以下输入：

- `largest_unresolved_span`
- `hook_waste_ratio`
- `duplicate_candidate_hook_count`
- `promoted_anchor_count`
- `gap_rescue_fail_count`
- `speaker_turn_overlap_missing_flag`
- `global_monotonic_violation_count`

### 6.2 受限 Seed 系统

`seed` 保留，但必须从“高召回候选生成器”改为“高精度骨架锚发现器”。

#### 6.2.1 保留的 seed 类别

1. `exact`
2. `strong_normalized`
3. `bounded_sequence`
   - 仅允许短连续 hook 合并，且必须局部唯一
4. `bounded_merge`
   - 仅允许短连续 token 合并，且必须相邻、局部唯一

#### 6.2.2 首版禁止的 seed 行为

1. 任意长跨度的 `token_i + token_j` 远距离 merge 候选
2. 只凭低信息 token 形成 hard anchor
3. 在一个局部窗口内存在多个近似等价候选时，仍然把候选直接升为 hard anchor
4. 仅凭 pronunciation hint 直接形成可提交 hard anchor

#### 6.2.3 严格筛选标准

每个 seed 至少要经过以下检查：

1. **长度约束**
   - merge/sequence 只允许短跨度
2. **邻接约束**
   - token merge 必须相邻
   - hook sequence 必须连续
3. **唯一性约束**
   - 局部搜索窗内不得有多个近似同分候选
4. **低信息词惩罚**
   - `a/an/the/of/to/in` 等功能词默认不单独形成 primary anchor
5. **语言敏感约束**
   - 英文关注 contraction / apostrophe / hyphen
   - 中文关注字词 merge/split
   - 日文关注 script/normalization

#### 6.2.4 重复词 / 同 hook 多候选消歧

当同一个 `hook_text` 在同一局部窗口内可对应多个相同或近似 `token/slot` 时，不允许使用“最近原则”“先到先得”或 cursor 硬贴。

首版统一引入 `ambiguity cluster` 概念：

1. 候选阶段允许同一个 hook 形成多个局部候选
2. 但这些候选必须被显式收束到同一个歧义簇中
3. 歧义簇中的候选只有在以下条件满足时才能升为 `primary_anchor`：
   - 局部唯一性明显成立
   - 与前后已确认锚点保持单调
   - 与预期对角线距离最小
   - 不跨越 hard speaker / punctuation boundary
   - 不是单独由低信息重复词撑起
4. 如果歧义簇无法在主锚链阶段唯一化：
   - 不得强升为 `primary_anchor`
   - 只允许保留为 `secondary candidate` 或 `rejected`
   - 后续仅可在 gap rescue 中继续判定
5. 如果 gap rescue 后仍不能唯一化：
   - 不得强挂
   - 宁可保留 unresolved，再交给残余小 gap 插值资格判断

硬约束：

1. 一个 hook 在最终结果中不能同时归属多个 token
2. 合法的 `N slot -> 1 hook merge` 只允许发生在连续 token 上
3. 仅因为一句话里有多个相同字词，不能自动推断为 merge

#### 6.2.5 `disambiguation score` 判定表

为了避免重复词场景下依赖拍脑袋式 tie-break，首版为 ambiguity cluster 引入统一 `disambiguation score`。这个分数只用于“同一 hook 的多个候选谁更可能是正确归属”，不替代全局 `anchor trust score`。

推荐由以下维度组成：

1. `surface_match_score`
   - 词面 / normalized / script-aware 匹配强度
2. `diagonal_distance_penalty`
   - 候选 `(unit_index, hook_index)` 与局部预期对角线的距离惩罚
3. `left_context_support`
   - 与左侧已确认 anchor 的顺序与跨度相容性
4. `right_context_support`
   - 与右侧已确认 anchor 的顺序与跨度相容性
5. `local_uniqueness_margin`
   - 当前候选相对于同簇次优候选的优势
6. `boundary_cross_penalty`
   - 跨越 speaker / punctuation hard boundary 时的惩罚
7. `low_information_penalty`
   - 低信息重复词的额外惩罚

首版不要求在线学习权重，但要求有**稳定的优先级顺序**：

1. 先满足单调和边界合法性
2. 再看 `surface_match_score`
3. 再看 `local_uniqueness_margin`
4. 最后用 `diagonal_distance_penalty` 与邻域上下文细化排序

#### 6.2.6 直接拒绝规则（Hard Reject）

以下情况不进入 `primary_anchor` 候选集，直接拒绝：

1. 候选跨越非连续 token，却试图形成 `N slot -> 1 hook merge`
2. 候选会破坏与已确认左右 anchor 的严格单调关系
3. 候选落在 hard speaker boundary 另一侧，且没有独立局部解释
4. 候选与同簇次优项几乎同分，无法形成明显唯一性优势
5. 候选只由低信息重复词支撑，缺乏上下文支撑
6. 候选需要依赖远距离 merge 或无界 sequence 才能成立

#### 6.2.7 `primary / secondary / unresolved` 三态决策

同 hook 多候选场景下，首版明确三态：

1. `primary_anchor`
   - 歧义簇已被唯一化，且满足全部 trust / disambiguation gate
2. `secondary_candidate`
   - 当前无法安全唯一化，但在 gap rescue 中仍有希望结合局部上下文判定
3. `unresolved_candidate`
   - 当前既不能唯一化，也不适合进入 rescue 晋升链

决策规则：

1. `primary_anchor` 只出现在主锚链前
2. `secondary_candidate` 只能在 gap rescue 成功后，通过 promoter 升级为 `secondary_anchor`
3. `unresolved_candidate` 不得参与主锚链，也不得在没有新证据的情况下强行升级

### 6.3 高可信锚点评估体系

首版引入统一 `anchor trust` 评估，不再把 “被匹配到” 直接视为可信锚点。

#### 6.3.1 评分维度

1. `seed_strength`
   - `exact > strong_normalized > bounded_sequence > bounded_merge > pronunciation_soft`
2. `local_uniqueness_margin`
   - 与次优候选的分差
3. `neighbor_chain_support`
   - 是否能与前后高可信点形成稳定单调链
4. `temporal_consistency`
   - 与邻近锚点的时间距离是否合理
5. `structural_consistency`
   - 是否跨越 punctuation/speaker hard boundary
6. `token_information_value`
   - 低信息 token 降权
7. `provenance_consistency`
   - 是否和 `source_chunk_ids / coverage` 一致

#### 6.3.2 Trust Tier

1. `primary_anchor`
   - 可直接进入主锚链 DP
2. `secondary_anchor`
   - 只能在 gap rescue 成功后晋升
3. `boundary_only`
   - 不能参与 timing finalize，只可作为边界提示
4. `rejected`
   - 仅保留 diagnostics

#### 6.3.3 首版硬规则

1. 只有 `primary_anchor` 能进入第 0 轮 sparse DP 主锚链。
2. 只有通过 promote gate 的 gap 匹配点，才能升为 `secondary_anchor`。
3. `secondary_anchor` 不允许反向推翻高 trust 的 `primary_anchor`，只允许切小 gap。
4. ambiguity cluster 中的候选，必须同时通过 `anchor trust` 与 `disambiguation score` 双门禁，才能升级为 `primary_anchor`。
5. 如果 ambiguity cluster 只在 gap rescue 后才被唯一化，该点最多升级为 `secondary_anchor`，不回溯篡改原始 `primary_anchor` 语义。

### 6.4 用 block-level sparse DP 代替当前贪心 ChainSolver

首版不做全窗 token x hook 大 DP，而做 **candidate block graph 上的稀疏 DP**。

#### 6.4.1 输入

- 通过 trust gate 的 primary candidate / local blocks

#### 6.4.2 目标

- 选出总分最高、严格单调、结构边界一致的全局主锚链

#### 6.4.3 必须满足的约束

1. `unit_indices` 严格递增
2. `hook_indices` 严格递增
3. 不允许跨越 hard speaker / punctuation boundary 形成不合理 block
4. 不允许后选 block 回跳到前缀 hook 区间

#### 6.4.4 与当前贪心的关键差异

1. 当前实现只做局部 suffix reseed
2. sparse DP 会在全局范围上比较多条合法链
3. 主锚链一旦选定，后续 gap rescue 只能在锚链之间工作，不能任意改写主链

### 6.5 锚岛之间的 gap 跑局部 DP/NW

在两个已确认锚岛之间，切出局部 `slow token span + fast hook span`，进行受限 DP/NW。

#### 6.5.1 gap rescue 的前提

1. 左右边界锚点至少达到最低 trust 阈值
2. gap span 不得无限扩大，必须使用 corridor
3. 本轮 rescue 只能处理当前 gap，不影响其他 island

#### 6.5.2 gap rescue 的输出

1. `matched_pairs`
2. `local_alignment_score`
3. `promotion_candidates`
4. `failure_reason`

#### 6.5.3 gap rescue 的禁止行为

1. 对齐失败后仍默认继续插值
2. 把低分、不唯一或结构冲突的点晋升成锚点
3. 局部 rescue 直接推翻整条主锚链

### 6.6 2~3 轮 bounded densification

gap rescue 不是只跑一轮。首版采用有限轮次的锚点增密：

1. **第 0 轮**
   - primary anchors -> sparse DP -> main anchor chain
2. **第 1 轮**
   - 对各 gap 跑 local DP/NW
   - 高可信匹配点晋升为 secondary anchors
3. **第 2 轮**
   - 用新增锚点切小 gap，对仍较大的 gap 再跑一次 local DP/NW
4. **停止条件**
   - 没有新的 promoted anchors
   - 残余 gap 已很小
   - 到达上限轮次
   - 触发 fuse

首版上限固定为 **2 次 rescue，最多 3 轮 timing 收敛**，不允许无限迭代。

### 6.7 插值资格（Interpolation Eligibility）

插值保留，但必须受资格控制。

#### 6.7.1 允许插值的条件

1. gap 已经过 local DP/NW rescue
2. 当前残余 gap token 数很小
3. 左右边界 anchor trust 足够高
4. 可用时间窗与 token 数匹配
5. 没有 monotonic violation

#### 6.7.2 禁止插值的条件

1. 左右边界锚点任何一侧 trust 不足
2. 本 gap rescue 失败
3. 残余 token 数过多
4. span 极小但 token 过多
5. gap 横跨 hard speaker boundary 且未重切

### 6.8 层级熔断策略

gap rescue 失败后，必须熔断，但分三层：

#### 6.8.1 gap 级熔断

触发条件：

1. local DP/NW 无法建立稳定路径
2. promotion candidates 全部被 trust gate 拒绝
3. 局部匹配与边界锚点严重不一致

动作：

1. 停止该 gap 的后续插值/晋升
2. 标记 `gap_state = no_alignment`
3. 记录 fail reason

#### 6.8.2 local rollback / reseed

在 gap 级熔断后，如果边界 anchor trust 不高，允许：

1. 回滚 gap 邻近 1~2 个低 trust anchor
2. 仅在该局部 corridor 内重新做一次 seed 搜索
3. 重新跑 local block + sparse DP / local NW

首版只允许 **一次 local rollback**。

#### 6.8.3 window 级熔断

触发条件：

1. 多个大 gap 连续失败
2. local rollback 后仍失败
3. promoted anchor 过少，remaining gap 过大
4. 全窗 monotonic / validity 不成立

动作：

1. 禁止继续正常 timeanchored 提交
2. 进入安全替代链

安全替代优先级：

1. 安全的旧对齐链 / bounded fallback alignment
2. bounded fast-timebase borrow
3. 仅在单 source chunk 或未来支持整窗原子替代时，才允许 `fast_direct`

### 6.9 轻量 rescue gate，而非多算法路由

这里保留一个极简 gate，触发逻辑如下：

1. `main chain` 成功且残余很小
   - 直接进 interpolation eligibility
2. `main chain` 成功但存在中等 gap
   - 触发 gap rescue
3. `main chain` 已不稳定
   - 直接 fuse，不继续 densification

这不是多主线算法路由，只是单主线的修复门禁。

### 6.10 核心算法边界与 `WindowAlignmentState`

为了避免新方案落地后演变成“算法逻辑散落在多个 service 中互相回调”，首版必须引入统一核心状态对象 `WindowAlignmentState`，并强制采用单向主线。

#### 6.10.1 单一状态对象

`WindowAlignmentState` 至少收口以下信息：

1. `input_view`
2. `raw_seed_candidates`
3. `trusted_primary_candidates`
4. `ambiguity_clusters`
5. `local_blocks`
6. `main_anchor_chain`
7. `anchor_islands`
8. `open_gaps`
9. `promoted_secondary_anchors`
10. `rescue_round_index`
11. `fuse_events`
12. `timeline_validity`

任何模块都只能消费当前 state 并返回新的 state，不允许隐式改写外部全局状态。

#### 6.10.2 单向数据流

核心算法固定为以下顺序：

1. `SeedDiscovery`
2. `AnchorTrustEvaluator`
3. `SparseChainSolver`
4. `GapRescueAligner`
5. `DensificationPromoter`
6. `TemporalEnvelopeBuilder`
7. `TimelineValidityValidator`

只有 `GapRescueAligner + DensificationPromoter` 之间允许有限轮次循环；其他模块一律单次前向执行。

#### 6.10.3 清晰职责边界

1. `SeedDiscovery`
   - 只产候选，不做最终归属决策
2. `AnchorTrustEvaluator`
   - 只做 trust tier / reject reason 评估
3. `SparseChainSolver`
   - 只选主锚链，不做 timing finalize
4. `GapRescueAligner`
   - 只在 island 之间对局部 span 做 DP/NW
5. `DensificationPromoter`
   - 只负责 secondary anchor 晋升
6. `TemporalEnvelopeBuilder`
   - 只负责 timing 收口与残余 gap 插值资格执行
7. `TimelineValidityValidator`
   - 只负责合法性门禁
8. `AlignmentStageService`
   - 只做 route、speaker bridge、atomic commit，不做核心对齐决策

#### 6.10.4 首版明确禁止的混乱模式

1. `TemporalEnvelopeBuilder` 再偷偷做对齐判定
2. `HookClaimResolver` 再去做重复词消歧
3. `AlignmentStageService` 里堆叠核心算法分支
4. gap rescue 失败后绕过 fuse 直接补时
5. 多个模块各自维护一套 trust / fallback / validity 规则

#### 6.10.5 设计目标

最终算法应呈现为“一条完整主线 + 一个有限 rescue 循环 + 一套统一状态”，而不是分散在多个 service 中互相打补丁。

## 7. 必须同步落地的非算法支撑项

### 7.1 speaker 真值桥接

虽然本计划核心是前半链，但 speaker 漏切已明确不是纯切分层问题，首版必须同步修：

1. `AlignmentStageService` 不再读取旧 `decision_ingress.tokens`
2. timeline speaker truth 必须基于 `anchored_token_units` 注入正确入口
3. 若窗口与 timeline 明显重叠但注入为空，必须记 diagnostics，必要时进入 degraded / invalid

### 7.2 多 chunk window 原子提交

算法再好，如果提交继续 owner-only，重复字幕仍会发生。首版必须同步收口：

1. timeanchored 输出必须按 coverage span-set 替换
2. 多 chunk window invalid 时，不允许 owner-only fast fallback
3. 安全替代链也必须尊重整窗替换边界

## 8. 多语言 hook / token 粒度审计

在首轮真正改算法前，先做一轮中英日粒度审计，确认 `seed` 应保留到什么程度。

### 8.1 审计目标

统计以下关系的占比与典型场景：

1. `1:1 exact/normalized`
2. `1:N split`
3. `N:1 merge`
4. `N:M complex`
5. `script/normalization mismatch`
6. `low-information false anchor`

### 8.2 审计范围

1. 英文测试任务
2. 中文现有 text aligner 基线
3. 日文现有 text aligner 基线
4. 至少抽取一批真实 jobs trace

### 8.3 审计结论如何影响实现

1. 如果绝大多数为 `1:1`，可继续收缩 seed
2. 如果 `1:N/N:1` 高发，则保留 bounded seed
3. 如果日文主要问题是 script/normalization，则日文策略要偏向 normalization/pronunciation，而不是复杂 merge seed

## 9. Chunked Plan

## Chunk 1: 审计基线与契约冻结

### Task 1.1: 输出多语言 hook/token 粒度审计结论

**Files:**
- Create: `docs/superpowers/specs/2026-03-28-timeanchored-hook-token-grain-audit.md`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_seed_discovery.py`
- Modify: `backend/tests/timeanchored_alignment/test_alignment_preparation.py`
- Modify: `backend/tests/timeanchored_alignment/test_text_aligner_en.py`
- Modify: `backend/tests/timeanchored_alignment/test_text_aligner_zh.py`
- Modify: `backend/tests/timeanchored_alignment/test_text_aligner_ja.py`

- [ ] **Step 1: 汇总现有中英日粒度不一致基线，列出确证样例**
- [ ] **Step 2: 补充缺失的 bounded merge / bounded sequence / low-information false anchor 测试**
- [ ] **Step 3: 从真实 jobs 中补 1 批 trace 统计样例**
- [ ] **Step 4: 写出审计结论文档，明确首版保留/禁止的 seed 类型**
- [ ] **Step 5: 运行相关测试，确认基线稳定**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/timeanchored_alignment/test_anchor_mount_seed_discovery.py backend/tests/timeanchored_alignment/test_alignment_preparation.py backend/tests/timeanchored_alignment/test_text_aligner_en.py backend/tests/timeanchored_alignment/test_text_aligner_zh.py backend/tests/timeanchored_alignment/test_text_aligner_ja.py -v
```

Expected:

- 粒度相关测试全部通过
- 审计文档形成明确的 seed 边界结论

### Task 1.2: 冻结新的核心契约

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/contracts.py`
- Create: `backend/app/services/timeanchored_alignment/anchor_mount/window_alignment_state.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_contracts.py`
- Create: `backend/tests/timeanchored_alignment/test_window_alignment_state.py`

- [ ] **Step 1: 为 candidate / block / promoted anchor / validity state 增加明确契约字段**
- [ ] **Step 2: 定义 `trust_tier / reject_reason / gap_state / timeline_validity / ambiguity_cluster_id`**
- [ ] **Step 3: 新建 `WindowAlignmentState`，锁住单向状态流所需字段**
- [ ] **Step 4: 写契约与 state 测试，锁住字段与不变量**
- [ ] **Step 5: 运行契约测试**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/timeanchored_alignment/test_anchor_mount_contracts.py backend/tests/timeanchored_alignment/test_window_alignment_state.py -v
```

Expected:

- 合同测试通过

## Chunk 2: 受限 Seed 与高可信锚点评估

### Task 2.1: 收缩 `SeedDiscovery`

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/seed_discovery.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/low_complexity_mask.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_seed_discovery.py`
- Create: `backend/tests/timeanchored_alignment/test_anchor_disambiguation.py`

- [ ] **Step 1: 移除远距离 merge 候选与无界 sequence 扩张**
- [ ] **Step 2: 仅保留 bounded seed 类型**
- [ ] **Step 3: 让低信息 token 默认不能单独形成 primary anchor**
- [ ] **Step 4: 为重复词 / 同 hook 多候选场景建立 ambiguity cluster 测试**
- [ ] **Step 5: 为 `disambiguation score` 的 hard reject / unresolved 分支建立测试**
- [ ] **Step 6: 增加对应失败/成功测试**
- [ ] **Step 7: 运行 seed 测试**

### Task 2.2: 引入 `AnchorTrustEvaluator`

**Files:**
- Create: `backend/app/services/timeanchored_alignment/anchor_mount/anchor_trust_evaluator.py`
- Create: `backend/tests/timeanchored_alignment/test_anchor_trust_evaluator.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/service.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_disambiguation.py`

- [ ] **Step 1: 定义 trust 维度与 tier**
- [ ] **Step 2: 对 seed/block 输出 `primary / secondary / boundary_only / rejected`**
- [ ] **Step 3: 增加独立 `disambiguation score` 规则，不与全局 trust 混用**
- [ ] **Step 4: 让重复词歧义簇只能在 `trust + disambiguation` 双门禁通过时进入 `primary_anchor`**
- [ ] **Step 5: 把 trust gate 接到主服务入口**
- [ ] **Step 6: 运行 trust 相关测试**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/timeanchored_alignment/test_anchor_mount_seed_discovery.py backend/tests/timeanchored_alignment/test_anchor_disambiguation.py backend/tests/timeanchored_alignment/test_anchor_trust_evaluator.py backend/tests/timeanchored_alignment/test_anchor_mount_service.py -v
```

Expected:

- 只有高 trust seed 能进入主锚链候选集

## Chunk 3: block-level sparse DP 主锚链

### Task 3.1: 用 sparse DP 重写 `ChainSolver`

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/local_extension.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/chain_solver.py`
- Create: `backend/tests/timeanchored_alignment/test_sparse_dp_chain_solver.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_disambiguation.py`

- [ ] **Step 1: 保留 `LocalExtension` 负责构造 bounded blocks**
- [ ] **Step 2: 让 `ChainSolver` 在 block graph 上做全局最优单调 DP**
- [ ] **Step 3: 增加 post-solve monotonic validator**
- [ ] **Step 4: 覆盖“贪心会错、DP 应选另一条链”的回归**
- [ ] **Step 5: 覆盖“同 hook 对多个重复词时，DP 结合上下文应选正确位置”的回归**
- [ ] **Step 6: 运行 solver 测试**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/timeanchored_alignment/test_sparse_dp_chain_solver.py backend/tests/timeanchored_alignment/test_anchor_disambiguation.py backend/tests/timeanchored_alignment/test_anchor_mount_service.py -v
```

Expected:

- 旧贪心坏例子被 sparse DP 纠正
- 输出链严格单调

## Chunk 4: Gap Rescue 与 Bounded Densification

### Task 4.1: 实现 gap 内局部 DP/NW

**Files:**
- Create: `backend/app/services/timeanchored_alignment/anchor_mount/gap_rescue_aligner.py`
- Create: `backend/tests/timeanchored_alignment/test_gap_rescue_aligner.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/service.py`

- [ ] **Step 1: 定义 gap rescue 输入输出**
- [ ] **Step 2: 在左右边界 trust 合法时，对 gap 跑局部 DP/NW**
- [ ] **Step 3: rescue 失败时返回显式 failure reason**
- [ ] **Step 4: 运行 gap rescue 测试**

### Task 4.2: 实现 2~3 轮 bounded densification

**Files:**
- Create: `backend/app/services/timeanchored_alignment/anchor_mount/densification_promoter.py`
- Create: `backend/tests/timeanchored_alignment/test_densification_promoter.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/service.py`

- [ ] **Step 1: 定义 promoted anchor 的 trust gate**
- [ ] **Step 2: 只有 ambiguity cluster 在 rescue 后被唯一化时，才允许该点从 `secondary_candidate` 升级为 `secondary_anchor`**
- [ ] **Step 3: 第 1 轮 rescue 后允许把高可信点晋升为 secondary anchors**
- [ ] **Step 4: 第 2 轮仅处理剩余较大 gap**
- [ ] **Step 5: 加入固定停止条件**
- [ ] **Step 6: 运行 densification 测试**

### Task 4.3: 把插值缩到残余小 gap

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/temporal_envelope_builder.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_punctuation_and_envelopes.py`

- [ ] **Step 1: 引入 interpolation eligibility**
- [ ] **Step 2: rescue 失败或边界 trust 不足时，禁止默认 inferred 插值**
- [ ] **Step 3: 只对极小残余 gap 保留插值**
- [ ] **Step 4: 运行 envelope 测试**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/timeanchored_alignment/test_gap_rescue_aligner.py backend/tests/timeanchored_alignment/test_densification_promoter.py backend/tests/timeanchored_alignment/test_anchor_mount_punctuation_and_envelopes.py backend/tests/timeanchored_alignment/test_anchor_mount_service.py -v
```

Expected:

- 大 gap 不再直接被默认插值吞掉
- promoted anchors 能显著缩小剩余 gap

## Chunk 5: 层级熔断与合法性门禁

### Task 5.1: 实现 `TimelineValidityValidator`

**Files:**
- Create: `backend/app/services/timeanchored_alignment/anchor_mount/timeline_validity_validator.py`
- Create: `backend/tests/timeanchored_alignment/test_timeline_validity_validator.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/service.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/fallback_gate.py`

- [ ] **Step 1: 校验全局单调、压缩率、残余 gap、speaker overlap 等规则**
- [ ] **Step 2: 输出 `valid / repairable / quarantined / fatal`**
- [ ] **Step 3: 用新状态替代单布尔 `should_fallback` 的核心语义**
- [ ] **Step 4: 运行 validity / fallback 测试**

### Task 5.2: 实现层级熔断

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/service.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_service.py`
- Modify: `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`

- [ ] **Step 1: 先做 gap 级 fuse**
- [ ] **Step 2: 失败后允许一次 local rollback / reseed**
- [ ] **Step 3: 多次失败升级到 window 级 fuse**
- [ ] **Step 4: 首版安全替代链优先走安全对齐/受限时基借用，不直接整窗快流替代**
- [ ] **Step 5: 运行 routing / service 测试**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/timeanchored_alignment/test_timeline_validity_validator.py backend/tests/timeanchored_alignment/test_anchor_mount_service.py backend/tests/timeanchored_alignment/test_alignment_stage_routing.py backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py -v
```

Expected:

- rescue 失败的 gap 不再继续向后插值
- window 级 fuse 能阻断坏时间轴提交

## Chunk 6: 支撑闭环修复与回归

### Task 6.1: 修 speaker 真值桥接

**Files:**
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- Modify: `backend/app/services/textflow/decision_ingress_adapter.py`
- Modify: `backend/tests/timeanchored_alignment/test_alignment_stage_trace_payloads.py`
- Modify: `backend/tests/textflow/test_decision_layer_ingress_context.py`

- [ ] **Step 1: 改为从 `anchored_token_units` 计算窗口范围**
- [ ] **Step 2: timeline speaker truth 正确注入 Decision 输入**
- [ ] **Step 3: 重叠明显但注入为空时输出 diagnostics**
- [ ] **Step 4: 运行 speaker / ingress 相关测试**

### Task 6.2: 修多 chunk 原子提交与安全降级边界

**Files:**
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- Modify: `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`
- Modify: `backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py`

- [ ] **Step 1: 用 coverage/span-set 替代 owner-only 提交假设**
- [ ] **Step 2: 多 chunk invalid 时禁止 owner-only `fast_direct`**
- [ ] **Step 3: 安全替代链必须遵守整窗替换边界**
- [ ] **Step 4: 运行 routing 回归**

### Task 6.3: jobs 回归与 trace 指标核验

**Files:**
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_service.py`
- Modify: `backend/tests/timeanchored_alignment/test_alignment_stage_trace_payloads.py`
- Modify: `backend/tests/timeanchored_alignment/test_job_punctuation_fact_chain_regression.py`

- [ ] **Step 1: 为本轮 root-cause jobs 添加结构化回归样例**
- [ ] **Step 2: 校验新增 observability 指标已写入 trace**
- [ ] **Step 3: 运行主回归**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/timeanchored_alignment/test_alignment_stage_trace_payloads.py backend/tests/timeanchored_alignment/test_anchor_mount_service.py backend/tests/timeanchored_alignment/test_alignment_stage_routing.py backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py backend/tests/timeanchored_alignment/test_job_punctuation_fact_chain_regression.py -v
```

Expected:

- 新指标完整输出
- 指定坏例子的结构性问题被锁住

## 10. 文档同步要求

首版主要写设计与实现计划，因此当前只新增本计划文档。真正开始落地代码后，必须同步检查并更新：

- `docs/superpowers/specs/2026-03-28-timeanchored-intelligent-repair-design.md`

若最终实现与本计划一致，再决定是否把稳定后的实现总结写入 `llmdoc/architecture/`。在代码未落地前，不更新 `llmdoc` 当前实现文档，避免把目标态写成现状。

## 11. 验证顺序建议

建议严格按以下顺序推进，避免过早实现复杂集成：

1. 粒度审计与契约冻结
2. 受限 seed + trust gate
3. sparse DP 主锚链
4. gap rescue + densification
5. fuse + validity
6. speaker bridge + atomic commit
7. jobs regression

## 12. 提交策略

1. 不自动执行 `git commit`
2. 每完成一个 chunk，暂停并通知 wgh 手动检查/提交
3. 如果中途发现用户已有改动与本计划冲突，先暂停并重新确认边界

## 13. 计划完成口径

本计划完成并不等于“字幕 bug 绝对为零”，而等于以下结构性闭环已经建立：

1. 主锚链来自高可信锚点，而非贪心拼接
2. gap 先 rescue 增密，再插值，而不是默认强塞
3. rescue 失败会熔断，不再伪造正常时间轴
4. speaker truth 与多 chunk 提交边界不再继续放大前半链错误

只有达到以上闭环，timeanchored 才算从“前半链 fail-open”进入“前半链可解释、可拒绝、可修复”的状态。
