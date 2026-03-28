# Timeanchored 智能判型与局部修复重设计

**Status:** Proposed

**Date:** 2026-03-28

**Owner:** wgh / Codex

## 1. 背景与问题定义

当前 timeanchored 主链在复杂窗口上暴露了两类结构性问题：

1. speaker 真值虽然已在 timeline 中存在，但没有稳定进入 Decision 主链，导致该切的地方不切。
2. AnchorMount 在复杂挂载场景下会继续产出“看起来有时间戳、但实际上不合法”的时间轴，后续 Decision 只能补锅，最终形成极短碎片、时间回跳、跨段混乱，甚至在多 chunk 窗口上造成重复字幕。

已确认的问题样例包括：

- `jobs/p-20260328-193846-tr-test-en-1-bi5n`
  - `#51 #52` 实际是 `#49` 的重复
  - `#64 #65 #66 #67` 实际是 `#62 #63` 的重复
- `jobs/p-20260328-202014-tr-test-en-1-248i`
  - `#31 #32 #33` 被错误切分且时间戳错误
  - `#42` 在 `Oh` 前存在 speaker 切换，但未切
  - `#49-#53` 时间戳混乱

这些问题说明：当前系统不是“只差一个阈值”，而是缺少“复杂失败场景的正式建模、局部修复机制、合法性验证、原子提交边界”。

## 2. 为什么不能简单地“挂不上就回退快流”

整窗直接回退快流不是根治方案，原因有四个：

1. timeanchored 是 `window-first` 模型，结果覆盖多个 `source_chunk_ids`；当前 `fast_direct` 是 owner chunk 本地提交。多 chunk 窗口一旦 owner-only 回退，就会形成“新结果只覆盖一部分、旧结果残留另一部分”，这正是重复字幕的结构性来源。
2. 快流结果只能提供较粗的时间参考，不能等价替代 slow truth 的文本、speaker 约束和窗口级切分语义。
3. 如果把“should_fallback=true”直接映射为整窗替换，系统会倾向于用低质量替代掩盖上游对齐失败，长期反而削弱对 timeanchored 核心问题的暴露与修复。
4. 用户需求不是“做不了就退”，而是“能智能判断，并尽量修好局部坏段，而不是让整窗质量被最差部分拖垮”。

因此，正确方向不是“取消 fallback”，也不是“整窗直接快流替代”，而是把快流降级为局部时基辅助者，把系统改造成“判型 -> 切分 -> 局部修复 -> 合法性提交”的闭环。

## 3. 设计目标

新设计必须同时满足以下目标：

1. 普通窗口仍走主链，不为少数坏例子拖慢所有 case。
2. 复杂窗口要能被识别为“局部可修”，而不是只得到一个模糊的 `should_fallback`。
3. speaker 真值必须前置成约束，而不是后补。
4. 系统要允许“保住大部分合法结果，只局部修坏段”，而不是整窗一刀切。
5. 任何时间轴只要不能证明合法，就不允许提交为最终字幕。
6. 多 chunk 窗口的提交必须原子覆盖，不允许 owner-only 替换。

## 4. 非目标

本方案不承诺：

1. 任意极差音频都能被完美挂载。
2. 快流辅助时基后的局部坏段一定与正常 timeanchored 段质量等价。
3. 未来所有字幕问题绝对为零。

本方案要保证的是：

1. 当前这类结构性重复字幕不会再以提交层残留形式出现。
2. 当前这类“时间回跳/极短碎片/过度压缩”的坏时间轴不会再被静默提交。
3. speaker 高置信 turn 不会再因为桥接层失效而悄悄丢失。
4. 修不好时，坏段会被隔离或局部降级，而不是污染整窗。

## 5. 根因归纳

### 5.1 speaker 真值桥接失效

timeline speaker turn 已存在，但注入 Decision 的桥接层仍使用旧字段形状，导致 Decision 实际消费的是 token 自带的旧 speaker，而不是 timeline 真值。

### 5.2 ChainSolver 只做局部冲突裁剪，不保证全局单调

当前 solver 允许保留前缀中更晚的 hook，再接受后缀里更早的 hook，从而形成文本顺序前进、时间却回跳的坏链。

### 5.3 TemporalEnvelopeBuilder 会硬插值，不会拒绝

当大量 unresolved/inferred token 落在极小时间窗内时，当前实现仍会强行分配时间，造成多个 token 被压进不合理的小片段。

### 5.4 `should_fallback` 语义过粗

当前只有一个布尔信号，无法区分：

- 仅少量局部 gap 可修
- 需要按 turn 拆窗重挂
- 时间轴已不可信，必须隔离
- 完全 fatal

### 5.5 提交粒度错误

timeanchored 处理的是窗口级覆盖集，但某些降级路径仍按 owner chunk 本地提交，导致多 chunk 窗口无法原子替换。

## 6. 目标态总览

目标态不再是：

`整窗挂载 -> should_fallback? -> 继续产出/整窗回退`

而是：

`窗口判型 -> 策略选择 -> anchor islands + repair gaps -> gap 级修复 -> 合法性验证 -> span-set 原子提交`

核心思想：

1. 先把“确定能证明合法”的部分收成 `anchor islands`
2. 把“不确定但可能可修”的部分标成 `repair gaps`
3. 只对 gap 做局部修复
4. 只有合法 span 才能进入提交

## 7. 核心设计

### 7.1 Window Complexity Analyzer

在正式挂载前引入窗口判型器，对每个 window 产出复杂度画像与推荐策略。至少包含以下指标：

- `hook_density`
- `hook_order_violation_risk`
- `speaker_switch_density`
- `cross_chunk_overlap_density`
- `max_unresolved_run`
- `punctuation_anchor_coverage`
- `timeline_turn_coverage`

判型结果不是布尔值，而是策略建议：

- `strict_monotonic`
- `turn_split_first`
- `phrase_block_mode`
- `repair_first`

### 7.2 Speaker 作为前置约束

speaker turn 不再只在提交前注入 Decision，而要在 preparation/alignment 前段就进入约束系统：

1. 高置信 speaker 边界可作为 block 切分参考
2. 横跨 speaker turn 的疑难段优先按 turn 拆开再挂
3. speaker 不再只是“切分证据”，而是“挂载边界的软硬约束之一”

这样像 `Oh` 前切点这类 case，不会因为后补失败而被整段继承旧 speaker。

### 7.3 Anchor Islands + Repair Gaps

重写挂载目标：不是尽量让所有 token 都有时间，而是先构造严格单调的合法岛。

- `anchor island`
  - 一段内部单调、边界可信、可证明合法的 token span
- `repair gap`
  - 岛与岛之间无法直接合法挂载的 span

只要某个 block 会破坏全局单调、压缩率超标或跨越不可信边界，就不能进入 island，必须进入 gap。

### 7.4 局部修复梯子

每个 `repair gap` 按固定梯子处理，而不是整窗回退：

1. `local_reanchor`
   - 缩小上下文，只重挂该 gap
2. `turn_aware_split`
   - 如 gap 横跨 speaker turn，则先按 turn 切小段再挂
3. `bounded_fast_timebase_borrow`
   - 仍挂不上时，仅借快流时间基做局部时间包络，不替换 slow truth 文本主导权
4. `quarantine`
   - 仍无法证明合法时，隔离该 gap，不允许其污染邻近合法段

这里的关键不是“快流替代”，而是“快流只作为局部时基辅助”。系统仍保留 slow truth 作为文本真相。

### 7.5 Timeline Validity Validator

在时间轴进入 Decision/提交前，必须统一跑合法性验证器。至少校验：

1. token 时间全局单调
2. block 内部无回跳
3. 压缩率不超阈值
4. unresolved run 与可用时间窗匹配
5. speaker/timeline 重叠显著时，不允许 speaker 注入为空

输出不再是 `should_fallback`，而是结构化状态：

- `valid`
- `repairable_local`
- `repairable_split`
- `quarantined`
- `fatal`

### 7.6 Span-Set Atomic Commit

提交层不再围绕 owner chunk，而围绕“本次 window 实际覆盖的 span 集合”：

1. 先明确本次 window 覆盖了哪些 chunk / 旧句子 / 时间段
2. 统一 replace 整个覆盖集
3. 只有被 validator 判为 `valid` 的 span 才能提交
4. `quarantined` span 不能和 `valid` span 混在同一次正常提交中伪装成完整结果

这样可以制度性杜绝“半窗新结果 + 半窗旧残留”。

## 8. 新状态机

### 8.1 旧状态机

- `anchor_mount ok`
- `anchor_mount should_fallback`
- `fast_direct`

这套状态机的问题是：信息量太少，不能驱动局部修复。

### 8.2 新状态机

每个 window 最终进入以下之一：

1. `valid`
   - 正常进入 Decision 与输出投影
2. `repairable_local`
   - 仅少量 gap 需要局部重挂
3. `repairable_split`
   - 需要按 turn/phrase 重切为更小子窗
4. `quarantined`
   - 大部分正常，但局部 span 不可信；只允许提交合法部分，并显式记录隔离段
5. `fatal`
   - 当前 window 无法生成可信输出，需显式失败或转入更高层恢复逻辑

这使系统具备“智能判断”的表达能力，而不是只能在“继续”与“整窗回退”之间二选一。

## 9. 为什么这不是“做不了就直接回退”

本方案明确拒绝把复杂失败场景简化成整窗回退，原因是：

1. 失败通常不是全窗均匀失败，而是局部 gap 失败。
2. 许多窗口中，80% 以上 token 本来就可以构成合法 island。
3. 整窗回退会把最差部分的低质量时基扩散到整窗，实际是过度降级。
4. 局部 gap 修复可以在保持 slow truth 文本主导权的同时，只借用必要的辅助时间信息。

因此，系统的优先级应是：

`局部修复 > 局部借时基 > 局部隔离 > 整窗失败`

而不是：

`挂不上 > 整窗快流替代`

## 10. 最小闭环重构范围

本方案接受局部重构，但范围必须限制在以下闭环：

1. `alignment_stage_service`
   - speaker 真值注入
   - 新状态机路由
   - span-set 提交
2. `anchor_mount/contracts.py`
   - block / island / gap / validity 契约
3. `anchor_mount/chain_solver.py`
   - 从贪心拼块改为合法 island 求解
4. `anchor_mount/temporal_envelope_builder.py`
   - 从硬插值改为“可拒绝的 envelope 生成”
5. `anchor_mount/service.py`
   - 引入复杂度判型、gap 修复、状态收敛
6. `decision_ingress_adapter.py`
   - speaker/timeline 真值与 gap 状态正确投影
7. 输出投影与提交相关模块
   - 从 owner chunk 提交改为 span-set 原子提交

这已经是解决当前问题所需的最小闭环，不应扩散为无关的大重写。

## 11. 可以严格保证的后果

完成该设计后，可以严格保证以下结构性后果被阻断：

1. 多 chunk window 不会再因 owner-only fallback 形成重复字幕残留。
2. 无法证明合法的时间轴不会再以正常字幕身份落地。
3. speaker 高置信真值不会再因为旧字段桥接失效而静默丢失。
4. 大量 token 被压进极小时间窗的异常，会在 validator 阶段被识别为 `repairable` 或 `quarantined`，而不是继续伪造正常输出。

## 12. 仍然存在的真实边界

即便完成该设计，以下边界仍需诚实保留：

1. 极端噪声、严重串话、极差 diarization 的 case，可能仍有局部段无法高质量恢复。
2. `bounded_fast_timebase_borrow` 只是时基辅助，不保证该局部段一定达到正常 anchor 成功段的质量。
3. 若上游文本真值本身错误，新的时间轴合法性设计也不能修正文本内容本身。

但新的制度保证是：修不好时系统会显式隔离，不再输出“看似完整、实际错误”的坏字幕。

## 13. 推荐落地顺序

建议按以下顺序实施：

1. 修 speaker 真值桥接，避免继续被旧字段污染。
2. 引入 validity validator 与新状态机，先把 fail-open 改成 fail-closed。
3. 重写 ChainSolver 与 TemporalEnvelopeBuilder，先保证合法 island 和拒绝能力。
4. 再引入 repair gap 梯子与局部借时基。
5. 最后把提交层改成 span-set 原子提交，彻底切断重复残留。

## 14. 最终结论

要避免当前问题，关键不是“Anchor 挂不上时直接选快流”，而是把 timeanchored 从“整窗硬算、失败继续、必要时粗暴替代”重构为“智能判型、局部修复、合法性先行、覆盖集原子提交”。

这套设计的核心价值不在于承诺“以后永远不会挂不上”，而在于建立以下制度闭环：

1. 能挂好的部分，继续高质量输出
2. 挂不好的部分，优先局部修复
3. 修不好的部分，局部隔离或受限借时基
4. 不能证明合法的结果，绝不再提交成最终字幕

这才是既保质量、又不靠粗暴回退掩盖问题的长期方案。
