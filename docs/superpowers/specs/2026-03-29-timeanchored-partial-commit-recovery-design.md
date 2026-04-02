# Timeanchored 局部提交与覆盖率重构设计

**Status:** Proposed

**Date:** 2026-03-29

**Owner:** wgh / Codex

## 1. 背景与问题定义

`timeanchored` 主链在重复英文对话场景下出现了成片的超长字幕不切分。问题任务样例：

- `jobs/p-20260329-201905-tr-test-en-1-pxac`

已确认这不是 SRT 写出层问题，也不是标点模型没有提供切点。根因是：

1. `AnchorMount` 在大量 slow window 上覆盖率不足，窗口被判为 `quarantined` 或 `fatal`。
2. [alignment_stage_service.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py) 现行降级语义是：`quarantined/fatal -> safe_window_fallback`。
3. `safe_window_fallback` 直接把整窗 `chosen_text_clean` 压成唯一一条 `SentenceSegment` 与唯一一条 `SubtitleItem`。

这意味着，哪怕窗口内部已经存在部分可靠锚点与可用标点事实，一旦窗口级合法性失败，系统也会跳过局部提交，直接产出一条跨十几秒到二十秒的大字幕。

因此，本问题不是“切分阈值偏了”，而是两个结构性缺陷叠加：

1. `AnchorMount` 在重复短语窗口上缺少足够稳健的覆盖率恢复能力。
2. 窗口级降级策略过于粗暴，失败语义只有“整窗一条”。

## 2. 现状根因归纳

### 2.1 候选歧义在重复短语窗口中急剧放大

[seed_discovery.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/seed_discovery.py) 目前主要依赖 `exact / normalized / merge / sequence` 的 surface 匹配。  
[ambiguity_cluster_builder.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/ambiguity_cluster_builder.py) 会把同 surface、同 span 的候选聚为冲突簇。  
[anchor_trust_evaluator.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/anchor_trust_evaluator.py) 再按固定先验与有限冲突惩罚筛掉大量候选。

在 `Okay / trolley / baby / five elderly people` 这类高重复英文对话里，surface 本身信息量过低，导致：

1. 候选很多，但都长得像。
2. trust 评分缺少更强的位置与邻域一致性信号。
3. 结果是主链前只有少量局部唯一片段能活下来。

### 2.2 ChainSolver 以固定硬阈值裁图，导致大量局部可用块无法进入主链

[chain_solver.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/chain_solver.py) 当前使用固定约束：

- `_MONOTONIC_CORRIDOR = 3`
- `_MAX_HOOK_JUMP = 12`
- `_MAX_UNIT_JUMP = 12`
- `_DISTANCE_RATIO_LIMIT = 3.0`

这些规则在普通窗口上有效，但在重复对话和局部漏挂场景下过于刚性。结果不是“稍差但可修”，而是直接让主链断成零碎岛。

### 2.3 Gap rescue 只覆盖主锚岛之间的内部 gap

[gap_rescue_aligner.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/gap_rescue_aligner.py) 当前有两个先天限制：

1. 只处理两个已确认 anchor island 之间的内部 gap，不处理 head/tail gap。
2. 只接受“unit 侧唯一 + hook 侧唯一”的局部精确匹配。

这会导致窗口头尾的大量 unresolved token 无法被挽救，即使中间已经有一小段主链成功。

### 2.4 窗口失败语义过粗

[timeline_validity_validator.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/timeline_validity_validator.py) 与 [fallback_gate.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/fallback_gate.py) 目前会把窗口状态归到：

- `valid`
- `repairable`
- `quarantined`
- `fatal`

但是 [alignment_stage_service.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py) 最终只关心一件事：

- `quarantined/fatal` 是否需要 `safe_window_fallback`

也就是说，状态虽然不是布尔值，但提交层实际上仍然是“整窗继续”与“整窗一条”二选一。

### 2.5 输出总线并不是根因，也不是主要重构对象

[output_projector.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/output_projection/output_projector.py) 与 [commit_scope_resolver.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/commit_scope_resolver.py) 的核心关注点是：

1. `SubtitleBatch`
2. window coverage
3. replace scope

它们并不强制要求“一个窗口只能有一种 route”或“一个窗口只能有一条结果”。  
因此本次重构的最小闭环应当集中在：

`anchor_mount -> decision_ingress -> alignment_stage_service`

而不是推翻整个后半段输出体系。

## 3. 设计目标

本方案必须同时满足以下目标：

1. 彻底移除“窗口一旦挂坏就整窗一条”的默认降级语义。
2. 允许窗口内已证明可靠的部分正常提交，不被坏 span 拖死。
3. 对坏 span 进行局部 fallback，而不是用整窗 fallback 掩盖上游失败。
4. 提高重复短语窗口上的 anchor 覆盖率，降低 `quarantined/fatal` 占比。
5. 保持现有 Decision / OutputProjection / CommitScope 总线主体不被推翻。
6. 所有局部恢复路径必须有 provenance，可解释、可观测、可回归。

## 4. 非目标

本方案首轮不做以下事情：

1. 不重写整个 `timeanchored` 为全新全局序列对齐器。
2. 不追求所有极端差音频都完美挂载。
3. 不把 fast 流提升为 slow truth 的文本替代者。
4. 不同时重构输出层、前端编辑模型和 checkpoint 提交协议。

## 5. 方案选择

### 5.1 方案 A：仅替换 `safe_window_fallback`

做法：

1. 保留现有 `AnchorMount` 主链。
2. 仅把整窗一条改为“按标点拆成多条 fallback”。

优点：

1. 见效快。
2. 对输出层改动小。

缺点：

1. 仍然没有保住窗口内已有的可靠 anchor。
2. `quarantined/fatal` 窗口数量不会下降。
3. 只是止血，不是根治。

### 5.2 方案 B：局部提交 + span 级降级 + 覆盖率重构

做法：

1. 重构窗口失败语义，允许部分提交。
2. 为 bad span 设计局部 fallback，而不是整窗 fallback。
3. 同步补 `head/tail rescue`、重复短语消歧、solver 软化。

优点：

1. 同时切断当前超长字幕问题和上游覆盖率问题。
2. 保持后半段输出总线基本不变。
3. 范围仍可收敛在最小闭环。

缺点：

1. 设计复杂度显著高于方案 A。
2. 需要补一组新契约和回归集。

### 5.3 方案 C：彻底重写全局对齐器

做法：

1. 放弃现有 seed/chain/gap rescue。
2. 改为新的全局序列图或 DP 系统。

优点：

1. 理论上自由度最高。

缺点：

1. 范围过大。
2. 当前没有足够证据说明必须重写到这个级别。
3. 风险与回归面都明显超出本次问题闭环。

### 5.4 结论

本次选择 **方案 B**。  
原因不是“想顺手重构”，而是当前可见代码证据已经表明：只改 fallback 不足以根治，而全量重写又超出最小闭环。

## 6. 目标态架构总览

目标态主链不再是：

`AnchorMount -> timeline_validity -> safe_window_fallback(整窗一条)`

而是：

`AnchorMount -> WindowRecoveryPlanner -> trusted/interpolated/fallback spans -> Decision / SpanFallback -> merged SubtitleBatch -> OutputProjection`

核心原则：

1. 先保住可靠 span。
2. 再局部处理坏 span。
3. 只有在整窗完全无法构造任何可提交 span 时，才允许最终 emergency fallback。

## 7. 核心设计

### 7.1 B1：降级语义重构

#### 7.1.1 新的提交语义

[alignment_stage_service.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py) 不再把 `quarantined/fatal` 直接映射为 `safe_window_fallback`。  
改为三层处理：

1. `window_valid_commit`
   - 窗口整体可用，按现有主链正常提交。
2. `window_partial_commit`
   - 窗口存在坏 span，但能提取出可信 span 集，提交可信部分并局部修复坏 span。
3. `window_emergency_fallback`
   - 整窗无法构造任何可信 span，才允许最后兜底。

#### 7.1.2 引入 `WindowRecoveryPlanner`

新增一个窗口恢复规划器，职责是：

1. 读取 `AnchorMountStageResult`、`DecisionIngressPackage`、timeline validity、open gaps、boundary evidences。
2. 按 token 连续区间把窗口拆成：
   - `trusted_span`
   - `interpolated_span`
   - `fallback_span`
3. 为每个 span 生成后续执行计划。

判型依据至少包括：

1. `mount_status`
2. `anchor_kind`
3. `match_confidence`
4. `timeline_validity`
5. `validity_reasons`
6. `largest_unresolved_span`
7. `open_gaps / anchor_islands`

#### 7.1.3 Span 级执行策略

每个 span 的处理规则如下：

1. `trusted_span`
   - 直接走现有 `DecisionIngressAdapter -> Decision -> OutputProjection`
2. `interpolated_span`
   - 允许保留，但 trace 中要标明映射质量与恢复来源
3. `fallback_span`
   - 不允许整窗一条
   - 需要在 span 内用现有 `punctuation_facts + boundary_evidences + fallback_clean_text` 做最小切分
   - 时间轴使用相邻锚点、coverage 边界或局部包络做局部映射

#### 7.1.4 `safe_window_fallback` 降级为 emergency only

现有 `_commit_safe_window_fallback_result()` 需要保留，但角色变化：

1. 默认路径移除。
2. 仅在以下条件同时满足时启用：
   - 没有任何 `trusted_span`
   - 没有任何可构造的 `fallback_span` 结果
   - 文本仍非空

并建议重命名为 `emergency_window_fallback`，避免继续把它当成常规降级通道。

### 7.2 B2：Anchor 覆盖率重构

#### 7.2.1 候选发现增强

[seed_discovery.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/seed_discovery.py) 需要从“仅看 token surface”升级到“surface + 局部位置签名”：

1. 为候选增加局部前后文 key。
2. 引入相对位置先验，例如 `unit_index / total_units` 与 `hook_index / total_hooks` 的对角线接近度。
3. 对低信息量短词提高重复惩罚。

这不是换模型，而是在现有候选系统上补足 disambiguation 的先天缺口。

#### 7.2.2 冲突簇与可信度评分增强

[ambiguity_cluster_builder.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/ambiguity_cluster_builder.py) 和 [anchor_trust_evaluator.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/anchor_trust_evaluator.py) 需要新增：

1. 邻域一致性评分
2. 对角线位置一致性
3. 重复 surface 风险等级
4. head/tail 边界可信度

目标不是让所有候选都通过，而是让“虽然重复，但在当前邻域下最像正确位置”的候选能稳定活下来。

#### 7.2.3 `ChainSolver` 从固定硬阈值改为“硬边界 + 软惩罚”

[chain_solver.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/chain_solver.py) 的重构原则：

1. 继续保留硬约束：
   - 不允许交叉
   - 不允许跨 speaker/turn 硬边界
   - 不允许跨硬句界
2. 把固定 `jump/corridor/distance ratio` 从硬拒绝改成按窗口尺度归一的软惩罚。
3. 对 prefix/suffix 弱连接保留“次优但单调”的通道，而不是直接裁掉。

这样做的目标是：  
从“只保住极少数局部唯一片段”变成“尽量形成可解释的最长单调链”。

#### 7.2.4 `GapRescueAligner` 扩成三类 gap

[gap_rescue_aligner.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/gap_rescue_aligner.py) 需要从只处理 internal gap 改成：

1. `prefix_gap`
2. `internal_gap`
3. `suffix_gap`

同时把当前“必须唯一匹配”放宽为：

1. 局部最优
2. 与上下游锚点单调一致
3. provenance 可解释

也就是说，允许“不是数学上唯一，但在当前局部最合理”的救援匹配进入恢复层。

### 7.3 复用现有 Decision / Output 总线

本方案不应另起一套输出系统。现有可复用事实如下：

1. [decision_ingress_adapter.py](f:/video_to_srt_gpu/backend/app/services/textflow/decision_ingress_adapter.py) 已经能把 `DecisionIngressPackage` 适配到 Decision 层。
2. [output_projector.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/output_projection/output_projector.py) 只关心 `SubtitleBatch + coverage`。
3. [commit_scope_resolver.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/commit_scope_resolver.py) 也只关心 replace scope 和 affected ids。

因此，B 方案的正确方向是：

1. 在窗口内部先得到多个 span 结果。
2. 把 span 结果合成为一个 owner `SubtitleBatch`。
3. 继续复用现有投影与提交总线。

## 8. 契约与数据结构变更

### 8.1 扩展 `AnchoredTokenUnit`

[contracts.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/contracts.py) 中的 `AnchoredTokenUnit` 当前缺少稳定的文本切片坐标。  
为支持 span 级 recovery，需要补充：

1. `token_index`
2. `char_start`
3. `char_end`

这样 span fallback 才能稳定切出 slow truth 的子文本，而不是靠重新拼接猜测。

### 8.2 新增恢复规划契约

建议新增以下对象：

1. `WindowSpanDecision`
   - span 的 token 范围、char 范围、time 范围、quality、route
2. `SpanFallbackPlan`
   - fallback 文本来源、切分证据、时间映射策略
3. `RecoveredWindowPlan`
   - 一个窗口的完整恢复计划

建议字段至少包括：

1. `span_id`
2. `span_kind`
3. `token_start/token_end`
4. `char_start/char_end`
5. `time_start/time_end`
6. `route`
7. `quality`
8. `reason_codes`
9. `provenance`

### 8.3 支持按 span 切片构造 Decision ingress

[decision_ingress_assembler.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/decision_ingress_assembler.py) 与 [decision_ingress_adapter.py](f:/video_to_srt_gpu/backend/app/services/textflow/decision_ingress_adapter.py) 需要支持：

1. 基于 window 内 span 对 `anchored_token_units` 做子集切片
2. 同步裁剪关联的 `punctuation_facts / boundary_evidences / locks`
3. 保留 window 级 metadata，并增加 span 级 provenance

### 8.4 调试与观测字段

为确保这类问题以后可回归，需要增加以下指标：

1. `trusted_span_count`
2. `fallback_span_count`
3. `emergency_fallback_used`
4. `prefix_rescue_count`
5. `suffix_rescue_count`
6. `duplicate_surface_cluster_count`
7. `diagonal_penalty_avg`
8. `span_level_sentence_count`

## 9. 与旧 `timeanchored stage_service` 语义对齐

仓库里已有 [stage_service.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/stage_service.py)，其中有：

1. `failed_spans`
2. `final_stream`
3. `base_result + fallback_result`

这说明“部分失败 + 局部恢复”的语义在仓库中并非全无先例。  
本次设计不要求直接复活旧实现，但建议复用它的核心思想：

1. 失败不是整窗失败，而是 span 集失败。
2. 最终产出是合并后的 stream，而不是简单的单一路由。

## 10. 迁移顺序

### 10.1 Phase 1：切断整窗一条

目标：

1. 引入 `WindowRecoveryPlanner`
2. 实现 `trusted/fallback span` 最小闭环
3. 把 `safe_window_fallback` 降为 emergency only

完成后应立即消除“整窗一条”的主故障形态。

### 10.2 Phase 2：补 head/tail rescue

目标：

1. 扩展 `GapRescueAligner`
2. 新增 prefix/suffix rescue
3. 打通 span fallback 的局部时间映射

完成后应明显降低失败窗口数量。

### 10.3 Phase 3：做重复短语消歧与 solver 软化

目标：

1. 增强候选发现与 trust 评分
2. 把 solver 固定硬阈值改为硬边界 + 软惩罚
3. 补充统计与回归对比

完成后应让 `repairable` 和 `valid` 窗口占比显著上升。

## 11. 测试策略

### 11.1 回归样例

至少覆盖以下窗口：

1. 本次失败任务中的典型 `fatal` 窗口
2. 本次失败任务中的典型 `quarantined` 窗口
3. 本次任务中的成功窗口 `sw-000018`
4. 一个单 chunk 正常窗口
5. 一个多 chunk 正常窗口

### 11.2 必测断言

1. 不再出现 `safe_window_fallback` 产出的十几秒单条字幕。
2. `quarantined/fatal` 窗口允许输出多条局部字幕。
3. 成功窗口的现有切分不回退。
4. `OutputProjection` 的 replace scope 不受破坏。
5. trace 中能区分每条字幕来自：
   - 主链 trusted span
   - 插值 span
   - fallback span
   - emergency fallback

### 11.3 指标验收

在本次样例任务上，至少要求：

1. 超长单条字幕数降为 0。
2. `emergency_window_fallback` 触发次数接近 0，且明显少于现有 `safe_window_fallback` 次数。
3. `trusted + fallback span` 组合能覆盖原失败窗口的大部分文本。

## 12. 风险与控制

### 12.1 风险：局部 span 合并后顺序错误

控制：

1. span 输出前后都按时间排序
2. 合并阶段做单调校验
3. 出现重叠时优先保留 provenance 更强的 span

### 12.2 风险：局部 fallback 过度切分

控制：

1. fallback span 仍复用现有标点事实与 boundary evidences
2. 不单纯按字符长度切
3. 保留最小时长与最短文本门限

### 12.3 风险：soft solver 导致误挂

控制：

1. 只把距离阈值变成软惩罚，不放松硬边界
2. 所有 soft 选边都记录 provenance
3. 新增针对重复短语窗口的定向回归集

## 13. 最终结论

本问题的根因不是一个参数，也不是一个写出层 bug，而是：

1. `AnchorMount` 在重复短语窗口上覆盖率恢复能力不足；
2. `quarantined/fatal -> safe_window_fallback(整窗一条)` 的降级设计过于粗暴。

因此，彻底修复必须同时做两件事：

1. 把窗口级失败语义改造成“部分提交 + span 级降级”；
2. 把 `AnchorMount` 从“局部唯一片段求生”升级到“可解释的局部最优恢复”。

这就是本次选择方案 B 的原因，也是当前最小闭环下的根治路径。
