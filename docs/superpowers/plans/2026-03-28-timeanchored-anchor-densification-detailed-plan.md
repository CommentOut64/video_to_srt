# Timeanchored Anchor Densification Detailed Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 timeanchored 前半链重构为一条统一、可解释、可熔断的核心算法主线：以高可信骨架锚为起点，用稀疏 DP 选主锚链，在 gap 中用局部 DP/NW 做有限轮次增密，仅对小残差 gap 插值，并用统一状态对象和原子提交闭环避免重复字幕、时间戳混乱和 speaker 漏切。

**Architecture:** 采用“删除旧的 fail-open 逻辑 -> 提炼统一 `WindowAlignmentState` -> 让 `AnchorMountCorePipeline` 成为唯一核心算法主线 -> 由 `AlignmentStageService` 仅做桥接/路由/提交”的结构。允许局部重构，但目标不是堆补丁，而是把现有 `seed / block / chain / gap / fuse / commit` 重组为清晰的一体化流水线，并最大限度复用已有 `NW core / punctuation / output projection / decision ingress` 代码。

**Tech Stack:** Python 3.10+、`backend/app/services/timeanchored_alignment/anchor_mount/*`、`backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`、`backend/app/services/alignment/nw_v2_core.py`、pytest、Windows PowerShell、timeanchored trace JSON

---

## 0. 文档定位

本文件是 **详细开发文档**，用于承接上一份概念性计划：

- 参考但不覆盖：[2026-03-28-timeanchored-anchor-densification-mainline.md](f:/video_to_srt_gpu/docs/superpowers/plans/2026-03-28-timeanchored-anchor-densification-mainline.md)

两份文档关系：

1. 上一份文档负责“我们为什么这样做、总体要做什么”
2. 本文负责“具体怎么重构、每个模块怎么落、哪些旧实现先删、核心代码长什么样”

## 1. 实施原则

### 1.1 一体化优先，不堆补丁

新方案必须体现为一条完整主线，而不是把新逻辑散落到：

- `SeedDiscovery`
- `ChainSolver`
- `TemporalEnvelopeBuilder`
- `AlignmentStageService`
- `HookClaimResolver`

这几个位置分别偷偷“补一刀”。

首版原则：

1. 核心算法只能有一个明确 orchestrator
2. 所有阶段共享同一个状态对象
3. 只有 gap rescue 允许有限轮循环
4. 桥接层不能反向篡改算法决策

### 1.2 先删旧实现，再接新主线

实现新方案前，必须先删除或禁用以下旧实现形态，否则新旧语义会互相污染：

1. `SeedDiscovery` 的无界远距离 merge 候选
2. `ChainSolver` 的贪心 suffix reseed 主逻辑
3. `TemporalEnvelopeBuilder` 对大 gap 的默认 inferred/unresolved 补时路径
4. `AlignmentStageService` 中把 `should_fallback=true` 映射为 owner-only fast fallback 的旧语义
5. speaker timeline 注入仍读取 `decision_ingress.tokens` 的旧路径

### 1.3 在重构前提下尽量复用现有代码

首版优先复用：

1. `NeedlemanWunschV2Core`
   - 文件：[nw_v2_core.py](f:/video_to_srt_gpu/backend/app/services/alignment/nw_v2_core.py)
   - 用于 gap 内局部 DP/NW
2. `PunctuationFactMapper`
   - 文件：[punctuation_fact_mapper.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/punctuation_fact_mapper.py)
3. `BoundaryEvidenceBuilder`
4. `CrossChunkLockBuilder`
5. `DecisionIngressAssembler`
6. `DecisionIngressAdapter`
7. `OutputProjector`

不复用但会被替换/重写的核心：

1. `SeedDiscovery` 的候选生成策略
2. `ChainSolver` 贪心选链策略
3. `TemporalEnvelopeBuilder` 大 gap 处理策略
4. `FallbackGate` 单布尔语义

## 2. 目标架构调整

### 2.1 重构后的模块边界

重构后建议形成如下结构：

```text
AlignmentPreparationPackage
        │
        ▼
AnchorMountAlignmentService
        │  只做 stage 级包装与依赖注入
        ▼
AnchorMountCorePipeline
        │
        ├─ SeedDiscovery
        ├─ AmbiguityClusterBuilder
        ├─ AnchorTrustEvaluator
        ├─ SparseChainSolver
        ├─ GapRescueAligner
        ├─ DensificationPromoter
        ├─ TemporalEnvelopeBuilder
        ├─ HookClaimResolver
        ├─ BoundaryEvidenceBuilder
        └─ TimelineValidityValidator
        │
        ▼
DecisionIngressAssembler
        │
        ▼
AlignmentStageService
        │  只做 speaker bridge / route / atomic commit
        ▼
Decision / OutputProjection / Dispatch
```

### 2.2 统一状态对象

核心算法只传递一个状态对象：

- `WindowAlignmentState`

这个对象是所有阶段的唯一共享真相，至少应包含：

```python
@dataclass(frozen=True)
class WindowAlignmentState:
    input_view: AnchorMountInputView
    raw_seed_candidates: tuple[AnchorCandidate, ...]
    ambiguity_clusters: tuple["AmbiguityCluster", ...]
    primary_candidates: tuple[AnchorCandidate, ...]
    local_blocks: tuple[LocalAlignmentBlock, ...]
    main_chain: "SparseChainResult | None"
    anchor_islands: tuple["AnchorIsland", ...]
    open_gaps: tuple["AnchorGap", ...]
    promoted_secondary_anchors: tuple[AnchorCandidate, ...]
    rescue_round_index: int
    fuse_events: tuple["FuseEvent", ...]
    timeline_validity: str
    diagnostics: dict[str, Any]
```

设计要求：

1. 各模块接收旧 state，返回新 state
2. 不允许模块偷偷改写外部字段
3. 算法 trace 直接从 state 派生

### 2.3 核心 orchestrator

建议新增：

- `backend/app/services/timeanchored_alignment/anchor_mount/core_pipeline.py`

职责：

1. 执行唯一主线
2. 控制 rescue 轮次
3. 管理 fuse 升级
4. 产出 `AnchorMountResult`

核心骨架：

```python
class AnchorMountCorePipeline:
    def run(self, input_view: AnchorMountInputView) -> WindowAlignmentState:
        state = WindowAlignmentState.bootstrap(input_view)

        state = self._seed_discovery.run(state)
        state = self._ambiguity_cluster_builder.run(state)
        state = self._anchor_trust_evaluator.run(state)
        state = self._sparse_chain_solver.run(state)

        while self._should_run_gap_rescue(state):
            state = self._gap_rescue_aligner.run(state)
            state = self._densification_promoter.run(state)
            state = self._sparse_chain_solver.run_with_secondary_anchors(state)
            if self._should_fuse(state):
                state = self._fuse(state)
                break

        state = self._temporal_envelope_builder.run(state)
        state = self._hook_claim_resolver.run(state)
        state = self._boundary_evidence_builder.run(state)
        state = self._timeline_validity_validator.run(state)
        return state
```

这个类是“一体化架构”的核心。  
`service.py` 以后不再自己串十几个步骤，而是只调用 `core_pipeline.run()`。

## 3. Delete-First 清理清单

### 3.1 必删旧逻辑

#### A. `SeedDiscovery` 里的无界 merge 候选

当前问题点：

- [seed_discovery.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/seed_discovery.py)
- 当前存在 `for next_unit_index in range(unit_index + 1, len(input_view.token_units))`

要删除的语义：

1. 当前 token 与任意后续 token 的远距离拼接
2. 无上界 merge
3. 仅凭拼接成功就形成 hard merge 候选

删除后替换为：

1. 只允许相邻 token merge
2. merge 长度有硬上限
3. merge 必须经过 ambiguity + trust gate

#### B. 贪心 `ChainSolver`

当前问题点：

- [chain_solver.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/chain_solver.py)

要删除的语义：

1. `_find_conflict_start()` + suffix score 的贪心 reseed
2. 只检查局部冲突而不检查全局最优

替换为：

1. block graph sparse DP
2. 全局单调链优化

#### C. 大 gap 默认插值

当前问题点：

- [temporal_envelope_builder.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/temporal_envelope_builder.py)

要删除的语义：

1. 只要有前后 resolved token，就直接 inferred
2. 即使 token 过多 / span 过小也继续塞时间

替换为：

1. gap rescue 优先
2. 仅对极小残余 gap 插值
3. 大 gap rescue 失败时显式 fuse / invalid

#### D. `should_fallback` 的旧路由语义

当前问题点：

- [alignment_stage_service.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py)

要删除的语义：

1. `should_fallback=true` 直接被视为 fast fallback 触发原因
2. 特别是 owner-only fast fallback

替换为：

1. `timeline_validity`
2. `gap_fuse/window_fuse`
3. safe fallback route

#### E. 旧 speaker 注入路径

当前问题点：

- [alignment_stage_service.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py)

要删除的语义：

1. 读取 `decision_ingress.tokens`
2. 把 speaker 真值作为后补兼容注入

替换为：

1. 基于 `anchored_token_units` 的窗口范围
2. 正式 speaker truth injection

### 3.2 删除顺序

建议按顺序清理：

1. 删 `SeedDiscovery` 无界 merge
2. 删贪心 `ChainSolver`
3. 删大 gap 默认插值
4. 删旧 fallback 路由
5. 删旧 speaker 注入

原因是：前面三项是核心算法清理，后面两项是外层桥接清理。

## 4. File-Level 实施蓝图

### 4.1 新增文件

#### `anchor_mount/core_pipeline.py`

职责：

1. 统一 orchestration
2. 限制 rescue 循环
3. 汇总 fuse / validity

#### `anchor_mount/window_alignment_state.py`

职责：

1. 定义唯一核心状态对象
2. 避免算法状态散落在 service / result / metrics 中

#### `anchor_mount/ambiguity_cluster_builder.py`

职责：

1. 将同 hook 多候选收束为 `ambiguity cluster`
2. 给后续 disambiguation 提供稳定输入

核心数据结构：

```python
@dataclass(frozen=True)
class AmbiguityCluster:
    cluster_id: str
    hook_span: tuple[int, ...]
    unit_span: tuple[int, ...]
    candidate_ids: tuple[str, ...]
    candidate_unit_spans: tuple[tuple[int, ...], ...]
    candidate_hook_spans: tuple[tuple[int, ...], ...]
    cluster_tags: tuple[str, ...]
    conflict_basis: tuple[str, ...]
```

设计约束：

1. `AmbiguityCluster` 不是“文本类别”的标签桶，而是“不能同时成为 primary anchor 的候选冲突图”
2. 是否进入同一 cluster，取决于候选间是否存在互斥关系，而不是只看字面是否重复
3. `cluster_tags` 只做诊断，不承担主判定逻辑
4. `conflict_basis` 必须显式记录冲突来源，至少包括：
   - `same_hook_span`
   - `same_unit_span`
   - `merge_width_competition`
   - `duplicate_surface`
   - `boundary_cross_risk`
   - `low_information_risk`

首版不再使用 `repeated_token / repeated_phrase / merge_competition` 这种单标签主分类，因为它既不稳定，也不足以支撑后续 trust/disambiguation 决策。

#### `anchor_mount/anchor_trust_evaluator.py`

职责：

1. 计算 trust tier
2. 给出 reject reason
3. 与 `disambiguation score` 分工

核心约束：

1. `disambiguation score` 只回答“同 cluster 内谁更可能正确”
2. `trust score` 回答“该候选有没有资格进入主线”
3. 两者不能混成一个分数，否则后续会重新长回黑箱

建议输出形态：

```python
@dataclass(frozen=True)
class AnchorTrustReport:
    candidate_id: str
    trust_score: float  # [0, 1]
    trust_tier: str     # primary / secondary / boundary_only / rejected
    reject_reason: str | None
    diagnostics: dict[str, float | str]
```

建议伪代码：

```python
def evaluate_candidate(candidate: AnchorCandidate, context: CandidateContext) -> AnchorTrustReport:
    reject_reason = hard_reject(candidate, context)
    if reject_reason is not None:
        return AnchorTrustReport(
            candidate_id=candidate.candidate_id,
            trust_score=0.0,
            trust_tier="rejected",
            reject_reason=reject_reason,
            diagnostics={"hard_reject": reject_reason},
        )

    score = 0.0
    score += 0.30 * kind_prior(candidate.anchor_kind)
    score += 0.20 * uniqueness_margin(candidate, context)
    score += 0.15 * context_support(candidate, context)
    score += 0.10 * information_value(candidate, context)
    score += 0.10 * boundary_consistency(candidate, context)
    score += 0.10 * provenance_consistency(candidate, context)
    score += 0.05 * temporal_plausibility(candidate, context)
    score -= ambiguity_penalty(candidate, context)
    score -= low_information_penalty(candidate, context)

    if score >= 0.80 and disambiguation_margin(candidate, context) >= 0.15:
        tier = "primary"
    elif score >= 0.62:
        tier = "secondary"
    elif score >= 0.45:
        tier = "boundary_only"
    else:
        tier = "rejected"

    return AnchorTrustReport(
        candidate_id=candidate.candidate_id,
        trust_score=max(0.0, min(1.0, score)),
        trust_tier=tier,
        reject_reason=None if tier != "rejected" else "trust_below_threshold",
        diagnostics={...},
    )
```

#### `anchor_mount/gap_rescue_aligner.py`

职责：

1. 在 island 之间做局部 DP/NW
2. 尽量复用 `NeedlemanWunschV2Core`

#### `anchor_mount/densification_promoter.py`

职责：

1. 将 rescue 成功的局部匹配点升级为 `secondary anchors`
2. 防止 promoted anchors 越权修改 primary anchors

#### `anchor_mount/timeline_validity_validator.py`

职责：

1. 统一做 monotonic / compression / residual gap 校验
2. 输出 `valid / repairable / quarantined / fatal`

### 4.2 重构/修改文件

#### `anchor_mount/service.py`

重构目标：

1. 从“算法主逻辑容器”降级为“stage wrapper”
2. 只负责：
   - 依赖构造
   - 调 `core_pipeline`
   - 从 state 收口成 `AnchorMountResult`

建议改造后形态：

```python
class AnchorMountAlignmentService:
    def __init__(...):
        self._core_pipeline = AnchorMountCorePipeline(...)

    def align(...):
        input_view = self._ingress_validator.validate(...)
        state = self._core_pipeline.run(input_view)
        result = self._result_builder.build(state)
        decision_ingress = self._decision_ingress_assembler.build(...)
        return AnchorMountStageResult(...)
```

#### `anchor_mount/seed_discovery.py`

重构目标：

1. 只生成 bounded raw candidates
2. 不直接决定 hard anchor

核心代码片段：

```python
MAX_TOKEN_MERGE_WIDTH = 2
MAX_HOOK_SEQUENCE_WIDTH = 3

def _build_bounded_merge_candidates(...):
    for start in range(len(token_units)):
        for width in range(2, MAX_TOKEN_MERGE_WIDTH + 1):
            span = token_units[start:start + width]
            if len(span) != width:
                break
            if not _is_contiguous(span):
                continue
            yield ...
```

#### `anchor_mount/chain_solver.py`

重构目标：

1. 改成 sparse DP solver
2. 删除贪心 `_find_conflict_start`

设计说明：

1. 新 solver 的首要收益是“全局最优单调链”，不是单纯降复杂度
2. 当前贪心实现最坏同样接近 `O(B^2)`；新 solver 不能只把双层循环换个名字
3. 首版不直接上 segment tree，而是先把问题改写为“稀疏兼容图上的 DAG longest-path”
4. 因此实际复杂度表述为 `O(B + E)`，其中 `E` 是通过兼容性筛选后保留下来的边数，而不是默认全连接 `B^2`

兼容边必须先经过以下 gate，再进入 DP：

1. strict monotonic：`prev.unit_end < curr.unit_start` 且 `prev.hook_end < curr.hook_start`
2. diagonal corridor：块间偏移不允许超过 corridor
3. boundary legality：不跨 hard speaker / punctuation boundary
4. distance sanity：token/hook 跨度比例不能明显失衡

核心代码片段：

```python
def solve(self, blocks: tuple[LocalAlignmentBlock, ...]) -> SparseChainResult:
    ordered = sorted(blocks, key=_sort_key)
    edges = build_sparse_compatibility_edges(
        ordered,
        monotonic_corridor=self._corridor,
        max_hook_jump=self._max_hook_jump,
        max_unit_jump=self._max_unit_jump,
    )
    best_score = [float("-inf")] * len(ordered)
    parent = [-1] * len(ordered)

    for i, block in enumerate(ordered):
        best_score[i] = block.score
        for j in incoming_neighbors(edges, i):
            score = best_score[j] + transition_bonus(ordered[j], block) + block.score
            if score > best_score[i]:
                best_score[i] = score
                parent[i] = j

    return _reconstruct_best_chain(ordered, best_score, parent)
```

实现注意：

1. 不接受“只检查最近 K 个 blocks”的硬裁剪，因为它会误杀长 gap 后的正确前驱
2. 首版先通过 bounded seed、ambiguity 收束与兼容边稀疏化把 `E` 压小
3. 只有在 trace 证明 `B` 和 `E` 仍然过大时，第二阶段才考虑索引优化

#### `anchor_mount/temporal_envelope_builder.py`

重构目标：

1. 只处理：
   - anchored / merged islands
   - 小残差 gap 插值
2. 大 gap 处理完全前移到 rescue

核心代码片段：

```python
if gap.token_count > MAX_INTERPOLATION_RESIDUAL:
    raise LargeResidualGapError(...)

if not _is_interpolation_eligible(left_anchor, right_anchor, gap):
    raise IneligibleInterpolationError(...)
```

#### `alignment_stage_service.py`

重构目标：

1. `timeline_validity` 决定路由
2. speaker truth 进入正式桥接
3. 提交改为 span-set atomic commit

## 5. 重复词 / 同 hook 多候选的具体实现

### 5.1 两阶段消歧

建议明确分两阶段：

1. **主链前消歧**
   - 由 `ambiguity cluster + trust + disambiguation score` 决定能否形成 `primary_anchor`
2. **gap 内消歧**
   - 如果主链前不能唯一化，则保留为 `secondary_candidate`
   - 只在 gap rescue 后结合新上下文再尝试晋升

补充约束：

1. 一个 hook span 在同一轮里最多只能对应一个 `primary_anchor`
2. 同一 `unit_span` 也不能同时被多个 `primary_anchor` 占用
3. gap rescue 后即使唯一化成功，也只允许升级为 `secondary_anchor`，不反写已确定的 primary 语义

### 5.2 `disambiguation score` 代码骨架

```python
def score_candidate(
    candidate: AnchorCandidate,
    *,
    cluster: AmbiguityCluster,
    left_anchor: AnchorCandidate | None,
    right_anchor: AnchorCandidate | None,
) -> float:
    score = 0.0
    score += surface_match_score(candidate)
    score += left_context_support(candidate, left_anchor)
    score += right_context_support(candidate, right_anchor)
    score -= diagonal_distance_penalty(candidate, cluster)
    score -= boundary_cross_penalty(candidate)
    score -= low_information_penalty(candidate)
    return score
```

### 5.3 Hard Reject 代码骨架

```python
def hard_reject(candidate: AnchorCandidate, context: CandidateContext) -> str | None:
    if not is_contiguous_merge(candidate):
        return "non_contiguous_merge"
    if breaks_monotonicity(candidate, context):
        return "monotonic_violation"
    if crosses_hard_boundary(candidate, context):
        return "hard_boundary_cross"
    if lacks_uniqueness_margin(candidate, context):
        return "ambiguous_duplicate"
    if relies_on_low_information_only(candidate, context):
        return "low_information_duplicate"
    return None
```

### 5.4 关键实现约束

1. `HookClaimResolver` 不再承担重复词消歧职责
2. 消歧结果必须在 `SparseChainSolver` 前就显式落到 candidate tier
3. 如果 gap rescue 后仍不能唯一化，不允许后置偷偷硬贴
4. `AmbiguityCluster` 必须保留 trace：候选数、冲突原因、被拒原因、最终晋升结果

## 6. 稀疏 DP 主锚链的具体实现

### 6.1 为什么不用全窗大 DP

原因：

1. 大 DP 成本高
2. 会把 primary anchors 和 gap rescue 混在一起
3. 不利于保持主锚链语义稳定

### 6.2 为什么 block-level sparse DP 合适

因为它同时满足：

1. 全局最优优于贪心
2. 输入稀疏、可解释
3. 自然支持重复词消歧后的 candidate tier

### 6.2.1 复杂度与性能口径

必须明确：

1. 新方案不是为了把理论复杂度从 `O(B^2)` 神奇降到 `O(B log B)`
2. 真正收益来自：
   - 主链不再局部贪心
   - 候选集合变小
   - 兼容边稀疏化
   - trace 可观测
3. 需要新增以下指标，避免性能争论只停留在口头：
   - `block_count`
   - `compatibility_edge_count`
   - `solver_elapsed_ms`
   - `avg_in_degree`
   - `max_in_degree`

首版性能策略：

1. 先做 sparse compatibility graph
2. 再通过 trace 验证是否真出现高热点
3. 如果仍有 `100+ blocks / 高 edge density` 的窗口，再进入第二阶段优化

### 6.2.2 第二阶段可选优化（不纳入首版强依赖）

只有在 trace 证明确有必要时，才考虑：

1. 对兼容前驱做索引化查询
2. 对 block 按 hook_end 建 segment tree / Fenwick-like 索引
3. 对极大窗口引入更窄的动态 corridor

这些优化不是首版前置条件，避免在正确性尚未稳定前过早引入复杂状态。

### 6.3 transition 设计

建议 transition score 至少考虑：

1. 与前块的 hook 距离是否合理
2. 与前块的 token 距离是否合理
3. 是否跨越 speaker / punctuation hard boundary
4. 是否维持近似对角趋势

## 7. Gap Rescue 与 Bounded Densification 的具体实现

### 7.1 gap rescue 复用 NW core

建议直接复用：

- [nw_v2_core.py](f:/video_to_srt_gpu/backend/app/services/alignment/nw_v2_core.py)

封装一个局部适配层，而不是另写一套 DP 引擎。

局部适配器形态：

```python
class GapRescueAligner:
    def align_gap(self, gap: AnchorGap) -> GapRescueResult:
        seq1 = [token.normalized_text for token in gap.token_units]
        seq2 = [hook.normalized_text for hook in gap.fast_hooks]
        path = self._nw_core.align(seq1, seq2, match_fn=self._match_fn)
        return self._to_rescue_result(path, gap)
```

### 7.2 rescue 成功的晋升规则

只有同时满足以下条件的点才能晋升：

1. 匹配强度高
2. 在 local path 中唯一
3. 不破坏全局主锚链单调性
4. 不跨 hard boundary
5. rescue 后确实能把 gap 切小

### 7.3 bounded densification 的停止条件

```python
def should_continue(state: WindowAlignmentState) -> bool:
    if state.rescue_round_index >= MAX_RESCUE_ROUNDS:
        return False
    if state.last_round_report.promoted_anchor_count <= 0:
        return False
    if state.last_round_report.promoted_anchor_mean_trust < MIN_PROMOTED_TRUST:
        return False
    if state.last_round_report.gap_shrink_ratio < MIN_GAP_SHRINK_RATIO:
        return False
    if state.elapsed_ms >= state.repair_budget_ms:
        return False
    if max_gap_token_count(state.open_gaps) <= SMALL_RESIDUAL_GAP:
        return False
    return True
```

设计说明：

1. 首版采用“动态 early-stop + 小硬上限”
2. `MAX_RESCUE_ROUNDS` 建议固定为 `2`，对应“第 0 轮主链 + 最多 2 次 rescue pass”
3. 不建议把 `3~4` 轮作为默认，因为第 3 轮之后新增锚点越来越依赖 promoted anchors，自激漂移风险显著上升
4. 质量信号必须至少覆盖：
   - promoted anchor 数量
   - promoted anchor 平均 trust
   - 最大 gap 缩减比例
   - rescue 总耗时

## 8. 层级熔断的具体实现

### 8.1 fuse 不是 fallback

要明确：

1. `fuse` 是停止继续美化坏结果
2. `fallback` 是进入安全替代链

两者不能混在一个布尔值里。

### 8.2 fuse 状态对象

```python
@dataclass(frozen=True)
class FuseEvent:
    level: str  # gap / local_reseed / window
    reason: str
    gap_id: str | None
    round_index: int
    diagnostics: dict[str, float | str | int]
```

### 8.2.1 fuse 的语义边界

必须明确区分：

1. `gap_fuse`
   - 放弃继续修某个 gap
   - 该 gap 禁止再插值
   - 窗口其他合法部分仍可继续
2. `window_fuse`
   - 停止整个窗口的 timeanchored 修复主线
   - 后续只允许进入 validator 与 safe fallback route

`fuse` 是过程事件；`timeline_validity` 是最终判定。  
`window_fuse` 往往会导致 `quarantined/fatal`，但二者不是同一个字段。

### 8.2.2 `_should_fuse()` 的判断逻辑

建议至少满足以下之一时触发：

1. 连续 `gap_rescue` 失败次数超过阈值
2. 本轮 `promoted_anchor_mean_trust` 低于阈值
3. 本轮 `gap_shrink_ratio` 低于阈值
4. 出现全局 monotonic violation
5. trusted coverage 低于最低线
6. rescue 总耗时超预算

伪代码：

```python
def should_fuse(state: WindowAlignmentState) -> bool:
    report = state.last_round_report
    if state.global_monotonic_violation_count > 0:
        return True
    if report.consecutive_failed_gaps >= MAX_FAILED_GAPS:
        return True
    if report.promoted_anchor_count == 0 and report.pending_large_gap_count > 0:
        return True
    if report.promoted_anchor_mean_trust < MIN_PROMOTED_TRUST:
        return True
    if report.gap_shrink_ratio < MIN_GAP_SHRINK_RATIO and report.pending_large_gap_count > 0:
        return True
    if state.elapsed_ms >= state.repair_budget_ms:
        return True
    return False
```

### 8.3 local rollback / reseed 规则

只允许回滚：

1. gap 邻近、trust 不高的 1~2 个 anchor

绝不允许回滚：

1. 高 trust primary anchors
2. 已作为 island 核心骨架的锚点
3. 跨 speaker hard boundary 的锚点

代码骨架：

```python
def select_rollback_targets(state: WindowAlignmentState, gap: AnchorGap) -> tuple[str, ...]:
    candidates = nearby_low_trust_anchors(state, gap)
    return tuple(anchor.anchor_id for anchor in candidates[:2])
```

## 9. 合法性门禁的具体实现

### 9.1 `TimelineValidityValidator`

建议最少输出：

```python
@dataclass(frozen=True)
class TimelineValidityReport:
    state: str  # valid / repairable / quarantined / fatal
    monotonic_violation_count: int
    compressed_run_count: int
    residual_gap_token_count: int
    invalid_gap_ids: tuple[str, ...]
    reasons: tuple[str, ...]
```

### 9.2 与 stage 路由对接

`AlignmentStageService` 不再判断：

- `should_fallback == True`

而判断：

1. `timeline_validity == "valid"`
   - 正常提交
2. `timeline_validity in {"repairable", "quarantined"}`
   - 根据 span-set 与 safe fallback 规则处理
3. `timeline_validity == "fatal"`
   - 禁止正常 timeanchored commit

### 9.3 `timeline_validity` 与 safe fallback 的关系

首版路由表必须写死，避免重新回到 `should_fallback -> fast_direct`：

1. `valid`
   - 正常提交 timeanchored 输出
2. `repairable`
   - 仍停留在主线内继续 repair，不外跳
3. `quarantined`
   - 禁止正常 timeanchored commit
   - 若存在 `safe_window_fallback`，则以同一 coverage 做整窗保守降级
   - 若不存在，则显式失败，不做 owner-only 替换
4. `fatal`
   - 不提交 timeanchored 结果
   - 尝试 `safe_window_fallback`
   - 再失败则显式报错

首版不引入“部分 valid islands 正常提交、剩余 gap 留空”的字幕版本管理。  
原因是 commit scope 与 segment split 复杂度过高，容易在收尾层再次引入新不一致。  
因此首版原则是：局部修复尽量在主线内部完成；一旦升级到 `quarantined/fatal`，就按整窗 coverage 做安全兜底或失败闭环。

## 10. speaker bridge 与原子提交

### 10.1 speaker bridge

必须改为从：

- `anchored_token_units`

计算窗口时间范围，然后用 timeline overlap 注入正式 `speaker_turns`。

具体规则：

1. 优先使用 `anchored_token_units` 的时间并集
2. 若 `anchored_token_units` 为空，则退到 `coverage.core_segments`
3. 若 `coverage.core_segments` 也为空，则退到 `source_chunk_ids` 对应的已知时间窗
4. 若三者都不可用，只记录 diagnostics，不强注入 speaker turn
5. 窗口跨多个 turn 时，注入所有重叠 turn，不是只选一个 dominant speaker
6. token 级 speaker 回填必须满足 overlap/midpoint/confidence 门槛；否则只作为窗口级 evidence，不硬写 token speaker

伪代码：

```python
def resolve_window_time_span(package: DecisionIngressPackage, timeline_turns: Sequence[Turn]) -> TimeSpan | None:
    spans = collect_token_spans(package.anchored_token_units)
    if not spans:
        spans = collect_coverage_spans(package.coverage)
    if not spans:
        spans = lookup_chunk_spans(package.source_chunk_ids)
    return merge_spans(spans) if spans else None

def inject_speaker_turns(package: DecisionIngressPackage, timeline_turns: Sequence[Turn]) -> list[TurnProjection]:
    window_span = resolve_window_time_span(package, timeline_turns)
    if window_span is None:
        record_diagnostic("speaker_window_span_missing")
        return []

    selected = []
    for turn in timeline_turns:
        overlap = overlap_ratio(window_span, turn.span)
        if overlap <= 0.0:
            continue
        selected.append(project_turn(turn, overlap=overlap))
    return selected
```

### 10.2 span-set atomic commit

必须从“owner chunk commit”改成：

```python
@dataclass(frozen=True)
class CommitScope:
    window_id: str
    source_chunk_ids: tuple[str, ...]
    source_chunk_indices: tuple[int, ...]
    coverage_segments: tuple[tuple[float, float], ...]
    affected_segment_ids: tuple[str, ...]
    timeline_validity: str
    generation_id: str
```

提交时：

1. 先算 scope
2. 再 replace scope 覆盖集
3. 多 chunk invalid 时不允许 owner-only 替换

### 10.2.1 `CommitScopeResolver` 的首版边界

首版不引入完整字幕版本管理系统，但必须显式增加 `CommitScopeResolver`，职责如下：

1. 以 `coverage.core_segments + source_chunk_ids/source_chunk_indices` 作为 scope 主边界
2. 从既有字幕 segment provenance 中解析 `affected_segment_ids`
3. 若旧 segment 仅与 scope 部分重叠，则：
   - 若投影层支持 split，则先 split 再 replace
   - 若不支持 split，则把 scope 向 segment 边界扩张后整体 replace
4. 每次 commit 必须带 `generation_id`，避免旧窗口结果回写覆盖新窗口

伪代码：

```python
def build_commit_scope(window_result: WindowResult, existing_segments: Sequence[SubtitleSegment]) -> CommitScope:
    coverage_segments = normalize_coverage_segments(window_result.coverage)
    affected = []
    for segment in existing_segments:
        if overlaps_scope(segment, coverage_segments, window_result.source_chunk_ids):
            affected.append(segment.segment_id)
    return CommitScope(
        window_id=window_result.window_id,
        source_chunk_ids=window_result.source_chunk_ids,
        source_chunk_indices=window_result.source_chunk_indices,
        coverage_segments=coverage_segments,
        affected_segment_ids=tuple(affected),
        timeline_validity=window_result.timeline_validity,
        generation_id=window_result.generation_id,
    )
```

首版重点不是“极致精细替换”，而是先制度性杜绝：

1. owner-only fallback
2. 多 chunk window 半替换半残留
3. 旧窗口结果回写覆盖新窗口

### 10.3 Safe Fallback 首版口径

删除旧 `anchor_mount_should_fallback -> fast_direct` 后，必须保留安全兜底，但首版兜底必须满足：

1. coverage 级，而不是 owner chunk 级
2. 保留 slow truth 文本，不把快流文本重新升格为真相
3. fast 只提供保守时间基，不承担切分真值

首版兜底顺序：

1. `safe_window_fallback`
   - 对整窗 coverage 生成保守时间轴
   - 文本仍来自 slow truth / chosen text
2. `bounded_timebase_borrow`
   - 仅在 window 级保守回退需要时使用快流时间基
3. 显式失败
   - 如果 coverage 级安全兜底也无法构造，则不提交任何伪完整字幕

首版不再允许：

1. owner-only `fast_direct`
2. `should_fallback=True` 直接映射成快流直通
3. 多 chunk invalid 时只替换 owner chunk

## 11. 修改文件列表

### 11.1 Create

- `backend/app/services/timeanchored_alignment/anchor_mount/core_pipeline.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/window_alignment_state.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/ambiguity_cluster_builder.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/anchor_trust_evaluator.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/gap_rescue_aligner.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/densification_promoter.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/timeline_validity_validator.py`
- `backend/app/pipelines/dual_pipeline/services/commit_scope_resolver.py`
- `backend/tests/timeanchored_alignment/test_window_alignment_state.py`
- `backend/tests/timeanchored_alignment/test_anchor_disambiguation.py`
- `backend/tests/timeanchored_alignment/test_anchor_trust_evaluator.py`
- `backend/tests/timeanchored_alignment/test_gap_rescue_aligner.py`
- `backend/tests/timeanchored_alignment/test_densification_promoter.py`
- `backend/tests/timeanchored_alignment/test_timeline_validity_validator.py`
- `backend/tests/timeanchored_alignment/test_sparse_dp_chain_solver.py`
- `backend/tests/timeanchored_alignment/test_job_p20260328_193846_no_duplicate.py`
- `backend/tests/timeanchored_alignment/test_job_p20260328_202014_no_timestamp_chaos.py`
- `backend/tests/timeanchored_alignment/test_anchor_mount_performance_benchmark.py`

### 11.2 Modify

- `backend/app/services/timeanchored_alignment/anchor_mount/contracts.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/service.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/seed_discovery.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/low_complexity_mask.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/local_extension.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/chain_solver.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/temporal_envelope_builder.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/hook_claim_resolver.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/fallback_gate.py`
- `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- `backend/app/services/textflow/decision_ingress_adapter.py`
- `backend/app/services/timeanchored_alignment/output_projection/output_projector.py`
- `backend/tests/timeanchored_alignment/test_anchor_mount_seed_discovery.py`
- `backend/tests/timeanchored_alignment/test_anchor_mount_service.py`
- `backend/tests/timeanchored_alignment/test_anchor_mount_punctuation_and_envelopes.py`
- `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`
- `backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py`
- `backend/tests/timeanchored_alignment/test_alignment_stage_trace_payloads.py`
- `backend/tests/textflow/test_decision_layer_ingress_context.py`

### 11.3 Delete / Retire

首版不一定物理删文件，但必须删掉以下旧实现分支：

- `seed_discovery.py` 的远距离 merge 逻辑
- `chain_solver.py` 的旧贪心 `_find_conflict_start` 主逻辑
- `temporal_envelope_builder.py` 的大 gap 默认 inferred 路径
- `alignment_stage_service.py` 中 owner-only `anchor_mount_should_fallback -> fast_direct`
- `alignment_stage_service.py` 中旧 speaker 注入路径

## 12. 分阶段实施建议

先明确：这不是四个可独立上线的小 feature，而是一次**中等规模的一体化重构**。  
下面的 `Phase A/B/C/D` 是执行切片与 review checkpoint，不是“每个 phase 完成即可上线”的承诺。

### Phase A: Delete-First + 契约重建

1. 删除旧分支
2. 新建 `WindowAlignmentState`
3. 建立 ambiguity / trust / validity 契约

### Phase B: Core Pipeline 成形

1. 引入 `core_pipeline.py`
2. 接上受限 seed
3. 接上 sparse DP

### Phase C: Gap Rescue 主线

1. 接入 `GapRescueAligner`
2. 接入 `DensificationPromoter`
3. 收口插值资格

### Phase D: 外层闭环

1. speaker bridge
2. atomic commit
3. safe fallback

建议排期口径：

1. 以 2~3 周的完整开发窗口来估算
2. 每个 phase 完成后都要做阶段回归，但默认仍在同一特性分支内持续推进
3. 只有当 `validity + commit + safe fallback` 三者全部闭环后，才具备实际替换旧主线的条件

## 13. 验证命令建议

### 13.1 核心算法

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' `
  backend/tests/timeanchored_alignment/test_window_alignment_state.py `
  backend/tests/timeanchored_alignment/test_anchor_disambiguation.py `
  backend/tests/timeanchored_alignment/test_anchor_trust_evaluator.py `
  backend/tests/timeanchored_alignment/test_sparse_dp_chain_solver.py `
  backend/tests/timeanchored_alignment/test_gap_rescue_aligner.py `
  backend/tests/timeanchored_alignment/test_densification_promoter.py `
  backend/tests/timeanchored_alignment/test_timeline_validity_validator.py -v
```

### 13.2 集成桥接

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' `
  backend/tests/timeanchored_alignment/test_anchor_mount_service.py `
  backend/tests/timeanchored_alignment/test_alignment_stage_routing.py `
  backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py `
  backend/tests/timeanchored_alignment/test_alignment_stage_trace_payloads.py `
  backend/tests/timeanchored_alignment/test_job_p20260328_193846_no_duplicate.py `
  backend/tests/timeanchored_alignment/test_job_p20260328_202014_no_timestamp_chaos.py `
  backend/tests/textflow/test_decision_layer_ingress_context.py -v
```

### 13.3 性能验证口径

`test_anchor_mount_performance_benchmark.py` 首版建议作为非阻塞 benchmark：

1. 默认不进入主 CI 阻塞链
2. 记录 `block_count / compatibility_edge_count / solver_elapsed_ms / rescue_elapsed_ms`
3. 用于比较：
   - 旧贪心 solver
   - 新 sparse compatibility DP
   - 是否出现异常 edge density

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' `
  backend/tests/timeanchored_alignment/test_anchor_mount_performance_benchmark.py -v
```

## 14. 完成定义

这份详细开发文档对应的“完成”，不是把若干补丁打进现有链路，而是达到下面这个结构性目标：

1. 删除旧的 fail-open 关键分支
2. 用统一 `WindowAlignmentState` 把核心算法收口
3. 用 `core_pipeline` 承载唯一主线
4. 让 `seed / chain / gap rescue / densification / fuse / validity` 分工清晰
5. 让 `AlignmentStageService` 退出核心算法决策，只保留桥接与提交职责

只有达到这个形态，后续维护才不会重新长成一坨混杂调用链。
