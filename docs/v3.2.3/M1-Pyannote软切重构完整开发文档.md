# M1: Pyannote 软切重构完整开发文档

> 版本：V3.2.0+dev.20260214.06
> 状态：待实施

---

## 1. 文档概述

### 1.1 背景与目标

当前主链路中，`L6` 已是“文本边界切分”而非 speaker/turn 硬切，但说话人变化证据尚未形成统一软切裁决闭环，导致以下问题：

1. **漏切**：speaker_change 未被可靠转化为可执行切分点。
2. **错切**：单点抖动或低置信证据可能诱发不稳定切分。
3. **不可追溯**：缺乏“窗口-锚点-裁决”的统一决策记录。

M1 目标：

1. 把 Pyannote 从“硬切命令源”改为“证据源”。
2. 统一 `speaker_change -> 待切窗口 -> 锚点裁决 -> CutPlan -> L6执行`。
3. 为 M2 后处理简化与 NW V2 预埋稳定契约，不重复返工。

### 1.2 核心设计理念

```text
历史问题（误用边界）：
  边界点 -> 直接切句

M1 目标（软切）：
  speaker_change -> 证据分级 -> 待切窗口 -> 锚点评分 -> CutPlan -> L6执行
```

### 1.3 设计原则

| 原则 | 说明 |
|---|---|
| 主链收口 | 决策收口在 L6，避免新增复杂硬层 |
| 证据驱动 | 所有切分必须有证据链与风险标签 |
| 渐进落地 | 通过可选字段与回退路径渐进集成 |
| 可观测性 | 每次切分保留 `window_id/reason/risk` |

---

## 2. 架构总览

### 2.1 层级定位

M1 保留现有 L1-L7 分层，不新增物理“L5.5层”。

1. **逻辑上**：存在“L5.5 决策子层”（`EvidenceFusion + SoftCutDecisionEngine`）。
2. **实现上**：该子层落在 L6 入口路径，由 `segmentation_processor` 消费 `CutPlan` 执行。

```text
L5(语义注入输出) -> [逻辑L5.5: 证据融合与决策] -> L6(按CutPlan执行) -> L7
```

### 2.2 模块架构图

```text
┌─────────────────────────────────────────────────────────────┐
│                      AsyncDualPipeline                      │
│  ┌──────────────┐   ┌──────────────────────────────┐        │
│  │ L5 Output    │-->|  逻辑L5.5: Evidence+Decision │        │
│  │AnnotatedWord │   │  (生成 CutPlan)              │        │
│  └──────────────┘   └──────────────┬───────────────┘        │
│                                     │                        │
│                          ┌──────────▼───────────┐            │
│                          │ L6 Segmentation      │            │
│                          │ 按 CutPlan 执行切分   │            │
│                          └──────────┬───────────┘            │
│                                     │                        │
│                                  L7 输出                      │
└─────────────────────────────────────────────────────────────┘

证据输入：Timeline / Pause / Punctuation / (M3后接LLM语义)
```

### 2.3 数据流概览

```text
输入源：
1) SpeakerTimeline / Diarization / Segmentation
2) AnnotatedWord[] (L5)
3) PunctTrack / Pause signals

处理流程：
1) EvidenceBuilder 构建证据
2) EvidenceFusion 融合证据
3) SoftCutDecisionEngine 生成 CutPlan
4) L6 按 CutPlan 执行切分

输出：
1) SentenceSegment[]
2) Trace(window_id/reason/risk/anchor)
3) deferred 状态（用于增量修订）
```

### 2.4 与现有后处理关系

| 现有模块 | M1 改造 |
|---|---|
| `FinalSplitter` | 保留，作为锚点评分的停顿/标点信号来源 |
| `SegmentationProcessor.process()` | 改造，按 `CutPlan` 执行切分，不再自行硬判全部边界 |
| `SemanticInjector` | 保留，标点注入逻辑不变 |
| `DefaultAligner` | 保留，L4 对齐逻辑不变 |

调用链变化：

1. 现有：`L5 -> L6(SegmentationProcessor 自行切分) -> L7`
2. M1 后：`L5 -> EvidenceFusion -> SoftCutDecisionEngine -> L6(按 CutPlan 执行) -> L7`

### 2.5 M1 先测与 M2 即插接入策略

目标：先在当前架构中验证 soft-cut 收益，再在 M2 完成后处理简化后无缝切换，不二次改 L6 主链接口。

1. M1 测试期（现架构）：
   - 保留 `cut_plan=None` 旧路径作为基线。
   - 开启 `cut_plan` 新路径做 A/B 对比，验证漏切/错切收益。
   - 所有新决策必须落 trace（`window_id/reason/risk`），为 M2 回放对比提供样本。
2. M2 接入期（四模块）：
   - 用 `FactBuilder + EvidenceFusion + SoftCutDecisionEngine + OutputTrace` 替换 M1 内部实现。
   - `L6Input.cut_plan`、`L6Output.applied_cut_plan` 保持不变。
   - pipeline 仅替换 CutPlan 生产者，不改 L6/L7 消费口。
3. 切换原则：
   - 不改调用边界，只换内部策略。
   - 新增能力一律通过可选字段追加，不破坏旧字段语义。

---

## 3. 核心数据契约

### 3.1 证据类型定义

#### 3.1.1 SpeakerChangeEvidence（说话人变化证据）

```python
from dataclasses import dataclass
from typing import Literal


@dataclass
class SpeakerChangeEvidence:
    time: float
    from_speaker: str
    to_speaker: str
    pyannote_confidence: float
    pause_duration: float
    embedding_distance: float
    embedding_threshold: float
    is_abrupt_energy_shift: bool
    level: Literal["high", "mid", "low"]
    tags: set[str]  # rebound_merged, suspected_missed, cross_chunk_deferred
```

#### 3.1.2 AnchorScore（锚点评分）

```python
from dataclasses import dataclass


@dataclass
class AnchorScore:
    anchor_type: str  # pause_anchor / word_boundary / semantic_anchor / punctuation_anchor
    anchor_time: float
    base_score: float
    distance_penalty: float
    final_score: float
    source: str
```

评分口径：

1. `pause_anchor`: 0.8
2. `word_boundary`: 0.6
3. `semantic_anchor`: 0.5（M1 代理，M3 后由 LLM 主供）
4. `punctuation_anchor`: 0.3
5. `final_score = base_score - abs(anchor_time - t_change) * 0.1`

#### 3.1.3 CutWindow（待切窗口）

```python
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class CutWindow:
    window_id: str
    trigger_time: float
    trigger_level: Literal["high", "mid", "low"]
    start_time: float
    end_time: float
    chunk_id: str
    candidate_anchors: list[AnchorScore] = field(default_factory=list)
    state: Literal["open", "resolved", "forced", "expired"] = "open"
```

窗口口径：

1. 范围：`[t_change - window_before, t_change + window_after]`。
2. 默认：`window_before=0.5s`、`window_after=0.3s`（偏向提前切）。
3. 不跨 chunk：跨边界拆分为两个执行窗口，后半段标记 `cross_chunk_deferred`。
4. 窗口冲突：高置信窗口优先，低置信窗口收缩/并入；并入后低置信证据仅作辅助，不再独立触发落刀。

### 3.2 决策类型定义

#### 3.2.1 CutDecision（切分决策）

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class CutDecision:
    time: float
    window_id: str
    reason: str  # speaker_change / pause / length_pressure / deferred_forced / high_forced
    risk: Optional[str]  # deferred / singleton_unavoidable / overflow_forced
    anchor_type: str
    anchor_score: float
    depends_on_fast_draft: bool
    time_range: tuple[float, float]
```

#### 3.2.2 DeferredCut（延迟切分）

```python
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class DeferredCut:
    deferred_id: str
    window_id: str
    created_at: float
    expected_resolve_by: float
    state: Literal["pending", "resolved", "forced", "expired"] = "pending"
    resolution_decision: Optional[CutDecision] = None
```

#### 3.2.3 CutPlan（切分计划）

```python
from dataclasses import dataclass, field


@dataclass
class CutPlan:
    plan_id: str
    block_id: str
    decisions: list[CutDecision] = field(default_factory=list)
    deferred_cuts: list[DeferredCut] = field(default_factory=list)
    generation_report: dict = field(default_factory=dict)
```

### 3.3 类型注册（alignment/types.py 扩展）

M1 目标扩展（保持兼容）：

```python
@dataclass
class L6Input:
    annotated_words: List[AnnotatedWord]
    vad_intervals: Optional[List[Tuple[float, float]]] = None
    cut_plan: Optional[CutPlan] = None  # M1 新增，可选字段


@dataclass
class L6Output:
    sentence_segments: List[SentenceSegment]
    words_for_split: List[WordTimestamp]
    segmentation_report: Dict[str, Any] = field(default_factory=dict)
    applied_cut_plan: Optional[CutPlan] = None  # M1 新增，可选字段
```

兼容策略：

1. `cut_plan=None` 时走现有默认切分路径。
2. 新字段均为可选，不破坏旧调用方。

### 3.4 为 M2 预埋的四模块契约

```python
@dataclass
class AlignedFacts:
    words: list[AlignedWord]
    speaker_turns: list[SpeakerTurn]
    fast_draft_cuts: list[float]
    time_axis_version: str = "m1_legacy"
    nw_alignment_score: float | None = None
    word_time_confidence: list[float] | None = None


@dataclass
class FusedEvidence:
    speaker_changes: list[SpeakerChangeEvidence]
    pause_anchors: list[PauseAnchor]
    semantic_anchors: list[SemanticAnchor]
    punctuation_anchors: list[PunctuationAnchor]


@dataclass
class FinalSegments:
    segments: list[SentenceSegment]
    trace: list[SegmentTrace]
```

字段口径：

1. M1 阶段 `time_axis_version` 固定为 `m1_legacy`，`nw_alignment_score/word_time_confidence` 可空。
2. M2 引入 NW V2 后，填充 `time_axis_version="m2_nw_v2"` 与对齐置信字段。
3. Soft-cut 决策层必须兼容“有/无 NW 字段”两种输入，保证 M1 可先测。

### 3.5 M1 阶段 semantic_anchor 代理实现

M1 未接入 LLM 时，`semantic_anchor` 由轻量规则代理生成：

1. 句末标点代理：`。！？.!?` 生成 `source="punct_proxy"`，默认 `confidence=0.5`。
2. 连词前置代理：在“但是/然后/所以/不过”等连词前生成 `source="conjunction_rule"`，默认 `confidence=0.3`。
3. M3 替换口径：接入 LLM 后，`source` 切换为 `llm`，`confidence` 由模型直接输出。

### 3.6 M1 -> M2 接口冻结清单

以下字段在 M2 前后保持语义稳定，不允许改名或改含义：

1. `CutDecision.window_id/reason/risk/anchor_type/anchor_score/time_range`
2. `DeferredCut.window_id/state/expected_resolve_by`
3. `CutPlan.decisions/deferred_cuts/generation_report`
4. `L6Input.cut_plan` 与 `L6Output.applied_cut_plan`
5. `segmentation_report` 中 soft-cut 聚合统计键名

冻结规则：

1. 允许新增可选字段，不允许删除或重定义既有字段。
2. 若必须变更字段语义，先新增新字段并保留旧字段至少一个开发周期。
3. M2 上线前必须通过“字段兼容回放测试”（M1 样本可被 M2 正确重放）。

---

## 4. 核心模块实现

### 4.1 新增文件清单

| 文件路径 | 职责 | 行数估计 |
|---|---|---|
| `backend/app/services/segmentation/soft_cut/types.py` | 软切契约定义 | ~180 |
| `backend/app/services/segmentation/soft_cut/evidence_builder.py` | 证据构建 | ~220 |
| `backend/app/services/segmentation/soft_cut/evidence_fusion.py` | 证据融合 | ~220 |
| `backend/app/services/segmentation/soft_cut/decision_engine.py` | 窗口管理与决策 | ~280 |
| `backend/app/services/segmentation/soft_cut/__init__.py` | 模块导出 | ~20 |

说明：路径落在 `segmentation` 域，保证“主逻辑在 L6”。

### 4.2 EvidenceBuilder（证据构建器）

职责：

1. 从 timeline/diarization 构建 `SpeakerChangeEvidence`。
2. 从词与停顿构建 `AnchorScore` 候选。
3. 输出窗口触发所需基础事实，不做最终裁决。

关键逻辑：

1. `speaker_change` 分级：
   - `score = pyannote_confidence + pause_bonus + embedding_bonus + energy_bonus`
   - `>=0.8 -> high`，`>=0.5 -> mid`，否则 `low`
2. 去抖冷却：
   - 默认 `cooldown_ms=300`
   - 动态区间：快语速可降至 `180-220ms`，连续回弹可升至 `350-450ms`
   - 回弹 `A->B->A` 合并为抖动标签 `rebound_merged`

分级伪代码：

```python
def classify_speaker_change(evidence: SpeakerChangeEvidence) -> str:
    score = evidence.pyannote_confidence
    if evidence.pause_duration > 0.30:
        score += 0.15
    if evidence.embedding_distance > evidence.embedding_threshold:
        score += 0.20
    if evidence.is_abrupt_energy_shift:
        score += 0.05
    if score >= 0.80:
        return "high"
    if score >= 0.50:
        return "mid"
    return "low"
```

去抖伪代码：

```python
def debounce_speaker_changes(changes: list[SpeakerChangeEvidence], cooldown_ms: int = 300):
    result = []
    for change in changes:
        if not result:
            result.append(change)
            continue
        last = result[-1]
        gap = change.time - last.time
        if gap < cooldown_ms / 1000:
            if change.to_speaker == last.from_speaker:
                # A->B->A 回弹：视作抖动合并
                last.tags.add("rebound_merged")
            else:
                # 非回弹短间隔：保留为真实快速对话
                result.append(change)
        else:
            result.append(change)
    return result
```

### 4.3 EvidenceFusion（证据融合服务）

职责：

1. 窗口内融合 pause/word/semantic/punctuation 锚点。
2. 计算 `final_score` 并排序。
3. 处理冲突窗口（高置信优先）。

锚点选择规则：

1. 选 `final_score` 最高锚点。
2. 同分选离 `t_change` 最近锚点。
3. 无锚点进入 `deferred`。

M2 对接预埋：

1. 若 `AlignedFacts.word_time_confidence` 存在，允许作为锚点评分修正项。
2. 若 `time_axis_version="m2_nw_v2"`，允许降低距离惩罚系数以利用更准时基。
3. 若上述字段缺失，保持 M1 默认评分，不阻塞现架构测试。

### 4.4 SoftCutDecisionEngine（软切决策引擎）

职责：

1. 生成/管理 `CutWindow`。
2. 应用 P0-P3 约束，输出 `CutPlan`。
3. 管理 `deferred` 生命周期与强制策略。

核心决策：

1. `high`：必须切；超时阈值复用 `deferred_max_wait_sec`；超时标记 `high_forced`。
2. `mid`：按“锚点质量 + 长度压力”矩阵决策。
3. `low`：入 `evidence_queue`，慢流或 chunk 结束消费，必要时升级触发 `subtitle.revised`。

mid 决策矩阵：

| 锚点质量 | 当前句长 | 决策 |
|---|---|---|
| 高（`>=0.6`） | 任意 | 立即切 |
| 中（`0.4-0.6`） | `>15`词 | 立即切 |
| 中（`0.4-0.6`） | `<=15`词 | 延后 |
| 低（`<0.4`） | `>25`词 | 立即切（长度压力） |
| 低（`<0.4`） | `<=25`词 | 延后 |

长度压力阈值：

1. `length_pressure_soft=15`
2. `length_pressure_hard=25`

`low` 级别消费流程：

1. 写入时机：speaker_change 分级为 `low` 后立即写入 `evidence_queue`。
2. 消费时机：慢流覆盖返回时即时消费，或 chunk 结束批量消费。
3. 消费动作：若升级为 `mid/high`，触发 `subtitle.revised` 增量修订。

deferred 生命周期：

1. `created`：窗口创建但暂无合格锚点。
2. `waiting`：等待后续 `N 秒` 或 `M 个词`。
3. `resolved`：命中锚点切分，标记 `deferred_resolved`。
4. `forced`：超时强制词边界切，标记 `deferred_forced`。
5. `revisit`：慢流或 LLM（M3）回证据后重评。

默认阈值：

1. `deferred_max_wait_sec=2.5`
2. `deferred_max_words=18`
3. `deferred_force_min_word_boundary_gap=0.08`
4. `MAX_DEFERRED_WINDOWS=3`（超上限强制处理最旧窗口并标记 `overflow_forced`）

### 4.5 切分优先级与强约束

1. `P0 强约束`：禁止句中硬切；禁止单点抖动直接切句。
2. `P1 主目标`：speaker_change 必须触发待切窗口。
3. `P2 落点锚点`：停顿与词边界 > 语义锚点 > 标点锚点。
4. `P3 兜底`：无高质量锚点时进入 deferred，超时再词边界强制切。

合并约束：

1. 禁跨高置信 speaker_change 合并。
2. 低置信 speaker_change 允许延后，不允许静默吞并。
3. 1 词孤句优先并入邻句；无法并入时保留并打 `singleton_unavoidable`。

P0 冲突顺序：

1. 禁句中硬切。
2. 禁跨高置信 speaker_change 合并。
3. 最后才考虑避免 1 词孤句。

---

## 5. 现有文件修改清单

### 5.1 修改文件总览

| 文件 | 目标改造 |
|---|---|
| `backend/app/services/alignment/types.py` | 扩展 `L6Input/L6Output` 可选 soft-cut 字段 |
| `backend/app/services/timeline/diarization_service.py` | 强化证据输出，保持“事实层”职责 |
| `backend/app/services/timeline/turn_builder.py` | 保留 turn 事实，不输出硬切命令 |
| `backend/app/services/segmentation/segmentation_processor.py` | 新增按 `CutPlan` 执行路径 + 默认回退 |
| `backend/app/pipelines/async_dual_pipeline.py` | 串接 Evidence/Decision 到 L6 输入 |

### 5.2 alignment/types.py 修改

1. `L6Input` 新增可选 `cut_plan`。
2. `L6Output` 新增可选 `applied_cut_plan`。
3. 保持原字段不变，避免下游破坏。

### 5.3 diarization_service.py 修改

1. 输出 speaker 变化证据必要字段：
   - `pyannote_confidence`
   - `is_overlap`
   - 时间范围
2. 不在该层做切分判决。
3. 未知人数场景遵循自动估计或 min/max 约束，不固定人数硬编码。

### 5.4 turn_builder.py 重构

1. 继续产出 turn 事实（`speaker_id/start/end/boundary_confidence`）。
2. 禁止把 turn-change 直接下发为“强制切分命令”。
3. 允许为 soft-cut 提供辅助证据（如长停顿、边界置信度）。

### 5.5 segmentation_processor.py 扩展

目标：

1. `cut_plan` 存在时：按 `CutPlan` 执行切分。
2. `cut_plan` 缺失时：保持当前默认路径（`FinalSplitter`）。
3. 输出 `segmentation_report` 增加 `window_id/reason/risk` 聚合统计。

### 5.6 async_dual_pipeline.py 集成

1. 在 L5->L6 之间生成 `CutPlan`。
2. 维护 `pending_deferred` 跨 chunk 状态。
3. 慢流返回后仅重算受影响窗口：

```python
def find_affected_windows(existing_plan: CutPlan, slow_result: SlowGroupResult) -> list[str]:
    affected = []
    for decision in existing_plan.decisions:
        if decision.depends_on_fast_draft and slow_result.covers(decision.time_range):
            affected.append(decision.window_id)
        if decision.risk == "deferred" and slow_result.has_new_anchors(decision.time_range):
            affected.append(decision.window_id)
    return sorted(set(affected))
```

---

## 6. 分阶段实施计划

### 6.1 阶段概览

| 阶段 | 名称 | 目标 | 预计工作量 |
|---|---|---|---|
| A | 契约定义 | 完成软切类型与 L6 扩展字段 | 小 |
| B | 证据层 | 实现 EvidenceBuilder 与窗口构建 | 中 |
| C | 决策层 | 实现 EvidenceFusion + DecisionEngine | 中 |
| D | 集成层 | 接入 L6 与 pipeline，保留回退 | 中 |
| E | 验证层 | 指标评估、回归、参数收敛 | 中 |

### 6.2 Phase A: 契约定义

任务：

1. 定义 `SpeakerChangeEvidence/AnchorScore/CutWindow/CutDecision/DeferredCut/CutPlan`。
2. 扩展 `alignment/types.py` 的 `L6Input/L6Output`。
3. 明确字段约束：
   - `window_id` 必填
   - `reason/risk` 必入 trace
   - `tags` 枚举化

验收：

1. 类型检查通过。
2. 无循环依赖。

### 6.3 Phase B: 证据层

任务：

1. 实现证据提取与分级。
2. 实现窗口生成与冲突处理。
3. 实现锚点评分（含距离惩罚）。

验收：

1. `high/mid/low` 分布可观测。
2. 去抖后有效事件数稳定。

### 6.4 Phase C: 决策层

任务：

1. 实现 mid 决策矩阵。
2. 实现 deferred 生命周期与 `high_forced`。
3. 实现 low 证据消费触发逻辑。

验收：

1. 决策路径覆盖 complete。
2. deferred 有终态，不允许静默积压。

### 6.5 Phase D: 集成层

任务：

1. `segmentation_processor` 新增 CutPlan 执行路径。
2. `async_dual_pipeline` 串接 soft-cut 生成与增量更新。
3. 保留默认回退路径。

验收：

1. 不破坏现有任务主流程。
2. 回退路径可用。

### 6.6 Phase E: 验证层

任务：

1. 单测 + 集成测试 + 文本域评估。
2. 参数调优（按依赖顺序）。
3. 固化失败排查清单。

验收：

1. 无句中硬切回归。
2. 主指标达到或优于基线。

### 6.7 参数规划与调参顺序

首批参数：

1. `speaker_change_min_score`
2. `debounce_cooldown_ms`
3. `soft_cut_window_before_sec`
4. `soft_cut_window_after_sec`
5. `anchor_pause_min_sec`
6. `length_pressure_soft`
7. `length_pressure_hard`
8. `deferred_max_wait_sec`
9. `deferred_max_words`
10. `singleton_merge_max_duration_sec`

依赖关系：

1. `speaker_change_min_score` + `debounce_cooldown_ms` 决定有效事件数量与 high/mid/low 分布。
2. `soft_cut_window_before_sec` + `soft_cut_window_after_sec` + `anchor_pause_min_sec` 决定可命中的锚点质量。
3. `length_pressure_soft/hard` 决定 mid 级别是否提前切分。
4. `deferred_max_wait_sec/deferred_max_words` 决定 deferred 解决率与 forced 比例。
5. `singleton_merge_max_duration_sec` 决定兜底回并范围与孤句率。

建议调参顺序：

1. 先调 `speaker_change_min_score`（漏切/错切平衡）。
2. 再调 `debounce_cooldown_ms`（抑抖但保留快速对话）。
3. 再调窗口与停顿参数（落刀精度）。
4. 再调长度压力参数（长句与提前切平衡）。
5. 最后调 deferred 与 singleton 参数（兜底行为）。

### 6.8 M1 到 M2 的实施节奏

阶段顺序（最终口径）：

1. 先完成 `M1-A -> M1-B -> M1-C -> M1-D -> M1-E` 一轮闭环（在现架构验证 soft-cut）。
2. 再进入 `M2`（后处理简化 + NW V2），但继续产出同一 `CutPlan` 契约。
3. M2 完成后执行“生产者切换”：CutPlan 生产从 M1 逻辑实现切到 M2 四模块实现。
4. 切换后先小样本回放，再全量灰度放量，始终保留回退开关。

关于 `E` 阶段（验证层）：

1. `M1-E` 不是一次性结束项，而是持续门禁项。
2. M2 开发与切换期间继续沿用 `M1-E` 同一指标与同一评测口径做回归卡闸。
3. 任一阶段出现 P0 违规或主指标退化即阻断继续放量。

切换完成判据：

1. `L6/L7` 接口与调用方零改动。
2. M1 历史样本在 M2 引擎下可回放并达到不降级指标。
3. 回退到 M1 生产者时行为可恢复。

---

## 7. 测试与评估

### 7.1 单元测试用例

1. EvidenceBuilder：
   - 分级公式
   - 去抖与回弹
   - 锚点评分
2. DecisionEngine：
   - 窗口生成/冲突
   - mid 决策矩阵
   - deferred/high_forced

### 7.2 集成测试用例

1. `cut_plan` 存在时 L6 正确按计划执行。
2. `cut_plan` 缺失时 L6 默认路径保持稳定。
3. 慢流返回后仅受影响窗口被重算。
4. SSE 修订事件 `subtitle.revised` 可用。

### 7.3 评估指标

主指标（clean_set）：

1. `change_hit_rate`
2. `cross_speaker_miss_rate`
3. `extra_cut_in_clean_scope`
4. `singleton_sentence_rate`
5. `mid_sentence_wrong_split_rate`

观察指标（overlap_set）：

1. `overlap_observation_hit_rate`
2. `deferred_resolved_rate`
3. `deferred_forced_rate`
4. `overflow_forced_count`

### 7.4 评估数据集

推荐用“标注 SRT -> JSONL”流程：

```bash
python backend/scripts/export_speaker_marked_srt_to_jsonl.py ^
  --input-srt tests/test_en_1_speaker.srt ^
  --output-jsonl jobs/test_en_1_speaker.speaker_units.jsonl ^
  --output-changes-jsonl jobs/test_en_1_speaker.speaker_changes.jsonl ^
  --default-tolerance-sec 0.45
```

口径：

1. 允许 `11.5` 编号与粗粒度切口。
2. 命中采用容差窗口，不做毫秒级硬命中。
3. 重叠样本不计入主分。

### 7.5 标注规范（长期复用）

1. 标注文件使用 SRT，speaker 用 `（S1）/（S2）`。
2. 在确认切换点的字幕末尾加 `//`。
3. 支持插入编号（如 `11.5`），编号不参与评测。
4. 重叠语音标记 `（OVL）/[OVL]/#OVL#`，默认 `eval_include=false`。
5. 时间戳允许粗标注，评测以容差窗口命中为准。

### 7.6 每轮固定执行流程

1. 准备 `clean_set` 与 `overlap_set`。
2. 运行 `dual_full/smart_patch/fast_only` 三种模式。
3. 收集最终 SRT、`subtitle_speaker_links`（如可用）、soft-cut trace。
4. 计算主指标（clean）与观察指标（overlap）。
5. 归档参数快照、指标 JSON、失败案例。

### 7.7 回归闸门

1. `change_hit_rate` 不低于稳定基线。
2. `cross_speaker_miss_rate` 连续两轮下降或持平。
3. `extra_cut_in_clean_scope` 不高于基线 `+10%`。
4. `singleton_sentence_rate` 保持 `<2%` 或不高于基线。
5. 任何 P0 违规（句中硬切）直接判失败。

### 7.8 失败排查顺序

1. 先查 P0 违规（句中硬切/跨高置信误并）。
2. 再查分级阈值（是否过高导致漏切）。
3. 再查去抖参数（是否误杀快速对话）。
4. 再查 deferred（是否堆积或 forced 过高）。
5. 最后查锚点权重与窗口参数。

### 7.9 测试注意点

1. 重叠语音与 `UNKNOWN` 样本不参与主指标。
2. 不用毫秒级硬命中评估粗标注切口。
3. 跨版本比较必须对齐参数快照。
4. 指标定义、脚本参数、产物路径必须同步更新。

### 7.10 M1 先测 + M2 接入验收

1. 基线组：`cut_plan=None`（当前默认路径）。
2. M1 组：`cut_plan` 由 M1 逻辑生成。
3. M2 组：`cut_plan` 由 M2 四模块生成（接口不变）。
4. 对比要求：
   - M1 组需优于基线组或至少在主指标不退化。
   - M2 组相对 M1 组需达到“无接口改动 + 指标不退化”。
5. 强制验收：
   - 任一组出现 P0 违规即失败。
   - 任一组不能回放历史样本即失败。

---

## 8. 风险与缓解

### 8.1 技术风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 窗口过宽或过窄 | 锚点误选/漏切 | 调整 `window_before/after` |
| 抖动过滤过强 | 误杀快对话 | 动态 `debounce_cooldown_ms` |
| deferred 积压 | 长尾修订/内存压力 | `MAX_DEFERRED_WINDOWS=3` |
| low 证据长期不消费 | 漏切残留 | chunk 结束批量消费 + 慢流触发消费 |

### 8.2 兼容性风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| `L6Input/L6Output` 扩展 | 下游调用变化 | 字段保持可选 |
| 新模块引入 | 导入耦合 | 按域分层与清晰边界 |
| 参数新增 | 调参混乱 | 固化调参顺序与默认值 |
| M1/M2 契约漂移 | M2 接入返工 | 冻结 `CutPlan` 关键字段并做回放兼容测试 |

### 8.3 缓解策略

1. 保留回退路径（无 CutPlan 时走默认切分）。
2. 先灰度再放量，持续观测主指标。
3. 所有异常路径必须带风险标签并可回放。

### 8.4 回退策略

1. 参数过激导致切分震荡时，降级 `smart_patch`（仅高置信生效）。
2. deferred 堆积导致晚修订过多时，启用窗口上限强制策略。
3. Pyannote 噪声升高时，提高 `speaker_change_min_score` 并提升停顿锚点权重。

---

## 9. 附录

### 9.1 术语表

| 术语 | 定义 |
|---|---|
| 硬切 | 说话人变化点直接切句 |
| 软切 | speaker_change 触发窗口后选锚点切句 |
| 待切窗口 | 围绕变化点搜索最优切分位置的时间范围 |
| deferred | 当前窗口无合格锚点时延后处理 |
| high_forced | high 级别超时后词边界强制切 |

### 9.2 参考文档

1. `docs/v3.2.3/M1-Pyannote软切重构与接口预埋.md`
2. `docs/v3.2.3/M2-后处理简化与Needleman-Wunsch-V2融合.md`
3. `docs/v3.2.3/重构执行总纲-M1-M5.md`
4. `llmdoc/architecture/text-processing-v3.2.3-evaluation.md`
5. `llmdoc/architecture/speaker-refactor-phase2-timeline-domain.md`

### 9.3 文档维护规则

1. 本文档只描述当前目标态，不记录流水账式历史。
2. 若方案变化，采用整段替换，不追加“补丁注释”。
3. 文档修改后需与代码契约同步核对。

### 9.4 关键字段约束

1. `CutDecision.window_id` 必填，且必须可回溯到具体窗口与触发证据。
2. `CutDecision.reason/risk` 必须写入 trace，保证“为何切/为何未切”可回放。
3. `SpeakerChangeEvidence.tags` 仅允许枚举标签，禁止自由文本污染统计口径。

---

## 10. 前端说话人编辑与后端唯一真源方案

### 10.1 目标与原则

1. 说话人相关数据（`speaker_id/turn_id/颜色/绑定来源`）以**后端**为唯一真源。
2. 前端状态仅作为展示缓存，不拥有最终写入权。
3. 所有前端人工改 speaker 行为必须先写后端，再由后端广播回前端。
4. 草稿字幕不参与 speaker 最终归属判定，避免把不稳定结果固化到 UI。

### 10.2 数据真源边界

后端真源分层：

1. `subtitle_speaker_links`：句级 speaker 绑定真源（可被人工改绑）。
2. `speaker_profiles`：说话人展示信息真源（名称、颜色、锁定态）。
3. `turn_speaker_links`：时轴检测事实真源（系统写入，前端不可直接改）。
4. `speaker_audit_logs`：人工操作审计真源（可回放与排障）。

前端边界：

1. 不在本地生成或持久化“最终 speaker 归属”。
2. 不允许绕过后端直接改句子内 `speaker_id` 并视为生效。
3. 页面刷新后必须以接口回传结果覆盖本地缓存。

### 10.3 草稿与定稿的字段约束（强制）

#### 10.3.1 草稿事件（`subtitle.draft`）

1. **禁止携带**说话人标签字段：
   - `speaker_id`
   - `turn_id`
   - `speaker_color_key`
   - `binding_source`
   - `speaker_label`
2. 草稿仅允许包含文本与时间相关字段（`text/start/end/words/confidence`）。
3. 前端对草稿字幕不做说话人染色。

#### 10.3.2 定稿/修订事件（`subtitle.replace_chunk` / `subtitle.finalized` / `subtitle.revised`）

1. **必须携带**句级 speaker 字段：
   - `speaker_id`
   - `turn_id`（可空）
   - `speaker_color_key`
   - `binding_source`（`auto`/`user`）
   - `speaker_label`（展示别名，缺失时前端回退 `speaker_id`）
2. 前端仅对定稿与修订字幕启用说话人染色。
3. 若字段缺失，按数据不完整处理并记录告警，不回退到“猜测 speaker”。

### 10.4 前端人工改 speaker 的同步闭环

1. 用户在前端修改 speaker（改绑/改名/改色）。
2. 前端调用后端接口：
   - `PATCH /api/speakers/{job_id}/subtitles/{sentence_index}`（改绑 speaker）
   - `PATCH /api/speakers/{job_id}/profiles/{speaker_id}`（改名/改色/锁定）
   - `POST /api/speakers/{job_id}/profiles/merge`（合并 speaker）
3. 后端在单事务内写入真源表与审计表。
4. 后端完成写入后立即广播 SSE：
   - 句子改绑：`subtitle.revised`
   - profile 变更：`subtitle.speaker_profiles`
5. 前端仅消费后端广播结果更新 UI，禁止“前端先改后端失败仍保留本地结果”。

### 10.5 一致性与冲突处理

1. 写接口响应必须返回 `updated_at`（毫秒时间戳）用于并发判断。
2. SSE 事件必须带 `revision_id` 与 `updated_at`，前端按“时间新者覆盖旧者”处理。
3. 多端并发修改采用“后端最后写入生效（Last Write Wins）+ 审计可追溯”。
4. 前端若收到 409/422，必须回滚本地临时态并拉取后端最新数据。

### 10.6 与 M1 soft-cut 的联动约束

1. `subtitle.revised` 若由 soft-cut 触发，必须附带：
   - `affected_window_ids`
   - `reason/risk`（来自 `CutDecision`）
2. 修订只允许作用于受影响窗口，不允许全量覆写无关句子。
3. `window_id/reason/risk` 与 speaker 归属变更必须可联合回放。

### 10.7 接口返回与拉取口径

1. `GET /api/transcription-text/{job_id}` 返回的 `segments` 必须是定稿口径，并包含 speaker 字段。
2. 恢复场景（checkpoint/snapshot）返回的字幕同样遵守“草稿无 speaker、定稿有 speaker”。
3. 前端首屏加载时以该接口为准重建状态，SSE 仅做增量更新。

### 10.8 验收标准

1. 草稿事件中 speaker 字段出现率必须为 `0`。
2. 定稿与修订事件中 speaker 必填字段完整率必须为 `100%`。
3. 前端改 speaker 后，`<=1s` 内可收到后端回推并更新 UI。
4. 刷新页面后，speaker 展示与后端查询结果一致。
5. 多端同时编辑后，不出现前端与后端长期分叉状态。

---

> 文档结束
