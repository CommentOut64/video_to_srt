# M1 Pyannote软切重构与接口预埋

> Type: Architecture | Status: Active
> Version: V3.2.0+dev.20260214.03
> Scope: 说话人软切与后处理简化接口预埋；不包含 LLM 语义主用切分

## 1. Summary

* **Goal**: 把 Pyannote 从“硬切命令”改为“证据输入”，同时一次性预埋 M2 后处理简化所需的数据契约，避免二次拆改。
* **交付原则**:
  - 说话人变化只触发“待切窗口”，不直接落刀。
  - 先落契约，再落策略；先可追溯，再做激进优化。
  - 所有决策必须输出风险标签与证据来源。

## 2. Diagram

* `Pyannote diarization(exclusive)` + `segmentation frames` + `embedding变化` + `pause` -> `SpeakerEvidenceTrack`
* `SpeakerEvidenceTrack` -> `speaker_change分级(high/mid/low)` -> `待切窗口`
* `待切窗口` -> `锚点裁决` -> `切分/延迟(deferred)` -> `OutputTrace`
* `M1契约对象` -> `M2四模块主链直接复用`

## 3. Key Components

* `backend/app/services/timeline/diarization_service.py`: 提供说话人变化原始证据。
* `backend/app/services/timeline/segmentation_service.py`: 提供帧级变化分数与候选边界。
* `backend/app/services/timeline/speaker_timeline_service.py`: 统一组装时间线事实与边界证据。
* `backend/app/services/segmentation/segmentation_processor.py`: 执行软切窗口裁决。
* `backend/app/services/alignment/types.py`: 新增软切契约对象（M1 预埋）。
* `backend/app/pipelines/async_dual_pipeline.py`: 下传事实与证据，不下传硬切命令。

### 3.1 与现有后处理的关系

| 现有模块 | M1 改造 |
|---|---|
| `FinalSplitter` | 保留，作为锚点评分的停顿/标点信号来源 |
| `SegmentationProcessor.process()` | 改造，按 `CutPlan` 执行切分，不再自行硬判全部边界 |
| `SemanticInjector` | 保留，标点注入逻辑不变 |
| `DefaultAligner` | 保留，L4 对齐逻辑不变 |

调用链变化：

1. 现有：`L5 -> L6(SegmentationProcessor 自行切分) -> L7`
2. M1 后：`L5 -> EvidenceFusion -> SoftCutDecisionEngine -> L6(按 CutPlan 执行) -> L7`

## 4. 切分优先级与强约束

### 4.1 P0-P3 优先级

1. `P0 强约束`：禁句中硬切、禁单点抖动直接切句。
2. `P1 主目标`：speaker_change 必须触发待切窗口。
3. `P2 落点锚点`：停顿与词边界 > 语义锚点 > 标点锚点。
4. `P3 兜底`：窗口内无高质量锚点时，延迟到最近可接受停顿或超时强制词边界切。

### 4.2 合并约束

1. 禁跨高置信 speaker_change 合并。
2. 允许在低置信 speaker_change 条件下延迟切分。
3. 1 词孤句默认并入邻句，无法并入时保留并打标。
4. 上述约束作用于所有合并环节（短句回并、窗口合并、跨批次拼接）。

### 4.3 P0 冲突解决顺序（最终版）

1. 禁句中硬切。
2. 禁跨高置信 speaker_change 合并。
3. 尽量避免 1 词孤句。  
说明：必要时允许 1 词孤句并标记 `singleton_unavoidable`。

### 4.4 待切窗口定义

1. 窗口范围：`[t_change - window_before, t_change + window_after]`。
2. 默认参数：`window_before=0.5s`、`window_after=0.3s`（偏向提前切，降低句中截断风险）。
3. 窗口标识：每个窗口必须分配唯一 `window_id`，用于追溯、增量更新与恢复。
4. 窗口冲突处理：
   - 两个窗口重叠时，高置信窗口优先；
   - 低置信窗口执行收缩，必要时并入高置信窗口。
   - 并入后，低置信窗口的 speaker_change 证据降级为高置信窗口的辅助证据，不再独立触发落刀判定。
5. 窗口与 chunk 边界：
   - 窗口不跨 chunk 边界；
   - 跨边界时拆分为前后两个执行窗口，后半段标记 `cross_chunk_deferred`。

### 4.5 锚点评分与选择

1. 锚点类型与基础分：

| 类型 | 基础分 | 说明 |
|---|---:|---|
| `pause_anchor` | 0.8 | 停顿 > 300ms |
| `word_boundary` | 0.6 | 词边界 + 短停顿 |
| `semantic_anchor` | 0.5 | M1 由标点代理，M3 后由 LLM 提供 |
| `punctuation_anchor` | 0.3 | 纯标点模型输出 |

2. 距离惩罚：`final_score = base_score - abs(anchor_time - t_change) * 0.1`。
3. 选择规则：
   - 窗口内选 `final_score` 最高锚点；
   - 同分时选离 `t_change` 最近锚点；
   - 无可用锚点时进入 `deferred`。

## 5. speaker_change 分级标准（必须可计算）

### 5.1 多因子评分公式

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

### 5.2 分级动作

1. `high`：必须切（窗口内必须落刀，必要时触发超时强制切）。
2. `mid`：可切可延后，由 `SoftCutDecisionEngine` 根据锚点质量与长度压力决定。
3. `low`：仅记录证据，不立即切；进入慢流/LLM 复核候选。

补充约束：

1. `high` 级别超时阈值复用 `deferred_max_wait_sec`。
2. `high` 超时后在词边界强制切分，并标记 `high_forced`。

### 5.3 mid/low 消费规则

1. `mid` 立即切条件：命中高质量停顿锚点，且不触发 P0 冲突。
2. `mid` 延后条件：仅命中低质量锚点，或切分后会形成高风险孤句。
3. `low` 消费链路：先写入 `evidence_queue`，再由慢流复核；M3 开启后再进入 LLM 复核。

### 5.4 mid 级别决策矩阵

| 锚点质量 | 当前句长 | 决策 |
|---|---|---|
| 高（`>=0.6`） | 任意 | 立即切 |
| 中（`0.4-0.6`） | `>15` 词 | 立即切 |
| 中（`0.4-0.6`） | `<=15` 词 | 延后 |
| 低（`<0.4`） | `>25` 词 | 立即切（长度压力） |
| 低（`<0.4`） | `<=25` 词 | 延后 |

长度压力阈值：

1. `length_pressure_soft=15`（进入偏向切分区间）。
2. `length_pressure_hard=25`（触发强制切分判定）。

### 5.5 low 级别证据消费流程

1. 写入时机：speaker_change 分级为 `low` 后立即写入 `evidence_queue`。
2. 消费时机：
   - 慢流返回且覆盖该时间范围时即时消费；
   - 或 chunk 结束时批量消费。
3. 消费动作：
   - 若慢流提供新证据（例如更高置信度锚点），执行重新分级；
   - 若 `low` 升级为 `mid/high`，触发句段修订并通过 SSE 发送 `subtitle.revised`。
4. M3 扩展：LLM 作为额外消费者，可进一步升级或确认 `low` 证据。

## 6. 去抖与冷却机制

### 6.1 机制目标

1. 清理 Pyannote 抖动回弹。
2. 不误杀真实快速对话。

### 6.2 动态冷却参数

1. 默认 `cooldown_ms=300`。
2. 语速快时降至 `180-220ms`。
3. 连续回弹时升至 `350-450ms`。

### 6.3 参考伪代码

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
                last.confidence = max(last.confidence, change.confidence)
                last.tags.add("rebound_merged")
            else:
                result.append(change)  # 快速对话豁免
        else:
            result.append(change)
    return result
```

## 7. “漏切不是无解”的工程化方案

1. **证据融合补检**：当 pyannote 无变化但 `embedding_distance`、停顿、韵律突变同时异常时，生成 `suspected_missed_change` 事件。
2. **延迟纠错**：窗口未命中时进入 deferred，后续捕捉到停顿/语义锚点再补切。
3. **慢流复核**：将 `low` 与 `suspected_missed_change` 推入慢流复核队列。
4. **LLM 复核预留**：M3 引入后，低置信冲突交由 LLM 语义边界模型二次判决。
5. **风险标注可回放**：漏切链路必须可追溯到“证据不足”而非静默失败。

## 8. deferred_cut 生命周期

### 8.1 生命周期定义

1. `created`：speaker_change 触发窗口但未找到可接受锚点。
2. `waiting`：在后续 `N 秒` 或 `M 个词`内等待锚点。
3. `resolved`：命中锚点后落刀，标记 `deferred_resolved`。
4. `forced`：超时仍未命中，词边界强制切，标记 `deferred_forced`。
5. `revisit`：慢流或 LLM 提供新证据后重评。

### 8.2 默认阈值

1. `deferred_max_wait_sec=2.5`
2. `deferred_max_words=18`
3. `deferred_force_min_word_boundary_gap=0.08`

口径统一：

1. `high` 级别窗口的超时判断复用 `deferred_max_wait_sec`，避免双阈值漂移。

### 8.3 前端显示策略

1. `created/waiting` 阶段显示暂定合并句。
2. `resolved/forced` 通过增量替换事件修订字幕。

### 8.4 堆积上限

```python
MAX_DEFERRED_WINDOWS = 3
```

超过上限时，强制处理最旧窗口并标记 `overflow_forced`。

## 9. 为 M2 预埋的统一契约（M1 必须落地）

### 9.1 FactBuilder 输出（事实层）

```python
@dataclass
class AlignedFacts:
    words: list[AlignedWord]
    speaker_turns: list[SpeakerTurn]
    fast_draft_cuts: list[float]
```

### 9.2 EvidenceFusion 输出（证据层）

```python
@dataclass
class FusedEvidence:
    speaker_changes: list[SpeakerChangeEvidence]
    pause_anchors: list[PauseAnchor]
    semantic_anchors: list[SemanticAnchor]
    punctuation_anchors: list[PunctuationAnchor]
```

### 9.3 DecisionEngine 输出（决策层）

```python
@dataclass
class CutPlan:
    decisions: list[CutDecision]
    deferred: list[DeferredCut]
    merged_singletons: list[MergedSingleton]
```

### 9.4 OutputTrace 输出（追踪层）

```python
@dataclass
class FinalSegments:
    segments: list[SentenceSegment]
    trace: list[SegmentTrace]
```

### 9.5 M1 阶段 `semantic_anchor` 代理实现

M1 阶段未引入 LLM，`SemanticAnchor` 由规则代理生成：

1. 句末标点代理：
   - 在 `。！？.!?` 位置生成 `SemanticAnchor(confidence=0.5, source="punct_proxy")`。
2. 连词前置代理：
   - 在“但是/然后/所以/不过”等连词前生成 `SemanticAnchor(confidence=0.3, source="conjunction_rule")`。
3. M3 替换策略：
   - M3 引入 LLM 后，`source` 切换为 `llm`，`confidence` 使用模型输出。

### 9.6 快慢流增量更新（M1 预埋）

#### 9.6.1 `window_id` 机制

每个 `CutDecision` 必须关联 `window_id`，用于：

1. 追溯当前决策来源；
2. 慢流返回后定位受影响窗口；
3. pause/resume 恢复时定位状态。

#### 9.6.2 受影响窗口判定

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

#### 9.6.3 增量更新流程

1. 慢流返回后，先计算受影响窗口集合；
2. 仅对受影响窗口重跑 `EvidenceFusion + SoftCutDecisionEngine`；
3. 合并结果回 `CutPlan` 并更新 trace；
4. 通过 SSE 推送 `subtitle.revised`。

### 9.7 关键子对象字段定义（M1 最小契约）

```python
from dataclasses import dataclass
from typing import Literal, Optional


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


@dataclass
class PauseAnchor:
    time: float
    duration: float
    confidence: float
    source: str  # pause_detector / vad_pause


@dataclass
class SemanticAnchor:
    time: float
    confidence: float
    source: str  # punct_proxy / conjunction_rule / llm


@dataclass
class PunctuationAnchor:
    time: float
    punctuation: str
    confidence: float
    source: str  # punctuation_model


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

字段约束：

1. `CutDecision.window_id` 必填，且必须可回溯到 `SpeakerChangeEvidence` 或窗口兜底来源。
2. `CutDecision.reason` 与 `risk` 需同步写入 trace，便于回放“为何切/为何未切”。
3. `SpeakerChangeEvidence.tags` 仅允许写入枚举化标签，避免自由文本污染统计口径。

## 10. 参数规划（M1 首批）

1. `speaker_change_min_score`
2. `soft_cut_window_sec`
3. `anchor_pause_min_sec`
4. `singleton_merge_max_duration_sec`
5. `deferred_max_wait_sec`
6. `deferred_max_words`
7. `debounce_cooldown_ms`
8. `soft_cut_window_before_sec`
9. `soft_cut_window_after_sec`
10. `length_pressure_soft`
11. `length_pressure_hard`

### 10.1 参数依赖与调参顺序

参数依赖关系：

1. `speaker_change_min_score` 与 `debounce_cooldown_ms` 共同决定有效 speaker_change 数量与 high/mid/low 分布。
2. `soft_cut_window_before_sec` 与 `soft_cut_window_after_sec` 决定锚点搜索范围，`anchor_pause_min_sec` 决定可用停顿锚点数量。
3. `length_pressure_soft` 与 `length_pressure_hard` 决定 mid 级别在低锚点质量下是否提前切分。
4. `deferred_max_wait_sec` 与 `deferred_max_words` 共同决定 deferred 的解决率与强制切比例。
5. `singleton_merge_max_duration_sec` 决定兜底回并范围，影响孤句率与跨边界误并风险。
6. 兼容关系：若仅配置 `soft_cut_window_sec`，按 `before:after=5:3` 拆分为 `soft_cut_window_before_sec` 与 `soft_cut_window_after_sec`。

建议调参顺序：

1. 先调 `speaker_change_min_score`（先平衡漏切/错切）。
2. 再调 `debounce_cooldown_ms`（抑制抖动，保留真实快切换）。
3. 再调 `soft_cut_window_before_sec`、`soft_cut_window_after_sec`、`anchor_pause_min_sec`（提升落刀精度）。
4. 再调 `length_pressure_soft`、`length_pressure_hard`（平衡长句拖延与提前切分）。
5. 最后调 `deferred_max_wait_sec`、`deferred_max_words`、`singleton_merge_max_duration_sec`（优化兜底行为）。

## 11. 测试与验收

1. 回归维度：漏切、错切、单词句、句中硬切、跨 speaker 错并。
2. 指标目标：
  - `gt_cut_hit_rate` 提升。
  - `singleton_sentence_rate` 下降至 `<2%`。
  - `mid_sentence_wrong_split_rate` 连续任务集下降。
3. 行为断言：
  - 不允许句中硬切。
  - 高置信 speaker_change 不得被跨边界合并。
  - 所有 deferred 都有结果状态，禁止静默丢失。

## 12. 风险与回退

1. 风险：参数过于激进导致切分震荡。  
回退：降级为 `smart_patch`（仅高置信生效）。
2. 风险：deferred 堆积导致晚修订过多。  
回退：启用窗口上限强制策略。
3. 风险：Pyannote 不稳定导致证据噪声。  
回退：提升 `speaker_change_min_score`，增加停顿锚点权重。

## 13. 未来测试与评估完整流程（执行版）

### 13.1 测试目标

1. 验证 soft-cut 是否在“漏切下降”的同时控制“错切与孤句副作用”。
2. 验证 speaker_change 分级、deferred 生命周期、去抖冷却是否按预期工作。
3. 验证评测流程可长期复用，且不依赖“精确到毫秒”的人工切口。

### 13.2 标注数据规范（推荐）

1. 输入标注文件使用 SRT（人工可读、易维护）。
2. 每条字幕标注 speaker：`（S1）` / `（S2）`。
3. 在你确认是“说话人切换点”的条目末尾加 `//`。
4. 编号不做强约束，可使用 `11.5` 之类插入编号。
5. 若是重叠语音，标记 `（OVL）` 或 `[OVL]` 或 `#OVL#`，默认 `eval_include=false`。
6. 时间戳允许粗略估计，不要求严格毫秒精度。

### 13.3 数据转换步骤

1. 将标注 SRT 转为 `speaker_units`：
   - 每行保留 `start/end/speaker_sequence/units/cut_positions/eval_include`。
2. 可选同时生成 `speaker_changes`（用于切换评测）：
   - 自动输出 `change_id/time/from_speaker/to_speaker/tolerance_sec`。
3. 使用脚本：

```bash
python backend/scripts/export_speaker_marked_srt_to_jsonl.py ^
  --input-srt tests/test_en_1_speaker.srt ^
  --output-jsonl jobs/test_en_1_speaker.speaker_units.jsonl ^
  --output-changes-jsonl jobs/test_en_1_speaker.speaker_changes.jsonl ^
  --default-tolerance-sec 0.45
```

### 13.4 容差与过滤规则（宽松口径）

1. 切换命中采用“窗口命中”，不是“时间点完全相等”。
2. 默认容差：
   - `tolerance_sec = max(default_tolerance_sec, min(0.8, segment_duration * 0.12))`
   - `default_tolerance_sec` 建议起步 `0.45`，需要更宽松可用 `0.6~0.8`。
3. 排除规则：
   - `is_overlap=true` 的样本不计入主指标；
   - `UNKNOWN` 或无法确认说话人的样本不计入主指标。
4. 重叠样本单独输出观察报告，不参与主分。

### 13.5 执行流程（每次迭代固定执行）

1. 准备数据集：
   - `clean_set`（无重叠、说话人明确）
   - `overlap_set`（仅观察，不打主分）
2. 跑三种模式：
   - `dual_full`
   - `smart_patch`
   - `fast_only`（无 GPU 回退验证）
3. 收集产物：
   - 最终 SRT
   - `subtitle_speaker_links`（如可用）
   - soft-cut trace（`split_reason/split_risk/window_id`）
4. 评测计算：
   - 主指标在 `clean_set` 上计算；
   - 观察指标在 `overlap_set` 上记录。
5. 结果归档：
   - 保存参数快照、指标 JSON、关键失败案例。

### 13.6 指标体系（主指标 + 观察指标）

1. 主指标（必须达标）：
   - `change_hit_rate`（切换命中率）
   - `cross_speaker_miss_rate`（跨 speaker 漏切率）
   - `extra_cut_in_clean_scope`（干净样本内多切）
   - `singleton_sentence_rate`
   - `mid_sentence_wrong_split_rate`
2. 观察指标（不直接卡闸）：
   - `overlap_observation_hit_rate`
   - `deferred_resolved_rate`
   - `deferred_forced_rate`
   - `overflow_forced_count`

### 13.7 回归闸门（建议）

1. `change_hit_rate` 不低于上一稳定基线。
2. `cross_speaker_miss_rate` 连续两轮下降或持平。
3. `extra_cut_in_clean_scope` 不高于基线 `+10%`。
4. `singleton_sentence_rate` 维持 `<2%` 或不高于基线。
5. 任一 P0 违规（句中硬切）直接判失败。

### 13.8 失败排查顺序（固定）

1. 先看 P0 违规：是否句中硬切或跨高置信边界误并。
2. 再看 speaker_change 分级：阈值是否过高导致漏切。
3. 再看去抖参数：是否误杀快速对话。
4. 再看 deferred：是否堆积、是否大量 forced。
5. 最后看锚点权重：停顿/标点/语义权重是否失衡。

### 13.9 注意点（必须遵守）

1. 不把重叠语音样本混入主指标。
2. 不把 `UNKNOWN` 说话人样本用于主指标比较。
3. 不用严格毫秒命中评估粗标注切口。
4. 不跨版本比较未对齐参数的结果。
5. 每次评测必须保存参数快照，否则结果不可复现。
6. 文档中的指标定义、脚本参数、产物路径必须同步更新，禁止“代码已改、文档未改”。
