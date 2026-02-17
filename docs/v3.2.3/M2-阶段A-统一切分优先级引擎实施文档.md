# M2 阶段A：统一切分优先级引擎实施文档

> Type: Architecture | Status: Active
> Version: V3.2.0+dev.20260217.01
> Scope: `speaker/pause/punctuation/semantic/llm` 多来源切分优先级统一与可配置化

## 1. Summary

* **Goal**: 将现有“speaker 驱动开窗 + 标点补充”的局部逻辑升级为统一优先级引擎，使切分来源可按配置快速调权、切换和下线（含未来 LLM 接入）。
* **关键约束**:
  - `speaker` 不是硬边界（避免 pyannote 误判和抖动放大）。
  - `speaker` 在软决策中默认最高权重。
  - 切分优先级必须“仅改配置即可调整”，不得依赖分支代码重写。

## 2. Diagram

* `AnnotatedWord + speaker_change_facts + anchors` -> `PriorityEvidenceBuilder` -> `PriorityFusion` -> `SoftCutDecisionEngine` -> `CutPlan`
* `runtime.soft_cut.priority.profile` -> `PriorityPolicy` -> `score/rank/gate` -> `split decisions`
* `Profile 切换`：`punct_boost` -> `llm_ramp` -> `llm_primary_no_punct`

## 3. Key Components

* `backend/app/services/segmentation/soft_cut/evidence_builder.py`: 证据构建与开窗策略（现状仅 speaker 驱动，需扩展为多来源可触发）。
* `backend/app/services/segmentation/soft_cut/evidence_fusion.py`: 多来源锚点融合评分（需接入来源权重策略）。
* `backend/app/services/segmentation/soft_cut/decision_engine.py`: 决策引擎（需从固定 reason 改为来源驱动 reason/risk）。
* `backend/app/pipelines/dual_pipeline/implementation.py`: L5->L6 证据构建与 CutPlan 组装入口（需接入 profile 与任务级开关透传）。
* `backend/app/services/textflow/decision_layer.py`: CutPlan 执行与回退路径（需保证回退链保留标点能力，不吞标点切分）。
* `backend/app/services/text_pipeline_config.py`: `segmentation.soft_cut` 参数入口（需新增 priority 配置结构）。
* `backend/app/services/model_runtime_config_service.py`: runtime 默认值与覆盖来源（需新增 priority profile 配置）。

## 4. 现状缺口（必须先修）

1. `soft-cut` 开窗当前依赖 `speaker_change_facts`，无 speaker 变化时直接返回 `no_cut_windows`，导致标点/语义证据不能独立触发切分。
2. 标点锚点存在但基础分低，且只能作为窗口内候选，无法在“无 speaker 窗口”场景生效。
3. `decision_engine` 的 `split_reason` 仅返回 `pause/speaker_change`，无法表达 `punctuation/semantic/llm`。
4. L6 回退 `FinalSplitter` 时仅传 `words`，未统一透传 `clean_text/punctuation_positions`，在注入受阻场景可能弱化标点切分。

## 5. 设计目标（阶段A）

1. 提供统一优先级协议：`enabled_sources + weights + min_confidence + trigger_threshold + tiebreak_order`。
2. 允许非 speaker 来源触发开窗（标点、语义、未来 LLM）。
3. 保持 `speaker` 软边界属性，但默认最高权重。
4. 支持三类无代码迁移：
  - 仅提高标点优先级。
  - LLM 上线后降低标点、提高 LLM。
  - 直接关闭标点来源并稳定运行。

## 6. 统一契约（新增）

### 6.1 切分来源枚举

```python
class SplitEvidenceSource(str, Enum):
    SPEAKER = "speaker"
    PAUSE = "pause"
    PUNCTUATION = "punctuation"
    SEMANTIC = "semantic"
    LLM = "llm"
    FORCE = "force"
```

### 6.2 统一候选结构

```python
@dataclass
class SplitEvidence:
    source: SplitEvidenceSource
    anchor_time: float
    confidence: float
    base_score: float
    final_score: float
    reason: str
    risk: str
    metadata: dict[str, Any]
```

说明：
1. `final_score` 由策略引擎统一计算，来源侧只提供原始信息。
2. `reason/risk` 由来源模板生成，禁止在决策层写死 `if anchor_type == ...`。

### 6.3 优先级配置结构

```python
@dataclass
class SourcePriorityRule:
    enabled: bool
    weight: float
    min_confidence: float
    trigger_threshold: float

@dataclass
class SoftCutPriorityProfile:
    profile_name: str
    tiebreak_order: list[str]
    merge_window_ms: int
    source_rules: dict[str, SourcePriorityRule]
```

## 7. 决策算法（阶段A统一版）

1. 候选收集：收集 `speaker/pause/punctuation/semantic`（预留 `llm`）。
2. 来源门控：过滤 `enabled=false` 或 `confidence < min_confidence` 的候选。
3. 开窗触发：任一来源 `confidence * weight >= trigger_threshold` 均可触发窗口。
4. 评分计算：`score = weight * confidence * stability_factor * boundary_quality_factor`。
5. 冲突收敛：在 `merge_window_ms` 时间窗内做 NMS，仅保留最高分候选。
6. 同分裁决：按 `tiebreak_order` 决定来源优先级。
7. 护栏处理：应用 `min_segment_duration/max_segment_duration/min_tokens`，避免碎句。
8. 生成输出：输出 `CutDecision.reason/risk/source/window_id`，保证可追溯。

## 8. 默认优先级分配（阶段A落地值）

说明：`speaker` 作为软边界但高权重，不做硬切。

1. `speaker`：`weight=0.85`，`min_confidence=0.45`，`trigger_threshold=0.42`
2. `pause`：`weight=0.55`，`min_confidence=0.35`，`trigger_threshold=0.30`
3. `semantic`：`weight=0.45`，`min_confidence=0.30`，`trigger_threshold=0.24`
4. `punctuation`（当前过渡期）：`weight=0.65`，`min_confidence=0.35`，`trigger_threshold=0.28`
5. `llm`（预留，阶段A可关闭）：`weight=0.00`，`enabled=false`

默认同分顺序：`speaker > punctuation > pause > semantic > llm`。

## 9. 可快速切换的 profile 设计

### 9.1 `punct_boost_transition`（当前推荐）

1. 目标：先修复“标点切分被忽略”问题。
2. 配置特征：`speaker` 高权重，`punctuation` 次高并可独立触发窗口。

### 9.2 `llm_ramp_up`（LLM 灰度期）

1. 目标：逐步把语义主导权交给 LLM。
2. 配置特征：提高 `llm`，下调 `punctuation`，保留 `speaker` 高权重。

### 9.3 `llm_primary_no_punct`（LLM 稳定期）

1. 目标：移除标点主链依赖。
2. 配置特征：`punctuation.enabled=false`，其余来源仍可稳定运行。

## 10. 代码改造清单（按文件）

1. `backend/app/services/segmentation/soft_cut/types.py`
  - 新增 `SplitEvidenceSource` 与 `SplitEvidence`。
  - `CutDecision` 增加 `source` 字段，保留旧字段兼容。
2. `backend/app/services/segmentation/soft_cut/evidence_builder.py`
  - 扩展为“多来源触发窗口”，不再仅依赖 speaker。
  - 新增来源级去抖与候选过滤。
3. `backend/app/services/segmentation/soft_cut/evidence_fusion.py`
  - 引入 `SoftCutPriorityProfile`，按 profile 统一评分。
  - 保留现有融合结果结构，增加 `source_score_report`。
4. `backend/app/services/segmentation/soft_cut/decision_engine.py`
  - 把 `_resolve_anchor_reason` 改为来源映射表驱动。
  - reason 统一输出：`speaker_change/pause/punctuation/semantic/llm/forced_guard`。
5. `backend/app/pipelines/dual_pipeline/implementation.py`
  - `_build_fused_evidence_for_l6` 接入 profile。
  - 无 speaker 时仍允许由 punctuation/semantic 触发窗口。
6. `backend/app/services/textflow/decision_layer.py`
  - `FinalSplitter.split(...)` 回退调用补齐 `clean_text/punctuation_positions` 透传路径。
7. `backend/app/services/text_pipeline_config.py`
  - 新增 `segmentation.soft_cut.priority.*` 配置读取（支持 dotted/nested）。
8. `backend/app/services/model_runtime_config_service.py`
  - 新增 priority profile 默认值与有效键元数据。

## 11. Runtime 参数规范（阶段A新增）

建议统一归入 `runtime.segmentation`：

```yaml
segmentation:
  soft_cut.priority.active_profile: punct_boost_transition
  soft_cut.priority.merge_window_ms: 120
  soft_cut.priority.profile.punct_boost_transition.tiebreak_order: [speaker, punctuation, pause, semantic, llm]
  soft_cut.priority.profile.punct_boost_transition.source.speaker.enabled: true
  soft_cut.priority.profile.punct_boost_transition.source.speaker.weight: 0.85
  soft_cut.priority.profile.punct_boost_transition.source.punctuation.enabled: true
  soft_cut.priority.profile.punct_boost_transition.source.punctuation.weight: 0.65
```

要求：
1. 支持 profile 热切换（新任务生效）。
2. 缺省 profile 不存在时回退 `punct_boost_transition`。
3. 配置非法时降级到内置默认并写告警日志。

## 12. 测试计划（阶段A必须补齐）

1. 单测：`evidence_builder`
  - 无 speaker_change 时，标点可触发窗口。
  - `enabled_sources` 关闭后来源不再参与候选。
2. 单测：`evidence_fusion`
  - 不同 profile 下相同输入得到不同排序结果。
  - 同分时按 `tiebreak_order` 稳定决策。
3. 单测：`decision_engine`
  - `reason` 输出覆盖 `speaker/pause/punctuation/semantic`。
  - `source` 与 `reason` 一致性校验。
4. 集成测试：`dual_pipeline`
  - `speaker + punctuation` 冲突场景可解释、无无因切分。
  - `no_speaker_change_facts` 场景仍可得到标点切分。
5. 回归测试：`decision_layer`
  - CutPlan 为空时，回退路径保留标点切分能力。

## 13. 验收标准（DoD）

1. 切分优先级可通过 profile 调整，无需修改核心流程代码。
2. `speaker` 保持软边界，不出现“一刀切硬边界”行为。
3. 标点优先级上调后，`no_speaker_change_facts` 场景分句率提升且误切不显著上升。
4. profile 切换到 `llm_primary_no_punct` 后，系统可在无标点来源下稳定运行。
5. 调试输出必须包含：`source/reason/risk/score/profile`。

## 14. 风险与回滚

1. 风险：标点提权过高导致过切。
  - 处置：下调 `punctuation.weight`，提高 `min_confidence`。
2. 风险：speaker 抖动导致候选密集。
  - 处置：提高 `merge_window_ms` 与 speaker 去抖阈值。
3. 风险：profile 配置缺失导致行为漂移。
  - 处置：启用内置默认 profile 并记录 `fallback_profile`。
4. 回滚开关：
  - `soft_cut.priority.active_profile` 回切旧配置。
  - `soft_cut.enable=false` 回退 `FinalSplitter` 默认路径。

## 15. 与后续阶段关系

1. 阶段B（任务级 speaker 参数）依赖本阶段的 profile 机制做任务级覆盖。
2. 阶段C（`speaker_count=0` 自动）仅改变 speaker 证据输入，不改变优先级引擎接口。
3. 阶段D（7->4 命名治理）仅改名和诊断键，不改变阶段A契约语义。
