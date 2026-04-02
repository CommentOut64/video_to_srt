# Timeanchored 局部提交与覆盖率重构 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `quarantined/fatal -> safe_window_fallback(整窗一条)` 改造成 `partial commit + span fallback + emergency only fallback`，并补上重复短语窗口的覆盖率恢复能力，彻底消除本次双流样例中的超长未切分字幕。

**Architecture:** 以“最小闭环局部重构”为原则，只重构 `anchor_mount -> decision_ingress -> alignment_stage_service` 这一段失败语义，不推翻现有 `Decision -> OutputProjection -> CommitScope` 总线。实现顺序固定为三段：先切断整窗单条降级，再补 `prefix/suffix rescue` 与 span fallback，最后软化 solver 与重复短语消歧，并用 job 级回归锁死结果。

**Tech Stack:** Python 3.10+、FastAPI 后端、现有 dual_pipeline/timeanchored_alignment/textflow 架构、pytest、PowerShell 命令行。

---

## 0. 文件结构与职责边界

**核心代码**
- Create: `backend/app/services/timeanchored_alignment/window_recovery_contracts.py`
  - 定义 `WindowSpanDecision`、`SpanFallbackPlan`、`RecoveredWindowPlan` 等窗口局部恢复契约。
- Create: `backend/app/services/timeanchored_alignment/window_recovery_planner.py`
  - 读取 `AnchorMountStageResult + DecisionIngressPackage`，生成 `trusted/interpolated/fallback` spans。
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/contracts.py`
  - 为 `AnchoredTokenUnit` 增补 `token_index/char_start/char_end`，必要时补 span 级 provenance 字段。
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/decision_ingress_assembler.py`
  - 透传新的 token slice 坐标与 span 恢复元数据。
- Modify: `backend/app/services/textflow/decision_ingress_adapter.py`
  - 支持基于 span 的 ingress 子集构造，保证 `annotated_words / boundary / punctuation` 可按 span 裁剪。
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
  - 用 `WindowRecoveryPlanner` 接管 `quarantined/fatal` 提交路径；将 `safe_window_fallback` 改成 emergency only。
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/gap_rescue_aligner.py`
  - 扩展 `prefix/internal/suffix` gap rescue。
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/seed_discovery.py`
  - 增加局部位置签名与重复短词风险抑制。
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/ambiguity_cluster_builder.py`
  - 扩展重复 surface、边界与邻域冲突标签。
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/anchor_trust_evaluator.py`
  - 增加邻域一致性、对角线位置一致性、head/tail 风险评分。
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/chain_solver.py`
  - 保留硬边界，改固定 jump/corridor/distance 为窗口尺度归一的软惩罚。

**测试**
- Create: `backend/tests/timeanchored_alignment/test_window_recovery_planner.py`
- Create: `backend/tests/timeanchored_alignment/test_job_p20260329_201905_no_long_unsplit_subtitles.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_contracts.py`
- Modify: `backend/tests/textflow/test_decision_ingress_adapter.py`
- Modify: `backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py`
- Modify: `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`
- Modify: `backend/tests/timeanchored_alignment/test_gap_rescue_aligner.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_seed_discovery.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_trust_evaluator.py`
- Modify: `backend/tests/timeanchored_alignment/test_sparse_dp_chain_solver.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_service.py`

**文档同步**
- Modify: `llmdoc/architecture/dual-flow-finalization.md`
- Modify: `llmdoc/architecture/text-processing-layered-architecture.md`
- Modify: `llmdoc/architecture/preparation-language-adapter.md`

**固定约束**
- 不新增“另起一套输出层”的兼容分支。
- 不允许新的整窗文本兜底取代已存在的局部可靠 span。
- 所有新 trace/metric 都必须能区分 `trusted_span / interpolated_span / fallback_span / emergency_fallback`。
- commit 由 wgh 手动执行；计划里的“提交”步骤一律表示“等待 wgh 手动提交”。

---

## Chunk 1: 先锁契约与红灯门禁

### Task 1.1: 扩展 `AnchoredTokenUnit` 契约，为 span 级恢复补足切片坐标

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/contracts.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/decision_ingress_assembler.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_contracts.py`

- [ ] **Step 1: 写失败测试，要求 `AnchoredTokenUnit` 暴露 token/char 级坐标**

```python
def test_decision_ingress_token_units_expose_token_and_char_slice_coordinates() -> None:
    package = _build_decision_ingress_package()
    token = package.anchored_token_units[0]
    assert token.token_index == 0
    assert token.char_start == 0
    assert token.char_end > token.char_start
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_anchor_mount_contracts.py -k "char_slice or token_index" -v`
Expected: FAIL with missing field or assertion failure

- [ ] **Step 3: 最小实现契约扩展**

```python
@dataclass(frozen=True)
class AnchoredTokenUnit:
    token_index: int
    char_start: int
    char_end: int
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_anchor_mount_contracts.py -k "char_slice or token_index" -v`
Expected: PASS

- [ ] **Step 5: 等待 wgh 手动提交**

### Task 1.2: 新增窗口恢复契约测试，锁定 `trusted/interpolated/fallback` span 语义

**Files:**
- Create: `backend/app/services/timeanchored_alignment/window_recovery_contracts.py`
- Create: `backend/tests/timeanchored_alignment/test_window_recovery_planner.py`

- [ ] **Step 1: 写失败测试，定义最小恢复计划形状**

```python
def test_recovered_window_plan_exposes_span_level_routes() -> None:
    plan = RecoveredWindowPlan(
        spans=(
            WindowSpanDecision(span_id="s0", span_kind="trusted", route="decision"),
            WindowSpanDecision(span_id="s1", span_kind="fallback", route="span_fallback"),
        )
    )
    assert [item.span_kind for item in plan.spans] == ["trusted", "fallback"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_window_recovery_planner.py -k "span_level_routes" -v`
Expected: FAIL with import or contract error

- [ ] **Step 3: 实现最小契约文件**

```python
@dataclass(frozen=True)
class WindowSpanDecision:
    span_id: str
    span_kind: str
    route: str
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_window_recovery_planner.py -k "span_level_routes" -v`
Expected: PASS

- [ ] **Step 5: 等待 wgh 手动提交**

### Task 1.3: 红灯门禁，禁止 `fatal/quarantined` 默认直落整窗单条

**Files:**
- Modify: `backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py`
- Modify: `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`

- [ ] **Step 1: 写失败测试，要求 `fatal/quarantined` 优先走 partial commit 规划，而不是直接 safe fallback**

```python
def test_quarantined_window_prefers_partial_commit_over_safe_window_fallback() -> None:
    text_route, edge_route, final_route, error_code = service._resolve_anchor_mount_routes(...)
    assert final_route == "partial_commit"
    assert edge_route != "safe_window_fallback"
```

- [ ] **Step 2: 写失败测试，要求 emergency fallback 只在无可恢复 span 时触发**

```python
def test_emergency_fallback_requires_zero_recoverable_spans() -> None:
    assert service._should_use_safe_window_fallback(stage_result=stage_result) is False
```

- [ ] **Step 3: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py -k "partial_commit or emergency_fallback" -v`
Expected: FAIL

- [ ] **Step 4: 等待 wgh 手动提交（仅红灯）**

---

## Chunk 2: 接入 `WindowRecoveryPlanner`，先切断整窗单条

### Task 2.1: 先实现 planner 的最小分 span 规则

**Files:**
- Create: `backend/app/services/timeanchored_alignment/window_recovery_planner.py`
- Modify: `backend/tests/timeanchored_alignment/test_window_recovery_planner.py`

- [ ] **Step 1: 写失败测试，要求 unresolved 头尾被识别成 `fallback_span`，中间稳定锚点保留为 `trusted_span`**

```python
def test_window_recovery_planner_splits_prefix_suffix_fallback_and_middle_trusted() -> None:
    plan = planner.build(stage_result=_build_stage_result_with_prefix_suffix_gaps())
    assert [span.span_kind for span in plan.spans] == ["fallback", "trusted", "fallback"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_window_recovery_planner.py -k "prefix_suffix_fallback" -v`
Expected: FAIL

- [ ] **Step 3: 实现最小 planner**

```python
class WindowRecoveryPlanner:
    def build(self, *, stage_result):
        # 基于 mount_status/open_gap/timeline_validity 生成连续 spans
        return RecoveredWindowPlan(spans=tuple(spans))
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_window_recovery_planner.py -k "prefix_suffix_fallback" -v`
Expected: PASS

- [ ] **Step 5: 等待 wgh 手动提交**

### Task 2.2: Decision ingress 支持按 span 切片构造子包

**Files:**
- Modify: `backend/app/services/textflow/decision_ingress_adapter.py`
- Modify: `backend/tests/textflow/test_decision_ingress_adapter.py`

- [ ] **Step 1: 写失败测试，要求 adapter 可以只消费 span 内 token/boundary/punctuation**

```python
def test_decision_ingress_adapter_can_build_span_scoped_input() -> None:
    result = adapter.build(package=package, span_selector=(1, 3))
    assert [w.word for w in result.decision_input.annotated_words] == ["brave", "world"]
    assert all(item.split_idx < 2 for item in result.decision_input.canonical_candidate_boundaries)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/textflow/test_decision_ingress_adapter.py -k "span_scoped_input" -v`
Expected: FAIL

- [ ] **Step 3: 实现最小 span slice 逻辑**

```python
def build(self, *, package, span_selector=None, ...):
    token_units = _slice_token_units(package.anchored_token_units, span_selector)
    boundary_evidences = _slice_boundary_evidences(package.boundary_evidences, span_selector)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/textflow/test_decision_ingress_adapter.py -k "span_scoped_input" -v`
Expected: PASS

- [ ] **Step 5: 等待 wgh 手动提交**

### Task 2.3: `alignment_stage_service` 改成 partial commit 主路径

**Files:**
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- Modify: `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`
- Modify: `backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py`

- [ ] **Step 1: 写失败测试，要求 `fatal/quarantined` 窗口保留多条局部字幕**

```python
def test_alignment_stage_partial_commit_keeps_multiple_sentences_for_bad_window() -> None:
    asyncio.run(pipeline._run_alignment_stage(ctx))
    assert len(ctx.final_sentences) >= 2
    assert ctx.finalization_metrics["timeanchored_final_route"] == "partial_commit"
```

- [ ] **Step 2: 写失败测试，要求 emergency fallback 不再作为常规 edge route**

```python
def test_alignment_stage_no_longer_marks_quarantined_window_as_safe_window_fallback_by_default() -> None:
    assert ctx.finalization_metrics["timeanchored_edge_route"] != "safe_window_fallback"
```

- [ ] **Step 3: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_alignment_stage_routing.py -k "partial_commit or safe_window_fallback" -v`
Expected: FAIL

- [ ] **Step 4: 实现最小 integration**

```python
plan = self._window_recovery_planner.build(stage_result=stage_result, ctx=ctx)
if plan.has_recoverable_spans:
    final_sentences = self._commit_partial_window_result(...)
else:
    self._commit_safe_window_fallback_result(...)
```

- [ ] **Step 5: 运行目标测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_alignment_stage_routing.py -k "partial_commit or safe_window_fallback" -v`
Expected: PASS

- [ ] **Step 6: 等待 wgh 手动提交**

### Task 2.4: 实现 span fallback 最小切分，禁止 fallback span 重新压成单句

**Files:**
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- Modify: `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`

- [ ] **Step 1: 写失败测试，要求 fallback span 复用 punctuation/boundary 产出多条结果**

```python
def test_fallback_span_uses_local_boundaries_instead_of_single_window_sentence() -> None:
    result = service._build_fallback_span_sentences(...)
    assert len(result) == 2
    assert result[0].end <= result[1].start
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_alignment_stage_routing.py -k "fallback_span_uses_local_boundaries" -v`
Expected: FAIL

- [ ] **Step 3: 实现最小局部 fallback**

```python
def _build_fallback_span_sentences(...):
    # 基于 punctuation_facts + boundary_evidences + char slice 切 text
    return sentences
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_alignment_stage_routing.py -k "fallback_span_uses_local_boundaries" -v`
Expected: PASS

- [ ] **Step 5: 等待 wgh 手动提交**

---

## Chunk 3: 补 Anchor 覆盖率恢复

### Task 3.1: `GapRescueAligner` 扩展 prefix/suffix gap

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/gap_rescue_aligner.py`
- Modify: `backend/tests/timeanchored_alignment/test_gap_rescue_aligner.py`

- [ ] **Step 1: 写失败测试，要求只有单个 anchor island 时也能生成 head/tail open gap**

```python
def test_gap_rescue_aligner_builds_prefix_and_suffix_gap_for_single_island() -> None:
    state = aligner.run(state=_build_state_with_single_middle_island())
    gap_ids = [gap.gap_id for gap in state.open_gaps]
    assert gap_ids == ["prefix-gap-0", "suffix-gap-0"]
```

- [ ] **Step 2: 写失败测试，要求 prefix/suffix rescue 不再受“双边唯一 island”限制**

```python
def test_gap_rescue_aligner_can_rescue_prefix_gap_with_local_best_match() -> None:
    state = aligner.run(state=_build_state_with_prefix_duplicate_hooks())
    assert any(match.anchor_kind in {"exact", "normalized"} for match in state.open_gaps[0].rescue_matches)
```

- [ ] **Step 3: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_gap_rescue_aligner.py -k "prefix or suffix" -v`
Expected: FAIL

- [ ] **Step 4: 实现最小 prefix/suffix rescue**

```python
if len(islands) == 1:
    gaps.extend(self._build_edge_gaps(...))
```

- [ ] **Step 5: 运行测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_gap_rescue_aligner.py -k "prefix or suffix" -v`
Expected: PASS

- [ ] **Step 6: 等待 wgh 手动提交**

### Task 3.2: Seed/ambiguity/trust 提升重复短语窗口区分能力

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/seed_discovery.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/ambiguity_cluster_builder.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/anchor_trust_evaluator.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_seed_discovery.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_trust_evaluator.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_disambiguation.py`

- [ ] **Step 1: 写失败测试，要求重复短词候选包含局部位置/邻域区分信号**

```python
def test_seed_discovery_tags_duplicate_short_tokens_with_position_signature() -> None:
    candidates = SeedDiscovery().discover(input_view=_build_duplicate_dialogue_input())
    assert any("position_signature" in candidate.candidate_id for candidate in candidates)
```

- [ ] **Step 2: 写失败测试，要求 trust 评分优先选择邻域一致的候选**

```python
def test_anchor_trust_evaluator_prefers_diagonal_consistent_candidate_in_duplicate_cluster() -> None:
    assert report_primary.trust_score > report_off_diagonal.trust_score
```

- [ ] **Step 3: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_anchor_mount_seed_discovery.py backend/tests/timeanchored_alignment/test_anchor_trust_evaluator.py backend/tests/timeanchored_alignment/test_anchor_disambiguation.py -k "duplicate or diagonal" -v`
Expected: FAIL

- [ ] **Step 4: 实现最小增强**

```python
duplicate_penalty = ...
diagonal_consistency = ...
neighbor_consistency = ...
```

- [ ] **Step 5: 运行测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_anchor_mount_seed_discovery.py backend/tests/timeanchored_alignment/test_anchor_trust_evaluator.py backend/tests/timeanchored_alignment/test_anchor_disambiguation.py -k "duplicate or diagonal" -v`
Expected: PASS

- [ ] **Step 6: 等待 wgh 手动提交**

### Task 3.3: `ChainSolver` 改成硬边界 + 软惩罚

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/chain_solver.py`
- Modify: `backend/tests/timeanchored_alignment/test_sparse_dp_chain_solver.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_service.py`

- [ ] **Step 1: 写失败测试，要求弱偏移 block 不再被硬阈值整体裁掉，但仍保持单调**

```python
def test_chain_solver_keeps_monotonic_near_diagonal_blocks_with_soft_penalty() -> None:
    result = ChainSolver().solve(input_view=input_view, blocks=blocks)
    assert result.committed_blocks
    assert result.unresolved_unit_indices != (0, 1, 2, 3)
```

- [ ] **Step 2: 写失败测试，要求跨 speaker/turn 与跨硬句界仍然是硬拒绝**

```python
def test_chain_solver_still_rejects_cross_speaker_boundary_block() -> None:
    assert offending_block not in result.committed_blocks
```

- [ ] **Step 3: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_sparse_dp_chain_solver.py backend/tests/timeanchored_alignment/test_anchor_mount_service.py -k "soft_penalty or cross_speaker_boundary" -v`
Expected: FAIL

- [ ] **Step 4: 实现最小 soft-penalty solver**

```python
compatibility_cost = diagonal_penalty + distance_penalty
if crosses_hard_boundary:
    return False
```

- [ ] **Step 5: 运行测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_sparse_dp_chain_solver.py backend/tests/timeanchored_alignment/test_anchor_mount_service.py -k "soft_penalty or cross_speaker_boundary" -v`
Expected: PASS

- [ ] **Step 6: 等待 wgh 手动提交**

---

## Chunk 4: 锁 job 回归、指标与文档同步

### Task 4.1: 为本次任务新增 job 级回归，锁死“超长字幕清零”

**Files:**
- Create: `backend/tests/timeanchored_alignment/test_job_p20260329_201905_no_long_unsplit_subtitles.py`

- [ ] **Step 1: 写回归测试，直接针对本次任务目录断言无超长单条字幕**

```python
def test_job_p20260329_201905_has_no_long_unsplit_subtitles() -> None:
    snapshot = run_job_fixture("p-20260329-201905-tr-test-en-1-pxac")
    assert max(item.duration for item in snapshot.sentences) < 12.0
    assert max(item.char_count for item in snapshot.sentences) < 160
```

- [ ] **Step 2: 写回归测试，要求 fallback 结果最多为 span 级，不得整窗一条**

```python
def test_job_p20260329_201905_uses_partial_commit_not_window_safe_fallback() -> None:
    assert "safe_fallback:0" not in snapshot.segment_ids
```

- [ ] **Step 3: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_job_p20260329_201905_no_long_unsplit_subtitles.py -v`
Expected: FAIL

- [ ] **Step 4: 在完成 Chunk 2-3 实现后重跑并转绿**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_job_p20260329_201905_no_long_unsplit_subtitles.py -v`
Expected: PASS

- [ ] **Step 5: 等待 wgh 手动提交**

### Task 4.2: 补 observability 指标，方便以后排查同类问题

**Files:**
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- Modify: `backend/tests/timeanchored_alignment/test_alignment_stage_trace_payloads.py`

- [ ] **Step 1: 写失败测试，要求 trace/metrics 暴露 span 级恢复统计**

```python
def test_alignment_stage_trace_includes_window_recovery_metrics() -> None:
    summary = service._build_decision_ingress_trace_summary(...)
    assert "trusted_span_count" in summary["metrics"]
    assert "fallback_span_count" in summary["metrics"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_alignment_stage_trace_payloads.py -k "window_recovery_metrics" -v`
Expected: FAIL

- [ ] **Step 3: 实现最小指标回写**

```python
ctx.finalization_metrics["timeanchored_trusted_span_count"] = ...
ctx.finalization_metrics["timeanchored_fallback_span_count"] = ...
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_alignment_stage_trace_payloads.py -k "window_recovery_metrics" -v`
Expected: PASS

- [ ] **Step 5: 等待 wgh 手动提交**

### Task 4.3: 同步 `llmdoc` 当前实现文档

**Files:**
- Modify: `llmdoc/architecture/dual-flow-finalization.md`
- Modify: `llmdoc/architecture/text-processing-layered-architecture.md`
- Modify: `llmdoc/architecture/preparation-language-adapter.md`

- [ ] **Step 1: 重写过时段落，删除“`quarantined/fatal` 直接整窗 safe fallback”旧叙述**

```markdown
旧：窗口失败后直接提交 safe_window_fallback 单句
新：窗口失败进入 WindowRecoveryPlanner，优先 partial commit，emergency fallback 仅在零 recoverable span 时启用
```

- [ ] **Step 2: 增补 span recovery、prefix/suffix rescue、指标字段的当前实现说明**

```markdown
- trusted/interpolated/fallback span 语义
- span-scoped decision ingress
- emergency_window_fallback 触发前提
```

- [ ] **Step 3: 自查文档与代码状态一致，不追加“补丁式说明”**

Run: `rg -n "safe_window_fallback|partial commit|WindowRecoveryPlanner" llmdoc/architecture`
Expected: only current-state wording remains

- [ ] **Step 4: 等待 wgh 手动提交**

---

## 5. 推荐执行顺序

1. 先做 Chunk 1，把红灯测试和契约锁住。
2. 立即做 Chunk 2，把“整窗一条”主故障形态切断。
3. 再做 Chunk 3，提高覆盖率，避免 partial commit 过度依赖 fallback span。
4. 最后做 Chunk 4，锁 job 回归并同步架构文档。

## 6. 完整验证命令

- [ ] **Step 1: 跑本次改动的核心单测集合**

Run:

```powershell
$env:PYTHONPATH='backend'
python -m pytest `
  backend/tests/timeanchored_alignment/test_anchor_mount_contracts.py `
  backend/tests/timeanchored_alignment/test_window_recovery_planner.py `
  backend/tests/textflow/test_decision_ingress_adapter.py `
  backend/tests/timeanchored_alignment/test_alignment_stage_routing.py `
  backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py `
  backend/tests/timeanchored_alignment/test_gap_rescue_aligner.py `
  backend/tests/timeanchored_alignment/test_anchor_mount_seed_discovery.py `
  backend/tests/timeanchored_alignment/test_anchor_trust_evaluator.py `
  backend/tests/timeanchored_alignment/test_sparse_dp_chain_solver.py `
  backend/tests/timeanchored_alignment/test_anchor_mount_service.py `
  backend/tests/timeanchored_alignment/test_alignment_stage_trace_payloads.py `
  backend/tests/timeanchored_alignment/test_job_p20260329_201905_no_long_unsplit_subtitles.py -v
```

Expected: PASS

- [ ] **Step 2: 跑现有相邻回归，确认没有打坏正常窗**

Run:

```powershell
$env:PYTHONPATH='backend'
python -m pytest `
  backend/tests/timeanchored_alignment/test_job_p20260328_193846_no_duplicate.py `
  backend/tests/timeanchored_alignment/test_job_p20260328_202014_no_timestamp_chaos.py `
  backend/tests/timeanchored_alignment/test_output_projection.py `
  backend/tests/timeanchored_alignment/test_output_integration.py -v
```

Expected: PASS

- [ ] **Step 3: 如仓库已有格式化/静态检查入口，则补跑**

Run: `python -m pytest --collect-only backend/tests/timeanchored_alignment > $null`
Expected: exit code 0

- [ ] **Step 4: 等待 wgh 手动提交最终结果**

## 7. 完成判定

只有同时满足以下条件，才能向 wgh 声称“修复完成”：

1. 本次样例任务不再出现十几秒以上且数百字符的单条字幕。
2. `safe_window_fallback` 已降为 emergency only。
3. `fatal/quarantined` 窗口可以产生多条局部字幕或显式 emergency，不再默认整窗一条。
4. 正常窗口回归仍通过。
5. `llmdoc` 已同步为当前真实实现。
