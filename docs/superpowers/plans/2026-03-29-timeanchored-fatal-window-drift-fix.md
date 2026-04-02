# Timeanchored Fatal Window Drift Fix Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `fatal/quarantined` timeanchored window 被错误整窗提交后导致的大面积字幕时间漂移问题，确保 `partial_commit` 真正只提交单调可靠的 recoverable spans。

**Architecture:** 以最小闭环方式修复 `window_recovery_plan -> partial_commit -> decision ingress slice -> output merge` 这一条链，不改 AnchorMount 求解器主体，不新增第二套输出总线。先锁 planner 的硬门禁，再接通 span 级提交，最后补合并门禁、trace 与文档回归。

**Tech Stack:** Python 3.10+、FastAPI 后端、dual_pipeline/timeanchored_alignment/textflow、pytest、PowerShell。

---

## 0. 文件结构与职责边界

**核心代码**
- Modify: `backend/app/services/timeanchored_alignment/window_recovery_contracts.py`
  - 为 `RecoveredWindowPlan` 增补 window 级硬故障语义与 planner 输出所需的原因字段。
- Modify: `backend/app/services/timeanchored_alignment/window_recovery_planner.py`
  - 从“仅按 mount_status 判 recoverable”升级为“结合 monotonic_violation / span backtrack / 邻接冲突”的真实恢复规划器。
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
  - 新增 `partial_commit` 的 span 级提交路径、局部 batch 合并与失败升级逻辑。
- Modify: `backend/app/services/textflow/decision_ingress_adapter.py`
  - 补强 span 级 trace/compat 元信息，确保 partial commit 可观测、可审计。

**测试**
- Modify: `backend/tests/timeanchored_alignment/test_window_recovery_planner.py`
- Modify: `backend/tests/textflow/test_decision_ingress_adapter.py`
- Modify: `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`
- Modify: `backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py`
- Create: `backend/tests/timeanchored_alignment/test_alignment_stage_partial_commit_monotonicity.py`

**文档同步**
- Modify: `llmdoc/architecture/dual-flow-finalization.md`
- Modify: `llmdoc/architecture/preparation-language-adapter.md`

**固定约束**
- 不允许新增新的 owner-only fallback 提交通道。
- 不允许把 `timestamp_backtrack_fix` 当作 partial commit 的合法输入修复手段。
- 只要 partial commit 合并结果不单调，必须升级为 `safe_window_fallback`。
- commit 由 wgh 手动执行；计划中的提交步骤一律表示“等待 wgh 手动提交”。

---

## Chunk 1: 锁死 planner 的硬门禁

### Task 1.1: 为 `fatal/quarantined + monotonic_violation` 建立红灯测试

**Files:**
- Modify: `backend/tests/timeanchored_alignment/test_window_recovery_planner.py`

- [ ] **Step 1: 写失败测试，要求 monotonic_violation 不能仅因中间 trusted island 存在就放行 recoverable**

```python
def test_window_recovery_plan_does_not_treat_monotonic_violation_island_as_recoverable() -> None:
    plan = planner.build(stage_result=_build_stage_result_with_backtracking_trusted_island())
    assert plan.has_recoverable_spans is False
    assert plan.emergency_fallback_only is True
```

- [ ] **Step 2: 写失败测试，要求 span 内时间回退必须降级为 fallback**

```python
def test_window_recovery_plan_demotes_backtracking_span_to_fallback() -> None:
    plan = planner.build(stage_result=_build_stage_result_with_span_backtrack())
    assert [span.span_kind for span in plan.spans] == ["fallback", "fallback", "fallback"]
```

- [ ] **Step 3: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_window_recovery_planner.py -k "monotonic_violation or backtracking_span" -v`
Expected: FAIL，当前 planner 仍返回 `has_recoverable_spans=True`

- [ ] **Step 4: 等待 wgh 手动提交（红灯阶段）**

### Task 1.2: 实现 planner 的 window/span 级硬门禁

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/window_recovery_contracts.py`
- Modify: `backend/app/services/timeanchored_alignment/window_recovery_planner.py`

- [ ] **Step 1: 扩展恢复契约，补齐 emergency only 与原因字段**

```python
@dataclass(frozen=True)
class RecoveredWindowPlan:
    spans: tuple[WindowSpanDecision, ...] = field(default_factory=tuple)
    emergency_fallback_only: bool = False
    reason_codes: tuple[str, ...] = field(default_factory=tuple)
```

- [ ] **Step 2: 实现最小 planner 规则**

```python
def build(self, *, stage_result: object) -> RecoveredWindowPlan:
    timeline_validity = _resolve_timeline_validity(stage_result)
    reasons = _resolve_timeline_reasons(stage_result)
    spans = _build_spans_from_tokens(...)
    spans = _demote_backtracking_spans(spans)
    if "monotonic_violation" in reasons and not _spans_form_monotonic_commit(spans):
        return RecoveredWindowPlan(spans=tuple(spans), emergency_fallback_only=True, reason_codes=("monotonic_violation",))
    return RecoveredWindowPlan(spans=tuple(spans), reason_codes=tuple(reasons))
```

- [ ] **Step 3: 运行目标测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_window_recovery_planner.py -k "monotonic_violation or backtracking_span" -v`
Expected: PASS

- [ ] **Step 4: 运行同文件全量测试**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_window_recovery_planner.py -v`
Expected: PASS

- [ ] **Step 5: 等待 wgh 手动提交**

---

## Chunk 2: 接通真正的 partial commit

### Task 2.1: 锁定 `partial_commit` 必须传 `span_selector`

**Files:**
- Modify: `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`
- Modify: `backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py`
- Modify: `backend/tests/textflow/test_decision_ingress_adapter.py`

- [ ] **Step 1: 写失败测试，要求 partial commit 调用 `DecisionIngressAdapter.build(..., span_selector=...)`**

```python
def test_partial_commit_passes_span_selector_to_decision_ingress_adapter(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(service._decision_ingress_adapter, "build", lambda **kwargs: calls.append(kwargs) or _build_adapter_result())
    service._commit_timeanchored_main_chain_result(...)
    assert any(call.get("span_selector") == (9, 24) for call in calls)
```

- [ ] **Step 2: 写失败测试，要求 emergency only window 不得走 partial commit**

```python
def test_emergency_fallback_window_skips_partial_commit(monkeypatch) -> None:
    assert service._resolve_anchor_mount_safe_window_fallback_reason(stage_result=stage_result) == "anchor_mount_fatal"
```

- [ ] **Step 3: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_alignment_stage_routing.py -k "span_selector or emergency_fallback" -v`
Expected: FAIL

- [ ] **Step 4: 等待 wgh 手动提交（红灯阶段）**

### Task 2.2: 在 `alignment_stage_service` 中新增 span 级 partial commit

**Files:**
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`

- [ ] **Step 1: 提取 span 级提交辅助函数**

```python
def _build_partial_commit_spans(self, *, stage_result: AnchorMountStageResult) -> tuple[WindowSpanDecision, ...]:
    plan = getattr(stage_result, "window_recovery_plan", None)
    if plan is None or getattr(plan, "emergency_fallback_only", False):
        return tuple()
    return tuple(span for span in plan.spans if span.route == "decision")
```

- [ ] **Step 2: 为每个 recoverable span 单独构造 decision ingress**

```python
for span in partial_spans:
    adapter_result = self._decision_ingress_adapter.build(
        package=stage_result.decision_ingress,
        span_selector=(span.token_start, span.token_end),
        speaker_id=speaker_id,
        turn_id=turn_id,
    )
```

- [ ] **Step 3: 对每个 span 独立跑 Decision，收集局部 `subtitle_batch`**

```python
decision_output = host._decision_processor.process(...)
partial_batches.append(decision_output.subtitle_batch)
partial_sentences.extend(_finalize_partial_sentences(...))
```

- [ ] **Step 4: 运行目标测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_alignment_stage_routing.py -k "span_selector or emergency_fallback" -v`
Expected: PASS

- [ ] **Step 5: 等待 wgh 手动提交**

### Task 2.3: 合并 partial batches，并在不单调时升级 fallback

**Files:**
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- Create: `backend/tests/timeanchored_alignment/test_alignment_stage_partial_commit_monotonicity.py`

- [ ] **Step 1: 写失败测试，要求 partial 合并结果必须单调**

```python
def test_partial_commit_merge_rejects_overlapping_sentence_ranges() -> None:
    with pytest.raises(ValueError, match="partial_commit_non_monotonic"):
        service._merge_partial_commit_batches(partial_batches=_build_overlapping_batches())
```

- [ ] **Step 2: 写失败测试，要求不单调时升级成 safe window fallback**

```python
def test_partial_commit_non_monotonic_upgrades_to_safe_window_fallback(monkeypatch) -> None:
    service._commit_timeanchored_main_chain_result(...)
    assert safe_fallback_mock.called is True
```

- [ ] **Step 3: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_alignment_stage_partial_commit_monotonicity.py -v`
Expected: FAIL

- [ ] **Step 4: 实现局部 batch 合并与门禁**

```python
def _merge_partial_commit_batches(...):
    items = sorted(all_items, key=lambda item: (float(item.start), float(item.end), str(item.segment_id)))
    _assert_monotonic_items(items)
    return owner_batch
```

- [ ] **Step 5: 运行目标测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_alignment_stage_partial_commit_monotonicity.py -v`
Expected: PASS

- [ ] **Step 6: 等待 wgh 手动提交**

---

## Chunk 3: 补 trace、文档与回归门禁

### Task 3.1: 补 span 级 trace 与 compat 元数据

**Files:**
- Modify: `backend/app/services/textflow/decision_ingress_adapter.py`
- Modify: `backend/tests/textflow/test_decision_ingress_adapter.py`

- [ ] **Step 1: 写失败测试，要求 span selector 出现在 compat_report 中**

```python
def test_decision_ingress_adapter_reports_span_selector_in_compat_report() -> None:
    result = adapter.build(package=package, span_selector=(9, 24))
    assert result.compat_report["span_selector"] == [9, 24]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/textflow/test_decision_ingress_adapter.py -k "span_selector_in_compat_report" -v`
Expected: FAIL

- [ ] **Step 3: 实现最小 trace 补强**

```python
compat_report["span_selector"] = list(span_selector) if span_selector is not None else None
```

- [ ] **Step 4: 运行目标测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/textflow/test_decision_ingress_adapter.py -k "span_selector_in_compat_report" -v`
Expected: PASS

- [ ] **Step 5: 等待 wgh 手动提交**

### Task 3.2: 文档同步为当前真实行为

**Files:**
- Modify: `llmdoc/architecture/dual-flow-finalization.md`
- Modify: `llmdoc/architecture/preparation-language-adapter.md`

- [ ] **Step 1: 重写 `partial_commit` 语义段**

```md
- `partial_commit` 不再只是路由标签；提交层必须按 `window_recovery_plan.spans` 构造 span-scoped decision ingress。
- 含 `monotonic_violation` 的窗口，只有单调 recoverable spans 才允许继续提交；否则升级为 `safe_window_fallback`。
```

- [ ] **Step 2: 重写 `safe_window_fallback` 触发条件段**

```md
- `safe_window_fallback` 仅在 `emergency_fallback_only` 或 partial merge 非单调时触发。
```

- [ ] **Step 3: 人工检查文档整体连贯**

Run: `rg -n "partial_commit|safe_window_fallback|monotonic_violation" llmdoc/architecture/dual-flow-finalization.md llmdoc/architecture/preparation-language-adapter.md`
Expected: 能看到旧说法已被新说法整体替换，无“追加补丁式说明”

- [ ] **Step 4: 等待 wgh 手动提交**

### Task 3.3: 跑最小回归集，锁死这次问题

**Files:**
- Test only

- [ ] **Step 1: 跑 planner / ingress / routing / monotonicity 相关测试**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_window_recovery_planner.py backend/tests/textflow/test_decision_ingress_adapter.py backend/tests/timeanchored_alignment/test_alignment_stage_routing.py backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py backend/tests/timeanchored_alignment/test_alignment_stage_partial_commit_monotonicity.py -q`
Expected: PASS

- [ ] **Step 2: 补跑 timeanchored 对齐阶段回归**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/test_async_dual_pipeline_alignment_stage.py -q`
Expected: PASS

- [ ] **Step 3: 如条件允许，复跑问题 job 做人工验收**

Run: `python -m backend.tools.<按仓库实际复跑入口填写>`
Expected: `fatal/quarantined` 窗口不再整窗进入 Decision，trace 中可看到 span-scoped partial commit

- [ ] **Step 4: 等待 wgh 手动提交**

---

## 执行备注

- 若 Chunk 1 完成后发现大量现网坏窗口都会被直接判成 `emergency_fallback_only`，先不要继续扩大到 AnchorMount 覆盖率重构；本计划只解决“错误 partial 放行”和“整窗病提交”。
- 若 Chunk 2 中发现单 span Decision 输出仍出现 `timestamp_backtrack_fix_count > 0`，应立即停止继续堆逻辑，回到 planner 继续收紧 span 选择，而不是在 Decision 后补丁修时间。
- 本计划故意不包含 solver/seed/gap rescue 优化；那是下一阶段的覆盖率修复，不与这次漂移止血混做一单。
