# Timeanchored Segmentation Root-Cause Repair Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 彻底消除 timeanchored 主链里“hook 变相驱动切分 + Decision 边界抢切”导致的英文乱切分，不依赖 Decision 后兜底或输出层补丁。

**Architecture:** `AnchorMount` 只输出稀疏、可解释、与 hook 解耦的对齐层专属边界证据；`Decision` 不再把 `gap/pause/duration` 直接升格为可执行切点，而是基于 token 时间流做滚动式 segment 规划，把 `hard_limit` 降级为约束。整条链路保持 `Window -> token-unit -> sparse boundary evidence -> planner` 的单一语义，不允许后置 fallback 修句。

**Tech Stack:** Python, pytest, timeanchored alignment, textflow decision layer, postprocess trace JSON, Windows PowerShell

---

## 0. 当前证据与定性

- `jobs/p-20260328-175350-tr-test-en-1-qvvs` 的 [40_decision.output.json](f:/video_to_srt_gpu/jobs/p-20260328-175350-tr-test-en-1-qvvs/debug/postprocess/chunk_0001/40_decision.output.json) 显示 `17.9s-23.0s` 被切成 `killinging five / strangers is probably / more beneficial / for society, so I'll do that`，驱动原因是连续 `gap_pause` 加一次 `hard_limit_forced`。
- 同一 chunk 的 [21_anchor_mount.output.json](f:/video_to_srt_gpu/jobs/p-20260328-175350-tr-test-en-1-qvvs/debug/postprocess/chunk_0001/21_anchor_mount.output.json) 显示 `anchored_count=43`、`unresolved_count=0`、`alignment_score=1.0`，说明这段坏例子不是“挂载失败”。
- 当前 [boundary_evidence_builder.py](f:/video_to_srt_gpu/backend/app/services/timeanchored_alignment/anchor_mount/boundary_evidence_builder.py) 里 `anchor_block_close` 仍由相邻 token 的 `source_hook_ids !=` 触发，这会把 hook 差异变相抬升成边界语义。
- 当前 [decision_layer.py](f:/video_to_srt_gpu/backend/app/services/textflow/decision_layer.py) 的 `_collect_boundary_evidences_for_scoring()` 会把 `canonical_candidate_boundaries + punctuation + gap + speaker + duration_guard` 全部拼成候选边界，再逐个过阈值，不是“整段方案决策”。

## 1. 目标态约束

### 1.1 上游契约

- `hook` 只允许用于：
  - 时间挂载
  - claim/finalize
  - cross-chunk lock
  - trace/调试
- `hook` 禁止用于：
  - 生成 `BoundaryEvidence`
  - 进入 `canonical_candidate_boundaries`
  - 作为 Decision 可执行切分信号
- `anchor_block_close` 若保留，只能来自真实 `alignment block` 边界，而不是 hook 切换。

### 1.2 Decision 契约

- `Decision` 的输入仍是：
  - `annotated_words`
  - `canonical_punctuation_facts`
  - `canonical_candidate_boundaries`
  - `speaker_turns`
  - token 时间流本身
- 但 `Decision` 的内部模型改为：
  - 先把上述输入统一转成 `boundary features`
  - 再以“当前 segment 从哪开始、下一刀切到哪最合理”为对象做滚动规划
  - `hard_limit` 只作为“当前 segment 的最大允许长度”约束，不再作为预生成候选边界
- 禁止继续保留 timeanchored 路径里的“边界抢切”语义：
  - `gap_pause` 不能再直接等价于一刀
  - `blank_valley` 不能再直接等价于一刀
  - `hard_limit_forced` 不能再是预排候选

### 1.3 验收标准

- 上游 `boundary_evidence_count` 应明显稀疏，不能再接近 token 数。
- qvvs `chunk_0001` 的 `17.9s-23.0s` 不再出现 2-3 词英文碎片连续输出。
- `Decision` trace 能解释每一刀为何被选中，以及为何其他候选被拒绝。
- 进入 `Decision` 后不依赖 `FinalSplitter`/fast draft 去修正文句。

---

## 2. File Map

### 2.1 Create

- `backend/app/services/textflow/ingress_segment_planner.py`
- `backend/tests/textflow/test_ingress_segment_planner.py`

### 2.2 Modify

- `backend/app/services/timeanchored_alignment/anchor_mount/contracts.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/local_extension.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/chain_solver.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/temporal_envelope_builder.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/boundary_evidence_builder.py`
- `backend/app/services/timeanchored_alignment/anchor_mount/service.py`
- `backend/app/services/textflow/decision_layer.py`
- `backend/tests/timeanchored_alignment/test_boundary_evidence_builder.py`
- `backend/tests/timeanchored_alignment/test_anchor_mount_service.py`
- `backend/tests/textflow/test_decision_layer_ingress_context.py`
- `llmdoc/architecture/dual-flow-finalization.md`
- `llmdoc/architecture/preparation-language-adapter.md`
- `llmdoc/architecture/text-processing-layered-architecture.md`

### 2.3 Responsibility Split

- `anchor_mount/contracts.py`
  - 明确 block 级 provenance 字段，让 `AnchorMountItem` 能表达“属于哪个 alignment block”
- `temporal_envelope_builder.py`
  - 把 `ChainSolveResult.committed_blocks` 的 block 归属投影进 item/envelope
- `boundary_evidence_builder.py`
  - 仅使用 `source_chunk_ids` 与 `alignment_block_id` 生成稀疏边界证据
- `ingress_segment_planner.py`
  - 封装 timeanchored 规划器，负责 feature 归一化、滚动选刀、短碎片惩罚、hard-limit 约束
- `decision_layer.py`
  - 作为主入口编排 planner，并输出新的 trace 统计

---

## Chunk 1: 锁死根因回归

### Task 1.1: 写失败测试，锁住“hook 不能变相成为切分依据”

**Files:**
- Modify: `backend/tests/timeanchored_alignment/test_boundary_evidence_builder.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_service.py`

- [ ] **Step 1: 在 `test_boundary_evidence_builder.py` 写失败测试，要求同一 block 内的 hook 变化不能产出 `anchor_block_close`**

```python
def test_boundary_evidence_builder_ignores_per_token_hook_churn_inside_same_block():
    items = (
        AnchorMountItem(
            unit_id="u0",
            unit_index=0,
            token_text="killinging",
            display_text="killinging",
            normalized_text="killinging",
            speaker_id=None,
            turn_id=None,
            source_chunk_ids=("chunk-1",),
            source_chunk_indices=(1,),
            mount_status="anchored",
            anchor_kind="lexical",
            envelope_index=0,
            source_hook_ids=("hook-30",),
            match_confidence=1.0,
            cross_chunk_lock_ids=tuple(),
            alignment_block_id="block-a",
        ),
        AnchorMountItem(
            unit_id="u1",
            unit_index=1,
            token_text="five",
            display_text="five",
            normalized_text="five",
            speaker_id=None,
            turn_id=None,
            source_chunk_ids=("chunk-1",),
            source_chunk_indices=(1,),
            mount_status="anchored",
            anchor_kind="lexical",
            envelope_index=1,
            source_hook_ids=("hook-31",),
            match_confidence=1.0,
            cross_chunk_lock_ids=tuple(),
            alignment_block_id="block-a",
        ),
    )

    evidences = BoundaryEvidenceBuilder().build(
        items=items,
        envelopes=_build_envelopes_for_items(items),
        punctuation_facts=tuple(),
        cross_chunk_locks=tuple(),
    )

    assert {item.reason for item in evidences} == set()
```

- [ ] **Step 2: 运行测试，确认以“缺少 `alignment_block_id` 字段或仍产出 `anchor_block_close`”失败**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/timeanchored_alignment/test_boundary_evidence_builder.py::test_boundary_evidence_builder_ignores_per_token_hook_churn_inside_same_block -v
```

Expected: FAIL

- [ ] **Step 3: 在 `test_anchor_mount_service.py` 写失败测试，锁住 `AnchorMountItem` 必须能表达 block 归属**

```python
def test_anchor_mount_service_projects_alignment_block_identity_into_items():
    result = service.align(preparation=preparation, language="en")
    block_ids = [item.alignment_block_id for item in result.anchor_mount_result.items]
    assert any(block_id for block_id in block_ids)
```

- [ ] **Step 4: 运行测试，确认失败**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/timeanchored_alignment/test_anchor_mount_service.py::test_anchor_mount_service_projects_alignment_block_identity_into_items -v
```

Expected: FAIL

- [ ] **Step 5: 暂停并通知 wgh，准备进入实现**

---

### Task 1.2: 写失败测试，锁住 qvvs 类英文多词短碎片乱切

**Files:**
- Create: `backend/tests/textflow/test_ingress_segment_planner.py`
- Modify: `backend/tests/textflow/test_decision_layer_ingress_context.py`

- [ ] **Step 1: 新建失败测试，复刻 qvvs 的坏例子时间结构**

```python
def test_timeanchored_planner_does_not_fragment_english_multiword_clause():
    words = [
        WordTimestamp(word="Oh", start=16.04, end=16.10),
        WordTimestamp(word="God", start=16.52, end=16.94),
        WordTimestamp(word="killinging", start=18.20, end=18.68),
        WordTimestamp(word="five", start=18.74, end=19.52),
        WordTimestamp(word="strangers", start=19.94, end=20.24),
        WordTimestamp(word="is", start=20.30, end=20.44),
        WordTimestamp(word="probably", start=20.46, end=20.78),
        WordTimestamp(word="more", start=20.82, end=21.04),
        WordTimestamp(word="beneficial", start=21.08, end=21.62),
        WordTimestamp(word="for", start=21.70, end=21.82),
        WordTimestamp(word="society", start=21.84, end=22.30),
        WordTimestamp(word="so", start=22.36, end=22.50),
        WordTimestamp(word="I'll", start=22.52, end=22.72),
        WordTimestamp(word="do", start=22.74, end=22.88),
        WordTimestamp(word="that", start=22.90, end=23.30),
    ]

    output = processor._run_segmentation_core(
        decision_input,
        stream_id="timeanchored:qvvs-window",
        chunk_index=1,
        is_last_chunk=True,
    )

    texts = [segment.text for segment in output.sentence_segments]
    assert "killinging five" not in texts
    assert "strangers is probably" not in texts
    assert "more beneficial" not in texts
    assert len(texts) <= 2
```

- [ ] **Step 2: 运行测试，确认当前实现失败**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/textflow/test_ingress_segment_planner.py::test_timeanchored_planner_does_not_fragment_english_multiword_clause -v
```

Expected: FAIL

- [ ] **Step 3: 在 `test_decision_layer_ingress_context.py` 写失败测试，锁住 hard limit 必须基于已接受切点滚动重算**

```python
def test_decision_layer_treats_hard_limit_as_rolling_constraint_not_static_candidate():
    output = processor._run_segmentation_core(...)
    reasons = output.applied_cut_plan.generation_report["reason_stats"]
    assert reasons.get("hard_limit_forced", 0) <= 1
    assert [segment.text for segment in output.sentence_segments].count("killinging five") == 0
```

- [ ] **Step 4: 运行测试，确认失败**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/textflow/test_decision_layer_ingress_context.py::test_decision_layer_treats_hard_limit_as_rolling_constraint_not_static_candidate -v
```

Expected: FAIL

- [ ] **Step 5: 暂停并通知 wgh，准备进入实现**

---

## Chunk 2: 修复 AnchorMount 边界证据契约

### Task 2.1: 把 block 身份投影到 item/envelope，禁止 hook 驱动边界

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/contracts.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/local_extension.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/chain_solver.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/temporal_envelope_builder.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_service.py`

- [ ] **Step 1: 在 `contracts.py` 给 `AnchorMountItem` 增加 `alignment_block_id`，必要时给 `TemporalEnvelope` 增加 block 诊断字段**

```python
@dataclass(frozen=True)
class AnchorMountItem:
    ...
    alignment_block_id: str | None
```

- [ ] **Step 2: 在 `local_extension.py` 和 `chain_solver.py` 保留稳定 block identity，避免后续只能回看 hook**

```python
LocalAlignmentBlock(
    block_id=f"block-{index}",
    ...
)
```

- [ ] **Step 3: 在 `temporal_envelope_builder.py` 把 committed block 归属投影到每个 `AnchorMountItem`**

```python
unit_to_block_id = _build_unit_block_map(solve_result.committed_blocks)
...
alignment_block_id=unit_to_block_id.get(unit_index)
```

- [ ] **Step 4: 运行相关测试，确认转绿**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/timeanchored_alignment/test_anchor_mount_service.py backend/tests/timeanchored_alignment/test_boundary_evidence_builder.py -v
```

Expected: PASS

- [ ] **Step 5: 暂停并通知 wgh 手动确认 checkpoint**

---

### Task 2.2: 让 `anchor_block_close` 真正只表达 block 边界

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/boundary_evidence_builder.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/service.py`
- Modify: `backend/tests/timeanchored_alignment/test_boundary_evidence_builder.py`

- [ ] **Step 1: 在 `boundary_evidence_builder.py` 删除 `source_hook_ids !=` 生成 `anchor_block_close` 的逻辑**

```python
if (
    items[index].alignment_block_id
    and items[index + 1].alignment_block_id
    and items[index].alignment_block_id != items[index + 1].alignment_block_id
):
    ...
```

- [ ] **Step 2: 为 `anchor_block_close` 增加 block-aware metadata，保留可观测性**

```python
metadata={
    **base_metadata,
    "left_block_id": items[index].alignment_block_id,
    "right_block_id": items[index + 1].alignment_block_id,
}
```

- [ ] **Step 3: 在 `service.py` 的 metrics 中补充 block 边界统计，便于 trace 判断是否稀疏**

```python
"block_boundary_count": sum(
    1 for item in boundary_evidences if item.reason == "anchor_block_close"
)
```

- [ ] **Step 4: 运行测试，确认 `anchor_block_close` 只在 block 切换时出现**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/timeanchored_alignment/test_boundary_evidence_builder.py backend/tests/timeanchored_alignment/test_anchor_mount_service.py backend/tests/timeanchored_alignment/test_job_punctuation_fact_chain_regression.py -v
```

Expected: PASS

- [ ] **Step 5: 暂停并通知 wgh 手动确认 checkpoint**

---

## Chunk 3: 用滚动 segment planner 替换 Decision 边界抢切

### Task 3.1: 新建 timeanchored planner，把“边界候选”改成“整段规划”

**Files:**
- Create: `backend/app/services/textflow/ingress_segment_planner.py`
- Create: `backend/tests/textflow/test_ingress_segment_planner.py`

- [ ] **Step 1: 在新测试文件中写失败测试，要求 planner 输出的是 segment 方案而不是边界列表**

```python
def test_ingress_segment_planner_selects_next_cut_from_segment_state():
    planner = IngressSegmentPlanner()
    plan = planner.build_plan(words=words, data=decision_input, stream_id="timeanchored:test", chunk_index=1)
    assert plan.decisions
    assert plan.generation_report["generated_by"] == "timeanchored_segment_planner"
```

- [ ] **Step 2: 运行测试，确认失败**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/textflow/test_ingress_segment_planner.py::test_ingress_segment_planner_selects_next_cut_from_segment_state -v
```

Expected: FAIL

- [ ] **Step 3: 在新文件实现最小 planner 骨架**

```python
class IngressSegmentPlanner:
    def build_plan(self, *, words, data, stream_id, chunk_index):
        ...
```

- [ ] **Step 4: 运行测试，确认骨架通路通过**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/textflow/test_ingress_segment_planner.py::test_ingress_segment_planner_selects_next_cut_from_segment_state -v
```

Expected: PASS

- [ ] **Step 5: 暂停并通知 wgh 手动确认 checkpoint**

---

### Task 3.2: 把 `gap/punctuation/speaker/alignment` 降级为 feature，`hard_limit` 改成约束

**Files:**
- Modify: `backend/app/services/textflow/ingress_segment_planner.py`
- Modify: `backend/app/services/textflow/decision_layer.py`
- Modify: `backend/tests/textflow/test_ingress_segment_planner.py`
- Modify: `backend/tests/textflow/test_decision_layer_ingress_context.py`

- [ ] **Step 1: 在 planner 测试里写失败测试，锁住 `hard_limit` 不能再作为预生成候选**

```python
def test_ingress_segment_planner_uses_hard_limit_as_constraint():
    plan = planner.build_plan(words=words, data=decision_input, stream_id="timeanchored:test", chunk_index=1)
    assert plan.generation_report["feature_stats"]["duration_guard_candidates"] == 0
    assert plan.generation_report["constraint_stats"]["hard_limit_hits"] >= 0
```

- [ ] **Step 2: 运行测试，确认失败**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/textflow/test_ingress_segment_planner.py::test_ingress_segment_planner_uses_hard_limit_as_constraint -v
```

Expected: FAIL

- [ ] **Step 3: 在 planner 中实现统一 feature 归一化**

```python
features = [
    *self._collect_alignment_features(data.canonical_candidate_boundaries),
    *self._collect_punctuation_features(data.canonical_punctuation_facts),
    *self._collect_gap_features(words),
    *self._collect_turn_features(data.aligned_facts),
]
```

- [ ] **Step 4: 用滚动状态替代全窗边界筛选**

```python
segment_start_idx = 0
while segment_start_idx < len(words) - 1:
    candidate_ends = self._enumerate_candidate_endings(...)
    selected = self._select_best_segment_end(
        segment_start_idx=segment_start_idx,
        candidate_ends=candidate_ends,
        max_segment_sec=max_segment_sec,
    )
    if selected is None:
        break
    decisions.append(self._build_cut_decision(...))
    segment_start_idx = selected.split_idx + 1
```

- [ ] **Step 5: 给 segment 评分增加通用“最小可行句”约束，而不是 case-by-case 后置 merge**

```python
score -= self._short_fragment_penalty(
    word_count=current_word_count,
    duration_sec=current_duration_sec,
    has_terminal_punct=has_terminal_punct,
    closure_strength=closure_strength,
)
```

- [ ] **Step 6: 在 `decision_layer.py` 中，把 timeanchored ingress 路由到新 planner；禁止同一路径继续走旧的 `_build_decision_scored_cut_plan`**

```python
if self._is_timeanchored_ingress(stream_id=stream_id, ingress_context=data.ingress_context):
    return self._ingress_segment_planner.build_plan(...)
```

- [ ] **Step 7: 运行 textflow 回归，确认 qvvs 类坏例子转绿**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/textflow/test_ingress_segment_planner.py backend/tests/textflow/test_decision_layer_ingress_context.py -v
```

Expected: PASS

- [ ] **Step 8: 暂停并通知 wgh 手动确认 checkpoint**

---

### Task 3.3: 清掉 timeanchored 路径里的旧语义残留

**Files:**
- Modify: `backend/app/services/textflow/decision_layer.py`
- Modify: `backend/tests/textflow/test_decision_layer_ingress_context.py`

- [ ] **Step 1: 写失败测试，锁住 timeanchored 生成报告不再声明 `decision_layer_boundary_scoring`**

```python
def test_timeanchored_ingress_uses_segment_planner_generation_report():
    output = processor._run_segmentation_core(...)
    assert output.applied_cut_plan.generation_report["generated_by"] == "timeanchored_segment_planner"
```

- [ ] **Step 2: 运行测试，确认失败**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/textflow/test_decision_layer_ingress_context.py::test_timeanchored_ingress_uses_segment_planner_generation_report -v
```

Expected: FAIL

- [ ] **Step 3: 删除 timeanchored 路径里仍旧依赖 `_collect_boundary_evidences_for_scoring()` 的调用**

```python
if self._is_timeanchored_ingress(...):
    return self._ingress_segment_planner.build_plan(...)
```

- [ ] **Step 4: 保留旧 boundary scoring 仅给非-timeanchored 调用方；若已无调用，直接删除相关 helper**

```python
def _build_decision_scored_cut_plan(...):
    ...
```

- [ ] **Step 5: 运行回归，确认 timeanchored trace 已完全切到 planner 语义**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/textflow/test_ingress_segment_planner.py backend/tests/textflow/test_decision_layer_ingress_context.py backend/tests/textflow/test_decision_ingress_adapter.py -v
```

Expected: PASS

- [ ] **Step 6: 暂停并通知 wgh 手动确认 checkpoint**

---

## Chunk 4: 强化 trace、文档同步与样本复核

### Task 4.1: 把 planner 诊断写进 trace，避免以后再靠猜

**Files:**
- Modify: `backend/app/services/textflow/decision_layer.py`
- Modify: `backend/tests/textflow/test_decision_layer_ingress_context.py`

- [ ] **Step 1: 写失败测试，要求 trace 中出现 feature/constraint/rejection 统计**

```python
def test_timeanchored_trace_contains_planner_feature_and_constraint_stats():
    output = processor._run_segmentation_core(...)
    report = output.segmentation_report["soft_cut_stats"]
    assert "feature_stats" in report
    assert "constraint_stats" in report
    assert "rejection_stats" in report
```

- [ ] **Step 2: 运行测试，确认失败**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/textflow/test_decision_layer_ingress_context.py::test_timeanchored_trace_contains_planner_feature_and_constraint_stats -v
```

Expected: FAIL

- [ ] **Step 3: 在 planner/decision trace 中写入以下字段**

```python
"feature_stats": {
    "alignment_boundary_count": ...,
    "punctuation_feature_count": ...,
    "gap_feature_count": ...,
    "speaker_feature_count": ...,
},
"constraint_stats": {
    "hard_limit_hits": ...,
    "rolling_resets": ...,
},
"rejection_stats": {
    "short_fragment": ...,
    "low_closure": ...,
    "adjacent_fragment_chain": ...,
}
```

- [ ] **Step 4: 运行测试，确认通过**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/textflow/test_decision_layer_ingress_context.py -v
```

Expected: PASS

- [ ] **Step 5: 暂停并通知 wgh 手动确认 checkpoint**

---

### Task 4.2: 覆写文档，让最新语义与代码对齐

**Files:**
- Modify: `llmdoc/architecture/dual-flow-finalization.md`
- Modify: `llmdoc/architecture/preparation-language-adapter.md`
- Modify: `llmdoc/architecture/text-processing-layered-architecture.md`

- [ ] **Step 1: 覆写文档中的 `hook` 语义，明确 hook 只属于对齐内部，不属于切句证据**

- [ ] **Step 2: 覆写文档中的 `Decision` 语义，明确 `hard_limit` 是约束，不是候选边界**

- [ ] **Step 3: 覆写文档中的主链描述，明确 timeanchored 使用滚动 segment planner，而不是边界抢切**

- [ ] **Step 4: 自查三份文档，确认没有残留旧叙事**

Check:

```powershell
rg -n "hook.*boundary|boundary_scoring|hard_limit_forced|slot-" llmdoc/architecture
```

Expected: 仅剩与新语义一致的结果

- [ ] **Step 5: 暂停并通知 wgh 手动确认 checkpoint**

---

### Task 4.3: 用 qvvs 样本重新验收

**Files:**
- No code changes required unless rerun暴露新问题

- [ ] **Step 1: 使用当前代码重新跑 `test_en_1.mp4`，产出新的 postprocess trace**

- [ ] **Step 2: 对比以下文件，确认主链语义已变化**

Compare:

- `jobs/<new-job>/debug/postprocess/chunk_0001/21_anchor_mount.output.json`
- `jobs/<new-job>/debug/postprocess/chunk_0001/31_decision_ingress.output.json`
- `jobs/<new-job>/debug/postprocess/chunk_0001/40_decision.output.json`
- `jobs/<new-job>/test_en_1.srt`

- [ ] **Step 3: 逐条验收以下条件**

- `boundary_evidence_count` 不再接近 token 数
- `anchor_block_close` 不再由 per-token hook churn 触发
- `soft_cut_stats.reason_stats` 不再在同一句里连续出现多次 `gap_pause`
- `17.9s-23.0s` 不再被切成 4 段短碎片

- [ ] **Step 4: 如果 rerun 仍有碎切，继续往 planner 的 segment viability 评分补充证据，而不是加输出层 merge**

- [ ] **Step 5: 汇总结果并交给 wgh 手动决定是否进入下一轮实现**

---

## 3. 统一验证命令

### 3.1 单元/契约回归

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' `
  backend/tests/timeanchored_alignment/test_boundary_evidence_builder.py `
  backend/tests/timeanchored_alignment/test_anchor_mount_service.py `
  backend/tests/timeanchored_alignment/test_job_punctuation_fact_chain_regression.py `
  backend/tests/textflow/test_ingress_segment_planner.py `
  backend/tests/textflow/test_decision_layer_ingress_context.py `
  backend/tests/textflow/test_decision_ingress_adapter.py -v
```

Expected: PASS

### 3.2 文档/语义自查

```powershell
rg -n "source_hook_ids !=|reason=\"anchor_block_close\"|decision_layer_boundary_scoring|hard_limit_forced" backend llmdoc
```

Expected:

- `source_hook_ids !=` 不再出现在 `boundary_evidence_builder.py`
- timeanchored 主链不再把 `decision_layer_boundary_scoring` 作为生成器
- `hard_limit_forced` 仅作为 planner 诊断理由存在，不能再作为预生成候选路径残留

---

## 4. Out of Scope

- 不重写 Preparation/window builder，除非 rerun 证明 token-unit 本身错误。
- 不在输出层新增 merge/fallback 兜底。
- 不为了当前问题引入新的全局配置项。

---

## 5. 完成定义

- hook 已从切分语义中彻底移除。
- `anchor_block_close` 成为真正稀疏的 block 边界证据，或在无法成立时被删除。
- timeanchored `Decision` 已切换为滚动式 segment planner。
- qvvs 坏例子不再复现。
- 架构文档已同步到最新真实实现。
