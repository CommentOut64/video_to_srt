# Timeanchored Owner Truth Separation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不重写 decoder / decision 主体的前提下，把 timeanchored 主链中的 `window_text_truth` 与 `owner_text_truth` 正式分离，阻断多 chunk window 把整窗文本误当 owner commit truth 的结构性错误。

**Architecture:** Selection 层生成双真相，Preparation 只消费 owner truth 进入 canonical 与正式 provenance，OutputProjection 显式区分 window scope 与 replace scope，southbound cleanup 只按 owner commit scope 执行。窗口级 slow text 仍可保留给 punctuation、diagnostics 和后续辅助能力，但不再拥有默认提交权。

**Tech Stack:** Python, pytest, timeanchored alignment, textflow output chain, Windows PowerShell

---

## 0. Scope

- 只改 `selection -> preparation -> output projection` 最小闭环。
- 不重写 decoder / decision 主算法。
- 不在本计划中直接修改 prompt 或 slow window builder。
- commit 由 wgh 手动执行，本计划不包含自动提交步骤。

## 1. File Map

### 1.1 Modify

- `backend/app/services/timeanchored_alignment/contracts.py`
- `backend/app/services/timeanchored_alignment/selection/contracts.py`
- `backend/app/services/timeanchored_alignment/selection/service.py`
- `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- `backend/app/services/timeanchored_alignment/preparation/contracts.py`
- `backend/app/services/timeanchored_alignment/preparation/assembler.py`
- `backend/app/services/timeanchored_alignment/output_projection/output_projector.py`
- `backend/tests/timeanchored_alignment/test_selection_service.py`
- `backend/tests/timeanchored_alignment/test_no_inline_preparation_left.py`
- `backend/tests/timeanchored_alignment/test_alignment_preparation.py`
- `backend/tests/timeanchored_alignment/test_output_projection.py`
- `backend/tests/timeanchored_alignment/test_output_projection_contracts.py`
- `backend/tests/timeanchored_alignment/test_output_integration.py`
- `backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py`
- `llmdoc/architecture/preparation-language-adapter.md`
- `llmdoc/architecture/segmentation-alignment-services.md`
- `llmdoc/architecture/dual-flow-finalization.md`

### 1.2 Responsibilities

- `contracts.py`
  - 升级 `SelectedTextTruth` 为双真相正式契约
- `selection/*`
  - 生成 `window truth + owner truth`
- `alignment_stage_service.py`
  - 移除 preparation promotion，收口双真相 trace
- `preparation/*`
  - 只让 owner truth 进入 canonical，并正式携带 commit scope
- `output_projector.py`
  - 区分 window scope 与 replace scope
- `tests/*`
  - 锁住 owner/window truth separation 的正式行为
- `llmdoc/*`
  - 覆写旧的“整窗 promotion / 整窗 replace scope”叙事

---

## Chunk 1: 用失败测试锁住“真相分层”

### Task 1.1: 为 Selection 双真相新增失败测试

**Files:**
- Modify: `backend/tests/timeanchored_alignment/test_selection_service.py`
- Modify: `backend/app/services/timeanchored_alignment/selection/service.py`

- [ ] **Step 1: 新增多 source chunk window 夹具，构造“chosen_track 含前一个 chunk 文本”的场景**

```python
ready_window = _build_multi_source_ready_window(
    owner_chunk_id="chunk-33",
    source_units=[
        ("chunk-32", "在中午12点50分 ..."),
        ("chunk-33", "这家商店的老板有个15岁的儿子 ..."),
    ],
)
chosen_track = _build_track(
    text="在中午12点50分 ... 这家商店的老板有个15岁的儿子 ...",
    source="chosen",
)
```

- [ ] **Step 2: 写失败测试，断言 `SelectionOutcome.selected_text_truth` 必须同时暴露 window truth 与 owner truth**

```python
assert outcome.selected_text_truth.window_text.startswith("在中午12点50分")
assert outcome.selected_text_truth.owner_text.startswith("这家商店的老板")
assert outcome.selected_text_truth.owner_source_chunk_ids == ("chunk-33",)
assert outcome.selected_text_truth.window_source_chunk_ids == ("chunk-32", "chunk-33")
assert outcome.selected_text_truth.text == outcome.selected_text_truth.owner_text
```

- [ ] **Step 3: 运行失败测试，确认当前实现还只有单一 truth**

Run:

```powershell
$env:PYTHONPATH='backend'; .\.venv\Scripts\python.exe -m pytest -o addopts= backend/tests/timeanchored_alignment/test_selection_service.py -q
```

Expected: FAIL，原因是 `SelectedTextTruth` 尚未提供双真相字段

### Task 1.2: 为 Selection owner 提取策略写失败测试

**Files:**
- Modify: `backend/tests/timeanchored_alignment/test_selection_service.py`

- [ ] **Step 1: 新增失败测试，锁住“window text 无法稳定裁剪时，owner truth 退回 owner source text”**

```python
assert outcome.selected_text_truth.owner_text == "owner source text"
assert outcome.selected_text_truth.metadata["owner_text_resolve_mode"] == "owner_source_fallback"
```

- [ ] **Step 2: 运行该测试，确认当前实现失败**

Run:

```powershell
$env:PYTHONPATH='backend'; .\.venv\Scripts\python.exe -m pytest -o addopts= backend/tests/timeanchored_alignment/test_selection_service.py::test_selection_service_falls_back_to_owner_source_text_when_window_text_cannot_be_clipped -q
```

Expected: FAIL

---

## Chunk 2: 用失败测试锁住“Preparation 不再放大 truth”

### Task 2.1: 为 AlignmentStage preparation 输入移除 promotion 写失败测试

**Files:**
- Modify: `backend/tests/timeanchored_alignment/test_no_inline_preparation_left.py`
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`

- [ ] **Step 1: 覆写现有多 source chunk 测试预期，不再允许 `_resolve_preparation_text_inputs()` 把 owner truth 提升成整窗 source text**

```python
assert promoted_selected_text_truth.owner_text == "这家商店的老板有个15岁的儿子 ..."
assert promoted_selected_text_truth.window_text.startswith("在中午12点50分")
assert promoted_selected_text_truth.owner_source_chunk_ids == ("chunk-33",)
assert promoted_whisper["text_clean"] == promoted_selected_text_truth.owner_text
```

- [ ] **Step 2: 运行目标测试，确认当前实现仍在做整窗 promotion**

Run:

```powershell
$env:PYTHONPATH='backend'; .\.venv\Scripts\python.exe -m pytest -o addopts= backend/tests/timeanchored_alignment/test_no_inline_preparation_left.py::test_run_timeanchored_main_chain_promotes_window_source_text_when_owner_text_is_shorter -q
```

Expected: FAIL

### Task 2.2: 为 Preparation canonical / commit scope 写失败测试

**Files:**
- Modify: `backend/tests/timeanchored_alignment/test_alignment_preparation.py`
- Modify: `backend/app/services/timeanchored_alignment/preparation/contracts.py`
- Modify: `backend/app/services/timeanchored_alignment/preparation/assembler.py`

- [ ] **Step 1: 新增失败测试，锁住 canonical 只能来自 owner truth**

```python
assert package.canonical_sequence.normalized_text == selected_text_truth.owner_text
assert package.canonical_sequence.metadata["window_text"] == selected_text_truth.window_text
```

- [ ] **Step 2: 新增失败测试，锁住 preparation 正式携带 commit scope**

```python
assert package.commit_scope_chunk_ids == ("chunk-33",)
assert package.commit_scope_chunk_indices == (33,)
assert package.source_chunk_ids == ("chunk-32", "chunk-33")
```

- [ ] **Step 3: 运行 alignment preparation 测试，确认当前实现失败**

Run:

```powershell
$env:PYTHONPATH='backend'; .\.venv\Scripts\python.exe -m pytest -o addopts= backend/tests/timeanchored_alignment/test_alignment_preparation.py -q
```

Expected: FAIL，原因是 canonical 仍使用单一 truth，且 bundle 还没有正式 commit scope

---

## Chunk 3: 实现双真相契约与 selection owner 提取

### Task 3.1: 升级 `SelectedTextTruth` 契约

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/contracts.py`
- Test: `backend/tests/timeanchored_alignment/test_selection_service.py`

- [ ] **Step 1: 在 `SelectedTextTruth` 中引入正式双真相字段**

```python
owner_text: str
owner_source_chunk_ids: tuple[str, ...]
window_text: str
window_source_chunk_ids: tuple[str, ...]
```

- [ ] **Step 2: 保留兼容属性**

```python
@property
def text(self) -> str:
    return self.owner_text

@property
def source_chunk_ids(self) -> tuple[str, ...]:
    return self.owner_source_chunk_ids
```

- [ ] **Step 3: 运行 selection 测试，确认契约层通过**

Run:

```powershell
$env:PYTHONPATH='backend'; .\.venv\Scripts\python.exe -m pytest -o addopts= backend/tests/timeanchored_alignment/test_selection_service.py -q
```

Expected: 之前的“字段缺失”失败消失，但 owner 提取行为可能仍失败

### Task 3.2: 在 selection service 中生成双真相

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/selection/contracts.py`
- Modify: `backend/app/services/timeanchored_alignment/selection/service.py`
- Test: `backend/tests/timeanchored_alignment/test_selection_service.py`

- [ ] **Step 1: 扩展 `WindowSelectionScope`，让它显式带 owner chunk id / index**

```python
owner_chunk_id: str = ""
owner_chunk_index: int | None = None
```

- [ ] **Step 2: 在 `TextSelectionService` 中从 `ready_window.source_units` 推导 owner scope**

- [ ] **Step 3: 新增 owner text 提取逻辑**

```python
owner_source_text = self._build_owner_source_text(...)
owner_text = self._resolve_owner_text(
    window_text=window_text,
    owner_source_text=owner_source_text,
)
```

- [ ] **Step 4: 多 source chunk window 下，只有在 owner source units 覆盖整窗时，owner truth 才允许复用整窗 text**

- [ ] **Step 5: 跑 selection 测试，确认双真相与 fallback 规则转绿**

Run:

```powershell
$env:PYTHONPATH='backend'; .\.venv\Scripts\python.exe -m pytest -o addopts= backend/tests/timeanchored_alignment/test_selection_service.py -q
```

Expected: PASS

---

## Chunk 4: 实现 preparation truth 收口与 commit scope 正式化

### Task 4.1: 删除 preparation promotion 语义

**Files:**
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- Test: `backend/tests/timeanchored_alignment/test_no_inline_preparation_left.py`

- [ ] **Step 1: 把 `_resolve_preparation_text_inputs()` 改为双真相标准化，不再做整窗文本 promotion**

- [ ] **Step 2: trace summary 补充 owner/window truth 长度与 scope**

```python
"owner_text_len": len(selected_text_truth.owner_text),
"window_text_len": len(selected_text_truth.window_text),
"owner_source_chunk_ids": ...,
"window_source_chunk_ids": ...,
```

- [ ] **Step 3: 运行定向测试，确认 preparation 输入不再被放大**

Run:

```powershell
$env:PYTHONPATH='backend'; .\.venv\Scripts\python.exe -m pytest -o addopts= backend/tests/timeanchored_alignment/test_no_inline_preparation_left.py -q
```

Expected: PASS

### Task 4.2: 让 Preparation 正式携带 commit scope

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/preparation/contracts.py`
- Modify: `backend/app/services/timeanchored_alignment/preparation/assembler.py`
- Test: `backend/tests/timeanchored_alignment/test_alignment_preparation.py`

- [ ] **Step 1: 为 `PreparationBundle` 新增正式 commit scope 字段**

```python
commit_scope_chunk_ids: tuple[str, ...]
commit_scope_chunk_indices: tuple[int, ...]
```

- [ ] **Step 2: `AlignmentPreparationAssembler.prepare()` 改为**

1. canonical sequence 用 `owner_text_truth`
2. `window_text_truth` 只写入 metadata / debug_refs
3. `commit_scope_chunk_ids` 来自 `SelectedTextTruth.owner_source_chunk_ids`

- [ ] **Step 3: provenance metadata 明确记录双真相来源**

```python
"selected_owner_chunk_ids": [...],
"selected_window_chunk_ids": [...],
```

- [ ] **Step 4: 运行 preparation 测试，确认 owner canonical / commit scope 转绿**

Run:

```powershell
$env:PYTHONPATH='backend'; .\.venv\Scripts\python.exe -m pytest -o addopts= backend/tests/timeanchored_alignment/test_alignment_preparation.py -q
```

Expected: PASS

---

## Chunk 5: 实现 output projection 的 replace scope 分离

### Task 5.1: 为 output projection 契约写失败测试

**Files:**
- Modify: `backend/tests/timeanchored_alignment/test_output_projection.py`
- Modify: `backend/tests/timeanchored_alignment/test_output_projection_contracts.py`
- Modify: `backend/tests/timeanchored_alignment/test_output_integration.py`

- [ ] **Step 1: 新增失败测试，锁住 `OutputProjectionInput` 同时带 window scope 与 replace scope**

```python
assert data.source_chunk_ids == ("chunk-32", "chunk-33")
assert data.replace_scope_chunk_ids == ("chunk-33",)
```

- [ ] **Step 2: 新增失败测试，锁住 projector 不能再用整窗 source scope 填充 `replace_scope_chunk_ids`**

```python
assert result.sentence_records[0].replace_scope_chunk_ids == ("chunk-33",)
assert projection_meta["source_chunk_ids"] == ["chunk-32", "chunk-33"]
assert projection_meta["replace_scope_chunk_ids"] == ["chunk-33"]
```

- [ ] **Step 3: 新增 integration 测试，锁住 `window_group` cleanup 只清 owner replace scope**

```python
assert payloads[0]["source_chunk_ids"] == ["chunk-33"]
```

- [ ] **Step 4: 运行目标测试，确认当前实现失败**

Run:

```powershell
$env:PYTHONPATH='backend'; .\.venv\Scripts\python.exe -m pytest -o addopts= backend/tests/timeanchored_alignment/test_output_projection.py backend/tests/timeanchored_alignment/test_output_projection_contracts.py backend/tests/timeanchored_alignment/test_output_integration.py -q
```

Expected: FAIL

### Task 5.2: 实现 replace scope 分离

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/output_projection/output_projector.py`
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- Test: `backend/tests/timeanchored_alignment/test_output_projection.py`
- Test: `backend/tests/timeanchored_alignment/test_output_projection_contracts.py`
- Test: `backend/tests/timeanchored_alignment/test_output_integration.py`
- Test: `backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py`

- [ ] **Step 1: 扩展 `OutputProjectionInput`**

```python
replace_scope_chunk_ids: tuple[str, ...]
replace_scope_chunk_indices: tuple[int, ...]
```

- [ ] **Step 2: projector 默认使用 commit scope 作为 `replace_scope_chunk_ids`**

- [ ] **Step 3: `AlignmentStageService._build_projected_output_batches()` 用 `preparation.commit_scope_chunk_ids` 填入 projection input**

- [ ] **Step 4: 运行 output projection / routing 测试**

Run:

```powershell
$env:PYTHONPATH='backend'; .\.venv\Scripts\python.exe -m pytest -o addopts= backend/tests/timeanchored_alignment/test_output_projection.py backend/tests/timeanchored_alignment/test_output_projection_contracts.py backend/tests/timeanchored_alignment/test_output_integration.py backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py -q
```

Expected: PASS

---

## Chunk 6: 文档覆写与统一验证

### Task 6.1: 覆写 llmdoc 中过时叙事

**Files:**
- Modify: `llmdoc/architecture/preparation-language-adapter.md`
- Modify: `llmdoc/architecture/segmentation-alignment-services.md`
- Modify: `llmdoc/architecture/dual-flow-finalization.md`

- [ ] **Step 1: 删掉“多 source chunk window 允许 preparation 提升为整窗文本”的描述**

- [ ] **Step 2: 明确新增术语**

1. `window_text_truth`
2. `owner_text_truth`
3. `commit_scope_chunk_ids`

- [ ] **Step 3: 明确 output projection 现在区分**

1. window coverage metadata
2. southbound replace scope

- [ ] **Step 4: 用 `rg` 自查旧叙事是否还有残留**

Run:

```powershell
rg -n "提升为 preparation 输入|整窗文本|replace_scope_chunk_ids.*source_chunk_ids|SelectedTextTruth 默认是 canonical 文本真源" llmdoc/architecture
```

Expected: 结果只剩新语义或无命中

### Task 6.2: 统一回归

**Files:**
- No additional file changes

- [ ] **Step 1: 跑 selection / preparation / output projection 的定向回归**

Run:

```powershell
$env:PYTHONPATH='backend'; .\.venv\Scripts\python.exe -m pytest -o addopts= backend/tests/timeanchored_alignment/test_selection_service.py backend/tests/timeanchored_alignment/test_no_inline_preparation_left.py backend/tests/timeanchored_alignment/test_alignment_preparation.py backend/tests/timeanchored_alignment/test_output_projection.py backend/tests/timeanchored_alignment/test_output_projection_contracts.py backend/tests/timeanchored_alignment/test_output_integration.py backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py -q
```

Expected: PASS

- [ ] **Step 2: 如条件允许，补跑当前坏样本**

检查：

- `jobs/<new-job>/debug/postprocess/chunk_0003/05_selection.output.json`
- `jobs/<new-job>/debug/postprocess/chunk_0021/10_preparation.input.json`
- `jobs/<new-job>/debug/postprocess/chunk_0024/11_preparation.output.json`
- `jobs/<new-job>/test_en_1.srt`

- [ ] **Step 3: 按以下标准验收**

1. `05_selection.output.json` 中 owner truth 不再包含明显非 owner 文本
2. `10_preparation.input.json` 的 selected text 不再被整窗 promotion 放大
3. `50_output_projection.output.json` 中 `replace_scope_chunk_ids` 与 `source_chunk_ids` 可区分
4. 最新 SRT 不再出现当前这种 5 秒 / 10 秒级大空洞

---

## 2. Out of Scope

- 不改 decoder 主算法。
- 不改 decision planner 主算法。
- 不在本轮引入新的 recovery planner。
- 不做无关文档整理。

## 3. Done Criteria

- `SelectedTextTruth` 正式完成 owner/window 双真相分离。
- preparation 不再做整窗 truth promotion。
- output projection 正式完成 window scope / replace scope 分离。
- llmdoc 已覆写旧叙事。
- 定向 pytest 回归通过。
