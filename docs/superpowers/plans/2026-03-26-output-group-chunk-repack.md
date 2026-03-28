# 输出层重打包 Chunk（脱钩旧 VAD Chunk）Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从根源消除“切分后字幕被强制投影回旧 chunk 导致丢句/误清理”的系统性风险，让输出层以重打包后的 output chunk 为真源，同时保持处理层 `finalized_indices` 完整性口径不变。

**Architecture:** 保留现有 `ready_slow_window/source_chunk_indices` 作为计算与完成态统计口径；输出链路不再按 source chunk fan-out 替换，而是按 window 级 output-group 一次性产出权威 batch，并携带 `source_chunk_ids` 作为清理作用域。前端 `replace_chunk` 投影改为“按 source scope 清旧 + 写入 output-group 定稿 + 拒收 scope 内迟到 draft”，从而解除对旧 chunk 等价性的硬依赖。

**Tech Stack:** Python 3.10+（dataclass、pytest）、Vue3/Pinia/Vitest、SSE `subtitle.replace_chunk` 事件链、现有 `timeanchored_alignment` 与 `editorEventProjector`。

---

## 0. ROI Gate（本次选择 B：局部重构）

当前问题同时跨越 `OutputProjector -> StreamingSubtitleManager -> EditorEventProjector -> 完成态收口` 四个高耦合点，无法通过单点补丁稳定修复；继续补丁会继续增加 non-owner 空清理等 special case。选择 **B（局部重构）**：仅重构“输出主语义”最小闭环，处理层 chunk 与完成态计数保持不动，避免扩散到 ASR/VAD 主流程。

## 文件结构与职责边界

- Modify: `backend/app/services/timeanchored_alignment/output_projection/output_projector.py`
  - 职责：从“按 coverage bindings 分桶输出”改为“window output-group 输出”，并输出 `replace_scope_chunk_ids` 元数据。
- Modify: `backend/app/services/streaming_subtitle.py`
  - 职责：新增 scope-aware 替换入口（清理 source scope，再写入 output-group），移除 non-owner 空桶预期清理触发路径。
- Modify: `backend/app/services/textflow/output_dispatch_adapter.py`
  - 职责：把 `source_chunk_ids` 透传到输出 payload（便于前端投影与调试）。
- Modify: `frontend/src/stores/editor/editorEventProjector.js`
  - 职责：`replace_chunk` 增加 source scope 清理与 finalized scope 拒收机制，解除“必须同逻辑 chunk”假设。
- Modify: `frontend/src/views/EditorView.vue`
  - 职责：保持 `replace_chunk` 事件入口不变，补充新 payload 字段兼容检查（不改交互流程）。
- Modify: `frontend/src/stores/projectStore.js`
  - 职责：legacy 模式兼容 `source_chunk_ids` 清理（避免 V1 分支残留旧草稿）。
- Modify: `backend/tests/timeanchored_alignment/test_output_projection.py`
- Modify: `backend/tests/timeanchored_alignment/test_output_integration.py`
- Modify/Create: `backend/tests/test_streaming_subtitle_replace_chunk_empty.py`（新增 scoped replace 用例）
- Modify: `frontend/tests/stores/editor/editorEventProjector.spec.js`
- Modify: `frontend/tests/regression/phasef-editor-realtime-authoritative-reload.regression.spec.js`
- Modify: `llmdoc/architecture/async-dual-pipeline.md`
- Modify: `llmdoc/architecture/sse-events.md`
- Modify: `llmdoc/changelog.md`

---

## Chunk 1: 回归门禁先行（先红后绿）

### Task 1.1: 输出投影契约改为 output-group（禁止空桶 fan-out）

**Files:**
- Modify: `backend/tests/timeanchored_alignment/test_output_projection.py`
- Modify: `backend/tests/timeanchored_alignment/test_output_integration.py`

- [ ] **Step 1: 写失败测试（投影结果只产出非空 output-group batch）**

```python
def test_output_projector_emits_single_window_group_batch_with_scope_metadata():
    projected = OutputProjector().project(_build_input())
    assert len(projected) == 1
    batch = projected[0]
    assert batch.chunk_id.startswith("ow-")
    assert list(batch.diagnostics["projection"]["replace_scope_chunk_ids"]) == ["chunk-0", "chunk-1", "chunk-2"]
```

- [ ] **Step 2: 运行测试确认失败（当前仍是按 source chunk fan-out）**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_output_projection.py -k window_group -v`
Expected: FAIL（当前实现返回 3 个 batch，且包含 empty batch）

- [ ] **Step 3: 写失败测试（输出链路不再产生 non-owner 空清理调用）**

```python
def test_output_chain_no_longer_dispatches_non_owner_empty_cleanup_batches():
    projected = OutputProjector().project(_build_projection_input())
    subtitle_manager = _DummySubtitleManager()
    processor = OutputLayerProcessor(subtitle_manager=subtitle_manager)
    for batch in projected:
        processor.process(OutputLayerInput(chunk_index=batch.chunk_id, sentence_segments=[], subtitle_batch=batch))
    assert subtitle_manager.calls == [(projected[0].chunk_id, 1)]
```

- [ ] **Step 4: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_output_integration.py -k non_owner_empty_cleanup -v`
Expected: FAIL

- [ ] **Step 5: 等待 wgh 手动提交（当前仅测试红灯）**

### Task 1.2: 字幕管理器新增“scope 清理 + group 写入”契约测试

**Files:**
- Modify: `backend/tests/test_streaming_subtitle_replace_chunk_empty.py`

- [ ] **Step 1: 写失败测试（replace_chunk_batch 支持 replace_scope_chunk_ids）**

```python
def test_replace_chunk_batch_scope_cleanup_then_write_output_group():
    batch = SubtitleBatch(
        chunk_id="ow-window-0033",
        chunk_index=None,
        items=(item("seg-1", "ow-window-0033", 240.0, 242.0, "demo"),),
        diagnostics={"projection": {"projection_mode": "window_group", "replace_scope_chunk_ids": ["32", "33"]}},
    )
    manager.replace_chunk_batch(batch)
    assert manager.get_chunk_sentence_indices("32") == []
    assert manager.get_chunk_sentence_indices("33") == []
    assert len(manager.get_chunk_sentence_indices("ow-window-0033")) == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/test_streaming_subtitle_replace_chunk_empty.py -k scope_cleanup -v`
Expected: FAIL

- [ ] **Step 3: 等待 wgh 手动提交（当前仅测试红灯）**

### Task 1.3: 前端投影增加 source scope 清理与迟到 draft 拒收测试

**Files:**
- Modify: `frontend/tests/stores/editor/editorEventProjector.spec.js`
- Modify: `frontend/tests/regression/phasef-editor-realtime-authoritative-reload.regression.spec.js`

- [ ] **Step 1: 写失败测试（group 定稿可清理 source scope 旧草稿）**

```javascript
it('projectReplaceChunk 应按 source_chunk_ids 清理旧逻辑 chunk 并写入 output-group 定稿', async () => {
  projector.projectDraft({ index: 0, chunk_index: 32, sentence: { text: 'd1', start: 0, end: 1 } })
  projector.projectDraft({ index: 1, chunk_index: 33, sentence: { text: 'd2', start: 1, end: 2 } })
  projector.projectReplaceChunk({
    chunk_uid: 'ow-window-0033',
    source_chunk_ids: ['32', '33'],
    new_indices: [10],
    sentences: [{ text: 'final', start: 0, end: 2, source: 'whisper' }],
  })
  await new Promise((r) => queueMicrotask(r))
  expect(docStore.getCount()).toBe(1)
})
```

- [ ] **Step 2: 写失败测试（scope 已定稿后拒收迟到 draft）**

```javascript
it('projectDraft 不应接收 source scope 已定稿后的迟到草稿', async () => {
  projector.projectReplaceChunk({
    chunk_uid: 'ow-window-0033',
    source_chunk_ids: ['32', '33'],
    new_indices: [10],
    sentences: [{ text: 'final', start: 0, end: 2 }],
  })
  await new Promise((r) => queueMicrotask(r))
  const accepted = projector.projectDraft({ index: 99, chunk_index: 32, sentence: { text: 'late', start: 0, end: 1 } })
  expect(accepted).toBe(false)
})
```

- [ ] **Step 3: 运行测试确认失败**

Run: `cd frontend; npm run test:run -- tests/stores/editor/editorEventProjector.spec.js -t "source_chunk_ids"`
Expected: FAIL

- [ ] **Step 4: 等待 wgh 手动提交（当前仅测试红灯）**

---

## Chunk 2: 后端输出主语义切换为 output-group

### Task 2.1: OutputProjector 改为 window-group 输出

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/output_projection/output_projector.py`
- Test: `backend/tests/timeanchored_alignment/test_output_projection.py`

- [ ] **Step 1: 实现最小改造（只输出非空 window-group batch）**

```python
group_chunk_id = f"ow-{validated.window_id}"
projection_meta = {
    "projection_mode": "window_group",
    "owner_chunk_id": str(validated.owner_chunk_id),
    "source_chunk_ids": [str(x) for x in validated.source_chunk_ids],
    "source_chunk_indices": [int(x) for x in validated.source_chunk_indices],
    "replace_scope_chunk_ids": [str(x) for x in validated.source_chunk_ids],
}
SubtitleBatch(chunk_id=group_chunk_id, chunk_index=None, items=group_items, diagnostics={...})
```

- [ ] **Step 2: 运行 Task 1.1 测试并转绿**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_output_projection.py -k window_group -v`
Expected: PASS

- [ ] **Step 3: 等待 wgh 手动提交**

### Task 2.2: StreamingSubtitleManager 增加 scope-aware replace，移除 non-owner 空清理触发

**Files:**
- Modify: `backend/app/services/streaming_subtitle.py`
- Test: `backend/tests/test_streaming_subtitle_replace_chunk_empty.py`

- [ ] **Step 1: 实现 `replace_chunk_scope(...)`（清理 scope，再写 target group）**

```python
def replace_chunk_scope(self, *, target_chunk_ref, scope_chunk_refs, sentences):
    scope_indices = []
    for ref in scope_chunk_refs:
        scope_indices.extend(self.get_chunk_sentence_indices(ref))
    self._remove_chunk_alias_mappings(...)  # for each scope ref
    return self.replace_chunk(target_chunk_ref, sentences)
```

- [ ] **Step 2: 改造 `replace_chunk_batch(...)` 路由**

```python
if projection_mode == "window_group" and replace_scope_chunk_ids:
    return self.replace_chunk_scope(
        target_chunk_ref=subtitle_batch.chunk_id,
        scope_chunk_refs=replace_scope_chunk_ids,
        sentences=sentences,
    )
# 删除 projection_non_owner_cleanup 分支
```

- [ ] **Step 3: 运行 Task 1.2 测试并转绿**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/test_streaming_subtitle_replace_chunk_empty.py -k scope_cleanup -v`
Expected: PASS

- [ ] **Step 4: 运行回归测试（空替换语义仍兼容 legacy）**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/test_streaming_subtitle_replace_chunk_empty.py -v`
Expected: PASS

- [ ] **Step 5: 等待 wgh 手动提交**

### Task 2.3: 输出 payload 增加 source scope 可观测字段

**Files:**
- Modify: `backend/app/services/textflow/output_dispatch_adapter.py`
- Test: `backend/tests/timeanchored_alignment/test_output_integration.py`

- [ ] **Step 1: 在 output payload 增加 `source_chunk_ids` 字段**

```python
projection = dict(subtitle_batch.diagnostics.get("projection") or {})
payload["source_chunk_ids"] = list(projection.get("replace_scope_chunk_ids") or projection.get("source_chunk_ids") or [])
```

- [ ] **Step 2: 运行输出链路集成测试并转绿**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_output_integration.py -k non_owner_empty_cleanup -v`
Expected: PASS

- [ ] **Step 3: 等待 wgh 手动提交**

---

## Chunk 3: 前端按 source scope 收口（与旧 chunk 等价性解耦）

### Task 3.1: projector 支持 source scope 清理

**Files:**
- Modify: `frontend/src/stores/editor/editorEventProjector.js`
- Test: `frontend/tests/stores/editor/editorEventProjector.spec.js`

- [ ] **Step 1: 实现 source scope 解析与清理集合合并**

```javascript
const sourceScopeKeys = (data?.source_chunk_ids || []).map((v) => normalizeLogicalChunkKey(v)).filter((v) => v !== null)
const sourceScopeLocalIds = sourceScopeKeys.flatMap((k) => collectLocalIdsForLogicalChunk(docStore, k))
const oldLocalIds = uniqueLocalIds([...oldLocalIdsFromIndices, ...oldLocalIdsFromPrimaryKeys, ...sourceScopeLocalIds, ...collectLocalIdsForLogicalChunk(docStore, logicalChunkKey)])
```

- [ ] **Step 2: 允许“仅清理”替换事件（`sentences=[]` 但有 source scope）**

```javascript
if (replacementItems.length === 0 && sourceScopeKeys.length === 0) return false
```

- [ ] **Step 3: 运行 Task 1.3 第一条测试并转绿**

Run: `cd frontend; npm run test:run -- tests/stores/editor/editorEventProjector.spec.js -t "按 source_chunk_ids 清理"`
Expected: PASS

- [ ] **Step 4: 等待 wgh 手动提交**

### Task 3.2: projector 增加 finalized scope 拒收迟到 draft

**Files:**
- Modify: `frontend/src/stores/editor/editorEventProjector.js`
- Modify: `frontend/src/views/EditorView.vue`
- Test: `frontend/tests/stores/editor/editorEventProjector.spec.js`

- [ ] **Step 1: 增加 scope finalized 状态与 draft 拒收判断**

```javascript
const finalizedScopeKeys = new Set()
sourceScopeKeys.forEach((k) => finalizedScopeKeys.add(k))
if (logicalChunkKey !== null && finalizedScopeKeys.has(logicalChunkKey)) return false
```

- [ ] **Step 2: 增加 session 初始化重置（避免跨任务污染）**

```javascript
function resetProjectionState() {
  finalizedScopeKeys.clear()
}
```

- [ ] **Step 3: 在 `EditorView` 初始化 editor core 时调用 reset**

```javascript
editorEventProjector.resetProjectionState?.()
```

- [ ] **Step 4: 运行 Task 1.3 第二条测试并转绿**

Run: `cd frontend; npm run test:run -- tests/stores/editor/editorEventProjector.spec.js -t "迟到草稿"`
Expected: PASS

- [ ] **Step 5: 运行回归测试并转绿**

Run: `cd frontend; npm run test:run -- tests/regression/phasef-editor-realtime-authoritative-reload.regression.spec.js`
Expected: PASS

- [ ] **Step 6: 等待 wgh 手动提交**

### Task 3.3: legacy store 兼容 `source_chunk_ids`（非 V2 防回退）

**Files:**
- Modify: `frontend/src/stores/projectStore.js`
- Test: `frontend/tests/editor/projectStore.spec.js`

- [ ] **Step 1: 在 `replaceChunk(chunk_id, sentences, options)` 增加 scope 清理入口**

```javascript
const scope = Array.isArray(options?.source_chunk_ids) ? options.source_chunk_ids : []
scope.forEach((scopeChunkId) => clearChunkSubtitles(scopeChunkId))
```

- [ ] **Step 2: 新增兼容测试并运行**

Run: `cd frontend; npm run test:run -- tests/editor/projectStore.spec.js -t "source_chunk_ids"`
Expected: PASS

- [ ] **Step 3: 等待 wgh 手动提交**

---

## Chunk 4: 联调验收、文档同步与发布闸门

### Task 4.1: 后端+前端关键回归全绿

**Files:**
- Test only (no code)

- [ ] **Step 1: 运行后端聚合测试**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_output_projection.py backend/tests/timeanchored_alignment/test_output_integration.py backend/tests/test_streaming_subtitle_replace_chunk_empty.py -v`
Expected: PASS

- [ ] **Step 2: 运行前端聚合测试**

Run: `cd frontend; npm run test:run -- tests/stores/editor/editorEventProjector.spec.js tests/regression/phasef-editor-realtime-authoritative-reload.regression.spec.js tests/editor/projectStore.spec.js`
Expected: PASS

- [ ] **Step 3: 构建前端**

Run: `cd frontend; npm run build`
Expected: Build 成功（无阻断错误）

- [ ] **Step 4: 等待 wgh 手动提交**

### Task 4.2: llmdoc 覆写同步（禁止追加补丁式注释）

**Files:**
- Modify: `llmdoc/architecture/async-dual-pipeline.md`
- Modify: `llmdoc/architecture/sse-events.md`
- Modify: `llmdoc/changelog.md`

- [ ] **Step 1: 更新架构文档（输出主语义改为 output-group + source scope）**

```md
- 输出层不再要求与旧 VAD chunk 等价。
- replace_chunk 载荷新增 source_chunk_ids，消费侧按 scope 收口。
```

- [ ] **Step 2: 更新事件契约文档（replace_chunk 字段与空事件语义）**

```md
subtitle.replace_chunk:
- chunk_uid: output-group id
- source_chunk_ids: 需要清理/拒收草稿的 processing chunk 作用域
```

- [ ] **Step 3: 在 changelog 记录迁移与回归命令**

- [ ] **Step 4: 等待 wgh 手动提交**

### Task 4.3: 目标问题复现用例验收

**Files:**
- Runtime artifacts: `jobs/p-20260326-032327-tr-cut-994d`

- [ ] **Step 1: 对同类输入运行一次完整链路（或回放测试）**

Run: `python backend/scripts/<项目现有任务入口>.py --job-id p-20260326-032327-tr-cut-994d`
Expected: 产出链路成功（无 non-owner 空清理日志）

- [ ] **Step 2: 检查 #48/#49 间缺口**

Run: `python backend/scripts/<项目现有SRT检查脚本>.py --srt "F:/video_to_srt_gpu/jobs/p-20260326-032327-tr-cut-994d/日本毒可乐-cut.srt"`
Expected: 缺口消失或降为正常停顿

- [ ] **Step 3: 抽查 checkpoint 中映射与终态完整性**

Run: `python - <<'PY'
import json, pathlib
p = pathlib.Path(r'F:/video_to_srt_gpu/jobs/p-20260326-032327-tr-cut-994d/checkpoint.json')
j = json.loads(p.read_text(encoding='utf-8'))
print('finalized', len(j.get('transcription',{}).get('finalized_indices',[])))
print('total', j.get('preprocessing',{}).get('total_chunks'))
PY`
Expected: `finalized == total`，且字幕文本不缺段

- [ ] **Step 4: 等待 wgh 手动提交与发布决策**
