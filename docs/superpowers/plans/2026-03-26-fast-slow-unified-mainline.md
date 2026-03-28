# 快慢流统一主线（Preparation→Alignment→Decision→Output）Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立单一语义主线：快流与慢流都先进入 Preparation，再由 Alignment 选边，进入 Decision 后不再区分快慢，最终统一 Output 与标点处理。

**Architecture:** 以“单主线、双入口”为核心：`fast(chunk)` 与 `slow(window)` 只在输入粒度不同，语义处理链完全一致。禁止新增“兼容层 + 特判分支”绕过主链；优先复用现有 `AlignmentPreparationAssembler`、`AnchorMountAlignmentService`、`DecisionIngressAdapter`、`OutputProjector(window_group)` 框架。

**Tech Stack:** Python 3.10+、现有 dual_pipeline/timeanchored_alignment/textflow 架构、pytest、前端 Vitest 回归链路。

---

## 0. 架构约束（必须遵守）

- `fast` 与 `slow` 必须共用同一语义链：标准化、逆标准化、结构保护、标点 evidence、切分决策、输出投影。
- `is_fast_only_mode` 不得再影响切分/标点语义，只可保留性能优化或观测用途。
- 所有模式都通过 Alignment 层接口出 Decision；极速模式允许“单路短路选边”，但不能旁路统一链路。
- 不新增新兼容层；仅在现有主干函数中收敛分叉并删除失效 legacy 路径。

---

## 1. 模式映射（最终行为定义）

1. **极速模式（sensevoice_only）**
   - `fast -> preparation(chunk) -> alignment(单路短路=fast) -> decision -> output`
2. **复核模式（当前阶段按双流执行）**
   - `fast+slow -> preparation(同语义) -> alignment(选边) -> decision -> output`
3. **双流模式**
   - 与复核模式一致。

---

## 2. 复用优先的切入点

- 复用 Preparation 主实现：`backend/app/services/timeanchored_alignment/preparation/assembler.py`
- 复用 Alignment 主实现：`backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py` + `anchor_mount/service.py`
- 复用 Decision 主实现：`backend/app/pipelines/dual_pipeline/services/textflow_facade_service.py`
- 复用 Output 主实现：`backend/app/services/timeanchored_alignment/output_projection/output_projector.py` + `output_dispatch_adapter.py` + `streaming_subtitle.py`
- 仅删除/收敛分叉：
  - `alignment_stage_service.run` 内 fast direct 早退分支
  - `whisper_skipped` 直接 finalize 分支
  - `run_collection_scoring_decision_once(... is_fast_only_mode=True)` 语义差异点

---

## 3. 文件结构与职责边界

**核心代码**
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
  - 统一模式路由；快慢都进入 preparation+alignment；删除语义旁路。
- Modify: `backend/app/pipelines/dual_pipeline/services/textflow_facade_service.py`
  - 去除 `is_fast_only_mode` 的语义分叉（切分/标点策略统一）。
- Modify: `backend/app/pipelines/dual_pipeline/implementation.py`
  - 收敛 `_run_collection_scoring_decision_once` 的调用语义，保留接口兼容但不再传递语义分叉。
- Modify: `backend/app/services/timeanchored_alignment/preparation/assembler.py`
  - 复用现有 `external_punct_track` 合并能力，补足 fast 输入组装场景。
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/fallback_gate.py`
  - 保持质量门职责不变，仅用于选边，不再触发独立语义通道。

**输出与契约（以复用为主）**
- Keep/Verify: `backend/app/services/timeanchored_alignment/output_projection/output_projector.py`
- Keep/Verify: `backend/app/services/textflow/output_dispatch_adapter.py`
- Keep/Verify: `backend/app/services/streaming_subtitle.py`

**测试**
- Modify/Create: `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`
- Modify/Create: `backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py`
- Modify/Create: `backend/tests/timeanchored_alignment/test_alignment_preparation.py`
- Modify/Create: `backend/tests/timeanchored_alignment/test_punctuation_fact_chain_contract.py`
- Modify/Create: `backend/tests/timeanchored_alignment/test_punctuation_fact_chain_fail_closed.py`
- Modify/Create: `backend/tests/timeanchored_alignment/test_output_projection.py`
- Modify/Create: `backend/tests/timeanchored_alignment/test_output_integration.py`
- Modify: `backend/tests/textflow/test_job_p20260323_regressions.py`
- Verify: `frontend/tests/stores/editor/editorEventProjector.spec.js`
- Verify: `frontend/tests/regression/phasef-editor-realtime-authoritative-reload.regression.spec.js`

**文档同步**
- Modify: `llmdoc/architecture/async-dual-pipeline.md`
- Modify: `llmdoc/architecture/sse-events.md`
- Modify: `llmdoc/changelog.md`

---

## Chunk 1: 先锁路线（红灯门禁）

### Task 1.1: 模式路由契约测试（禁止旁路）

**Files:**
- Modify: `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`

- [ ] **Step 1: 写失败测试（sensevoice_only 仍走 preparation+alignment 接口）**
```python
def test_sensevoice_only_still_enters_unified_preparation_alignment_pipeline():
    # 断言：不调用 fast_direct finalize 旁路；调用统一 preparation/selection 接口
    assert unified_path_called is True
    assert fast_direct_shortcut_called is False
```

- [ ] **Step 2: 写失败测试（whisper_skipped 走单路选边而非旧 finalize）**
```python
def test_whisper_skipped_uses_single_candidate_alignment_not_legacy_finalize():
    assert single_candidate_alignment_called is True
    assert legacy_finalize_called is False
```

- [ ] **Step 3: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_alignment_stage_routing.py -k "sensevoice_only or whisper_skipped" -v`
Expected: FAIL

- [ ] **Step 4: 等待 wgh 手动提交（仅红灯）**

### Task 1.2: 语义统一门禁测试（禁止 fast-only 语义分叉）

**Files:**
- Modify: `backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py`
- Modify: `backend/tests/timeanchored_alignment/test_stage_service.py`

- [ ] **Step 1: 写失败测试（选 fast/选 slow 下切分规则集一致）**
```python
def test_selected_fast_and_selected_slow_share_same_decision_semantics():
    assert decision_semantic_signature_fast == decision_semantic_signature_slow
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py -k decision_semantics -v`
Expected: FAIL

- [ ] **Step 3: 等待 wgh 手动提交（仅红灯）**

---

## Chunk 2: Preparation 双入口统一（chunk/window 同语义）

### Task 2.1: fast chunk 输入接入 Preparation，不新增兼容层

**Files:**
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- Modify: `backend/app/services/timeanchored_alignment/preparation/assembler.py`
- Modify: `backend/tests/timeanchored_alignment/test_alignment_preparation.py`

- [ ] **Step 1: 写失败测试（fast 输入可生成与 slow 同结构 preparation 包）**
```python
def test_fast_chunk_can_build_preparation_package_with_same_contract_as_slow_window():
    assert package.slow_text.window_text.text != ""
    assert isinstance(package.slow_text.punctuation_evidences, tuple)
    assert package.compat is not None
```

- [ ] **Step 2: 实现最小改造（复用现有 assembler.prepare）**
```python
# alignment_stage_service:
# - 为 fast 构造单 chunk ReadySlowWindow/WindowTimeBase 输入
# - 调用同一 assembler.prepare(... external_punct_track=ctx.punct_track)
```

- [ ] **Step 3: 运行测试并转绿**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_alignment_preparation.py -k "fast_chunk or external_punct" -v`
Expected: PASS

- [ ] **Step 4: 等待 wgh 手动提交**

### Task 2.2: 标点事实链在 fast/slow 都完整

**Files:**
- Modify: `backend/tests/timeanchored_alignment/test_punctuation_fact_chain_contract.py`
- Modify: `backend/tests/timeanchored_alignment/test_job_punctuation_fact_chain_regression.py`

- [ ] **Step 1: 写失败测试（fast 路也必须有 preparation punctuation evidences）**
```python
def test_fast_preparation_keeps_punctuation_evidences():
    assert preparation_punctuation_count > 0
```

- [ ] **Step 2: 写失败测试（punctuation_facts 不得在 fast 路断链）**
```python
def test_fast_selected_path_produces_punctuation_facts():
    assert punctuation_fact_count > 0
```

- [ ] **Step 3: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_punctuation_fact_chain_contract.py -v`
Expected: FAIL

- [ ] **Step 4: 等待 wgh 手动提交（仅红灯）**

---

## Chunk 3: Alignment 统一选边（含极速单路短路）

### Task 3.1: 统一候选消费模型，移除 fast direct 语义旁路

**Files:**
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- Modify: `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`

- [ ] **Step 1: 实现候选模型**
```python
# candidates = [fast_candidate?, slow_candidate?]
# if len(candidates)==1: 走 alignment 单路短路选择
# else: 走既有选边逻辑
# 禁止 _commit_fast_direct_result 直接产出终稿
```

- [ ] **Step 2: 运行路由测试并转绿**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_alignment_stage_routing.py -v`
Expected: PASS

- [ ] **Step 3: 等待 wgh 手动提交**

### Task 3.2: 质量门只影响“选边”，不影响“语义链”

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/fallback_gate.py`
- Modify: `backend/tests/timeanchored_alignment/test_punctuation_fact_chain_fail_closed.py`

- [ ] **Step 1: 写失败测试（anchor_mount_should_fallback 触发后仍走统一 decision 链）**
```python
def test_quality_gate_fallback_keeps_unified_decision_semantics():
    assert selected_route in {"fast", "slow"}
    assert decision_entrypoint == "unified"
```

- [ ] **Step 2: 实现最小改造（质量门信号仅写入选边上下文）**

- [ ] **Step 3: 运行测试并转绿**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_punctuation_fact_chain_fail_closed.py -v`
Expected: PASS

- [ ] **Step 4: 等待 wgh 手动提交**

---

## Chunk 4: Decision 语义统一（去 fast-only 分叉）

### Task 4.1: 清理 `is_fast_only_mode` 语义影响点

**Files:**
- Modify: `backend/app/pipelines/dual_pipeline/services/textflow_facade_service.py`
- Modify: `backend/app/pipelines/dual_pipeline/implementation.py`
- Modify: `backend/tests/timeanchored_alignment/test_stage_service.py`

- [ ] **Step 1: 写失败测试（fast/slow 输入下 soft-cut 规则参数一致）**
```python
def test_soft_cut_semantic_config_not_switched_by_fast_only_flag():
    assert soft_cut_config_fast == soft_cut_config_slow
```

- [ ] **Step 2: 实现最小改造**
```python
# 保留 is_fast_only_mode 参数（兼容调用面）
# 但不再用于开启/关闭 CJK weak punct / semantic anchors / fast_draft 特殊阈值语义
```

- [ ] **Step 3: 运行测试并转绿**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_stage_service.py -k soft_cut -v`
Expected: PASS

- [ ] **Step 4: 等待 wgh 手动提交**

---

## Chunk 5: Output 统一契约复用（不再额外分叉）

### Task 5.1: 统一输出路径验证（window_group + source_scope）

**Files:**
- Verify/Adjust: `backend/app/services/timeanchored_alignment/output_projection/output_projector.py`
- Verify/Adjust: `backend/app/services/textflow/output_dispatch_adapter.py`
- Verify/Adjust: `backend/app/services/streaming_subtitle.py`
- Modify: `backend/tests/timeanchored_alignment/test_output_projection.py`
- Modify: `backend/tests/timeanchored_alignment/test_output_integration.py`
- Modify: `backend/tests/textflow/test_job_p20260323_regressions.py`

- [ ] **Step 1: 写失败测试（fast 选中时也产出 window_group）**
```python
def test_fast_selected_output_still_uses_window_group_projection_contract():
    assert projection_mode == "window_group"
    assert source_chunk_ids
```

- [ ] **Step 2: 写失败测试（不再出现 legacy non-owner empty warning 路径）**
```python
def test_no_non_owner_empty_warning_under_unified_projection_contract():
    assert "定稿句子为空" not in warning_logs
```

- [ ] **Step 3: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_output_projection.py backend/tests/timeanchored_alignment/test_output_integration.py backend/tests/textflow/test_job_p20260323_regressions.py -k "window_group or non_owner" -v`
Expected: FAIL

- [ ] **Step 4: 最小实现并回归转绿**

- [ ] **Step 5: 等待 wgh 手动提交**

---

## Chunk 6: 联调回归与文档覆写同步

### Task 6.1: 后端回归矩阵

**Files:**
- Test only

- [ ] **Step 1: 运行 timeanchored 主测试集**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment -v`
Expected: PASS

- [ ] **Step 2: 运行 textflow 关键回归**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/textflow/test_job_p20260323_regressions.py backend/tests/textflow/test_decision_ingress_adapter.py -v`
Expected: PASS

- [ ] **Step 3: 运行路由与事实链重点集**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_alignment_stage_routing.py backend/tests/timeanchored_alignment/test_punctuation_fact_chain_contract.py backend/tests/timeanchored_alignment/test_punctuation_fact_chain_fail_closed.py -v`
Expected: PASS

- [ ] **Step 4: 等待 wgh 手动提交**

### Task 6.2: 前端契约回归（确认无需新增兼容实现）

**Files:**
- Test only

- [ ] **Step 1: 运行 editor projector 回归**

Run: `cd frontend; npm run test:run -- tests/stores/editor/editorEventProjector.spec.js`
Expected: PASS

- [ ] **Step 2: 运行实时回归**

Run: `cd frontend; npm run test:run -- tests/regression/phasef-editor-realtime-authoritative-reload.regression.spec.js`
Expected: PASS

- [ ] **Step 3: 运行 legacy store 兼容测试**

Run: `cd frontend; npm run test:run -- tests/editor/projectStore.spec.js -t "source_chunk_ids"`
Expected: PASS

- [ ] **Step 4: 等待 wgh 手动提交**

### Task 6.3: 文档覆写同步（禁止追加式补丁说明）

**Files:**
- Modify: `llmdoc/architecture/async-dual-pipeline.md`
- Modify: `llmdoc/architecture/sse-events.md`
- Modify: `llmdoc/changelog.md`

- [ ] **Step 1: 覆写模式路由章节（明确单主线、双入口）**
- [ ] **Step 2: 覆写事件契约章节（统一 `window_group/source_chunk_ids`）**
- [ ] **Step 3: 记录迁移影响与回归命令**
- [ ] **Step 4: 等待 wgh 手动提交**

---

## 全量发布门禁（必须全部满足）

- [ ] 所有模式都满足：`Preparation -> Alignment -> Decision -> Output` 主线不变。
- [ ] 不存在 `fast_direct/sensevoice_only/whisper_skipped` 语义旁路直出。
- [ ] `is_fast_only_mode` 不再改变切分/标点规则集。
- [ ] `punctuation_chain_broken_flag` 不再只是告警，必须被拦截或受控降级。
- [ ] 输出统一 `window_group + source_chunk_ids` 契约，回归全绿。
- [ ] 文档已覆写同步，未残留旧链路叙事。

---

## 执行备注

- 本计划严格遵循“少分支、少兼容层、复用现有框架”的实现策略。
- 本计划不要求新建并行架构，不引入额外 adapter 层，仅在现有主干函数收敛。
- 任何 commit 由 wgh 手动执行，执行过程中不自动提交。
