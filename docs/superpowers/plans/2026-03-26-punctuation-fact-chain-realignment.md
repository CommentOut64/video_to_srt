# 标点事实链回归总纲 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `PunctuationEvidence -> PunctuationFact -> BoundaryHint -> Decision` 重新成为唯一主链，彻底消除“快流有标点结果但 L6 无标点事实”的偏离。

**Architecture:** 保持 `window-first`，将标点事实输入从“慢流文本字符扫描唯一来源”升级为“字符扫描 + 显式标点位置事实”的双通道收口，并在 Preparation 层统一合并。Decision 前禁止再依赖字符串回推边界，所有边界由事实链显式传递。删除遗留 compat 包装路径，统一通过 `DecisionIngressPackage` 进入 Decision。

**Tech Stack:** Python 3.10+, dataclasses, 现有 `timeanchored_alignment`、`punctuation_processor`、`decision_layer`、pytest。

---

## Chunk 1: 锁定偏离并建立回归门禁

### Task 1.1: 建立“slow 选文 + 标点不断链”失败测试

**Files:**
- Create: `backend/tests/timeanchored_alignment/test_punctuation_fact_chain_contract.py`
- Modify: `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`
- Test: `backend/tests/timeanchored_alignment/test_punctuation_fact_chain_contract.py`

- [ ] **Step 1: 写失败测试（Preparation 级）**

```python
def test_preparation_keeps_punctuation_evidence_when_slow_text_is_unpunctuated_but_punct_track_exists():
    # 1) chosen_source=slow
    # 2) slow text_clean 无句末标点
    # 3) punct_track.positions 含句末标点
    # 期望: preparation.slow_text.punctuation_evidences 非空
    assert len(package.slow_text.punctuation_evidences) > 0
```

- [ ] **Step 2: 写失败测试（AnchorMount 级）**

```python
def test_anchor_mount_receives_sentence_end_punctuation_facts_from_preparation():
    # 期望: DecisionIngressPackage.punctuation_facts 含 sentence_end
    assert any(f.punct_class == "sentence_end" for f in ingress.punctuation_facts)
```

- [ ] **Step 3: 运行失败测试确认当前偏离存在**

Run: `cd backend && python -m pytest tests/timeanchored_alignment/test_punctuation_fact_chain_contract.py -v`
Expected: FAIL（当前实现 `punctuation_evidences` 为 0）

- [ ] **Step 4: 等待 wgh 手动确认是否进入实现阶段**

### Task 1.2: 加入运行时断链哨兵指标

**Files:**
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- Modify: `backend/app/services/timeanchored_alignment/preparation/assembler.py`
- Test: `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`

- [ ] **Step 1: 写失败测试（日志/诊断字段）**

```python
def test_alignment_stage_emits_punctuation_chain_health_metrics():
    # 期望包含:
    # chosen_source, punct_track_positions_total, preparation_punctuation_count,
    # punctuation_chain_broken_flag
    assert payload["punctuation_chain_broken_flag"] in (0, 1)
```

- [ ] **Step 2: 运行测试确认缺失字段**

Run: `cd backend && python -m pytest tests/timeanchored_alignment/test_alignment_stage_routing.py -k punctuation_chain -v`
Expected: FAIL

- [ ] **Step 3: 等待 wgh 手动确认是否进入实现阶段**

---

## Chunk 2: 在 Preparation 层重建“标点事实双通道”

### Task 2.1: 扩展 Preparation 输入契约，允许显式标点事实进入

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/preparation/assembler.py`
- Modify: `backend/app/services/timeanchored_alignment/preparation/contracts.py`
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- Test: `backend/tests/timeanchored_alignment/test_alignment_preparation.py`

- [ ] **Step 1: 写失败测试（assembler.prepare 新参数行为）**

```python
def test_assembler_accepts_external_punct_track_and_merges_evidence():
    package = assembler.prepare(..., external_punct_track=punct_track)
    assert len(package.slow_text.punctuation_evidences) > 0
```

- [ ] **Step 2: 运行测试确认接口尚未支持**

Run: `cd backend && python -m pytest tests/timeanchored_alignment/test_alignment_preparation.py -k external_punct -v`
Expected: FAIL

- [ ] **Step 3: 最小实现 `prepare(..., external_punct_track=None)`**

```python
def prepare(..., external_punct_track: PunctTrack | None = None):
    lexical_evidences = ...
    external_evidences = ...
    punctuation_evidences = merge_and_dedupe(lexical_evidences, external_evidences)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd backend && python -m pytest tests/timeanchored_alignment/test_alignment_preparation.py -k external_punct -v`
Expected: PASS

- [ ] **Step 5: 等待 wgh 手动提交**

### Task 2.2: 提炼公共 remap 工具，统一坐标映射口径

**Files:**
- Create: `backend/app/services/punctuation/punct_position_mapper.py`
- Modify: `backend/app/services/punctuation/punctuation_processor.py`
- Modify: `backend/app/services/timeanchored_alignment/preparation/punctuation_evidence_builder.py`
- Test: `backend/tests/test_punctuation_position_mapper.py`

- [ ] **Step 1: 写失败测试（严格/宽松 remap 行为）**

```python
def test_punct_position_mapper_tolerant_mode_can_remap_equivalent_text_refs():
    mapped = mapper.remap(...)
    assert mapped
```

- [ ] **Step 2: 运行失败测试**

Run: `cd backend && python -m pytest tests/test_punctuation_position_mapper.py -v`
Expected: FAIL

- [ ] **Step 3: 实现公共 mapper 并替换两处重复逻辑**

```python
class PunctPositionMapper:
    def remap(self, source_ref: str, target_ref: str, positions: Sequence[PuncPosition], mode: str) -> list[PuncPosition]:
        ...
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd backend && python -m pytest tests/test_punctuation_position_mapper.py -v`
Expected: PASS

- [ ] **Step 5: 等待 wgh 手动提交**

---

## Chunk 3: 清理偏离路径，收敛到文档定义的单一路径

### Task 3.1: 删除 `finalize_timeanchored_stream` 旧 compat 包装

**Files:**
- Modify: `backend/app/pipelines/dual_pipeline/services/textflow_facade_service.py`
- Modify: `backend/app/pipelines/dual_pipeline/implementation.py`
- Test: `backend/tests/timeanchored_alignment/test_no_finalize_timeanchored_stream_wrapper.py`

- [ ] **Step 1: 写失败测试（禁止调用旧包装）**

```python
def test_timeanchored_pipeline_does_not_use_finalize_timeanchored_stream_wrapper():
    # 通过 monkeypatch 旧方法并断言未被调用
    assert legacy_wrapper_call_count == 0
```

- [ ] **Step 2: 运行失败测试**

Run: `cd backend && python -m pytest tests/timeanchored_alignment/test_no_finalize_timeanchored_stream_wrapper.py -v`
Expected: FAIL

- [ ] **Step 3: 移除旧入口并改为直接走 DecisionIngressAdapter**

```python
# 删除或断路 finalize_timeanchored_stream
# 对齐阶段只保留 DecisionIngressPackage -> Decision
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd backend && python -m pytest tests/timeanchored_alignment/test_no_finalize_timeanchored_stream_wrapper.py -v`
Expected: PASS

- [ ] **Step 5: 等待 wgh 手动提交**

### Task 3.2: 强化 Decision 前事实链断裂 fail-closed

**Files:**
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/service.py`
- Test: `backend/tests/timeanchored_alignment/test_punctuation_fact_chain_fail_closed.py`

- [ ] **Step 1: 写失败测试（有标点输入但事实链为空时必须告警/标记）**

```python
def test_fail_closed_when_punct_track_exists_but_fact_chain_is_empty():
    assert metrics["punctuation_chain_broken_flag"] == 1
```

- [ ] **Step 2: 运行失败测试**

Run: `cd backend && python -m pytest tests/timeanchored_alignment/test_punctuation_fact_chain_fail_closed.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 fail-closed 标记与结构化诊断**

```python
if punct_track_has_sentence_end and punctuation_fact_count == 0:
    metrics["punctuation_chain_broken_flag"] = 1
    metrics["punctuation_chain_broken_reason"] = "punct_track_not_projected_to_preparation"
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd backend && python -m pytest tests/timeanchored_alignment/test_punctuation_fact_chain_fail_closed.py -v`
Expected: PASS

- [ ] **Step 5: 等待 wgh 手动提交**

---

## Chunk 4: 端到端验证与回归门禁

### Task 4.1: 任务级回归测试（复现本次偏离场景）

**Files:**
- Create: `backend/tests/e2e/test_job_punctuation_fact_chain_regression.py`
- Test: `backend/tests/e2e/test_job_punctuation_fact_chain_regression.py`

- [ ] **Step 1: 写失败测试（mock/fixture job）**

```python
def test_job_has_punctuation_points_then_preparation_should_not_be_zero():
    # 断言:
    # 1) 有 split_points
    # 2) preparation_punctuation_count > 0
    # 3) default_splitter 占比低于阈值
    assert ratio_default_splitter < 0.8
```

- [ ] **Step 2: 运行失败测试**

Run: `cd backend && python -m pytest tests/e2e/test_job_punctuation_fact_chain_regression.py -v`
Expected: FAIL

- [ ] **Step 3: 修正断言口径并保证稳定**

```python
# 使用固定 fixture，避免依赖线上模型波动
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd backend && python -m pytest tests/e2e/test_job_punctuation_fact_chain_regression.py -v`
Expected: PASS

- [ ] **Step 5: 等待 wgh 手动提交**

### Task 4.2: 全链路测试回归

**Files:**
- Test: `backend/tests/timeanchored_alignment/`
- Test: `backend/tests/textflow/`

- [ ] **Step 1: 运行 timeanchored 测试集**

Run: `cd backend && python -m pytest tests/timeanchored_alignment -v`
Expected: PASS

- [ ] **Step 2: 运行 textflow 测试集**

Run: `cd backend && python -m pytest tests/textflow -v`
Expected: PASS

- [ ] **Step 3: 运行关键集成测试**

Run: `cd backend && python -m pytest tests/test_punctuation_runtime_overrides.py tests/test_runtime_param_integration.py -v`
Expected: PASS

- [ ] **Step 4: 等待 wgh 手动提交**

---

## Chunk 5: 文档回正与防再偏离机制

### Task 5.1: 重写总纲对应章节，清除偏离叙事

**Files:**
- Modify: `devdoc/v3.3.0/anchor/10-前半链统一分层重构总纲.md`
- Modify: `llmdoc/architecture/preparation-language-adapter.md`
- Modify: `llmdoc/architecture/dual-flow-finalization.md`

- [ ] **Step 1: 文档替换（不是追加）**

```text
更新“标点事实链”章节为当前真实实现：
PunctTrack -> Preparation Evidence Merge -> PunctuationFact -> BoundaryHint -> Decision
```

- [ ] **Step 2: 删除僵尸描述**

```text
删除仍暗示“slow 文本字符扫描是唯一来源”的段落
删除 finalize_timeanchored_stream 仍为主链的段落
```

- [ ] **Step 3: 文档一致性回读**

Run: `rg -n "finalize_timeanchored_stream|window -> chunk|fallback_punctuation_positions.*主真相" devdoc llmdoc`
Expected: 无与新主链冲突描述

### Task 5.2: 增加 CI 漂移守卫

**Files:**
- Create: `backend/tests/architecture/test_punctuation_fact_chain_arch_guard.py`
- Modify: `backend/pytest.ini`（如需注册 marker）

- [ ] **Step 1: 写守卫测试**

```python
def test_arch_guard_no_legacy_finalize_wrapper_usage():
    # 静态扫描关键路径，禁止回引旧包装
    assert "finalize_timeanchored_stream(" not in code
```

- [ ] **Step 2: 运行守卫测试**

Run: `cd backend && python -m pytest tests/architecture/test_punctuation_fact_chain_arch_guard.py -v`
Expected: PASS

- [ ] **Step 3: 等待 wgh 手动提交**

---

## 全量验证门禁

- [ ] 运行：`cd backend && python -m pytest tests/timeanchored_alignment tests/textflow tests/architecture -v`
- [ ] 运行：`cd backend && python -m pytest tests/e2e/test_job_punctuation_fact_chain_regression.py -v`
- [ ] 运行：`cd backend && python -m pytest tests/test_punctuation_runtime_overrides.py tests/test_runtime_param_integration.py -v`
- [ ] 对照真实任务回放检查：`preparation_punctuation_count` 不再长期为 0，`default_splitter` 占比下降
- [ ] 文档同步完成且无追加式补丁描述

---

**执行注意**
- 严格遵守 TDD：先失败测试，再最小实现，再回归验证。
- 任何提交由 wgh 手动执行；计划执行过程中不自动 commit。
- 若出现“修复 2 次仍无法通过”情况，暂停实现并回到总纲重新收敛边界。
