# Timeanchored 去 Slot / Token-Unit 主链收口 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留 `Window` 作为慢流最小处理单元的前提下，彻底移除 timeanchored 主链里的 `slot` 契约，让 `Decision` 直接基于 token 级时间事实与对齐层专属边界证据做唯一切分裁决，并把所有遗留兼容命名、半状态路由、文档旧叙事一次性清理干净。

**Architecture:** `Preparation` 只冻结 `token-unit + provenance + punctuation_evidences`，`AnchorMount` 只做 token 级时间挂载与对齐层专属 `BoundaryEvidence` 构造，`DecisionIngress` 只负责把 window 级真相投影成 `DecisionLayerInput`，`Decision` 是唯一切句层。上游不再用 `slot` 预离散化切分空间，也不允许通过后置 fallback 掩盖上游证据链问题。

**Tech Stack:** Python 3.10+、`dual_pipeline / timeanchored_alignment / textflow` 主链、pytest、Windows PowerShell、`llmdoc` 架构文档同步。

---

## 0. 当前状态快照（2026-03-28 承接点）

### 0.1 已完成的基线改造

- `Preparation` 已切到 `PreparedTokenUnit` 主契约，`source_attribution_binder.py` 已替换为 `token_provenance_binder.py`。
- `AnchorMount` 已切到 `token_units / AnchoredTokenUnit / token-indexed punctuation facts / boundary_evidences`。
- `DecisionIngressAdapter` 已不再给 `AnnotatedWord` 打 `slot_index / slot_id`。
- `Decision` 已删除 `slot -> word remap` 主链，直接消费 token 索引标点事实。
- `text_protection` 已不再传播 `slot_index / slot_id`。

### 0.2 已存在的回归检查点

以下检查点来自本轮之前已落地的工作，需要保留为历史基线，不在本计划内重复声明为“已验证完成”：

- 命令：

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' backend/tests/timeanchored_alignment/test_alignment_preparation_contracts.py backend/tests/timeanchored_alignment/test_anchor_mount_contracts.py backend/tests/textflow/test_decision_ingress_adapter.py backend/tests/textflow/test_decision_layer_ingress_context.py backend/tests/timeanchored_alignment/test_token_provenance_binder.py backend/tests/timeanchored_alignment/test_alignment_preparation.py backend/tests/timeanchored_alignment/test_anchor_mount_seed_discovery.py backend/tests/timeanchored_alignment/test_anchor_mount_punctuation_and_envelopes.py backend/tests/timeanchored_alignment/test_anchor_mount_service.py backend/tests/timeanchored_alignment/test_punctuation_fact_chain_contract.py backend/tests/timeanchored_alignment/test_text_protection_ext.py -v
```

- 结果：`46 passed`

### 0.3 当前源码已确认的残留问题

- `backend/app/services/timeanchored_alignment/anchor_mount/boundary_hint_assembler.py` 仍保留旧文件名/类名，且当前还在生成 `punctuation_sentence_end / gap_pause / blank_valley / speaker_change` 等重复证据，不符合最终契约。
- `backend/app/services/textflow/decision_ingress_adapter.py` 当前仍把 `canonical_candidate_boundaries` 设为 `tuple()`，对齐层专属证据没有进入 Decision 的 canonical 边界入口。
- `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py` 仍引用 `decision_ingress.tokens / boundary_hints` 等旧字段名，且 `should_fallback` 仍通过“先走 timeanchored 提交，再 force_fast 选边”的半状态方式生效。
- `backend/app/pipelines/dual_pipeline/services/anchor_mount_graph_renderer.py` 与其测试仍使用 `slot_id / slot_index` 术语。
- 以下文档仍描述旧 `slot-space / BoundaryHint / slot->word` 叙事：
  - `llmdoc/architecture/dual-flow-finalization.md`
  - `llmdoc/architecture/preparation-language-adapter.md`
  - `llmdoc/architecture/text-processing-layered-architecture.md`
- 尚未跑完 routing / graph / job regression 的更大范围回归：
  - `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`
  - `backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py`
  - `backend/tests/timeanchored_alignment/test_anchor_mount_graph_renderer.py`
  - `backend/tests/textflow/test_job_p20260323_regressions.py`

---

## 1. 根因机制与硬约束

### 1.1 根因判断

- 本问题的根因不是某个阈值偏高或偏低，而是 `slot` 让上游同时承担了 provenance、标点细分、发音 hint 细分和切分空间离散化这四种职责。
- 一旦 `slot` 在上游被切碎，`Decision` 即使名义上“自己打分”，也只能在被提前离散化后的窄空间里做选择，结果就会在“整段不切”和“碎切过度”之间来回摆动。
- 因此本计划只接受“从上游恢复正确事实链”的修正，不接受靠后处理 fallback、补丁式 merge、临时阈值兜底来掩盖问题。

### 1.2 `Window`、`token-unit`、`ingress` 的正确角色

- `Window` 继续作为慢流最小处理单元。
- `token-unit` 是进入 Decision 前的最小决策单元，不再使用 `slot` 作为中间语义层。
- `DecisionIngress` 只是“把 AnchorMount 的 window 级真相投影成 Decision 所需 DTO”的适配层，不负责生成可执行切点，不负责额外离散化，也不负责替下游做切分决策。

### 1.3 无 `slot` 时如何避免“长 Window 切不下去”

- 切分能力必须来自 `Decision` 对以下事实的统一打分，而不是来自上游把文本切碎：
  - `canonical_punctuation_facts`
  - `canonical_candidate_boundaries`（仅对齐层专属 reason）
  - `speaker_turns`
  - token 级时间映射自身可计算出的 `gap / duration / merge` 信号
- 如果一个长 `Window` 最终仍切不下去，排查方向必须是：
  - 上游 `token-unit` 是否冻结错误
  - AnchorMount 的时间包络是否失真
  - 标点事实是否没有正确映射到 token 索引
  - 对齐层专属边界证据是否缺失
- 不允许以“给 FinalSplitter 再加一个兜底切刀”作为解决方案。

### 1.4 最终允许进入 Decision 的输入

- `annotated_words`，来源于 `AnchoredTokenUnit -> AnnotatedWord` 的稳定投影
- `canonical_punctuation_facts(left_token_index/right_token_index)`
- `speaker_turns`
- `canonical_candidate_boundaries`，且仅允许以下对齐层专属 reason：
  - `lexical_boundary`
  - `anchor_block_close`

### 1.5 明确禁止的重复输入

- 上游禁止再把以下通用信号作为可执行边界输入注入 Decision：
  - `gap_pause`
  - `blank_valley`
  - `speaker_change`
  - `duration_guard`
  - `punctuation_sentence_end`
  - `punctuation_soft`

这些都必须由 `Decision` 自己基于 token 时间流、turn 流和 punctuation facts 统一计算。

---

## 2. 目标态数据流

```text
ReadySlowWindow
  -> Preparation(
       token_units,
       punctuation_evidences,
       pronunciation_hints,
       fast_hooks,
       provenance
     )
  -> AnchorMount(
       anchored_token_units,
       punctuation_facts[token-indexed],
       boundary_evidences[alignment-specific only],
       cross_chunk_locks
     )
  -> DecisionIngress(
       DecisionLayerInput
     )
  -> Decision
  -> Output
```

### 2.1 层间职责边界

- `Preparation`：冻结文本单元与 provenance，不做预切句。
- `AnchorMount`：建立 token 级时间挂载、标点事实和对齐层专属边界证据，不做最终切句。
- `DecisionIngress`：投影 DTO，不新增切点逻辑。
- `Decision`：唯一切句层。
- `Output`：只做输出与落盘，不改句边界。

---

## 3. 剩余工作地图

### 3.1 核心代码

- Rename: `backend/app/services/timeanchored_alignment/anchor_mount/boundary_hint_assembler.py` -> `backend/app/services/timeanchored_alignment/anchor_mount/boundary_evidence_builder.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/service.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/__init__.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/decision_ingress_assembler.py`
- Modify: `backend/app/services/textflow/decision_ingress_adapter.py`
- Modify: `backend/app/services/textflow/decision_layer.py`
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- Modify: `backend/app/pipelines/dual_pipeline/services/anchor_mount_graph_renderer.py`

### 3.2 测试

- Create: `backend/tests/timeanchored_alignment/test_boundary_evidence_builder.py`
- Modify: `backend/tests/textflow/test_decision_ingress_adapter.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_service.py`
- Modify: `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`
- Modify: `backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_graph_renderer.py`
- Modify: `backend/tests/textflow/test_job_p20260323_regressions.py`
- Modify: `backend/tests/timeanchored_alignment/test_job_punctuation_fact_chain_regression.py`

### 3.3 文档

- Modify: `llmdoc/architecture/dual-flow-finalization.md`
- Modify: `llmdoc/architecture/preparation-language-adapter.md`
- Modify: `llmdoc/architecture/text-processing-layered-architecture.md`

### 3.4 提交策略

- 所有“Commit”步骤均改为“暂停并通知 wgh 手动提交检查点”。
- 任何阶段都不自动执行 `git commit`。

---

## Chunk 5: AnchorMount 证据链最终收口

### Task 5.1: 将旧 `BoundaryHint` 语义收缩为真正的 `BoundaryEvidence`

**Files:**
- Rename: `backend/app/services/timeanchored_alignment/anchor_mount/boundary_hint_assembler.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/service.py`
- Modify: `backend/app/services/timeanchored_alignment/anchor_mount/__init__.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_service.py`
- Create: `backend/tests/timeanchored_alignment/test_boundary_evidence_builder.py`

- [ ] **Step 1: 写失败测试，锁死“只允许对齐层专属 reason”**

```python
def test_boundary_evidence_builder_only_emits_alignment_specific_reasons():
    evidences = builder.build(...)
    assert {item.reason for item in evidences} <= {"lexical_boundary", "anchor_block_close"}
```

- [ ] **Step 2: 写失败测试，锁死“标点、gap、speaker 不再通过 builder 进入 Decision”**

```python
def test_boundary_evidence_builder_does_not_emit_punctuation_gap_or_speaker_reasons():
    reasons = {item.reason for item in builder.build(...)}
    assert "punctuation_sentence_end" not in reasons
    assert "gap_pause" not in reasons
    assert "speaker_change" not in reasons
```

- [ ] **Step 3: 运行红灯测试**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_anchor_mount_service.py backend/tests/timeanchored_alignment/test_boundary_evidence_builder.py backend/tests/timeanchored_alignment/test_punctuation_fact_chain_contract.py -v
```

Expected: FAIL，失败点集中在旧 builder 仍输出重复 reason、旧文件名/类名仍存在。

- [ ] **Step 4: 新建 `BoundaryEvidenceBuilder`，只保留 `lexical_boundary / anchor_block_close`**

- [ ] **Step 5: 删除旧 `boundary_hint_assembler.py` 文件与导入，统一改名**

- [ ] **Step 6: 更新 `AnchorMountAlignmentService` 指标命名**
  - `boundary_hint_count -> boundary_evidence_count`
  - 不再把 punctuation / gap / speaker 重复统计成对齐层边界候选

- [ ] **Step 7: 重新运行上述测试，确认转绿**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_anchor_mount_service.py backend/tests/timeanchored_alignment/test_boundary_evidence_builder.py backend/tests/timeanchored_alignment/test_punctuation_fact_chain_contract.py -v
```

Expected: PASS

- [ ] **Step 8: 暂停并通知 wgh 手动提交检查点**

---

## Chunk 6: Decision ingress 收口为唯一可执行入口

### Task 6.1: 让对齐层专属证据真正进入 `canonical_candidate_boundaries`

**Files:**
- Modify: `backend/app/services/textflow/decision_ingress_adapter.py`
- Modify: `backend/tests/textflow/test_decision_ingress_adapter.py`

- [ ] **Step 1: 写失败测试，要求 adapter 把 alignment-specific evidences 写入 `canonical_candidate_boundaries`**

```python
def test_decision_ingress_adapter_forwards_alignment_boundary_evidences():
    result = adapter.build(package=package)
    reasons = {item.reason for item in result.decision_input.canonical_candidate_boundaries}
    assert reasons <= {"lexical_boundary", "anchor_block_close"}
    assert len(result.decision_input.canonical_candidate_boundaries) > 0
```

- [ ] **Step 2: 写失败测试，要求 adapter 不通过 `fused_evidence` 重新注入可执行 `gap/speaker/punctuation` 边界**

```python
def test_decision_ingress_adapter_keeps_alignment_boundaries_single_sourced():
    result = adapter.build(package=package)
    assert result.decision_input.canonical_candidate_boundaries
    assert all(item.reason != "speaker_change" for item in result.decision_input.canonical_candidate_boundaries)
```

- [ ] **Step 3: 运行红灯测试**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest backend/tests/textflow/test_decision_ingress_adapter.py backend/tests/textflow/test_decision_layer_ingress_context.py -v
```

Expected: FAIL，失败点集中在 `canonical_candidate_boundaries` 为空或仍存在重复证据路径。

- [ ] **Step 4: 修改 adapter**
  - `AnchoredTokenUnit -> AnnotatedWord` 保持 1:1 稳定投影
  - `boundary_evidences` 直接进入 `canonical_candidate_boundaries`
  - `fused_evidence` 若仍需保留，只能作为兼容/诊断镜像，不得成为 timeanchored 主链的第二条可执行切点通道

- [ ] **Step 5: 明确 `Decision` 的唯一打分入口**
  - `canonical_punctuation_facts`
  - `canonical_candidate_boundaries`
  - `speaker_turns`
  - token 时间流内生 `gap / duration` 信号

- [ ] **Step 6: 重新运行上述测试，确认转绿**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest backend/tests/textflow/test_decision_ingress_adapter.py backend/tests/textflow/test_decision_layer_ingress_context.py -v
```

Expected: PASS

- [ ] **Step 7: 暂停并通知 wgh 手动提交检查点**

---

## Chunk 7: `should_fallback` 变成真实路由，而不是半状态标签

### Task 7.1: 在进入 Decision 之前真实切到 `fast_direct`

**Files:**
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- Modify: `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`
- Modify: `backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py`
- Modify: `backend/tests/timeanchored_alignment/test_job_punctuation_fact_chain_regression.py`

- [ ] **Step 1: 写失败测试，要求 `anchor_mount_should_fallback` 在进入 Decision 前直接走快流路由**

```python
def test_anchor_mount_should_fallback_short_circuits_before_decision():
    service.run(ctx)
    assert fast_direct_commit_called is True
    assert decision_ingress_adapter_called is False
```

- [ ] **Step 2: 写失败测试，要求 timeanchored 不再“提交慢流文本，只把边界选成 fast”**

```python
def test_anchor_mount_force_fast_is_route_not_post_commit_label():
    service.run(ctx)
    assert commit_timeanchored_main_chain_called is False
    assert record.call_args.kwargs["reason"] == "anchor_mount_should_fallback"
```

- [ ] **Step 3: 运行红灯测试**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_alignment_stage_routing.py backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py backend/tests/timeanchored_alignment/test_job_punctuation_fact_chain_regression.py -v
```

Expected: FAIL，失败点集中在 `should_fallback` 仍只是 metadata/选边标签，而非真实路由。

- [ ] **Step 4: 修改 `alignment_stage_service.py`**
  - `should_fallback=True` 时直接走 `fast_direct`
  - 不再调用 `_commit_timeanchored_main_chain_result(...)`
  - 不再通过临时改写 `ctx.edge_selection_mode = "force_fast"` 实现半状态切路

- [ ] **Step 5: 同步修正旧字段访问**
  - `decision_ingress.tokens -> decision_ingress.anchored_token_units`
  - `decision_ingress.boundary_hints -> decision_ingress.boundary_evidences`
  - `boundary_hint_count -> boundary_evidence_count`

- [ ] **Step 6: 重新运行上述测试，确认转绿**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_alignment_stage_routing.py backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py backend/tests/timeanchored_alignment/test_job_punctuation_fact_chain_regression.py -v
```

Expected: PASS

- [ ] **Step 7: 暂停并通知 wgh 手动提交检查点**

---

## Chunk 8: 诊断、图渲染、术语彻底去 slot

### Task 8.1: 清理 graph / trace / metrics 中的旧字段名

**Files:**
- Modify: `backend/app/pipelines/dual_pipeline/services/anchor_mount_graph_renderer.py`
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- Modify: `backend/tests/timeanchored_alignment/test_anchor_mount_graph_renderer.py`

- [ ] **Step 1: 写失败测试，要求图渲染输出 `unit_id / unit_index` 而不是 `slot_id / slot_index`**

```python
def test_anchor_mount_graph_renderer_uses_unit_labels():
    payload = renderer.build_graph_payload(...)
    assert "unit_id" in payload["mounts"][0]
    assert "slot_id" not in payload["mounts"][0]
```

- [ ] **Step 2: 写失败测试，要求 trace/metrics 不再出现 `boundary_hint_count / slot_count`**

```python
def test_alignment_stage_metrics_use_boundary_evidence_and_token_unit_names():
    metrics = build_metrics(...)
    assert "boundary_hint_count" not in metrics
    assert "slot_count" not in metrics
```

- [ ] **Step 3: 运行红灯测试**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_anchor_mount_graph_renderer.py backend/tests/timeanchored_alignment/test_anchor_mount_service.py -v
```

Expected: FAIL

- [ ] **Step 4: 修改图渲染、trace 和指标名**
  - `slot_id -> unit_id`
  - `slot_index -> unit_index`
  - `slot_count -> unit_count` 或 `token_unit_count`
  - `boundary_hint_count -> boundary_evidence_count`

- [ ] **Step 5: 用全文搜索校验主链代码和测试中已无旧术语**

Run:

```powershell
rg -n "slot_index|slot_id|left_slot_index|right_slot_index|split_after_slot_id|SlowSlot|BoundaryHint|boundary_hint" backend/app/services/timeanchored_alignment backend/app/services/textflow backend/app/pipelines/dual_pipeline/services backend/tests/timeanchored_alignment backend/tests/textflow
```

Expected: 无输出；若有输出，只允许出现在与本计划无关的历史归档目录之外，必须继续清理。

- [ ] **Step 6: 暂停并通知 wgh 手动提交检查点**

---

## Chunk 9: llmdoc 覆写，同步当前真实实现

### Task 9.1: 覆写 3 份架构文档，删除旧 `slot-space` 叙事

**Files:**
- Modify: `llmdoc/architecture/dual-flow-finalization.md`
- Modify: `llmdoc/architecture/preparation-language-adapter.md`
- Modify: `llmdoc/architecture/text-processing-layered-architecture.md`

- [ ] **Step 1: 覆写 `dual-flow-finalization.md`**
  - `slow_text.slots -> slow_text.token_units`
  - `BoundaryHint -> BoundaryEvidence`
  - 删除“Decision 先做 slot->word 映射”叙述
  - 明确 `should_fallback=true` 时在进入 Decision 前直接切 `fast_direct`

- [ ] **Step 2: 覆写 `preparation-language-adapter.md`**
  - `PunctuationEvidence -> PunctuationFact -> BoundaryHint -> Decision`
  - 改为 `PunctuationEvidence -> token-indexed PunctuationFact + alignment-specific BoundaryEvidence -> Decision`
  - 明确 `DecisionIngress` 只是 DTO 投影，不生成可执行切点

- [ ] **Step 3: 覆写 `text-processing-layered-architecture.md`**
  - 删除 `slot` 索引契约说明
  - 明确 timeanchored ingress 进入 L6 时已经是 token 索引语义
  - 明确 L6 的唯一可执行边界输入是 `canonical_punctuation_facts + canonical_candidate_boundaries + speaker_turns + token 时间流`

- [ ] **Step 4: 用全文搜索验证文档已无旧叙事**

Run:

```powershell
rg -n "slot-space|slot_index|slot_id|SlowSlot|BoundaryHint|boundary_hints|slot->word" llmdoc/architecture/dual-flow-finalization.md llmdoc/architecture/preparation-language-adapter.md llmdoc/architecture/text-processing-layered-architecture.md
```

Expected: 无输出

- [ ] **Step 5: 暂停并通知 wgh 手动提交检查点**

---

## Chunk 10: 扩大回归并回放问题样本

### Task 10.1: 跑更大范围回归矩阵

**Files:**
- Test only

- [ ] **Step 1: 运行 routing / ingress / graph 回归**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest backend/tests/textflow/test_decision_ingress_adapter.py backend/tests/textflow/test_decision_layer_ingress_context.py backend/tests/timeanchored_alignment/test_alignment_stage_routing.py backend/tests/timeanchored_alignment/test_timeanchored_main_chain_routing.py backend/tests/timeanchored_alignment/test_anchor_mount_graph_renderer.py backend/tests/timeanchored_alignment/test_anchor_mount_service.py -v
```

Expected: PASS

- [ ] **Step 2: 运行作业级 regression**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest backend/tests/textflow/test_job_p20260323_regressions.py backend/tests/timeanchored_alignment/test_job_punctuation_fact_chain_regression.py -v
```

Expected: PASS

- [ ] **Step 3: 对照问题样本目录做 trace 复盘**
  - 样本目录：`jobs/p-20260328-020454-tr-test-en-1-yh75`
  - 优先检查这些文件：
    - `10_preparation.output.json`
    - `21_anchor_mount.output.json`
    - `21_anchor_mount.graph.json`
    - `31_decision_ingress.output.json`
    - `40_decision.output.json`

- [ ] **Step 4: 核对 4 个关键验收点**
  - `Preparation` 输出应是稳定 `token_units`，而不是被标点或 hint 提前切碎的伪单元
  - `AnchorMount` 输出的 `boundary_evidences` reason 只能是 `lexical_boundary / anchor_block_close`
  - `DecisionIngress` 的 `canonical_candidate_boundaries` 应已承接上述 evidences，且不再为空
  - `Decision` 若仍出现碎切，必须继续追查上游 token/time truth，不允许再在 `FinalSplitter` 后补丁式修复

- [ ] **Step 5: 暂停并通知 wgh 手动提交最终检查点**

---

## 发布门禁（全部满足才算完成）

- [ ] `slot` 已退出 timeanchored 主链契约、实现、诊断、测试与文档。
- [ ] `BoundaryHint` 文件名、类名、字段名与叙事已全部退出当前代码树。
- [ ] `DecisionIngress` 已把对齐层专属边界证据写入 `canonical_candidate_boundaries`。
- [ ] `Decision` 不再依赖任何 `slot->word` 或等价 remap。
- [ ] `should_fallback` 已变成进入 Decision 前的真实快流路由，不再保留半状态。
- [ ] `anchor_mount_graph_renderer`、trace、metrics、tests 已统一切到 `unit_* / token_unit_*` 术语。
- [ ] 3 份 `llmdoc` 架构文档已完全同步到 token-unit 主链现状。
- [ ] 问题样本 `jobs/p-20260328-020454-tr-test-en-1-yh75` 的碎切根因已从上游事实链修正，而不是靠后置 fallback 掩盖。

---

## 执行备注

- 该计划是“在当前 dirty worktree 上继续收口”的执行文档，不要求新建 worktree。
- 若执行中发现某处仍靠兼容属性维持运行，不要补新的 compat bridge，应直接把调用方切到新契约。
- 若样本仍出现“一个 Window 还是切不下去”，只能从 `token_units / punctuation_facts / boundary_evidences / envelopes / speaker_turns` 这条上游事实链继续找根因，不允许往 `Decision` 之后再加兜底层。
