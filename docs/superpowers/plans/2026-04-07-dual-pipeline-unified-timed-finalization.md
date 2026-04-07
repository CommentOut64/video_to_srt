# Dual Pipeline Unified Timed Finalization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除 `slow_text_fallback`，把所有 final 结果统一成基于 timed facts 的输入，经同一切分层输出 `sentence_records`，并让快流草稿切分收口到统一规则。

**Architecture:** 先收口 final 路由与输出契约，再引入 `FastTimedIngressAdapter`，让 `timeanchored final` 和 `fast timed final` 进入同一个 `Decision -> Output` 链。随后把 `SemanticBuffer` 从“切句者”降级为“缓冲/flush 层”，由统一 split policy 驱动 draft/final 切分，从而保证“时间戳优先、speaker 最高优先级切分、语言仅做特化”的规则在两条链上保持一致。

**Tech Stack:** Python 3.10+、FastAPI 后端运行时、`backend/app/pipelines/dual_pipeline/*`、`backend/app/services/textflow/*`、`backend/app/services/punctuation/*`、`backend/app/services/segmentation/*`、pytest、Windows PowerShell

---

## 0. 实施约束

1. 本计划按高风险共享逻辑修改处理，使用 TDD 或“先补失败回归测试，再写最小实现”的方式推进。
2. 不执行自动 commit。AGENTS 明确要求 commit 由 wgh 手动执行，因此每个任务只保留“准备提交检查点”，不包含 `git commit` 命令。
3. 当前工作区是脏的，实施时不能覆盖已有未提交修改，尤其是：
   - `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
   - `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`
   - `backend/app/services/timeanchored_alignment/decoder/local_repair.py`
4. 文档必须同步覆写，不允许追加式补丁。

## 1. 文件蓝图

### 1.1 新增文件

- `backend/app/services/textflow/fast_timed_ingress_adapter.py`
  - 把 fast timed facts 投影成与 `AlignmentPathAdapter` 同形的 `DecisionLayerInput`
- `backend/tests/textflow/test_fast_timed_ingress_adapter.py`
  - 适配器单元测试
- `backend/tests/dual_pipeline/test_unified_split_policy.py`
  - 统一 split policy 的优先级与语言特化测试

### 1.2 修改文件

- `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
  - 删除 `slow_text_fallback` 路由，改成 `fast timed final` 统一回退
- `backend/app/pipelines/dual_pipeline/implementation.py`
  - 收紧 final 输出入口，只允许经过统一 `Decision -> Output(sentence_records)`
- `backend/app/services/textflow/alignment_path_adapter.py`
  - 与新 `FastTimedIngressAdapter` 对齐输入契约
- `backend/app/services/textflow/decision_layer.py`
  - 显式消费统一切分 policy，并锁定 speaker 优先级
- `backend/app/services/punctuation/semantic_buffer.py`
  - 删除正式切句职责，仅保留缓冲/flush
- `backend/app/services/segmentation/unified_splitter.py`
  - 下沉 draft 切分职责并消费统一 policy
- `backend/app/services/punctuation/final_splitter.py`
  - 与统一 policy 对齐，保证 final 路径优先级固定
- `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`
  - 路由回归测试
- `backend/tests/timeanchored_alignment/test_phase3_decoder_shadow.py`
  - reject 语义回归测试
- `backend/tests/textflow/test_decision_layer*.py`
  - 如仓库已有相关测试文件，补充 speaker/pause/标点优先级断言
- `llmdoc/architecture/fast-worker.md`
  - 覆写快流草稿与 final 口径
- `llmdoc/architecture/segmentation-alignment-services.md`
  - 覆写统一切分与统一 final 路由口径
- `llmdoc/architecture/dual-flow-finalization.md`
  - 覆写 final fallback 语义

### 1.3 可删除/退役逻辑

- `AlignmentStageService._commit_slow_text_fallback_result()`
- 所有绕过 `Decision -> Output(sentence_records)` 的 final 直接提交分支
- `SemanticBuffer` 中以 `split_end_indices -> _build_sentences()` 为中心的正式草稿切句职责

## Chunk 1: Final 路由与输出契约收口

### Task 1: 锁定 slow reject 不再走 text-only final

**Files:**
- Modify: `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`
- Modify: `backend/tests/timeanchored_alignment/test_phase3_decoder_shadow.py`
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`

- [ ] **Step 1: 补失败回归测试，锁定 reject 路由**

在 [test_alignment_stage_routing.py](f:/video_to_srt_gpu/backend/tests/timeanchored_alignment/test_alignment_stage_routing.py) 增加至少两个测试：

```python
def test_selected_slow_low_confidence_falls_back_to_fast_timed_final(...):
    ...
    assert route == "fast_timed_final"
    assert slow_text_fallback_called is False

def test_selected_fast_reject_goes_to_fast_timed_final(...):
    ...
    assert route == "fast_timed_final"
```

在 [test_phase3_decoder_shadow.py](f:/video_to_srt_gpu/backend/tests/timeanchored_alignment/test_phase3_decoder_shadow.py) 增加 reject 语义测试：

```python
def test_alignment_low_confidence_is_final_reject_signal(...):
    ...
    assert stage_result.failure_semantic == "alignment_low_confidence"
    assert can_emit_slow_final is False
```

- [ ] **Step 2: 运行新增测试，确认当前实现失败**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' `
  backend/tests/timeanchored_alignment/test_alignment_stage_routing.py `
  backend/tests/timeanchored_alignment/test_phase3_decoder_shadow.py -v
```

Expected:

1. 至少有一条断言失败
2. 当前实现仍尝试走 `slow_text_fallback` 或仍允许 slow final

- [ ] **Step 3: 删除 slow text final 路由，改成 fast timed final 路由**

在 [alignment_stage_service.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py) 做最小实现：

1. 删除 `_should_commit_slow_text_fallback_on_timeanchored_rejection()`
2. 删除 `_commit_slow_text_fallback_result()`
3. 新增或改造统一路由辅助方法，例如：

```python
def _should_commit_fast_timed_final_on_timeanchored_rejection(...)-> bool:
    return stage_result is not None and self._is_unrecoverable_alignment_low_confidence(...)
```

4. 把 `selected slow + alignment_low_confidence` 也改成走 fast timed final
5. 若 fast timed facts 不可用，显式报错并 fail-closed

- [ ] **Step 4: 重跑路由测试，确认通过**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' `
  backend/tests/timeanchored_alignment/test_alignment_stage_routing.py `
  backend/tests/timeanchored_alignment/test_phase3_decoder_shadow.py -v
```

Expected:

1. 所有新增测试 PASS
2. 不再有 `slow_text_fallback` 路由断言残留

- [ ] **Step 5: 检查工作区只包含预期改动**

Run:

```powershell
git status --short
```

Expected:

1. 只新增/修改本任务涉及文件
2. 不回滚用户原有修改

### Task 2: 引入 `FastTimedIngressAdapter`，统一进入 `Decision -> Output`

**Files:**
- Create: `backend/app/services/textflow/fast_timed_ingress_adapter.py`
- Create: `backend/tests/textflow/test_fast_timed_ingress_adapter.py`
- Modify: `backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py`
- Modify: `backend/app/pipelines/dual_pipeline/implementation.py`
- Modify: `backend/app/services/textflow/alignment_path_adapter.py`

- [ ] **Step 1: 先写适配器失败测试**

在 [test_fast_timed_ingress_adapter.py](f:/video_to_srt_gpu/backend/tests/textflow/test_fast_timed_ingress_adapter.py) 增加至少三个测试：

```python
def test_fast_timed_ingress_adapter_builds_decision_input_from_fast_words():
    ...
    assert output.annotated_words
    assert output.aligned_facts is not None

def test_fast_timed_ingress_adapter_preserves_speaker_turn_facts_when_enabled():
    ...
    assert output.aligned_facts.speaker_turns

def test_fast_timed_ingress_adapter_and_alignment_path_adapter_share_shape():
    ...
    assert set(vars(fast_input).keys()) == set(vars(anchor_input).keys())
```

- [ ] **Step 2: 运行适配器测试，确认当前缺失实现**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' `
  backend/tests/textflow/test_fast_timed_ingress_adapter.py -v
```

Expected:

1. FAIL，原因是文件或类不存在

- [ ] **Step 3: 写最小适配器实现**

在 [fast_timed_ingress_adapter.py](f:/video_to_srt_gpu/backend/app/services/textflow/fast_timed_ingress_adapter.py) 实现最小适配器：

```python
class FastTimedIngressAdapter:
    def build(
        self,
        *,
        chunk,
        fast_words,
        language,
        speaker_turns,
        punctuation_facts,
    ) -> DecisionLayerInput:
        ...
```

实现要求：

1. 输入只使用 fast timed facts
2. 不伪造 slow path
3. 输出字段形状和 `AlignmentPathAdapter` 一致
4. speaker 开启时透传 `speaker_turns`

- [ ] **Step 4: 接入 alignment stage 与 implementation 主链**

在 [alignment_stage_service.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py) 和 [implementation.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/implementation.py) 中完成接线：

1. fast timed final 不再自己造 final sentence
2. 改为：

```python
decision_input = self._fast_timed_ingress_adapter.build(...)
decision_output = host._run_decision_layer(...)
host._emit_output_layer(... sentence_records=decision_output.sentence_records ...)
```

3. 所有 final 路径都必须最终调用 `_emit_output_layer(... sentence_records=...)`

- [ ] **Step 5: 写契约回归测试**

在现有测试文件中补断言：

```python
def test_fast_timed_final_emits_sentence_records(...):
    ...
    assert output_layer_input.sentence_records
    assert output_layer_input.subtitle_batch is not None or True
```

同时补一个负向测试，确认旧的空 `sentence_records` 提交已不存在。

- [ ] **Step 6: 运行契约测试**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' `
  backend/tests/textflow/test_fast_timed_ingress_adapter.py `
  backend/tests/timeanchored_alignment/test_alignment_stage_routing.py -v
```

Expected:

1. PASS
2. 不再出现 `_emit_output_layer 需要 sentence_records`

- [ ] **Step 7: 准备提交检查点**

Run:

```powershell
git diff -- backend/app/pipelines/dual_pipeline/services/alignment_stage_service.py
git diff -- backend/app/pipelines/dual_pipeline/implementation.py
git diff -- backend/app/services/textflow/fast_timed_ingress_adapter.py
```

Expected:

1. diff 只体现统一 final 路由与适配器接线
2. 没有顺手掺入草稿切分重构

## Chunk 2: 统一切分规则与 `SemanticBuffer` 收口

### Task 3: 引入共享 split policy，锁定 speaker > 标点 > pause > hard limit

**Files:**
- Create: `backend/tests/dual_pipeline/test_unified_split_policy.py`
- Modify: `backend/app/services/segmentation/unified_splitter.py`
- Modify: `backend/app/services/punctuation/final_splitter.py`
- Modify: `backend/app/services/textflow/decision_layer.py`

- [ ] **Step 1: 先写统一规则测试**

在 [test_unified_split_policy.py](f:/video_to_srt_gpu/backend/tests/dual_pipeline/test_unified_split_policy.py) 新增至少五个测试：

```python
def test_speaker_change_has_highest_priority_when_enabled():
    ...
    assert split_reason == "speaker_change"

def test_speaker_change_cannot_invent_timestamp_boundary():
    ...
    assert split_reason != "speaker_change"

def test_english_sparse_punctuation_uses_pause_or_hard_limit():
    ...
    assert len(sentences) >= 2

def test_short_sentences_are_not_over_fragmented():
    ...
    assert len(sentences) == 1

def test_final_and_draft_follow_same_priority_order():
    ...
    assert draft_reasons == final_reasons
```

- [ ] **Step 2: 运行规则测试，确认当前行为不满足**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' `
  backend/tests/dual_pipeline/test_unified_split_policy.py -v
```

Expected:

1. FAIL
2. 至少体现当前 draft/final 规则不一致或少标点长句未补切

- [ ] **Step 3: 在 `UnifiedSplitter` 与 `FinalSplitter` 中引入共享 policy**

最小实现方式：

1. 在 [unified_splitter.py](f:/video_to_srt_gpu/backend/app/services/segmentation/unified_splitter.py) 抽出共享 policy 编译函数或快照数据类
2. 在 [final_splitter.py](f:/video_to_srt_gpu/backend/app/services/punctuation/final_splitter.py) 读取同一份 policy
3. 在 [decision_layer.py](f:/video_to_srt_gpu/backend/app/services/textflow/decision_layer.py) 使用同一优先级定义

建议骨架：

```python
@dataclass(frozen=True)
class SplitPolicySnapshot:
    speaker_enabled: bool
    strong_punctuation: tuple[str, ...]
    weak_punctuation: tuple[str, ...]
    min_duration: float
    soft_pause: float
    long_pause: float
    hard_limit: float
    language_group: str
```

- [ ] **Step 4: 将 priority 顺序固化到 final 与 draft**

必须保证两条链共享这个顺序：

1. `speaker_change`
2. `strong_sentence_end_punctuation`
3. `stable_punctuation`
4. `reliable_pause_or_gap`
5. `length_or_hard_limit`
6. `no_split`

如果某层已有现成候选打分，不重写全算法，只要把优先级映射统一即可。

- [ ] **Step 5: 重跑统一规则测试**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' `
  backend/tests/dual_pipeline/test_unified_split_policy.py -v
```

Expected:

1. PASS
2. 英文少标点长句能补切
3. speaker 优先级断言成立

### Task 4: 收缩 `SemanticBuffer`，草稿切分统一下沉

**Files:**
- Modify: `backend/app/services/punctuation/semantic_buffer.py`
- Modify: `backend/app/pipelines/dual_pipeline/implementation.py`
- Modify: `backend/app/services/segmentation/unified_splitter.py`
- Modify: 相关 draft 测试文件（按仓库现有命名补充）

- [ ] **Step 1: 先写草稿回归测试**

在现有 dual pipeline / punctuation / segmentation 测试中补两个回归：

```python
def test_semantic_buffer_only_buffers_and_flushes_not_final_split():
    ...
    assert semantic_buffer_output_has_timed_words is True
    assert final_sentence_count_assertion_is_done_by_splitter is True

def test_sparse_punctuation_draft_sentence_is_split_by_unified_splitter():
    ...
    assert len(draft_sentences) >= 2
```

- [ ] **Step 2: 运行草稿回归测试，确认当前失败**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' `
  backend/tests/dual_pipeline/test_unified_split_policy.py `
  backend/tests/timeanchored_alignment/test_alignment_stage_routing.py -v
```

Expected:

1. 至少一条草稿相关断言失败
2. 能复现 `SemanticBuffer` 仍主导切句的现状

- [ ] **Step 3: 收缩 `SemanticBuffer` 职责**

在 [semantic_buffer.py](f:/video_to_srt_gpu/backend/app/services/punctuation/semantic_buffer.py) 做最小重构：

1. 保留：
   - pending text / words 聚合
   - speaker flush
   - max pending / force flush
   - punctuation 决策透传
2. 删除或降级：
   - `split_end_indices` 驱动的正式句边界生成
   - `_build_sentences()` 作为正式草稿切句真源

可接受的中间态：

```python
class BufferedTimedChunk:
    text: str
    audio_range: tuple[float, float]
    timed_words: list[...]
    speaker_id: str | None
```

- [ ] **Step 4: 在 implementation 中改成“buffer -> splitter -> draft sentences”**

在 [implementation.py](f:/video_to_srt_gpu/backend/app/pipelines/dual_pipeline/implementation.py) 中把 `_emit_semantic_sentences()` 改成：

1. `SemanticBuffer.add()` 返回缓冲块
2. `UnifiedSplitter.split_draft(...)` 基于缓冲块统一切分
3. 再由 subtitle manager 推送草稿

- [ ] **Step 5: 跑定向草稿测试**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' `
  backend/tests/dual_pipeline/test_unified_split_policy.py -v
```

Expected:

1. PASS
2. 少标点英文草稿不再整块糊住

- [ ] **Step 6: 做工作区检查**

Run:

```powershell
git diff -- backend/app/services/punctuation/semantic_buffer.py
git diff -- backend/app/services/segmentation/unified_splitter.py
git diff -- backend/app/pipelines/dual_pipeline/implementation.py
```

Expected:

1. 只体现职责收缩与切分下沉
2. 不夹带无关格式化或命名清洗

## Chunk 3: 文档同步与端到端回归

### Task 5: 覆写 llmdoc，统一对外口径

**Files:**
- Modify: `llmdoc/architecture/fast-worker.md`
- Modify: `llmdoc/architecture/segmentation-alignment-services.md`
- Modify: `llmdoc/architecture/dual-flow-finalization.md`

- [ ] **Step 1: 重写过期叙述，不做追加**

每个文档至少同步以下事实：

1. `slow_text_fallback` 已删除
2. final fallback 已统一成 timed final
3. `SemanticBuffer` 不再是正式切句真源
4. speaker 开启时是最高优先级切分信号

- [ ] **Step 2: 人工核对文档与代码口径**

Run:

```powershell
Select-String -Path llmdoc\architecture\fast-worker.md, `
  llmdoc\architecture\segmentation-alignment-services.md, `
  llmdoc\architecture\dual-flow-finalization.md `
  -Pattern "slow_text_fallback|SemanticBuffer|fast timed final|sentence_records"
```

Expected:

1. 文档中不再把 `slow_text_fallback` 描述为活跃正式路径
2. 文档明确 `sentence_records` 是正式输出真源

### Task 6: 回归验证与交付检查点

**Files:**
- Test: `backend/tests/timeanchored_alignment/test_alignment_stage_routing.py`
- Test: `backend/tests/timeanchored_alignment/test_phase3_decoder_shadow.py`
- Test: `backend/tests/textflow/test_fast_timed_ingress_adapter.py`
- Test: `backend/tests/dual_pipeline/test_unified_split_policy.py`

- [ ] **Step 1: 跑最小闭环测试集**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' `
  backend/tests/timeanchored_alignment/test_alignment_stage_routing.py `
  backend/tests/timeanchored_alignment/test_phase3_decoder_shadow.py `
  backend/tests/textflow/test_fast_timed_ingress_adapter.py `
  backend/tests/dual_pipeline/test_unified_split_policy.py -v
```

Expected:

1. 全部 PASS
2. 不再出现 `_emit_output_layer 需要 sentence_records`

- [ ] **Step 2: 如仓库已有更大范围相关套件，再跑一轮扩展验证**

Run:

```powershell
$env:PYTHONPATH='backend'; python -m pytest -o addopts='' `
  backend/tests/timeanchored_alignment `
  backend/tests/textflow -k "alignment or split or ingress" -v
```

Expected:

1. 与本次改动直接相关的测试全部 PASS
2. 若有失败，必须先定位是旧脏改动影响还是本次回归

- [ ] **Step 3: 端到端样例复核清单**

对以下样例重新跑或复核产物：

1. `F:\video_to_srt_gpu\jobs\p-20260407-114227-tr-test-en-1-0mgb`
2. 与 `#15 / #22` 对应的 checkpoint 样例
3. 已知 `alignment_low_confidence` 坏窗样例

必须确认：

1. `chunk_0005` 不再卡死
2. slow reject 时 final 走 `fast timed final`
3. 草稿内部可切句不再糊成一条
4. 时间戳没有为了切分被重新伪造

- [ ] **Step 4: 交付前检查**

Run:

```powershell
git status --short
```

Expected:

1. 只剩计划内代码、测试、文档修改
2. 没有误删用户现有文件
3. 准备好由 wgh 手动决定是否提交

## 完成定义

达到以下条件才算完成本计划：

1. `slow_text_fallback` 已从活跃 runtime 路由中删除
2. 所有 final 路径都通过统一 `Decision -> Output(sentence_records)`
3. `FastTimedIngressAdapter` 成为 reject 后唯一的 timed final 回退入口
4. `SemanticBuffer` 只保留缓冲/flush 职责
5. draft 与 final 共享同一套 split priority 规则
6. llmdoc 已同步覆写为当前真实状态
7. 定向测试与最小闭环验证全部通过

