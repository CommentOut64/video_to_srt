# Timeanchored Alignment Inspector Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个与主流水线默认执行路径隔离的对齐层分析脚本，既能读取现有 timeanchored trace 做取证，也能在独立 sandbox 中重放指定音频/片段并输出全量 alignment 证据。

**Architecture:** 采用“独立 CLI + 独立分析服务”的结构。CLI 只负责参数解析、sandbox 目录创建和结果落盘；核心分析逻辑封装为可测试的服务，复用现有 `AlignmentPreparationAssembler`、`AnchorMountAlignmentService`、`AlignmentStageService` 和 trace writer，但所有 replay 落盘都写入 `job_dir/debug/alignment_inspector/<run_id>/...`，不进入主任务默认 `debug/postprocess` 目录。

**Tech Stack:** Python, pytest, pathlib, argparse, 现有 timeanchored alignment 服务、现有 postprocess trace writer

---

## Chunk 1: 分析入口与隔离边界

### Task 1: 锁定脚本与 sandbox 目录职责

**Files:**
- Create: `backend/scripts/analyze_timeanchored_alignment.py`
- Create: `backend/app/services/timeanchored_alignment/debug_alignment_inspector.py`
- Test: `backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py`

- [ ] **Step 1: 写失败测试，刻画 sandbox 目录与参数入口**

```python
def test_build_analysis_run_dir_uses_job_debug_alignment_inspector(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    run_dir = build_analysis_run_dir(job_dir=job_dir, run_label="demo")
    assert run_dir == job_dir / "debug" / "alignment_inspector" / "demo"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py -k "build_analysis_run_dir" -v`
Expected: FAIL，提示目标函数不存在

- [ ] **Step 3: 实现最小 sandbox 目录构建与 CLI 参数骨架**

```python
def build_analysis_run_dir(*, job_dir: Path, run_label: str) -> Path:
    return job_dir / "debug" / "alignment_inspector" / run_label
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py -k "build_analysis_run_dir" -v`
Expected: PASS

### Task 2: 增加 inspect / replay 双模式 CLI

**Files:**
- Modify: `backend/scripts/analyze_timeanchored_alignment.py`
- Test: `backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py`

- [ ] **Step 1: 写失败测试，刻画 CLI 参数语义**

```python
def test_parse_args_supports_inspect_and_replay_modes() -> None:
    args = parse_args(["inspect", "--job-dir", "F:/jobs/x"])
    assert args.mode == "inspect"
    args = parse_args(["replay", "--job-dir", "F:/jobs/x", "--audio-path", "F:/audio.wav"])
    assert args.mode == "replay"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py -k "parse_args_supports" -v`
Expected: FAIL

- [ ] **Step 3: 实现最小 CLI 解析**

```python
parser = argparse.ArgumentParser(...)
sub = parser.add_subparsers(dest="mode", required=True)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py -k "parse_args_supports" -v`
Expected: PASS

## Chunk 2: 现有 trace 取证模式

### Task 3: 实现 inspect 模式的 trace 聚合

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/debug_alignment_inspector.py`
- Modify: `backend/scripts/analyze_timeanchored_alignment.py`
- Test: `backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py`

- [ ] **Step 1: 写失败测试，读取现有 job trace 并汇总窗口指标**

```python
def test_inspect_trace_job_summarizes_window_metrics() -> None:
    report = inspect_trace_job(job_dir=_fixture_job_dir())
    assert report["window_count"] > 0
    assert "timeline_validity_counts" in report
    assert "hypothesis_evidence" in report
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py -k "inspect_trace_job_summarizes_window_metrics" -v`
Expected: FAIL

- [ ] **Step 3: 实现 trace 聚合器**

```python
def inspect_trace_job(*, job_dir: Path) -> dict[str, Any]:
    # 读取 21_anchor_mount.graph.json / 31_decision_ingress / 40_decision
    # 汇总 mount_status、coverage、gap_rescue、speaker_bridge 与 fallback 证据
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py -k "inspect_trace_job_summarizes_window_metrics" -v`
Expected: PASS

### Task 4: 输出四类假设对应证据

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/debug_alignment_inspector.py`
- Test: `backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py`

- [ ] **Step 1: 写失败测试，要求报告带假设证据分区**

```python
def test_inspect_trace_job_emits_hypothesis_evidence_sections() -> None:
    report = inspect_trace_job(job_dir=_fixture_job_dir())
    evidence = report["hypothesis_evidence"]
    assert set(evidence) == {
        "slow_text_unreliable",
        "anchor_mount_failed",
        "anchor_rejected",
        "chain_or_dp_failed",
    }
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py -k "hypothesis_evidence_sections" -v`
Expected: FAIL

- [ ] **Step 3: 实现假设证据映射**

```python
report["hypothesis_evidence"] = {
    "slow_text_unreliable": {...},
    "anchor_mount_failed": {...},
    "anchor_rejected": {...},
    "chain_or_dp_failed": {...},
}
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py -k "hypothesis_evidence_sections" -v`
Expected: PASS

## Chunk 3: 独立 replay 模式

### Task 5: 实现 replay sandbox 和独立 trace 落盘

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/debug_alignment_inspector.py`
- Modify: `backend/scripts/analyze_timeanchored_alignment.py`
- Test: `backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py`

- [ ] **Step 1: 写失败测试，确保 replay 使用独立 sandbox job_dir**

```python
def test_replay_context_uses_analysis_sandbox_job_dir(tmp_path: Path) -> None:
    sandbox = build_analysis_run_dir(job_dir=tmp_path, run_label="demo")
    ctx = build_replay_context(job_dir=tmp_path, sandbox_job_dir=sandbox, ...)
    assert Path(ctx.job_dir) == sandbox
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py -k "sandbox_job_dir" -v`
Expected: FAIL

- [ ] **Step 3: 实现 replay 上下文构建**

```python
ctx = ProcessingContext(...)
ctx.job_dir = sandbox_job_dir
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py -k "sandbox_job_dir" -v`
Expected: PASS

### Task 6: 复用现有 preparation + anchor mount 最小闭环

**Files:**
- Modify: `backend/app/services/timeanchored_alignment/debug_alignment_inspector.py`
- Test: `backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py`

- [ ] **Step 1: 写失败测试，刻画 replay 返回 preparation / anchor_mount / window_recovery 报告**

```python
def test_replay_alignment_window_returns_stage_artifacts(monkeypatch) -> None:
    result = replay_alignment_window(...)
    assert "preparation" in result
    assert "anchor_mount" in result
    assert "window_recovery" in result
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py -k "replay_alignment_window_returns_stage_artifacts" -v`
Expected: FAIL

- [ ] **Step 3: 实现最小 replay 服务**

```python
preparation = AlignmentPreparationAssembler(...).prepare(...)
stage_result = AnchorMountAlignmentService().execute(...)
window_recovery = WindowRecoveryPlanner().build(stage_result=stage_result)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py -k "replay_alignment_window_returns_stage_artifacts" -v`
Expected: PASS

## Chunk 4: 端到端验证与真实数据输出

### Task 7: 用旧 trace job 验证 inspect 模式

**Files:**
- Modify: `backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py`

- [ ] **Step 1: 运行 inspect 模式测试**

Run: `$env:PYTHONPATH='backend'; python -m pytest backend/tests/timeanchored_alignment/test_debug_alignment_inspector.py -k "inspect" -v`
Expected: PASS

- [ ] **Step 2: 手动运行 inspect CLI，验证输出 JSON**

Run: `$env:PYTHONPATH='backend'; python backend/scripts/analyze_timeanchored_alignment.py inspect --job-dir F:\\video_to_srt_gpu\\jobs\\p-20260329-233522-tr-test-en-1-hplj`
Expected: 输出带 `timeline_validity_counts`、`mount_status_counts`、`hypothesis_evidence` 的 JSON

### Task 8: 用用户提供音频执行 replay 模式

**Files:**
- Modify: 无

- [ ] **Step 1: 在拿到真实音频路径后运行 replay CLI**

Run: `$env:PYTHONPATH='backend'; python backend/scripts/analyze_timeanchored_alignment.py replay --job-dir <目标job目录> --audio-path <错误音频路径> [--clip-start <sec> --clip-end <sec>]`
Expected: 在 `<job_dir>/debug/alignment_inspector/<run_id>/debug/postprocess/` 生成独立 trace，不污染主任务默认 trace

- [ ] **Step 2: 校验 replay 输出是否包含 candidate/trust/solver 证据**

Run: `Get-ChildItem -Recurse <job_dir>\\debug\\alignment_inspector\\<run_id>`
Expected: 至少包含 preparation、anchor_mount、decision_ingress、decision、report.json

- [ ] **Step 3: 汇总并回答四类假设**

Run: 无
Expected: 基于 replay 真实数据给出“慢流文本问题 / 挂载失败 / 可信度拒绝 / DP 链求解失败”四类证据结论
