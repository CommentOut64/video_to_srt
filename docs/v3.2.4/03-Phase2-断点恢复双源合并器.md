# Phase 2: 断点恢复双源合并器

> Type: Implementation Plan | Status: Ready
> Phase: 2 (P1 控制面对齐)
> 目标: 恢复路径同时保留"单元边界控制"与"转录细粒度数据"
> 灰度开关: `RESUME_MERGER_ENABLED`
> 依赖: Phase 0 (状态机)

## 1. 问题分析

### 当前双源断裂

```
runtime_state.db（控制面）           checkpoint.json（数据面）
├─ last_unit_commits                 ├─ fast_processed_indices
│  ├─ vad_chunk                      ├─ slow_processed_indices
│  ├─ triage_chunk                   ├─ finalized_indices
│  ├─ separation_chunk               ├─ previous_whisper_text
│  └─ langid_chunk                   ├─ sentences_snapshot
└─ unit_journal[]                    ├─ sentence_count
                                     └─ chunk_sentences_map
```

**问题**: `orchestrator.py:461` 检测到 `runtime_state.db` 存在时**早返回**，构建的 `runtime_checkpoint` 仅包含预处理边界信息，**完全忽略** `checkpoint.json` 中的转录细粒度数据（`finalized_indices`、`sentences_snapshot` 等）。

**后果**: 暂停恢复后丢失已定稿的字幕索引，导致已完成的 chunk 被重新处理。

### 恢复路径当前行为

```python
# orchestrator.py:459-500
if runtime_state_service.has_runtime_state():
    snapshot = runtime_state_service.load_snapshot()
    # ... 仅从 runtime_state 构建预处理状态 ...
    runtime_checkpoint = { "runtime_state": ..., "preprocessing": ... }

    # checkpoint.json 只在没有 runtime_state 时才使用（L500+）
    # 两者没有合并逻辑
```

## 2. 新增文件

### 2.1 `backend/app/services/checkpoint/resume_state_merger.py`（~150行）

```python
"""
恢复状态合并器 - 双源合并策略。

设计决策：
- 预处理完成位：runtime_state.db 优先（单元边界更可靠）
- 转录细粒度字段：checkpoint.json 优先（转录进度更精确）
- 冲突时记录 merge_decision 到诊断信息
- 独立模块，不侵入现有恢复路径
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.services.checkpoint.runtime_checkpoint_models import RuntimeCheckpointSnapshot
from app.services.checkpoint.runtime_checkpoint_service import RuntimeCheckpointService
from app.services.job.checkpoint_manager import CheckpointManagerV37

logger = logging.getLogger(__name__)


@dataclass
class MergeDecision:
    """单个合并决策记录。"""
    field_name: str
    source: str          # "runtime_state" | "checkpoint_json" | "default"
    reason: str
    value_summary: str   # 值的摘要（不含完整数据，避免日志膨胀）


@dataclass
class MergedResumeState:
    """合并后的恢复状态 - 包含两个源的最佳数据。"""

    # 预处理边界（来自 runtime_state.db）
    vad_completed: bool = False
    spectral_triage_completed: bool = False
    separation_completed: bool = False
    langid_completed: bool = False

    # 转录细粒度（来自 checkpoint.json）
    fast_processed_indices: Set[int] = field(default_factory=set)
    slow_processed_indices: Set[int] = field(default_factory=set)
    finalized_indices: Set[int] = field(default_factory=set)
    previous_whisper_text: str = ""
    sentences_snapshot: List[Dict[str, Any]] = field(default_factory=list)
    sentence_count: int = 0
    chunk_sentences_map: Dict[int, List[int]] = field(default_factory=dict)

    # 诊断信息
    merge_decisions: List[MergeDecision] = field(default_factory=list)
    runtime_state_available: bool = False
    checkpoint_json_available: bool = False


class ResumeStateMerger:
    """
    双源恢复合并器。

    合并规则：
    1. 预处理完成位 → runtime_state.db 优先
       - 基于 last_unit_commits 判断各阶段完成状态
       - 更可靠（事务性写入）

    2. 转录细粒度字段 → checkpoint.json 优先
       - fast/slow/finalized_indices
       - previous_whisper_text
       - sentences_snapshot
       - 更精确（转录进度细节）

    3. 冲突决策
       - 如果两个源对"是否需要重处理"有分歧，取保守值（不跳过）
       - 记录所有决策到 merge_decisions
    """

    def __init__(self, job_dir: Path):
        self._job_dir = job_dir
        self._logger = logging.getLogger(f"{__name__}.{job_dir.name}")

    def merge(self) -> MergedResumeState:
        """执行双源合并，返回完整恢复状态。"""
        result = MergedResumeState()

        # 源 1: runtime_state.db
        runtime_snapshot = self._load_runtime_state()
        result.runtime_state_available = runtime_snapshot is not None

        # 源 2: checkpoint.json
        checkpoint_data = self._load_checkpoint()
        result.checkpoint_json_available = checkpoint_data is not None

        if not runtime_snapshot and not checkpoint_data:
            self._logger.info("无恢复数据，全新执行")
            return result

        # 合并预处理边界（runtime_state 优先）
        self._merge_preprocessing(result, runtime_snapshot, checkpoint_data)

        # 合并转录细粒度（checkpoint.json 优先）
        self._merge_transcription(result, runtime_snapshot, checkpoint_data)

        self._logger.info(
            "合并完成: runtime=%s, checkpoint=%s, decisions=%d",
            result.runtime_state_available,
            result.checkpoint_json_available,
            len(result.merge_decisions),
        )

        return result

    def _load_runtime_state(self) -> Optional[RuntimeCheckpointSnapshot]:
        """加载 runtime_state.db 快照。"""
        try:
            service = RuntimeCheckpointService(job_dir=self._job_dir)
            if service.has_runtime_state():
                return service.load_snapshot()
        except Exception as exc:
            self._logger.warning("加载 runtime_state.db 失败: %s", exc)
        return None

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """加载 checkpoint.json。"""
        try:
            mgr = CheckpointManagerV37(self._job_dir, self._logger)
            return mgr.load_checkpoint()
        except Exception as exc:
            self._logger.warning("加载 checkpoint.json 失败: %s", exc)
        return None

    def _merge_preprocessing(
        self,
        result: MergedResumeState,
        runtime: Optional[RuntimeCheckpointSnapshot],
        checkpoint: Optional[Dict[str, Any]],
    ) -> None:
        """合并预处理边界 - runtime_state.db 优先。"""
        if runtime and runtime.last_unit_commits:
            commits = runtime.last_unit_commits
            preprocess_commit = commits.get("preprocess", "")

            result.vad_completed = preprocess_commit in {
                "vad_chunk", "triage_chunk", "separation_chunk",
                "langid_chunk", "speaker_chunk",
            }
            result.spectral_triage_completed = preprocess_commit in {
                "triage_chunk", "separation_chunk",
                "langid_chunk", "speaker_chunk",
            }
            result.separation_completed = preprocess_commit in {
                "separation_chunk", "langid_chunk", "speaker_chunk",
            }
            result.langid_completed = preprocess_commit in {
                "langid_chunk", "speaker_chunk",
            }

            result.merge_decisions.append(MergeDecision(
                field_name="preprocessing",
                source="runtime_state",
                reason="runtime_state.db 提供事务性单元边界",
                value_summary=f"preprocess_commit={preprocess_commit}",
            ))

        elif checkpoint:
            prep = checkpoint.get("preprocessing", {}) or {}
            result.vad_completed = bool(prep.get("vad_completed"))
            result.spectral_triage_completed = bool(prep.get("spectral_triage_completed"))
            result.separation_completed = bool(prep.get("separation_completed"))

            result.merge_decisions.append(MergeDecision(
                field_name="preprocessing",
                source="checkpoint_json",
                reason="runtime_state.db 不可用，降级到 checkpoint.json",
                value_summary=f"vad={result.vad_completed}, triage={result.spectral_triage_completed}",
            ))

    def _merge_transcription(
        self,
        result: MergedResumeState,
        runtime: Optional[RuntimeCheckpointSnapshot],
        checkpoint: Optional[Dict[str, Any]],
    ) -> None:
        """合并转录细粒度 - checkpoint.json 优先。"""
        if not checkpoint:
            result.merge_decisions.append(MergeDecision(
                field_name="transcription",
                source="default",
                reason="checkpoint.json 不可用，转录从头开始",
                value_summary="empty",
            ))
            return

        trans = checkpoint.get("transcription", {}) or {}

        # 快流已处理索引
        result.fast_processed_indices = set(
            trans.get("fast_processed_indices", [])
            or trans.get("processed_indices", [])  # 兼容旧格式
            or []
        )

        # 慢流已处理索引
        result.slow_processed_indices = set(
            trans.get("slow_processed_indices", []) or []
        )

        # 已对齐（定稿）索引
        result.finalized_indices = set(
            trans.get("finalized_indices", []) or []
        )

        # Whisper 上文
        result.previous_whisper_text = trans.get("previous_whisper_text", "")

        # 字幕快照
        result.sentences_snapshot = trans.get("sentences_snapshot", []) or []
        result.sentence_count = trans.get("sentence_count", 0)
        result.chunk_sentences_map = trans.get("chunk_sentences_map", {}) or {}

        result.merge_decisions.append(MergeDecision(
            field_name="transcription",
            source="checkpoint_json",
            reason="checkpoint.json 提供转录细粒度数据",
            value_summary=(
                f"fast={len(result.fast_processed_indices)}, "
                f"slow={len(result.slow_processed_indices)}, "
                f"finalized={len(result.finalized_indices)}, "
                f"sentences={len(result.sentences_snapshot)}"
            ),
        ))

        # 冲突检测: runtime_state 的 transcription_hint（如果存在）
        if runtime and hasattr(runtime, 'transcription_hint') and runtime.transcription_hint:
            hint = runtime.transcription_hint
            # 如果 runtime 记录的进度比 checkpoint 更新，记录但不覆盖
            if hint.get("last_committed_chunk_index", -1) > max(
                result.finalized_indices, default=-1
            ):
                result.merge_decisions.append(MergeDecision(
                    field_name="transcription_conflict",
                    source="runtime_state",
                    reason="runtime_state 记录更新的提交点，但保留 checkpoint.json 的细粒度数据",
                    value_summary=f"runtime_hint={hint}",
                ))
```

## 3. 修改文件

### 3.1 `backend/app/services/checkpoint/runtime_checkpoint_models.py`

**改动**: `RuntimeCheckpointSnapshot` 增加可选 `transcription_hint`

```python
@dataclass(frozen=True)
class RuntimeCheckpointSnapshot:
    """运行时状态快照。"""
    last_unit_commits: dict[str, str] = field(default_factory=dict)

    # Phase 2 新增: 转录摘要索引（不复制全部 checkpoint，仅存关键提交点）
    transcription_hint: dict[str, Any] | None = None
```

### 3.2 `backend/app/services/checkpoint/runtime_checkpoint_service.py`

**改动 1**: 单元提交增加事务保护

```python
def record_unit_committed(self, stage: str, unit_id: str, payload: dict):
    """记录单元提交（事务保护）。"""
    with self.repository.transaction() as conn:
        self.repository.append_unit_journal(
            stage=stage,
            unit_id=unit_id,
            status="committed",
            payload_json=json.dumps(payload),
            conn=conn,
        )
        self.repository.upsert_unit_commit(
            stage=stage,
            last_unit_id=unit_id,
            conn=conn,
        )
```

**改动 2**: 新增 `save_transcription_hint` 方法

```python
def save_transcription_hint(self, hint: Dict[str, Any]) -> None:
    """保存转录摘要到 runtime_state（仅关键索引，不含完整数据）。"""
    self.repository.upsert_control_signal(
        key="transcription_hint",
        value=json.dumps(hint),
    )
```

### 3.3 `backend/app/pipelines/orchestrator.py`

**改动**: `build_resume_context` 使用 `ResumeStateMerger`

```python
from app.services.checkpoint.resume_state_merger import ResumeStateMerger

def build_resume_context(self, job: JobState, ...) -> ResumeContext:
    """构建恢复上下文 - 使用双源合并器。"""

    if not RESUME_MERGER_ENABLED:
        # 灰度关闭时使用旧逻辑
        return self._build_resume_context_legacy(job, ...)

    merger = ResumeStateMerger(job_dir=Path(job.dir))
    merged = merger.merge()

    # 计算安全跳过索引
    safe_indices: Set[int] = set()
    if merged.finalized_indices:
        max_finalized = max(merged.finalized_indices)
        safe_indices = set(range(max_finalized + 1))
    elif merged.fast_processed_indices and merged.slow_processed_indices:
        safe_indices = merged.fast_processed_indices & merged.slow_processed_indices

    return ResumeContext(
        checkpoint={
            "preprocessing": {
                "vad_completed": merged.vad_completed,
                "spectral_triage_completed": merged.spectral_triage_completed,
                "separation_completed": merged.separation_completed,
            },
            "transcription": {
                "fast_processed_indices": list(merged.fast_processed_indices),
                "slow_processed_indices": list(merged.slow_processed_indices),
                "finalized_indices": list(merged.finalized_indices),
                "previous_whisper_text": merged.previous_whisper_text,
                "sentences_snapshot": merged.sentences_snapshot,
                "sentence_count": merged.sentence_count,
                "chunk_sentences_map": merged.chunk_sentences_map,
            },
        },
        safe_processed_indices=safe_indices,
        fast_processed_indices=merged.fast_processed_indices,
        slow_processed_indices=merged.slow_processed_indices,
        finalized_indices=merged.finalized_indices,
        previous_whisper_text=merged.previous_whisper_text,
        sentences_snapshot=merged.sentences_snapshot,
        sentence_count=merged.sentence_count,
        chunk_sentences_map=merged.chunk_sentences_map,
        merge_decisions=[d.__dict__ for d in merged.merge_decisions],
    )
```

### 3.4 `backend/app/schemas/resume_context.py`

**改动**: 增加 `merge_decisions` 诊断字段

```python
@dataclass
class ResumeContext:
    # ... 现有字段 ...

    # Phase 2 新增: 合并决策诊断
    merge_decisions: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_merged_resume(self) -> bool:
        """是否经过双源合并。"""
        return len(self.merge_decisions) > 0
```

## 4. 验收标准

1. **不丢 finalized_indices**: 暂停后恢复，已定稿的字幕索引完整保留
2. **不丢 sentences_snapshot**: 恢复后字幕快照正确推送到前端
3. **预处理不倒退**: 重启恢复时，已完成的预处理阶段不会重跑
4. **合并决策可追踪**: 每次恢复在日志中记录完整的 `merge_decisions`
5. **灰度安全**: 关闭 `RESUME_MERGER_ENABLED` 后回退到旧恢复路径

## 5. 测试要点

```python
# test_resume_state_merger.py

def test_merge_both_sources():
    """双源同时存在时正确合并。"""
    # 构造 runtime_state.db（预处理已完成到 separation）
    # 构造 checkpoint.json（转录已完成 fast[0-8], slow[0-6], finalized[0-4]）
    merger = ResumeStateMerger(job_dir)
    result = merger.merge()

    assert result.separation_completed is True       # 来自 runtime_state
    assert result.finalized_indices == {0,1,2,3,4}   # 来自 checkpoint.json
    assert len(result.merge_decisions) >= 2

def test_merge_only_checkpoint():
    """仅 checkpoint.json 存在时降级。"""
    # 无 runtime_state.db
    merger = ResumeStateMerger(job_dir)
    result = merger.merge()

    assert not result.runtime_state_available
    assert result.checkpoint_json_available
    # 预处理完成位来自 checkpoint.json 降级

def test_merge_only_runtime():
    """仅 runtime_state.db 存在时，转录从头开始。"""
    # 无 checkpoint.json
    merger = ResumeStateMerger(job_dir)
    result = merger.merge()

    assert result.runtime_state_available
    assert not result.checkpoint_json_available
    assert len(result.fast_processed_indices) == 0

def test_finalized_indices_never_regress():
    """PBT INV-5: finalized_indices 恢复后不倒退。"""
    # 暂停前记录 finalized_indices = {0,1,2,3}
    # 恢复后
    merged = merger.merge()
    assert merged.finalized_indices >= {0,1,2,3}

def test_merge_decisions_audit():
    """所有合并决策可追踪。"""
    merged = merger.merge()
    for decision in merged.merge_decisions:
        assert decision.field_name
        assert decision.source in {"runtime_state", "checkpoint_json", "default"}
        assert decision.reason
```
