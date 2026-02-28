"""
恢复上下文数据类 - V3.2.0+dev.20260125.06

用于断点续传时传递恢复状态给预处理和转录流水线。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class ResumeContext:
    """恢复上下文 - 断点续传所需的完整状态"""
    checkpoint: Optional[Dict[str, Any]] = None
    safe_processed_indices: Set[int] = field(default_factory=set)
    fast_processed_indices: Set[int] = field(default_factory=set)
    slow_processed_indices: Set[int] = field(default_factory=set)
    finalized_indices: Set[int] = field(default_factory=set)
    previous_whisper_text: str = ""
    sentences_snapshot: List[Dict[str, Any]] = field(default_factory=list)
    sentence_count: int = 0
    chunk_sentences_map: Dict[Any, List[int]] = field(default_factory=dict)
    merge_decisions: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_resuming(self) -> bool:
        """是否处于恢复模式"""
        return self.checkpoint is not None

    @property
    def is_merged_resume(self) -> bool:
        """是否经过双源合并恢复。"""
        return len(self.merge_decisions) > 0

    @classmethod
    def from_checkpoint(cls, checkpoint: Optional[Dict[str, Any]]) -> "ResumeContext":
        """从检查点构建恢复上下文"""
        if not checkpoint:
            return cls(checkpoint=None)

        transcription = checkpoint.get("transcription", {}) or {}
        fast_worker = transcription.get("fast_worker", {}) or {}
        slow_worker = transcription.get("slow_worker", {}) or {}
        alignment = transcription.get("alignment", {}) or {}
        context_state = slow_worker.get("context_state", {}) or {}

        # 解析快流已处理索引
        fast_indices = set(
            fast_worker.get("processed_indices", [])
            or transcription.get("fast_processed_indices", [])
            or transcription.get("processed_indices", [])
            or []
        )
        # 解析慢流已处理索引
        slow_indices = set(
            slow_worker.get("processed_indices", [])
            or transcription.get("slow_processed_indices", [])
            or []
        )
        # 解析已对齐索引
        finalized_indices = set(
            alignment.get("finalized_indices", [])
            or transcription.get("finalized_indices", [])
            or []
        )

        # 解析 Whisper 上下文
        previous_whisper_text = (
            context_state.get("previous_whisper_text")
            or transcription.get("previous_whisper_text")
            or ""
        )

        # 计算安全已处理索引（可跳过的索引）
        safe_indices: Set[int] = set()
        if finalized_indices:
            max_finalized = max(finalized_indices)
            safe_indices = set(range(max_finalized + 1))
        elif fast_indices and slow_indices:
            safe_indices = fast_indices & slow_indices
        elif slow_indices:
            safe_indices = set(slow_indices)

        # 解析字幕快照
        sentences_snapshot = transcription.get("sentences_snapshot", []) or []
        sentence_count = transcription.get("sentence_count", 0) or 0
        raw_chunk_map = transcription.get("chunk_sentences_map", {}) or {}
        chunk_sentences_map: Dict[Any, List[int]] = {}
        if raw_chunk_map:
            for raw_key, raw_indices in raw_chunk_map.items():
                normalized_key: Any = raw_key
                if isinstance(raw_key, str) and raw_key.lstrip("-").isdigit():
                    try:
                        normalized_key = int(raw_key)
                    except ValueError:
                        normalized_key = raw_key
                if not isinstance(raw_indices, list):
                    continue
                normalized_indices: List[int] = []
                for raw_index in raw_indices:
                    try:
                        normalized_indices.append(int(raw_index))
                    except (TypeError, ValueError):
                        continue
                chunk_sentences_map[normalized_key] = normalized_indices

        return cls(
            checkpoint=checkpoint,
            safe_processed_indices=safe_indices,
            fast_processed_indices=fast_indices,
            slow_processed_indices=slow_indices,
            finalized_indices=finalized_indices,
            previous_whisper_text=previous_whisper_text,
            sentences_snapshot=sentences_snapshot,
            sentence_count=sentence_count,
            chunk_sentences_map=chunk_sentences_map,
        )
