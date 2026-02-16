"""
转录恢复轻量工具函数。

说明：
- 仅放置纯数据修正逻辑，避免路由层依赖过重导致单测难以隔离。
"""

from __future__ import annotations

from typing import Any, Dict, List


def force_finalize_snapshot_when_finished(
    *,
    job_status: str,
    sentences_snapshot: List[Dict[str, Any]],
) -> int:
    """
    finished 态强制收口快照草稿标记。

    返回值:
        int: 被修正的句子数量
    """
    if str(job_status).lower() != "finished":
        return 0

    changed_count = 0
    for sentence in sentences_snapshot:
        is_draft = bool(sentence.get("_is_draft", False))
        is_finalized = sentence.get("_is_finalized")
        if is_finalized is None:
            is_finalized = not is_draft
        if is_draft or not bool(is_finalized):
            sentence["_is_draft"] = False
            sentence["_is_finalized"] = True
            changed_count += 1
    return changed_count


def force_finalize_segments_when_finished(
    *,
    job_status: str,
    segments: List[Dict[str, Any]],
) -> int:
    """
    finished 态下统一收口返回段落，避免前端长期停留“生成中”。

    返回值:
        int: 被修正的段落数量
    """
    if str(job_status).lower() != "finished":
        return 0

    changed_count = 0
    for seg in segments:
        is_draft = bool(seg.get("is_draft", False))
        is_finalized = seg.get("is_finalized")
        if is_finalized is None:
            is_finalized = not is_draft
        if is_draft or not bool(is_finalized):
            seg["is_draft"] = False
            seg["is_finalized"] = True
            changed_count += 1
    return changed_count
