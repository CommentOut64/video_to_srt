"""
项目字幕与同音索引的增量同步辅助。

设计说明：
- 这里采用增量同步而非整表重建，针对少量字幕编辑只更新受影响的句子；
- 若同步失败，显式将索引状态标记为 failed，强制下次查询走全量重建兜底。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Sequence

from .runtime import get_homophone_service
from .service import SentenceRecord


logger = logging.getLogger(__name__)


def detect_language_from_segments(segments: Sequence[dict[str, Any]]) -> str:
    """根据当前项目字幕推断主语言，保持增量更新与现有索引语言一致。"""
    joined_text = "\n".join(str(item.get("text", "")) for item in segments)
    if re.search(r"[ぁ-んァ-ン]", joined_text):
        return "ja"
    if re.search(r"[\u4e00-\u9fff]", joined_text):
        return "zh"
    if re.search(r"[A-Za-z]", joined_text):
        return "en"
    return "zh"


def sync_project_index_delta(
    *,
    project_id: str,
    segments: Sequence[dict[str, Any]],
    updated_sentence_indices: Iterable[int] = (),
    removed_sentence_indices: Iterable[int] = (),
) -> bool:
    """
    将项目字幕的小范围变更同步到已存在的同音索引。

    返回值：
    - `True`：存在 ready 索引且已尝试同步；
    - `False`：当前没有可增量同步的 ready 索引，调用方无需视为错误。
    """
    service = get_homophone_service()
    state = service.get_index_status(project_id)
    if state is None or state.status != "ready":
        return False

    segment_map = {
        int(item.get("id", 0)): item
        for item in segments
        if item.get("id") is not None
    }
    normalized_updated = sorted({int(item) for item in updated_sentence_indices})
    normalized_removed = sorted(
        {int(item) for item in removed_sentence_indices}
        | {index for index in normalized_updated if index not in segment_map}
    )
    synced_records = [
        SentenceRecord(index=index, text=str(segment_map[index].get("text", "")))
        for index in normalized_updated
        if index in segment_map
    ]
    if not synced_records and not normalized_removed:
        return False

    try:
        if normalized_removed:
            service.remove_sentences(
                project_id=project_id,
                revision=state.revision,
                sentence_indices=normalized_removed,
            )
        if synced_records:
            language = detect_language_from_segments(list(segments))
            service.sync_sentences(
                project_id=project_id,
                revision=state.revision,
                language=language,
                sentences=synced_records,
            )
        return True
    except Exception:
        service.mark_index_failed(project_id=project_id, revision=state.revision)
        logger.exception(
            "同音索引增量同步失败，已标记为 failed: project_id=%s updated=%s removed=%s",
            project_id,
            normalized_updated,
            normalized_removed,
        )
        return True
