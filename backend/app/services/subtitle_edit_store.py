import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# V3.2.0+dev.20260124.02: 用户字幕编辑落盘与叠加
EDIT_STORE_FILENAME = "subtitle_edits.json"


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """原子写入 JSON 文件，避免写入中断导致文件损坏。"""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def _normalize_index(value: Any) -> Optional[int]:
    """将索引字段标准化为 int。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_edit_store_path(job_dir: Path) -> Path:
    """获取用户编辑落盘文件路径。"""
    return job_dir / EDIT_STORE_FILENAME


def _build_state(
    data: Dict[str, Any]
) -> Tuple[Dict[int, Dict[str, Any]], Set[int], int]:
    raw_edits = data.get("edits", {})
    edits: Dict[int, Dict[str, Any]] = {}
    for key, value in raw_edits.items():
        index = _normalize_index(key)
        if index is None:
            continue
        if isinstance(value, dict):
            edits[index] = value

    raw_deleted = data.get("deleted_indices", [])
    deleted_indices: Set[int] = set()
    for item in raw_deleted:
        index = _normalize_index(item)
        if index is not None:
            deleted_indices.add(index)

    next_manual_index = data.get("next_manual_index")
    if next_manual_index is None:
        manual_indices = [idx for idx in edits.keys() if idx < 0]
        next_manual_index = min(manual_indices) - 1 if manual_indices else -1

    return edits, deleted_indices, int(next_manual_index)


def _load_state(
    job_dir: Path
) -> Tuple[Dict[int, Dict[str, Any]], Set[int], int]:
    """读取编辑状态（包含删除列表与手动索引序列）。"""
    path = get_edit_store_path(job_dir)
    if not path.exists():
        return {}, set(), -1

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _build_state(data)
    except Exception as exc:
        logger.warning("读取用户编辑落盘失败: %s", exc)
        return {}, set(), -1


def _persist_state(
    job_dir: Path,
    edits: Dict[int, Dict[str, Any]],
    deleted_indices: Set[int],
    next_manual_index: int
) -> None:
    path = get_edit_store_path(job_dir)
    payload = {
        "version": "3.2.0",
        "updated_at": time.time(),
        "edits": {str(k): v for k, v in edits.items()},
        "deleted_indices": sorted(deleted_indices),
        "next_manual_index": next_manual_index
    }
    _atomic_write_json(path, payload)


def load_edits(job_dir: Path) -> Dict[int, Dict[str, Any]]:
    """加载用户编辑记录。"""
    edits, _, _ = _load_state(job_dir)
    return edits


def load_deleted_indices(job_dir: Path) -> Set[int]:
    """加载已删除的字幕索引列表。"""
    _, deleted_indices, _ = _load_state(job_dir)
    return deleted_indices


def save_edit(
    job_dir: Path,
    sentence_index: int,
    update: Dict[str, Any],
    original_text: Optional[str] = None
) -> None:
    """保存单条用户编辑记录（覆盖式，保留原始文本）。"""
    edits, deleted_indices, next_manual_index = _load_state(job_dir)

    entry = edits.get(sentence_index, {})
    if original_text and not entry.get("original_text"):
        entry["original_text"] = original_text

    for key in ("text", "start", "end"):
        if key in update and update[key] is not None:
            entry[key] = update[key]

    entry["updated_at"] = time.time()
    edits[sentence_index] = entry

    _persist_state(job_dir, edits, deleted_indices, next_manual_index)


def add_deletion(job_dir: Path, sentence_index: int) -> None:
    """记录用户删除字幕的动作。"""
    edits, deleted_indices, next_manual_index = _load_state(job_dir)
    deleted_indices.add(sentence_index)
    # 删除手动字幕时，从 edits 中移除
    if sentence_index in edits and sentence_index < 0:
        del edits[sentence_index]
    _persist_state(job_dir, edits, deleted_indices, next_manual_index)


# V3.2.4+dev.20260303.01: undo/redo 后端增量同步支持函数


def remove_deletion(job_dir: Path, sentence_index: int) -> None:
    """撤销删除：从 deleted_indices 中移除（仅用于正索引字幕）。"""
    edits, deleted_indices, next_manual_index = _load_state(job_dir)
    if sentence_index not in deleted_indices:
        return
    deleted_indices.discard(sentence_index)
    _persist_state(job_dir, edits, deleted_indices, next_manual_index)


def remove_manual_entry(job_dir: Path, sentence_index: int) -> bool:
    """移除手动新增字幕（负索引），用于撤销新增。"""
    if sentence_index >= 0:
        return False
    edits, deleted_indices, next_manual_index = _load_state(job_dir)
    if sentence_index not in edits:
        return False
    del edits[sentence_index]
    _persist_state(job_dir, edits, deleted_indices, next_manual_index)
    return True


def restore_manual_entry(
    job_dir: Path,
    sentence_index: int,
    text: str,
    start: float,
    end: float,
) -> bool:
    """恢复已删除的手动字幕（负索引）：将数据写回 edits 并从 deleted_indices 移除。

    add_deletion 对负索引会 del edits[index]，仅靠 remove_deletion
    无法恢复丢失的数据，需要调用方提供完整内容重建。
    """
    if sentence_index >= 0:
        return False
    edits, deleted_indices, next_manual_index = _load_state(job_dir)
    edits[sentence_index] = {
        "text": text,
        "start": start,
        "end": end,
        "source": "manual",
        "updated_at": time.time(),
    }
    deleted_indices.discard(sentence_index)
    _persist_state(job_dir, edits, deleted_indices, next_manual_index)
    return True


def create_manual_entry(
    job_dir: Path,
    text: str,
    start: float,
    end: float
) -> Tuple[int, Dict[str, Any]]:
    """创建手动新增字幕记录，并返回索引。"""
    edits, deleted_indices, next_manual_index = _load_state(job_dir)
    new_index = int(next_manual_index)
    next_manual_index = new_index - 1

    entry = {
        "text": text,
        "start": start,
        "end": end,
        "source": "manual",
        "updated_at": time.time()
    }
    edits[new_index] = entry
    if new_index in deleted_indices:
        deleted_indices.remove(new_index)
    _persist_state(job_dir, edits, deleted_indices, next_manual_index)
    return new_index, entry


def _apply_edit_to_sentence(sentence: Dict[str, Any], edit: Dict[str, Any]) -> None:
    """将编辑内容叠加到单条句子数据上。"""
    existing_text = sentence.get("text", "")

    if edit.get("text") is not None:
        if sentence.get("original_text") is None:
            sentence["original_text"] = edit.get("original_text") or existing_text
        sentence["text"] = edit["text"]
        sentence["text_clean"] = edit["text"]
        sentence["confidence"] = None
        sentence["display_confidence"] = None
        sentence["confidence_source"] = "manual"

    if edit.get("start") is not None:
        sentence["start"] = edit["start"]
    if edit.get("end") is not None:
        sentence["end"] = edit["end"]

    sentence["is_modified"] = True


def apply_edits_to_sentences_snapshot(
    sentences_snapshot: list,
    edits: Dict[int, Dict[str, Any]]
) -> int:
    """将用户编辑叠加到 sentences_snapshot。"""
    updated_count = 0
    if not sentences_snapshot or not edits:
        return updated_count

    for sentence in sentences_snapshot:
        index = _normalize_index(sentence.get("_index"))
        if index is None:
            index = _normalize_index(sentence.get("index"))
        if index is None:
            continue
        edit = edits.get(index)
        if not edit:
            continue
        _apply_edit_to_sentence(sentence, edit)
        updated_count += 1

    return updated_count


def apply_edits_to_segments(
    segments: list,
    edits: Dict[int, Dict[str, Any]]
) -> int:
    """将用户编辑叠加到 API 输出的 segments。"""
    updated_count = 0
    if not segments or not edits:
        return updated_count

    for segment in segments:
        index = _normalize_index(segment.get("id"))
        if index is None:
            continue
        edit = edits.get(index)
        if not edit:
            continue
        _apply_edit_to_sentence(segment, edit)
        updated_count += 1

    return updated_count


def apply_deletions_to_segments(
    segments: list,
    deleted_indices: Set[int]
) -> list:
    """过滤被用户删除的字幕。"""
    if not segments or not deleted_indices:
        return segments
    return [segment for segment in segments if _normalize_index(segment.get("id")) not in deleted_indices]


def build_manual_segments(
    edits: Dict[int, Dict[str, Any]],
    deleted_indices: Set[int]
) -> list:
    """从编辑记录中构建手动新增字幕列表。"""
    manual_segments = []
    for index, entry in edits.items():
        if index >= 0:
            continue
        if index in deleted_indices:
            continue
        if entry.get("text") is None:
            continue
        manual_segments.append({
            "id": index,
            "start": entry.get("start", 0),
            "end": entry.get("end", 0),
            "text": entry.get("text", ""),
            "confidence": None,
            "display_confidence": None,
            "confidence_source": "manual",
            "source": entry.get("source", "manual"),
            "is_modified": True,
            "original_text": entry.get("original_text")
        })
    return manual_segments


def load_transcription_snapshot(
    job_dir: Path
) -> Optional[Tuple[Path, Dict[str, Any], str]]:
    """加载字幕快照数据（checkpoint 优先）。"""
    checkpoint_path = job_dir / "checkpoint.json"
    snapshot_path = job_dir / "transcription_text.json"

    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return checkpoint_path, data, "checkpoint"
        except Exception as exc:
            logger.warning("读取 checkpoint 失败: %s", exc)

    if snapshot_path.exists():
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return snapshot_path, data, "snapshot"
        except Exception as exc:
            logger.warning("读取字幕快照失败: %s", exc)

    return None


def persist_snapshot(path: Path, data: Dict[str, Any]) -> None:
    """写回字幕快照数据。"""
    _atomic_write_json(path, data)


def _get_sentences_snapshot(data: Dict[str, Any], kind: str) -> list:
    if kind == "checkpoint":
        return data.get("transcription", {}).get("sentences_snapshot", [])
    return data.get("sentences_snapshot", [])


def get_sentence_from_snapshot(
    data: Dict[str, Any],
    sentence_index: int,
    kind: str
) -> Optional[Dict[str, Any]]:
    """从快照数据中查找句子。"""
    sentences_snapshot = _get_sentences_snapshot(data, kind)
    for sentence in sentences_snapshot:
        index = _normalize_index(sentence.get("_index"))
        if index is None:
            index = _normalize_index(sentence.get("index"))
        if index == sentence_index:
            return sentence
    return None


def apply_edit_to_snapshot_data(
    data: Dict[str, Any],
    sentence_index: int,
    update: Dict[str, Any],
    original_text: Optional[str],
    kind: str
) -> bool:
    """将编辑应用到快照数据。"""
    sentences_snapshot = _get_sentences_snapshot(data, kind)
    for sentence in sentences_snapshot:
        index = _normalize_index(sentence.get("_index"))
        if index is None:
            index = _normalize_index(sentence.get("index"))
        if index != sentence_index:
            continue

        if original_text and not sentence.get("original_text"):
            sentence["original_text"] = original_text

        _apply_edit_to_sentence(sentence, update)
        return True

    return False
