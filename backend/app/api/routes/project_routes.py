"""
Project API 路由。
"""

from __future__ import annotations

import logging
import json
import shutil
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.config import FLAVOR
from app.services.file_service import FileManagementService
from app.services.project_service import get_project_service
from app.services.subtitle_doc_service import get_subtitle_doc_service
from app.services.subtitle_edit_store import (
    add_deletion,
    create_manual_entry,
    get_edit_store_path,
    load_deleted_indices,
    load_edits,
    remove_deletion,
    remove_manual_entry,
    restore_manual_entry,
    save_edit,
)
from app.services.sse_service import get_sse_manager
from app.utils.ass_converter import ASSConverter
from app.utils.text_utils import segments_to_srt

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["projects"])
_SUPPRESS_PROJECT_SUBTITLE_EVENTS: ContextVar[bool] = ContextVar(
    "suppress_project_subtitle_events",
    default=False,
)

# V3.2.4+dev.20260304.12: 统一导入媒体格式白名单，保持前后端一致。
SUPPORTED_IMPORT_MEDIA_EXTENSIONS = frozenset(
    FileManagementService.VIDEO_EXTENSIONS | FileManagementService.AUDIO_EXTENSIONS
)
MEDIA_UPLOAD_CHUNK_SIZE = 1024 * 1024


class ProjectTitleUpdateRequest(BaseModel):
    """更新项目标题请求。"""

    title: str = Field(..., min_length=1, max_length=200)


class SubtitleCreateRequest(BaseModel):
    """新增字幕请求。"""

    text: str = Field(default="")
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)


class SubtitleUpdateRequest(BaseModel):
    """更新字幕请求。"""

    text: Optional[str] = None
    start: Optional[float] = Field(default=None, ge=0.0)
    end: Optional[float] = Field(default=None, ge=0.0)


# V3.2.4+dev.20260303.01: undo/redo 批量同步请求模型


class BatchSyncUpdateItem(BaseModel):
    segment_id: str = Field(..., min_length=1)
    text: Optional[str] = None
    start: Optional[float] = Field(default=None, ge=0.0)
    end: Optional[float] = Field(default=None, ge=0.0)


class BatchSyncCreateItem(BaseModel):
    text: str = Field(default="")
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    restore_segment_id: Optional[str] = Field(default=None)


class BatchSyncDeleteItem(BaseModel):
    segment_id: str = Field(..., min_length=1)


class BatchSyncRequest(BaseModel):
    updates: list[BatchSyncUpdateItem] = Field(default_factory=list)
    creates: list[BatchSyncCreateItem] = Field(default_factory=list)
    deletes: list[BatchSyncDeleteItem] = Field(default_factory=list)


class EditorOpsApplyRequest(BaseModel):
    """批量编辑命令请求。"""

    session_id: str = Field(..., min_length=1)
    base_revision: Optional[int] = Field(default=None, ge=0)
    ops: list[dict[str, Any]] = Field(default_factory=list, min_length=1)


class ProjectSubtitleTimeOffsetRequest(BaseModel):
    """项目字幕时间偏移请求。"""

    offset: float = Field(
        ...,
        ge=-10.0,
        le=10.0,
        description="偏移量（秒），正值延后，负值提前，范围 -10.0 到 10.0",
    )


def _publish_project_subtitle_event(project_id: str, event_type: str, data: dict) -> None:
    if _SUPPRESS_PROJECT_SUBTITLE_EVENTS.get():
        return
    sse_manager = get_sse_manager()
    sse_manager.broadcast_sync(f"project:{project_id}", f"subtitle.{event_type}", data)


def _decode_upload_content(raw_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise HTTPException(status_code=400, detail="字幕文件编码不支持，请使用 UTF-8")


def _detect_subtitle_format(filename: str, hint: Optional[str]) -> Literal["srt", "ass", "vtt"]:
    normalized_hint = str(hint or "").strip().lower()
    if normalized_hint in {"srt", "ass", "vtt"}:
        return normalized_hint  # type: ignore[return-value]

    suffix = Path(filename or "").suffix.lower()
    if suffix == ".srt":
        return "srt"
    if suffix == ".ass":
        return "ass"
    if suffix == ".vtt":
        return "vtt"
    raise HTTPException(status_code=400, detail="无法识别字幕格式，请显式传入 subtitle_format")


def _parse_subtitle_content(content: str, fmt: str) -> list:
    """根据格式调用对应解析器"""
    sds = get_subtitle_doc_service()
    if fmt == "ass":
        return sds.parse_ass(content)
    elif fmt == "vtt":
        return sds.parse_vtt(content)
    return sds.parse_srt(content)


def _read_local_file(path: str) -> str:
    """读取本地文件内容，自动检测编码"""
    raw_bytes = Path(path).read_bytes()
    return _decode_upload_content(raw_bytes)


def _normalize_upload_filename(filename: str) -> str:
    """规范化上传文件名，避免路径注入和空文件名。"""
    normalized_name = Path(str(filename or "")).name.strip()
    if not normalized_name:
        raise HTTPException(status_code=400, detail="媒体文件名不能为空")
    return normalized_name


def _validate_import_media_extension(filename: str) -> str:
    """校验导入媒体扩展名是否在白名单内。"""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_IMPORT_MEDIA_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_IMPORT_MEDIA_EXTENSIONS))
        normalized_suffix = suffix or "<none>"
        raise HTTPException(
            status_code=400,
            detail=f"不支持的媒体格式: {normalized_suffix}，支持格式: {supported}",
        )
    return suffix


async def _persist_uploaded_media_file(
    upload_file: UploadFile,
    target_path: Path,
    chunk_size: int = MEDIA_UPLOAD_CHUNK_SIZE,
) -> int:
    """
    分块写入上传媒体，避免一次性读取大文件导致内存峰值和超时放大。
    """
    total_written_bytes = 0
    with target_path.open("wb") as output_fp:
        while True:
            chunk = await upload_file.read(chunk_size)
            if not chunk:
                break
            output_fp.write(chunk)
            total_written_bytes += len(chunk)
    return total_written_bytes


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


EDITOR_OPS_STATE_FILE = ".editor_ops_state.json"
EDITOR_OPS_STATE_VERSION = 1


def _editor_ops_state_path(project_dir: Path) -> Path:
    return project_dir / EDITOR_OPS_STATE_FILE


def _load_editor_ops_state(project_dir: Path) -> dict[str, Any]:
    state_path = _editor_ops_state_path(project_dir)
    if not state_path.exists():
        return {
            "version": EDITOR_OPS_STATE_VERSION,
            "server_revision": 0,
            "sessions": {},
        }

    try:
        with state_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        logger.warning("editor-ops 状态文件损坏，已重建: %s", state_path)
        return {
            "version": EDITOR_OPS_STATE_VERSION,
            "server_revision": 0,
            "sessions": {},
        }

    server_revision = payload.get("server_revision", 0)
    sessions = payload.get("sessions", {})
    return {
        "version": EDITOR_OPS_STATE_VERSION,
        "server_revision": int(server_revision) if isinstance(server_revision, int) else 0,
        "sessions": sessions if isinstance(sessions, dict) else {},
    }


def _save_editor_ops_state(project_dir: Path, state: dict[str, Any]) -> None:
    state_path = _editor_ops_state_path(project_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_state = {
        "version": EDITOR_OPS_STATE_VERSION,
        "server_revision": int(state.get("server_revision", 0) or 0),
        "sessions": state.get("sessions", {}),
    }
    with state_path.open("w", encoding="utf-8") as file:
        json.dump(normalized_state, file, ensure_ascii=False, indent=2)


def _backup_subtitle_edit_store(project_dir: Path) -> Optional[Path]:
    edit_store_path = get_edit_store_path(project_dir)
    if not edit_store_path.exists():
        return None
    backup_path = edit_store_path.with_suffix(f"{edit_store_path.suffix}.editor_ops.bak")
    shutil.copy2(edit_store_path, backup_path)
    return backup_path


def _restore_subtitle_edit_store(project_dir: Path, backup_path: Optional[Path]) -> None:
    edit_store_path = get_edit_store_path(project_dir)
    if backup_path is None:
        if edit_store_path.exists():
            edit_store_path.unlink()
        return

    if backup_path.exists():
        shutil.copy2(backup_path, edit_store_path)
        backup_path.unlink(missing_ok=True)


def _cleanup_subtitle_edit_store_backup(backup_path: Optional[Path]) -> None:
    if backup_path is not None:
        backup_path.unlink(missing_ok=True)


def _error_code_for_status(status_code: int) -> str:
    if status_code == 404:
        return "SEGMENT_NOT_FOUND"
    if status_code == 409:
        return "REVISION_CONFLICT"
    if status_code == 400:
        return "INVALID_REQUEST"
    return "INTERNAL_ERROR"


def _editor_ops_error(
    *,
    status_code: int,
    message: str,
    server_revision: Optional[int] = None,
    failed_op_id: Optional[str] = None,
    error_code: Optional[str] = None,
    details: Optional[dict[str, Any]] = None,
) -> JSONResponse:
    payload: dict[str, Any] = {
        "success": False,
        "error_code": error_code or _error_code_for_status(status_code),
        "message": message,
    }
    if server_revision is not None:
        payload["server_revision"] = server_revision
    if failed_op_id:
        payload["failed_op_id"] = failed_op_id
    if details:
        payload["details"] = details
    return JSONResponse(status_code=status_code, content=payload)


def _collect_project_segments(project_dir: Path) -> list[dict]:
    runtime_segments = _load_runtime_subtitle_segments(project_dir)
    if runtime_segments:
        return _compose_runtime_segments_with_user_edits(project_dir, runtime_segments)
    return get_subtitle_doc_service().load_segments(project_dir)


def _sync_homophone_index_for_project_change(
    *,
    project_id: str,
    project_dir: Path,
    updated_sentence_indices: list[int] | None = None,
    removed_sentence_indices: list[int] | None = None,
) -> None:
    """在 ready 索引存在时按句增量同步；无索引时静默跳过。"""
    from app.services.homophone.project_sync import sync_project_index_delta

    sync_project_index_delta(
        project_id=project_id,
        segments=_collect_project_segments(project_dir),
        updated_sentence_indices=updated_sentence_indices or [],
        removed_sentence_indices=removed_sentence_indices or [],
    )


def _normalize_editor_segment_id(value: Any) -> str:
    return str(value or "").strip()


def _resolve_editor_segment_id(
    *,
    project_dir: Path,
    op: dict[str, Any],
    request_bindings: dict[str, str],
    segment_field: str,
    client_ref_field: str,
) -> Optional[str]:
    explicit_segment_id = _normalize_editor_segment_id(op.get(segment_field))
    if explicit_segment_id:
        return explicit_segment_id

    client_ref_id = _normalize_editor_segment_id(op.get(client_ref_field))
    if client_ref_id and client_ref_id in request_bindings:
        return request_bindings[client_ref_id]

    if client_ref_id:
        existing_segment = _find_segment_by_segment_id(
            _collect_project_segments(project_dir),
            client_ref_id,
        )
        if existing_segment is not None:
            return client_ref_id

    return None


def _require_ms_field(op: dict[str, Any], container_key: str, field_key: str) -> int:
    container = op.get(container_key)
    if not isinstance(container, dict) or container.get(field_key) is None:
        raise HTTPException(
            status_code=400,
            detail=f"{container_key}.{field_key} 不能为空",
        )
    try:
        return int(container[field_key])
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{container_key}.{field_key} 必须是整数毫秒",
        ) from exc


def _require_text_field(op: dict[str, Any], container_key: str, field_key: str = "text") -> str:
    container = op.get(container_key)
    if not isinstance(container, dict) or container.get(field_key) is None:
        raise HTTPException(
            status_code=400,
            detail=f"{container_key}.{field_key} 不能为空",
        )
    return str(container.get(field_key, ""))


def _ms_to_seconds(ms: int) -> float:
    return float(ms) / 1000.0


def _build_updated_entity(
    segment: dict[str, Any],
    *,
    client_ref_id: Optional[str],
    revision: Optional[int] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "client_ref_id": client_ref_id,
        "segment_id": str(segment.get("segment_id", "") or ""),
        "text": str(segment.get("text", "") or ""),
        "start": _safe_float(segment.get("start"), 0.0),
        "end": _safe_float(segment.get("end"), _safe_float(segment.get("start"), 0.0)),
    }
    if revision is not None:
        payload["revision"] = revision
    return payload


def _build_editor_op_result(
    *,
    op_id: str,
    created_bindings: Optional[list[dict[str, str]]] = None,
    updated_entities: Optional[list[dict[str, Any]]] = None,
    events: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    return {
        "normalized_result": {
            "op_id": op_id,
            "applied": True,
            "warnings": [],
        },
        "created_bindings": created_bindings or [],
        "updated_entities": updated_entities or [],
        "events": events or [],
    }


def _aggregate_editor_op_result(
    *,
    result_entry: dict[str, Any],
    request_bindings: dict[str, str],
    created_bindings: list[dict[str, str]],
    normalized_results: list[dict[str, Any]],
    updated_entities: list[dict[str, Any]],
) -> None:
    for binding in result_entry.get("created_bindings", []):
        client_ref_id = _normalize_editor_segment_id(binding.get("client_ref_id"))
        segment_id = _normalize_editor_segment_id(binding.get("segment_id"))
        if client_ref_id and segment_id:
            request_bindings[client_ref_id] = segment_id
            created_bindings.append(
                {
                    "client_ref_id": client_ref_id,
                    "segment_id": segment_id,
                }
            )
    normalized_result = result_entry.get("normalized_result")
    if isinstance(normalized_result, dict):
        normalized_results.append(normalized_result)
    for entity in result_entry.get("updated_entities", []):
        if isinstance(entity, dict):
            updated_entities.append(entity)


async def _apply_editor_op(
    *,
    project_id: str,
    project_dir: Path,
    op: dict[str, Any],
    request_bindings: dict[str, str],
) -> dict[str, Any]:
    op_id = _normalize_editor_segment_id(op.get("op_id"))
    op_type = _normalize_editor_segment_id(op.get("type"))
    if not op_id:
        raise HTTPException(status_code=400, detail="op_id 不能为空")
    if not op_type:
        raise HTTPException(status_code=400, detail="type 不能为空")

    if op_type == "update_text":
        segment_id = _resolve_editor_segment_id(
            project_dir=project_dir,
            op=op,
            request_bindings=request_bindings,
            segment_field="segment_id",
            client_ref_field="client_ref_id",
        )
        if not segment_id:
            raise HTTPException(status_code=404, detail="未找到 update_text 对应字幕")
        updated_text = _require_text_field(op, "after")
        response = await update_project_subtitle(
            project_id,
            segment_id,
            SubtitleUpdateRequest(text=updated_text),
        )
        return _build_editor_op_result(
            op_id=op_id,
            updated_entities=[
                _build_updated_entity(
                    response["data"],
                    client_ref_id=_normalize_editor_segment_id(op.get("client_ref_id")) or None,
                )
            ],
            events=[
                {
                    "event_type": "edited",
                    "data": {
                        "segment_id": segment_id,
                        "segment": response["data"],
                        "source": "project_api",
                        "is_update": True,
                    },
                }
            ],
        )

    if op_type == "update_timing":
        segment_id = _resolve_editor_segment_id(
            project_dir=project_dir,
            op=op,
            request_bindings=request_bindings,
            segment_field="segment_id",
            client_ref_field="client_ref_id",
        )
        if not segment_id:
            raise HTTPException(status_code=404, detail="未找到 update_timing 对应字幕")
        start_ms = _require_ms_field(op, "after", "start_ms")
        end_ms = _require_ms_field(op, "after", "end_ms")
        response = await update_project_subtitle(
            project_id,
            segment_id,
            SubtitleUpdateRequest(
                start=_ms_to_seconds(start_ms),
                end=_ms_to_seconds(end_ms),
            ),
        )
        return _build_editor_op_result(
            op_id=op_id,
            updated_entities=[
                _build_updated_entity(
                    response["data"],
                    client_ref_id=_normalize_editor_segment_id(op.get("client_ref_id")) or None,
                )
            ],
            events=[
                {
                    "event_type": "edited",
                    "data": {
                        "segment_id": segment_id,
                        "segment": response["data"],
                        "source": "project_api",
                        "is_update": True,
                    },
                }
            ],
        )

    if op_type == "insert_subtitle":
        client_ref_id = _normalize_editor_segment_id(op.get("client_ref_id"))
        if not client_ref_id:
            raise HTTPException(status_code=400, detail="insert_subtitle.client_ref_id 不能为空")
        start_ms = _require_ms_field(op, "after", "start_ms")
        end_ms = _require_ms_field(op, "after", "end_ms")
        text = _require_text_field(op, "after")
        response = await create_project_subtitle(
            project_id,
            SubtitleCreateRequest(
                text=text,
                start=_ms_to_seconds(start_ms),
                end=_ms_to_seconds(end_ms),
            ),
        )
        segment = response["data"]
        segment_id = _normalize_editor_segment_id(segment.get("segment_id"))
        request_bindings[client_ref_id] = segment_id
        return _build_editor_op_result(
            op_id=op_id,
            created_bindings=[
                {
                    "client_ref_id": client_ref_id,
                    "segment_id": segment_id,
                }
            ],
            updated_entities=[
                _build_updated_entity(segment, client_ref_id=client_ref_id)
            ],
            events=[
                {
                    "event_type": "added",
                    "data": {
                        "segment": segment,
                        "source": "project_api",
                        "is_update": True,
                    },
                }
            ],
        )

    if op_type == "delete_subtitle":
        segment_id = _resolve_editor_segment_id(
            project_dir=project_dir,
            op=op,
            request_bindings=request_bindings,
            segment_field="segment_id",
            client_ref_field="client_ref_id",
        )
        if not segment_id:
            raise HTTPException(status_code=404, detail="未找到 delete_subtitle 对应字幕")
        await delete_project_subtitle(project_id, segment_id)
        return _build_editor_op_result(
            op_id=op_id,
            events=[
                {
                    "event_type": "deleted",
                    "data": {
                        "segment_id": segment_id,
                        "source": "project_api",
                        "is_update": True,
                    },
                }
            ],
        )

    if op_type == "split_subtitle":
        source_segment_id = _resolve_editor_segment_id(
            project_dir=project_dir,
            op=op,
            request_bindings=request_bindings,
            segment_field="segment_id",
            client_ref_field="client_ref_id",
        )
        if not source_segment_id:
            raise HTTPException(status_code=404, detail="未找到 split_subtitle 对应字幕")
        created_client_ref_id = _normalize_editor_segment_id(op.get("created_client_ref_id"))
        if not created_client_ref_id:
            raise HTTPException(status_code=400, detail="created_client_ref_id 不能为空")
        kept_start_ms = _require_ms_field(op, "after_kept", "start_ms")
        kept_end_ms = _require_ms_field(op, "after_kept", "end_ms")
        created_start_ms = _require_ms_field(op, "after_created", "start_ms")
        created_end_ms = _require_ms_field(op, "after_created", "end_ms")
        kept_text = _require_text_field(op, "after_kept")
        created_text = _require_text_field(op, "after_created")

        kept_response = await update_project_subtitle(
            project_id,
            source_segment_id,
            SubtitleUpdateRequest(
                text=kept_text,
                start=_ms_to_seconds(kept_start_ms),
                end=_ms_to_seconds(kept_end_ms),
            ),
        )
        created_response = await create_project_subtitle(
            project_id,
            SubtitleCreateRequest(
                text=created_text,
                start=_ms_to_seconds(created_start_ms),
                end=_ms_to_seconds(created_end_ms),
            ),
        )
        created_segment = created_response["data"]
        created_segment_id = _normalize_editor_segment_id(created_segment.get("segment_id"))
        request_bindings[created_client_ref_id] = created_segment_id
        return _build_editor_op_result(
            op_id=op_id,
            created_bindings=[
                {
                    "client_ref_id": created_client_ref_id,
                    "segment_id": created_segment_id,
                }
            ],
            updated_entities=[
                _build_updated_entity(
                    kept_response["data"],
                    client_ref_id=_normalize_editor_segment_id(op.get("client_ref_id")) or None,
                ),
                _build_updated_entity(
                    created_segment,
                    client_ref_id=created_client_ref_id,
                ),
            ],
            events=[
                {
                    "event_type": "edited",
                    "data": {
                        "segment_id": source_segment_id,
                        "segment": kept_response["data"],
                        "source": "project_api",
                        "is_update": True,
                    },
                },
                {
                    "event_type": "added",
                    "data": {
                        "segment": created_segment,
                        "source": "project_api",
                        "is_update": True,
                    },
                },
            ],
        )

    if op_type == "merge_subtitle":
        kept_segment_id = _resolve_editor_segment_id(
            project_dir=project_dir,
            op=op,
            request_bindings=request_bindings,
            segment_field="kept_segment_id",
            client_ref_field="kept_client_ref_id",
        )
        removed_segment_id = _resolve_editor_segment_id(
            project_dir=project_dir,
            op=op,
            request_bindings=request_bindings,
            segment_field="removed_segment_id",
            client_ref_field="removed_client_ref_id",
        )
        if not kept_segment_id or not removed_segment_id:
            raise HTTPException(status_code=404, detail="未找到 merge_subtitle 对应字幕")
        start_ms = _require_ms_field(op, "after", "start_ms")
        end_ms = _require_ms_field(op, "after", "end_ms")
        text = _require_text_field(op, "after")
        kept_response = await update_project_subtitle(
            project_id,
            kept_segment_id,
            SubtitleUpdateRequest(
                text=text,
                start=_ms_to_seconds(start_ms),
                end=_ms_to_seconds(end_ms),
            ),
        )
        await delete_project_subtitle(project_id, removed_segment_id)
        return _build_editor_op_result(
            op_id=op_id,
            updated_entities=[
                _build_updated_entity(
                    kept_response["data"],
                    client_ref_id=_normalize_editor_segment_id(op.get("kept_client_ref_id")) or None,
                )
            ],
            events=[
                {
                    "event_type": "edited",
                    "data": {
                        "segment_id": kept_segment_id,
                        "segment": kept_response["data"],
                        "source": "project_api",
                        "is_update": True,
                    },
                },
                {
                    "event_type": "deleted",
                    "data": {
                        "segment_id": removed_segment_id,
                        "source": "project_api",
                        "is_update": True,
                    },
                },
            ],
        )

    if op_type == "move_boundary":
        upper_segment_id = _resolve_editor_segment_id(
            project_dir=project_dir,
            op=op,
            request_bindings=request_bindings,
            segment_field="upper_segment_id",
            client_ref_field="upper_client_ref_id",
        )
        lower_segment_id = _resolve_editor_segment_id(
            project_dir=project_dir,
            op=op,
            request_bindings=request_bindings,
            segment_field="lower_segment_id",
            client_ref_field="lower_client_ref_id",
        )
        if not upper_segment_id or not lower_segment_id:
            raise HTTPException(status_code=404, detail="未找到 move_boundary 对应字幕")
        upper_end_ms = _require_ms_field(op, "after", "upper_end_ms")
        lower_start_ms = _require_ms_field(op, "after", "lower_start_ms")
        if upper_end_ms != lower_start_ms:
            raise HTTPException(status_code=400, detail="共享边界必须保持一致")
        upper_response = await update_project_subtitle(
            project_id,
            upper_segment_id,
            SubtitleUpdateRequest(end=_ms_to_seconds(upper_end_ms)),
        )
        lower_response = await update_project_subtitle(
            project_id,
            lower_segment_id,
            SubtitleUpdateRequest(start=_ms_to_seconds(lower_start_ms)),
        )
        return _build_editor_op_result(
            op_id=op_id,
            updated_entities=[
                _build_updated_entity(
                    upper_response["data"],
                    client_ref_id=_normalize_editor_segment_id(op.get("upper_client_ref_id")) or None,
                ),
                _build_updated_entity(
                    lower_response["data"],
                    client_ref_id=_normalize_editor_segment_id(op.get("lower_client_ref_id")) or None,
                ),
            ],
            events=[
                {
                    "event_type": "edited",
                    "data": {
                        "segment_id": upper_segment_id,
                        "segment": upper_response["data"],
                        "source": "project_api",
                        "is_update": True,
                    },
                },
                {
                    "event_type": "edited",
                    "data": {
                        "segment_id": lower_segment_id,
                        "segment": lower_response["data"],
                        "source": "project_api",
                        "is_update": True,
                    },
                },
            ],
        )

    if op_type == "batch_replace":
        replacements = op.get("replacements")
        if not isinstance(replacements, list):
            raise HTTPException(status_code=400, detail="replacements 必须是数组")
        entities: list[dict[str, Any]] = []
        for replacement in replacements:
            if not isinstance(replacement, dict):
                raise HTTPException(status_code=400, detail="replacement 项必须是对象")
            segment_id = _resolve_editor_segment_id(
                project_dir=project_dir,
                op=replacement,
                request_bindings=request_bindings,
                segment_field="segment_id",
                client_ref_field="client_ref_id",
            )
            if not segment_id:
                raise HTTPException(status_code=404, detail="batch_replace 存在未绑定字幕")
            updated_text = _require_text_field(replacement, "after")
            response = await update_project_subtitle(
                project_id,
                segment_id,
                SubtitleUpdateRequest(text=updated_text),
            )
            entities.append(
                _build_updated_entity(
                    response["data"],
                    client_ref_id=_normalize_editor_segment_id(replacement.get("client_ref_id")) or None,
                )
            )
        return _build_editor_op_result(
            op_id=op_id,
            updated_entities=entities,
            events=[
                {
                    "event_type": "edited",
                    "data": {
                        "segment_id": entity["segment_id"],
                        "segment": entity,
                        "source": "project_api",
                        "is_update": True,
                    },
                }
                for entity in entities
            ],
        )

    raise HTTPException(status_code=400, detail=f"未知命令类型: {op_type}")


def _get_transcription_service() -> Any:
    from app.core.config import config as app_config
    from app.services.transcription_service import get_transcription_service

    return get_transcription_service(str(app_config.JOBS_DIR))


def _resolve_project_identity(project_id: str) -> Any:
    from app.services.project_id_resolver import get_project_id_resolver

    return get_project_id_resolver().resolve_or_fail(project_id)


def _get_global_subtitle_time_offset() -> float:
    from app.services.user_config_service import get_user_config_service

    return float(get_user_config_service().get_subtitle_time_offset())


def _find_runtime_job_for_project(
    transcription_service: Any,
    project_id: str,
    *,
    legacy_job_id: Optional[str] = None,
) -> Optional[Any]:
    runtime_job = transcription_service.get_job(project_id)
    if runtime_job is not None:
        return runtime_job

    normalized_legacy_job_id = str(legacy_job_id or "").strip()
    if normalized_legacy_job_id:
        runtime_job = transcription_service.get_job(normalized_legacy_job_id)
        if runtime_job is not None:
            return runtime_job

    job_lifecycle = getattr(transcription_service, "job_lifecycle", None)
    runtime_jobs = getattr(job_lifecycle, "jobs", None)
    if isinstance(runtime_jobs, dict):
        for candidate in runtime_jobs.values():
            if str(getattr(candidate, "project_id", "") or "").strip() == project_id:
                return candidate

    return None


def _normalize_runtime_sentence_to_segment(sentence: dict[str, Any]) -> Optional[dict[str, Any]]:
    """将 runtime 句子快照标准化为 project 字幕段。"""
    if not isinstance(sentence, dict):
        return None

    raw_index = sentence.get("_index", sentence.get("index"))
    try:
        legacy_index = int(raw_index)
    except (TypeError, ValueError):
        return None

    text = str(sentence.get("text", "") or "")
    if not text.strip():
        return None

    start = _safe_float(sentence.get("start"), 0.0)
    end = _safe_float(sentence.get("end"), start)
    if end < start:
        end = start

    sentence_uid = str(
        sentence.get("sentence_uid")
        or sentence.get("segment_id")
        or f"seg-{legacy_index}"
    )
    chunk_uid = sentence.get("chunk_uid")

    return {
        "segment_id": sentence_uid,
        "sentence_uid": sentence_uid,
        "chunk_uid": str(chunk_uid) if chunk_uid is not None else None,
        "text": text,
        "start": start,
        "end": end,
        "legacy_index": legacy_index,
        "source_type": "transcribe",
        "is_modified": bool(sentence.get("is_modified", False)),
        "original_text": sentence.get("original_text"),
    }


def _load_runtime_subtitle_segments(project_dir: Path) -> list[dict]:
    """
    从 runtime_state / checkpoint 加载运行态字幕基线。

    返回空列表表示当前项目无运行态字幕基线（例如纯导入编辑项目）。
    """
    runtime_payload: Optional[dict] = None
    try:
        from app.services.checkpoint import RuntimeCheckpointService

        runtime_service = RuntimeCheckpointService(job_dir=project_dir)
        runtime_payload = runtime_service.load_subtitle_runtime()
    except Exception as exc:
        logger.warning("读取 runtime 字幕快照失败，降级 checkpoint 路径: %s", exc)
    sentences_snapshot: list[dict[str, Any]] = []

    if runtime_payload:
        raw_snapshot = runtime_payload.get("sentences_snapshot", [])
        if isinstance(raw_snapshot, list):
            sentences_snapshot = [item for item in raw_snapshot if isinstance(item, dict)]

    if not sentences_snapshot:
        restored_from_checkpoint = _restore_segments_from_checkpoint(project_dir)
        if restored_from_checkpoint:
            return restored_from_checkpoint
        restored_from_srt = _restore_segments_from_srt(project_dir)
        if restored_from_srt:
            return restored_from_srt
        return []

    segments: list[dict] = []
    for sentence in sentences_snapshot:
        segment = _normalize_runtime_sentence_to_segment(sentence)
        if segment is not None:
            segments.append(segment)

    segments.sort(
        key=lambda item: (
            float(item.get("start", 0.0)),
            float(item.get("end", 0.0)),
            int(item.get("legacy_index", 0) or 0),
        )
    )
    return segments


def _is_user_delta_entry(entry: dict) -> bool:
    """判断编辑条目是否为用户增量（避免把历史全量快照当作覆盖源）。"""
    if not isinstance(entry, dict):
        return False
    if entry.get("source") == "manual":
        return True
    if entry.get("is_modified") is True:
        return True
    if entry.get("original_text") is not None:
        return True
    return False


def _manual_segment_id(index: int) -> str:
    return f"manual-{abs(int(index))}"


def _compose_runtime_segments_with_user_edits(
    project_dir: Path,
    runtime_segments: list[dict],
) -> list[dict]:
    """构建“运行态基线 + 用户增量”的组合视图。"""
    edits = load_edits(project_dir)
    deleted_indices = set(load_deleted_indices(project_dir))

    composed: list[dict] = []
    for segment in runtime_segments:
        legacy_index = int(segment.get("legacy_index", 0))
        if legacy_index in deleted_indices:
            continue
        merged_segment = dict(segment)
        edit_entry = edits.get(legacy_index)
        if edit_entry and _is_user_delta_entry(edit_entry):
            if edit_entry.get("text") is not None:
                merged_segment["original_text"] = merged_segment.get("text")
                merged_segment["text"] = str(edit_entry.get("text", ""))
            if edit_entry.get("start") is not None:
                merged_segment["start"] = _safe_float(edit_entry.get("start"), merged_segment.get("start", 0.0))
            if edit_entry.get("end") is not None:
                merged_segment["end"] = _safe_float(edit_entry.get("end"), merged_segment.get("end", merged_segment.get("start", 0.0)))
            if float(merged_segment["end"]) < float(merged_segment["start"]):
                merged_segment["end"] = merged_segment["start"]
            merged_segment["is_modified"] = True
        composed.append(merged_segment)

    # 手动新增字幕（负索引）
    for index, edit_entry in edits.items():
        if int(index) >= 0:
            continue
        if int(index) in deleted_indices:
            continue
        text = str(edit_entry.get("text", "") or "")
        # 手动新增字幕允许空文本（仅时间块占位）。
        # 若这里过滤空文本，会导致“新增后立即导出”丢块，与前端强制保存栅栏语义不一致。
        start = _safe_float(edit_entry.get("start"), 0.0)
        end = _safe_float(edit_entry.get("end"), start)
        if end < start:
            end = start
        composed.append(
            {
                "segment_id": _manual_segment_id(int(index)),
                "sentence_uid": None,
                "chunk_uid": "chunk:manual",
                "text": text,
                "start": start,
                "end": end,
                "legacy_index": int(index),
                "source_type": "manual",
                "is_modified": True,
                "original_text": edit_entry.get("original_text"),
            }
        )

    composed.sort(
        key=lambda item: (
            float(item.get("start", 0.0)),
            float(item.get("end", 0.0)),
            int(item.get("legacy_index", 0) or 0),
        )
    )
    return composed


def _find_segment_by_segment_id(segments: list[dict], segment_id: str) -> Optional[dict]:
    normalized = str(segment_id or "").strip()
    if not normalized:
        return None
    for segment in segments:
        if str(segment.get("segment_id", "")).strip() == normalized:
            return segment
    return None


def _find_segment_by_legacy_index(segments: list[dict], sentence_index: int) -> Optional[dict]:
    target_index = int(sentence_index)
    for segment in segments:
        legacy_index = segment.get("legacy_index")
        if legacy_index is None:
            continue
        try:
            if int(legacy_index) == target_index:
                return segment
        except (TypeError, ValueError):
            continue
    return None


def _build_runtime_export_segments(project_dir: Path) -> Optional[list[dict]]:
    """
    构建导出使用的字幕段：
    1) 优先 runtime 基线 + 用户增量；
    2) 无 runtime 时回退 subtitle_doc（导入/纯编辑模式）。
    """
    runtime_segments = _load_runtime_subtitle_segments(project_dir)
    if not runtime_segments:
        return None
    return _compose_runtime_segments_with_user_edits(project_dir, runtime_segments)


def _export_segments_as_srt(segments: list[dict]) -> str:
    raw_segments = [
        {
            "start": float(item.get("start", 0.0) or 0.0),
            "end": float(item.get("end", item.get("start", 0.0)) or item.get("start", 0.0)),
            "text": str(item.get("text", "") or ""),
        }
        for item in segments
    ]
    return segments_to_srt(raw_segments)


def _export_segments_as_ass(project_dir: Path, segments: list[dict]) -> str:
    style = ASSConverter.STYLE_PRESETS["default"]
    content_parts: list[str] = [
        ASSConverter.generate_script_info(title=project_dir.name),
        "\n[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        ASSConverter.generate_style_section(style),
        "\n[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for seg in segments:
        start = ASSConverter.format_ass_timestamp(float(seg.get("start", 0.0) or 0.0))
        end = ASSConverter.format_ass_timestamp(
            float(seg.get("end", seg.get("start", 0.0)) or seg.get("start", 0.0))
        )
        text = str(seg.get("text", "") or "").replace("\n", "\\N")
        content_parts.append(
            f"Dialogue: 0,{start},{end},{style.name},,0,0,0,,{text}"
        )
    return "\n".join(content_parts)


def _restore_segments_from_checkpoint(project_dir: Path) -> list[dict]:
    """
    从 checkpoint/transcription_text 恢复字幕段（仅用于缺失真源时修复）。
    """
    candidate_paths = [
        project_dir / "checkpoint.json",
        project_dir / "transcription_text.json",
    ]
    for snapshot_path in candidate_paths:
        if not snapshot_path.exists():
            continue
        try:
            with open(snapshot_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception:
            continue

        transcription_payload = payload.get("transcription", payload)
        sentences = transcription_payload.get("sentences_snapshot", [])
        if not isinstance(sentences, list) or not sentences:
            continue

        segments = []
        for sentence in sentences:
            if not isinstance(sentence, dict):
                continue
            text = str(sentence.get("text", "") or "")
            if not text.strip():
                continue
            raw_index = sentence.get("_index", sentence.get("index"))
            try:
                legacy_index = int(raw_index)
            except (TypeError, ValueError):
                legacy_index = len(segments)
            start = float(sentence.get("start", 0.0) or 0.0)
            end = float(sentence.get("end", start) or start)
            if end < start:
                end = start
            segments.append(
                {
                    "segment_id": str(
                        sentence.get("sentence_uid")
                        or sentence.get("segment_id")
                        or f"seg-{legacy_index}"
                    ),
                    "sentence_uid": sentence.get("sentence_uid"),
                    "chunk_uid": sentence.get("chunk_uid"),
                    "text": text,
                    "start": start,
                    "end": end,
                    "legacy_index": legacy_index,
                    "source_type": "transcribe",
                    "is_modified": bool(sentence.get("is_modified", False)),
                    "original_text": sentence.get("original_text"),
                }
            )

        if segments:
            segments.sort(
                key=lambda item: (
                    float(item["start"]),
                    float(item["end"]),
                    int(item.get("legacy_index", 0) or 0),
                )
            )
            return segments
    return []


def _restore_segments_from_srt(project_dir: Path) -> list[dict]:
    """
    从目录中最新 SRT 文件恢复字幕段（仅用于缺失真源时修复）。
    """
    subtitle_doc_service = get_subtitle_doc_service()
    srt_files = sorted(
        project_dir.glob("*.srt"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    for srt_path in srt_files:
        try:
            raw_bytes = srt_path.read_bytes()
        except Exception:
            continue
        try:
            content = _decode_upload_content(raw_bytes)
        except HTTPException:
            continue
        parsed_segments = subtitle_doc_service.parse_srt(content)
        segments = []
        for index, segment in enumerate(parsed_segments):
            normalized_segment = dict(segment)
            normalized_segment["segment_id"] = str(
                normalized_segment.get("segment_id") or f"seg-{index}"
            )
            normalized_segment["legacy_index"] = int(index)
            segments.append(normalized_segment)
        if segments:
            return segments
    return []


@router.post("/import")
async def import_project(
    subtitle_file: UploadFile = File(...),
    title: str = Form(default=""),
    flavor: str = Form(default=FLAVOR),
    subtitle_format: Optional[str] = Form(default=None),
    video_file: Optional[UploadFile] = File(default=None),
):
    """导入字幕并创建项目。"""
    project_service = get_project_service()

    raw_bytes = await subtitle_file.read()
    content = _decode_upload_content(raw_bytes)
    detected_format = _detect_subtitle_format(subtitle_file.filename or "", subtitle_format)

    segments = _parse_subtitle_content(content, detected_format)

    if not segments:
        raise HTTPException(status_code=400, detail="字幕文件为空或解析失败")

    # V3.2.4+dev.20260228.01: 确保 title 不为空，避免回退到 project_id (UUID)
    raw_filename = subtitle_file.filename or ""
    fallback_title = Path(raw_filename).stem.strip() or raw_filename.strip()
    resolved_title = (title or "").strip() or fallback_title or "导入字幕"

    project = project_service.create_import_project(
        title=resolved_title,
        subtitle_segments=segments,
        video_path=None,
        flavor=flavor,
    )
    project_dir = project_service.get_project_dir(project.project_id)
    if project_dir is None:
        raise HTTPException(status_code=500, detail="项目目录创建失败")

    if video_file is not None and video_file.filename:
        media_filename = _normalize_upload_filename(video_file.filename)
        _validate_import_media_extension(media_filename)
        target_path = project_dir / media_filename
        try:
            written_size = await _persist_uploaded_media_file(video_file, target_path)
        except HTTPException:
            target_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            target_path.unlink(missing_ok=True)
            logger.exception(
                "导入项目保存媒体文件失败: project_id=%s, file=%s",
                project.project_id,
                media_filename,
            )
            raise HTTPException(status_code=500, detail="媒体文件保存失败，请重试") from exc
        finally:
            await video_file.close()

        if written_size <= 0:
            target_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="媒体文件为空")

        project_service.refresh_media_assets(project.project_id)
        project = project_service.get_project(project.project_id) or project

    return {
        "success": True,
        "data": project.to_dict(),
        "redirect_url": f"/editor/project/{project.project_id}",
    }


@router.post("/import-local")
async def import_project_local(
    subtitle_filename: str = Form(...),
    media_filename: Optional[str] = Form(default=None),
    title: str = Form(default=""),
    flavor: str = Form(default=FLAVOR),
):
    """从 input 目录的本地文件导入项目（无需上传）。"""
    from app.core.config import config as app_config

    file_service = FileManagementService(
        str(app_config.INPUT_DIR), str(app_config.OUTPUT_DIR)
    )
    project_service = get_project_service()

    # 1. 读取字幕文件
    subtitle_path = file_service.get_input_file_path(subtitle_filename)
    if not Path(subtitle_path).exists():
        raise HTTPException(status_code=404, detail=f"字幕文件不存在: {subtitle_filename}")

    content = _read_local_file(subtitle_path)
    detected_format = _detect_subtitle_format(subtitle_filename, None)
    segments = _parse_subtitle_content(content, detected_format)

    if not segments:
        raise HTTPException(status_code=400, detail="字幕文件为空或解析失败")

    # 2. 可选：获取媒体文件路径
    media_path: Optional[str] = None
    if media_filename:
        _validate_import_media_extension(media_filename)
        media_path = file_service.get_input_file_path(media_filename)
        if not Path(media_path).exists():
            raise HTTPException(status_code=404, detail=f"媒体文件不存在: {media_filename}")

    # V3.2.4+dev.20260228.01: 确保 title 不为空，优先使用去扩展名的文件名
    local_fallback_title = Path(subtitle_filename).stem.strip() or subtitle_filename.strip()
    resolved_local_title = (title or "").strip() or local_fallback_title or "导入字幕"

    # 3. 创建项目（媒体文件通过硬链接关联到项目目录）
    project = project_service.create_import_project(
        title=resolved_local_title,
        subtitle_segments=segments,
        video_path=media_path,
        flavor=flavor,
    )

    return {
        "success": True,
        "data": project.to_dict(),
        "redirect_url": f"/editor/project/{project.project_id}",
    }


@router.get("")
async def list_projects(flavor: Optional[str] = None):
    """列出项目。"""
    project_service = get_project_service()
    projects = project_service.list_projects(flavor=flavor)
    return {"success": True, "data": [item.to_dict() for item in projects]}


@router.get("/{project_id}")
async def get_project(project_id: str):
    """获取项目详情。"""
    project_service = get_project_service()
    resolved_project_id = str(project_id or "").strip()
    try:
        identity = _resolve_project_identity(project_id)
        resolved_project_id = str(getattr(identity, "project_id", "") or resolved_project_id)
    except Exception:
        resolved_project_id = str(project_id or "").strip()

    project = (
        project_service.get_project(resolved_project_id)
        or project_service.find_project_by_alias(project_id)
    )
    if project is None:
        raise HTTPException(status_code=404, detail="项目不存在")
    return {"success": True, "data": project.to_dict()}


@router.patch("/{project_id}/title")
async def update_project_title(project_id: str, body: ProjectTitleUpdateRequest):
    """更新项目标题。"""
    project_service = get_project_service()
    resolved_project_id = str(project_id or "").strip()
    try:
        identity = _resolve_project_identity(project_id)
        resolved_project_id = str(getattr(identity, "project_id", "") or resolved_project_id)
    except Exception:
        resolved_project_id = str(project_id or "").strip()

    is_success = project_service.update_title(resolved_project_id, body.title)
    if not is_success:
        raise HTTPException(status_code=404, detail="项目不存在")
    project = project_service.get_project(resolved_project_id)
    return {"success": True, "data": project.to_dict() if project else None}


@router.get("/{project_id}/subtitle-time-offset")
async def get_project_subtitle_time_offset(project_id: str):
    """读取项目级字幕时间偏移（无任务级覆盖时回退全局）。"""
    project_service = get_project_service()
    global_offset = _get_global_subtitle_time_offset()
    resolved_project_id = str(project_id or "").strip()
    legacy_job_id: Optional[str] = None
    try:
        identity = _resolve_project_identity(project_id)
        resolved_project_id = str(getattr(identity, "project_id", "") or project_id)
        legacy_job_id = str(getattr(identity, "legacy_job_id", "") or "") or None
    except Exception as exc:
        logger.warning(
            "解析项目身份失败，降级按 project_id 读取字幕偏移: project_id=%s error=%s",
            project_id,
            exc,
        )
        alias_project = project_service.find_project_by_alias(project_id)
        if alias_project is not None:
            resolved_project_id = str(getattr(alias_project, "project_id", "") or resolved_project_id)
            legacy_job_id = str(getattr(alias_project, "job_id", "") or "") or None

    project_dir = project_service.get_project_dir(resolved_project_id)
    if project_dir is None:
        raise HTTPException(status_code=404, detail="项目不存在")

    transcription_service = _get_transcription_service()
    runtime_job = _find_runtime_job_for_project(
        transcription_service,
        resolved_project_id,
        legacy_job_id=legacy_job_id,
    )
    if runtime_job is None:
        return {"success": True, "offset": global_offset, "source": "global"}

    task_offset = getattr(runtime_job, "subtitle_time_offset", None)
    if task_offset is None:
        return {"success": True, "offset": global_offset, "source": "global"}
    return {"success": True, "offset": float(task_offset), "source": "project"}


@router.post("/{project_id}/subtitle-time-offset")
async def set_project_subtitle_time_offset(
    project_id: str,
    body: ProjectSubtitleTimeOffsetRequest,
):
    """设置项目级字幕时间偏移（等于全局时不保存任务覆盖值）。"""
    project_service = get_project_service()
    global_offset = _get_global_subtitle_time_offset()
    normalized = round(float(body.offset), 3)
    global_normalized = round(float(global_offset), 3)

    resolved_project_id = str(project_id or "").strip()
    legacy_job_id: Optional[str] = None
    try:
        identity = _resolve_project_identity(project_id)
        resolved_project_id = str(getattr(identity, "project_id", "") or project_id)
        legacy_job_id = str(getattr(identity, "legacy_job_id", "") or "") or None
    except Exception as exc:
        logger.warning(
            "解析项目身份失败，降级按 project_id 写入字幕偏移: project_id=%s error=%s",
            project_id,
            exc,
        )
        alias_project = project_service.find_project_by_alias(project_id)
        if alias_project is not None:
            resolved_project_id = str(getattr(alias_project, "project_id", "") or resolved_project_id)
            legacy_job_id = str(getattr(alias_project, "job_id", "") or "") or None

    project_dir = project_service.get_project_dir(resolved_project_id)
    if project_dir is None:
        raise HTTPException(status_code=404, detail="项目不存在")

    transcription_service = _get_transcription_service()
    runtime_job = _find_runtime_job_for_project(
        transcription_service,
        resolved_project_id,
        legacy_job_id=legacy_job_id,
    )
    if runtime_job is None:
        logger.warning(
            "项目无运行态任务，字幕偏移回退全局: project_id=%s requested_offset=%s",
            resolved_project_id,
            normalized,
        )
        return {"success": True, "offset": global_normalized, "source": "global"}

    if normalized == global_normalized:
        runtime_job.subtitle_time_offset = None
        source = "global"
        effective_offset = global_normalized
    else:
        runtime_job.subtitle_time_offset = normalized
        source = "project"
        effective_offset = normalized

    if str(getattr(runtime_job, "project_id", "") or "").strip() != resolved_project_id:
        runtime_job.project_id = resolved_project_id

    if not transcription_service.save_job_meta(runtime_job):
        raise HTTPException(status_code=500, detail="项目字幕时间偏移保存失败")

    return {"success": True, "offset": effective_offset, "source": source}


@router.get("/{project_id}/subtitles")
async def list_project_subtitles(project_id: str):
    """获取项目字幕。"""
    project_service = get_project_service()
    subtitle_doc_service = get_subtitle_doc_service()
    project_dir = project_service.get_project_dir(project_id)
    if project_dir is None:
        raise HTTPException(status_code=404, detail="项目不存在")
    runtime_segments = _load_runtime_subtitle_segments(project_dir)
    if runtime_segments:
        segments = _compose_runtime_segments_with_user_edits(project_dir, runtime_segments)
        return {"success": True, "data": segments}

    # 纯导入/仅编辑项目：沿用 subtitle_doc 读取。
    segments = subtitle_doc_service.load_segments(project_dir)
    return {"success": True, "data": segments}


@router.post("/{project_id}/subtitles")
async def create_project_subtitle(project_id: str, body: SubtitleCreateRequest):
    """新增项目字幕。"""
    if body.end < body.start:
        raise HTTPException(status_code=400, detail="结束时间必须大于等于开始时间")

    project_service = get_project_service()
    subtitle_doc_service = get_subtitle_doc_service()
    project_dir = project_service.get_project_dir(project_id)
    if project_dir is None:
        raise HTTPException(status_code=404, detail="项目不存在")

    runtime_segments = _load_runtime_subtitle_segments(project_dir)
    if runtime_segments:
        legacy_index, entry = create_manual_entry(
            project_dir,
            body.text or "",
            float(body.start),
            float(body.end),
        )
        segment = {
            "segment_id": _manual_segment_id(legacy_index),
            "sentence_uid": None,
            "chunk_uid": "chunk:manual",
            "text": str(entry.get("text", body.text or "")),
            "start": _safe_float(entry.get("start"), float(body.start)),
            "end": _safe_float(entry.get("end"), float(body.end)),
            "legacy_index": int(legacy_index),
            "source_type": "manual",
            "is_modified": True,
            "original_text": entry.get("original_text"),
        }
        _sync_homophone_index_for_project_change(
            project_id=project_id,
            project_dir=project_dir,
            updated_sentence_indices=[int(legacy_index)],
        )
        _publish_project_subtitle_event(
            project_id,
            "added",
            {"segment": segment, "source": "project_api", "is_update": True},
        )
        return {"success": True, "data": segment}

    segment = subtitle_doc_service.create_segment(
        project_dir=project_dir,
        text=body.text,
        start=body.start,
        end=body.end,
    )
    _sync_homophone_index_for_project_change(
        project_id=project_id,
        project_dir=project_dir,
        updated_sentence_indices=[int(segment.get("legacy_index", 0))],
    )
    _publish_project_subtitle_event(
        project_id,
        "added",
        {"segment": segment, "source": "project_api", "is_update": True},
    )
    return {"success": True, "data": segment}


@router.patch("/{project_id}/subtitles/{segment_id}")
async def update_project_subtitle(
    project_id: str,
    segment_id: str,
    body: SubtitleUpdateRequest,
):
    """更新项目字幕。"""
    update_payload = body.model_dump(exclude_none=True)
    if not update_payload:
        raise HTTPException(status_code=400, detail="更新内容不能为空")
    if (
        update_payload.get("start") is not None
        and update_payload.get("end") is not None
        and float(update_payload["end"]) < float(update_payload["start"])
    ):
        raise HTTPException(status_code=400, detail="结束时间必须大于等于开始时间")

    project_service = get_project_service()
    subtitle_doc_service = get_subtitle_doc_service()
    project_dir = project_service.get_project_dir(project_id)
    if project_dir is None:
        raise HTTPException(status_code=404, detail="项目不存在")

    runtime_segments = _load_runtime_subtitle_segments(project_dir)
    if runtime_segments:
        composed_before = _compose_runtime_segments_with_user_edits(project_dir, runtime_segments)
        current_segment = _find_segment_by_segment_id(composed_before, segment_id)
        if current_segment is None:
            raise HTTPException(status_code=404, detail="字幕段不存在")

        target_index = int(current_segment.get("legacy_index", 0))
        save_edit(
            project_dir,
            target_index,
            update_payload,
            original_text=str(current_segment.get("text", "") or ""),
        )

        composed_after = _compose_runtime_segments_with_user_edits(project_dir, runtime_segments)
        segment = _find_segment_by_segment_id(composed_after, segment_id)
        if segment is None:
            raise HTTPException(status_code=404, detail="字幕段不存在")
        _sync_homophone_index_for_project_change(
            project_id=project_id,
            project_dir=project_dir,
            updated_sentence_indices=[target_index],
        )
        _publish_project_subtitle_event(
            project_id,
            "edited",
            {
                "segment_id": segment_id,
                "segment": segment,
                "source": "project_api",
                "is_update": True,
            },
        )
        return {"success": True, "data": segment}

    is_success = subtitle_doc_service.update_segment(project_dir, segment_id, update_payload)
    if not is_success:
        raise HTTPException(status_code=404, detail="字幕段不存在")

    segment = subtitle_doc_service.get_segment(project_dir, segment_id)
    _sync_homophone_index_for_project_change(
        project_id=project_id,
        project_dir=project_dir,
        updated_sentence_indices=[int(segment.get("legacy_index", 0))] if segment else [],
    )
    _publish_project_subtitle_event(
        project_id,
        "edited",
        {
            "segment_id": segment_id,
            "segment": segment,
            "source": "project_api",
            "is_update": True,
        },
    )
    return {"success": True, "data": segment}


@router.delete("/{project_id}/subtitles/{segment_id}")
async def delete_project_subtitle(project_id: str, segment_id: str):
    """删除项目字幕。"""
    project_service = get_project_service()
    subtitle_doc_service = get_subtitle_doc_service()
    project_dir = project_service.get_project_dir(project_id)
    if project_dir is None:
        raise HTTPException(status_code=404, detail="项目不存在")

    runtime_segments = _load_runtime_subtitle_segments(project_dir)
    if runtime_segments:
        composed_before = _compose_runtime_segments_with_user_edits(project_dir, runtime_segments)
        current_segment = _find_segment_by_segment_id(composed_before, segment_id)
        if current_segment is None:
            raise HTTPException(status_code=404, detail="字幕段不存在")
        target_index = int(current_segment.get("legacy_index", 0))
        add_deletion(project_dir, target_index)
        _sync_homophone_index_for_project_change(
            project_id=project_id,
            project_dir=project_dir,
            removed_sentence_indices=[target_index],
        )
        _publish_project_subtitle_event(
            project_id,
            "deleted",
            {"segment_id": segment_id, "source": "project_api", "is_update": True},
        )
        return {"success": True, "data": {"segment_id": segment_id, "is_deleted": True}}

    segment = subtitle_doc_service.get_segment(project_dir, segment_id)
    is_success = subtitle_doc_service.delete_segment(project_dir, segment_id)
    if not is_success:
        raise HTTPException(status_code=404, detail="字幕段不存在")

    _sync_homophone_index_for_project_change(
        project_id=project_id,
        project_dir=project_dir,
        removed_sentence_indices=[int(segment.get("legacy_index", 0))] if segment else [],
    )
    _publish_project_subtitle_event(
        project_id,
        "deleted",
        {"segment_id": segment_id, "source": "project_api", "is_update": True},
    )
    return {"success": True, "data": {"segment_id": segment_id, "is_deleted": True}}


@router.post("/{project_id}/editor-ops:apply")
async def apply_editor_ops(project_id: str, body: EditorOpsApplyRequest):
    """统一批量编辑命令入口。"""
    project_service = get_project_service()
    project_dir = project_service.get_project_dir(project_id)
    if project_dir is None:
        return _editor_ops_error(
            status_code=404,
            message="项目不存在",
            error_code="PROJECT_NOT_FOUND",
        )

    if not body.ops:
        return _editor_ops_error(
            status_code=400,
            message="ops 不能为空",
            error_code="INVALID_REQUEST",
        )

    state = _load_editor_ops_state(project_dir)
    sessions = state.setdefault("sessions", {})
    session_ops = sessions.setdefault(body.session_id, {})
    current_revision = int(state.get("server_revision", 0) or 0)

    request_bindings: dict[str, str] = {}
    created_bindings: list[dict[str, str]] = []
    normalized_results: list[dict[str, Any]] = []
    updated_entities: list[dict[str, Any]] = []
    new_ops: list[dict[str, Any]] = []
    seen_op_ids: set[str] = set()

    for op in body.ops:
        if not isinstance(op, dict):
            return _editor_ops_error(
                status_code=400,
                message="ops 项必须是对象",
                server_revision=current_revision,
                error_code="INVALID_REQUEST",
            )
        op_id = _normalize_editor_segment_id(op.get("op_id"))
        if not op_id:
            return _editor_ops_error(
                status_code=400,
                message="存在缺少 op_id 的命令",
                server_revision=current_revision,
                error_code="INVALID_REQUEST",
            )
        if op_id in seen_op_ids:
            return _editor_ops_error(
                status_code=400,
                message=f"请求内存在重复 op_id: {op_id}",
                server_revision=current_revision,
                error_code="INVALID_REQUEST",
                failed_op_id=op_id,
            )
        seen_op_ids.add(op_id)
        existing_entry = session_ops.get(op_id)
        if isinstance(existing_entry, dict):
            _aggregate_editor_op_result(
                result_entry=existing_entry,
                request_bindings=request_bindings,
                created_bindings=created_bindings,
                normalized_results=normalized_results,
                updated_entities=updated_entities,
            )
        else:
            new_ops.append(op)

    if new_ops and body.base_revision is not None and body.base_revision != current_revision:
        return _editor_ops_error(
            status_code=409,
            message="客户端基准版本过旧",
            server_revision=current_revision,
            error_code="REVISION_CONFLICT",
            failed_op_id=_normalize_editor_segment_id(new_ops[0].get("op_id")),
            details={"reason": "base_revision_outdated"},
        )

    if not new_ops:
        return {
            "success": True,
            "server_revision": current_revision,
            "created_bindings": created_bindings,
            "normalized_results": normalized_results,
            "updated_entities": updated_entities,
        }

    backup_path = _backup_subtitle_edit_store(project_dir)
    staged_entries: dict[str, dict[str, Any]] = {}
    suppress_token = _SUPPRESS_PROJECT_SUBTITLE_EVENTS.set(True)

    try:
        for op in body.ops:
            op_id = _normalize_editor_segment_id(op.get("op_id"))
            if op_id in session_ops:
                continue
            result_entry = await _apply_editor_op(
                project_id=project_id,
                project_dir=project_dir,
                op=op,
                request_bindings=request_bindings,
            )
            staged_entries[op_id] = result_entry
            _aggregate_editor_op_result(
                result_entry=result_entry,
                request_bindings=request_bindings,
                created_bindings=created_bindings,
                normalized_results=normalized_results,
                updated_entities=updated_entities,
            )
    except HTTPException as exc:
        _restore_subtitle_edit_store(project_dir, backup_path)
        detail = exc.detail if isinstance(exc.detail, str) else "editor-ops 执行失败"
        return _editor_ops_error(
            status_code=exc.status_code,
            message=str(detail),
            server_revision=current_revision,
            failed_op_id=_normalize_editor_segment_id(op.get("op_id")) if isinstance(op, dict) else None,
        )
    except Exception as exc:
        _restore_subtitle_edit_store(project_dir, backup_path)
        logger.exception("editor-ops 执行异常: project_id=%s", project_id)
        return _editor_ops_error(
            status_code=500,
            message=f"editor-ops 执行失败: {exc}",
            server_revision=current_revision,
            failed_op_id=_normalize_editor_segment_id(op.get("op_id")) if isinstance(op, dict) else None,
        )
    finally:
        _SUPPRESS_PROJECT_SUBTITLE_EVENTS.reset(suppress_token)
        _cleanup_subtitle_edit_store_backup(backup_path)

    next_revision = current_revision + 1
    for entry in staged_entries.values():
        entry["applied_revision"] = next_revision
        for entity in entry.get("updated_entities", []):
            if isinstance(entity, dict):
                entity["revision"] = next_revision

    for entity in updated_entities:
        entity["revision"] = next_revision

    session_ops.update(staged_entries)
    state["server_revision"] = next_revision
    _save_editor_ops_state(project_dir, state)

    for op in body.ops:
        op_id = _normalize_editor_segment_id(op.get("op_id"))
        entry = staged_entries.get(op_id)
        if not isinstance(entry, dict):
            continue
        for event in entry.get("events", []):
            if not isinstance(event, dict):
                continue
            event_type = _normalize_editor_segment_id(event.get("event_type"))
            data = event.get("data")
            if event_type and isinstance(data, dict):
                _publish_project_subtitle_event(project_id, event_type, data)

    return {
        "success": True,
        "server_revision": next_revision,
        "created_bindings": created_bindings,
        "normalized_results": normalized_results,
        "updated_entities": updated_entities,
    }


# V3.2.4+dev.20260303.01: undo/redo 批量同步端点


@router.post("/{project_id}/subtitles/batch-sync")
async def batch_sync_project_subtitles(project_id: str, body: BatchSyncRequest):
    """undo/redo 后端增量同步（支持 runtime 与 subtitle_doc 双路径）。"""
    project_service = get_project_service()
    subtitle_doc_service = get_subtitle_doc_service()
    project_dir = project_service.get_project_dir(project_id)
    if project_dir is None:
        raise HTTPException(status_code=404, detail="项目不存在")

    runtime_segments = _load_runtime_subtitle_segments(project_dir)
    is_runtime = bool(runtime_segments)
    results: dict[str, Any] = {
        "updated": 0,
        "created": 0,
        "deleted": 0,
        "created_segments": [],
        "errors": [],
    }

    # 前置校验 end >= start
    for item in body.updates:
        if (
            item.start is not None
            and item.end is not None
            and item.end < item.start
        ):
            results["errors"].append(
                f"update: {item.segment_id} end({item.end}) < start({item.start})"
            )
    for item in body.creates:
        if item.end < item.start:
            results["errors"].append(
                f"create: end({item.end}) < start({item.start})"
            )
    if results["errors"]:
        return {"success": False, "data": results}

    if is_runtime:
        _batch_sync_runtime(project_id, project_dir, body, runtime_segments, results)
    else:
        _batch_sync_subtitle_doc(
            project_id, project_dir, body, subtitle_doc_service, results
        )

    return {
        "success": len(results["errors"]) == 0,
        "data": results,
    }


def _batch_sync_runtime(
    project_id: str,
    project_dir: Path,
    body: BatchSyncRequest,
    runtime_segments: list[dict],
    results: dict[str, Any],
) -> None:
    """Runtime 路径批量同步。"""
    composed = _compose_runtime_segments_with_user_edits(project_dir, runtime_segments)
    updated_indices: set[int] = set()
    removed_indices: set[int] = set()

    # 1. updates
    for item in body.updates:
        seg = _find_segment_by_segment_id(composed, item.segment_id)
        if not seg:
            results["errors"].append(f"update: {item.segment_id} 不存在")
            continue
        payload: dict[str, Any] = {}
        if item.text is not None:
            payload["text"] = item.text
        if item.start is not None:
            payload["start"] = item.start
        if item.end is not None:
            payload["end"] = item.end
        if payload:
            save_edit(
                project_dir,
                int(seg["legacy_index"]),
                payload,
                original_text=str(seg.get("text", "")),
            )
            updated_indices.add(int(seg["legacy_index"]))
            results["updated"] += 1
            composed_after = _compose_runtime_segments_with_user_edits(
                project_dir, runtime_segments
            )
            updated_segment = _find_segment_by_segment_id(
                composed_after, item.segment_id
            )
            if updated_segment is not None:
                _publish_project_subtitle_event(
                    project_id,
                    "edited",
                    {
                        "segment_id": item.segment_id,
                        "segment": updated_segment,
                        "source": "project_api",
                        "is_update": True,
                    },
                )

    # 2. deletes
    for item in body.deletes:
        seg = _find_segment_by_segment_id(composed, item.segment_id)
        if not seg:
            results["errors"].append(f"delete: {item.segment_id} 不存在")
            continue
        idx = int(seg["legacy_index"])
        if idx < 0:
            remove_manual_entry(project_dir, idx)
        else:
            add_deletion(project_dir, idx)
        removed_indices.add(idx)
        results["deleted"] += 1
        _publish_project_subtitle_event(
            project_id,
            "deleted",
            {
                "segment_id": item.segment_id,
                "source": "project_api",
                "is_update": True,
            },
        )

    # 3. creates
    for item in body.creates:
        if item.restore_segment_id:
            restored_index = _handle_runtime_restore(
                project_id, project_dir, item, runtime_segments, results
            )
            if restored_index is not None:
                updated_indices.add(int(restored_index))
        else:
            new_index, entry = create_manual_entry(
                project_dir, item.text or "", float(item.start), float(item.end)
            )
            segment_id = _manual_segment_id(new_index)
            segment = {
                "segment_id": segment_id,
                "sentence_uid": None,
                "chunk_uid": "chunk:manual",
                "text": str(entry.get("text", item.text or "")),
                "start": float(entry.get("start", float(item.start))),
                "end": float(entry.get("end", float(item.end))),
                "legacy_index": int(new_index),
                "source_type": "manual",
                "is_modified": True,
                "original_text": entry.get("original_text"),
            }
            results["created_segments"].append({
                "segment_id": segment_id,
                "legacy_index": new_index,
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"],
            })
            updated_indices.add(int(new_index))
            results["created"] += 1
            _publish_project_subtitle_event(
                project_id,
                "added",
                {"segment": segment, "source": "project_api", "is_update": True},
            )

    _sync_homophone_index_for_project_change(
        project_id=project_id,
        project_dir=project_dir,
        updated_sentence_indices=sorted(updated_indices),
        removed_sentence_indices=sorted(removed_indices),
    )


def _handle_runtime_restore(
    project_id: str,
    project_dir: Path,
    item: BatchSyncCreateItem,
    runtime_segments: list[dict],
    results: dict[str, Any],
) -> int | None:
    """Runtime 路径恢复已删除字幕。"""
    seg_id = item.restore_segment_id
    if not seg_id:
        return None

    # 区分正索引（runtime 基线）与负索引（手动新增）
    if seg_id.startswith("manual-"):
        try:
            original_index = -int(seg_id.split("-", 1)[1])
        except (ValueError, IndexError):
            results["errors"].append(f"restore: {seg_id} 索引解析失败")
            return None
        ok = restore_manual_entry(
            project_dir, original_index, item.text or "", float(item.start), float(item.end)
        )
        if ok:
            segment = {
                "segment_id": seg_id,
                "sentence_uid": None,
                "chunk_uid": "chunk:manual",
                "text": str(item.text or ""),
                "start": float(item.start),
                "end": float(item.end),
                "legacy_index": int(original_index),
                "source_type": "manual",
                "is_modified": True,
                "original_text": None,
            }
            results["created_segments"].append({
                "segment_id": seg_id,
                "legacy_index": original_index,
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"],
            })
            results["created"] += 1
            _publish_project_subtitle_event(
                project_id,
                "added",
                {"segment": segment, "source": "project_api", "is_update": True},
            )
            return int(original_index)
        else:
            results["errors"].append(f"restore: {seg_id} 手动字幕恢复失败")
            return None
    else:
        target = _find_segment_by_segment_id(runtime_segments, seg_id)
        if not target:
            results["errors"].append(f"restore: {seg_id} 未找到")
            return None
        idx = int(target["legacy_index"])
        remove_deletion(project_dir, idx)
        # 如果内容有变，补存编辑
        delta: dict[str, Any] = {}
        if item.text and item.text != str(target.get("text", "")):
            delta["text"] = item.text
        if item.start is not None and abs(item.start - float(target.get("start", 0))) > 0.001:
            delta["start"] = item.start
        if item.end is not None and abs(item.end - float(target.get("end", 0))) > 0.001:
            delta["end"] = item.end
        if delta:
            save_edit(project_dir, idx, delta)
        composed_after = _compose_runtime_segments_with_user_edits(
            project_dir, runtime_segments
        )
        restored_segment = _find_segment_by_segment_id(composed_after, seg_id)
        restored_text = (
            str(restored_segment.get("text", ""))
            if restored_segment is not None
            else str(item.text or target.get("text", ""))
        )
        restored_start = (
            float(restored_segment.get("start", 0.0))
            if restored_segment is not None
            else float(item.start if item.start is not None else target.get("start", 0.0))
        )
        restored_end = (
            float(restored_segment.get("end", restored_start))
            if restored_segment is not None
            else float(item.end if item.end is not None else target.get("end", restored_start))
        )
        results["created_segments"].append({
            "segment_id": seg_id,
            "legacy_index": idx,
            "text": restored_text,
            "start": restored_start,
            "end": restored_end,
        })
        results["created"] += 1
        _publish_project_subtitle_event(
            project_id,
            "added",
            {
                "segment": restored_segment
                if restored_segment is not None
                else {
                    "segment_id": seg_id,
                    "legacy_index": idx,
                    "text": restored_text,
                    "start": restored_start,
                    "end": restored_end,
                    "source_type": "restored",
                    "is_modified": True,
                },
                "source": "project_api",
                "is_update": True,
            },
        )
        return idx
    return None


def _batch_sync_subtitle_doc(
    project_id: str,
    project_dir: Path,
    body: BatchSyncRequest,
    subtitle_doc_service: Any,
    results: dict[str, Any],
) -> None:
    """Subtitle Doc 路径批量同步。"""
    updated_indices: set[int] = set()
    removed_indices: set[int] = set()
    # 1. updates
    for item in body.updates:
        payload = {
            k: v
            for k, v in {"text": item.text, "start": item.start, "end": item.end}.items()
            if v is not None
        }
        ok = subtitle_doc_service.update_segment(project_dir, item.segment_id, payload)
        if ok:
            results["updated"] += 1
            segment = subtitle_doc_service.get_segment(project_dir, item.segment_id)
            if segment is not None:
                updated_indices.add(int(segment.get("legacy_index", 0)))
                _publish_project_subtitle_event(
                    project_id,
                    "edited",
                    {
                        "segment_id": item.segment_id,
                        "segment": segment,
                        "source": "project_api",
                        "is_update": True,
                    },
                )
        else:
            results["errors"].append(f"update: {item.segment_id} 不存在")

    # 2. deletes
    for item in body.deletes:
        segment = subtitle_doc_service.get_segment(project_dir, item.segment_id)
        ok = subtitle_doc_service.delete_segment(project_dir, item.segment_id)
        if ok:
            results["deleted"] += 1
            if segment is not None:
                removed_indices.add(int(segment.get("legacy_index", 0)))
            _publish_project_subtitle_event(
                project_id,
                "deleted",
                {
                    "segment_id": item.segment_id,
                    "source": "project_api",
                    "is_update": True,
                },
            )
        else:
            results["errors"].append(f"delete: {item.segment_id} 不存在")

    # 3. creates
    for item in body.creates:
        if item.restore_segment_id:
            restored_index = _handle_subtitle_doc_restore(
                project_id, project_dir, item, subtitle_doc_service, results
            )
            if restored_index is not None:
                updated_indices.add(int(restored_index))
        else:
            new_seg = subtitle_doc_service.create_segment(
                project_dir, item.text or "", float(item.start), float(item.end)
            )
            results["created_segments"].append({
                "segment_id": new_seg.get("segment_id"),
                "legacy_index": new_seg.get("legacy_index"),
                "text": new_seg.get("text"),
                "start": new_seg.get("start"),
                "end": new_seg.get("end"),
            })
            updated_indices.add(int(new_seg.get("legacy_index", 0)))
            results["created"] += 1
            _publish_project_subtitle_event(
                project_id,
                "added",
                {"segment": new_seg, "source": "project_api", "is_update": True},
            )

    _sync_homophone_index_for_project_change(
        project_id=project_id,
        project_dir=project_dir,
        updated_sentence_indices=sorted(updated_indices),
        removed_sentence_indices=sorted(removed_indices),
    )


def _handle_subtitle_doc_restore(
    project_id: str,
    project_dir: Path,
    item: BatchSyncCreateItem,
    subtitle_doc_service: Any,
    results: dict[str, Any],
) -> int | None:
    """Subtitle Doc 路径恢复已删除字幕（通过 tombstone 映射）。"""
    seg_id = item.restore_segment_id
    if not seg_id:
        return None

    if seg_id.startswith("manual-"):
        try:
            original_index = -int(seg_id.split("-", 1)[1])
        except (ValueError, IndexError):
            results["errors"].append(f"restore: {seg_id} 索引解析失败")
            return None
        ok = restore_manual_entry(
            project_dir, original_index, item.text or "", float(item.start), float(item.end)
        )
        if ok:
            segment = {
                "segment_id": seg_id,
                "legacy_index": int(original_index),
                "text": str(item.text or ""),
                "start": float(item.start),
                "end": float(item.end),
                "source_type": "manual",
                "is_modified": True,
                "is_deleted": False,
            }
            results["created_segments"].append({
                "segment_id": seg_id,
                "legacy_index": original_index,
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"],
            })
            results["created"] += 1
            _publish_project_subtitle_event(
                project_id,
                "added",
                {"segment": segment, "source": "project_api", "is_update": True},
            )
            return int(original_index)
        else:
            results["errors"].append(f"restore: {seg_id} 手动字幕恢复失败")
            return None
    else:
        # 通过 tombstone 恢复：restore_segment 查询 _deleted_segment_map
        update: dict[str, Any] = {}
        if item.text:
            update["text"] = item.text
        if item.start is not None:
            update["start"] = item.start
        if item.end is not None:
            update["end"] = item.end
        restored_index = subtitle_doc_service.restore_segment(
            project_dir, seg_id, update or None
        )
        if restored_index is not None:
            restored_segment = subtitle_doc_service.get_segment(project_dir, seg_id)
            restored_text = (
                str(restored_segment.get("text", ""))
                if restored_segment is not None
                else str(item.text or "")
            )
            restored_start = (
                float(restored_segment.get("start", 0.0))
                if restored_segment is not None
                else float(item.start if item.start is not None else 0.0)
            )
            restored_end = (
                float(restored_segment.get("end", restored_start))
                if restored_segment is not None
                else float(item.end if item.end is not None else restored_start)
            )
            results["created_segments"].append({
                "segment_id": seg_id,
                "legacy_index": restored_index,
                "text": restored_text,
                "start": restored_start,
                "end": restored_end,
            })
            results["created"] += 1
            _publish_project_subtitle_event(
                project_id,
                "added",
                {
                    "segment": restored_segment
                    if restored_segment is not None
                    else {
                        "segment_id": seg_id,
                        "legacy_index": restored_index,
                        "text": restored_text,
                        "start": restored_start,
                        "end": restored_end,
                        "source_type": "restored",
                        "is_modified": True,
                    },
                    "source": "project_api",
                    "is_update": True,
                },
            )
            return int(restored_index)
        else:
            results["errors"].append(f"restore: {seg_id} tombstone 中未找到")
            return None
    return None


@router.patch("/{project_id}/subtitles/legacy/{sentence_index}")
async def update_project_subtitle_by_legacy_index(
    project_id: str,
    sentence_index: int,
    body: SubtitleUpdateRequest,
):
    """兼容入口：按 legacy_index 更新项目字幕。"""
    project_service = get_project_service()
    subtitle_doc_service = get_subtitle_doc_service()
    project_dir = project_service.get_project_dir(project_id)
    if project_dir is None:
        raise HTTPException(status_code=404, detail="项目不存在")

    runtime_segments = _load_runtime_subtitle_segments(project_dir)
    if runtime_segments:
        composed_segments = _compose_runtime_segments_with_user_edits(project_dir, runtime_segments)
        target_segment = _find_segment_by_legacy_index(composed_segments, sentence_index)
        if target_segment is None:
            raise HTTPException(status_code=404, detail="字幕段不存在")
        target_segment_id = str(target_segment.get("segment_id", "") or "").strip()
        if not target_segment_id:
            raise HTTPException(status_code=404, detail="字幕段不存在")
        return await update_project_subtitle(project_id, target_segment_id, body)

    segments = subtitle_doc_service.load_segments(project_dir)
    target_segment = _find_segment_by_legacy_index(segments, sentence_index)
    if target_segment is None:
        raise HTTPException(status_code=404, detail="字幕段不存在")
    target_segment_id = str(target_segment.get("segment_id", "") or "").strip()
    if not target_segment_id:
        raise HTTPException(status_code=404, detail="字幕段不存在")
    return await update_project_subtitle(project_id, target_segment_id, body)


@router.delete("/{project_id}/subtitles/legacy/{sentence_index}")
async def delete_project_subtitle_by_legacy_index(project_id: str, sentence_index: int):
    """兼容入口：按 legacy_index 删除项目字幕。"""
    project_service = get_project_service()
    subtitle_doc_service = get_subtitle_doc_service()
    project_dir = project_service.get_project_dir(project_id)
    if project_dir is None:
        raise HTTPException(status_code=404, detail="项目不存在")

    runtime_segments = _load_runtime_subtitle_segments(project_dir)
    if runtime_segments:
        composed_segments = _compose_runtime_segments_with_user_edits(project_dir, runtime_segments)
        target_segment = _find_segment_by_legacy_index(composed_segments, sentence_index)
        if target_segment is None:
            raise HTTPException(status_code=404, detail="字幕段不存在")
        target_segment_id = str(target_segment.get("segment_id", "") or "").strip()
        if not target_segment_id:
            raise HTTPException(status_code=404, detail="字幕段不存在")
        return await delete_project_subtitle(project_id, target_segment_id)

    segments = subtitle_doc_service.load_segments(project_dir)
    target_segment = _find_segment_by_legacy_index(segments, sentence_index)
    if target_segment is None:
        raise HTTPException(status_code=404, detail="字幕段不存在")
    target_segment_id = str(target_segment.get("segment_id", "") or "").strip()
    if not target_segment_id:
        raise HTTPException(status_code=404, detail="字幕段不存在")
    return await delete_project_subtitle(project_id, target_segment_id)


@router.get("/{project_id}/export")
async def export_project_subtitles(project_id: str, format: Literal["srt", "ass"] = "srt"):
    """导出项目字幕文本。"""
    project_service = get_project_service()
    subtitle_doc_service = get_subtitle_doc_service()
    project_dir = project_service.get_project_dir(project_id)
    if project_dir is None:
        raise HTTPException(status_code=404, detail="项目不存在")

    runtime_segments = _build_runtime_export_segments(project_dir)
    if runtime_segments is not None:
        if format == "srt":
            content = _export_segments_as_srt(runtime_segments)
        else:
            content = _export_segments_as_ass(project_dir, runtime_segments)
    else:
        if format == "srt":
            content = subtitle_doc_service.export_srt(project_dir)
        else:
            content = subtitle_doc_service.export_ass(project_dir)
    return {"success": True, "data": {"format": format, "content": content}}
