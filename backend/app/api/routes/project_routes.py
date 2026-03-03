"""
Project API 路由。
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import FLAVOR
from app.services.file_service import FileManagementService
from app.services.project_service import get_project_service
from app.services.subtitle_doc_service import get_subtitle_doc_service
from app.services.subtitle_edit_store import (
    add_deletion,
    create_manual_entry,
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


class ProjectSubtitleTimeOffsetRequest(BaseModel):
    """项目字幕时间偏移请求。"""

    offset: float = Field(
        ...,
        ge=-10.0,
        le=10.0,
        description="偏移量（秒），正值延后，负值提前，范围 -10.0 到 10.0",
    )


def _publish_project_subtitle_event(project_id: str, event_type: str, data: dict) -> None:
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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
        if not text.strip():
            continue
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
        video_bytes = await video_file.read()
        target_path = project_dir / video_file.filename
        target_path.write_bytes(video_bytes)
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
        _publish_project_subtitle_event(
            project_id,
            "deleted",
            {"segment_id": segment_id, "source": "project_api", "is_update": True},
        )
        return {"success": True, "data": {"segment_id": segment_id, "is_deleted": True}}

    is_success = subtitle_doc_service.delete_segment(project_dir, segment_id)
    if not is_success:
        raise HTTPException(status_code=404, detail="字幕段不存在")

    _publish_project_subtitle_event(
        project_id,
        "deleted",
        {"segment_id": segment_id, "source": "project_api", "is_update": True},
    )
    return {"success": True, "data": {"segment_id": segment_id, "is_deleted": True}}


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
            _handle_runtime_restore(
                project_id, project_dir, item, runtime_segments, results
            )
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
            results["created"] += 1
            _publish_project_subtitle_event(
                project_id,
                "added",
                {"segment": segment, "source": "project_api", "is_update": True},
            )


def _handle_runtime_restore(
    project_id: str,
    project_dir: Path,
    item: BatchSyncCreateItem,
    runtime_segments: list[dict],
    results: dict[str, Any],
) -> None:
    """Runtime 路径恢复已删除字幕。"""
    seg_id = item.restore_segment_id
    if not seg_id:
        return

    # 区分正索引（runtime 基线）与负索引（手动新增）
    if seg_id.startswith("manual-"):
        try:
            original_index = -int(seg_id.split("-", 1)[1])
        except (ValueError, IndexError):
            results["errors"].append(f"restore: {seg_id} 索引解析失败")
            return
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
        else:
            results["errors"].append(f"restore: {seg_id} 手动字幕恢复失败")
    else:
        target = _find_segment_by_segment_id(runtime_segments, seg_id)
        if not target:
            results["errors"].append(f"restore: {seg_id} 未找到")
            return
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


def _batch_sync_subtitle_doc(
    project_id: str,
    project_dir: Path,
    body: BatchSyncRequest,
    subtitle_doc_service: Any,
    results: dict[str, Any],
) -> None:
    """Subtitle Doc 路径批量同步。"""
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
        ok = subtitle_doc_service.delete_segment(project_dir, item.segment_id)
        if ok:
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
        else:
            results["errors"].append(f"delete: {item.segment_id} 不存在")

    # 3. creates
    for item in body.creates:
        if item.restore_segment_id:
            _handle_subtitle_doc_restore(
                project_id, project_dir, item, subtitle_doc_service, results
            )
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
            results["created"] += 1
            _publish_project_subtitle_event(
                project_id,
                "added",
                {"segment": new_seg, "source": "project_api", "is_update": True},
            )


def _handle_subtitle_doc_restore(
    project_id: str,
    project_dir: Path,
    item: BatchSyncCreateItem,
    subtitle_doc_service: Any,
    results: dict[str, Any],
) -> None:
    """Subtitle Doc 路径恢复已删除字幕（通过 tombstone 映射）。"""
    seg_id = item.restore_segment_id
    if not seg_id:
        return

    if seg_id.startswith("manual-"):
        try:
            original_index = -int(seg_id.split("-", 1)[1])
        except (ValueError, IndexError):
            results["errors"].append(f"restore: {seg_id} 索引解析失败")
            return
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
        else:
            results["errors"].append(f"restore: {seg_id} 手动字幕恢复失败")
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
        else:
            results["errors"].append(f"restore: {seg_id} tombstone 中未找到")


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
