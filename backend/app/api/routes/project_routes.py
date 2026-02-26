"""
Project API 路由。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import FLAVOR
from app.services.file_service import FileManagementService
from app.services.project_service import get_project_service
from app.services.subtitle_doc_service import get_subtitle_doc_service
from app.services.sse_service import get_sse_manager

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

    project = project_service.create_import_project(
        title=title or (subtitle_file.filename or ""),
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

    # 3. 创建项目（媒体文件通过硬链接关联到项目目录）
    project = project_service.create_import_project(
        title=title or subtitle_filename,
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
    project = project_service.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="项目不存在")
    return {"success": True, "data": project.to_dict()}


@router.patch("/{project_id}/title")
async def update_project_title(project_id: str, body: ProjectTitleUpdateRequest):
    """更新项目标题。"""
    project_service = get_project_service()
    is_success = project_service.update_title(project_id, body.title)
    if not is_success:
        raise HTTPException(status_code=404, detail="项目不存在")
    project = project_service.get_project(project_id)
    return {"success": True, "data": project.to_dict() if project else None}


@router.get("/{project_id}/subtitles")
async def list_project_subtitles(project_id: str):
    """获取项目字幕。"""
    project_service = get_project_service()
    subtitle_doc_service = get_subtitle_doc_service()
    project_dir = project_service.get_project_dir(project_id)
    if project_dir is None:
        raise HTTPException(status_code=404, detail="项目不存在")
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

    is_success = subtitle_doc_service.delete_segment(project_dir, segment_id)
    if not is_success:
        raise HTTPException(status_code=404, detail="字幕段不存在")

    _publish_project_subtitle_event(
        project_id,
        "deleted",
        {"segment_id": segment_id, "source": "project_api", "is_update": True},
    )
    return {"success": True, "data": {"segment_id": segment_id, "is_deleted": True}}


@router.get("/{project_id}/export")
async def export_project_subtitles(project_id: str, format: Literal["srt", "ass"] = "srt"):
    """导出项目字幕文本。"""
    project_service = get_project_service()
    subtitle_doc_service = get_subtitle_doc_service()
    project_dir = project_service.get_project_dir(project_id)
    if project_dir is None:
        raise HTTPException(status_code=404, detail="项目不存在")

    if format == "srt":
        content = subtitle_doc_service.export_srt(project_dir)
    else:
        content = subtitle_doc_service.export_ass(project_dir)
    return {"success": True, "data": {"format": format, "content": content}}
