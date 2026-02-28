"""
Legacy 兼容路由。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
import logging

from fastapi import APIRouter, HTTPException

from app.core.config import config
from app.services.legacy_projection_service import get_legacy_projection_service
from app.services.project_service import get_project_service

router = APIRouter(prefix="/api/legacy", tags=["legacy"])
logger = logging.getLogger(__name__)


def _load_job_status_from_meta(job_id: str) -> Optional[str]:
    job_meta_path = Path(config.JOBS_DIR) / job_id / "job_meta.json"
    if not job_meta_path.exists():
        return None
    try:
        with open(job_meta_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        status = payload.get("status")
        return str(status) if status is not None else None
    except Exception:
        return None


def _safe_get_job_from_runtime(job_id: str):
    """
    尝试从运行态任务管理器读取任务。

    说明：
    - Lite 模式或依赖缺失时，允许失败并回退到 job_meta.json。
    """
    try:
        from app.services.transcription_service import get_transcription_service

        transcription_service = get_transcription_service(str(config.JOBS_DIR))
        return transcription_service.get_job(job_id)
    except Exception:
        return None


def _bind_project_id_to_runtime_job(job_id: str, project_id: str) -> None:
    """
    将 resolve 得到的 project_id 回绑到运行态任务，避免后续链路出现 project_id 漂移。
    """
    runtime_job = _safe_get_job_from_runtime(job_id)
    if runtime_job is None:
        return
    current_project_id = getattr(runtime_job, "project_id", None)
    if current_project_id == project_id:
        return
    try:
        runtime_job.project_id = project_id
    except Exception as exc:
        logger.warning("运行态任务绑定 project_id 失败: job_id=%s, project_id=%s, err=%s", job_id, project_id, exc)


@router.get("/tasks/{job_id}/resolve")
async def resolve_legacy_task(job_id: str):
    """
    解析旧任务为项目 ID。
    """
    legacy_service = get_legacy_projection_service()
    try:
        project_id, is_newly_migrated = legacy_service.resolve(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"解析旧任务失败: {exc}") from exc

    _bind_project_id_to_runtime_job(job_id, project_id)
    return {
        "success": True,
        "project_id": project_id,
        "is_newly_migrated": is_newly_migrated,
        "redirect_url": f"/editor/project/{project_id}",
    }


@router.get("/tasks/{job_id}/status")
async def get_legacy_status(job_id: str):
    """
    返回旧任务兼容状态。
    """
    legacy_service = get_legacy_projection_service()
    project_service = get_project_service()

    try:
        project_id, is_newly_migrated = legacy_service.resolve(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"读取旧任务状态失败: {exc}") from exc

    _bind_project_id_to_runtime_job(job_id, project_id)
    project = project_service.get_project(project_id)
    job = _safe_get_job_from_runtime(job_id)

    status = None
    phase = None
    progress = None
    if job is not None:
        status = job.status
        phase = job.phase
        progress = job.progress
    else:
        status = _load_job_status_from_meta(job_id) or "unknown"

    return {
        "success": True,
        "project_id": project_id,
        "job_id": job_id,
        "is_newly_migrated": is_newly_migrated,
        "status": status,
        "phase": phase,
        "progress": progress,
        "redirect_url": f"/editor/project/{project_id}",
        "project": project.to_dict() if project else None,
    }
