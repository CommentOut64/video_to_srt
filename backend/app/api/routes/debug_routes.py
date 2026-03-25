"""
调试相关 API 路由。
V3.2.0+dev.20260201.02
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Path as ApiPath
from fastapi.responses import FileResponse

from app.core.config import config


router = APIRouter(prefix="/api/debug", tags=["debug"])


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_FILENAME_PATTERN = re.compile(
    r"^(?:\d{2}_[a-z0-9_.]+\.(?:json|svg|html)|manifest\.json)$",
    re.IGNORECASE,
)


def _resolve_job_dir(identifier: str) -> Path:
    normalized = str(identifier or "").strip()
    if not normalized or not _IDENTIFIER_PATTERN.fullmatch(normalized):
        raise HTTPException(status_code=400, detail="非法任务标识")
    return Path(config.JOBS_DIR) / normalized


def _resolve_postprocess_file(*, identifier: str, chunk_index: int, filename: str) -> Path:
    safe_name = str(filename or "").strip()
    if not _FILENAME_PATTERN.fullmatch(safe_name):
        raise HTTPException(status_code=400, detail="非法文件名")
    job_dir = _resolve_job_dir(identifier)
    if safe_name == "manifest.json":
        target = job_dir / "debug" / "postprocess" / "manifest.json"
    else:
        target = (
            job_dir
            / "debug"
            / "postprocess"
            / f"chunk_{max(0, int(chunk_index)):04d}"
            / safe_name
        )
    if not target.exists():
        raise HTTPException(status_code=404, detail="调试文件不存在")
    return target


@router.get("/punctuation/{identifier}")
def get_punctuation_debug(identifier: str):
    """获取指定任务的标点调试输出（jsonl）。"""
    job_dir = _resolve_job_dir(identifier)
    debug_file = job_dir / "debug" / "punctuation.jsonl"
    if not debug_file.exists():
        raise HTTPException(status_code=404, detail="调试文件不存在")
    return FileResponse(
        path=debug_file,
        media_type="application/jsonl",
        filename="punctuation.jsonl",
    )


@router.get("/postprocess/{identifier}/manifest")
def get_postprocess_manifest(identifier: str) -> Dict[str, Any]:
    """读取后处理追踪清单。"""
    manifest_path = _resolve_postprocess_file(
        identifier=identifier,
        chunk_index=0,
        filename="manifest.json",
    )
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("manifest 非对象")
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"读取 manifest 失败: {exc}") from exc


@router.get("/postprocess/{identifier}/chunk/{chunk_index}/{filename}")
def get_postprocess_chunk_file(
    identifier: str,
    chunk_index: int = ApiPath(..., ge=0),
    filename: str = ApiPath(...),
):
    """下载指定 chunk 的后处理追踪文件。"""
    file_path = _resolve_postprocess_file(
        identifier=identifier,
        chunk_index=chunk_index,
        filename=filename,
    )
    suffix = file_path.suffix.lower()
    media_type = "application/json"
    if suffix == ".svg":
        media_type = "image/svg+xml"
    elif suffix == ".html":
        media_type = "text/html; charset=utf-8"
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=file_path.name,
    )
