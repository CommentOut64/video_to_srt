"""
调试相关 API 路由。
V3.2.0+dev.20260201.02
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import config


router = APIRouter(prefix="/api/debug", tags=["debug"])


@router.get("/punctuation/{identifier}")
def get_punctuation_debug(identifier: str):
    """获取指定任务的标点调试输出（jsonl）。"""
    job_dir = Path(config.JOBS_DIR) / identifier
    debug_file = job_dir / "debug" / "punctuation.jsonl"
    if not debug_file.exists():
        raise HTTPException(status_code=404, detail="调试文件不存在")
    return FileResponse(
        path=debug_file,
        media_type="application/jsonl",
        filename="punctuation.jsonl",
    )
