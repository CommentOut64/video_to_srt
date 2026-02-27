"""
流式订阅路由。

职责：
1. 提供项目级 SSE 订阅入口（project channel）。
2. 统一复用 SSEManager 的订阅与心跳机制。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.services.project_service import get_project_service
from app.services.sse_service import get_sse_manager
from app.services.subtitle_doc_service import get_subtitle_doc_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stream", tags=["stream"])


@router.get("/project/{project_id}")
async def stream_project_events(project_id: str, request: Request):
    """
    订阅项目级 SSE 事件流。

    频道：
    - `project:{project_id}`

    主要事件：
    - `subtitle.added`
    - `subtitle.edited`
    - `subtitle.deleted`
    """
    project_service = get_project_service()
    subtitle_doc_service = get_subtitle_doc_service()
    sse_manager = get_sse_manager()

    project = project_service.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="项目不存在")

    channel_id = f"project:{project_id}"

    def get_initial_state() -> dict:
        """
        连接建立后推送初始快照。

        说明：
        - 前端当前主要依赖增量事件；初始快照用于诊断和可观测性。
        """
        project_dir = project_service.get_project_dir(project_id)
        if project_dir is None:
            return {
                "project_id": project_id,
                "title": project.title,
                "segment_count": 0,
                "updated_at": project.updated_at,
            }
        try:
            segments = subtitle_doc_service.load_segments(project_dir)
            segment_count = len(segments)
        except Exception as exc:
            logger.warning("读取项目字幕初始快照失败: %s", exc)
            segment_count = 0

        return {
            "project_id": project_id,
            "title": project.title,
            "mode": project.mode,
            "flavor": project.flavor,
            "job_id": project.job_id,
            "segment_count": segment_count,
            "updated_at": project.updated_at,
        }

    return StreamingResponse(
        sse_manager.subscribe(
            channel_id,
            request,
            initial_state_callback=get_initial_state,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

