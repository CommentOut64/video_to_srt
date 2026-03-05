"""
模型管理API路由（基于 ModelManager V2）
"""

import logging
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.model_manager_v2 import get_model_manager_v2
from app.services.sse_service import get_sse_manager
from app.services.model_download_event_bus import get_model_download_event_bus
from app.services.model_bootstrap_service import get_model_bootstrap_service

router = APIRouter(prefix="/api/models", tags=["models"])
logger = logging.getLogger(__name__)

model_manager = get_model_manager_v2()


def _resolve_bootstrap_mode_for_api() -> str:
    raw_mode = str(os.getenv("MODEL_BOOTSTRAP_MODE", "")).strip().lower()
    if raw_mode in {"strict", "background", "off"}:
        return raw_mode
    legacy_auto = str(os.getenv("MODEL_BOOTSTRAP_AUTO", "")).strip().lower()
    if legacy_auto:
        if legacy_auto in {"0", "false", "no", "off"}:
            return "off"
        return "background"
    return "background"


class BootstrapTriggerRequest(BaseModel):
    """后台模型自愈触发请求。"""

    model_ids: Optional[List[str]] = Field(default=None, description="可选：只触发指定模型")
    is_force: bool = Field(default=False, alias="force", description="是否强制修复已就绪模型")


@router.get("/", response_model=List[dict])
async def list_models():
    """列出所有模型状态。"""
    try:
        status = model_manager.list_status()
        return list(status.values())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型状态失败: {str(e)}")


@router.get("/bootstrap/status")
async def get_bootstrap_status():
    """获取模型后台自愈状态。"""
    try:
        bootstrap_service = get_model_bootstrap_service(model_manager=model_manager)
        return {
            "success": True,
            "mode": _resolve_bootstrap_mode_for_api(),
            "required_models": bootstrap_service.resolve_required_model_ids(),
            **bootstrap_service.get_status(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取后台自愈状态失败: {str(e)}")


@router.post("/bootstrap/trigger")
async def trigger_bootstrap(req: BootstrapTriggerRequest):
    """手动触发后台模型自愈。"""
    try:
        bootstrap_service = get_model_bootstrap_service(model_manager=model_manager)
        is_started = bootstrap_service.start_non_blocking(
            model_ids=req.model_ids,
            is_force=req.is_force,
        )
        return {
            "success": True,
            "started": is_started,
            "status": bootstrap_service.get_status(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"触发后台自愈失败: {str(e)}")


@router.post("/{model_id}/ensure")
async def ensure_model(model_id: str):
    """确保模型可用（必要时下载或校验）。"""
    try:
        path = model_manager.ensure_available(model_id)
        return {"success": True, "message": f"模型可用: {path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"确保模型失败: {str(e)}")


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """删除本地模型缓存（如存在）。"""
    try:
        success = model_manager.delete_model(model_id)
        if not success:
            raise HTTPException(status_code=400, detail="模型不存在或未下载")
        return {"success": True, "message": f"已删除模型 {model_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


@router.get("/punct/{language}")
async def select_punctuation_model(language: str):
    """按语言选择标点模型，返回模型ID或404。"""
    try:
        model_id = model_manager.select_punct_model(language)
        if not model_id:
            raise HTTPException(status_code=404, detail="未找到合适的标点模型")
        return {"success": True, "model_id": model_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"选择标点模型失败: {str(e)}")


@router.get("/metrics")
async def metrics():
    """Prometheus 指标导出。"""
    text = model_manager.metrics_text()
    from fastapi.responses import Response
    return Response(content=text, media_type="text/plain; version=0.0.4")


@router.get("/events")
async def model_events(request: Request):
    """模型下载 SSE 事件流（频道：models）。"""
    sse_manager = get_sse_manager()
    event_bus = get_model_download_event_bus()

    return StreamingResponse(
        sse_manager.subscribe("models", request, initial_state_callback=event_bus.get_snapshot),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
