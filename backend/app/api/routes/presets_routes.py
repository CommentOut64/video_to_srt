"""
自定义预设 API 路由

端点：
- GET    /api/presets/custom    获取所有自定义预设
- POST   /api/presets/custom    新增自定义预设
- DELETE /api/presets/custom/{preset_id}  删除自定义预设
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/presets", tags=["presets"])


class CreatePresetRequest(BaseModel):
    """新增预设请求"""
    name: str = Field(..., min_length=1, max_length=50, description="预设名称")
    config: Dict[str, Any] = Field(..., description="预设配置")


@router.get("/custom")
async def get_custom_presets():
    """获取所有自定义预设"""
    from app.services.presets_service import get_all_presets

    try:
        presets = get_all_presets()
        return {"success": True, "presets": presets}
    except Exception as e:
        logger.error(f"获取自定义预设失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/custom")
async def create_custom_preset(req: CreatePresetRequest):
    """新增自定义预设"""
    from app.services.presets_service import add_preset

    try:
        preset = add_preset(req.name, req.config)
        return {"success": True, "preset": preset}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"新增自定义预设失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/custom/{preset_id}")
async def delete_custom_preset(preset_id: str):
    """删除自定义预设"""
    from app.services.presets_service import delete_preset

    try:
        deleted = delete_preset(preset_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="预设不存在")
        return {"success": True, "message": f"预设 {preset_id} 已删除"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除自定义预设失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
