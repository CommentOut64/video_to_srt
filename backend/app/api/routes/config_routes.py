"""
用户配置API路由
提供前端设置菜单使用的配置接口
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/config", tags=["config"])


# ========== 请求/响应模型 ==========

class SubtitleTimeOffsetRequest(BaseModel):
    """字幕时间偏移请求"""
    offset: float = Field(
        ...,
        ge=-10.0,
        le=10.0,
        description="偏移量（秒），正值延后，负值提前，范围 -10.0 到 10.0"
    )


class DefaultModelRequest(BaseModel):
    """默认模型请求"""
    model_id: Optional[str] = Field(
        None,
        description="模型ID，None表示清除用户选择"
    )


class ConfigUpdateRequest(BaseModel):
    """通用配置更新请求"""
    updates: Dict[str, Any] = Field(
        ...,
        description="要更新的配置项键值对"
    )


# ========== API 端点 ==========

@router.get("/all")
async def get_all_config():
    """
    获取所有用户配置
    
    Returns:
        所有配置项的字典
    """
    from app.services.user_config_service import get_user_config_service
    
    try:
        config_service = get_user_config_service()
        config_data = config_service.get_all_config()
        return {
            "success": True,
            "config": config_data
        }
    except Exception as e:
        logger.error(f"获取配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/subtitle-time-offset")
async def get_subtitle_time_offset():
    """
    获取字幕时间戳全局偏移
    
    Returns:
        offset: 偏移量（秒）
    """
    from app.services.user_config_service import get_user_config_service
    
    try:
        config_service = get_user_config_service()
        offset = config_service.get_subtitle_time_offset()
        return {
            "success": True,
            "offset": offset,
            "description": "正值延后字幕显示，负值提前字幕显示"
        }
    except Exception as e:
        logger.error(f"获取字幕时间偏移失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/subtitle-time-offset")
async def set_subtitle_time_offset(req: SubtitleTimeOffsetRequest):
    """
    设置字幕时间戳全局偏移
    
    用于音画不同步时的全局校正：
    - 正值：字幕延后显示（音频比画面快时使用）
    - 负值：字幕提前显示（音频比画面慢时使用）
    
    Args:
        offset: 偏移量（秒），范围 -10.0 到 10.0
        
    Returns:
        success: 是否设置成功
    """
    from app.services.user_config_service import get_user_config_service
    
    try:
        config_service = get_user_config_service()
        success = config_service.set_subtitle_time_offset(req.offset)
        
        if success:
            return {
                "success": True,
                "offset": req.offset,
                "message": f"字幕时间偏移已设置为 {req.offset} 秒"
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="设置失败，请检查偏移量范围（-10.0 到 10.0）"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"设置字幕时间偏移失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/default-model")
async def get_default_model():
    """
    获取默认预加载模型
    
    Returns:
        model_id: 模型ID，None表示未设置
    """
    from app.services.user_config_service import get_user_config_service
    
    try:
        config_service = get_user_config_service()
        model_id = config_service.get_default_preload_model()
        return {
            "success": True,
            "model_id": model_id
        }
    except Exception as e:
        logger.error(f"获取默认模型失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/default-model")
async def set_default_model(req: DefaultModelRequest):
    """
    设置默认预加载模型
    
    Args:
        model_id: 模型ID，None表示清除用户选择
        
    Returns:
        success: 是否设置成功
    """
    from app.services.user_config_service import get_user_config_service
    
    try:
        config_service = get_user_config_service()
        success = config_service.set_default_preload_model(req.model_id)
        
        if success:
            return {
                "success": True,
                "model_id": req.model_id,
                "message": f"默认模型已设置为 {req.model_id}" if req.model_id else "默认模型已清除"
            }
        else:
            raise HTTPException(status_code=400, detail="设置失败")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"设置默认模型失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/update")
async def update_config(req: ConfigUpdateRequest):
    """
    批量更新配置
    
    Args:
        updates: 要更新的配置项键值对
        
    Returns:
        success: 是否更新成功
    """
    from app.services.user_config_service import get_user_config_service
    
    try:
        config_service = get_user_config_service()
        success = config_service.update_config(req.updates)
        
        if success:
            return {
                "success": True,
                "updated_keys": list(req.updates.keys()),
                "message": "配置已更新"
            }
        else:
            raise HTTPException(status_code=400, detail="更新失败")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")


# ========== V3.1.2+dev.20260113.01: Proxy配置管理 ==========

class ProxyConfigRequest(BaseModel):
    """Proxy配置更新请求"""
    auto_trigger_720p: bool = Field(
        ...,
        description="是否自动触发720p转码"
    )


@router.get("/proxy")
async def get_proxy_config():
    """
    获取Proxy配置

    Returns:
        auto_trigger_720p: 是否自动触发720p转码
    """
    from app.core.config import config

    return {
        "auto_trigger_720p": config.PROXY_CONFIG.get('auto_trigger_720p', False)
    }


@router.put("/proxy")
async def update_proxy_config(req: ProxyConfigRequest):
    """
    更新Proxy配置

    Args:
        auto_trigger_720p: 是否自动触发720p转码

    Returns:
        success: 是否更新成功
    """
    from app.core.config import config

    try:
        config.PROXY_CONFIG['auto_trigger_720p'] = req.auto_trigger_720p

        logger.info(f"[Config] Proxy配置已更新: auto_trigger_720p={req.auto_trigger_720p}")

        return {
            "success": True,
            "message": "Proxy配置已更新",
            "config": {
                "auto_trigger_720p": req.auto_trigger_720p
            }
        }
    except Exception as e:
        logger.error(f"更新Proxy配置失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新配置失败: {str(e)}")
