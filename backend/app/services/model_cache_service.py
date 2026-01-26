"""
ModelCacheService - 模型缓存管理

集中处理模型缓存清理逻辑，避免分散在业务服务中。
"""
from __future__ import annotations

import logging
from typing import Optional


class ModelCacheService:
    """模型缓存管理器"""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def clear_whisper_cache(self) -> None:
        """清理 Whisper 模型缓存"""
        from app.services.model_manager_v2 import get_model_manager_v2

        manager = get_model_manager_v2()
        manager.unload_all()
        self.logger.info("ModelManagerV2 缓存已清空")

    def clear_model_cache(self) -> None:
        """兼容旧接口：清理模型缓存"""
        self.clear_whisper_cache()


_model_cache_service: Optional[ModelCacheService] = None


def get_model_cache_service(
    logger: Optional[logging.Logger] = None,
) -> ModelCacheService:
    """获取模型缓存服务（单例）"""
    global _model_cache_service
    if _model_cache_service is None:
        _model_cache_service = ModelCacheService(logger=logger)
    elif logger is not None:
        _model_cache_service.logger = logger
    return _model_cache_service
