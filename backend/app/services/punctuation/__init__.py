"""
标点服务包入口。
"""
from typing import Any

__all__ = ["PunctuationService", "get_punctuation_service"]


def __getattr__(name: str) -> Any:
    """延迟导入，避免不必要的重依赖加载。"""
    if name in __all__:
        from app.services.punctuation.service import PunctuationService, get_punctuation_service

        return {"PunctuationService": PunctuationService, "get_punctuation_service": get_punctuation_service}[name]
    raise AttributeError(name)
