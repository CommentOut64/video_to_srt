"""
API路由模块初始化
"""

__all__ = ["create_file_router", "create_transcription_router", "create_speaker_router"]


def __getattr__(name: str):
    """
    延迟导入路由工厂，避免 Lite 模式在包初始化时触发 Full 依赖链
    """
    if name == "create_file_router":
        from .file_routes import create_file_router

        return create_file_router
    if name == "create_transcription_router":
        from .transcription_routes import create_transcription_router

        return create_transcription_router
    if name == "create_speaker_router":
        from .speaker_routes import create_speaker_router

        return create_speaker_router
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(__all__)
