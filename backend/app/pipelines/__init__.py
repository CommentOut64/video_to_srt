"""
流水线模块

包含:
- async_dual_pipeline: 异步双流流水线（V3.1.0+）
- preprocessing_pipeline: 预处理流水线（新架构 - Stage模式）
"""

__all__ = [
    "AsyncDualPipeline",
    "get_async_dual_pipeline",
    "PreprocessingPipeline",
    "get_preprocessing_pipeline",
]


def __getattr__(name: str):
    """延迟导入，避免无关依赖在导入时被强制加载。"""
    if name in {"AsyncDualPipeline", "get_async_dual_pipeline"}:
        from app.pipelines.async_dual_pipeline import (
            AsyncDualPipeline,
            get_async_dual_pipeline,
        )

        return {"AsyncDualPipeline": AsyncDualPipeline, "get_async_dual_pipeline": get_async_dual_pipeline}[name]
    if name in {"PreprocessingPipeline", "get_preprocessing_pipeline"}:
        from app.pipelines.preprocessing_pipeline import (
            PreprocessingPipeline,
            get_preprocessing_pipeline,
        )

        return {
            "PreprocessingPipeline": PreprocessingPipeline,
            "get_preprocessing_pipeline": get_preprocessing_pipeline,
        }[name]
    raise AttributeError(f"module 'app.pipelines' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
