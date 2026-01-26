"""
流水线模块

包含:
- async_dual_pipeline: 异步双流流水线（V3.1.0+）
- preprocessing_pipeline: 预处理流水线（新架构 - Stage模式）
"""

from app.pipelines.async_dual_pipeline import (
    AsyncDualPipeline,
    get_async_dual_pipeline
)

from app.pipelines.preprocessing_pipeline import (
    PreprocessingPipeline,
    get_preprocessing_pipeline
)

__all__ = [
    # 异步双流流水线（V3.1.0+ - 生产使用）
    'AsyncDualPipeline',
    'get_async_dual_pipeline',

    # 预处理流水线（新架构 - Stage模式）
    'PreprocessingPipeline',
    'get_preprocessing_pipeline',
]
