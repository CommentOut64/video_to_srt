"""
AsyncDualPipeline 对外入口（编排层外观）。

说明：
- 核心业务算法已迁移到 `app.pipelines.dual_pipeline.kernel`。
- 本文件仅保留对外稳定入口与依赖注入兼容层，避免调用方受重构影响。
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from app.pipelines.dual_pipeline import kernel as _kernel

if TYPE_CHECKING:
    from app.utils.cancellation_token import CancellationToken

# 调试钩子默认指向 Kernel 实现，测试可在本模块 monkeypatch 后由 _sync_debug_hooks 回写。
append_debug_dual_time_compare_line = _kernel.append_debug_dual_time_compare_line
append_debug_layer_diag_line = _kernel.append_debug_layer_diag_line
append_debug_layer_trace_line = _kernel.append_debug_layer_trace_line
append_debug_m2_stage0_line = _kernel.append_debug_m2_stage0_line
append_debug_whisper_line = _kernel.append_debug_whisper_line
write_debug_json_payload = _kernel.write_debug_json_payload


class AsyncDualPipeline(_kernel.AsyncDualPipelineKernel):
    """
    对外兼容类。

    设计说明（Facade Pattern）：
    - 保持原有 `app.pipelines.async_dual_pipeline.AsyncDualPipeline` 导入路径不变。
    - 具体实现下沉至 Kernel，当前类只承担入口稳定性与依赖注入承接职责。
    """

    @staticmethod
    def _sync_debug_hooks() -> None:
        """
        同步可 monkeypatch 的调试函数到 Kernel 模块。
        """
        if hasattr(_kernel, "sync_debug_hooks"):
            _kernel.sync_debug_hooks(
                hook_append_debug_dual_time_compare_line=append_debug_dual_time_compare_line,
                hook_append_debug_layer_diag_line=append_debug_layer_diag_line,
                hook_append_debug_layer_trace_line=append_debug_layer_trace_line,
                hook_append_debug_m2_stage0_line=append_debug_m2_stage0_line,
                hook_append_debug_whisper_line=append_debug_whisper_line,
                hook_write_debug_json_payload=write_debug_json_payload,
            )

    async def run(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        self._sync_debug_hooks()
        return await super().run(*args, **kwargs)

    async def _run_alignment_stage(self, ctx):  # type: ignore[override]
        self._sync_debug_hooks()
        return await super()._run_alignment_stage(ctx)


def get_async_dual_pipeline(
    job_id: str,
    queue_maxsize: int = 5,
    logger: Optional[logging.Logger] = None,
    cancellation_token: Optional["CancellationToken"] = None,
) -> AsyncDualPipeline:
    """
    获取异步双流流水线实例（兼容旧入口）。
    """
    return AsyncDualPipeline(
        job_id=job_id,
        queue_maxsize=queue_maxsize,
        logger=logger,
        cancellation_token=cancellation_token,
    )


def __getattr__(name: str) -> Any:
    """
    兼容历史调用方直接从模块读取类型/工具函数的场景。
    """
    return getattr(_kernel, name)


__all__ = [
    "AsyncDualPipeline",
    "get_async_dual_pipeline",
]
