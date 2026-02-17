"""
双流流水线内核入口。

说明：
- `implementation.py` 承载完整实现细节；
- 本模块作为稳定内核入口，负责实现层装配与兼容导出。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from app.pipelines.dual_pipeline import implementation as _implementation
from app.pipelines.dual_pipeline.services import DiagnosticTraceService

if TYPE_CHECKING:
    from app.utils.cancellation_token import CancellationToken

# 可替换调试钩子（默认指向实现层）
append_debug_dual_time_compare_line = _implementation.append_debug_dual_time_compare_line
append_debug_layer_diag_line = _implementation.append_debug_layer_diag_line
append_debug_layer_trace_line = _implementation.append_debug_layer_trace_line
append_debug_m2_stage0_line = _implementation.append_debug_m2_stage0_line
append_debug_whisper_line = _implementation.append_debug_whisper_line
write_debug_json_payload = _implementation.write_debug_json_payload


class AsyncDualPipelineKernel(_implementation.AsyncDualPipelineKernel):
    """
    内核入口类（Kernel Facade）。

    设计模式：Facade Pattern
    - 对外稳定暴露 `AsyncDualPipelineKernel` 类型；
    - 具体算法实现下沉到 `implementation.py`，便于后续按领域继续拆分。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._diagnostic_trace_service = DiagnosticTraceService(logger=self.logger)

    def _emit_whisper_debug(
        self,
        job_dir: Optional[Path],
        *,
        group_id: str,
        flush_reason: str,
        whisper_result: Dict[str, Any],
        chunk_indices: List[int],
    ) -> None:
        self._diagnostic_trace_service.emit_whisper_debug(
            job_dir,
            group_id=group_id,
            flush_reason=flush_reason,
            whisper_result=whisper_result,
            chunk_indices=chunk_indices,
            append_debug_whisper_line=append_debug_whisper_line,
        )

    def _emit_layer_diagnostics(
        self,
        ctx: Any,
        *,
        tracks: Any,
        punct_track: Optional[Any],
        alignment_result: Any,
        injection_stats: Dict[str, Any],
        split_stats: Dict[str, Any],
        final_sentences: List[Any],
    ) -> None:
        self._diagnostic_trace_service.emit_layer_diagnostics(
            ctx,
            tracks=tracks,
            punct_track=punct_track,
            alignment_result=alignment_result,
            injection_stats=injection_stats,
            split_stats=split_stats,
            final_sentences=final_sentences,
            append_debug_layer_diag_line=append_debug_layer_diag_line,
        )

    def _emit_layer_trace_full(
        self,
        ctx: Any,
        *,
        tracks: Any,
        sv_result: Dict[str, Any],
        whisper_result: Dict[str, Any],
        arbitration_output: Any,
        punct_track: Optional[Any],
        alignment_result: Any,
        aligned_facts: Any,
        fused_evidence: Any,
        words_for_split: Sequence[Any],
        injection_stats: Dict[str, Any],
        split_stats: Dict[str, Any],
        final_sentences: Sequence[Any],
        output_traces: Sequence[Any],
    ) -> None:
        self._diagnostic_trace_service.emit_layer_trace_full(
            ctx,
            tracks=tracks,
            sv_result=sv_result,
            whisper_result=whisper_result,
            arbitration_output=arbitration_output,
            punct_track=punct_track,
            alignment_result=alignment_result,
            aligned_facts=aligned_facts,
            fused_evidence=fused_evidence,
            words_for_split=words_for_split,
            injection_stats=injection_stats,
            split_stats=split_stats,
            final_sentences=final_sentences,
            output_traces=output_traces,
            append_debug_layer_trace_line=append_debug_layer_trace_line,
        )


def get_async_dual_pipeline_kernel(
    job_id: str,
    queue_maxsize: int = 5,
    logger: Optional[logging.Logger] = None,
    cancellation_token: Optional["CancellationToken"] = None,
) -> AsyncDualPipelineKernel:
    """
    获取异步双流流水线内核实例。
    """
    return AsyncDualPipelineKernel(
        job_id=job_id,
        queue_maxsize=queue_maxsize,
        logger=logger,
        cancellation_token=cancellation_token,
    )


def sync_debug_hooks(
    *,
    hook_append_debug_dual_time_compare_line: Any,
    hook_append_debug_layer_diag_line: Any,
    hook_append_debug_layer_trace_line: Any,
    hook_append_debug_m2_stage0_line: Any,
    hook_append_debug_whisper_line: Any,
    hook_write_debug_json_payload: Any,
) -> None:
    """
    将外观层的可替换调试钩子同步到实现层模块。
    """
    global append_debug_dual_time_compare_line
    global append_debug_layer_diag_line
    global append_debug_layer_trace_line
    global append_debug_m2_stage0_line
    global append_debug_whisper_line
    global write_debug_json_payload

    append_debug_dual_time_compare_line = hook_append_debug_dual_time_compare_line
    append_debug_layer_diag_line = hook_append_debug_layer_diag_line
    append_debug_layer_trace_line = hook_append_debug_layer_trace_line
    append_debug_m2_stage0_line = hook_append_debug_m2_stage0_line
    append_debug_whisper_line = hook_append_debug_whisper_line
    write_debug_json_payload = hook_write_debug_json_payload

    _implementation.append_debug_dual_time_compare_line = hook_append_debug_dual_time_compare_line
    _implementation.append_debug_layer_diag_line = hook_append_debug_layer_diag_line
    _implementation.append_debug_layer_trace_line = hook_append_debug_layer_trace_line
    _implementation.append_debug_m2_stage0_line = hook_append_debug_m2_stage0_line
    _implementation.append_debug_whisper_line = hook_append_debug_whisper_line
    _implementation.write_debug_json_payload = hook_write_debug_json_payload


def __getattr__(name: str) -> Any:
    """
    透传实现层符号，兼容历史直接导入。
    """
    return getattr(_implementation, name)


__all__ = [
    "AsyncDualPipelineKernel",
    "get_async_dual_pipeline_kernel",
    "sync_debug_hooks",
]
