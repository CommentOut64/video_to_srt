"""
对齐阶段循环门面服务。

设计模式：Facade Pattern。
原因：将最终消费者循环从实现类抽离，进一步缩小主实现类职责。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List, Optional

from app.schemas.pipeline_context import ProcessingContext
from app.utils.cancellation_token import CancelledException, PausedException


class AlignLoopService:
    """对齐阶段循环门面。"""

    def __init__(self, *, host: Any) -> None:
        self._host = host

    async def run(
        self,
        *,
        results: List[ProcessingContext],
        job_dir: Optional[Path] = None,
        total_chunks: int = 0,
        base_align_count: int = 0,
        initial_finalized_indices: Optional[set] = None,
    ) -> None:
        """执行对齐阶段循环。"""
        host = self._host
        token = host.cancellation_token
        if job_dir and host._runtime_checkpoint_service is None:
            try:
                from app.services.checkpoint import RuntimeCheckpointService

                host._runtime_checkpoint_service = RuntimeCheckpointService(job_dir=job_dir)
            except Exception as exc:
                host.logger.warning("Finalize 断点服务初始化失败，降级继续: %s", exc)

        finalized_indices = set(initial_finalized_indices) if initial_finalized_indices else set()
        host._finalized_indices = finalized_indices
        if finalized_indices:
            host._last_align_chunk_index = max(finalized_indices)
        pause_requested = False
        idle_poll_interval = 0.5

        try:
            while True:
                try:
                    ctx = await asyncio.wait_for(
                        host.queue_final.get(),
                        timeout=idle_poll_interval,
                    )
                except asyncio.TimeoutError:
                    if token:
                        token.raise_if_canceled()
                    continue

                if ctx.error:
                    if isinstance(ctx.error, CancelledException):
                        host.logger.info(f"上游取消信号: {ctx.error}")
                    else:
                        host.logger.error(f"上游错误: {ctx.error}")
                    raise ctx.error

                if ctx.is_end:
                    break

                chunk_index = ctx.chunk_index

                if host._is_finalize_batch_committed(chunk_index):
                    host.logger.info(
                        "Finalize 命中已提交 batch，跳过重算: chunk_index=%s",
                        chunk_index,
                    )
                    finalized_indices.add(chunk_index)
                    host._last_align_chunk_index = chunk_index
                    results.append(ctx)
                    if host.progress_emitter and total_chunks > 0:
                        total_processed = len(finalized_indices)
                        host.progress_emitter.update_align(
                            total_processed,
                            total_chunks,
                            message=f"对齐: {total_processed}/{total_chunks}",
                        )
                    continue

                host._record_finalize_batch_unit(
                    chunk_index=chunk_index,
                    status="started",
                    payload={
                        "chunk_index": int(chunk_index),
                        "job_id": host.job_id,
                    },
                )

                if token:
                    token.enter_atomic_region(f"align_stage_chunk_{chunk_index}")

                try:
                    await host._run_alignment_stage(ctx)

                    results.append(ctx)
                    finalized_indices.add(chunk_index)
                    host._last_align_chunk_index = chunk_index
                    host._record_finalize_batch_unit(
                        chunk_index=chunk_index,
                        status="committed",
                        payload={
                            "chunk_index": int(chunk_index),
                            "final_sentence_count": int(len(ctx.final_sentences or [])),
                        },
                    )

                    if host.progress_emitter and total_chunks > 0:
                        total_processed = len(finalized_indices)
                        host.progress_emitter.update_align(
                            total_processed,
                            total_chunks,
                            message=f"对齐: {total_processed}/{total_chunks}",
                        )
                finally:
                    if token:
                        has_pending = token.exit_atomic_region()
                        if has_pending:
                            host.logger.debug(
                                f"[v3.1.0] 对齐阶段 chunk {chunk_index} 完成后检测到待处理请求"
                            )

                if token and job_dir:
                    subtitle_checkpoint_data = {}
                    if host.subtitle_manager:
                        subtitle_checkpoint_data = host.subtitle_manager.to_checkpoint_data()

                    host._finalized_indices = finalized_indices
                    checkpoint_data = {
                        "transcription": {
                            "align_processed_count": len(finalized_indices),
                            "last_align_chunk_index": chunk_index,
                            "completed_chunks": len(results),
                            "finalized_indices": list(finalized_indices),
                            **subtitle_checkpoint_data,
                        }
                    }
                    try:
                        token.check_and_save(checkpoint_data, job_dir)
                    except PausedException as exc:
                        if not pause_requested:
                            host.logger.debug("[V3.1.0] 对齐阶段捕获暂停信号，继续排空 queue_final")
                        pause_requested = True
                        if not host.pause_exception:
                            host.pause_exception = exc

            host.logger.info("对齐阶段循环完成")

        except CancelledException as exc:
            host.logger.info(f"对齐阶段循环取消: {exc}")
            host.errors.append(exc)

        except Exception as exc:
            host.logger.error(f"对齐阶段循环异常: {exc}", exc_info=True)
            host.errors.append(exc)
        finally:
            if pause_requested:
                host.logger.debug("[V3.1.0] 对齐阶段已排空所有上下文，等待上层暂停")
