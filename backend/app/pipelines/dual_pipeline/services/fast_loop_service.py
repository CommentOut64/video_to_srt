"""
FastWorker 循环门面服务。

设计模式：Facade Pattern。
原因：将 Fast 生产者循环从实现类抽离，降低主类体积并保持单一职责。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List, Optional, Set

from app.schemas.pipeline_context import ProcessingContext
from app.services.audio.chunk_engine import AudioChunk
from app.utils.cancellation_token import CancelledException, PausedException


class FastLoopService:
    """FastWorker 生产者循环门面。"""

    def __init__(self, *, host: Any) -> None:
        self._host = host

    async def _put_queue_inter_with_guard(
        self,
        *,
        payload: ProcessingContext,
        token: Any,
        timeout_seconds: float = 0.5,
        max_retry: int = 6,
        allow_drop: bool = False,
        reason: str = "",
    ) -> bool:
        """
        带取消守卫的 queue_inter 投递。

        Why:
        - 下游提前退出时，`await queue.put(...)` 可能因背压永久阻塞
        - 取消链路必须优先退出，不能被队列写入卡死
        """
        host = self._host
        retries = 0
        while True:
            try:
                await asyncio.wait_for(
                    host.queue_inter.put(payload),
                    timeout=timeout_seconds,
                )
                return True
            except asyncio.TimeoutError:
                retries += 1
                if token:
                    token.raise_if_canceled()
                if retries >= max_retry:
                    if allow_drop:
                        host.logger.warning(
                            "FastWorker 投递 queue_inter 超时，丢弃载荷: reason=%s, retries=%s",
                            reason,
                            retries,
                        )
                        return False
                    raise RuntimeError(
                        f"FastWorker 投递 queue_inter 超时: reason={reason}, retries={retries}"
                    )

    async def run(
        self,
        *,
        chunks: List[AudioChunk],
        full_audio_array: Optional[Any] = None,
        full_audio_sr: int = 16000,
        job_dir: Optional[Path] = None,
        processed_indices: Optional[Set[int]] = None,
        total_chunks: int = 0,
    ) -> None:
        """执行 FastWorker 主循环。"""
        host = self._host
        token = host.cancellation_token
        processed_indices = processed_indices or set()
        total_chunks = total_chunks or len(chunks)

        base_fast_count = len(processed_indices)
        fast_processed_count = 0
        pause_requested = False
        should_send_end_signal = True
        last_chunk_index: Optional[int] = None

        try:
            for i, chunk in enumerate(chunks):
                if i in processed_indices:
                    host.logger.debug(f"[FastWorker] 跳过已处理的 chunk {i}")
                    continue

                ctx = ProcessingContext(
                    job_id=host.job_id,
                    chunk_index=i,
                    audio_chunk=chunk,
                    job_dir=job_dir,
                    debug_punctuation=host.debug_punctuation,
                    full_audio_array=full_audio_array,
                    full_audio_sr=full_audio_sr,
                )

                if token:
                    token.enter_atomic_region(f"fast_worker_chunk_{i}")

                try:
                    await host.fast_worker.process(ctx)
                    if ctx.sv_result:
                        host._validate_l0_result(ctx.sv_result, source="fast", chunk_index=i)
                    normalized = host._normalize_sensevoice_result(ctx)
                    await host._apply_fast_punctuation(ctx, normalized)
                    host._context_cache[ctx.chunk_index] = ctx
                    used_semantic = await host._emit_draft_sentences(
                        ctx,
                        is_final_output=False,
                    )

                    if (
                        not host._enable_bridge_batches
                        or (not host._consume_turn_groups_only and not used_semantic)
                    ):
                        await self._put_queue_inter_with_guard(
                            payload=ctx,
                            token=token,
                            reason=f"chunk_{i}",
                        )
                    fast_processed_count += 1
                    last_chunk_index = i

                    if host.progress_emitter:
                        total_processed = base_fast_count + fast_processed_count
                        host.progress_emitter.update_fast(
                            total_processed,
                            total_chunks,
                            message=f"SenseVoice: {total_processed}/{total_chunks}",
                        )
                finally:
                    if token:
                        has_pending = token.exit_atomic_region()
                        if has_pending:
                            host.logger.debug(
                                f"[v3.1.0] FastWorker chunk {i} 完成后检测到待处理请求"
                            )

                if token and job_dir:
                    processed_indices.add(i)
                    host._fast_processed_indices = processed_indices
                    checkpoint_data = {
                        "transcription": {
                            "fast_processed_indices": list(processed_indices),
                            "fast_processed_count": len(processed_indices),
                            "total_chunks": len(chunks),
                        }
                    }
                    try:
                        token.check_and_save(checkpoint_data, job_dir)
                    except PausedException as exc:
                        pause_requested = True
                        if not host.pause_exception:
                            host.pause_exception = exc
                        host.logger.debug(
                            f"[V3.1.0] FastWorker 捕获暂停信号，已完成 {len(processed_indices)} / {len(chunks)} 个 Chunk"
                        )
                        break

        except CancelledException as exc:
            host.logger.info(f"FastWorker 循环取消: {exc}")
            host.errors.append(exc)
            should_send_end_signal = False

            error_ctx = ProcessingContext(
                job_id=host.job_id,
                chunk_index=-1,
                audio_chunk=None,
                job_dir=job_dir,
                debug_punctuation=host.debug_punctuation,
                is_end=True,
                error=exc,
            )
            await self._put_queue_inter_with_guard(
                payload=error_ctx,
                token=token,
                allow_drop=True,
                reason="cancel_error_ctx",
            )
            return

        except Exception as exc:
            host.logger.error(f"FastWorker 循环异常: {exc}", exc_info=True)
            host.errors.append(exc)

            error_ctx = ProcessingContext(
                job_id=host.job_id,
                chunk_index=-1,
                audio_chunk=None,
                job_dir=job_dir,
                debug_punctuation=host.debug_punctuation,
                is_end=True,
                error=exc,
            )
            await self._put_queue_inter_with_guard(
                payload=error_ctx,
                token=token,
                allow_drop=True,
                reason="exception_error_ctx",
            )
            should_send_end_signal = False
            return
        finally:
            if should_send_end_signal:
                if last_chunk_index is not None:
                    await host._flush_semantic_buffer(
                        is_final_output=False,
                        chunk_index=last_chunk_index,
                    )
                end_ctx = ProcessingContext(
                    job_id=host.job_id,
                    chunk_index=-1,
                    audio_chunk=None,
                    job_dir=job_dir,
                    debug_punctuation=host.debug_punctuation,
                    is_end=True,
                )
                await self._put_queue_inter_with_guard(
                    payload=end_ctx,
                    token=token,
                    allow_drop=True,
                    reason="end_ctx",
                )
                if pause_requested:
                    host.logger.debug("[V3.1.0] FastWorker 已发送暂停结束信号，等待下游排空")
                else:
                    host.logger.info("FastWorker 循环完成")
