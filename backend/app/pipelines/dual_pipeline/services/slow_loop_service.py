"""
SlowWorker 循环门面服务。

设计模式：Facade Pattern。
原因：将中间消费者-生产者循环从实现类抽离，主类只保留调度入口。
"""

from __future__ import annotations

import asyncio
import copy
import time
from pathlib import Path
from typing import Any, Optional

from app.schemas.pipeline_context import ProcessingContext
from app.services.bridge.turn_group_builder import TurnGroupEnvelope
from app.utils.cancellation_token import CancelledException, PausedException


class SlowLoopService:
    """SlowWorker 循环门面。"""

    def __init__(self, *, host: Any) -> None:
        self._host = host

    async def _put_queue_final_with_guard(
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
        带取消守卫的 queue_final 投递。

        Why:
        - 对齐阶段提前退出时，SlowWorker 的 put 会被背压阻塞
        - 取消/删除链路必须保证可收敛退出
        """
        host = self._host
        retries = 0
        while True:
            try:
                await asyncio.wait_for(
                    host.queue_final.put(payload),
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
                            "SlowWorker 投递 queue_final 超时，丢弃载荷: reason=%s, retries=%s",
                            reason,
                            retries,
                        )
                        return False
                    raise RuntimeError(
                        f"SlowWorker 投递 queue_final 超时: reason={reason}, retries={retries}"
                    )

    async def run(
        self,
        *,
        job_dir: Optional[Path] = None,
        total_chunks: int = 0,
        base_slow_count: int = 0,
        initial_slow_processed_indices: Optional[set] = None,
    ) -> None:
        """执行 SlowWorker 循环。"""
        host = self._host
        token = host.cancellation_token
        slow_processed_count = 0
        slow_processed_indices = (
            set(initial_slow_processed_indices) if initial_slow_processed_indices else set()
        )
        host._slow_processed_indices = slow_processed_indices
        if slow_processed_indices:
            host._last_slow_chunk_index = max(slow_processed_indices)
        pause_requested = False
        idle_poll_interval = 0.5

        try:
            while True:
                try:
                    payload = await asyncio.wait_for(
                        host.queue_inter.get(),
                        timeout=idle_poll_interval,
                    )
                except asyncio.TimeoutError:
                    if token:
                        token.raise_if_canceled()
                    if host.bridge_controller and host.bridge_controller.should_flush_on_idle(
                        host.queue_inter.qsize()
                    ):
                        await host._flush_bridge_controller()
                    envelope = host._turn_group_builder.flush_idle(now=time.time())
                    if envelope:
                        await host._enqueue_turn_group(envelope)
                    continue

                if isinstance(payload, TurnGroupEnvelope):
                    if host._is_turn_group_committed(payload.group.group_id):
                        host.logger.info(
                            "W-Epoch 命中已提交 TurnGroup，跳过重算: group_id=%s",
                            payload.group.group_id,
                        )
                        continue
                    host._record_turn_group_unit(
                        group=payload.group,
                        status="started",
                        payload={
                            "source_chunks": list(payload.group.source_chunks),
                            "speaker_id": payload.group.speaker_id,
                            "flush_reason": payload.group.flush_reason,
                        },
                    )
                    if await host._process_turn_group(
                        payload,
                        job_dir=job_dir,
                        total_chunks=total_chunks,
                        slow_processed_indices=slow_processed_indices,
                        token=token,
                    ):
                        pause_requested = True
                    continue

                ctx = payload

                if ctx.is_end or ctx.error:
                    await self._put_queue_final_with_guard(
                        payload=ctx,
                        token=token,
                        allow_drop=True,
                        reason="upstream_end_or_error",
                    )
                    break

                chunk_index = ctx.chunk_index

                if token:
                    token.enter_atomic_region(f"slow_worker_chunk_{chunk_index}")

                try:
                    sv_result = ctx.sv_result or {}
                    chunk = ctx.audio_chunk

                    skip_whisper = False
                    whisper_result: Optional[dict] = None
                    prompt: Optional[str] = None

                    if host.is_patching_mode and sv_result:
                        if host._should_skip_whisper(sv_result, chunk):
                            skip_whisper = True
                            ctx.whisper_skipped = True
                            ctx.whisper_result = {}
                            host.logger.debug(
                                f"Chunk {chunk_index}: SenseVoice 质量足够，跳过 Whisper "
                                f"(confidence={sv_result.get('confidence', 0):.2f})"
                            )

                    if not skip_whisper:
                        sv_context = sv_result.get("text_clean", "") if sv_result else None
                        pause_gap_sec = None
                        if host._last_prompt_audio_end is not None and chunk is not None:
                            pause_gap_sec = max(
                                0.0,
                                float(chunk.start) - float(host._last_prompt_audio_end),
                            )
                        prompt = host._build_whisper_prompt(
                            sv_context,
                            pause_gap_sec=pause_gap_sec,
                        )
                        audio_with_overlap = host._extract_audio_with_overlap(ctx)

                        whisper_result = await host.slow_worker.infer(
                            audio_with_overlap,
                            initial_prompt=prompt,
                            cancel_checker=token.raise_if_canceled if token else None,
                        )
                        host._validate_l0_result(
                            whisper_result,
                            source="slow",
                            chunk_index=chunk_index,
                        )
                        whisper_text_raw = str(whisper_result.get("raw_text") or "")
                        whisper_result["text_raw"] = whisper_text_raw
                        whisper_result["prompt"] = prompt
                        base_text = whisper_result.get("min_clean_text") or whisper_text_raw
                        whisper_result["text"] = host._whisper_sanitizer.sanitize_minimal(
                            str(base_text or ""),
                            prompt=prompt,
                        )
                        if host._hallucination_detector.is_hallucination(whisper_result, prompt):
                            host.logger.warning(
                                f"Chunk {chunk_index}: 检测到 Whisper 幻觉，回退到 SenseVoice"
                            )
                            host._reset_prompt_cache(reason="hallucination")
                            fallback_text = sv_result.get("text_clean", "")
                            whisper_result["text"] = fallback_text
                            whisper_result["text_itn_raw"] = (
                                sv_result.get("text_itn_raw") or fallback_text
                            )
                            whisper_result["text_clean"] = fallback_text
                            whisper_result["language"] = sv_result.get("language", "auto")
                            whisper_result["is_hallucination"] = True
                        else:
                            whisper_language = chunk.language or whisper_result.get("language") or "auto"
                            normalized = host._text_normalizer.normalize(
                                whisper_result.get("text", ""),
                                whisper_language,
                            )
                            whisper_result["text_itn_raw"] = normalized.text_itn_raw
                            whisper_result["text_clean"] = normalized.text_clean
                            whisper_result["text"] = (
                                normalized.text_clean or whisper_result.get("text", "")
                            )
                            whisper_result["language"] = whisper_language

                        ctx.whisper_result = copy.deepcopy(whisper_result)
                        ctx.whisper_skipped = False
                        if not whisper_result.get("is_hallucination"):
                            host._update_prompt_cache(
                                whisper_result.get("text", ""),
                                confidence=whisper_result.get("confidence"),
                                whisper_result=whisper_result,
                            )
                            if chunk is not None:
                                host._last_prompt_audio_end = float(chunk.end)

                        host.logger.debug(
                            f"Chunk {chunk_index}: Whisper 推理完成 "
                            f"(text_length={len(whisper_result.get('text', ''))})"
                        )

                    await self._put_queue_final_with_guard(
                        payload=ctx,
                        token=token,
                        reason=f"chunk_{chunk_index}",
                    )
                    host._context_cache.pop(chunk_index, None)
                    slow_processed_count += 1
                    slow_processed_indices.add(chunk_index)
                    host._last_slow_chunk_index = chunk_index

                    if host.progress_emitter and total_chunks > 0:
                        total_processed = len(slow_processed_indices)
                        host.progress_emitter.update_slow(
                            total_processed,
                            total_chunks,
                            message=f"Whisper: {total_processed}/{total_chunks}",
                        )
                finally:
                    if token:
                        has_pending = token.exit_atomic_region()
                        if has_pending:
                            host.logger.debug(
                                f"[v3.1.0] SlowWorker chunk {chunk_index} 完成后检测到待处理请求"
                            )

                if token and job_dir:
                    previous_whisper_text = host.previous_whisper_text or ""
                    host._slow_processed_indices = slow_processed_indices
                    checkpoint_data = {
                        "transcription": {
                            "slow_processed_count": len(slow_processed_indices),
                            "slow_processed_indices": list(slow_processed_indices),
                            "previous_whisper_text": previous_whisper_text,
                            "last_slow_chunk_index": chunk_index,
                        }
                    }
                    try:
                        token.check_and_save(checkpoint_data, job_dir)
                    except PausedException as exc:
                        if not pause_requested:
                            host.logger.debug("[V3.1.0] SlowWorker 捕获暂停信号，继续排空 queue_inter")
                        pause_requested = True
                        if not host.pause_exception:
                            host.pause_exception = exc

            host.logger.info("SlowWorker 循环完成")

        except CancelledException as exc:
            host.logger.info(f"SlowWorker 循环取消: {exc}")
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
            await self._put_queue_final_with_guard(
                payload=error_ctx,
                token=token,
                allow_drop=True,
                reason="cancel_error_ctx",
            )

        except Exception as exc:
            host.logger.error(f"SlowWorker 循环异常: {exc}", exc_info=True)
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
            await self._put_queue_final_with_guard(
                payload=error_ctx,
                token=token,
                allow_drop=True,
                reason="exception_error_ctx",
            )
        finally:
            if pause_requested:
                host.logger.debug("[V3.1.0] SlowWorker 已完成排空，等待对齐阶段同步完成")
