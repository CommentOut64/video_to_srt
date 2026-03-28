"""
极速模式编排门面服务。

设计模式：Facade Pattern。
原因：将 `sensevoice_only` 专属编排流程从主实现类抽离，避免核心类继续膨胀。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Set

from app.schemas.pipeline_context import ProcessingContext
from app.services.audio.chunk_engine import AudioChunk
from app.utils.cancellation_token import PausedException


class SensevoiceOrchestratorService:
    """极速模式（sensevoice_only）执行门面。"""

    def __init__(self, *, host: Any) -> None:
        self._host = host

    async def run(
        self,
        *,
        audio_chunks: List[AudioChunk],
        full_audio_array: Optional[Any] = None,
        full_audio_sr: int = 16000,
        job_dir: Optional[Path] = None,
        processed_indices: Optional[Set[int]] = None,
    ) -> List[ProcessingContext]:
        """执行极速模式主流程。"""
        host = self._host
        host.logger.info(f"极速模式开始: {len(audio_chunks)} 个 Chunk")

        results: List[ProcessingContext] = []
        token = host.cancellation_token
        processed_indices = processed_indices or set()
        host._fast_processed_indices = processed_indices
        host._finalized_indices = processed_indices
        total_chunks = len(audio_chunks)
        host._audio_chunks_by_index = {chunk.index: chunk for chunk in audio_chunks}
        host._decision_processor.reset_state()
        host._soft_cut_pending_deferred_by_stream.clear()

        for i, chunk in enumerate(audio_chunks):
            if i in processed_indices:
                host.logger.debug(f"跳过已处理的 chunk {i}")
                continue

            ctx = ProcessingContext(
                job_id=host.job_id,
                chunk_index=i,
                audio_chunk=chunk,
                job_dir=job_dir,
                debug_punctuation=host.debug_punctuation,
                edge_selection_mode=getattr(host, "_edge_selection_mode", "auto"),
                full_audio_array=full_audio_array,
                full_audio_sr=full_audio_sr,
            )

            if token:
                token.enter_atomic_region(f"fast_chunk_{i}")

            try:
                await host.fast_worker.process(ctx)
                if ctx.sv_result:
                    host._validate_l0_result(ctx.sv_result, source="fast", chunk_index=i)
                normalized = host._normalize_sensevoice_result(ctx)
                await host._apply_fast_punctuation(ctx, normalized)
                ctx.whisper_skipped = True
                await host._run_alignment_stage(ctx)
                results.append(ctx)
                if host.progress_emitter:
                    processed_count = len(results)
                    host.progress_emitter.update_fast(
                        processed_count,
                        total_chunks,
                        message=f"SenseVoice: {processed_count}/{total_chunks}",
                    )
            except Exception as exc:
                host.logger.error(f"Chunk {i} 处理失败: {exc}", exc_info=True)
                host.errors.append(exc)
            finally:
                if token:
                    has_pending = token.exit_atomic_region()
                    if has_pending:
                        host.logger.debug(f"[v3.1.0] Chunk {i} 处理完成后检测到待处理请求")

            if token and job_dir:
                processed_indices.add(i)
                host._last_align_chunk_index = i

                subtitle_checkpoint_data = {}
                if host.subtitle_manager:
                    subtitle_checkpoint_data = host.subtitle_manager.to_checkpoint_data()

                checkpoint_data = {
                    "transcription": {
                        "mode": "sensevoice_only",
                        "processed_indices": list(processed_indices),
                        "processed_count": len(processed_indices),
                        "total_chunks": len(audio_chunks),
                        "finalized_indices": list(processed_indices),
                        **subtitle_checkpoint_data,
                    }
                }
                try:
                    token.check_and_save(checkpoint_data, job_dir)
                except PausedException as exc:
                    if not host.pause_exception:
                        host.pause_exception = exc
                    host.logger.debug(
                        f"[V3.1.0] 极速模式捕获暂停信号，已处理 {len(processed_indices)} / {total_chunks} 个 Chunk"
                    )
                    break

        if host.errors:
            host.logger.error(f"极速模式执行中发生 {len(host.errors)} 个错误")
            raise host.errors[0]

        if host.pause_exception:
            host.is_pause_snapshot_saved = host._force_save_pause_checkpoint(job_dir, total_chunks)
            raise host.pause_exception

        host.logger.info(f"极速模式完成: {len(results)} 个 Chunk 已处理")
        return results

