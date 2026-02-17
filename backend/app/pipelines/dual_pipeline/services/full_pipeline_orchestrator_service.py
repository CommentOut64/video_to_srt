"""
完整双流编排门面服务。

设计模式：Facade Pattern。
原因：将 `_run_full_pipeline` 的任务编排职责从实现类中抽离，主类仅保留调度入口与依赖注入。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

from app.schemas.pipeline_context import ProcessingContext
from app.services.audio.chunk_engine import AudioChunk


class FullPipelineOrchestratorService:
    """复核/双流模式完整编排入口。"""

    def __init__(self, *, host: Any) -> None:
        self._host = host

    async def run(
        self,
        *,
        audio_chunks: List[AudioChunk],
        full_audio_array: Optional[Any] = None,
        full_audio_sr: int = 16000,
        job_dir: Optional[Path] = None,
        vad_intervals: Optional[List[Tuple[float, float]]] = None,
        processed_indices: Optional[Set[int]] = None,
        base_slow_count: int = 0,
        base_align_count: int = 0,
        initial_slow_processed_indices: Optional[set] = None,
        initial_finalized_indices: Optional[set] = None,
    ) -> List[ProcessingContext]:
        """执行完整三级流水线（复核/双流模式）。"""
        host = self._host

        # V3.1.0: 清理历史状态，避免重复抛出旧异常
        host.errors.clear()
        host.pause_exception = None
        if host._is_dual_time_experiment_enabled:
            host._dual_time_compare_accumulator = {
                "chunk_count": 0,
                "boundary_precision_sum": 0.0,
                "boundary_recall_sum": 0.0,
                "boundary_f1_sum": 0.0,
                "word_start_mae_ms_sum": 0.0,
                "word_end_mae_ms_sum": 0.0,
                "sentence_start_mae_ms_sum": 0.0,
                "sentence_end_mae_ms_sum": 0.0,
                "selected_experiment_count": 0,
            }
            host._dual_time_legacy_sentences_by_chunk.clear()
            host._dual_time_experiment_sentences_by_chunk.clear()

        # V3.2.0+dev.20260201.07: 记录完整音频与 Chunk 映射，供 Bridge 批次使用
        host._full_audio_array = full_audio_array
        host._full_audio_sr = full_audio_sr
        host._audio_chunks_by_index = {chunk.index: chunk for chunk in audio_chunks}
        host._context_cache = {}
        host._vad_intervals = list(vad_intervals) if vad_intervals else None
        host._l6_processor.reset_state()
        host._soft_cut_pending_deferred_by_stream.clear()

        # Phase 2: S-Epoch 预构建 Timeline 映射（speaker_id/turn_id）。
        await host._prepare_timeline_domain(
            audio_chunks=audio_chunks,
            full_audio_array=full_audio_array,
            full_audio_sr=full_audio_sr,
            job_dir=job_dir,
        )

        total_chunks = len(audio_chunks)
        host.logger.info(f"开始三级流水线: {total_chunks} 个 Chunk")

        results: List[ProcessingContext] = []
        processed_indices = processed_indices or set()
        host._fast_processed_indices = processed_indices

        task_fast = asyncio.create_task(
            host._fast_loop(
                audio_chunks,
                full_audio_array,
                full_audio_sr,
                job_dir,
                processed_indices,
                total_chunks,
            )
        )
        task_slow = asyncio.create_task(
            host._slow_loop(
                job_dir,
                total_chunks,
                base_slow_count,
                initial_slow_processed_indices,
            )
        )
        task_align = asyncio.create_task(
            host._align_loop(
                results,
                job_dir,
                total_chunks,
                base_align_count,
                initial_finalized_indices,
            )
        )

        await_results = await asyncio.gather(
            task_fast,
            task_slow,
            task_align,
            return_exceptions=True,
        )

        for res in await_results:
            if isinstance(res, Exception):
                host.logger.error(f"任务异常: {res}", exc_info=res)
                raise res

        if host.errors:
            host.logger.error(f"流水线执行中发生 {len(host.errors)} 个错误")
            raise host.errors[0]

        if host.pause_exception:
            host.is_pause_snapshot_saved = host._force_save_pause_checkpoint(job_dir, total_chunks)
            # V3.1.0: 等待队列排空后再通知上层暂停，避免进度回退
            raise host.pause_exception

        expected_chunk_indices = {int(chunk.index) for chunk in audio_chunks}
        finalized_indices = {
            int(idx)
            for idx in (getattr(host, "_finalized_indices", set()) or set())
        }
        missing_indices = sorted(expected_chunk_indices - finalized_indices)
        if missing_indices:
            subtitle_manager = getattr(host, "subtitle_manager", None)
            fast_processed_indices = {
                int(idx)
                for idx in (getattr(host, "_fast_processed_indices", set()) or set())
            }
            chunk_sentence_map = (
                dict(getattr(subtitle_manager, "chunk_sentences", {}) or {})
                if subtitle_manager is not None
                else {}
            )

            recoverable_missing = [
                idx
                for idx in missing_indices
                if idx in fast_processed_indices or idx in chunk_sentence_map
            ]
            unrecoverable_missing = sorted(set(missing_indices) - set(recoverable_missing))
            if unrecoverable_missing:
                raise RuntimeError(
                    "三级流水线定稿完整性校验失败: "
                    f"存在未进入定稿链的 chunk={unrecoverable_missing}"
                )

            recovered_missing: List[int] = []
            failed_recover_missing: List[int] = []
            for chunk_index in recoverable_missing:
                try:
                    if subtitle_manager is None:
                        failed_recover_missing.append(chunk_index)
                        continue
                    subtitle_manager.replace_chunk(chunk_index, [])
                    finalized_indices.add(chunk_index)
                    recovered_missing.append(chunk_index)
                except Exception as recover_exc:
                    host.logger.error(
                        f"三级流水线尾部收口失败: chunk={chunk_index} error={recover_exc}"
                    )
                    failed_recover_missing.append(chunk_index)

            host._finalized_indices = finalized_indices
            remaining_missing = sorted(expected_chunk_indices - finalized_indices)
            if failed_recover_missing or remaining_missing:
                raise RuntimeError(
                    "三级流水线定稿完整性校验失败: "
                    f"recover_failed={failed_recover_missing}, remaining_missing={remaining_missing}"
                )

            host.logger.warning(
                f"三级流水线完成前执行尾部收口: missing={missing_indices} "
                f"recovered={recovered_missing}"
            )

        host.logger.info(f"三级流水线完成: {len(results)} 个 Chunk 已处理")
        return results
