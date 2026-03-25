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
                run_result = host._finalize_sensevoice_only(ctx)
                final_sentences = list(run_result.final_sentences)
                host._assign_sentence_identity_by_timeline_overlap(
                    final_sentences,
                    fallback_chunk=ctx.audio_chunk,
                )
                output_traces = host._normalize_output_traces_for_sentences(
                    final_sentences=final_sentences,
                    output_traces=list(run_result.output_traces or []),
                    default_reason="sensevoice_only",
                )
                ctx.final_sentences = final_sentences
                injection_stats = dict(run_result.injection_stats)
                split_stats = dict(run_result.split_stats)
                soft_cut_observe_snapshot = host._update_soft_cut_observability(
                    split_stats=split_stats,
                    chunk_index=ctx.chunk_index,
                    stage="sensevoice_orchestrator",
                )
                alignment_result = run_result.alignment_result
                aligned_facts = run_result.aligned_facts
                fused_evidence = run_result.fused_evidence
                ctx.finalization_metrics = {
                    "coverage": alignment_result.coverage,
                    "gap_ratio": alignment_result.gap_ratio,
                    "alignment_score": alignment_result.alignment_score,
                    "gap_positions": list(alignment_result.gap_positions),
                    "gap_resolution": (
                        alignment_result.resolution.value if alignment_result.resolution else None
                    ),
                    **injection_stats,
                }
                for key, value in split_stats.items():
                    ctx.finalization_metrics[f"split_{key}"] = value
                for key, value in soft_cut_observe_snapshot.items():
                    ctx.finalization_metrics[f"soft_cut_obs_{key}"] = value
                ctx.finalization_metrics["fact_word_count"] = float(len(aligned_facts.annotated_words))
                ctx.finalization_metrics["fact_turn_count"] = float(len(aligned_facts.speaker_turns))
                ctx.finalization_metrics["fact_mapping_count"] = float(len(aligned_facts.time_mappings))
                ctx.finalization_metrics["evidence_speaker_change_count"] = float(
                    len(fused_evidence.speaker_changes)
                )
                ctx.finalization_metrics["evidence_pause_anchor_count"] = float(
                    len(fused_evidence.pause_anchors)
                )
                ctx.finalization_metrics["evidence_semantic_anchor_count"] = float(
                    len(fused_evidence.semantic_anchors)
                )
                ctx.finalization_metrics["evidence_punctuation_anchor_count"] = float(
                    len(fused_evidence.punctuation_anchors)
                )
                output_layer_result = host._emit_output_layer(
                    chunk_index=ctx.chunk_index,
                    sentence_segments=final_sentences,
                    language=str(run_result.detected_language or "auto"),
                    injection_report={
                        "mapping_coverage": float(
                            injection_stats.get("injection_mapping_coverage", 0.0)
                        ),
                        "mismatch_count": float(
                            injection_stats.get("injection_unmatched_total", 0.0)
                        ),
                        "error_code": str(injection_stats.get("injection_error_code", "") or ""),
                        "blocked": float(injection_stats.get("injection_blocked", 0.0)),
                    },
                    segmentation_report={
                        "boundary_score_stats": dict(split_stats),
                        "forced_split_count": float(split_stats.get("force_split_count", 0.0)),
                        "error_code": str(split_stats.get("error_code", "") or ""),
                    },
                    output_traces=output_traces,
                    default_trace_reason="sensevoice_only",
                    subtitle_batch=run_result.subtitle_batch,
                )
                ctx.finalization_metrics["l7_error_count"] = float(
                    len(output_layer_result.output_payload.get("errors", []))
                )
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

