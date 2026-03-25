"""
对齐阶段门面服务。

设计模式：Facade Pattern。
原因：收口对齐阶段的大段编排逻辑，让主实现类聚焦流水线调度与依赖装配。
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict
from typing import Any, Dict, Optional, Sequence

from app.models.sensevoice_models import SentenceSegment, TextSource
from app.pipelines.dual_pipeline.services.textflow_facade_service import Layer456RunResult
from app.schemas.pipeline_context import ProcessingContext
from app.services.alignment.types import PunctTrack
from app.services.language_policy import build_language_policy_snapshot
from app.services.textflow.decision_ingress_adapter import DecisionIngressAdapter
from app.services.timeanchored_alignment import AlignmentItem
from app.services.timeanchored_alignment.anchor_mount.service import (
    AnchorMountAlignmentService,
    AnchorMountStageResult,
)
from app.services.timeanchored_alignment.preparation import (
    AlignmentPreparationAssembler,
    AlignmentPreparationPackage,
)
from app.services.timeanchored_alignment.sentence_segmenter import SentenceSegmenter
from app.services.timeanchored_alignment.slow_window.contracts import (
    DialogueShapeSnapshot,
    PromptSeed,
    ReadySlowWindow,
    WindowBatchHint,
    WindowChunkBinding,
    WindowCoverage,
    WindowLanguageProfile,
    WindowSourceUnit,
)
from app.services.timeanchored_alignment.window_time_base_assembler import (
    WindowTimeBaseAssembler,
    WindowTimeBasePackage,
)


class AlignmentStageService:
    """双流对齐阶段执行门面。"""

    def __init__(self, *, host: Any) -> None:
        self._host = host
        self._timeanchored_preparation_assembler = AlignmentPreparationAssembler(
            logger=getattr(host, "logger", None),
            sanitizer=getattr(host, "_whisper_sanitizer", None),
            hallucination_detector=getattr(host, "_hallucination_detector", None),
        )
        self._timeanchored_sentence_segmenter = SentenceSegmenter()
        self._timeanchored_stage_service = AnchorMountAlignmentService()
        self._decision_ingress_adapter = DecisionIngressAdapter()

    async def run(self, ctx: ProcessingContext) -> None:
        """执行单个 chunk 的对齐阶段。"""
        host = self._host
        chunk = ctx.audio_chunk

        fast_direct_reason = self._resolve_fast_direct_reason(ctx=ctx)
        if fast_direct_reason:
            if ctx.sv_result is None:
                raise ValueError("对齐阶段缺少 SenseVoice 推理结果")
            self._commit_fast_direct_result(
                ctx=ctx,
                sv_result=ctx.sv_result,
                reason=fast_direct_reason,
            )
            return

        # V3.10: 快速路径 - SlowWorker 跳过时直接使用 SenseVoice
        if ctx.whisper_skipped:
            if ctx.sv_result is None:
                raise ValueError("对齐阶段缺少 SenseVoice 推理结果")
            host.logger.debug(
                f"Chunk {ctx.chunk_index}: Whisper 跳过，直接使用 SenseVoice 定稿"
            )
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
            alignment_result = run_result.alignment_result
            aligned_facts = run_result.aligned_facts
            fused_evidence = run_result.fused_evidence
            injection_stats = dict(run_result.injection_stats)
            split_stats = dict(run_result.split_stats)
            soft_cut_observe_snapshot = host._update_soft_cut_observability(
                split_stats=split_stats,
                chunk_index=ctx.chunk_index,
                stage="sensevoice_only_alignment_stage",
            )
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

            # 输出层薄层分发：即便走快路，也统一通过输出层入口。
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

            host.logger.debug(
                f"Chunk {ctx.chunk_index}: SenseVoice 定稿已推送 "
                f"({len(final_sentences)} 个句子) [智能复核-跳过]"
            )
            return

        # 阶段 1: 双流定稿主链（前置仲裁/标点域 + 四层主链）
        host.logger.debug(f"Chunk {ctx.chunk_index}: 双流对齐")

        if ctx.whisper_result is None or ctx.sv_result is None:
            raise ValueError("对齐阶段缺少必要的推理结果")

        whisper_result = ctx.whisper_result
        sv_result = ctx.sv_result

        # V3.2.0+dev.20260202.07: Whisper 清洗增强 + 规范化重算
        host._apply_whisper_full_sanitize(ctx)

        arbitration_output = host._run_arbitration(ctx, sv_result, whisper_result)
        arbitration_result = arbitration_output.arbitration_result
        ctx.arbitration_result = arbitration_result
        tracks = host._ensure_text_tracks(ctx)
        self._apply_arbitration_track_selection(
            host=host,
            chunk_index=ctx.chunk_index,
            tracks=tracks,
            arbitration_output=arbitration_output,
        )

        chosen_text_clean = host._select_text_for_alignment(tracks.chosen_track)
        host._apply_arbitration_text(
            whisper_result,
            sv_result,
            arbitration_result.chosen_source,
            chosen_text_clean,
        )
        # V3.2.0+dev.20260204.10: 删除慢流补跑分支，定稿标点仅走统一前置入口
        punct_track: Optional[PunctTrack] = None
        # V3.2.0+dev.20260204.11: 若仲裁选择 fast 且快流已产出同一 clean_text_ref 的 PunctTrack，直接复用避免重复跑模型
        if (
            arbitration_result.chosen_source == "fast"
            and ctx.punct_track is not None
            and tracks.chosen_track is not None
            and ctx.punct_track.clean_text_ref == tracks.chosen_track.text_clean
        ):
            punct_track = ctx.punct_track
        else:
            punct_track = await host._run_punctuation_layer(
                ctx,
                sv_result,
                whisper_result,
                arbitration_result.chosen_source,
            )
        ctx.punct_track = punct_track
        punctuation_positions = punct_track.positions if punct_track and punct_track.positions else None
        punctuation_clean_text = punct_track.clean_text_ref if punct_track else None
        if tracks.chosen_track:
            if chosen_text_clean and tracks.chosen_track.text_clean != chosen_text_clean:
                tracks.chosen_track.text_clean = chosen_text_clean
            if (
                punctuation_positions
                and punctuation_clean_text
                and punctuation_clean_text == tracks.chosen_track.text_clean
            ):
                tracks.chosen_track.punct_positions = list(punctuation_positions)
            else:
                if punctuation_positions:
                    host.logger.debug("评分层标点 clean_text 不一致，清空回写位置")
                tracks.chosen_track.punct_positions = []
        if host.bridge_controller:
            host.bridge_controller.record_arbitration_result(arbitration_result)
        host.logger.debug(
            "Chunk {}: 选文完成 chosen_source={} reason={} coverage={:.2f}",
            ctx.chunk_index,
            arbitration_result.chosen_source,
            arbitration_result.reason,
            arbitration_result.coverage,
        )

        # V3.2.0+dev.20260205.09: 集合层（仅对齐与 gap 修复）
        sv_words = host._build_sv_word_timestamps(sv_result, chunk)
        speaker_id = host._resolve_speaker_id_for_chunk(ctx.audio_chunk) if ctx.audio_chunk else None
        turn_id = host._resolve_turn_id_for_chunk(ctx.audio_chunk) if ctx.audio_chunk else None
        policy_snapshot = None
        language_hint = str(
            (tracks.chosen_track.language if tracks.chosen_track else "")
            or whisper_result.get("language")
            or getattr(chunk, "language", "")
            or "auto"
        )
        try:
            policy_snapshot = build_language_policy_snapshot(language_hint=language_hint)
            host.logger.debug(
                "语言策略快照注入: chunk={} language_tag={} policy_version={}",
                ctx.chunk_index,
                policy_snapshot.language_tag,
                policy_snapshot.policy_version,
            )
        except Exception:
            host.logger.exception(
                "语言策略快照编译失败，回退空快照注入: chunk={} language_hint={}",
                ctx.chunk_index,
                language_hint,
            )

        alignment_pipeline_mode = self._resolve_alignment_pipeline_mode()
        legacy_experiment_enabled = (
            alignment_pipeline_mode == "legacy"
            and bool(host._is_dual_time_experiment_enabled)
        )
        ctx.hetero_alignment_result = None
        ctx.hetero_alignment_report = None
        ctx.hetero_route = None

        timeanchored_available = self._is_timeanchored_main_chain_enabled(ctx)
        if alignment_pipeline_mode == "shadow":
            if timeanchored_available and self._should_sample_alignment_pipeline_shadow(ctx):
                stage_result = self._run_timeanchored_main_chain(
                    ctx=ctx,
                    tracks=tracks,
                    sv_result=sv_result,
                    whisper_result=whisper_result,
                    language_hint=language_hint,
                    speaker_id=speaker_id,
                    turn_id=turn_id,
                )
                self._record_hetero_alignment_result(
                    ctx=ctx,
                    stage_result=stage_result,
                    mode=alignment_pipeline_mode,
                    selected=False,
                    reason=(
                        "shadow_observed"
                        if stage_result is not None
                        else "shadow_timeanchored_failed"
                    ),
                )
            else:
                shadow_reason = (
                    "shadow_sample_skipped"
                    if timeanchored_available
                    else "shadow_time_base_unavailable"
                )
                self._record_hetero_alignment_result(
                    ctx=ctx,
                    stage_result=None,
                    mode=alignment_pipeline_mode,
                    selected=False,
                    reason=shadow_reason,
                )
        elif alignment_pipeline_mode in {"active", "default"}:
            if timeanchored_available:
                stage_result = self._run_timeanchored_main_chain(
                    ctx=ctx,
                    tracks=tracks,
                    sv_result=sv_result,
                    whisper_result=whisper_result,
                    language_hint=language_hint,
                    speaker_id=speaker_id,
                    turn_id=turn_id,
                )
                should_accept, reason = self._should_accept_timeanchored_result(
                    stage_result=stage_result,
                    mode=alignment_pipeline_mode,
                    ctx=ctx,
                )
                if should_accept and stage_result is not None:
                    self._commit_timeanchored_main_chain_result(
                        ctx=ctx,
                        stage_result=stage_result,
                        whisper_result=whisper_result,
                        sv_result=sv_result,
                        speaker_id=speaker_id,
                        turn_id=turn_id,
                    )
                    self._record_hetero_alignment_result(
                        ctx=ctx,
                        stage_result=stage_result,
                        mode=alignment_pipeline_mode,
                        selected=True,
                        reason=reason,
                    )
                    return
                self._record_hetero_alignment_result(
                    ctx=ctx,
                    stage_result=stage_result,
                    mode=alignment_pipeline_mode,
                    selected=False,
                    reason=reason,
                )
            else:
                self._record_hetero_alignment_result(
                    ctx=ctx,
                    stage_result=None,
                    mode=alignment_pipeline_mode,
                    selected=False,
                    reason=f"{alignment_pipeline_mode}_time_base_unavailable",
                )

        # 旧四层 legacy 已下线：新主链未被接受时直接失败，避免静默回退。
        hetero_reason = str(
            (ctx.hetero_alignment_result or {}).get("reason")
            or "timeanchored_not_selected"
        )
        raise RuntimeError(
            f"Chunk {ctx.chunk_index}: legacy_alignment_pipeline_disabled reason={hetero_reason}"
        )

        legacy_run = host._run_collection_scoring_decision_once(
            tracks=tracks,
            sv_result=sv_result,
            whisper_result=whisper_result,
            sv_words=sv_words,
            punctuation_positions=punctuation_positions,
            punctuation_clean_text=punctuation_clean_text,
            variant="legacy",
            speaker_id=speaker_id,
            turn_id=turn_id,
            policy_snapshot=policy_snapshot,
        )

        experiment_run: Optional[Layer456RunResult] = None
        selected_variant = "legacy"
        selected_reason = (
            "shadow_default" if host._dual_time_mode == "shadow" else "legacy_default"
        )
        compare_payload: Optional[Dict[str, Any]] = None

        if legacy_experiment_enabled:
            # Shadow/Active 都采用串行后处理：先 legacy，再 experiment，不并行占用 GPU。
            experiment_run = host._run_collection_scoring_decision_once(
                tracks=tracks,
                sv_result=sv_result,
                whisper_result=whisper_result,
                sv_words=sv_words,
                punctuation_positions=punctuation_positions,
                punctuation_clean_text=punctuation_clean_text,
                variant="experiment",
                speaker_id=speaker_id,
                turn_id=turn_id,
                policy_snapshot=policy_snapshot,
            )
            compare_payload = host._build_dual_time_compare_payload(
                ctx=ctx,
                legacy_run=legacy_run,
                experiment_run=experiment_run,
            )
            if host._dual_time_mode == "active":
                selected_variant, selected_reason = host._select_dual_time_variant(
                    compare_payload=compare_payload,
                    experiment_run=experiment_run,
                )
            else:
                selected_variant = "legacy"
                selected_reason = "shadow_force_legacy"
            compare_payload["selected_variant"] = selected_variant
            compare_payload["selected_reason"] = selected_reason
            host._append_dual_time_compare_debug(ctx.job_dir, compare_payload)
            host._update_dual_time_summary(ctx, compare_payload)
            if host._is_dual_time_write_debug_srt:
                host._dual_time_legacy_sentences_by_chunk[ctx.chunk_index] = copy.deepcopy(
                    legacy_run.final_sentences
                )
                host._dual_time_experiment_sentences_by_chunk[ctx.chunk_index] = copy.deepcopy(
                    experiment_run.final_sentences
                )
                host._write_dual_time_debug_srt(ctx.job_dir)

        run_result = legacy_run
        if selected_variant == "experiment" and experiment_run is not None:
            run_result = experiment_run

        host._maybe_record_m2_stage0_sample(
            ctx=ctx,
            selected_variant=selected_variant,
            selected_reason=selected_reason,
            legacy_run=legacy_run,
            experiment_run=experiment_run,
            active_run=run_result,
            compare_payload=compare_payload,
        )

        alignment_result = run_result.alignment_result
        aligned_facts = run_result.aligned_facts
        fused_evidence = run_result.fused_evidence
        words_for_split = run_result.words_for_split
        injection_stats = dict(run_result.injection_stats)
        split_stats = dict(run_result.split_stats)
        soft_cut_observe_snapshot = host._update_soft_cut_observability(
            split_stats=split_stats,
            chunk_index=ctx.chunk_index,
            stage="dual_or_patch_alignment_stage",
        )
        final_sentences = run_result.final_sentences
        output_traces = list(run_result.output_traces or [])
        host._assign_sentence_identity_by_timeline_overlap(
            final_sentences,
            fallback_chunk=ctx.audio_chunk,
        )
        output_traces = host._normalize_output_traces_for_sentences(
            final_sentences=final_sentences,
            output_traces=output_traces,
            default_reason="post_identity_binding",
        )

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
        ctx.finalization_metrics["fact_time_mapping_enabled"] = (
            1.0 if host._is_m2_time_mapping_enabled else 0.0
        )
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

        if compare_payload is not None:
            ctx.finalization_metrics["dual_time_mode"] = host._dual_time_mode
            ctx.finalization_metrics["dual_time_selected_variant"] = selected_variant
            ctx.finalization_metrics["dual_time_selected_reason"] = selected_reason
            ctx.finalization_metrics["dual_time_boundary_f1"] = compare_payload["comparison"][
                "boundary_f1"
            ]
            ctx.finalization_metrics["dual_time_word_start_mae_ms"] = compare_payload[
                "comparison"
            ]["word_start_mae_ms"]
            ctx.finalization_metrics["dual_time_legacy_time_source"] = compare_payload[
                "legacy"
            ]["alignment_time_source"]
            ctx.finalization_metrics["dual_time_experiment_time_source"] = compare_payload[
                "experiment"
            ]["alignment_time_source"]

        if tracks.chosen_track:
            mapping_cov = split_stats.get("mapping_coverage")
            if mapping_cov is not None:
                tracks.chosen_track.mapping_coverage = float(mapping_cov)
            ctx.finalization_metrics["itn_fallback"] = (
                1.0 if tracks.chosen_track.itn_fallback else 0.0
            )
        if tracks.sv_track:
            ctx.finalization_metrics["sv_itn_fallback"] = (
                1.0 if tracks.sv_track.itn_fallback else 0.0
            )
        if tracks.whisper_track:
            ctx.finalization_metrics["whisper_itn_fallback"] = (
                1.0 if tracks.whisper_track.itn_fallback else 0.0
            )

        host._emit_layer_diagnostics(
            ctx,
            tracks=tracks,
            punct_track=punct_track,
            alignment_result=alignment_result,
            injection_stats=injection_stats,
            split_stats=split_stats,
            final_sentences=final_sentences,
        )
        host._emit_layer_trace_full(
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
        )

        # V3.2.0+dev.20260204.08: 集合/评分域补跑候选埋点（仅统计，不触发补跑）
        host._record_punct_retry_candidates(ctx)

        ctx.final_sentences = final_sentences
        if ctx.arbitration_result:
            ctx.arbitration_result.gap_positions = list(alignment_result.gap_positions)
            host.logger.debug(
                "Chunk {}: 选文统计 coverage={:.2f} gap_positions={}",
                ctx.chunk_index,
                ctx.arbitration_result.coverage,
                ctx.arbitration_result.gap_positions,
            )

        # 阶段 2: 推送定稿（使用 Chunk 级别的批量替换）
        host.logger.debug(
            f"Chunk {ctx.chunk_index}: 对齐完成 - "
            f"final_sentences={len(final_sentences)}, "
            f"whisper_text_len={len(whisper_result.get('text', ''))}, "
            f"sv_text_clean_len={len(sv_result.get('text_clean', ''))}"
        )

        # 定稿链内兜底：若裁决层仍未产出句子，基于裁决层输入词流构建单句，禁止回读草稿链。
        if not final_sentences:
            host.logger.error(
                f"Chunk {ctx.chunk_index}: 定稿句子为空！"
                f"Whisper文本长度={len(whisper_result.get('text', ''))}, "
                f"SenseVoice文本长度={len(sv_result.get('text_clean', ''))}, "
                f"words_for_split={len(words_for_split)}, "
                f"chosen_text_len={len(chosen_text_clean)}"
            )
            fallback_sentence = host._build_final_fallback_sentence(words_for_split)
            fallback_error_code = "E_DECISION_SPLIT_EMPTY"
            if fallback_sentence is None:
                fallback_text = self._resolve_text_fallback_content(
                    chosen_text_clean=chosen_text_clean,
                    tracks=tracks,
                    whisper_result=whisper_result,
                    sv_result=sv_result,
                )
                fallback_confidence = self._resolve_text_fallback_confidence(
                    chosen_source=arbitration_result.chosen_source,
                    whisper_result=whisper_result,
                    sv_result=sv_result,
                )
                fallback_sentence = self._build_text_fallback_sentence(
                    text=fallback_text,
                    chunk=ctx.audio_chunk,
                    confidence=fallback_confidence,
                )
                fallback_error_code = "E_DECISION_SPLIT_EMPTY_TEXT_FALLBACK"
            if fallback_sentence is not None:
                fallback_sentence.speaker_id = speaker_id
                fallback_sentence.turn_id = turn_id
                fallback_sentence.confidence_source = (
                    "slow" if arbitration_result.chosen_source == "slow" else "fast"
                )
                final_sentences = [fallback_sentence]
                host._assign_sentence_identity_by_timeline_overlap(
                    final_sentences,
                    fallback_chunk=ctx.audio_chunk,
                )
                ctx.final_sentences = final_sentences
                split_stats["error_code"] = fallback_error_code
                host.logger.warning(
                    "Chunk {}: 触发定稿链内单句兜底（禁止草稿回退） error_code={}",
                    ctx.chunk_index,
                    fallback_error_code,
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
            default_trace_reason="default_splitter",
            subtitle_batch=run_result.subtitle_batch,
        )
        ctx.finalization_metrics["l7_error_count"] = float(
            len(output_layer_result.output_payload.get("errors", []))
        )
        if ctx.hetero_alignment_result is not None:
            ctx.finalization_metrics["alignment_pipeline_mode"] = str(
                ctx.hetero_alignment_result.get("mode", "")
            )
            ctx.finalization_metrics["alignment_pipeline_selected"] = (
                1.0 if bool(ctx.hetero_alignment_result.get("selected")) else 0.0
            )
            ctx.finalization_metrics["alignment_pipeline_reason"] = str(
                ctx.hetero_alignment_result.get("reason", "")
            )
        else:
            ctx.hetero_route = "legacy"
            ctx.hetero_alignment_result = {
                "mode": "legacy",
                "selected": True,
                "reason": "legacy_only",
                "route": "legacy",
                "sentence_count": int(len(final_sentences)),
                "failed_span_count": 0,
            }
            ctx.hetero_alignment_report = None
            ctx.finalization_metrics["alignment_pipeline_mode"] = "legacy"
            ctx.finalization_metrics["alignment_pipeline_selected"] = 1.0
            ctx.finalization_metrics["alignment_pipeline_reason"] = "legacy_only"

        host.logger.debug(
            f"Chunk {ctx.chunk_index}: 定稿已推送 "
            f"({len(final_sentences)} 个句子)"
        )

    @staticmethod
    def _is_timeanchored_main_chain_enabled(ctx: ProcessingContext) -> bool:
        if ctx.time_base_chunk is None:
            return False
        if ctx.whisper_result is None:
            return False
        return True

    def _try_run_timeanchored_main_chain(
        self,
        *,
        ctx: ProcessingContext,
        tracks: Any,
        sv_result: Dict[str, Any],
        whisper_result: Dict[str, Any],
        language_hint: str,
        speaker_id: Optional[str],
        turn_id: Optional[str],
    ) -> bool:
        stage_result = self._run_timeanchored_main_chain(
            ctx=ctx,
            tracks=tracks,
            sv_result=sv_result,
            whisper_result=whisper_result,
            language_hint=language_hint,
            speaker_id=speaker_id,
            turn_id=turn_id,
        )
        should_accept, _ = self._should_accept_timeanchored_result(
            stage_result=stage_result,
            mode="default",
            ctx=ctx,
        )
        if not should_accept or stage_result is None:
            return False
        self._commit_timeanchored_main_chain_result(
            ctx=ctx,
            stage_result=stage_result,
            whisper_result=whisper_result,
            sv_result=sv_result,
            speaker_id=speaker_id,
            turn_id=turn_id,
        )
        return True

    def _run_timeanchored_main_chain(
        self,
        *,
        ctx: ProcessingContext,
        tracks: Any,
        sv_result: Dict[str, Any],
        whisper_result: Dict[str, Any],
        language_hint: str,
        speaker_id: Optional[str],
        turn_id: Optional[str],
    ) -> Optional[AnchorMountStageResult]:
        host = self._host
        try:
            base_language = str(language_hint or "auto")
            fallback_text = self._resolve_text_fallback_content(
                chosen_text_clean=host._select_text_for_alignment(
                    getattr(tracks, "chosen_track", None)
                ),
                tracks=tracks,
                whisper_result=whisper_result,
                sv_result=sv_result,
            )
            ready_window, window_time_base = self._resolve_preparation_inputs(
                ctx=ctx,
                whisper_result=whisper_result,
                fallback_text=fallback_text,
                language_hint=base_language,
                speaker_id=speaker_id,
                turn_id=turn_id,
            )
            host.logger.debug(
                "Chunk {}: preparation 输入 window_id={} source_chunks={} source_units={} time_base_raw_units={} time_base_word_units={} fallback_text_len={}",
                ctx.chunk_index,
                ready_window.window_id,
                len(ready_window.source_chunk_ids),
                len(ready_window.source_units),
                len(window_time_base.raw_units),
                len(window_time_base.word_units),
                len(fallback_text),
            )
            preparation = self._timeanchored_preparation_assembler.prepare(
                ready_window=ready_window,
                window_time_base=window_time_base,
                whisper_result=whisper_result,
                default_language=base_language,
                fallback_text=fallback_text,
            )
            self._commit_preparation_context(ctx=ctx, preparation=preparation)

            return self._execute_prepared_timeanchored_stage(
                preparation=preparation,
                language=base_language,
            )
        except Exception:
            host.logger.exception(
                "Chunk {}: timeanchored 主链失败（legacy 已下线）",
                ctx.chunk_index,
            )
            return None

    def _execute_prepared_timeanchored_stage(
        self,
        *,
        preparation: AlignmentPreparationPackage,
        language: str,
    ) -> AnchorMountStageResult:
        self._host.logger.debug(
            "Chunk {}: preparation 执行 anchor_mount window_id={} slots={} fast_hooks={} punctuation_evidences={}",
            preparation.owner_chunk_index,
            preparation.window_id,
            len(preparation.slow_text.slots),
            len(preparation.fast_hooks),
            len(preparation.slow_text.punctuation_evidences),
        )
        return self._timeanchored_stage_service.execute(
            preparation=preparation,
            language=language,
        )

    def _commit_preparation_context(
        self,
        *,
        ctx: ProcessingContext,
        preparation: AlignmentPreparationPackage,
    ) -> None:
        ctx.alignment_preparation = preparation
        ctx.slow_window_meta = {
            "window_id": preparation.window_id,
            "owner_chunk_id": preparation.owner_chunk_id,
            "slot_count": len(preparation.slow_text.slots),
            "hook_count": len(preparation.fast_hooks),
            "source_chunk_ids": list(preparation.source_chunk_ids),
        }
        if isinstance(preparation.compat.time_base, WindowTimeBasePackage):
            ctx.window_time_base = preparation.compat.time_base
        self._host.logger.debug(
            "Chunk {}: preparation 已写回 ctx window_id={} slot_count={} hook_count={} pronunciation_hint_count={} window_time_base_units={}",
            ctx.chunk_index,
            preparation.window_id,
            len(preparation.slow_text.slots),
            len(preparation.fast_hooks),
            len(preparation.slow_text.pronunciation_hints),
            len(getattr(preparation.compat.time_base, "word_units", ()) or ()),
        )

    def _resolve_preparation_inputs(
        self,
        *,
        ctx: ProcessingContext,
        whisper_result: Dict[str, Any],
        fallback_text: str,
        language_hint: str,
        speaker_id: Optional[str],
        turn_id: Optional[str],
    ) -> tuple[ReadySlowWindow, WindowTimeBasePackage]:
        ready_window = ctx.ready_slow_window or self._build_compat_ready_slow_window(
            ctx=ctx,
            whisper_result=whisper_result,
            fallback_text=fallback_text,
            language_hint=language_hint,
            speaker_id=speaker_id,
            turn_id=turn_id,
        )
        window_time_base = ctx.window_time_base or self._build_compat_window_time_base(
            ctx=ctx,
            ready_window=ready_window,
            language_hint=language_hint,
        )
        ctx.ready_slow_window = ready_window
        ctx.window_time_base = window_time_base
        ctx.time_base_chunk = window_time_base
        return ready_window, window_time_base

    def _build_compat_ready_slow_window(
        self,
        *,
        ctx: ProcessingContext,
        whisper_result: Dict[str, Any],
        fallback_text: str,
        language_hint: str,
        speaker_id: Optional[str],
        turn_id: Optional[str],
    ) -> ReadySlowWindow:
        chunk = ctx.audio_chunk
        chunk_id = str(getattr(chunk, "chunk_id", "") or f"chunk-{ctx.chunk_index}")
        start = float(getattr(chunk, "start", 0.0) or 0.0) if chunk is not None else 0.0
        end = float(getattr(chunk, "end", start) or start) if chunk is not None else start
        if end <= start and ctx.time_base_chunk is not None:
            units = tuple(getattr(ctx.time_base_chunk, "word_units", ()) or getattr(ctx.time_base_chunk, "raw_units", ()))
            if units:
                start = float(units[0].start)
                end = float(units[-1].end)
        if end <= start:
            end = start + 0.01
        source_text = str(
            fallback_text
            or whisper_result.get("text_clean")
            or whisper_result.get("text")
            or whisper_result.get("text_itn_raw")
            or ""
        ).strip()
        semantic_chunk_id = f"compat-semantic-{ctx.chunk_index}"
        speaker = str(speaker_id or "unknown")
        ready_window = ReadySlowWindow(
            window_id=str(ctx.slow_window_id or f"compat-window-{ctx.chunk_index}"),
            owner_chunk_id=chunk_id,
            owner_chunk_index=int(ctx.chunk_index),
            window_mode="steady",
            flush_reason=str(ctx.slow_window_flush_reason or "compat_single_chunk"),
            audio_segments=((start, end),),
            coverage=WindowCoverage(
                core_segments=((start, end),),
                left_guard_sec=0.0,
                right_guard_sec=0.0,
                chunk_bindings=(
                    WindowChunkBinding(
                        chunk_id=chunk_id,
                        chunk_index=int(ctx.chunk_index),
                        chunk_start=start,
                        chunk_end=end,
                        overlap_ratio=1.0,
                        role="owner",
                        is_owner=True,
                    ),
                ),
            ),
            source_semantic_chunk_ids=(semantic_chunk_id,),
            source_chunk_ids=(chunk_id,),
            source_chunk_indices=(int(ctx.chunk_index),),
            source_units=(
                WindowSourceUnit(
                    unit_id=f"{chunk_id}:compat-source",
                    semantic_chunk_id=semantic_chunk_id,
                    text=source_text,
                    audio_start=start,
                    audio_end=end,
                    source_chunk_ids=(chunk_id,),
                    source_chunk_indices=(int(ctx.chunk_index),),
                    speaker_id=speaker,
                    turn_id=turn_id,
                    language=str(language_hint or "auto"),
                    arrived_at=end,
                ),
            ),
            dialogue_shape=DialogueShapeSnapshot(
                shape="single_speaker",
                speaker_count=1,
                dominant_speaker_id=speaker,
                dominant_speaker_ratio=1.0,
                speaker_switch_count=0,
                speaker_switch_density=0.0,
                turn_count=1,
                avg_turn_duration_sec=max(end - start, 0.01),
            ),
            language_profile=WindowLanguageProfile(
                primary_language=str(language_hint or "auto"),
                language_mix_state="single_language",
                decision_domains=("timeanchored_alignment",),
                should_bypass_whisper=False,
            ),
            prompt_seed=PromptSeed(text=source_text),
            batch_hint=WindowBatchHint(
                duration_bucket="compat_single_chunk",
                token_estimate=max(len(source_text), 1),
                acoustic_density_hint="unknown",
                queue_priority=0,
            ),
            created_at=end,
        )
        return ready_window

    @staticmethod
    def _build_compat_window_time_base(
        *,
        ctx: ProcessingContext,
        ready_window: ReadySlowWindow,
        language_hint: str,
    ) -> WindowTimeBasePackage:
        time_base = ctx.time_base_chunk
        if time_base is None:
            raise ValueError("对齐阶段缺少 time base，无法构建 AlignmentPreparation")
        if isinstance(time_base, WindowTimeBasePackage):
            return time_base
        chunk_bindings = tuple(getattr(ready_window.coverage, "chunk_bindings", ()) or ())
        return WindowTimeBasePackage(
            window_id=ready_window.window_id,
            language=str(getattr(time_base, "language", "") or language_hint or "auto"),
            raw_units=WindowTimeBaseAssembler._rebase_units(
                tuple(getattr(time_base, "raw_units", ()) or ()),
                binding=chunk_bindings[0] if chunk_bindings else None,
            ),
            word_units=WindowTimeBaseAssembler._rebase_units(
                tuple(getattr(time_base, "word_units", ()) or ()),
                binding=chunk_bindings[0] if chunk_bindings else None,
            ),
            quality=getattr(time_base, "quality"),
            source_chunk_ids=ready_window.source_chunk_ids,
            source_chunk_indices=ready_window.source_chunk_indices,
            chunk_bindings=chunk_bindings,
            metadata=dict(getattr(time_base, "metadata", {}) or {}),
            frame_stride=float(getattr(time_base, "frame_stride", 0.06) or 0.06),
            source=str(getattr(time_base, "source", "sensevoice_window") or "sensevoice_window"),
            contract_version=str(getattr(time_base, "contract_version", "1.0") or "1.0"),
        )

    def _commit_timeanchored_main_chain_result(
        self,
        *,
        ctx: ProcessingContext,
        stage_result: AnchorMountStageResult,
        whisper_result: Dict[str, Any],
        sv_result: Dict[str, Any],
        speaker_id: Optional[str],
        turn_id: Optional[str],
    ) -> None:
        host = self._host
        text_route, edge_route, final_route, stage_error_code = self._resolve_anchor_mount_routes(
            ctx=ctx,
            stage_result=stage_result,
        )
        preparation = ctx.alignment_preparation
        language = str(
            getattr(getattr(preparation, "compat", None), "text_truth", None).language
            if getattr(getattr(preparation, "compat", None), "text_truth", None) is not None
            else ""
            or whisper_result.get("language")
            or "auto"
        )
        chosen_text_clean = str(
            getattr(getattr(preparation, "slow_text", None), "window_text", None).text
            if getattr(getattr(preparation, "slow_text", None), "window_text", None) is not None
            else ""
            or ""
        ).strip()
        if not chosen_text_clean:
            chosen_text_clean = self._resolve_text_fallback_content(
                chosen_text_clean="",
                tracks=ctx.text_tracks,
                whisper_result=whisper_result,
                sv_result=sv_result,
            )
        adapter_result = self._decision_ingress_adapter.build(
            package=stage_result.decision_ingress,
            speaker_id=speaker_id,
            turn_id=turn_id,
        )
        decision_output = host._decision_processor.process(
            adapter_result.decision_input,
            stream_id=adapter_result.stream_id,
            chunk_index=adapter_result.chunk_index,
            is_last_chunk=host._is_last_chunk_index(adapter_result.chunk_index),
        )
        final_sentences = list(decision_output.sentence_segments)
        fallback_error_code = ""
        if not final_sentences:
            fallback_text = chosen_text_clean or adapter_result.decision_input.fallback_clean_text_ref
            if not fallback_text:
                fallback_text = chosen_text_clean
            fallback_confidence = self._resolve_text_fallback_confidence(
                chosen_source=str(getattr(ctx.arbitration_result, "chosen_source", "") or "slow"),
                whisper_result=whisper_result,
                sv_result=sv_result,
            )
            fallback_sentence = self._build_text_fallback_sentence(
                text=fallback_text,
                chunk=ctx.audio_chunk,
                confidence=fallback_confidence,
            )
            if fallback_sentence is not None:
                fallback_sentence.speaker_id = speaker_id
                fallback_sentence.turn_id = turn_id
                final_sentences = [fallback_sentence]
                fallback_error_code = "E_TIMEANCHORED_SEGMENT_EMPTY_FALLBACK"

        if host._final_grouper:
            final_sentences = host._final_grouper.group(final_sentences)
        for sentence in final_sentences:
            sentence.source = TextSource.WHISPER_PATCH
            sentence.is_draft = False
            sentence.is_finalized = True
            sentence.alignment_score = float(
                stage_result.anchor_mount_result.metrics.get("alignment_score")
                or 0.0
            )
            sentence.matched_ratio = float(
                stage_result.anchor_mount_result.metrics.get("coverage_ratio")
                or 0.0
            )
            sentence.confidence_source = host._resolve_sentence_confidence_source(
                sentence.words
            )
            if getattr(sentence, "speaker_id", None) is None:
                sentence.speaker_id = speaker_id
            if getattr(sentence, "turn_id", None) is None:
                sentence.turn_id = turn_id
        host._assign_sentence_identity_by_timeline_overlap(
            final_sentences,
            fallback_chunk=ctx.audio_chunk,
        )
        output_traces = host._normalize_output_traces_for_sentences(
            final_sentences=final_sentences,
            output_traces=list(decision_output.output_traces or []),
            default_reason="timeanchored_chain",
        )
        aligned_facts = adapter_result.decision_input.aligned_facts
        fused_evidence = adapter_result.decision_input.fused_evidence
        injection_stats = {
            "injection_positions_total": float(
                adapter_result.compat_report.get("fallback_punctuation_position_count", 0)
            ),
            "injection_unmatched_total": 0.0,
            "injection_miss_ratio": 0.0,
            "injection_mapping_coverage": (
                1.0
                if stage_result.decision_ingress.punctuation_facts
                else 0.0
            ),
            "injection_blocked": 0.0,
            "injection_error_code": "",
        }
        split_stats = dict(host._final_splitter.last_split_stats or {})
        split_stats.update(dict(decision_output.segmentation_report.get("boundary_score_stats", {})))
        split_stats.update(dict(decision_output.segmentation_report.get("soft_cut_stats", {})))
        split_stats["timeanchored_boundary_candidate_count"] = int(
            len(stage_result.decision_ingress.boundary_hints)
        )
        split_stats["timeanchored_boundary_hard_count"] = int(
            sum(1 for item in stage_result.decision_ingress.boundary_hints if bool(item.hard_flag))
        )
        split_stats["timeanchored_boundary_pause_count"] = int(
            sum(1 for item in stage_result.decision_ingress.boundary_hints if "pause" in str(item.reason))
        )
        split_stats["timeanchored_boundary_punct_count"] = int(
            sum(
                1
                for item in stage_result.decision_ingress.boundary_hints
                if str(item.reason).startswith("punctuation")
            )
        )
        split_stats["decision_ingress_adapter_fallback_punct_count"] = int(
            adapter_result.compat_report.get("fallback_punctuation_position_count", 0) or 0
        )
        soft_cut_observe_snapshot = host._update_soft_cut_observability(
            split_stats=split_stats,
            chunk_index=ctx.chunk_index,
            stage="timeanchored_alignment_stage",
        )

        output_layer_result = host._emit_output_layer(
            chunk_index=ctx.chunk_index,
            sentence_segments=final_sentences,
            language=language,
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
                "route": "timeanchored",
                "text_route": text_route,
                "edge_route": edge_route,
                "error_code": str(
                    split_stats.get("error_code", "")
                    or stage_error_code
                    or fallback_error_code
                ),
            },
            output_traces=output_traces,
            default_trace_reason="timeanchored_chain",
            subtitle_batch=decision_output.subtitle_batch,
        )
        ctx.final_sentences = list(final_sentences)
        ctx.finalization_metrics = {
            "coverage": float(
                stage_result.anchor_mount_result.metrics.get("coverage_ratio", 0.0) or 0.0
            ),
            "gap_ratio": float(
                stage_result.anchor_mount_result.metrics.get("largest_unresolved_span", 0.0)
                or 0.0
            ),
            "alignment_score": float(
                stage_result.anchor_mount_result.metrics.get("alignment_score")
                or 0.0
            ),
            "gap_positions": [],
            "gap_resolution": None,
            **injection_stats,
            "timeanchored_enabled": 1.0,
            "timeanchored_text_route": text_route,
            "timeanchored_edge_route": edge_route,
            "timeanchored_final_route": final_route,
            "timeanchored_item_count": float(len(stage_result.decision_ingress.tokens)),
            "timeanchored_sentence_count": float(len(final_sentences)),
            "timeanchored_failed_span_count": 0.0,
            "timeanchored_boundary_candidate_count": float(len(stage_result.decision_ingress.boundary_hints)),
            "l7_error_count": float(len(output_layer_result.output_payload.get("errors", []))),
        }
        for key, value in stage_result.anchor_mount_result.metrics.items():
            ctx.finalization_metrics[f"anchor_mount_{key}"] = value
        for key, value in split_stats.items():
            ctx.finalization_metrics[f"split_{key}"] = value
        for key, value in soft_cut_observe_snapshot.items():
            ctx.finalization_metrics[f"soft_cut_obs_{key}"] = value
        ctx.finalization_metrics["fact_word_count"] = float(len(aligned_facts.annotated_words))
        ctx.finalization_metrics["fact_turn_count"] = float(len(aligned_facts.speaker_turns))
        ctx.finalization_metrics["fact_mapping_count"] = float(len(aligned_facts.time_mappings))
        ctx.finalization_metrics["evidence_speaker_change_count"] = float(
            len(fused_evidence.speaker_changes) if fused_evidence is not None else 0
        )
        ctx.finalization_metrics["evidence_pause_anchor_count"] = float(
            len(fused_evidence.pause_anchors) if fused_evidence is not None else 0
        )
        ctx.finalization_metrics["evidence_semantic_anchor_count"] = float(
            len(fused_evidence.semantic_anchors) if fused_evidence is not None else 0
        )
        ctx.finalization_metrics["evidence_punctuation_anchor_count"] = float(
            len(fused_evidence.punctuation_anchors) if fused_evidence is not None else 0
        )
        if fallback_error_code:
            ctx.finalization_metrics["timeanchored_error_code"] = fallback_error_code
        host.logger.debug(
            "Chunk {}: timeanchored 主链完成 sentences={} route={}",
            ctx.chunk_index,
            len(final_sentences),
            final_route,
        )

    def _commit_fast_direct_result(
        self,
        *,
        ctx: ProcessingContext,
        sv_result: Dict[str, Any],
        reason: str,
    ) -> None:
        host = self._host
        chunk = ctx.audio_chunk
        language = str(
            getattr(chunk, "language", "")
            or sv_result.get("language")
            or "auto"
        )
        speaker_id = host._resolve_speaker_id_for_chunk(chunk)
        turn_id = host._resolve_turn_id_for_chunk(chunk)
        sv_words = host._build_sv_word_timestamps(sv_result, chunk)
        fast_stream = self._build_fast_direct_stream(words=sv_words)
        boundary_evidences = self._timeanchored_sentence_segmenter.collect_boundary_evidences(
            stream=fast_stream,
            language=language,
        )
        run_result = host._finalize_timeanchored_stream(
            final_stream=fast_stream,
            boundary_evidences=boundary_evidences,
            detected_language=language,
            chosen_text_clean=str(
                sv_result.get("text_clean")
                or sv_result.get("text_itn_raw")
                or sv_result.get("text")
                or ""
            ).strip(),
            punctuation_positions=[],
            punctuation_clean_text=str(
                sv_result.get("text_clean")
                or sv_result.get("text_itn_raw")
                or sv_result.get("text")
                or ""
            ).strip(),
            speaker_id=speaker_id,
            turn_id=turn_id,
            coverage=1.0 if fast_stream else 0.0,
            route_confidence=1.0 if fast_stream else 0.0,
            error_code="",
        )
        final_sentences = list(run_result.final_sentences)

        fallback_error_code = ""
        if not final_sentences:
            fallback_text = str(
                sv_result.get("text_clean")
                or sv_result.get("text_itn_raw")
                or sv_result.get("text")
                or ""
            ).strip()
            fallback_confidence = self._resolve_text_fallback_confidence(
                chosen_source="fast",
                whisper_result=ctx.whisper_result or {},
                sv_result=sv_result,
            )
            fallback_sentence = self._build_text_fallback_sentence(
                text=fallback_text,
                chunk=chunk,
                confidence=fallback_confidence,
            )
            if fallback_sentence is not None:
                fallback_sentence.speaker_id = speaker_id
                fallback_sentence.turn_id = turn_id
                final_sentences = [fallback_sentence]
                fallback_error_code = "E_FAST_DIRECT_SEGMENT_EMPTY_FALLBACK"

        for sentence in final_sentences:
            sentence.source = TextSource.SENSEVOICE
            sentence.is_draft = False
            sentence.is_finalized = True
            sentence.confidence_source = host._resolve_sentence_confidence_source(
                sentence.words
            )
            if getattr(sentence, "speaker_id", None) is None:
                sentence.speaker_id = speaker_id
            if getattr(sentence, "turn_id", None) is None:
                sentence.turn_id = turn_id

        host._assign_sentence_identity_by_timeline_overlap(
            final_sentences,
            fallback_chunk=chunk,
        )
        output_traces = host._normalize_output_traces_for_sentences(
            final_sentences=final_sentences,
            output_traces=list(run_result.output_traces or []),
            default_reason="fast_direct",
        )
        split_stats = dict(run_result.split_stats)
        soft_cut_observe_snapshot = host._update_soft_cut_observability(
            split_stats=split_stats,
            chunk_index=ctx.chunk_index,
            stage="fast_direct_alignment_stage",
        )

        output_layer_result = host._emit_output_layer(
            chunk_index=ctx.chunk_index,
            sentence_segments=final_sentences,
            language=language,
            injection_report={
                "mapping_coverage": float(
                    run_result.injection_stats.get("injection_mapping_coverage", 0.0)
                ),
                "mismatch_count": float(
                    run_result.injection_stats.get("injection_unmatched_total", 0.0)
                ),
                "error_code": str(
                    run_result.injection_stats.get("injection_error_code", "") or fallback_error_code
                ),
                "blocked": float(run_result.injection_stats.get("injection_blocked", 0.0)),
            },
            segmentation_report={
                "boundary_score_stats": dict(split_stats),
                "forced_split_count": float(split_stats.get("force_split_count", 0.0)),
                "route": "fast_direct",
                "text_route": "fast_direct",
                "edge_route": "fast",
                "error_code": str(split_stats.get("error_code", "") or fallback_error_code),
            },
            output_traces=output_traces,
            default_trace_reason="fast_direct",
            subtitle_batch=run_result.subtitle_batch,
        )

        ctx.final_sentences = list(final_sentences)
        ctx.hetero_route = "fast"
        ctx.hetero_alignment_result = {
            "mode": "fast_direct",
            "selected": True,
            "reason": reason,
            "route": "fast",
        }
        ctx.hetero_alignment_report = dict(ctx.hetero_alignment_result)
        ctx.finalization_metrics = {
            "coverage": run_result.alignment_result.coverage,
            "gap_ratio": run_result.alignment_result.gap_ratio,
            "alignment_score": run_result.alignment_result.alignment_score,
            "gap_positions": list(run_result.alignment_result.gap_positions),
            "gap_resolution": None,
            "fast_direct_enabled": 1.0,
            "alignment_pipeline_mode": "fast_direct",
            "alignment_pipeline_selected": 1.0,
            "alignment_pipeline_reason": reason,
            "alignment_pipeline_route": "fast",
            "timeanchored_enabled": 0.0,
            "timeanchored_text_route": "fast_direct",
            "timeanchored_edge_route": "fast",
            "timeanchored_final_route": "fast",
            "timeanchored_item_count": float(len(fast_stream)),
            "timeanchored_sentence_count": float(len(final_sentences)),
            "timeanchored_failed_span_count": 0.0,
            "timeanchored_boundary_candidate_count": float(len(boundary_evidences)),
            "l7_error_count": float(len(output_layer_result.output_payload.get("errors", []))),
        }
        for key, value in run_result.injection_stats.items():
            ctx.finalization_metrics[key] = value
        for key, value in split_stats.items():
            ctx.finalization_metrics[f"split_{key}"] = value
        for key, value in soft_cut_observe_snapshot.items():
            ctx.finalization_metrics[f"soft_cut_obs_{key}"] = value
        if fallback_error_code:
            ctx.finalization_metrics["timeanchored_error_code"] = fallback_error_code

        host.logger.debug(
            "Chunk {}: 快流直通定稿完成 sentences={} reason={}",
            ctx.chunk_index,
            len(final_sentences),
            reason,
        )

    def _record_hetero_alignment_result(
        self,
        *,
        ctx: ProcessingContext,
        stage_result: Optional[AnchorMountStageResult],
        mode: str,
        selected: bool,
        reason: str,
    ) -> None:
        _, _, route, _ = self._resolve_anchor_mount_routes(
            ctx=ctx,
            stage_result=stage_result,
        )
        ctx.hetero_route = route
        ctx.hetero_alignment_result = {
            "mode": str(mode),
            "selected": bool(selected),
            "reason": str(reason),
            "route": route,
            "sentence_count": (
                int(len(stage_result.decision_ingress.tokens))
                if stage_result is not None
                else 0
            ),
            "failed_span_count": 0,
        }
        ctx.hetero_alignment_report = (
            asdict(stage_result.pipeline_report)
            if stage_result is not None
            else None
        )
        ctx.finalization_metrics["alignment_pipeline_mode"] = str(mode)
        ctx.finalization_metrics["alignment_pipeline_selected"] = 1.0 if selected else 0.0
        ctx.finalization_metrics["alignment_pipeline_reason"] = str(reason)
        ctx.finalization_metrics["alignment_pipeline_route"] = route

    def _should_accept_timeanchored_result(
        self,
        *,
        stage_result: Optional[AnchorMountStageResult],
        mode: str,
        ctx: ProcessingContext,
    ) -> tuple[bool, str]:
        if stage_result is None:
            return False, f"{mode}_timeanchored_failed"
        _, _, route, _ = self._resolve_anchor_mount_routes(
            ctx=ctx,
            stage_result=stage_result,
        )
        if route == "error":
            return False, f"{mode}_gate_route_error"
        if not stage_result.decision_ingress.tokens:
            return False, f"{mode}_gate_empty_stream"
        if mode == "default":
            return True, "default_gate_pass"
        return True, f"{mode}_gate_pass"

    def _resolve_anchor_mount_routes(
        self,
        *,
        ctx: ProcessingContext,
        stage_result: Optional[AnchorMountStageResult],
    ) -> tuple[str, str, str, str | None]:
        token_count = len(stage_result.decision_ingress.tokens) if stage_result is not None else 0
        text_route = "slow" if token_count else "error"
        edge_selection_mode = str(getattr(ctx, "edge_selection_mode", "") or "").strip().lower()
        if edge_selection_mode not in {"force_fast", "prefer_fast", "force_slow", "prefer_slow"}:
            edge_selection_mode = str(
                getattr(self._host, "_edge_selection_mode", "auto") or "auto"
            ).strip().lower()
        if not token_count:
            edge_route = "error"
        elif edge_selection_mode in {"force_fast", "prefer_fast"}:
            edge_route = "fast"
        else:
            edge_route = "slow"
        final_route = edge_route if edge_route != "error" else text_route
        error_code = "anchor_mount_empty" if not token_count else None
        return text_route, edge_route, final_route, error_code

    def _resolve_fast_direct_reason(self, *, ctx: ProcessingContext) -> str:
        host = self._host
        if bool(getattr(host, "is_sensevoice_only", False)):
            return "sensevoice_only"
        ctx_mode = str(getattr(ctx, "edge_selection_mode", "") or "").strip().lower()
        host_mode = str(getattr(host, "_edge_selection_mode", "auto") or "auto").strip().lower()
        if ctx_mode == "force_fast" or host_mode == "force_fast":
            return "force_fast"
        return ""

    @staticmethod
    def _build_fast_direct_stream(*, words: Sequence[Any]) -> tuple[AlignmentItem, ...]:
        stream: list[AlignmentItem] = []
        cursor = 0.0
        for item in words or []:
            text = str(getattr(item, "word", "") or "")
            if not text:
                continue
            try:
                start = float(getattr(item, "start", 0.0) or 0.0)
                end = float(getattr(item, "end", start) or start)
            except (TypeError, ValueError):
                continue
            start = max(start, cursor)
            end = max(end, start)
            cursor = end
            confidence = getattr(item, "confidence", None)
            if confidence is not None:
                try:
                    confidence = max(0.0, min(1.0, float(confidence)))
                except (TypeError, ValueError):
                    confidence = None
            stream.append(
                AlignmentItem(
                    text=text,
                    start=start,
                    end=end,
                    status="direct",
                    source="fast",
                    confidence=confidence,
                    reason="fast_direct",
                )
            )
        return tuple(stream)

    def _should_sample_alignment_pipeline_shadow(self, ctx: ProcessingContext) -> bool:
        sample_rate = max(
            0.0,
            min(1.0, float(getattr(self._host, "_alignment_pipeline_shadow_sample_rate", 0.0))),
        )
        if sample_rate <= 0.0:
            return False
        if sample_rate >= 1.0:
            return True
        digest = hashlib.md5(f"{ctx.job_id}:{int(ctx.chunk_index)}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], byteorder="big", signed=False) % 10000
        threshold = int(sample_rate * 10000)
        return bucket < threshold

    def _resolve_alignment_pipeline_mode(self) -> str:
        mode = str(getattr(self._host, "_alignment_pipeline_mode", "default") or "default").strip().lower()
        if mode == "active":
            return "active"
        if mode in {"default", "legacy", "shadow", "timeanchored", "off"}:
            return "default"
        return "default"

    @staticmethod
    def _resolve_text_fallback_content(
        *,
        chosen_text_clean: str,
        tracks: Any,
        whisper_result: Dict[str, Any],
        sv_result: Dict[str, Any],
    ) -> str:
        if chosen_text_clean:
            return str(chosen_text_clean).strip()
        if tracks and getattr(tracks, "chosen_track", None):
            track_text = str(getattr(tracks.chosen_track, "text_clean", "") or "").strip()
            if track_text:
                return track_text
        whisper_text = str(
            whisper_result.get("text_clean")
            or whisper_result.get("text")
            or whisper_result.get("text_itn_raw")
            or ""
        ).strip()
        if whisper_text:
            return whisper_text
        return str(
            sv_result.get("text_clean")
            or sv_result.get("text_itn_raw")
            or sv_result.get("text")
            or ""
        ).strip()

    @staticmethod
    def _resolve_text_fallback_confidence(
        *,
        chosen_source: str,
        whisper_result: Dict[str, Any],
        sv_result: Dict[str, Any],
    ) -> Optional[float]:
        preferred = whisper_result if chosen_source == "slow" else sv_result
        fallback = sv_result if chosen_source == "slow" else whisper_result
        for item in (preferred, fallback):
            value = item.get("confidence")
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _build_text_fallback_sentence(
        *,
        text: str,
        chunk: Optional[Any],
        confidence: Optional[float],
    ) -> Optional[SentenceSegment]:
        normalized_text = str(text or "").strip()
        if not normalized_text or chunk is None:
            return None
        start = float(getattr(chunk, "start", 0.0) or 0.0)
        end = float(getattr(chunk, "end", start) or start)
        if end <= start:
            end = start + 0.01
        return SentenceSegment(
            text=normalized_text,
            text_clean=normalized_text,
            start=start,
            end=end,
            words=[],
            confidence=confidence,
            source=TextSource.WHISPER_PATCH,
            is_draft=False,
            is_finalized=True,
        )

    @staticmethod
    def _apply_arbitration_track_selection(
        *,
        host: Any,
        chunk_index: int,
        tracks: Any,
        arbitration_output: Any,
    ) -> None:
        arbitration_result = getattr(arbitration_output, "arbitration_result", None)
        if arbitration_output.chosen_text_track:
            tracks.chosen_track = arbitration_output.chosen_text_track
            return
        if (
            arbitration_result is not None
            and str(getattr(arbitration_result, "error_code", "") or "")
            == "E_L2_ARBITRATION_FORCED_SOURCE_MISSING"
        ):
            forced_source = str(getattr(arbitration_result, "forced_source", "") or "")
            raise ValueError(
                f"Chunk {chunk_index}: 强制选边源缺失，终止本 chunk 定稿。forced_source={forced_source}"
            )
        if tracks.chosen_track is None:
            fallback_track = tracks.sv_track or tracks.whisper_track
            if fallback_track:
                tracks.chosen_track = host._clone_text_track(
                    fallback_track,
                    source="chosen",
                )

