"""
对齐阶段门面服务。

设计模式：Facade Pattern。
原因：收口对齐阶段的大段编排逻辑，让主实现类聚焦流水线调度与依赖装配。
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

from app.models.sensevoice_models import SentenceSegment, TextSource
from app.pipelines.dual_pipeline.services.textflow_facade_service import Layer456RunResult
from app.schemas.pipeline_context import ProcessingContext
from app.services.alignment.types import CharMapping, PunctTrack, TextTrack, TextTrackBundle
from app.services.language_policy import build_language_policy_snapshot
from app.services.textflow.alignment_path_adapter import AlignmentPathAdapter
from app.services.timeanchored_alignment import AlignmentItem
from app.services.timeanchored_alignment.contracts import (
    SelectedTextTruth,
    TimeBasePackage,
    TimeBaseQuality,
    TimeBaseUnit,
)
from app.services.timeanchored_alignment.anchor_mount.service import (
    AnchorMountAlignmentService,
    AnchorMountStageResult,
)
from app.services.timeanchored_alignment.decoder import AlignmentDecoderService
from app.services.timeanchored_alignment.preparation import (
    AlignmentPreparationAssembler,
    PreparationBundle,
)
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
from app.services.timeanchored_alignment.output_projection.output_projector import (
    OutputProjectionInput,
    OutputProjector,
)
from app.services.timeanchored_alignment.selection import TextSelectionService
from app.pipelines.dual_pipeline.services.postprocess_trace_writer import PostprocessTraceWriter
from app.pipelines.dual_pipeline.services.anchor_mount_graph_renderer import (
    AnchorMountGraphRenderer,
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
        self._timeanchored_stage_service = AnchorMountAlignmentService()
        self._alignment_decoder_service = AlignmentDecoderService()
        self._alignment_path_adapter = AlignmentPathAdapter()
        self._output_projector = OutputProjector()
        self._selection_service = TextSelectionService(logger=getattr(host, "logger", None))
        self._postprocess_trace_writer = PostprocessTraceWriter(
            logger=getattr(host, "logger", None),
            enabled=bool(getattr(host, "_postprocess_trace_enabled", False)),
            level=str(getattr(host, "_postprocess_trace_level", "summary") or "summary"),
        )
        self._anchor_mount_graph_renderer = AnchorMountGraphRenderer()
        graph_mode = str(getattr(host, "_anchor_mount_graph", "off") or "off").strip().lower()
        if graph_mode not in {"off", "svg", "html", "both"}:
            graph_mode = "off"
        self._anchor_mount_graph_mode = graph_mode

    async def run(self, ctx: ProcessingContext) -> None:
        """执行单个 chunk 的对齐阶段。"""
        host = self._host
        chunk = ctx.audio_chunk

        fast_mode_reason = self._resolve_fast_direct_reason(ctx=ctx)
        if fast_mode_reason:
            if ctx.sv_result is None:
                raise ValueError("对齐阶段缺少 SenseVoice 推理结果")
            self._hydrate_fast_mode_unified_inputs(ctx=ctx)
        elif ctx.whisper_skipped and ctx.sv_result is not None and ctx.whisper_result is None:
            self._hydrate_fast_mode_unified_inputs(ctx=ctx)

        # 阶段 1: 双流定稿主链（前置仲裁/标点域 + 四层主链）
        host.logger.debug(f"Chunk {ctx.chunk_index}: 双流对齐")

        if ctx.whisper_result is None or ctx.sv_result is None:
            raise ValueError("对齐阶段缺少必要的推理结果")

        whisper_result = ctx.whisper_result
        sv_result = ctx.sv_result

        # V3.2.0+dev.20260202.07: Whisper 清洗增强 + 规范化重算
        host._apply_whisper_full_sanitize(ctx)

        tracks = host._ensure_text_tracks(ctx)
        quality_signals = host._build_quality_signals(
            sv_result=sv_result,
            whisper_result=whisper_result,
            tracks=tracks,
        )
        selection_inputs = self._selection_service.build_selection_inputs(
            ctx=ctx,
            tracks=tracks,
            quality_signals=quality_signals,
            sv_result=sv_result,
            whisper_result=whisper_result,
        )
        selection_outcome = self._selection_service.select(
            ctx=ctx,
            selection_inputs=selection_inputs,
            arbitration_processor=host._l2_processor,
            clone_text_track=host._clone_text_track,
        )
        arbitration_output = selection_outcome.arbitration_output
        arbitration_result = selection_outcome.arbitration_result
        ctx.selected_text_truth = selection_outcome.selected_text_truth
        ctx.selection_decision = selection_outcome.selection_decision
        ctx.selection_report = selection_outcome.selection_report
        ctx.arbitration_result = arbitration_result
        self._write_selection_trace(
            ctx=ctx,
            selection_outcome=selection_outcome,
        )
        chosen_text_clean = self._selection_service.apply_runtime_selection(
            tracks=tracks,
            whisper_result=whisper_result,
            sv_result=sv_result,
            selection_outcome=selection_outcome,
        )
        chosen_source = selection_outcome.selection_decision.chosen_source
        # V3.2.0+dev.20260204.10: 删除慢流补跑分支，定稿标点仅走统一前置入口
        punct_track: Optional[PunctTrack] = None
        # V3.2.0+dev.20260204.11: 若仲裁选择 fast 且快流已产出同一 clean_text_ref 的 PunctTrack，直接复用避免重复跑模型
        if (
            chosen_source == "fast"
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
                chosen_source,
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
            chosen_source,
            str(getattr(ctx.selection_report, "primary_reason_code", "") or arbitration_result.reason),
            arbitration_result.coverage,
        )

        # V3.2.0+dev.20260205.09: 集合层（仅对齐与 gap 修复）
        sv_words = host._build_sv_word_timestamps(sv_result, chunk)
        speaker_id = host._resolve_speaker_id_for_chunk(ctx.audio_chunk) if ctx.audio_chunk else None
        turn_id = host._resolve_turn_id_for_chunk(ctx.audio_chunk) if ctx.audio_chunk else None
        policy_snapshot = None
        language_hint = str(
            (getattr(ctx.selected_text_truth, "language_hint", "") or "")
            or (tracks.chosen_track.language if tracks.chosen_track else "")
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
                    force_fast_reason = self._resolve_anchor_mount_force_fast_reason(
                        stage_result=stage_result
                    )
                    if force_fast_reason:
                        self._commit_fast_direct_result(
                            ctx=ctx,
                            sv_result=sv_result,
                            reason=force_fast_reason,
                        )
                        self._record_hetero_alignment_result(
                            ctx=ctx,
                            stage_result=stage_result,
                            mode=alignment_pipeline_mode,
                            selected=True,
                            reason=force_fast_reason,
                        )
                        return
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
                    chosen_source=self._resolve_selected_source(ctx, default="slow"),
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
                    "slow"
                    if self._resolve_selected_source(ctx, default="slow") == "slow"
                    else "fast"
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
        if ctx.whisper_result is None and ctx.sv_result is None:
            return False
        return True

    def _hydrate_fast_mode_unified_inputs(self, *, ctx: ProcessingContext) -> None:
        self._ensure_time_base_for_fast_mode(ctx=ctx)
        if ctx.whisper_result is None:
            ctx.whisper_result = self._build_compat_whisper_result_from_fast(ctx=ctx)

    def _ensure_time_base_for_fast_mode(self, *, ctx: ProcessingContext) -> None:
        if ctx.time_base_chunk is not None:
            return
        sv_result = ctx.sv_result or {}
        if not sv_result:
            return
        chunk = ctx.audio_chunk
        sv_words = list(self._host._build_sv_word_timestamps(sv_result, chunk) or [])
        units: list[TimeBaseUnit] = []
        for item in sv_words:
            text = str(getattr(item, "word", "") or "").strip()
            if not text:
                continue
            try:
                start = float(getattr(item, "start", 0.0) or 0.0)
                end = float(getattr(item, "end", start) or start)
            except (TypeError, ValueError):
                continue
            if end <= start:
                end = start + 0.01
            confidence_raw = getattr(item, "confidence", 0.8)
            try:
                confidence = float(confidence_raw if confidence_raw is not None else 0.8)
            except (TypeError, ValueError):
                confidence = 0.8
            confidence = max(0.0, min(1.0, confidence))
            units.append(
                TimeBaseUnit(
                    text=text,
                    start=start,
                    end=end,
                    confidence=confidence,
                    token_type="word",
                    source="sensevoice",
                )
            )
        if not units:
            source_text = str(
                sv_result.get("text_clean")
                or sv_result.get("text_itn_raw")
                or sv_result.get("text")
                or ""
            ).strip()
            if source_text:
                start = float(getattr(chunk, "start", 0.0) or 0.0) if chunk is not None else 0.0
                end = float(getattr(chunk, "end", start) or start) if chunk is not None else start
                if end <= start:
                    end = start + 0.01
                units.append(
                    TimeBaseUnit(
                        text=source_text,
                        start=start,
                        end=end,
                        confidence=0.8,
                        token_type="raw",
                        source="sensevoice",
                    )
                )
        if not units:
            return
        language = str(
            sv_result.get("language")
            or getattr(chunk, "language", "")
            or "auto"
        )
        quality = TimeBaseQuality(
            blank_ratio=0.0,
            avg_max_prob=0.8,
            low_prob_ratio=0.0,
            unit_count=len(units),
            word_count=len(units),
        )
        ctx.time_base_chunk = TimeBasePackage(
            raw_units=tuple(units),
            word_units=tuple(units),
            quality=quality,
            language=language,
            source="sensevoice_fast_compat",
            metadata={"compat_generated": True},
        )

    @staticmethod
    def _build_compat_whisper_result_from_fast(*, ctx: ProcessingContext) -> Dict[str, Any]:
        sv_result = dict(ctx.sv_result or {})
        text = str(
            sv_result.get("text_clean")
            or sv_result.get("text_itn_raw")
            or sv_result.get("text")
            or ""
        ).strip()
        language = str(
            sv_result.get("language")
            or getattr(ctx.audio_chunk, "language", "")
            or "auto"
        )
        confidence = sv_result.get("confidence", 0.8)
        return {
            "text": text,
            "text_clean": text,
            "text_itn_raw": text,
            "language": language,
            "confidence": confidence,
            "segments": [],
            "raw_result": {"segments": []},
            "is_fast_compat": True,
        }

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
                chosen_text_clean=str(
                    getattr(getattr(ctx, "selected_text_truth", None), "text", "") or ""
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
            preparation_selected_text_truth, preparation_whisper_result = (
                self._resolve_preparation_text_inputs(
                    ready_window=ready_window,
                    selected_text_truth=getattr(ctx, "selected_text_truth", None),
                    whisper_result=whisper_result,
                    fallback_text=fallback_text,
                    language_hint=base_language,
                )
            )
            host.logger.debug(
                "Chunk {}: preparation 输入 window_id={} source_chunks={} source_units={} time_base_raw_units={} time_base_word_units={} selected_text_len={}",
                ctx.chunk_index,
                ready_window.window_id,
                len(ready_window.source_chunk_ids),
                len(ready_window.source_units),
                len(window_time_base.raw_units),
                len(window_time_base.word_units),
                len(preparation_selected_text_truth.text),
            )
            self._trace_write(
                ctx=ctx,
                filename="10_preparation.input.json",
                stage="preparation_input",
                summary_payload={
                    "window_id": str(ready_window.window_id),
                    "owner_chunk_id": str(ready_window.owner_chunk_id),
                    "owner_chunk_index": int(ready_window.owner_chunk_index),
                    "source_chunk_ids": list(ready_window.source_chunk_ids),
                    "source_chunk_indices": list(ready_window.source_chunk_indices),
                    "source_units_count": len(ready_window.source_units),
                    "time_base_raw_units_count": len(window_time_base.raw_units),
                    "time_base_word_units_count": len(window_time_base.word_units),
                    "selected_text_source": str(preparation_selected_text_truth.text_source),
                    "selected_text_len": len(preparation_selected_text_truth.text),
                    "language_hint": base_language,
                },
                full_payload={
                    "ready_window": ready_window,
                    "window_time_base": window_time_base,
                    "selected_text_truth": preparation_selected_text_truth,
                    "language_hint": base_language,
                    "whisper_result": preparation_whisper_result,
                    "sv_result": sv_result,
                },
            )
            preparation = self._timeanchored_preparation_assembler.prepare(
                ready_window=ready_window,
                window_time_base=window_time_base,
                selected_text_truth=preparation_selected_text_truth,
                whisper_result=preparation_whisper_result,
                default_language=base_language,
                external_punct_track=ctx.punct_track,
            )
            self._trace_write(
                ctx=ctx,
                filename="11_preparation.output.json",
                stage="preparation_output",
                summary_payload={
                    "window_id": str(preparation.window_id),
                    "canonical_token_count": len(preparation.canonical_sequence.tokens),
                    "observation_slice_count": len(preparation.acoustic_observation_pack.slices),
                    "punctuation_fact_count": len(preparation.external_stable_facts.punctuation_facts),
                    "source_chunk_count": len(preparation.source_chunk_ids),
                    "selected_text_source": str(preparation.provenance.text_source),
                },
                full_payload=preparation,
            )
            self._postprocess_trace_writer.write_layer_summary(
                job_dir=ctx.job_dir,
                layer_summary=preparation.report.summary,
            )
            self._commit_preparation_context(ctx=ctx, preparation=preparation)

            return self._execute_prepared_timeanchored_stage(
                ctx=ctx,
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
        ctx: ProcessingContext,
        preparation: PreparationBundle,
        language: str,
    ) -> AnchorMountStageResult:
        self._host.logger.debug(
            "Chunk {}: preparation 执行 anchor_mount window_id={} token_units={} fast_hooks={} punctuation_evidences={}",
            preparation.owner_chunk_index,
            preparation.window_id,
            len(preparation.slow_text.token_units),
            len(preparation.fast_hooks),
            len(preparation.slow_text.punctuation_evidences),
        )
        self._trace_write(
            ctx=ctx,
            filename="20_anchor_mount.input.json",
            stage="anchor_mount_input",
            summary_payload={
                "window_id": str(preparation.window_id),
                "owner_chunk_id": str(preparation.owner_chunk_id),
                "token_unit_count": len(preparation.slow_text.token_units),
                "hook_count": len(preparation.fast_hooks),
                "punctuation_evidence_count": len(preparation.slow_text.punctuation_evidences),
                "language": language,
            },
            full_payload={
                "preparation": preparation,
                "language": language,
            },
        )
        stage_result = self._timeanchored_stage_service.execute(
            preparation=preparation,
            language=language,
        )
        decoder_result = self._run_decoder_alignment(
            ctx=ctx,
            preparation=preparation,
        )
        setattr(ctx, "decoder_alignment_result", decoder_result)
        if isinstance(stage_result, AnchorMountStageResult):
            self._trace_write(
                ctx=ctx,
                filename="21_anchor_mount.output.json",
                stage="anchor_mount_output",
                summary_payload={
                    "window_id": str(preparation.window_id),
                    "token_count": len(stage_result.anchor_mount_result.items),
                    "boundary_evidence_count": len(stage_result.anchor_mount_result.boundary_evidences),
                    "cross_chunk_lock_count": len(stage_result.anchor_mount_result.cross_chunk_locks),
                    "should_fallback": bool(stage_result.anchor_mount_result.should_fallback),
                    "metrics": dict(stage_result.anchor_mount_result.metrics),
                },
                full_payload=stage_result,
            )
            if decoder_result is not None and self._should_run_decoder_shadow(ctx):
                self._write_decoder_shadow_trace(
                    ctx=ctx,
                    preparation=preparation,
                    stage_result=stage_result,
                    decoder_shadow_result=decoder_result,
                )
            self._emit_anchor_mount_graph(
                ctx=ctx,
                preparation=preparation,
                stage_result=stage_result,
            )
        return stage_result

    def _run_decoder_alignment(
        self,
        *,
        ctx: ProcessingContext,
        preparation: PreparationBundle,
    ) -> Any | None:
        try:
            return self._alignment_decoder_service.execute(preparation=preparation)
        except Exception:
            self._host.logger.exception(
                "Chunk {}: decoder AlignmentPath 执行失败",
                getattr(ctx, "chunk_index", -1),
            )
            return None

    def _should_run_decoder_shadow(self, ctx: ProcessingContext) -> bool:
        if not self._postprocess_trace_writer.enabled:
            return False
        return self._should_sample_alignment_pipeline_shadow(ctx)

    def _write_decoder_shadow_trace(
        self,
        *,
        ctx: ProcessingContext,
        preparation: PreparationBundle,
        stage_result: AnchorMountStageResult,
        decoder_shadow_result: Any,
    ) -> None:
        alignment_path = getattr(decoder_shadow_result, "alignment_path", None)
        alignment_report = getattr(decoder_shadow_result, "alignment_report", None)
        self._trace_write(
            ctx=ctx,
            filename="22_decoder_shadow.output.json",
            stage="decoder_shadow_output",
            summary_payload={
                "window_id": str(preparation.window_id),
                "route": str(getattr(alignment_report, "route", "") or ""),
                "failure_semantic": str(getattr(alignment_report, "failure_semantic", "") or ""),
                "path_token_count": len(getattr(alignment_path, "aligned_tokens", ()) or ()),
                "boundary_candidate_count": len(
                    getattr(decoder_shadow_result, "boundary_candidates", ()) or ()
                ),
                "low_confidence_span_count": len(
                    getattr(decoder_shadow_result, "low_confidence_spans", ()) or ()
                ),
            },
            full_payload={
                "alignment_path": alignment_path,
                "boundary_candidates": getattr(decoder_shadow_result, "boundary_candidates", ()),
                "low_confidence_spans": getattr(decoder_shadow_result, "low_confidence_spans", ()),
                "alignment_report": alignment_report,
                "diagnostics": getattr(decoder_shadow_result, "diagnostics", {}),
            },
        )
        self._trace_write(
            ctx=ctx,
            filename="23_decoder_shadow.diff.json",
            stage="decoder_shadow_diff",
            summary_payload=self._build_decoder_shadow_diff(
                stage_result=stage_result,
                decoder_shadow_result=decoder_shadow_result,
            ),
            full_payload=self._build_decoder_shadow_diff(
                stage_result=stage_result,
                decoder_shadow_result=decoder_shadow_result,
            ),
        )
        if alignment_report is not None:
            self._postprocess_trace_writer.write_layer_summary(
                job_dir=ctx.job_dir,
                layer_summary=alignment_report.summary,
            )

    @staticmethod
    def _build_decoder_shadow_diff(
        *,
        stage_result: AnchorMountStageResult,
        decoder_shadow_result: Any,
    ) -> dict[str, Any]:
        alignment_path = getattr(decoder_shadow_result, "alignment_path", None)
        decoder_tokens = tuple(getattr(alignment_path, "aligned_tokens", ()) or ())
        anchor_mount_result = getattr(stage_result, "anchor_mount_result", None)
        anchor_tokens = tuple(
            SimpleNamespace(
                start=float(getattr(envelope, "provisional_start", 0.0) or 0.0),
                end=float(getattr(envelope, "provisional_end", 0.0) or 0.0),
            )
            for envelope in (getattr(anchor_mount_result, "envelopes", ()) or ())
        )
        pair_count = min(len(decoder_tokens), len(anchor_tokens))
        avg_timing_delta_ms = 0.0
        if pair_count:
            deltas = []
            for index in range(pair_count):
                decoder_item = decoder_tokens[index]
                anchor_item = anchor_tokens[index]
                deltas.append(
                    abs(float(getattr(decoder_item, "start", 0.0) or 0.0) - float(getattr(anchor_item, "start", 0.0) or 0.0))
                )
                deltas.append(
                    abs(float(getattr(decoder_item, "end", 0.0) or 0.0) - float(getattr(anchor_item, "end", 0.0) or 0.0))
                )
            avg_timing_delta_ms = sum(deltas) / max(len(deltas), 1) * 1000.0
        return {
            "decoder": {
                "route": str(getattr(getattr(decoder_shadow_result, "alignment_report", None), "route", "") or ""),
                "path_token_count": len(decoder_tokens),
                "boundary_candidate_count": len(
                    getattr(decoder_shadow_result, "boundary_candidates", ()) or ()
                ),
                "low_confidence_span_count": len(
                    getattr(decoder_shadow_result, "low_confidence_spans", ()) or ()
                ),
            },
            "anchor_mount": {
                "path_token_count": len(anchor_tokens),
                "boundary_evidence_count": len(
                    getattr(anchor_mount_result, "boundary_evidences", ()) or ()
                ),
                "should_fallback": bool(stage_result.anchor_mount_result.should_fallback),
            },
            "diff": {
                "path_token_count_delta": int(len(decoder_tokens) - len(anchor_tokens)),
                "boundary_count_delta": int(
                    len(getattr(decoder_shadow_result, "boundary_candidates", ()) or ())
                    - len(getattr(anchor_mount_result, "boundary_evidences", ()) or ())
                ),
                "source_chunk_scope_match": tuple(getattr(alignment_path, "source_chunk_ids", ()) or ())
                == AlignmentStageService._resolve_anchor_mount_source_chunk_ids(
                    stage_result=stage_result
                ),
                "avg_timing_delta_ms": float(avg_timing_delta_ms),
            },
        }

    def _commit_preparation_context(
        self,
        *,
        ctx: ProcessingContext,
        preparation: PreparationBundle,
    ) -> None:
        ctx.alignment_preparation = preparation
        ctx.preparation_report = preparation.report
        ctx.text_truth = preparation.compat.text_truth
        ctx.protected_spans = list(preparation.compat.protected_spans)
        ctx.language_runs = list(preparation.compat.language_runs.runs)
        ctx.pronunciation_package = preparation.compat.pronunciation
        ctx.pronunciation_report = dict(preparation.compat.pronunciation_report)
        ctx.slow_window_meta = {
            "window_id": preparation.window_id,
            "canonical_token_count": len(preparation.canonical_sequence.tokens),
            "source_chunk_ids": list(preparation.source_chunk_ids),
        }
        if isinstance(preparation.compat.time_base, WindowTimeBasePackage):
            ctx.window_time_base = preparation.compat.time_base
        self._host.logger.debug(
            "Chunk {}: preparation 已写回 ctx window_id={} canonical_token_count={} compat_hook_count={} pronunciation_state_count={} window_time_base_units={}",
            ctx.chunk_index,
            preparation.window_id,
            len(preparation.canonical_sequence.tokens),
            len(preparation.fast_hooks),
            len(preparation.pronunciation_graph.state_nodes),
            len(getattr(preparation.compat.time_base, "word_units", ()) or ()),
        )

    def _require_decoder_alignment_result(
        self,
        *,
        ctx: ProcessingContext,
        preparation: PreparationBundle,
    ) -> Any:
        decoder_result = getattr(ctx, "decoder_alignment_result", None)
        if decoder_result is None:
            decoder_result = self._run_decoder_alignment(
                ctx=ctx,
                preparation=preparation,
            )
            setattr(ctx, "decoder_alignment_result", decoder_result)
        if decoder_result is None or getattr(decoder_result, "alignment_path", None) is None:
            route = str(
                getattr(getattr(decoder_result, "alignment_report", None), "route", "") or ""
            )
            raise RuntimeError(
                "timeanchored 主链缺少 AlignmentPath，"
                f"无法进入正式切分入口 route={route or 'unknown'}"
            )
        return decoder_result

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

    def _resolve_preparation_text_inputs(
        self,
        *,
        ready_window: ReadySlowWindow,
        selected_text_truth: SelectedTextTruth | None,
        whisper_result: Dict[str, Any],
        fallback_text: str,
        language_hint: str,
    ) -> tuple[SelectedTextTruth, Dict[str, Any]]:
        """
        为 preparation 选择窗口级文本输入。

        根因背景：
        - 某些窗口场景下 whisper_result 仅包含 owner chunk 文本；
        - 但 ready_window.source_units 可能覆盖多个 source chunk；
        - 若仍用 owner 文本驱动 preparation，会导致非 owner chunk 内容在窗口定稿中被吞掉。
        """
        normalized_whisper = dict(whisper_result or {})
        current_text = str(
            getattr(selected_text_truth, "text", "") or fallback_text or ""
        ).strip()
        if not current_text:
            current_text = str(
                normalized_whisper.get("text_clean")
                or normalized_whisper.get("text")
                or normalized_whisper.get("text_itn_raw")
                or ""
            ).strip()
        if isinstance(selected_text_truth, SelectedTextTruth):
            normalized_selected = selected_text_truth
        else:
            source_chunk_ids = tuple(
                str(item)
                for item in (
                    getattr(selected_text_truth, "source_chunk_ids", None)
                    or ready_window.source_chunk_ids
                )
            )
            quality = dict(getattr(selected_text_truth, "quality", {}) or {})
            metadata = dict(getattr(selected_text_truth, "metadata", {}) or {})
            metadata.setdefault("raw_text", current_text)
            normalized_selected = SelectedTextTruth(
                text=current_text or fallback_text or " ",
                text_source=str(getattr(selected_text_truth, "text_source", "") or "slow"),
                language_hint=str(
                    getattr(selected_text_truth, "language_hint", "")
                    or language_hint
                    or "auto"
                ),
                source_chunk_ids=source_chunk_ids,
                quality=quality,
                rejection_reasons=tuple(
                    getattr(selected_text_truth, "rejection_reasons", ()) or ()
                ),
                metadata=metadata,
            )
        if len(tuple(ready_window.source_units or ())) <= 1:
            return normalized_selected, normalized_whisper

        window_source_text = self._build_ready_window_source_text(ready_window=ready_window)
        if not window_source_text:
            return normalized_selected, normalized_whisper
        if self._key_char_length(window_source_text) <= self._key_char_length(current_text):
            return normalized_selected, normalized_whisper

        promoted = dict(normalized_whisper)
        for field in ("text", "text_clean", "text_itn_raw", "min_clean_text"):
            promoted[field] = window_source_text
        return (
            replace(
                normalized_selected,
                text=window_source_text,
                source_chunk_ids=tuple(str(item) for item in ready_window.source_chunk_ids),
                metadata={
                    **dict(getattr(normalized_selected, "metadata", {}) or {}),
                    "raw_text": window_source_text,
                    "promoted_from_window_source_units": True,
                },
            ),
            promoted,
        )

    @staticmethod
    def _build_ready_window_source_text(*, ready_window: ReadySlowWindow) -> str:
        texts = [
            str(getattr(unit, "text", "") or "").strip()
            for unit in tuple(getattr(ready_window, "source_units", ()) or ())
            if str(getattr(unit, "text", "") or "").strip()
        ]
        return " ".join(texts).strip()

    @staticmethod
    def _key_char_length(text: str) -> int:
        return sum(1 for char in str(text or "") if char.isalnum())

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
            getattr(getattr(preparation, "canonical_sequence", None), "language_hint", "")
            or (
                getattr(getattr(preparation, "compat", None), "text_truth", None).language
                if getattr(getattr(preparation, "compat", None), "text_truth", None) is not None
                else ""
            )
            or whisper_result.get("language")
            or "auto"
        )
        chosen_text_clean = str(
            getattr(getattr(preparation, "canonical_sequence", None), "normalized_text", "")
            if getattr(getattr(preparation, "canonical_sequence", None), "normalized_text", None) is not None
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
        decoder_result = self._require_decoder_alignment_result(
            ctx=ctx,
            preparation=preparation,
        )
        alignment_path = decoder_result.alignment_path
        adapter_result = self._alignment_path_adapter.build(
            preparation=preparation,
            decoder_result=decoder_result,
            speaker_id=speaker_id,
            turn_id=turn_id,
        )
        self._inject_timeline_turns_into_decision_input(
            decision_input=adapter_result.decision_input,
            time_axis_tokens=getattr(alignment_path, "aligned_tokens", ()) or (),
        )
        self._trace_write(
            ctx=ctx,
            filename="30_segmentation_ingress.input.json",
            stage="segmentation_ingress_input",
            summary_payload={
                "window_id": str(preparation.window_id),
                "aligned_token_count": len(getattr(alignment_path, "aligned_tokens", ()) or ()),
                "punctuation_fact_count": len(
                    getattr(adapter_result.decision_input, "canonical_punctuation_facts", ()) or ()
                ),
                "boundary_candidate_count": len(
                    getattr(adapter_result.decision_input, "canonical_candidate_boundaries", ()) or ()
                ),
                "coverage_owner_chunks": len(preparation.source_chunk_ids),
            },
            full_payload={
                "alignment_path": alignment_path,
                "alignment_report": getattr(decoder_result, "alignment_report", None),
                "boundary_candidates": getattr(decoder_result, "boundary_candidates", ()),
                "low_confidence_spans": getattr(decoder_result, "low_confidence_spans", ()),
            },
        )
        self._trace_write(
            ctx=ctx,
            filename="31_segmentation_ingress.output.json",
            stage="segmentation_ingress_output",
            summary_payload=self._build_segmentation_ingress_trace_summary(
                alignment_path=alignment_path,
                adapter_result=adapter_result,
            ),
            full_payload=adapter_result,
        )
        decision_output = host._decision_processor.process(
            adapter_result.decision_input,
            stream_id=adapter_result.stream_id,
            chunk_index=adapter_result.chunk_index,
            is_last_chunk=host._is_last_chunk_index(adapter_result.chunk_index),
        )
        self._trace_write(
            ctx=ctx,
            filename="40_decision.output.json",
            stage="decision_output",
            summary_payload={
                "sentence_count": len(getattr(decision_output, "aligned_sentences", ()) or ()),
                "trace_count": len(decision_output.output_traces),
                "segmentation_report": dict(decision_output.segmentation_report or {}),
                "has_subtitle_batch": bool(getattr(decision_output, "subtitle_batch", None)),
            },
            full_payload=decision_output,
        )
        final_sentences = list(decision_output.sentence_segments)
        fallback_error_code = ""
        if not final_sentences:
            fallback_text = chosen_text_clean or adapter_result.decision_input.fallback_clean_text_ref
            if not fallback_text:
                fallback_text = chosen_text_clean
            fallback_confidence = self._resolve_text_fallback_confidence(
                chosen_source=self._resolve_selected_source(ctx, default="slow"),
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
                if getattr(adapter_result.decision_input, "canonical_punctuation_facts", ())
                else 0.0
            ),
            "injection_blocked": 0.0,
            "injection_error_code": "",
        }
        split_stats = dict(host._final_splitter.last_split_stats or {})
        split_stats.update(dict(decision_output.segmentation_report.get("boundary_score_stats", {})))
        split_stats.update(dict(decision_output.segmentation_report.get("soft_cut_stats", {})))
        split_stats["timeanchored_boundary_candidate_count"] = int(
            len(getattr(alignment_path, "boundary_candidates", ()) or ())
        )
        split_stats["timeanchored_boundary_hard_count"] = int(
            sum(
                1
                for item in (getattr(alignment_path, "boundary_candidates", ()) or ())
                if bool(getattr(item, "hard_boundary", False))
            )
        )
        split_stats["timeanchored_boundary_lexical_count"] = int(
            sum(
                1
                for item in (getattr(alignment_path, "boundary_candidates", ()) or ())
                if str(getattr(item, "reason", "") or "") == "lexical_boundary"
            )
        )
        split_stats["timeanchored_boundary_anchor_block_close_count"] = int(
            sum(
                1
                for item in (getattr(alignment_path, "boundary_candidates", ()) or ())
                if str(getattr(item, "reason", "") or "") == "anchor_block_close"
            )
        )
        split_stats["segmentation_ingress_fallback_punct_count"] = int(
            adapter_result.compat_report.get("fallback_punctuation_position_count", 0) or 0
        )
        preparation_punctuation_count = int(
            len(getattr(getattr(preparation, "slow_text", None), "punctuation_evidences", ()) or ())
            if preparation is not None
            else 0
        )
        punctuation_chain_health = self._build_punctuation_chain_health_metrics(
            chosen_source=self._resolve_selected_source(ctx, default="unknown"),
            punct_track=ctx.punct_track,
            preparation_punctuation_count=preparation_punctuation_count,
            punctuation_fact_count=int(
                len(getattr(adapter_result.decision_input, "canonical_punctuation_facts", ()) or ())
            ),
        )
        split_stats["punctuation_chain_broken_flag"] = int(
            punctuation_chain_health["punctuation_chain_broken_flag"]
        )
        split_stats["punctuation_chain_track_sentence_end_count"] = int(
            punctuation_chain_health["punct_track_sentence_end_count"]
        )
        split_stats["punctuation_chain_preparation_count"] = int(
            punctuation_chain_health["preparation_punctuation_count"]
        )
        split_stats["punctuation_chain_fact_count"] = int(
            punctuation_chain_health["punctuation_fact_count"]
        )
        if punctuation_chain_health["punctuation_chain_broken_flag"]:
            host.logger.warning(
                "Chunk {}: punctuation_chain_broken chosen_source={} track_positions={} preparation_punctuation_count={} punctuation_fact_count={} reason={}",
                ctx.chunk_index,
                punctuation_chain_health["chosen_source"],
                punctuation_chain_health["punct_track_positions_total"],
                punctuation_chain_health["preparation_punctuation_count"],
                punctuation_chain_health["punctuation_fact_count"],
                punctuation_chain_health["punctuation_chain_broken_reason"],
            )
        soft_cut_observe_snapshot = host._update_soft_cut_observability(
            split_stats=split_stats,
            chunk_index=ctx.chunk_index,
            stage="timeanchored_alignment_stage",
        )

        projected_batches = self._build_projected_output_batches(
            preparation=preparation,
            stage_result=stage_result,
            decision_output=decision_output,
            split_stats=split_stats,
        )
        self._trace_write(
            ctx=ctx,
            filename="50_output_projection.output.json",
            stage="output_projection_output",
            summary_payload={
                "projected_chunk_count": len(projected_batches),
                "window_id": str(preparation.window_id),
                "source_chunk_indices": list(preparation.source_chunk_indices),
            },
            full_payload={
                "projected_batches": projected_batches,
                "window_id": str(preparation.window_id),
                "source_chunk_indices": list(preparation.source_chunk_indices),
            },
        )
        output_error_count = 0
        dispatch_payloads: list[dict[str, Any]] = []
        for projected_batch in projected_batches:
            output_layer_result = host._emit_output_layer(
                chunk_index=(
                    projected_batch.chunk_index
                    if projected_batch.chunk_index is not None
                    else projected_batch.chunk_id
                ),
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
                    "projection_chunk_count": int(len(projected_batches)),
                    "error_code": str(
                        split_stats.get("error_code", "")
                        or stage_error_code
                        or fallback_error_code
                    ),
                },
                output_traces=output_traces,
                default_trace_reason="timeanchored_chain",
                subtitle_batch=projected_batch,
            )
            output_error_count += len(output_layer_result.output_payload.get("errors", []))
            dispatch_payloads.append(
                {
                    "dispatch_chunk_index": (
                        projected_batch.chunk_index
                        if projected_batch.chunk_index is not None
                        else projected_batch.chunk_id
                    ),
                    "subtitle_count": len(getattr(projected_batch, "subtitles", ()) or ()),
                    "error_count": len(output_layer_result.output_payload.get("errors", [])),
                }
            )
        self._trace_write(
            ctx=ctx,
            filename="60_output_dispatch.payload.json",
            stage="output_dispatch_payload",
            summary_payload={
                "dispatch_count": len(dispatch_payloads),
                "dispatch_payloads": dispatch_payloads,
            },
            full_payload={
                "dispatch_payloads": dispatch_payloads,
                "projected_batches": projected_batches,
            },
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
            "timeanchored_item_count": float(len(getattr(alignment_path, "aligned_tokens", ()) or ())),
            "timeanchored_sentence_count": float(len(final_sentences)),
            "timeanchored_projected_chunk_count": float(len(projected_batches)),
            "timeanchored_failed_span_count": 0.0,
            "timeanchored_boundary_candidate_count": float(
                len(getattr(alignment_path, "boundary_candidates", ()) or ())
            ),
            "l7_error_count": float(output_error_count),
            **punctuation_chain_health,
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

    def _inject_timeline_turns_into_decision_input(
        self,
        *,
        decision_input: Any,
        time_axis_tokens: Sequence[Any],
    ) -> None:
        """
        将窗口内 timeline turn 真值注入 Decision 输入，避免 slot 级 speaker 继承导致误判。
        """
        aligned_facts = getattr(decision_input, "aligned_facts", None)
        if aligned_facts is None:
            return
        host = self._host
        timeline_turns = list(getattr(host, "_timeline_turns", []) or [])
        if not timeline_turns:
            return
        tokens = list(time_axis_tokens or ())
        if not tokens:
            return
        window_start = min(float(getattr(item, "start", 0.0) or 0.0) for item in tokens)
        window_end = max(float(getattr(item, "end", window_start) or window_start) for item in tokens)
        if window_end <= window_start:
            return
        selected_turns: list[dict[str, Any]] = []
        for turn in timeline_turns:
            turn_start = float(getattr(turn, "start", 0.0) or 0.0)
            turn_end = float(getattr(turn, "end", turn_start) or turn_start)
            if turn_end <= turn_start:
                continue
            if min(window_end, turn_end) <= max(window_start, turn_start):
                continue
            selected_turns.append(
                {
                    "turn_id": str(getattr(turn, "turn_id", "") or ""),
                    "speaker_id": str(getattr(turn, "speaker_id", "unknown") or "unknown"),
                    "start": turn_start,
                    "end": turn_end,
                    "source": "timeline_overlap",
                    "boundary_confidence": float(
                        getattr(turn, "boundary_confidence", 0.0) or 0.0
                    ),
                }
            )
        if selected_turns:
            aligned_facts.speaker_turns = selected_turns

    def _build_projected_output_batches(
        self,
        *,
        preparation: PreparationBundle,
        stage_result: AnchorMountStageResult,
        decision_output: Any,
        split_stats: Dict[str, Any],
    ) -> tuple[Any, ...]:
        owner_batch = getattr(decision_output, "subtitle_batch", None)
        if owner_batch is None:
            raise ValueError("timeanchored 主链缺少 subtitle_batch，无法执行 output projection")
        projection_input = OutputProjectionInput(
            window_id=str(preparation.window_id),
            source_chunk_ids=tuple(str(item) for item in preparation.source_chunk_ids),
            source_chunk_indices=tuple(int(item) for item in preparation.source_chunk_indices),
            coverage=preparation.coverage,
            carrier_batch=owner_batch,
            decision_metadata={
                "boundary_score_stats": dict(split_stats),
                "segmentation_report": dict(getattr(decision_output, "segmentation_report", {}) or {}),
                "segmentation_report_contract": (
                    asdict(getattr(decision_output, "segmentation_report_contract"))
                    if getattr(decision_output, "segmentation_report_contract", None) is not None
                    else None
                ),
            },
        )
        projected_batches = tuple(self._output_projector.project(projection_input))
        if not projected_batches:
            return (owner_batch,)
        return projected_batches

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
        tracks = ctx.text_tracks or TextTrackBundle()
        ctx.text_tracks = tracks
        normalize_sensevoice_result = getattr(host, "_normalize_sensevoice_result", None)
        if tracks.sv_track is None and callable(normalize_sensevoice_result):
            normalize_sensevoice_result(ctx)
            tracks = ctx.text_tracks or tracks
        if tracks.sv_track is None:
            fallback_text = str(
                sv_result.get("text_clean")
                or sv_result.get("text_itn_raw")
                or sv_result.get("text")
                or sv_result.get("raw_text")
                or ""
            ).strip()
            text_length = len(fallback_text)
            mapping = list(range(text_length))
            tracks.sv_track = TextTrack(
                raw_text=fallback_text,
                text_itn_raw=fallback_text,
                text_clean=fallback_text,
                char_mapping=[
                    CharMapping(raw_idx=index, clean_idx=index, punct=None)
                    for index in range(text_length)
                ],
                raw_to_clean=list(mapping),
                clean_to_raw=list(mapping),
                language=language,
                source="sensevoice",
                mapping_coverage=1.0 if text_length > 0 else 0.0,
            )
        if tracks.chosen_track is None:
            clone_text_track = getattr(host, "_clone_text_track", None)
            if callable(clone_text_track):
                tracks.chosen_track = clone_text_track(tracks.sv_track, source="chosen")
            else:
                tracks.chosen_track = tracks.sv_track

        punct_track = ctx.punct_track
        punctuation_positions = list(punct_track.positions) if punct_track and punct_track.positions else None
        punctuation_clean_text = (
            punct_track.clean_text_ref
            if punct_track and punct_track.clean_text_ref
            else str(
                getattr(tracks.chosen_track, "text_clean", "")
                or sv_result.get("text_clean")
                or sv_result.get("text_itn_raw")
                or sv_result.get("text")
                or ""
            ).strip()
        )
        policy_snapshot = None
        try:
            policy_snapshot = build_language_policy_snapshot(language_hint=language)
        except Exception:
            host.logger.exception(
                "Chunk {}: fast_direct 语言策略快照编译失败，回退空快照注入",
                ctx.chunk_index,
            )
        run_result = host._run_collection_scoring_decision_once(
            tracks=tracks,
            sv_result=sv_result,
            whisper_result=ctx.whisper_result or {},
            sv_words=sv_words,
            punctuation_positions=punctuation_positions,
            punctuation_clean_text=punctuation_clean_text,
            variant="legacy",
            speaker_id=speaker_id,
            turn_id=turn_id,
            policy_snapshot=policy_snapshot,
            is_fast_only_mode=True,
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
            "timeanchored_boundary_candidate_count": float(
                split_stats.get("boundary_candidate_count", 0.0) or 0.0
            ),
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

    def _trace_write(
        self,
        *,
        ctx: ProcessingContext,
        filename: str,
        stage: str,
        summary_payload: Any,
        full_payload: Any = None,
    ) -> None:
        writer = self._postprocess_trace_writer
        if not writer.enabled:
            return
        payload = full_payload if writer.level == "full" and full_payload is not None else summary_payload
        try:
            writer.write_stage(
                job_dir=ctx.job_dir,
                chunk_index=int(ctx.chunk_index),
                filename=filename,
                payload=payload,
                stage=stage,
            )
        except Exception:
            self._host.logger.exception(
                "Chunk {}: 写入后处理追踪失败 stage={} filename={}",
                ctx.chunk_index,
                stage,
                filename,
            )

    def _emit_anchor_mount_graph(
        self,
        *,
        ctx: ProcessingContext,
        preparation: PreparationBundle,
        stage_result: AnchorMountStageResult,
    ) -> None:
        if not self._postprocess_trace_writer.enabled:
            return
        if self._anchor_mount_graph_mode == "off":
            return
        try:
            graph_payload = self._anchor_mount_graph_renderer.build_graph_payload(
                window_id=str(preparation.window_id),
                items=stage_result.anchor_mount_result.items,
                envelopes=stage_result.anchor_mount_result.envelopes,
                boundary_evidences=stage_result.anchor_mount_result.boundary_evidences,
                hooks=preparation.fast_hooks,
                metrics=stage_result.anchor_mount_result.metrics,
            )
            self._postprocess_trace_writer.write_stage(
                job_dir=ctx.job_dir,
                chunk_index=int(ctx.chunk_index),
                filename="21_anchor_mount.graph.json",
                payload=graph_payload,
                stage="anchor_mount_graph",
            )
            if self._anchor_mount_graph_mode in {"svg", "both"}:
                svg_body = self._anchor_mount_graph_renderer.render_svg(graph_payload)
                self._postprocess_trace_writer.write_graph_artifact(
                    job_dir=ctx.job_dir,
                    chunk_index=int(ctx.chunk_index),
                    filename="21_anchor_mount.graph.svg",
                    body=svg_body,
                    stage="anchor_mount_graph",
                    media_type="image/svg+xml",
                )
            if self._anchor_mount_graph_mode in {"html", "both"}:
                svg_body = self._anchor_mount_graph_renderer.render_svg(graph_payload)
                html_body = (
                    "<!doctype html><html><head><meta charset='utf-8'>"
                    "<title>Anchor Mount Graph</title></head><body>"
                    f"{svg_body}</body></html>"
                )
                self._postprocess_trace_writer.write_graph_artifact(
                    job_dir=ctx.job_dir,
                    chunk_index=int(ctx.chunk_index),
                    filename="21_anchor_mount.graph.html",
                    body=html_body,
                    stage="anchor_mount_graph",
                    media_type="text/html",
                )
        except Exception:
            self._host.logger.exception(
                "Chunk {}: 生成锚点挂载图失败",
                ctx.chunk_index,
            )

    @staticmethod
    def _build_segmentation_ingress_trace_summary(
        *,
        alignment_path: Any,
        adapter_result: Any,
    ) -> dict[str, Any]:
        aligned_tokens = list(getattr(alignment_path, "aligned_tokens", ()) or ())
        boundary_candidates = list(getattr(alignment_path, "boundary_candidates", ()) or ())
        rows = [
            AlignmentStageService._serialize_alignment_path_token(index=index, token=item)
            for index, item in enumerate(aligned_tokens)
        ]
        return {
            "stream_id": str(getattr(adapter_result, "stream_id", "") or ""),
            "chunk_index": int(getattr(adapter_result, "chunk_index", 0) or 0),
            "fallback_clean_text_len": len(
                str(getattr(getattr(adapter_result, "decision_input", None), "fallback_clean_text_ref", "") or "")
            ),
            "compat_report": dict(getattr(adapter_result, "compat_report", {}) or {}),
            "aligned_tokens": rows,
            "boundary_candidates": [
                AlignmentStageService._serialize_alignment_boundary_candidate(item)
                for item in boundary_candidates
            ],
            "boundary_by_split": AlignmentStageService._build_boundary_groups_for_trace(
                token_units=aligned_tokens,
                boundary_evidences=boundary_candidates,
            ),
        }

    @staticmethod
    def _serialize_alignment_path_token(
        *,
        index: int,
        token: Any,
    ) -> dict[str, Any]:
        return {
            "token_index": int(index),
            "token_id": str(getattr(token, "token_id", "") or ""),
            "text": str(getattr(token, "text", "") or ""),
            "start": float(getattr(token, "start", 0.0) or 0.0),
            "end": float(getattr(token, "end", 0.0) or 0.0),
            "source_chunk_ids": [
                str(item) for item in (getattr(token, "source_chunk_ids", ()) or ())
            ],
            "confidence": float(getattr(token, "confidence", 0.0) or 0.0),
            "trace": dict(getattr(token, "trace", {}) or {}),
        }

    @staticmethod
    def _serialize_alignment_boundary_candidate(item: Any) -> dict[str, Any]:
        return {
            "split_idx": int(
                getattr(item, "split_token_index", getattr(item, "split_idx", 0)) or 0
            ),
            "event_time": float(getattr(item, "event_time", 0.0) or 0.0),
            "reason": str(getattr(item, "reason", "") or ""),
            "score": float(getattr(item, "score", 0.0) or 0.0),
            "hard_flag": bool(getattr(item, "hard_boundary", getattr(item, "hard_flag", False))),
            "source_chunk_ids": [
                str(value) for value in (getattr(item, "source_chunk_ids", ()) or ())
            ],
            "metadata": dict(getattr(item, "metadata", {}) or {}),
        }

    @staticmethod
    def _serialize_cross_chunk_lock_for_trace(item: Any) -> dict[str, Any]:
        return {
            "lock_id": str(getattr(item, "lock_id", "") or ""),
            "unit_ids": [str(value) for value in (getattr(item, "unit_ids", ()) or ())],
            "hook_ids": [str(value) for value in (getattr(item, "hook_ids", ()) or ())],
            "reason": str(getattr(item, "reason", "") or ""),
            "source_chunk_ids": [
                str(value) for value in (getattr(item, "source_chunk_ids", ()) or ())
            ],
            "source_chunk_indices": [
                int(value) for value in (getattr(item, "source_chunk_indices", ()) or ())
            ],
        }

    @staticmethod
    def _build_boundary_groups_for_trace(
        *,
        token_units: Sequence[Any],
        boundary_evidences: Sequence[Any],
    ) -> list[dict[str, Any]]:
        by_split_idx: dict[int, list[Any]] = {}
        for item in boundary_evidences:
            raw_split_idx = getattr(item, "split_idx", getattr(item, "split_token_index", -1))
            split_idx = int(raw_split_idx) if raw_split_idx is not None else -1
            if split_idx < 0:
                continue
            by_split_idx.setdefault(split_idx, []).append(item)
        rows: list[dict[str, Any]] = []
        for split_idx in sorted(by_split_idx):
            left_text = ""
            right_text = ""
            if 0 <= split_idx < len(token_units):
                left_text = str(
                    getattr(token_units[split_idx], "token_text", getattr(token_units[split_idx], "text", ""))
                    or ""
                )
            if 0 <= split_idx + 1 < len(token_units):
                right_text = str(
                    getattr(
                        token_units[split_idx + 1],
                        "token_text",
                        getattr(token_units[split_idx + 1], "text", ""),
                    )
                    or ""
                )
            evidences = by_split_idx[split_idx]
            rows.append(
                {
                    "split_idx": int(split_idx),
                    "left_text": left_text,
                    "right_text": right_text,
                    "reasons": [
                        str(getattr(item, "reason", "") or "")
                        for item in evidences
                    ],
                    "scores": [
                        float(getattr(item, "score", 0.0) or 0.0)
                        for item in evidences
                    ],
                    "blocked_by_lock": any(
                        bool((getattr(item, "metadata", {}) or {}).get("blocked_by_lock"))
                        for item in evidences
                    ),
                }
            )
        return rows

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
                int(len(stage_result.anchor_mount_result.items))
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

    @staticmethod
    def _build_punctuation_chain_health_metrics(
        *,
        chosen_source: str,
        punct_track: Optional[PunctTrack],
        preparation_punctuation_count: int,
        punctuation_fact_count: int,
    ) -> dict[str, Any]:
        positions = tuple(getattr(punct_track, "positions", ()) or ()) if punct_track is not None else tuple()
        sentence_end_marks = {"。", "！", "？", ".", "!", "?"}
        track_sentence_end_count = sum(
            1
            for item in positions
            if str(getattr(item, "punctuation", "") or "") in sentence_end_marks
        )
        broken_flag = 0
        broken_reason = ""
        if track_sentence_end_count > 0 and int(preparation_punctuation_count) <= 0:
            broken_flag = 1
            broken_reason = "punct_track_not_projected_to_preparation"
        elif track_sentence_end_count > 0 and int(punctuation_fact_count) <= 0:
            broken_flag = 1
            broken_reason = "preparation_not_mapped_to_punctuation_facts"
        return {
            "chosen_source": str(chosen_source or "unknown"),
            "punct_track_positions_total": int(len(positions)),
            "punct_track_sentence_end_count": int(track_sentence_end_count),
            "preparation_punctuation_count": int(preparation_punctuation_count),
            "punctuation_fact_count": int(punctuation_fact_count),
            "punctuation_chain_broken_flag": int(broken_flag),
            "punctuation_chain_broken_reason": str(broken_reason),
        }

    @staticmethod
    def _should_accept_timeanchored_result(
        *,
        stage_result: Optional[AnchorMountStageResult],
        mode: str,
        ctx: Optional[ProcessingContext] = None,
    ) -> tuple[bool, str]:
        if stage_result is None:
            return False, f"{mode}_timeanchored_failed"

        token_count = AlignmentStageService._resolve_anchor_mount_token_count(stage_result=stage_result)

        if ctx is not None and getattr(stage_result, "anchor_mount_result", None) is not None:
            edge_selection_mode = str(getattr(ctx, "edge_selection_mode", "") or "").strip().lower()
            if edge_selection_mode in {"force_fast", "prefer_fast"}:
                route = "fast" if token_count else "error"
            else:
                route = "slow" if token_count else "error"
        else:
            base_route = str(getattr(getattr(stage_result, "base_result", None), "route", "") or "")
            route = base_route or ("slow" if token_count else "error")

        if route == "error":
            return False, f"{mode}_gate_route_error"
        if token_count <= 0:
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
        token_count = self._resolve_anchor_mount_token_count(stage_result=stage_result)
        text_route = "slow" if token_count else "error"
        if self._should_force_fast_direct_on_anchor_mount_fallback(
            stage_result=stage_result
        ):
            edge_route = "fast" if token_count else "error"
            final_route = edge_route if edge_route != "error" else text_route
            error_code = "anchor_mount_empty" if not token_count else None
            return text_route, edge_route, final_route, error_code
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
    def _resolve_anchor_mount_source_chunk_ids(
        *,
        stage_result: Optional[AnchorMountStageResult],
    ) -> tuple[str, ...]:
        if stage_result is None:
            return tuple()
        anchor_mount_result = getattr(stage_result, "anchor_mount_result", None)
        if anchor_mount_result is None:
            return tuple()
        deduped_source_chunk_ids: list[str] = []
        seen_source_chunk_ids: set[str] = set()
        for token_unit in (getattr(anchor_mount_result, "items", ()) or ()):
            for chunk_id in (getattr(token_unit, "source_chunk_ids", ()) or ()):
                chunk_value = str(chunk_id).strip()
                if not chunk_value or chunk_value in seen_source_chunk_ids:
                    continue
                seen_source_chunk_ids.add(chunk_value)
                deduped_source_chunk_ids.append(chunk_value)
        return tuple(deduped_source_chunk_ids)

    @staticmethod
    def _resolve_anchor_mount_token_count(
        *,
        stage_result: Optional[Any],
    ) -> int:
        if stage_result is None:
            return 0
        anchor_mount_result = getattr(stage_result, "anchor_mount_result", None)
        if anchor_mount_result is not None:
            return int(len(getattr(anchor_mount_result, "items", ()) or ()))
        return int(len(getattr(stage_result, "final_stream", ()) or ()))

    @classmethod
    def _should_force_fast_direct_on_anchor_mount_fallback(
        cls,
        *,
        stage_result: Optional[AnchorMountStageResult],
    ) -> bool:
        if stage_result is None:
            return False
        anchor_mount_result = getattr(stage_result, "anchor_mount_result", None)
        if anchor_mount_result is None:
            return False
        if not bool(getattr(anchor_mount_result, "should_fallback", False)):
            return False
        source_chunk_ids = cls._resolve_anchor_mount_source_chunk_ids(
            stage_result=stage_result
        )
        if source_chunk_ids and len(source_chunk_ids) > 1:
            return False
        return True

    @classmethod
    def _resolve_anchor_mount_force_fast_reason(
        cls,
        *,
        stage_result: Optional[AnchorMountStageResult],
    ) -> str:
        """将 AnchorMount 的 fallback 质量信号映射为快流直通触发原因。"""
        if stage_result is None:
            return ""
        if cls._should_force_fast_direct_on_anchor_mount_fallback(
            stage_result=stage_result
        ):
            return "anchor_mount_should_fallback"
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

    def _write_selection_trace(
        self,
        *,
        ctx: ProcessingContext,
        selection_outcome: Any,
    ) -> None:
        writer = self._postprocess_trace_writer
        if not writer.enabled:
            return
        writer.write_stage(
            job_dir=ctx.job_dir,
            chunk_index=int(ctx.chunk_index),
            filename="04_selection.input.json",
            payload=selection_outcome.selection_inputs,
            stage="selection_input",
        )
        writer.write_stage(
            job_dir=ctx.job_dir,
            chunk_index=int(ctx.chunk_index),
            filename="05_selection.output.json",
            payload={
                "selected_text_truth": selection_outcome.selected_text_truth,
                "selection_decision": selection_outcome.selection_decision,
                "selection_report": selection_outcome.selection_report,
            },
            stage="selection_output",
        )
        writer.write_layer_summary(
            job_dir=ctx.job_dir,
            layer_summary=selection_outcome.selection_report.summary,
        )

    @staticmethod
    def _resolve_selected_source(ctx: ProcessingContext, *, default: str) -> str:
        selection_decision = getattr(ctx, "selection_decision", None)
        if selection_decision is not None:
            chosen_source = str(
                getattr(selection_decision, "chosen_source", "")
                or getattr(selection_decision, "accepted_text_source", "")
                or ""
            ).strip()
            if chosen_source:
                return chosen_source
        arbitration_result = getattr(ctx, "arbitration_result", None)
        if arbitration_result is not None:
            chosen_source = str(getattr(arbitration_result, "chosen_source", "") or "").strip()
            if chosen_source:
                return chosen_source
        return default

