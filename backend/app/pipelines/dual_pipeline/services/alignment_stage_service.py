"""
对齐阶段门面服务。

设计模式：Facade Pattern。
原因：收口对齐阶段的大段编排逻辑，让主实现类聚焦流水线调度与依赖装配。
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, replace
from typing import Any, Dict, Optional

from app.models.sensevoice_models import SentenceSegment, TextSource
from app.pipelines.dual_pipeline.services.textflow_facade_service import Layer456RunResult
from app.schemas.pipeline_context import ProcessingContext
from app.services.alignment.types import PunctTrack
from app.services.language_policy import build_language_policy_snapshot
from app.services.timeanchored_alignment import (
    ChunkWindow,
    ChunkProjector,
    EdgeSelector,
    LanguageRunFrontend,
    OutputAdapter,
    PhoneticAligner,
    PronunciationFrontend,
    SentenceSegmenter,
    SubtitleAssembler,
    TimeanchoredAlignmentStageService,
    TimeanchoredStageResult,
    TextAligner,
)
from app.services.timeanchored_alignment.adapters.slow import WhisperTextAdapter


class AlignmentStageService:
    """双流对齐阶段执行门面。"""

    def __init__(self, *, host: Any) -> None:
        self._host = host
        self._timeanchored_whisper_adapter = WhisperTextAdapter(
            sanitizer=getattr(host, "_whisper_sanitizer", None),
            hallucination_detector=getattr(host, "_hallucination_detector", None),
        )
        self._timeanchored_language_frontend = LanguageRunFrontend()
        self._timeanchored_pronunciation_frontend = PronunciationFrontend(
            language_run_frontend=self._timeanchored_language_frontend,
        )
        self._timeanchored_text_aligner = TextAligner(
            phonetic_aligner=PhoneticAligner(),
        )
        self._timeanchored_edge_selector = EdgeSelector()
        self._timeanchored_subtitle_assembler = SubtitleAssembler()
        self._timeanchored_sentence_segmenter = SentenceSegmenter()
        self._timeanchored_chunk_projector = ChunkProjector()
        self._timeanchored_output_adapter = OutputAdapter()
        self._timeanchored_stage_service = TimeanchoredAlignmentStageService(
            text_aligner=self._timeanchored_text_aligner,
            edge_selector=self._timeanchored_edge_selector,
            subtitle_assembler=self._timeanchored_subtitle_assembler,
            sentence_segmenter=self._timeanchored_sentence_segmenter,
            chunk_projector=self._timeanchored_chunk_projector,
            output_adapter=self._timeanchored_output_adapter,
        )

    async def run(self, ctx: ProcessingContext) -> None:
        """执行单个 chunk 的对齐阶段。"""
        host = self._host
        chunk = ctx.audio_chunk

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
    ) -> Optional[TimeanchoredStageResult]:
        host = self._host
        chunk = ctx.audio_chunk
        try:
            base_language = str(language_hint or "auto")
            text_truth = self._timeanchored_whisper_adapter.build_text_truth_package(
                whisper_result=whisper_result,
                default_language=base_language,
            )
            truth_text = str(text_truth.normalized_text or text_truth.raw_text or "").strip()
            if not truth_text:
                truth_text = self._resolve_text_fallback_content(
                    chosen_text_clean=host._select_text_for_alignment(
                        getattr(tracks, "chosen_track", None)
                    ),
                    tracks=tracks,
                    whisper_result=whisper_result,
                    sv_result=sv_result,
                )
                text_truth = replace(
                    text_truth,
                    raw_text=truth_text,
                    normalized_text=truth_text,
                )

            from app.services.text_protection import extract_protected_spans

            protected_spans = tuple(extract_protected_spans(truth_text))
            text_truth = replace(text_truth, protected_spans=protected_spans)
            language_runs = self._timeanchored_language_frontend.build_runs(
                text=truth_text,
                language_hint=base_language,
            )
            pronunciation = self._timeanchored_pronunciation_frontend.build_package(
                text=truth_text,
                language_hint=base_language,
                language_runs=language_runs.runs,
                dominant_language=language_runs.dominant_language,
            )

            ctx.text_truth = text_truth
            ctx.protected_spans = list(text_truth.protected_spans)
            ctx.language_runs = list(language_runs.runs)
            ctx.pronunciation_package = pronunciation
            ctx.pronunciation_report = {
                "token_count": len(pronunciation.token_units),
                "phone_count": len(pronunciation.phone_units),
                "window_kind": language_runs.window_kind,
                "foreign_run_ratio": float(language_runs.foreign_run_ratio),
            }

            chunk_start = float(getattr(chunk, "start", 0.0) or 0.0) if chunk is not None else 0.0
            chunk_end = (
                float(getattr(chunk, "end", chunk_start) or chunk_start)
                if chunk is not None
                else chunk_start
            )
            if chunk_end <= chunk_start:
                chunk_end = chunk_start + 0.01
            ctx_edge_mode = str(getattr(ctx, "edge_selection_mode", "") or "").strip().lower()
            host_edge_mode = str(getattr(host, "_edge_selection_mode", "auto") or "auto").strip().lower()
            edge_mode = ctx_edge_mode if ctx_edge_mode in {"force_fast", "force_slow"} else host_edge_mode

            return self._timeanchored_stage_service.execute(
                time_base=ctx.time_base_chunk,
                text_truth=text_truth,
                language_runs=language_runs,
                pronunciation=pronunciation,
                chunk_window=ChunkWindow(
                    chunk_ref=ctx.chunk_index,
                    start=chunk_start,
                    end=chunk_end,
                ),
                language=base_language,
                edge_selection_mode=edge_mode,
                speaker_id=speaker_id,
                turn_id=turn_id,
            )
        except Exception:
            host.logger.exception(
                "Chunk {}: timeanchored 主链失败（legacy 已下线）",
                ctx.chunk_index,
            )
            return None

    def _commit_timeanchored_main_chain_result(
        self,
        *,
        ctx: ProcessingContext,
        stage_result: TimeanchoredStageResult,
        whisper_result: Dict[str, Any],
        sv_result: Dict[str, Any],
        speaker_id: Optional[str],
        turn_id: Optional[str],
    ) -> None:
        host = self._host
        final_sentences = list(stage_result.sentence_segments)
        fallback_error_code = ""
        if not final_sentences:
            fallback_text = self._timeanchored_sentence_segmenter.compose_text(
                stream=stage_result.final_stream,
                language=str(
                    getattr(ctx.text_truth, "language", "")
                    or whisper_result.get("language")
                    or "auto"
                ),
            ).strip()
            if not fallback_text:
                fallback_text = str(
                    getattr(ctx.text_truth, "normalized_text", "")
                    or getattr(ctx.text_truth, "raw_text", "")
                    or ""
                ).strip()
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

        for sentence in final_sentences:
            if getattr(sentence, "speaker_id", None) is None:
                sentence.speaker_id = speaker_id
            if getattr(sentence, "turn_id", None) is None:
                sentence.turn_id = turn_id
        host._assign_sentence_identity_by_timeline_overlap(
            final_sentences,
            fallback_chunk=ctx.audio_chunk,
        )

        output_layer_result = host._emit_output_layer(
            chunk_index=ctx.chunk_index,
            sentence_segments=final_sentences,
            language=str(
                getattr(ctx.text_truth, "language", "")
                or whisper_result.get("language")
                or "auto"
            ),
            injection_report={
                "mapping_coverage": float(stage_result.base_result.metrics.coverage),
                "error_code": str(stage_result.base_result.error_code or ""),
            },
            segmentation_report={
                "route": "timeanchored",
                "text_route": stage_result.text_result.route,
                "edge_route": stage_result.edge_result.route,
                "error_code": str(stage_result.base_result.error_code or fallback_error_code),
            },
            output_traces=[],
            default_trace_reason="timeanchored_chain",
        )
        ctx.final_sentences = list(final_sentences)
        ctx.finalization_metrics = {
            "coverage": float(stage_result.base_result.metrics.coverage),
            "gap_ratio": 0.0,
            "alignment_score": float(stage_result.base_result.metrics.route_confidence),
            "gap_positions": [],
            "gap_resolution": None,
            "timeanchored_enabled": 1.0,
            "timeanchored_text_route": stage_result.text_result.route,
            "timeanchored_edge_route": stage_result.edge_result.route,
            "timeanchored_final_route": stage_result.base_result.route,
            "timeanchored_item_count": float(len(stage_result.final_stream)),
            "timeanchored_sentence_count": float(len(final_sentences)),
            "timeanchored_failed_span_count": float(len(stage_result.failed_spans)),
            "l7_error_count": float(len(output_layer_result.output_payload.get("errors", []))),
        }
        if fallback_error_code:
            ctx.finalization_metrics["timeanchored_error_code"] = fallback_error_code
        if ctx.arbitration_result:
            ctx.arbitration_result.gap_positions = []
        host.logger.debug(
            "Chunk {}: timeanchored 主链完成 sentences={} route={}",
            ctx.chunk_index,
            len(final_sentences),
            stage_result.base_result.route,
        )

    def _record_hetero_alignment_result(
        self,
        *,
        ctx: ProcessingContext,
        stage_result: Optional[TimeanchoredStageResult],
        mode: str,
        selected: bool,
        reason: str,
    ) -> None:
        route = str(stage_result.base_result.route) if stage_result is not None else "unavailable"
        ctx.hetero_route = route
        ctx.hetero_alignment_result = {
            "mode": str(mode),
            "selected": bool(selected),
            "reason": str(reason),
            "route": route,
            "sentence_count": (
                int(len(stage_result.sentence_segments))
                if stage_result is not None
                else 0
            ),
            "failed_span_count": (
                int(len(stage_result.failed_spans))
                if stage_result is not None
                else 0
            ),
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
    def _should_accept_timeanchored_result(
        *,
        stage_result: Optional[TimeanchoredStageResult],
        mode: str,
    ) -> tuple[bool, str]:
        if stage_result is None:
            return False, f"{mode}_timeanchored_failed"
        route = str(stage_result.base_result.route or "")
        if route == "error":
            return False, f"{mode}_gate_route_error"
        if not stage_result.final_stream:
            return False, f"{mode}_gate_empty_stream"
        if not stage_result.sentence_segments:
            return False, f"{mode}_gate_empty_sentences"
        if mode == "default":
            return True, "default_gate_pass"
        return True, f"{mode}_gate_pass"

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

