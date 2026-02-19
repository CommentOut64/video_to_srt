"""
对齐阶段门面服务。

设计模式：Facade Pattern。
原因：收口对齐阶段的大段编排逻辑，让主实现类聚焦流水线调度与依赖装配。
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from app.models.sensevoice_models import SentenceSegment, TextSource
from app.pipelines.dual_pipeline.services.textflow_facade_service import Layer456RunResult
from app.schemas.pipeline_context import ProcessingContext
from app.services.alignment.types import PunctTrack
from app.services.language_policy import build_language_policy_snapshot


class AlignmentStageService:
    """双流对齐阶段执行门面。"""

    def __init__(self, *, host: Any) -> None:
        self._host = host

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
        if arbitration_output.chosen_text_track:
            tracks.chosen_track = arbitration_output.chosen_text_track
        elif tracks.chosen_track is None:
            fallback_track = tracks.sv_track or tracks.whisper_track
            if fallback_track:
                tracks.chosen_track = host._clone_text_track(
                    fallback_track,
                    source="chosen",
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

        if host._is_dual_time_experiment_enabled:
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

        host.logger.debug(
            f"Chunk {ctx.chunk_index}: 定稿已推送 "
            f"({len(final_sentences)} 个句子)"
        )

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

