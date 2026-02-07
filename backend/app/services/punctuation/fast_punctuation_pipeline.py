"""
快流标点编排器（Fast Punctuation Pipeline）。
V3.2.0+dev.20260204.06
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from app.services.alignment.types import NormalizationResult, TextTrack
from app.services.punctuation.base import PunctuationResult, apply_punctuation, build_split_points
from app.services.punctuation.debug_utils import append_debug_punctuation_line
from app.services.punctuation.postprocess import (
    PunctuationPostprocessResult,
    get_postprocess_config,
    postprocess_punctuation,
)
from app.services.punctuation.punctuation_processor import PunctuationProcessor
from app.services.punctuation.scheduler import get_punctuation_scheduler
from app.services.sse_service import get_sse_manager

if TYPE_CHECKING:
    from app.services.audio.chunk_engine import AudioChunk
    from app.schemas.pipeline_context import ProcessingContext
    from app.services.punctuation.service import PunctuationService


class FastPunctuationPipeline:
    """快流标点编排器（外观模式：统一标点恢复/后处理/调度）。"""

    def __init__(
        self,
        job_id: str,
        punctuation_service: Optional["PunctuationService"],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.job_id = job_id
        self.punctuation_service = punctuation_service
        self.logger = logger or logging.getLogger(__name__)
        self._punctuation_scheduler = get_punctuation_scheduler()
        # V3.2.0+dev.20260204.06: 统一入口 - 快流标点也通过 L3Processor 产出 PunctTrack
        self._l3_processor = PunctuationProcessor(
            punctuation_service=punctuation_service,
            logger=self.logger,
        )

    async def apply(
        self,
        sv_result: Dict[str, Any],
        *,
        chunk: "AudioChunk",
        ctx: Optional["ProcessingContext"],
        normalization: NormalizationResult,
    ) -> Dict[str, Any]:
        """执行快流标点恢复与后处理，并写入元信息。"""
        if not sv_result or not self.punctuation_service:
            return sv_result

        raw_text = normalization.text_itn_raw or ""
        clean_text = normalization.text_clean or raw_text
        if not clean_text:
            return sv_result

        words = sv_result.get("words", [])
        language = chunk.language or sv_result.get("language") or "auto"

        try:
            chosen_track = self._resolve_chosen_track(ctx, sv_result, normalization, language)
            output = await self._l3_processor.process(
                data=self._l3_input(chosen_track, words)
            )
        except Exception as exc:
            self.logger.warning("快流标点处理失败，继续主流程: %s", exc)
            return sv_result

        punct_track = output.punct_track
        # 兼容：仍写入旧 metadata 结构，供 SemanticBuffer/debug 使用
        metadata = sv_result.setdefault("metadata", {})
        positions = list(punct_track.positions or [])
        post_config = get_postprocess_config("fast")
        post = postprocess_punctuation(
            raw_text=raw_text,
            clean_text=clean_text,
            raw_to_clean=normalization.raw_to_clean,
            clean_to_raw=normalization.clean_to_raw,
            words=words,
            language=language,
            mode="fast",
            candidates=positions,
            config=post_config,
        )
        model_id = punct_track.confidence_stats.get("model_id", "")
        processing_time_ms = punct_track.confidence_stats.get("processing_time_ms", 0.0)
        model_confidence = punct_track.confidence_stats.get("model_confidence", 0.0)
        model_result = PunctuationResult(
            text=apply_punctuation(clean_text, post.final_positions),
            split_points=[],
            punctuation_positions=[],
            confidence=float(model_confidence or 0.0),
            model_id=str(model_id or ""),
            processing_time_ms=float(processing_time_ms or 0.0),
        )
        metadata["punctuation"] = self._postprocess_result_to_dict(
            model_result,
            post,
            words,
            mode="fast",
        )

        decision = self._punctuation_scheduler.evaluate_fast(
            self._build_postprocess_result(model_result, post, words),
            sv_confidence=sv_result.get("confidence"),
        )
        metadata["punctuation_decision"] = {
            "is_slow_requested": decision.is_slow_requested,
            "reason": decision.reason,
            "mode": self._punctuation_scheduler.policy.mode.value,
        }
        if ctx is not None:
            ctx.punct_track = punct_track
        self._emit_debug_outputs(ctx, raw_text, metadata, sv_result.get("confidence"))
        return sv_result

    @staticmethod
    def _resolve_chosen_track(
        ctx: Optional["ProcessingContext"],
        sv_result: Dict[str, Any],
        normalization: NormalizationResult,
        language: str,
    ) -> TextTrack:
        if ctx is not None and getattr(ctx, "text_tracks", None) is not None:
            track = getattr(ctx.text_tracks, "sv_track", None)
            if track is not None:
                return track
        raw_text = str(sv_result.get("raw_text") or normalization.text_itn_raw or "")
        return TextTrack(
            raw_text=raw_text,
            text_itn_raw=normalization.text_itn_raw,
            text_clean=normalization.text_clean or normalization.text_itn_raw,
            char_mapping=normalization.char_mapping,
            raw_to_clean=normalization.raw_to_clean,
            clean_to_raw=normalization.clean_to_raw,
            language=language or "auto",
            source="sv",
            itn_fallback=normalization.itn_fallback,
            itn_fallback_reason=normalization.itn_fallback_reason,
            mapping_coverage=normalization.mapping_coverage,
        )

    @staticmethod
    def _l3_input(track: TextTrack, words: List[Dict[str, Any]]):
        # 延迟导入避免循环依赖
        from app.services.alignment.types import L3Input

        return L3Input(
            chosen_text_track=track,
            sv_punct_source=None,
            wh_punct_source=None,
            word_timestamps=words,
        )

    def _emit_debug_outputs(
        self,
        ctx: Optional["ProcessingContext"],
        raw_text: str,
        metadata: Dict[str, Any],
        sv_confidence: Optional[float],
    ) -> None:
        """输出标点调试信息（SSE + 文件）。"""
        if not ctx or not getattr(ctx, "debug_punctuation", False):
            return

        punctuation = metadata.get("punctuation", {}) if isinstance(metadata, dict) else {}
        if not punctuation:
            return

        payload = {
            "chunk_id": f"chunk-{ctx.chunk_index}",
            "original_text": raw_text,
            "punctuated_text": punctuation.get("text", ""),
            "split_points": punctuation.get("split_points", []),
            "model_id": punctuation.get("model_id", ""),
            "processing_time_ms": punctuation.get("processing_time_ms", 0.0),
            "confidence": punctuation.get("confidence", 0.0),
        }

        sse_manager = get_sse_manager()
        sse_manager.broadcast_sync(f"job:{self.job_id}", "debug.punctuation", payload)
        append_debug_punctuation_line(ctx.job_dir, payload, logger=self.logger)

        decision = metadata.get("punctuation_decision", {})
        if isinstance(decision, dict):
            scheduler_payload = {
                "chunk_id": payload["chunk_id"],
                "is_slow_requested": bool(decision.get("is_slow_requested", False)),
                "reason": str(decision.get("reason", "")),
                "mode": str(decision.get("mode", "")),
                "punct_confidence": punctuation.get("confidence", 0.0),
                "sv_confidence": sv_confidence,
            }
            sse_manager.broadcast_sync(
                f"job:{self.job_id}",
                "debug.punctuation_scheduler",
                scheduler_payload,
            )

    def _postprocess_result_to_dict(
        self,
        model_result: PunctuationResult,
        post_result: PunctuationPostprocessResult,
        words: List[Dict[str, Any]],
        *,
        mode: str,
    ) -> Dict[str, Any]:
        """将后处理结果转换为可序列化结构。"""
        positions = post_result.final_positions
        split_points = build_split_points(
            text=post_result.final_text,
            positions=positions,
            word_timestamps=words,
        )
        return {
            "text": post_result.final_text,
            "split_points": [
                {
                    "char_index": point.char_index,
                    "relative_time": point.relative_time,
                    "punctuation": point.punctuation,
                    "confidence": point.confidence,
                }
                for point in split_points
            ],
            "punctuation_positions": [
                {
                    "char_index": position.char_index,
                    "punctuation": position.punctuation,
                    "confidence": position.confidence,
                }
                for position in positions
            ],
            "confidence": self._estimate_positions_confidence(positions, model_result.confidence),
            "model_id": model_result.model_id,
            "processing_time_ms": model_result.processing_time_ms,
            "postprocess": {
                "mode": mode,
                "decision_log": [
                    {
                        "char_index": decision.char_index,
                        "punctuation": decision.punctuation,
                        "action": decision.action,
                        "reason": decision.reason,
                        "raw_conf": decision.raw_conf,
                        "cand_conf": decision.cand_conf,
                    }
                    for decision in post_result.decision_log
                ],
                "metrics": post_result.metrics,
            },
        }

    @staticmethod
    def _estimate_positions_confidence(
        positions: List[Any],
        fallback: float,
    ) -> float:
        """估算后处理标点置信度。"""
        if not positions:
            return float(fallback or 0.0)
        return sum(float(pos.confidence) for pos in positions) / len(positions)

    @staticmethod
    def _build_postprocess_result(
        model_result: PunctuationResult,
        post_result: PunctuationPostprocessResult,
        words: List[Dict[str, Any]],
    ) -> PunctuationResult:
        """构造用于调度器的标点结果。"""
        positions = post_result.final_positions
        split_points = build_split_points(
            text=post_result.final_text,
            positions=positions,
            word_timestamps=words,
        )
        return PunctuationResult(
            text=post_result.final_text,
            split_points=split_points,
            punctuation_positions=positions,
            confidence=FastPunctuationPipeline._estimate_positions_confidence(positions, model_result.confidence),
            model_id=model_result.model_id,
            processing_time_ms=model_result.processing_time_ms,
        )
