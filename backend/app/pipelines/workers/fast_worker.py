"""
FastWorker - 快流推理 Worker（CPU）

职责：
1. 执行 SenseVoice 推理
2. 填充 ProcessingContext.sv_result
"""
import copy
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from app.core.asr.engine import ASREngine
from app.core.asr.models import ASRResult
from app.schemas.pipeline_context import ProcessingContext
from app.services.audio.chunk_engine import AudioChunk
from app.services.punctuation.base import PunctuationResult, build_split_points
from app.services.punctuation.debug_utils import append_debug_punctuation_line
from app.services.punctuation.postprocess import (
    PunctuationPostprocessResult,
    build_clean_text,
    get_postprocess_config,
    postprocess_punctuation,
)
from app.services.punctuation.scheduler import get_punctuation_scheduler
from app.services.sse_service import get_sse_manager

if TYPE_CHECKING:
    from app.services.punctuation.service import PunctuationService


class FastWorker:
    """
    FastWorker - 快流推理 Worker（CPU）

    在三级流水线中负责：
    1. SenseVoice 推理
    2. 输出 sv_result（不做分句/推送）
    """

    def __init__(
        self,
        job_id: str,
        draft_engine: ASREngine,
        sensevoice_language: str = "auto",
        punctuation_service: Optional["PunctuationService"] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化 FastWorker

        Args:
            job_id: 任务 ID
            sensevoice_language: SenseVoice 语言设置
            draft_engine: 草稿引擎（必须提供）
            punctuation_service: 标点服务（可选）
            logger: 日志记录器
        """
        self.job_id = job_id
        self.sensevoice_language = sensevoice_language
        self.punctuation_service = punctuation_service
        self.logger = logger or logging.getLogger(__name__)
        self._punctuation_scheduler = get_punctuation_scheduler()

        if not draft_engine:
            raise ValueError("FastWorker 需要提供 draft_engine")
        self.draft_engine = draft_engine

    async def process(self, ctx: ProcessingContext):
        """
        处理单个 Chunk（快流）

        流程：
        1. SenseVoice 推理
        2. 填充 ctx.sv_result

        Args:
            ctx: 处理上下文
        """
        chunk = ctx.audio_chunk
        self.logger.debug(f"Chunk {ctx.chunk_index}: SenseVoice 快流推理")
        sv_result = await self._run_sensevoice(chunk)

        # V3.2.0+dev.20260130.01: FastWorker 标点集成 + 统一后处理（可选）
        if self.punctuation_service:
            sv_result = await self._apply_punctuation(sv_result, chunk, ctx)

        # V3.8 修复竞态条件：深拷贝 sv_result，避免下游修改影响其他协程
        ctx.sv_result = copy.deepcopy(sv_result)

    async def _run_sensevoice(self, chunk: AudioChunk) -> Dict[str, Any]:
        """
        运行 SenseVoice 推理

        Args:
            chunk: AudioChunk

        Returns:
            Dict: SenseVoice 推理结果
        """
        # V3.2.2+dev.20260201.02: 注入 chunk 的语言标签到 SenseVoice
        # 优先使用 chunk.language（LangID 检测结果），回退到全局设置
        language = chunk.language or self.sensevoice_language
        
        asr_result = await self.draft_engine.transcribe(
            chunk.audio,
            language=language,
            sample_rate=chunk.sample_rate,
            use_itn=True,
        )
        result = self._convert_asr_result(asr_result)

        # V3.2.2+dev.20260201.01: 语言融合逻辑
        result = self._fuse_language(chunk, result)

        return result

    def _fuse_language(
        self,
        chunk: AudioChunk,
        sv_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        V3.2.2+dev.20260201.01: 融合 LangID 和 SenseVoice 的语言检测结果

        策略：
        1. 如果 Chunk 的 language 不是 auto，则透传 Chunk 的语言标签
        2. 如果 Chunk 的 language 是 auto，则综合 language_confidence 和
           SenseVoice 语言标签打分决定最终语言

        打分规则：
        - LangID Top-2 置信度：直接作为分数
        - SenseVoice 语言标签：如果匹配则加分（基于 sv_language_info.confidence）

        Args:
            chunk: AudioChunk，包含 LangID 检测结果
            sv_result: SenseVoice 转录结果

        Returns:
            更新后的 sv_result
        """
        chunk_language = chunk.language or "auto"
        chunk_confidence = chunk.language_confidence or {}

        # 策略 1：非 auto 时透传
        if chunk_language != "auto":
            sv_result["language"] = chunk_language
            self.logger.debug(
                f"Chunk {chunk.index}: 透传 LangID 语言={chunk_language}"
            )
            return sv_result

        # 策略 2：auto 时融合判断
        sv_language = sv_result.get("language", "auto")
        sv_language_info = sv_result.get("sv_language_info")
        sv_confidence = 0.0
        if sv_language_info and isinstance(sv_language_info, dict):
            sv_confidence = sv_language_info.get("confidence", 0.0)

        # 如果 LangID 没有结果，直接使用 SenseVoice 的结果
        if not chunk_confidence:
            self.logger.debug(
                f"Chunk {chunk.index}: LangID 无结果，使用 SenseVoice 语言={sv_language}"
            )
            return sv_result

        # 打分融合：LangID 置信度 + SenseVoice 匹配加分
        # SenseVoice 匹配加分权重：0.3（可调整）
        SV_MATCH_WEIGHT = 0.3
        scores: Dict[str, float] = {}

        for lang, conf in chunk_confidence.items():
            scores[lang] = conf
            # 如果 SenseVoice 检测到同一语言，加分
            if lang == sv_language and sv_confidence > 0:
                scores[lang] += SV_MATCH_WEIGHT * sv_confidence

        # 选择得分最高的语言（得分相同时按语言代码字典序）
        if scores:
            best_lang = max(scores.keys(), key=lambda k: (scores[k], -ord(k[0]) if k else 0))
            sv_result["language"] = best_lang
            self.logger.debug(
                f"Chunk {chunk.index}: 语言融合 LangID={chunk_confidence}, "
                f"SV={sv_language}(conf={sv_confidence:.3f}) -> {best_lang} (score={scores[best_lang]:.3f})"
            )
        else:
            sv_result["language"] = sv_language
            self.logger.debug(
                f"Chunk {chunk.index}: 无有效分数，使用 SenseVoice 语言={sv_language}"
            )

        return sv_result

    def _convert_asr_result(self, asr_result: ASRResult) -> Dict[str, Any]:
        """将 ASRResult 转为旧的 SenseVoice 结果结构。"""
        event_tag = None
        if asr_result.event_tags:
            event_tag = " ".join(asr_result.event_tags)
        elif asr_result.metadata and asr_result.metadata.raw_tags:
            event_tag = asr_result.metadata.raw_tags.get("event_tag")

        words = []
        if asr_result.words:
            for word in asr_result.words:
                confidence = word.confidence if word.confidence is not None else 1.0
                words.append(
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "confidence": confidence,
                        "confidence_raw": word.confidence,
                        "confidence_display_raw": word.confidence_display_raw,
                        "is_pseudo": word.is_pseudo,
                        "token_type": word.token_type,
                    }
                )

        raw_tokens = asr_result.raw_tokens
        if raw_tokens is None and asr_result.metadata and asr_result.metadata.raw_tags:
            raw_tokens = asr_result.metadata.raw_tags.get("raw_tokens")

        # V3.2.2+dev.20260201.01: 提取 SenseVoice 语言标签置信度
        sv_language_info = None
        if asr_result.metadata and asr_result.metadata.raw_tags:
            sv_language_info = asr_result.metadata.raw_tags.get("sv_language_info")

        return {
            "text": asr_result.text,
            "text_clean": asr_result.text_clean or asr_result.text,
            "words": words,
            "raw_tokens": raw_tokens,
            "confidence": float(asr_result.confidence or 0.0),
            "language": asr_result.language or self.sensevoice_language,
            "emotion": asr_result.emotion,
            "event": event_tag,
            # V3.2.2+dev.20260201.01: SenseVoice 语言标签置信度
            "sv_language_info": sv_language_info,
        }

    async def _apply_punctuation(
        self,
        sv_result: Dict[str, Any],
        chunk: AudioChunk,
        ctx: Optional[ProcessingContext] = None,
    ) -> Dict[str, Any]:
        """调用标点服务并写入结果元信息。"""
        if not sv_result:
            return sv_result

        words = sv_result.get("words", [])
        cleaned_words = words

        raw_text = sv_result.get("text_clean") or sv_result.get("text") or ""
        clean_text, _, _ = build_clean_text(raw_text)
        if not clean_text:
            clean_text = raw_text
        language = chunk.language or sv_result.get("language") or self.sensevoice_language

        try:
            result = await self.punctuation_service.restore(
                text=clean_text,
                language=language,
                word_timestamps=cleaned_words,
            )
        except Exception as exc:
            self.logger.warning("FastWorker 标点恢复失败，继续主流程: %s", exc)
            return sv_result

        if result:
            metadata = sv_result.setdefault("metadata", {})
            post_config = get_postprocess_config("fast")
            post = postprocess_punctuation(
                raw_text=raw_text,
                words=cleaned_words,
                language=language,
                mode="fast",
                candidates=result.punctuation_positions,
                config=post_config,
            )
            metadata["punctuation"] = self._postprocess_result_to_dict(
                result,
                post,
                cleaned_words,
                mode="fast",
            )
            # V3.2.0+dev.20260129.02: 写入快流标点调度决策
            decision = self._punctuation_scheduler.evaluate_fast(
                self._build_postprocess_result(result, post, cleaned_words),
                sv_confidence=sv_result.get("confidence"),
            )
            metadata["punctuation_decision"] = {
                "is_slow_requested": decision.is_slow_requested,
                "reason": decision.reason,
                "mode": self._punctuation_scheduler.policy.mode.value,
            }
            self._emit_debug_outputs(ctx, raw_text, metadata, sv_result.get("confidence"))
        return sv_result

    def _emit_debug_outputs(
        self,
        ctx: Optional[ProcessingContext],
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

    def _merge_punctuation_timestamps(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并标点 token 的时间戳并移除标点 token。"""
        if not words:
            return []

        punctuation_set = set(",.!?;:'\"()[]{}，。！？；：、（）【】《》“”‘’「」『』")
        left_punct = set("([{“‘（【《「『")
        cleaned: List[Dict[str, Any]] = []
        words_copy = [word.copy() for word in words]

        for idx, word in enumerate(words_copy):
            token = word.get("word", "")
            if token in punctuation_set:
                merge_to_next = token in left_punct
                if merge_to_next:
                    merged = False
                    for next_word in words_copy[idx + 1:]:
                        if next_word.get("word", "") not in punctuation_set:
                            next_start = next_word.get("start", 0.0)
                            next_word["start"] = min(next_start, float(word.get("start", next_start)))
                            merged = True
                            break
                    if merged:
                        continue
                if cleaned:
                    prev = cleaned[-1]
                    prev_end = prev.get("end", 0.0)
                    prev["end"] = max(prev_end, float(word.get("end", prev_end)))
                else:
                    for next_word in words_copy[idx + 1:]:
                        if next_word.get("word", "") not in punctuation_set:
                            next_start = next_word.get("start", 0.0)
                            next_word["start"] = min(next_start, float(word.get("start", next_start)))
                            break
                continue
            cleaned.append(word)

        return cleaned

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
            confidence=FastWorker._estimate_positions_confidence(positions, model_result.confidence),
            model_id=model_result.model_id,
            processing_time_ms=model_result.processing_time_ms,
        )
