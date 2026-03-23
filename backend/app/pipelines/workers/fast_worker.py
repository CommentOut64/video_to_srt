"""
FastWorker - 快流推理 Worker（CPU）

职责：
1. 执行 SenseVoice 推理
2. 填充 ProcessingContext.sv_result
"""
import copy
import logging
from typing import Any, Dict, List, Optional

from app.core.asr.engine import ASREngine
from app.core.asr.models import ASRResult
from app.core.logging import resolve_loguru_logger
from app.schemas.pipeline_context import ProcessingContext
from app.services.audio.chunk_engine import AudioChunk
from app.services.model_runtime_config_service import get_model_runtime_config_service
from app.services.timeanchored_alignment.adapters.fast import SenseVoiceTimeAdapter
from app.services.timeanchored_alignment.time_base_builder import TimeBaseBuilder


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
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化 FastWorker

        Args:
            job_id: 任务 ID
            sensevoice_language: SenseVoice 语言设置
            draft_engine: 草稿引擎（必须提供）
            logger: 日志记录器
        """
        self.job_id = job_id
        self.sensevoice_language = sensevoice_language
        self.logger = resolve_loguru_logger(
            logger,
            __name__,
            job_id=job_id,
            layer="L0",
        )

        if not draft_engine:
            raise ValueError("FastWorker 需要提供 draft_engine")
        self.draft_engine = draft_engine
        self._time_base_builder: Optional[TimeBaseBuilder] = None

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
        log = self.logger.bind(chunk_index=ctx.chunk_index)
        log.debug(f"Chunk {ctx.chunk_index}: SenseVoice 快流推理")
        sv_result = await self._run_sensevoice(chunk)

        # V3.8 修复竞态条件：深拷贝 sv_result，避免下游修改影响其他协程
        ctx.sv_result = copy.deepcopy(sv_result)
        self._maybe_build_time_base(ctx)

    def _maybe_build_time_base(self, ctx: ProcessingContext) -> None:
        """按 feature flag 构建时间基底，不影响现有快流主链。"""
        if not self._is_timeanchored_alignment_enabled():
            return

        if self._time_base_builder is None:
            self._time_base_builder = TimeBaseBuilder(adapter=self._build_time_adapter())

        try:
            ctx.time_base_chunk = self._time_base_builder.build(ctx)
            ctx.time_base_report = {
                "enabled": True,
                "built": ctx.time_base_chunk is not None,
                "chunk_index": ctx.chunk_index,
            }
        except Exception:
            self.logger.exception("Chunk %s: 构建 time_base_chunk 失败，降级继续旧链路", ctx.chunk_index)
            ctx.time_base_chunk = None
            ctx.time_base_report = {
                "enabled": True,
                "built": False,
                "error": "time_base_build_failed",
            }

    def _build_time_adapter(self) -> SenseVoiceTimeAdapter:
        """尽力从 draft_engine 提取 vocab，无法提取时走紧凑轨迹回退。"""
        vocab = None
        blank_id = 0
        service = getattr(self.draft_engine, "service", None)
        decoder = getattr(service, "decoder", None)
        if decoder is not None and hasattr(decoder, "vocab"):
            vocab = getattr(decoder, "vocab", None)
            blank_id = int(getattr(decoder, "blank_id", 0))
        return SenseVoiceTimeAdapter(vocab=vocab, blank_id=blank_id)

    def _is_timeanchored_alignment_enabled(self) -> bool:
        """读取 alignment_pipeline flag（legacy 下线后默认强制开启）。"""
        runtime = get_model_runtime_config_service().get_effective_runtime_global()
        effective = runtime.get("effective", {}) if isinstance(runtime, dict) else {}
        override = runtime.get("override", {}) if isinstance(runtime, dict) else {}

        group: Dict[str, Any] = {}
        if isinstance(effective, dict):
            value = effective.get("alignment_pipeline", {})
            if isinstance(value, dict):
                group.update(value)
        if isinstance(override, dict):
            value = override.get("alignment_pipeline", {})
            if isinstance(value, dict):
                group.update(value)

        version = str(group.get("version", "") or "").lower()
        mode = str(group.get("mode", "") or "").lower()
        enabled = bool(group.get("enabled") or group.get("enable") or group.get("enable_timeanchored"))
        if version == "legacy" or mode in {"legacy", "off"}:
            self.logger.warning(
                "alignment_pipeline 收到 legacy/off 配置，legacy 已下线，强制启用 timeanchored time_base 构建"
            )
            return True
        if version == "timeanchored":
            return True
        if mode in {"shadow", "active", "default", "timeanchored"}:
            return True
        if enabled:
            return True
        return True

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
        
        # V3.2.0+dev.20260203.04: 对齐运行参数 use_itn，避免分支结果不一致
        runtime = get_model_runtime_config_service().get_effective_runtime_global()
        use_itn = runtime.get("effective", {}).get("sensevoice", {}).get("use_itn", True)

        asr_result = await self.draft_engine.transcribe(
            chunk.audio,
            language=language,
            sample_rate=chunk.sample_rate,
            use_itn=use_itn,
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
            log = self.logger.bind(chunk_index=chunk.index)
            log.debug(
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
            log = self.logger.bind(chunk_index=chunk.index)
            log.debug(
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
            log = self.logger.bind(chunk_index=chunk.index)
            log.debug(
                f"Chunk {chunk.index}: 语言融合 LangID={chunk_confidence}, "
                f"SV={sv_language}(conf={sv_confidence:.3f}) -> {best_lang} (score={scores[best_lang]:.3f})"
            )
        else:
            sv_result["language"] = sv_language
            log = self.logger.bind(chunk_index=chunk.index)
            log.debug(
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

        words = None
        if asr_result.words:
            words = []
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
        ctc_compact_trace = None
        ctc_logits = None
        ctc_frame_stride = None
        if asr_result.metadata and asr_result.metadata.raw_tags:
            ctc_compact_trace = asr_result.metadata.raw_tags.get("ctc_compact_trace")
            ctc_logits = asr_result.metadata.raw_tags.get("ctc_logits")
            ctc_frame_stride = asr_result.metadata.raw_tags.get("ctc_frame_stride")

        # V3.2.0+dev.20260203.10: L0 仅透传原始文本与置信度来源
        raw_text = asr_result.text if asr_result.text is not None else None
        result = {
            "raw_text": raw_text,
            "words": words,
            "raw_tokens": raw_tokens,
            "confidence": float(asr_result.confidence) if asr_result.confidence is not None else None,
            "confidence_source": "fast",
            "language": asr_result.language if asr_result.language is not None else None,
            "emotion": asr_result.emotion,
            "event": event_tag,
            # V3.2.2+dev.20260201.01: SenseVoice 语言标签置信度
            "sv_language_info": sv_language_info,
            "source": "fast",
            "segments": None,
        }
        if ctc_compact_trace is not None:
            result["ctc_compact_trace"] = ctc_compact_trace
        if ctc_logits is not None:
            result["ctc_logits"] = ctc_logits
        if ctc_frame_stride is not None:
            result["ctc_frame_stride"] = ctc_frame_stride
        return result

