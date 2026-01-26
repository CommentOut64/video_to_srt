"""
FastWorker - 快流推理 Worker（CPU）

职责：
1. 执行 SenseVoice 推理
2. 填充 ProcessingContext.sv_result
"""
import copy
import logging
from typing import Dict, Optional, Any

from app.core.asr.engine import ASREngine
from app.core.asr.models import ASRResult
from app.schemas.pipeline_context import ProcessingContext
from app.services.audio.chunk_engine import AudioChunk


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
        self.logger = logger or logging.getLogger(__name__)

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
        asr_result = await self.draft_engine.transcribe(
            chunk.audio,
            language=self.sensevoice_language,
            sample_rate=chunk.sample_rate,
            use_itn=True,
        )
        return self._convert_asr_result(asr_result)

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
                        "is_pseudo": word.is_pseudo,
                    }
                )

        return {
            "text": asr_result.text,
            "text_clean": asr_result.text_clean or asr_result.text,
            "words": words,
            "confidence": float(asr_result.confidence or 0.0),
            "language": asr_result.language or self.sensevoice_language,
            "emotion": asr_result.emotion,
            "event": event_tag,
        }
