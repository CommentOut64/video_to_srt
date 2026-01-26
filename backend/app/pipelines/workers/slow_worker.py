"""
SlowWorker - 慢流推理 Worker（GPU）

职责：
1. 执行 Whisper 推理
2. 返回推理结果（不做 Prompt 构建/幻觉检测/对齐）
"""
import logging
from typing import Dict, Optional, Any

from app.core.asr.engine import ASREngine
from app.core.asr.models import ASRResult


class SlowWorker:
    """
    SlowWorker - 慢流推理 Worker（GPU）

    在三级流水线中负责：
    1. Whisper 推理
    2. 返回结果
    """

    def __init__(
        self,
        patch_engine: ASREngine,
        whisper_language: str = "auto",
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化 SlowWorker

        Args:
            patch_engine: 复核引擎（必须提供）
            whisper_language: Whisper 语言设置
            logger: 日志记录器
        """
        if not patch_engine:
            raise ValueError("SlowWorker 需要提供 patch_engine")
        self.patch_engine = patch_engine
        self.whisper_language = whisper_language
        self.logger = logger or logging.getLogger(__name__)

    async def infer(
        self,
        audio: Any,
        initial_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行 Whisper 推理

        Args:
            audio: 音频数组
            initial_prompt: 提示词

        Returns:
            Dict: Whisper 推理结果
        """
        asr_result = await self.patch_engine.transcribe(
            audio,
            language=self.whisper_language,
            initial_prompt=initial_prompt,
            repetition_penalty=None,
            no_repeat_ngram_size=None,
        )
        return self._convert_asr_result(asr_result)

    def _convert_asr_result(self, asr_result: ASRResult) -> Dict[str, Any]:
        """将 ASRResult 转为 Whisper 结果结构。"""
        raw_result = {}
        if asr_result.metadata and asr_result.metadata.raw_tags:
            raw_result = asr_result.metadata.raw_tags.get("raw_result") or {}

        if not raw_result:
            raw_segments = []
            for segment in asr_result.segments:
                raw_segments.append(
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "avg_logprob": -0.5,
                        "no_speech_prob": 0.0,
                        "words": [],
                    }
                )
            raw_result = {
                "text": asr_result.text,
                "segments": raw_segments,
                "language": asr_result.language,
            }

        return {
            "text": asr_result.text,
            "confidence": float(asr_result.confidence or 0.0),
            "language": asr_result.language,
            "raw_result": raw_result,
        }
