"""
SlowWorker - 慢流推理 Worker（GPU）

职责：
1. 执行 Whisper 推理
2. 返回推理结果（不做 Prompt 构建/幻觉检测/对齐）
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any, List, TYPE_CHECKING

from app.core.asr.engine import ASREngine
from app.core.asr.models import ASRResult
from app.models.sensevoice_models import SentenceSegment
from app.services.bridge.batch_builder import BridgeBatch
from app.services.punctuation.base import PunctuationResult
from app.services.punctuation.semantic_buffer import PunctuationDecision
from app.services.whisper_buffer_pool import WhisperBufferPool, WhisperBufferConfig

if TYPE_CHECKING:
    from app.services.punctuation.service import PunctuationService


@dataclass
class SlowWorkerResult:
    """SlowWorker 批次处理结果。"""

    batch_id: str
    whisper_result: Dict[str, Any]
    source_sentences: List[SentenceSegment]
    slow_punctuation: Optional[PunctuationResult]
    punctuation_decision: Optional[PunctuationDecision]


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
        punctuation_service: Optional["PunctuationService"] = None,
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
        self.punctuation_service = punctuation_service
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

    async def process_batch(
        self,
        batch: BridgeBatch,
        *,
        full_audio_array: Any,
        full_audio_sr: int = 16000,
    ) -> SlowWorkerResult:
        """
        处理 Bridge 批次（V3.2.0+dev.20260201.07）。

        Args:
            batch: Bridge 批次
            full_audio_array: 完整音频数组（16kHz）
            full_audio_sr: 完整音频采样率
        """
        if not batch:
            raise ValueError("SlowWorker 批次为空")
        audio = self._concat_batch_audio(batch, full_audio_array, full_audio_sr)
        whisper_lang = batch.language or self.whisper_language
        prompt = batch.prompt or None

        asr_result = await self.patch_engine.transcribe(
            audio,
            language=whisper_lang,
            initial_prompt=prompt,
            word_timestamps=True,
            vad_filter=False,
            repetition_penalty=None,
            no_repeat_ngram_size=None,
        )
        whisper_result = self._convert_asr_result(asr_result)

        slow_punct_result: Optional[PunctuationResult] = None
        decision = batch.punctuation_decision
        if (
            self.punctuation_service
            and decision
            and decision.is_slow_requested
            and whisper_result.get("text")
        ):
            word_timestamps = self._extract_word_timestamps(whisper_result)
            slow_punct_result = await self.punctuation_service.restore(
                text=str(whisper_result.get("text", "")),
                language=whisper_lang,
                word_timestamps=word_timestamps if word_timestamps else None,
            )

        return SlowWorkerResult(
            batch_id=batch.batch_id,
            whisper_result=whisper_result,
            source_sentences=batch.sentences or [],
            slow_punctuation=slow_punct_result,
            punctuation_decision=decision,
        )

    def _concat_batch_audio(
        self,
        batch: BridgeBatch,
        full_audio_array: Any,
        full_audio_sr: int,
    ) -> Any:
        """使用 WhisperBufferPool 拼接批次音频（保留间隔）。"""
        if full_audio_array is None:
            raise ValueError("SlowWorker 批次拼接需要完整音频数组")
        if not batch.audio_segments:
            raise ValueError("SlowWorker 批次缺少音频片段")

        segments = sorted(batch.audio_segments, key=lambda item: item[0])
        config = WhisperBufferConfig()
        pool = WhisperBufferPool(config)
        # 采样率不匹配会导致间隔长度偏差，强制同步
        if getattr(pool, "_sample_rate", None) != full_audio_sr:
            pool._sample_rate = full_audio_sr

        audio_len = len(full_audio_array)
        for idx, (start, end) in enumerate(segments):
            start_sample = max(0, int(start * full_audio_sr))
            end_sample = min(audio_len, int(end * full_audio_sr))
            if end_sample <= start_sample:
                continue
            audio_slice = full_audio_array[start_sample:end_sample]
            pool.add_chunk(
                index=idx,
                start=float(start),
                end=float(end),
                audio=audio_slice,
            )

        if pool.is_empty:
            raise ValueError("SlowWorker 批次拼接失败：有效音频片段为空")

        concatenated, _, _ = pool.get_concatenated_audio(preserve_gaps=True)
        return concatenated

    @staticmethod
    def _extract_word_timestamps(whisper_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从 Whisper raw_result 中提取词级时间戳。"""
        raw = whisper_result.get("raw_result") if isinstance(whisper_result, dict) else None
        segments = raw.get("segments", []) if isinstance(raw, dict) else []
        words: List[Dict[str, Any]] = []
        for seg in segments:
            for word in seg.get("words", []) or []:
                words.append(
                    {
                        "word": str(word.get("word", "")),
                        "start": float(word.get("start", 0.0) or 0.0),
                        "end": float(word.get("end", 0.0) or 0.0),
                        "confidence": float(word.get("probability", 0.0) or 0.0),
                    }
                )
        return words

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
