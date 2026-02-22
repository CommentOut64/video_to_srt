"""
SlowWorker - 慢流推理 Worker（GPU）

职责：
1. 执行 Whisper 推理
2. 返回推理结果（不做 Prompt 构建/幻觉检测/对齐）
"""
import logging
from typing import Dict, Optional, Any, List, TYPE_CHECKING

from app.core.asr.engine import ASREngine
from app.core.asr.models import ASRResult
from app.core.logging import resolve_loguru_logger
from app.services.bridge.turn_group_models import TurnGroup
from app.services.whisper_buffer_pool import WhisperBufferPool, WhisperBufferConfig
from app.services.whisper.whisper_text_sanitizer import WhisperTextSanitizer

if TYPE_CHECKING:
    from app.services.punctuation.service import PunctuationService

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
        self.logger = resolve_loguru_logger(logger, __name__, layer="L0")
        self._whisper_sanitizer = WhisperTextSanitizer(logger=self.logger)

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
        return self._convert_asr_result(asr_result, prompt=initial_prompt)

    async def process_turn_group(
        self,
        group: TurnGroup,
        *,
        full_audio_array: Any,
        full_audio_sr: int = 16000,
        prompt_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """处理 TurnGroup（Phase 3 W-Epoch）。"""
        if not group:
            raise ValueError("SlowWorker TurnGroup 为空")
        if not group.audio_segments:
            raise ValueError("SlowWorker TurnGroup 缺少音频片段")

        log = self.logger.bind(group_id=group.group_id, speaker_id=group.speaker_id)
        log.debug("SlowWorker TurnGroup 推理开始")
        audio = self._concat_segments_audio(
            segments=group.audio_segments,
            full_audio_array=full_audio_array,
            full_audio_sr=full_audio_sr,
        )
        # Why: 禁止隐式回退到 group.prompt_text，避免关闭注入后仍发生上下文拼接污染。
        prompt = prompt_text if prompt_text else None
        asr_result = await self.patch_engine.transcribe(
            audio,
            language=group.language or self.whisper_language,
            initial_prompt=prompt,
            word_timestamps=True,
            vad_filter=False,
            repetition_penalty=None,
            no_repeat_ngram_size=None,
        )
        whisper_result = self._convert_asr_result(asr_result, prompt=prompt)
        whisper_result["source"] = "slow"
        whisper_result["speaker_id"] = group.speaker_id
        whisper_result["group_id"] = group.group_id
        whisper_result["turn_group_id"] = group.group_id
        whisper_result["target_turn_ids"] = list(group.target_turn_ids)
        whisper_result["context_turn_ids"] = list(group.context_turn_ids)
        whisper_result["flush_reason"] = group.flush_reason
        whisper_result["language"] = group.language or whisper_result.get("language")
        return whisper_result

    def _concat_segments_audio(
        self,
        *,
        segments: List[tuple[float, float]],
        full_audio_array: Any,
        full_audio_sr: int,
    ) -> Any:
        """按给定时间片拼接音频（保留间隔）。"""
        if full_audio_array is None:
            raise ValueError("SlowWorker 音频拼接需要完整音频数组")
        if not segments:
            raise ValueError("SlowWorker 音频拼接缺少时间片")

        sorted_segments = sorted(segments, key=lambda item: item[0])
        config = WhisperBufferConfig()
        pool = WhisperBufferPool(config)
        if getattr(pool, "_sample_rate", None) != full_audio_sr:
            pool._sample_rate = full_audio_sr

        audio_len = len(full_audio_array)
        for idx, (start, end) in enumerate(sorted_segments):
            start_sample = max(0, int(float(start) * full_audio_sr))
            end_sample = min(audio_len, int(float(end) * full_audio_sr))
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
            raise ValueError("SlowWorker 音频拼接失败：有效片段为空")

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

    def _convert_asr_result(
        self,
        asr_result: ASRResult,
        *,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
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

        raw_text = asr_result.text if asr_result.text is not None else None
        # V3.2.0+dev.20260203.10: L0 最小清洗仅产出 min_clean_text，不直接进入下游
        min_clean_text = None
        if raw_text:
            min_clean_text = self._whisper_sanitizer.sanitize_minimal(raw_text, prompt=prompt)
        segments = raw_result.get("segments") if isinstance(raw_result, dict) else None
        return {
            "raw_text": raw_text,
            "min_clean_text": min_clean_text,
            "confidence": float(asr_result.confidence) if asr_result.confidence is not None else None,
            "confidence_source": "slow",
            "language": asr_result.language if asr_result.language is not None else None,
            "raw_result": raw_result,
            "segments": segments,
            "source": "slow",
        }
