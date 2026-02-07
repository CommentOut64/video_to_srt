"""
Bridge 控制器（完整实现）。
V3.2.0+dev.20260202.01
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional, Tuple, Dict, Any

from app.core.logging import resolve_loguru_logger
from app.models.sensevoice_models import SentenceSegment
from app.services.bridge.batch_builder import BatchBuilder, BridgeBatch
from app.services.bridge.config import BridgeConfig
from app.services.bridge.prompt_builder import PromptBuilder
from app.services.bridge.sentence_queue import SentenceQueue
from app.services.punctuation.semantic_buffer import PunctuationDecision, SemanticChunk


class BridgeController:
    """Bridge 控制器（控制器模式）：统一调度队列、批次与 Prompt 构建，降低耦合。"""

    def __init__(
        self,
        config: Optional[BridgeConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config or BridgeConfig()
        self._logger = resolve_loguru_logger(logger, __name__, layer="L0")
        self._queue = SentenceQueue(
            maxsize=self._config.queue_maxsize,
            backpressure_timeout=self._config.backpressure_timeout,
            logger=self._logger,
        )
        self._batch_builder = BatchBuilder()
        self._prompt_builder = PromptBuilder()
        self._lock = asyncio.Lock()
        self._last_input_time = time.time()
        self._last_flush_time = self._last_input_time
        self._last_audio_end = 0.0
        self._last_language = "auto"
        self._last_speaker_id: Optional[str] = None
        self._last_tail_prompt = ""
        self._arbiter_feedback: Optional[PunctuationDecision] = None
        self._slow_results: Dict[str, Any] = {}
        # V3.2.0+dev.20260202.08: 记录最新仲裁结果（调试/后续策略用）
        self._last_arbitration: Optional[Any] = None

    def is_backpressure_active(self) -> bool:
        return self._queue.is_backpressure_active()

    async def wait_for_capacity(self) -> None:
        await self._queue.wait_for_capacity()

    def get_queue_size(self) -> int:
        return len(self._queue)

    def should_flush_on_idle(self, queue_inter_size: int, now: Optional[float] = None) -> bool:
        """GPU 空闲时的保守调度判断：队列水位+等待时长触发强制 flush。"""
        if not self._config.force_flush_on_gpu_idle:
            return False
        if queue_inter_size > 0:
            return False
        if len(self._queue) == 0:
            return False
        now = now or time.time()
        waited_ms = max(0.0, (now - self._last_input_time) * 1000.0)
        if len(self._queue) >= self._config.queue_high_watermark:
            return True
        if waited_ms >= self._config.dynamic_batch_max_wait_ms:
            return True
        if (
            len(self._queue) >= self._config.queue_low_watermark
            and waited_ms >= self._config.dynamic_batch_max_wait_ms / 2
        ):
            return True
        return False

    def update_arbiter_feedback(self, decision: Optional[PunctuationDecision]) -> None:
        """更新仲裁反馈，用于标点调度融合。"""
        self._arbiter_feedback = decision

    def record_slow_result(self, batch_id: str, result: Any) -> None:
        """记录 SlowWorker 批次结果，供后续仲裁使用。"""
        if not batch_id:
            return
        self._slow_results[batch_id] = result

    def record_arbitration_result(self, result: Any) -> None:
        """记录最新仲裁结果（便于调试与后续策略迭代）。"""
        self._last_arbitration = result

    def get_slow_result(self, batch_id: str) -> Optional[Any]:
        """获取 SlowWorker 批次结果。"""
        if not batch_id:
            return None
        return self._slow_results.get(batch_id)

    async def add_semantic_chunk(self, chunk: SemanticChunk) -> Optional[BridgeBatch]:
        """添加语义 Chunk，满足条件时输出批次。"""
        if chunk is None:
            return None

        async with self._lock:
            now = time.time()
            pre_flush = self._flush_if_starved(now)
            if pre_flush:
                return pre_flush

            pre_flush = self._flush_if_language_changed(chunk)
            if pre_flush:
                await self._push_chunk(chunk)
                self._last_input_time = now
                self._last_audio_end = max(self._last_audio_end, chunk.audio_range[1])
                self._last_language = chunk.language or self._last_language
                if chunk.speaker_id:
                    self._last_speaker_id = chunk.speaker_id
                return pre_flush

            pre_flush = self._flush_if_speaker_changed(chunk)
            if pre_flush:
                await self._push_chunk(chunk)
                # V3.2.0+dev.20260207.03: P0 speaker_change 状态同步补齐
                self._last_input_time = now
                self._last_audio_end = max(self._last_audio_end, chunk.audio_range[1])
                self._last_language = chunk.language or self._last_language
                if chunk.speaker_id:
                    self._last_speaker_id = chunk.speaker_id
                return pre_flush

            pre_flush = self._flush_if_long_pause(chunk)
            if pre_flush:
                await self._push_chunk(chunk)
                return pre_flush

            await self._push_chunk(chunk)
            self._last_input_time = now
            self._last_audio_end = max(self._last_audio_end, chunk.audio_range[1])
            self._last_language = chunk.language or self._last_language
            if chunk.speaker_id:
                self._last_speaker_id = chunk.speaker_id

            if self._should_flush():
                return self._build_batch(flush_reason="threshold_reached")
            return None

    async def flush(self, reason: str = "force") -> Optional[BridgeBatch]:
        """强制刷新缓冲区。"""
        async with self._lock:
            if len(self._queue) == 0:
                return None
            return self._build_batch(flush_reason=reason)

    def build_whisper_prompt(
        self,
        batch: BridgeBatch,
        history_context: Optional[str] = None,
    ) -> str:
        """为批次构建 Whisper Prompt。"""
        return self._prompt_builder.build_prompt(
            batch.sentences,
            history_context=history_context,
            tail_context=self._last_tail_prompt,
        )

    async def _push_chunk(self, chunk: SemanticChunk) -> None:
        is_pushed = await self._queue.push(chunk)
        if is_pushed:
            return

        if self._config.is_enable_frame_drop:
            dropped = self._queue.drop_oldest()
            if dropped:
                self._logger.bind(chunk_id=dropped.chunk_id).warning(
                    "Bridge 触发紧急丢帧：dropped_chunk=%s", dropped.chunk_id
                )

        await self._queue.wait_for_capacity()
        await self._queue.force_push(chunk)

    def _flush_if_starved(self, now: float) -> Optional[BridgeBatch]:
        if len(self._queue) == 0:
            return None
        if (now - self._last_input_time) < self._config.starvation_timeout:
            return None
        return self._build_batch(flush_reason="starvation_timeout")

    def _flush_if_speaker_changed(self, chunk: SemanticChunk) -> Optional[BridgeBatch]:
        if len(self._queue) == 0:
            return None
        if not chunk.speaker_id or not self._last_speaker_id:
            return None
        if chunk.speaker_id == self._last_speaker_id:
            return None
        return self._build_batch(flush_reason="speaker_change")

    def _flush_if_language_changed(self, chunk: SemanticChunk) -> Optional[BridgeBatch]:
        """检测语言切换，必要时刷新当前批次。"""
        if len(self._queue) == 0:
            return None
        current_language = self._normalize_language(chunk.language)
        last_language = self._normalize_language(self._last_language)
        if not current_language or not last_language:
            return None
        if current_language == last_language:
            return None
        return self._build_batch(flush_reason="language_change")

    def _flush_if_long_pause(self, chunk: SemanticChunk) -> Optional[BridgeBatch]:
        if len(self._queue) == 0:
            return None
        gap = chunk.audio_range[0] - self._last_audio_end
        if gap <= self._config.long_pause_threshold:
            return None
        return self._build_batch(flush_reason="long_pause")

    def _should_flush(self) -> bool:
        if len(self._queue) == 0:
            return False
        chunks = self._queue.items()
        total_sentences = sum(len(item.sentences) for item in chunks if item.sentences)
        total_duration, language = self._compute_duration_and_language(chunks)
        min_sentences, max_sentences, min_duration, max_duration = (
            self._compute_dynamic_thresholds(total_duration, total_sentences, language)
        )
        now = time.time()
        min_sentences, max_sentences, min_duration, max_duration, force_flush = (
            self._apply_dynamic_batch_adjustments(
                now,
                len(chunks),
                total_duration,
                total_sentences,
                min_sentences,
                max_sentences,
                min_duration,
                max_duration,
            )
        )

        if total_duration >= max_duration:
            return True
        if total_sentences >= max_sentences:
            return True
        if total_duration >= min_duration and total_sentences >= min_sentences:
            return True
        if force_flush and total_sentences > 0:
            return True
        return False

    def _apply_dynamic_batch_adjustments(
        self,
        now: float,
        queue_len: int,
        total_duration: float,
        total_sentences: int,
        min_sentences: int,
        max_sentences: int,
        min_duration: float,
        max_duration: float,
    ) -> Tuple[int, int, float, float, bool]:
        if not self._config.is_enable_dynamic_batch:
            return min_sentences, max_sentences, min_duration, max_duration, False

        waited_ms = max(0.0, (now - self._last_input_time) * 1000.0)
        force_flush = waited_ms >= self._config.dynamic_batch_max_wait_ms

        # 水位低且等待不足，避免过小批次
        if (
            queue_len <= self._config.queue_low_watermark
            and waited_ms < self._config.dynamic_batch_max_wait_ms
        ):
            return min_sentences, max_sentences, min_duration, max_duration, False

        # 水位高：缩小阈值，加速 flush
        if queue_len >= self._config.queue_high_watermark:
            min_sentences = max(1, min_sentences - 1)
            min_duration = max(0.0, min_duration - self._config.dynamic_duration_step)
            if total_duration <= 0.0 and total_sentences <= 0:
                force_flush = False

        if max_sentences < min_sentences:
            max_sentences = min_sentences
        if max_duration < min_duration:
            max_duration = min_duration
        return min_sentences, max_sentences, min_duration, max_duration, force_flush

    def _compute_duration_and_language(self, chunks: List[SemanticChunk]) -> Tuple[float, str]:
        if not chunks:
            return 0.0, "auto"
        start = min(item.audio_range[0] for item in chunks)
        end = max(item.audio_range[1] for item in chunks)
        duration = max(0.0, end - start)
        language = self._last_language or chunks[-1].language or "auto"
        return duration, language

    def _compute_dynamic_thresholds(
        self,
        total_duration: float,
        total_sentences: int,
        language: str,
    ) -> Tuple[int, int, float, float]:
        min_sentences = self._config.min_batch_sentences
        max_sentences = self._config.max_batch_sentences
        min_duration = self._config.min_batch_duration
        max_duration = self._config.max_batch_duration

        if not self._config.is_enable_dynamic_thresholds:
            return min_sentences, max_sentences, min_duration, max_duration

        speech_rate = self._estimate_speech_rate(language)
        if speech_rate is None or total_duration <= 0:
            return min_sentences, max_sentences, min_duration, max_duration

        is_fast, is_slow = self._classify_speech_rate(language, speech_rate)
        if is_fast:
            max_sentences = max(1, max_sentences - self._config.dynamic_sentence_step)
            max_duration = max(
                min_duration,
                max_duration - self._config.dynamic_duration_step,
            )
        elif is_slow:
            max_sentences = max_sentences + self._config.dynamic_sentence_step
            max_duration = max_duration + self._config.dynamic_duration_step

        if max_sentences < min_sentences:
            max_sentences = min_sentences
        if max_duration < min_duration:
            max_duration = min_duration
        return min_sentences, max_sentences, min_duration, max_duration

    def _estimate_speech_rate(self, language: str) -> Optional[float]:
        chunks = self._queue.items()
        if not chunks:
            return None
        total_duration, _ = self._compute_duration_and_language(chunks)
        if total_duration <= 0:
            return None
        token_count = 0
        for chunk in chunks:
            token_count += self._count_tokens(chunk.sentences, language)
        if token_count <= 0:
            return None
        return token_count / total_duration

    @staticmethod
    def _count_tokens(sentences: List[SentenceSegment], language: str) -> int:
        if not sentences:
            return 0
        lang = (language or "auto").lower()
        if lang.startswith(("zh", "ja")):
            return sum(len(sentence.text) for sentence in sentences if sentence and sentence.text)
        return sum(len(sentence.text.split()) for sentence in sentences if sentence and sentence.text)

    @staticmethod
    def _normalize_language(language: Optional[str]) -> Optional[str]:
        """标准化语言标签，auto/空值视为未知。"""
        if not language:
            return None
        normalized = str(language).strip().lower()
        if not normalized or normalized == "auto":
            return None
        return normalized

    @staticmethod
    def _classify_speech_rate(language: str, rate: float) -> Tuple[bool, bool]:
        lang = (language or "auto").lower()
        if lang.startswith(("zh", "ja")):
            return rate >= 6.0, rate <= 2.5
        return rate >= 3.5, rate <= 1.5

    def _build_batch(self, flush_reason: str) -> BridgeBatch:
        chunks = self._queue.pop_all()
        if not chunks:
            raise RuntimeError("Bridge 批次构建失败：缓冲区为空")

        self._last_flush_time = time.time()
        sentences = [sentence for chunk in chunks for sentence in chunk.sentences]
        previous_tail = self._last_tail_prompt
        self._last_tail_prompt = self._prompt_builder.build_tail(sentences)
        decision = self._merge_decisions(chunks)
        prompt = self._prompt_builder.build_prompt(
            sentences,
            tail_context=previous_tail,
        )

        batch = self._batch_builder.build(
            chunks,
            prompt=prompt,
            punctuation_decision=decision,
            overlap_duration=self._config.overlap_duration,
            flush_reason=flush_reason,
        )

        self._logger.bind(batch_id=batch.batch_id).debug(
            "Bridge 批次构建: batch_id=%s, sentences=%d, duration=%.2fs, reason=%s",
            batch.batch_id,
            len(batch.sentences),
            batch.total_duration,
            flush_reason,
        )
        return batch

    def _merge_decisions(self, chunks: List[SemanticChunk]) -> Optional[PunctuationDecision]:
        decisions = [chunk.punctuation_decision for chunk in chunks if chunk.punctuation_decision]
        if self._arbiter_feedback:
            decisions.append(self._arbiter_feedback)
        if not decisions:
            return None
        slow_decisions = [decision for decision in decisions if decision.is_slow_requested]
        if slow_decisions:
            reason = ";".join({decision.reason for decision in slow_decisions if decision.reason})
            mode = slow_decisions[0].mode
            return PunctuationDecision(is_slow_requested=True, reason=reason, mode=mode)
        return decisions[0]
