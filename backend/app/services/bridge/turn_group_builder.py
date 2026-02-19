"""
TurnGroup 构建器。

设计模式：Builder Pattern。
原因：将 TurnGroup 的聚合与 flush 决策集中封装，保证 SlowWorker
仅消费单 speaker 的稳定输入单元。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional

from app.models.sensevoice_models import SentenceSegment
from app.services.bridge.flush_policy import FlushPolicy, FlushPolicyConfig
from app.services.bridge.turn_group_models import TurnGroup
from app.services.punctuation.semantic_buffer import PunctuationDecision, SemanticChunk


@dataclass
class _PendingState:
    """当前待构建组状态。"""

    speaker_id: str = ""
    language: str = "auto"
    audio_segments: list[tuple[float, float]] = None
    sentences: list[SentenceSegment] = None
    source_chunks: list[str] = None
    target_turn_ids: list[str] = None
    context_turn_ids: list[str] = None
    punctuation_decisions: list[PunctuationDecision] = None
    token_count: int = 0
    first_arrived_at: float = 0.0
    last_arrived_at: float = 0.0
    last_audio_end: float = 0.0

    def __post_init__(self) -> None:
        if self.audio_segments is None:
            self.audio_segments = []
        if self.sentences is None:
            self.sentences = []
        if self.source_chunks is None:
            self.source_chunks = []
        if self.target_turn_ids is None:
            self.target_turn_ids = []
        if self.context_turn_ids is None:
            self.context_turn_ids = []
        if self.punctuation_decisions is None:
            self.punctuation_decisions = []


@dataclass
class TurnGroupEnvelope:
    """TurnGroup + 下游兼容信息封装。"""

    group: TurnGroup
    sentences: list[SentenceSegment]
    punctuation_decision: Optional[PunctuationDecision]


class TurnGroupBuilder:
    """基于 Timeline speaker 信号构建 TurnGroup。"""

    _PROMPT_SEED_MAX_CHARS = 160
    _PROMPT_SEED_MAX_SENTENCES = 3
    _PROMPT_SEED_MAX_KEYWORDS = 20
    _LATIN_WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]{1,31}")
    _CJK_WORD_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,6}")

    def __init__(
        self,
        *,
        flush_policy: Optional[FlushPolicy] = None,
        flush_config: Optional[FlushPolicyConfig] = None,
    ) -> None:
        self.flush_policy = flush_policy or FlushPolicy(flush_config)
        self._pending = _PendingState()
        self._group_counter = 0

    def add_chunk(
        self,
        chunk: SemanticChunk,
        *,
        speaker_id: Optional[str],
        turn_id: Optional[str],
        now: Optional[float] = None,
    ) -> list[TurnGroupEnvelope]:
        """追加语义块，并在命中策略时返回可消费的 TurnGroup。"""
        if chunk is None:
            return []

        now_ts = float(now) if now is not None else time.time()
        current_speaker = str(speaker_id or chunk.speaker_id or "unknown")
        current_language = str(chunk.language or "auto")

        outputs: list[TurnGroupEnvelope] = []
        if self._has_pending():
            is_language_changed = current_language != self._pending.language
            pause_gap = max(0.0, float(chunk.audio_range[0]) - self._pending.last_audio_end)
            is_speaker_changed = current_speaker != self._pending.speaker_id
            if is_speaker_changed:
                outputs.append(self._flush("speaker_change"))
            else:
                decision = self.flush_policy.evaluate_before_append(
                    is_language_changed=is_language_changed,
                    pause_gap_sec=pause_gap,
                )
                if decision.should_flush:
                    outputs.append(self._flush(decision.reason))

        self._append_chunk(
            chunk,
            speaker_id=current_speaker,
            language=current_language,
            turn_id=turn_id,
            now_ts=now_ts,
        )

        audio_sec = self._pending_duration()
        wait_sec = max(0.0, now_ts - self._pending.first_arrived_at)
        decision = self.flush_policy.evaluate_after_append(
            audio_sec=audio_sec,
            token_count=self._pending.token_count,
            wait_sec=wait_sec,
        )
        if decision.should_flush:
            outputs.append(self._flush(decision.reason))
        return outputs

    def flush_idle(self, *, now: Optional[float] = None) -> Optional[TurnGroupEnvelope]:
        """空闲检测触发 tail flush。"""
        if not self._has_pending():
            return None
        now_ts = float(now) if now is not None else time.time()
        idle_sec = max(0.0, now_ts - self._pending.last_arrived_at)
        decision = self.flush_policy.evaluate_idle(idle_sec=idle_sec)
        if not decision.should_flush:
            return None
        return self._flush(decision.reason)

    def flush(self, reason: str = "eof_flush") -> Optional[TurnGroupEnvelope]:
        """强制刷新。"""
        if not self._has_pending():
            return None
        return self._flush(reason)

    def _append_chunk(
        self,
        chunk: SemanticChunk,
        *,
        speaker_id: str,
        language: str,
        turn_id: Optional[str],
        now_ts: float,
    ) -> None:
        if not self._has_pending():
            self._pending.speaker_id = speaker_id
            self._pending.language = language
            self._pending.first_arrived_at = now_ts

        self._pending.last_arrived_at = now_ts
        self._pending.last_audio_end = max(self._pending.last_audio_end, float(chunk.audio_range[1]))
        self._pending.language = language
        self._pending.audio_segments.append((float(chunk.audio_range[0]), float(chunk.audio_range[1])))
        self._pending.source_chunks.extend(list(chunk.source_chunks or [chunk.chunk_id]))
        if chunk.sentences:
            self._pending.sentences.extend(chunk.sentences)
        if chunk.punctuation_decision:
            self._pending.punctuation_decisions.append(chunk.punctuation_decision)
        if turn_id and turn_id not in self._pending.target_turn_ids:
            self._pending.target_turn_ids.append(turn_id)
        if turn_id and turn_id not in self._pending.context_turn_ids:
            self._pending.context_turn_ids.append(turn_id)
        self._pending.token_count += self._count_tokens(chunk.text, language)

    def _flush(self, reason: str) -> TurnGroupEnvelope:
        self._group_counter += 1
        group_id = f"tg-{self._group_counter:06d}"
        prompt_text = self._build_prompt_seed()
        decision = self._merge_decisions(self._pending.punctuation_decisions)
        group = TurnGroup(
            group_id=group_id,
            target_turn_ids=list(self._pending.target_turn_ids),
            context_turn_ids=list(self._pending.context_turn_ids),
            speaker_id=self._pending.speaker_id,
            audio_segments=list(self._pending.audio_segments),
            prompt_text=prompt_text,
            flush_reason=reason,
            language=self._pending.language,
            source_chunks=list(self._pending.source_chunks),
        )
        envelope = TurnGroupEnvelope(
            group=group,
            sentences=list(self._pending.sentences),
            punctuation_decision=decision,
        )
        self._pending = _PendingState()
        return envelope

    def _has_pending(self) -> bool:
        return bool(self._pending.audio_segments)

    def _pending_duration(self) -> float:
        if not self._pending.audio_segments:
            return 0.0
        start = min(segment[0] for segment in self._pending.audio_segments)
        end = max(segment[1] for segment in self._pending.audio_segments)
        return max(0.0, float(end) - float(start))

    def _build_prompt_seed(self) -> str:
        """
        构建 TurnGroup 的提示词种子（关键词化 + 长度预算）。

        Why:
        - 避免把整段历史正文塞入 prompt_text，降低跨批次拼接污染与回显概率。
        - 保留实体词与短上下文线索，满足 Whisper 风格/术语引导。
        """
        recent_sentences = self._pending.sentences[-self._PROMPT_SEED_MAX_SENTENCES :]
        raw_texts = [
            str(sentence.text_clean or sentence.text or "").strip()
            for sentence in recent_sentences
            if sentence
        ]
        merged_text = " ".join(item for item in raw_texts if item).strip()
        if not merged_text:
            return ""

        keywords: list[str] = []
        seen = set()
        for candidate in self._LATIN_WORD_PATTERN.findall(merged_text):
            normalized = candidate.strip()
            lowered = normalized.lower()
            if not normalized or lowered in seen:
                continue
            seen.add(lowered)
            keywords.append(normalized)
            if len(keywords) >= self._PROMPT_SEED_MAX_KEYWORDS:
                break

        if len(keywords) < self._PROMPT_SEED_MAX_KEYWORDS:
            for candidate in self._CJK_WORD_PATTERN.findall(merged_text):
                normalized = candidate.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                keywords.append(normalized)
                if len(keywords) >= self._PROMPT_SEED_MAX_KEYWORDS:
                    break

        seed = " ".join(keywords).strip()
        if not seed:
            seed = merged_text
        if len(seed) > self._PROMPT_SEED_MAX_CHARS:
            seed = seed[-self._PROMPT_SEED_MAX_CHARS :].lstrip()
        return seed

    @staticmethod
    def _count_tokens(text: str, language: str) -> int:
        normalized = str(text or "")
        lang = (language or "auto").lower()
        if lang.startswith(("zh", "ja")):
            return len(normalized)
        return len([item for item in normalized.split() if item])

    @staticmethod
    def _merge_decisions(
        decisions: list[PunctuationDecision],
    ) -> Optional[PunctuationDecision]:
        if not decisions:
            return None
        slow_requested = [item for item in decisions if item.is_slow_requested]
        if slow_requested:
            reason = ";".join({item.reason for item in slow_requested if item.reason})
            return PunctuationDecision(
                is_slow_requested=True,
                reason=reason,
                mode=slow_requested[0].mode,
            )
        return decisions[0]

