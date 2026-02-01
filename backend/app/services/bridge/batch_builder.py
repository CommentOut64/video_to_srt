"""
Bridge 批次构建器。
V3.2.0+dev.20260201.04
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

from app.models.sensevoice_models import SentenceSegment
from app.services.punctuation.semantic_buffer import PunctuationDecision, SemanticChunk


@dataclass
class BridgeBatch:
    """Bridge 输出批次。"""

    batch_id: str
    text: str
    sentences: List[SentenceSegment]
    audio_segments: List[Tuple[float, float]]
    total_duration: float
    language: str
    speaker_id: Optional[str]
    prompt: str
    punctuation_decision: Optional[PunctuationDecision]
    overlap_audio: Optional[Any]
    flush_reason: str
    source_chunks: List[str]
    created_at: float = field(default_factory=time.time)


class BatchBuilder:
    """批次构建器（建造者模式）：集中负责聚合语义块与输出结构化批次。"""

    def build(
        self,
        chunks: List[SemanticChunk],
        *,
        prompt: str,
        punctuation_decision: Optional[PunctuationDecision],
        overlap_duration: float,
        flush_reason: str,
    ) -> BridgeBatch:
        ordered = sorted(chunks, key=lambda item: item.audio_range[0])
        sentences: List[SentenceSegment] = []
        audio_segments: List[Tuple[float, float]] = []
        source_chunks: List[str] = []
        text_parts: List[str] = []

        for item in ordered:
            audio_segments.append(item.audio_range)
            if item.source_chunks:
                source_chunks.extend(item.source_chunks)
            if item.sentences:
                sentences.extend(item.sentences)
            if item.text:
                text_parts.append(item.text)

        total_duration = self._compute_total_duration(audio_segments)
        language = self._select_language(ordered)
        speaker_id = self._select_speaker(ordered)
        text = "".join(text_parts)

        return BridgeBatch(
            batch_id=uuid.uuid4().hex,
            text=text,
            sentences=sentences,
            audio_segments=audio_segments,
            total_duration=total_duration,
            language=language,
            speaker_id=speaker_id,
            prompt=prompt,
            punctuation_decision=punctuation_decision,
            overlap_audio=None if overlap_duration <= 0 else None,
            flush_reason=flush_reason,
            source_chunks=source_chunks,
        )

    @staticmethod
    def _compute_total_duration(segments: List[Tuple[float, float]]) -> float:
        if not segments:
            return 0.0
        start = min(segment[0] for segment in segments)
        end = max(segment[1] for segment in segments)
        return max(0.0, end - start)

    @staticmethod
    def _select_language(chunks: List[SemanticChunk]) -> str:
        languages = [chunk.language for chunk in chunks if chunk.language]
        if not languages:
            return "auto"
        counts: dict[str, int] = {}
        for lang in languages:
            counts[lang] = counts.get(lang, 0) + 1
        return max(counts, key=counts.get)

    @staticmethod
    def _select_speaker(chunks: List[SemanticChunk]) -> Optional[str]:
        speakers = [chunk.speaker_id for chunk in chunks if chunk.speaker_id]
        if not speakers:
            return None
        unique = set(speakers)
        if len(unique) == 1:
            return speakers[0]
        return None
