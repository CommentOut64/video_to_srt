"""CanonicalTextStream 适配器（Phase 2）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import unicodedata

from app.services.alignment.types import AlignedFacts, TextTrack, TextTrackBundle
from app.services.punctuation.base import PuncPosition
from app.services.textflow.contracts import (
    CanonicalStreamDiagnostics,
    CanonicalTextStream,
    CoreToken,
    PunctuationFact,
    SegmentationIngressContext,
)
from app.services.timeanchored_alignment.contracts import BoundaryEvidence, ProtectedSpan


_INNER_WORD_CONNECTORS = {"'", "’", "-"}
_SENTENCE_END_PUNCT = {"。", "！", "？", ".", "!", "?", "…"}
_WEAK_PUNCT = {"，", ",", "、", "；", ";", "：", ":"}
_QUOTE_PUNCT = {"“", "”", "\"", "'", "‘", "’", "「", "」", "『", "』"}
_BRACKET_PUNCT = {"(", ")", "[", "]", "{", "}", "（", "）", "【", "】", "《", "》"}


@dataclass(frozen=True)
class _TokenSpan:
    index: int
    start: int
    end: int
    text: str


class CanonicalTextStreamAdapter:
    """将三轨文本适配为统一只读输入流。"""

    def build(
        self,
        *,
        stream_id: str,
        chunk_ref: str,
        ingress_context: Optional[SegmentationIngressContext] = None,
        tracks: TextTrackBundle,
        text_source: str,
        language: str = "auto",
        candidate_boundaries: Sequence[BoundaryEvidence] = (),
        protected_spans: Sequence[ProtectedSpan] = (),
        cross_chunk_context: Optional[Dict[str, Any]] = None,
        aligned_facts: Optional[AlignedFacts] = None,
        raw_mount_trace: Optional[Dict[str, Any]] = None,
        token_mapping_trace: Sequence[Dict[str, Any]] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CanonicalTextStream:
        chosen_track = tracks.chosen_track or self._resolve_track_by_text_source(tracks=tracks, text_source=text_source)
        if chosen_track is None:
            raise ValueError("CanonicalTextStreamAdapter 缺少 chosen_track，无法构建主输入流")

        clean_text = str(chosen_track.text_clean or chosen_track.text_itn_raw or chosen_track.raw_text or "")
        token_spans, char_to_token = self._build_token_spans(clean_text)
        tokens = self._build_tokens(
            token_spans=token_spans,
            text_source=text_source,
            clean_text=clean_text,
            chosen_track=chosen_track,
            aligned_facts=aligned_facts,
            stream_id=stream_id,
        )

        source_priority = self._source_priority(text_source)
        raw_fast_facts = self._extract_facts_from_track(
            track=tracks.sv_track,
            canonical_text=clean_text,
            char_to_token=char_to_token,
            source_label="fast",
            stream_id=stream_id,
            source_priority=source_priority,
        )
        raw_slow_facts = self._extract_facts_from_track(
            track=tracks.whisper_track,
            canonical_text=clean_text,
            char_to_token=char_to_token,
            source_label="slow",
            stream_id=stream_id,
            source_priority=source_priority,
        )
        raw_aligned_facts = self._extract_facts_from_track(
            track=tracks.chosen_track,
            canonical_text=clean_text,
            char_to_token=char_to_token,
            source_label="aligned",
            stream_id=stream_id,
            source_priority=source_priority,
        )

        # 仅当三轨都没有显式标点事实时，回退到 chosen 文本字符级提取。
        if not raw_fast_facts and not raw_slow_facts and not raw_aligned_facts:
            raw_aligned_facts = self._extract_facts_from_text(
                text=clean_text,
                char_to_token=char_to_token,
                source_label="aligned",
                stream_id=stream_id,
                source_priority=source_priority,
            )

        active_facts, dedup_log = self._deduplicate_active_facts(
            facts=[*raw_fast_facts, *raw_slow_facts, *raw_aligned_facts],
            source_priority=source_priority,
        )
        ingress_payload = self._serialize_ingress_context(ingress_context)
        diagnostics = CanonicalStreamDiagnostics(
            raw_fast_text=str(tracks.sv_track.text_clean if tracks.sv_track else ""),
            raw_slow_text=str(tracks.whisper_track.text_clean if tracks.whisper_track else ""),
            raw_aligned_text=str(tracks.chosen_track.text_clean if tracks.chosen_track else ""),
            raw_fast_punctuation=tuple(raw_fast_facts),
            raw_slow_punctuation=tuple(raw_slow_facts),
            raw_aligned_punctuation=tuple(raw_aligned_facts),
            dedup_log=tuple(dedup_log),
            raw_mount_trace=dict(raw_mount_trace or {}),
            token_mapping_trace=tuple(dict(item) for item in token_mapping_trace),
            ingress_context=ingress_payload,
        )
        metadata_payload = dict(metadata or {})
        if ingress_payload:
            metadata_payload["ingress_context"] = ingress_payload
        return CanonicalTextStream(
            stream_id=stream_id,
            chunk_ref=chunk_ref,
            language=str(language or "auto"),
            text_source=text_source,
            tokens=tuple(tokens),
            punctuation_facts=tuple(active_facts),
            candidate_boundaries=tuple(candidate_boundaries),
            protected_spans=tuple(protected_spans),
            cross_chunk_context=dict(cross_chunk_context or {}),
            diagnostics=diagnostics,
            metadata=metadata_payload,
        )

    @staticmethod
    def _serialize_ingress_context(
        ingress_context: Optional[SegmentationIngressContext],
    ) -> Dict[str, Any]:
        if ingress_context is None:
            return {}
        return ingress_context.to_dict()

    def _build_tokens(
        self,
        *,
        token_spans: Sequence[_TokenSpan],
        text_source: str,
        clean_text: str,
        chosen_track: TextTrack,
        aligned_facts: Optional[AlignedFacts],
        stream_id: str,
    ) -> List[CoreToken]:
        tokens: List[CoreToken] = []
        clean_to_word = list(getattr(chosen_track, "clean_to_word", []) or [])
        words = list(getattr(aligned_facts, "annotated_words", []) or [])
        for span in token_spans:
            start, end, confidence, speaker_id, turn_id = self._resolve_timing_and_meta(
                span=span,
                clean_text=clean_text,
                clean_to_word=clean_to_word,
                words=words,
            )
            normalized_text = span.text.casefold() if span.text.isascii() else span.text
            tokens.append(
                CoreToken(
                    token_id=f"{stream_id}:tk:{span.index}",
                    index=span.index,
                    text_core=span.text,
                    normalized_text=normalized_text,
                    start=start,
                    end=end,
                    confidence=confidence,
                    source=text_source,
                    speaker_id=speaker_id,
                    turn_id=turn_id,
                )
            )
        return tokens

    @staticmethod
    def _resolve_timing_and_meta(
        *,
        span: _TokenSpan,
        clean_text: str,
        clean_to_word: Sequence[Optional[int]],
        words: Sequence[Any],
    ) -> Tuple[float, float, Optional[float], Optional[str], Optional[str]]:
        if len(clean_to_word) == len(clean_text) and words:
            word_indices: List[int] = []
            for pos in range(span.start, span.end + 1):
                idx = clean_to_word[pos]
                if idx is None:
                    continue
                if int(idx) < 0 or int(idx) >= len(words):
                    continue
                if not word_indices or word_indices[-1] != int(idx):
                    word_indices.append(int(idx))
            if word_indices:
                first = words[word_indices[0]]
                last = words[word_indices[-1]]
                start = float(getattr(first, "start", span.index))
                end = float(getattr(last, "end", start + 0.001))
                if end < start:
                    end = start
                conf_values = [
                    float(getattr(words[word_idx], "confidence"))
                    for word_idx in word_indices
                    if getattr(words[word_idx], "confidence", None) is not None
                ]
                confidence = (sum(conf_values) / len(conf_values)) if conf_values else None
                speaker_id = getattr(first, "speaker_id", None)
                turn_id = getattr(first, "turn_id", None)
                return start, end, confidence, speaker_id, turn_id
        fallback_start = float(span.index)
        return fallback_start, fallback_start + 0.001, None, None, None

    def _extract_facts_from_track(
        self,
        *,
        track: Optional[TextTrack],
        canonical_text: str,
        char_to_token: Sequence[Optional[int]],
        source_label: str,
        stream_id: str,
        source_priority: Dict[str, int],
    ) -> List[PunctuationFact]:
        if track is None:
            return []
        same_text = str(track.text_clean or "") == canonical_text
        positions = list(getattr(track, "punct_positions", []) or [])
        facts: List[PunctuationFact] = []
        for idx, pos in enumerate(positions):
            mapped_char_index = int(pos.char_index) if same_text else None
            facts.append(
                self._build_fact(
                    stream_id=stream_id,
                    source_label=source_label,
                    local_index=idx,
                    punctuation=str(pos.punctuation or ""),
                    char_index=mapped_char_index,
                    char_to_token=char_to_token,
                    source_priority=source_priority,
                    confidence=float(pos.confidence),
                    metadata={
                        "char_index_raw": int(pos.char_index),
                        "text_mapped": same_text,
                    },
                )
            )
        return facts

    def _extract_facts_from_text(
        self,
        *,
        text: str,
        char_to_token: Sequence[Optional[int]],
        source_label: str,
        stream_id: str,
        source_priority: Dict[str, int],
    ) -> List[PunctuationFact]:
        positions: List[PuncPosition] = []
        for idx, char in enumerate(text):
            if self._is_punctuation_char(char) and not self._is_connector_as_core(text=text, idx=idx):
                positions.append(PuncPosition(char_index=idx, punctuation=char, confidence=1.0))
        return [
            self._build_fact(
                stream_id=stream_id,
                source_label=source_label,
                local_index=idx,
                punctuation=str(pos.punctuation or ""),
                char_index=int(pos.char_index),
                char_to_token=char_to_token,
                source_priority=source_priority,
                confidence=float(pos.confidence),
                metadata={"char_index_raw": int(pos.char_index), "text_mapped": True},
            )
            for idx, pos in enumerate(positions)
        ]

    def _build_fact(
        self,
        *,
        stream_id: str,
        source_label: str,
        local_index: int,
        punctuation: str,
        char_index: Optional[int],
        char_to_token: Sequence[Optional[int]],
        source_priority: Dict[str, int],
        confidence: float,
        metadata: Dict[str, Any],
    ) -> PunctuationFact:
        left_idx, right_idx, attach_mode = self._resolve_anchor(
            char_index=char_index,
            char_to_token=char_to_token,
        )
        normalized = punctuation.strip() or punctuation
        punct_class = self._classify_punctuation(normalized)
        return PunctuationFact(
            fact_id=f"{stream_id}:{source_label}:pf:{local_index}",
            left_token_index=left_idx,
            right_token_index=right_idx,
            attach_mode=attach_mode,
            raw_text=punctuation,
            normalized_text=normalized,
            punct_class=punct_class,
            source=source_label,
            priority=int(source_priority.get(source_label, 0)),
            is_sentence_end=punct_class == "sentence_end",
            metadata={"confidence": confidence, **metadata},
        )

    @staticmethod
    def _resolve_anchor(
        *,
        char_index: Optional[int],
        char_to_token: Sequence[Optional[int]],
    ) -> Tuple[Optional[int], Optional[int], str]:
        if char_index is None or char_index < 0 or char_index >= len(char_to_token):
            return None, None, "standalone"

        left_idx: Optional[int] = None
        right_idx: Optional[int] = None
        for pos in range(char_index - 1, -1, -1):
            if char_to_token[pos] is not None:
                left_idx = int(char_to_token[pos])
                break
        for pos in range(char_index + 1, len(char_to_token)):
            if char_to_token[pos] is not None:
                right_idx = int(char_to_token[pos])
                break

        if left_idx is None and right_idx is None:
            return None, None, "standalone"
        if left_idx is None:
            return None, right_idx, "leading"
        if right_idx is None:
            return left_idx, None, "trailing"
        return left_idx, right_idx, "between"

    def _deduplicate_active_facts(
        self,
        *,
        facts: Sequence[PunctuationFact],
        source_priority: Dict[str, int],
    ) -> Tuple[List[PunctuationFact], List[Dict[str, Any]]]:
        # 无锚点事实只归档，不参与同位同类去重。
        unmapped_facts = [
            fact for fact in facts if fact.left_token_index is None and fact.right_token_index is None
        ]
        mapped_facts = [fact for fact in facts if fact not in unmapped_facts]

        grouped: Dict[Tuple[Optional[int], Optional[int], str, str], List[PunctuationFact]] = {}
        for fact in mapped_facts:
            key = (
                fact.left_token_index,
                fact.right_token_index,
                fact.normalized_text,
                fact.punct_class,
            )
            grouped.setdefault(key, []).append(fact)

        active: List[PunctuationFact] = []
        dedup_log: List[Dict[str, Any]] = []
        for key, group in grouped.items():
            if len(group) == 1:
                active.append(group[0])
                continue
            sorted_group = sorted(
                group,
                key=lambda item: (
                    int(source_priority.get(item.source, 0)),
                    float(item.metadata.get("confidence", 0.0) or 0.0),
                ),
                reverse=True,
            )
            kept = sorted_group[0]
            active.append(kept)
            for dropped in sorted_group[1:]:
                dedup_log.append(
                    {
                        "reason": "dedup_lower_priority",
                        "left_token_index": key[0],
                        "right_token_index": key[1],
                        "normalized_text": key[2],
                        "punct_class": key[3],
                        "kept_fact_id": kept.fact_id,
                        "kept_source": kept.source,
                        "dropped_fact_id": dropped.fact_id,
                        "dropped_source": dropped.source,
                    }
                )
        active.extend(unmapped_facts)
        active.sort(key=lambda item: (item.left_token_index is None, item.left_token_index or -1, item.priority), reverse=False)
        return active, dedup_log

    @staticmethod
    def _source_priority(text_source: str) -> Dict[str, int]:
        if text_source == "fast":
            ordered = ("fast", "aligned", "slow", "injected")
        elif text_source == "slow":
            ordered = ("slow", "aligned", "fast", "injected")
        else:
            ordered = ("slow", "aligned", "fast", "injected")
        return {name: len(ordered) - idx for idx, name in enumerate(ordered)}

    @staticmethod
    def _resolve_track_by_text_source(*, tracks: TextTrackBundle, text_source: str) -> Optional[TextTrack]:
        if text_source == "fast":
            return tracks.sv_track
        if text_source == "slow":
            return tracks.whisper_track
        return tracks.chosen_track or tracks.whisper_track or tracks.sv_track

    def _build_token_spans(self, text: str) -> Tuple[List[_TokenSpan], List[Optional[int]]]:
        spans: List[_TokenSpan] = []
        char_to_token: List[Optional[int]] = [None] * len(text)
        cursor = 0
        while cursor < len(text):
            if self._is_core_char(text=text, idx=cursor):
                start = cursor
                cursor += 1
                while cursor < len(text) and self._is_core_char(text=text, idx=cursor):
                    cursor += 1
                end = cursor - 1
                token_text = text[start : end + 1].strip()
                if token_text:
                    token_index = len(spans)
                    spans.append(_TokenSpan(index=token_index, start=start, end=end, text=token_text))
                    for pos in range(start, end + 1):
                        if not text[pos].isspace():
                            char_to_token[pos] = token_index
                continue
            cursor += 1
        return spans, char_to_token

    def _is_core_char(self, *, text: str, idx: int) -> bool:
        char = text[idx]
        if char.isspace():
            return False
        if self._is_connector_as_core(text=text, idx=idx):
            return True
        return not self._is_punctuation_char(char)

    @staticmethod
    def _is_connector_as_core(*, text: str, idx: int) -> bool:
        char = text[idx]
        if char not in _INNER_WORD_CONNECTORS:
            return False
        if idx == 0 or idx == len(text) - 1:
            return False
        left = text[idx - 1]
        right = text[idx + 1]
        return left.isalnum() and right.isalnum()

    @staticmethod
    def _is_punctuation_char(char: str) -> bool:
        return bool(char) and unicodedata.category(char).startswith("P")

    @staticmethod
    def _classify_punctuation(punctuation: str) -> str:
        if punctuation in _SENTENCE_END_PUNCT:
            return "sentence_end"
        if punctuation in _WEAK_PUNCT:
            return "weak"
        if punctuation in _QUOTE_PUNCT:
            return "quote"
        if punctuation in _BRACKET_PUNCT:
            return "bracket"
        return "other"
