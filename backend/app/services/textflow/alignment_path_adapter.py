"""AlignmentPath -> DecisionLayerInput 适配器。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

from app.services.alignment.types import (
    AlignedFacts,
    AnnotatedWord,
    DecisionLayerInput,
    FusedEvidence,
)
from app.services.language_policy import build_language_policy_snapshot
from app.services.punctuation.base import PuncPosition
from app.services.textflow.contracts import (
    PunctuationFact as CanonicalPunctuationFact,
    SegmentationIngressContext,
)
from app.services.timeanchored_alignment.contracts import (
    AlignmentPath,
    BoundaryCandidate,
    BoundaryEvidence,
)
from app.services.timeanchored_alignment.decoder.contracts import DecoderShadowResult
from app.services.timeanchored_alignment.preparation.contracts import (
    PreparationBundle,
)


_SENTENCE_END_MARKS = frozenset({"。", "！", "？", ".", "!", "?"})
_WEAK_PUNCT_MARKS = frozenset({"，", "、", ",", ";", "；", ":", "："})


@dataclass(frozen=True)
class AlignmentPathAdapterResult:
    """AlignmentPath 适配产物。"""

    decision_input: DecisionLayerInput
    stream_id: str
    chunk_index: int
    ingress_context: SegmentationIngressContext
    compat_report: dict[str, Any]


class AlignmentPathAdapter:
    """把 Phase 3 AlignmentPath 投影到当前 Decision 入口。"""

    _INGRESS_VERSION = "alignment_path_v1"
    _FALLBACK_PROJECTION_MODE = "layer_internal_compat"

    def build(
        self,
        *,
        preparation: PreparationBundle,
        decoder_result: DecoderShadowResult,
        speaker_id: str | None = None,
        turn_id: str | None = None,
    ) -> AlignmentPathAdapterResult:
        alignment_path = decoder_result.alignment_path
        if alignment_path is None:
            route = str(getattr(decoder_result.alignment_report, "route", "") or "")
            raise ValueError(f"缺少正式 AlignmentPath，当前 route={route or 'unknown'}")

        aligned_tokens = tuple(alignment_path.aligned_tokens)
        annotated_words = self._build_annotated_words(
            aligned_tokens=aligned_tokens,
            fallback_speaker_id=speaker_id,
            fallback_turn_id=turn_id,
        )
        canonical_boundaries = self._build_boundary_evidences(
            aligned_tokens=aligned_tokens,
            boundary_candidates=tuple(decoder_result.boundary_candidates or ()),
        )
        policy_snapshot = build_language_policy_snapshot(
            language_hint=str(preparation.canonical_sequence.language_hint or "auto"),
            feature_scope="timeanchored_alignment",
        )
        canonical_punctuation_facts = self._build_canonical_punctuation_facts(
            preparation=preparation,
        )
        fallback_clean_text, fallback_positions = self._build_fallback_punct_projection(
            preparation=preparation,
        )
        ingress_context = SegmentationIngressContext(
            unit_kind="slow_window",
            unit_id=str(preparation.window_id),
            chunk_id=str(preparation.window_id),
            chunk_index=int(preparation.owner_chunk_index),
            slow_window_id=str(preparation.window_id),
            window_coverage=self._resolve_window_coverage_ratio(preparation.coverage),
            source_chunk_ids=tuple(str(item) for item in preparation.source_chunk_ids),
            projection_chunk_ids=(),
            metadata={
                "segmentation_input_version": self._INGRESS_VERSION,
                "fallback_projection_mode": self._FALLBACK_PROJECTION_MODE,
                "source_chunk_indices": [int(item) for item in preparation.source_chunk_indices],
                "alignment_path_id": str(alignment_path.path_id),
                "alignment_route": str(getattr(decoder_result.alignment_report, "route", "") or ""),
                "alignment_failure_semantic": str(
                    getattr(decoder_result.alignment_report, "failure_semantic", "") or ""
                ),
                "boundary_candidate_count": int(len(decoder_result.boundary_candidates or ())),
                "low_confidence_span_count": int(len(decoder_result.low_confidence_spans or ())),
            },
        )
        aligned_facts = AlignedFacts(
            annotated_words=list(annotated_words),
            alignment_score=float(alignment_path.route_confidence),
            gap_ratio=self._resolve_gap_ratio(aligned_tokens=aligned_tokens),
            gap_positions=[],
            speaker_turns=self._build_speaker_turns(
                preparation=preparation,
                fallback_speaker_id=speaker_id,
                fallback_turn_id=turn_id,
            ),
            fast_draft_cuts=[],
            time_axis_version="alignment_path_window_first",
            time_mappings=[
                {
                    "token_index": int(index),
                    "token_id": str(token.token_id),
                    "start": float(token.start),
                    "end": float(token.end),
                    "mapping_quality": "aligned_path",
                    "source_chunk_ids": [str(item) for item in token.source_chunk_ids],
                }
                for index, token in enumerate(aligned_tokens)
            ],
        )
        fused_evidence = FusedEvidence(
            speaker_changes=self._build_speaker_change_anchors(
                speaker_turns=aligned_facts.speaker_turns,
            ),
            pause_anchors=self._build_pause_anchors(preparation=preparation),
            semantic_anchors=self._build_semantic_anchors(
                boundary_candidates=tuple(decoder_result.boundary_candidates or ()),
            ),
            punctuation_anchors=[
                {
                    "fact_id": item.fact_id,
                    "left_token_index": item.left_token_index,
                    "right_token_index": item.right_token_index,
                    "punct_class": item.punct_class,
                    "normalized_text": item.normalized_text,
                    "boundary_weight": float(item.metadata.get("boundary_weight", 1.0) or 1.0),
                }
                for item in canonical_punctuation_facts
                if bool(item.is_sentence_end)
            ],
            evidence_report={
                "builder": "alignment_path_adapter",
                "boundary_candidate_count": int(len(canonical_boundaries)),
                "punctuation_fact_count": int(len(canonical_punctuation_facts)),
                "pause_fact_count": int(
                    len(getattr(preparation.external_stable_facts, "pause_facts", ()) or ())
                ),
            },
        )
        decision_input = DecisionLayerInput(
            annotated_words=annotated_words,
            aligned_facts=aligned_facts,
            fused_evidence=fused_evidence,
            fallback_clean_text_ref=fallback_clean_text,
            fallback_punctuation_positions=fallback_positions,
            canonical_punctuation_facts=canonical_punctuation_facts,
            canonical_candidate_boundaries=canonical_boundaries,
            policy_snapshot=policy_snapshot,
            allow_fast_draft_fallback=False,
            ingress_context=ingress_context,
        )
        return AlignmentPathAdapterResult(
            decision_input=decision_input,
            stream_id=f"timeanchored:{preparation.window_id}",
            chunk_index=int(preparation.owner_chunk_index),
            ingress_context=ingress_context,
            compat_report={
                "segmentation_input_version": self._INGRESS_VERSION,
                "fallback_projection_mode": self._FALLBACK_PROJECTION_MODE,
                "fallback_punctuation_position_count": int(len(fallback_positions)),
                "canonical_punctuation_fact_count": int(len(canonical_punctuation_facts)),
                "alignment_boundary_count": int(len(canonical_boundaries)),
                "alignment_path_token_count": int(len(aligned_tokens)),
            },
        )

    @classmethod
    def _build_annotated_words(
        cls,
        *,
        aligned_tokens: Sequence[Any],
        fallback_speaker_id: str | None,
        fallback_turn_id: str | None,
    ) -> list[AnnotatedWord]:
        words: list[AnnotatedWord] = []
        for token in aligned_tokens:
            token_text = str(getattr(token, "text", "") or "").strip()
            if not token_text:
                continue
            parts = cls._split_token_text(token_text)
            if not parts:
                continue
            start = float(getattr(token, "start", 0.0) or 0.0)
            end = float(getattr(token, "end", start) or start)
            duration = max(end - start, 0.001)
            total_weight = float(sum(max(len(part), 1) for part in parts))
            trace = dict(getattr(token, "trace", {}) or {})
            cursor = start
            for index, part in enumerate(parts):
                weight = float(max(len(part), 1)) / total_weight
                piece_duration = duration * weight
                piece_start = cursor
                piece_end = (
                    end
                    if index == len(parts) - 1
                    else max(piece_start + 1e-3, cursor + piece_duration)
                )
                cursor = piece_end
                words.append(
                    AnnotatedWord(
                        word=str(part),
                        start=float(piece_start),
                        end=float(piece_end),
                        confidence=float(getattr(token, "confidence", 1.0) or 1.0),
                        confidence_source="aligned",
                        is_pseudo=False,
                        speaker_id=str(trace.get("speaker_id") or fallback_speaker_id or "") or None,
                        turn_id=str(trace.get("turn_id") or fallback_turn_id or "") or None,
                        track_id="alignment_path",
                    )
                )
        return words

    @staticmethod
    def _split_token_text(text: str) -> list[str]:
        normalized = str(text or "").strip()
        if not normalized:
            return []
        parts = [part for part in re.split(r"\s+", normalized) if part]
        return parts if parts else [normalized]

    @staticmethod
    def _build_boundary_evidences(
        *,
        aligned_tokens: Sequence[Any],
        boundary_candidates: Sequence[BoundaryCandidate],
    ) -> tuple[BoundaryEvidence, ...]:
        boundaries: list[BoundaryEvidence] = []
        for candidate in boundary_candidates:
            split_index = int(candidate.split_token_index)
            if split_index < 0 or split_index >= len(aligned_tokens) - 1:
                continue
            left = aligned_tokens[split_index]
            right = aligned_tokens[split_index + 1]
            boundaries.append(
                BoundaryEvidence(
                    split_idx=split_index,
                    event_time=float(candidate.event_time),
                    left_end=float(getattr(left, "end", 0.0) or 0.0),
                    right_start=float(getattr(right, "start", 0.0) or 0.0),
                    reason=str(candidate.reason),
                    score=float(candidate.score),
                    hard_flag=bool(candidate.hard_boundary),
                    metadata={
                        "evidence_source": "alignment_path",
                        "source_chunk_ids": [str(item) for item in candidate.source_chunk_ids],
                        **dict(candidate.metadata or {}),
                    },
                )
            )
        return tuple(boundaries)

    @classmethod
    def _build_canonical_punctuation_facts(
        cls,
        *,
        preparation: PreparationBundle,
    ) -> tuple[CanonicalPunctuationFact, ...]:
        facts: list[CanonicalPunctuationFact] = []
        tokens = tuple(preparation.canonical_sequence.tokens or ())
        for index, item in enumerate(preparation.external_stable_facts.punctuation_facts or ()):
            mark = str(getattr(item, "mark", "") or "")
            if not mark:
                continue
            source_char_index = int(getattr(item, "source_char_index", -1) or -1)
            attach_side = str(getattr(item, "attach_side", "after") or "after")
            left_token_index = cls._resolve_left_token_index(
                tokens=tokens,
                char_index=source_char_index,
            )
            if left_token_index is None:
                continue
            right_token_index = (
                left_token_index + 1 if left_token_index + 1 < len(tokens) else None
            )
            punct_class = cls._resolve_punct_class(mark=mark)
            facts.append(
                CanonicalPunctuationFact(
                    fact_id=f"{preparation.window_id}:punct:{index}",
                    left_token_index=left_token_index,
                    right_token_index=right_token_index,
                    attach_mode="trailing" if attach_side == "after" else "leading",
                    raw_text=mark,
                    normalized_text=mark,
                    punct_class=punct_class,
                    source=cls._resolve_punctuation_source(
                        evidence_source=str(getattr(item, "evidence_source", "") or "")
                    ),
                    priority=100 if punct_class == "sentence_end" else 10,
                    is_sentence_end=punct_class == "sentence_end",
                    metadata={
                        "source_char_index": source_char_index,
                        "boundary_weight": 1.0 if punct_class == "sentence_end" else 0.65,
                    },
                )
            )
        return tuple(facts)

    @classmethod
    def _build_fallback_punct_projection(
        cls,
        *,
        preparation: PreparationBundle,
    ) -> tuple[str, list[PuncPosition]]:
        clean_text = str(preparation.canonical_sequence.normalized_text or "")
        positions = [
            PuncPosition(
                char_index=int(getattr(item, "source_char_index", 0) or 0),
                punctuation=str(getattr(item, "mark", "") or ""),
                confidence=1.0,
            )
            for item in (preparation.external_stable_facts.punctuation_facts or ())
            if str(getattr(item, "mark", "") or "")
        ]
        return clean_text, positions

    @staticmethod
    def _build_speaker_turns(
        *,
        preparation: PreparationBundle,
        fallback_speaker_id: str | None,
        fallback_turn_id: str | None,
    ) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        for item in (preparation.external_stable_facts.speaker_turns or ()):
            turns.append(
                {
                    "turn_id": str(item.get("turn_id") or fallback_turn_id or ""),
                    "speaker_id": str(item.get("speaker_id") or fallback_speaker_id or "unknown"),
                    "start": float(item.get("audio_start", 0.0) or 0.0),
                    "end": float(item.get("audio_end", 0.0) or 0.0),
                    "source": "alignment_path_adapter",
                    "boundary_confidence": 1.0,
                }
            )
        return turns

    @staticmethod
    def _build_speaker_change_anchors(
        *,
        speaker_turns: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        anchors: list[dict[str, Any]] = []
        ordered_turns = sorted(
            list(speaker_turns or ()),
            key=lambda item: (float(item.get("start", 0.0) or 0.0), float(item.get("end", 0.0) or 0.0)),
        )
        for left, right in zip(ordered_turns, ordered_turns[1:]):
            if str(left.get("speaker_id") or "") == str(right.get("speaker_id") or ""):
                continue
            anchors.append(
                {
                    "split_idx": -1,
                    "decision_time": float(right.get("start", 0.0) or 0.0),
                    "score": 1.0,
                    "reason": "speaker_change",
                }
            )
        return anchors

    @staticmethod
    def _build_pause_anchors(
        *,
        preparation: PreparationBundle,
    ) -> list[dict[str, Any]]:
        anchors: list[dict[str, Any]] = []
        for item in (preparation.external_stable_facts.pause_facts or ()):
            anchors.append(
                {
                    "split_idx": -1,
                    "decision_time": float(item.get("pause_start", 0.0) or 0.0),
                    "score": max(0.0, min(1.0, float(item.get("duration", 0.0) or 0.0) / 0.5)),
                    "reason": "pause",
                }
            )
        return anchors

    @staticmethod
    def _build_semantic_anchors(
        *,
        boundary_candidates: Sequence[BoundaryCandidate],
    ) -> list[dict[str, Any]]:
        anchors: list[dict[str, Any]] = []
        for item in boundary_candidates:
            if str(item.reason) not in {"blank_boundary_support", "punctuation_sentence_end"}:
                continue
            anchors.append(
                {
                    "split_idx": int(item.split_token_index),
                    "decision_time": float(item.event_time),
                    "score": float(item.score),
                    "reason": str(item.reason),
                }
            )
        return anchors

    @staticmethod
    def _resolve_gap_ratio(*, aligned_tokens: Sequence[Any]) -> float:
        if len(aligned_tokens) <= 1:
            return 0.0
        max_gap = max(
            (
                max(
                    float(getattr(right, "start", 0.0) or 0.0)
                    - float(getattr(left, "end", 0.0) or 0.0),
                    0.0,
                )
                for left, right in zip(aligned_tokens, aligned_tokens[1:])
            ),
            default=0.0,
        )
        duration = max(
            float(getattr(aligned_tokens[-1], "end", 0.0) or 0.0)
            - float(getattr(aligned_tokens[0], "start", 0.0) or 0.0),
            0.0,
        )
        if duration <= 0.0:
            return 0.0
        return max(0.0, min(1.0, max_gap / duration))

    @staticmethod
    def _resolve_window_coverage_ratio(coverage: Any) -> float | None:
        core_segments = list(getattr(coverage, "core_segments", ()) or ())
        if not core_segments:
            return None
        total = sum(max(float(end) - float(start), 0.0) for start, end in core_segments)
        if total <= 0.0:
            return None
        return 1.0

    @staticmethod
    def _resolve_left_token_index(
        *,
        tokens: Sequence[Any],
        char_index: int,
    ) -> int | None:
        fallback_index: int | None = None
        for index, token in enumerate(tokens):
            token_start = int(getattr(token, "char_start", 0) or 0)
            token_end = int(getattr(token, "char_end", 0) or 0)
            if token_start <= char_index < token_end:
                return int(index)
            if char_index >= token_end - 1:
                fallback_index = int(index)
        return fallback_index

    @staticmethod
    def _resolve_punct_class(*, mark: str) -> str:
        if mark in _SENTENCE_END_MARKS:
            return "sentence_end"
        if mark in _WEAK_PUNCT_MARKS:
            return "weak"
        return "other"

    @staticmethod
    def _resolve_punctuation_source(*, evidence_source: str) -> str:
        normalized = str(evidence_source or "").strip().lower()
        if "punct_track" in normalized:
            return "injected"
        if "slow" in normalized:
            return "slow"
        return "aligned"


__all__ = ["AlignmentPathAdapter", "AlignmentPathAdapterResult"]
