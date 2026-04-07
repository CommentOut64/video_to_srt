"""Fast timed facts -> DecisionLayerInput 适配器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from app.services.alignment.types import (
    AlignedFacts,
    AnnotatedWord,
    DecisionLayerInput,
    FusedEvidence,
)
from app.services.language_policy import build_language_policy_snapshot
from app.services.punctuation.base import PuncPosition
from app.services.textflow.alignment_path_adapter import AlignmentPathAdapter
from app.services.textflow.contracts import SegmentationIngressContext


@dataclass(frozen=True)
class FastTimedIngressAdapterResult:
    """Fast timed 输入适配产物。"""

    decision_input: DecisionLayerInput
    stream_id: str
    chunk_index: int
    ingress_context: SegmentationIngressContext
    compat_report: dict[str, Any]


class FastTimedIngressAdapter:
    """把 fast timed 词级事实投影到统一 Decision 入口。"""

    _INGRESS_VERSION = "fast_timed_v1"
    _FALLBACK_PROJECTION_MODE = "fast_word_timestamps"

    def build(
        self,
        *,
        chunk: Any,
        fast_words: Sequence[Any],
        language: str,
        speaker_turns: Sequence[dict[str, Any]] | None,
        punctuation_positions: Sequence[PuncPosition] | None,
        fallback_clean_text: str,
        speaker_id: str | None = None,
        turn_id: str | None = None,
    ) -> FastTimedIngressAdapterResult:
        words = tuple(item for item in (fast_words or ()) if str(getattr(item, "word", "") or "").strip())
        if not words:
            raise ValueError("FastTimedIngressAdapter 需要非空 fast_words")

        chunk_index = int(getattr(chunk, "index", 0) or 0)
        chunk_id = str(getattr(chunk, "chunk_id", "") or f"chunk-{chunk_index}")
        annotated_words, time_mappings = self._build_annotated_words_and_mappings(
            words=words,
            fallback_speaker_id=speaker_id,
            fallback_turn_id=turn_id,
        )
        normalized_turns = self._normalize_speaker_turns(
            speaker_turns=speaker_turns,
            fallback_speaker_id=speaker_id,
            fallback_turn_id=turn_id,
        )
        fallback_positions = list(punctuation_positions or [])
        policy_snapshot = build_language_policy_snapshot(
            language_hint=language,
            feature_scope="timeanchored_alignment",
        )
        ingress_context = SegmentationIngressContext(
            unit_kind="chunk",
            unit_id=chunk_id,
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            source_chunk_ids=(chunk_id,),
            projection_chunk_ids=(),
            metadata={
                "segmentation_input_version": self._INGRESS_VERSION,
                "fallback_projection_mode": self._FALLBACK_PROJECTION_MODE,
                "alignment_route": "fast_timed_final",
                "alignment_failure_semantic": "",
                "time_mapping_count": int(len(time_mappings)),
                "speaker_turn_count": int(len(normalized_turns)),
            },
        )
        aligned_facts = AlignedFacts(
            annotated_words=list(annotated_words),
            alignment_score=self._resolve_alignment_score(words=words),
            gap_ratio=AlignmentPathAdapter._resolve_gap_ratio(aligned_tokens=words),
            gap_positions=[],
            speaker_turns=normalized_turns,
            fast_draft_cuts=[],
            time_axis_version="fast_timed_direct",
            time_mappings=time_mappings,
        )
        fused_evidence = FusedEvidence(
            speaker_changes=AlignmentPathAdapter._build_speaker_change_anchors(
                speaker_turns=normalized_turns,
            ),
            pause_anchors=[],
            semantic_anchors=[],
            punctuation_anchors=[],
            evidence_report={
                "builder": "fast_timed_ingress_adapter",
                "time_mapping_count": int(len(time_mappings)),
                "fallback_punctuation_position_count": int(len(fallback_positions)),
                "speaker_turn_count": int(len(normalized_turns)),
            },
        )
        decision_input = DecisionLayerInput(
            annotated_words=annotated_words,
            aligned_facts=aligned_facts,
            fused_evidence=fused_evidence,
            fallback_clean_text_ref=str(fallback_clean_text or "").strip(),
            fallback_punctuation_positions=fallback_positions,
            canonical_punctuation_facts=(),
            canonical_candidate_boundaries=(),
            policy_snapshot=policy_snapshot,
            allow_fast_draft_fallback=False,
            ingress_context=ingress_context,
        )
        return FastTimedIngressAdapterResult(
            decision_input=decision_input,
            stream_id=f"fast_timed:{chunk_id}",
            chunk_index=chunk_index,
            ingress_context=ingress_context,
            compat_report={
                "segmentation_input_version": self._INGRESS_VERSION,
                "fallback_projection_mode": self._FALLBACK_PROJECTION_MODE,
                "fallback_punctuation_position_count": int(len(fallback_positions)),
                "speaker_turn_count": int(len(normalized_turns)),
                "time_mapping_count": int(len(time_mappings)),
            },
        )

    @staticmethod
    def _build_annotated_words_and_mappings(
        *,
        words: Sequence[Any],
        fallback_speaker_id: str | None,
        fallback_turn_id: str | None,
    ) -> tuple[list[AnnotatedWord], list[dict[str, Any]]]:
        annotated_words: list[AnnotatedWord] = []
        time_mappings: list[dict[str, Any]] = []
        for index, item in enumerate(words):
            word = str(getattr(item, "word", "") or "").strip()
            if not word:
                continue
            start = float(getattr(item, "start", 0.0) or 0.0)
            end = float(getattr(item, "end", start) or start)
            annotated_words.append(
                AnnotatedWord(
                    word=word,
                    start=start,
                    end=end,
                    confidence=getattr(item, "confidence", None),
                    confidence_source=str(
                        getattr(item, "confidence_source", "") or "fast"
                    ),
                    is_pseudo=bool(getattr(item, "is_pseudo", False)),
                    speaker_id=fallback_speaker_id,
                    turn_id=fallback_turn_id,
                    track_id="fast_timed",
                )
            )
            time_mappings.append(
                {
                    "token_index": int(index),
                    "token_id": f"fast:{index}",
                    "word": word,
                    "start": start,
                    "end": end,
                    "mapping_quality": "fast_timed_direct",
                }
            )
        return annotated_words, time_mappings

    @staticmethod
    def _normalize_speaker_turns(
        *,
        speaker_turns: Sequence[dict[str, Any]] | None,
        fallback_speaker_id: str | None,
        fallback_turn_id: str | None,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in list(speaker_turns or ()):
            start = float(item.get("start", 0.0) or 0.0)
            end = float(item.get("end", start) or start)
            if end <= start:
                continue
            normalized.append(
                {
                    "turn_id": str(item.get("turn_id") or fallback_turn_id or ""),
                    "speaker_id": str(item.get("speaker_id") or fallback_speaker_id or "unknown"),
                    "start": start,
                    "end": end,
                    "source": str(item.get("source") or "fast_timed_ingress_adapter"),
                    "boundary_confidence": float(item.get("boundary_confidence", 1.0) or 1.0),
                }
            )
        return normalized

    @staticmethod
    def _resolve_alignment_score(*, words: Sequence[Any]) -> float:
        confidences = [
            float(getattr(item, "confidence", 0.0) or 0.0)
            for item in words
            if getattr(item, "confidence", None) is not None
        ]
        if not confidences:
            return 1.0
        return max(0.0, min(1.0, sum(confidences) / len(confidences)))


__all__ = ["FastTimedIngressAdapter", "FastTimedIngressAdapterResult"]
