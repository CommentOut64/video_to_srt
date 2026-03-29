"""DecisionIngressPackage -> DecisionLayerInput 适配器。"""

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
from app.services.punctuation.base import PuncPosition
from app.services.textflow.contracts import (
    PunctuationFact as CanonicalPunctuationFact,
    SegmentationIngressContext,
)
from app.services.timeanchored_alignment.anchor_mount.contracts import (
    AnchoredTokenUnit,
    CrossChunkLock,
    DecisionIngressPackage,
    PunctuationFact,
    PunctuationPairState,
)
from app.services.timeanchored_alignment.contracts import BoundaryEvidence


@dataclass(frozen=True)
class DecisionIngressAdapterResult:
    """Decision ingress 适配产物。"""

    decision_input: DecisionLayerInput
    stream_id: str
    chunk_index: int
    ingress_context: SegmentationIngressContext
    compat_report: dict[str, Any]


class DecisionIngressAdapter:
    """把 AnchorMount 的稳定出口投影到当前 Decision 入口。"""

    _INGRESS_VERSION = "window_first_compat_v1"
    _FALLBACK_PROJECTION_MODE = "layer_internal_compat"

    def build(
        self,
        *,
        package: DecisionIngressPackage,
        speaker_id: str | None = None,
        turn_id: str | None = None,
    ) -> DecisionIngressAdapterResult:
        token_units = tuple(package.anchored_token_units)
        annotated_words = self._build_annotated_words(
            token_units=token_units,
            fallback_speaker_id=speaker_id,
            fallback_turn_id=turn_id,
        )
        boundary_evidences = self._build_boundary_evidences(
            boundary_evidences=package.boundary_evidences,
        )
        canonical_punctuation_facts = self._build_canonical_punctuation_facts(
            punctuation_facts=package.punctuation_facts,
        )
        fallback_clean_text, fallback_positions = self._build_fallback_punct_projection(
            token_units=token_units,
            punctuation_facts=package.punctuation_facts,
        )
        ingress_context = SegmentationIngressContext(
            unit_kind="slow_window",
            unit_id=str(package.window_id),
            chunk_id=str(package.owner_chunk_id),
            chunk_index=int(package.owner_chunk_index),
            slow_window_id=str(package.window_id),
            window_coverage=self._resolve_window_coverage_ratio(package.coverage),
            source_chunk_ids=tuple(str(item) for item in package.source_chunk_ids),
            projection_chunk_ids=(),
            metadata={
                "decision_ingress_version": self._INGRESS_VERSION,
                "fallback_projection_mode": self._FALLBACK_PROJECTION_MODE,
                "source_chunk_indices": [int(item) for item in package.source_chunk_indices],
                "cross_chunk_locks": [
                    self._serialize_cross_chunk_lock(item) for item in package.cross_chunk_locks
                ],
                "punctuation_pair_states": [
                    self._serialize_pair_state(item) for item in package.punctuation_pair_states
                ],
                "quality_metrics": dict(package.quality_metrics or {}),
                "timeline_validity": str(package.timeline_validity),
                "should_fallback": bool(package.should_fallback),
                "candidate_boundary_count": int(len(boundary_evidences)),
            },
        )
        aligned_facts = AlignedFacts(
            annotated_words=list(annotated_words),
            alignment_score=float(
                package.quality_metrics.get("alignment_score")
                or package.quality_metrics.get("soft_anchor_ratio")
                or package.quality_metrics.get("coverage_ratio")
                or 0.0
            ),
            gap_ratio=float(package.quality_metrics.get("largest_unresolved_span", 0.0) or 0.0),
            gap_positions=[],
            speaker_turns=self._build_speaker_turns(token_units=token_units),
            # Why: timeanchored 不再由 ingress 下发可执行切点；切分统一由 Decision 层评分并执行。
            fast_draft_cuts=[],
            time_axis_version="anchor_mount_window_first",
            time_mappings=[
                {
                    "token_index": int(index),
                    "unit_id": str(token.unit_id),
                    "start": float(token.start),
                    "end": float(token.end),
                    "mapping_quality": str(token.mount_status or "aligned"),
                    "source_chunk_ids": list(token.source_chunk_ids),
                    "source_chunk_indices": list(token.source_chunk_indices),
                }
                for index, token in enumerate(token_units)
            ],
        )
        fused_evidence = FusedEvidence(
            speaker_changes=[
                self._serialize_boundary_anchor(item)
                for item in boundary_evidences
                if str(item.reason) == "speaker_change"
                and not bool((item.metadata or {}).get("blocked_by_lock"))
            ],
            pause_anchors=[
                self._serialize_boundary_anchor(item)
                for item in boundary_evidences
                if "pause" in str(item.reason)
                and not bool((item.metadata or {}).get("blocked_by_lock"))
            ],
            semantic_anchors=[
                self._serialize_boundary_anchor(item)
                for item in boundary_evidences
                if str(item.reason) in {"lexical_boundary", "anchor_block_close"}
                and not bool((item.metadata or {}).get("blocked_by_lock"))
            ],
            punctuation_anchors=[
                {
                    "fact_id": item.fact_id,
                    "left_token_index": item.left_token_index,
                    "right_token_index": item.right_token_index,
                    "punct_class": item.punct_class,
                    "normalized_text": item.normalized_text,
                    "boundary_weight": float(item.boundary_weight),
                }
                for item in package.punctuation_facts
                if str(item.punct_class) == "sentence_end"
            ],
            evidence_report={
                "builder": "decision_ingress_adapter",
                "boundary_evidence_count": len(boundary_evidences),
                "punctuation_fact_count": len(package.punctuation_facts),
                "cross_chunk_lock_count": len(package.cross_chunk_locks),
            },
        )
        decision_input = DecisionLayerInput(
            annotated_words=annotated_words,
            aligned_facts=aligned_facts,
            fused_evidence=fused_evidence,
            fallback_clean_text_ref=fallback_clean_text,
            fallback_punctuation_positions=fallback_positions,
            canonical_punctuation_facts=canonical_punctuation_facts,
            # Why: 对齐层专属边界证据必须进入 canonical_candidate_boundaries，
            # 由 Decision 与 punctuation/speaker/gap 信号统一评分。
            canonical_candidate_boundaries=boundary_evidences,
            policy_snapshot=package.policy_snapshot,
            allow_fast_draft_fallback=False,
            ingress_context=ingress_context,
        )
        return DecisionIngressAdapterResult(
            decision_input=decision_input,
            stream_id=f"timeanchored:{package.window_id}",
            chunk_index=int(package.owner_chunk_index),
            ingress_context=ingress_context,
            compat_report={
                "decision_ingress_version": self._INGRESS_VERSION,
                "fallback_projection_mode": self._FALLBACK_PROJECTION_MODE,
                "timeline_validity": str(package.timeline_validity),
                "fallback_clean_text_length": len(fallback_clean_text),
                "fallback_punctuation_position_count": len(fallback_positions),
                "canonical_punctuation_fact_count": len(canonical_punctuation_facts),
                "canonical_boundary_count": len(boundary_evidences),
            },
        )

    @classmethod
    def _build_annotated_words(
        cls,
        *,
        token_units: Sequence[AnchoredTokenUnit],
        fallback_speaker_id: str | None,
        fallback_turn_id: str | None,
    ) -> list[AnnotatedWord]:
        words: list[AnnotatedWord] = []
        for token in token_units:
            token_text = str(token.token_text or "").strip()
            if not token_text:
                continue
            parts = cls._split_token_text(token_text)
            if not parts:
                continue
            start = float(token.start)
            end = float(token.end)
            duration = max(end - start, 0.001)
            total_weight = float(sum(max(len(part), 1) for part in parts))
            cursor = start
            for idx, part in enumerate(parts):
                weight = float(max(len(part), 1)) / total_weight
                piece_duration = duration * weight
                piece_start = cursor
                piece_end = end if idx == len(parts) - 1 else max(piece_start + 1e-3, cursor + piece_duration)
                cursor = piece_end
                annotated = AnnotatedWord(
                    word=str(part),
                    start=float(piece_start),
                    end=float(piece_end),
                    confidence=float(token.match_confidence or 1.0),
                    confidence_source="aligned",
                    is_pseudo=False,
                    speaker_id=token.speaker_id or fallback_speaker_id,
                    turn_id=token.turn_id or fallback_turn_id,
                    track_id="decision_ingress",
                )
                words.append(annotated)
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
        boundary_evidences: Sequence[BoundaryEvidence],
    ) -> tuple[BoundaryEvidence, ...]:
        return tuple(
            item
            for item in boundary_evidences
            if not bool((item.metadata or {}).get("blocked_by_lock"))
        )

    @staticmethod
    def _build_canonical_punctuation_facts(
        *,
        punctuation_facts: Sequence[PunctuationFact],
    ) -> tuple[CanonicalPunctuationFact, ...]:
        facts: list[CanonicalPunctuationFact] = []
        for item in punctuation_facts:
            normalized_text = str(item.normalized_text or "")
            source = str(item.source or "aligned").strip().lower()
            if source not in {"fast", "slow", "aligned", "injected"}:
                source = "aligned"
            facts.append(
                CanonicalPunctuationFact(
                    fact_id=str(item.fact_id),
                    left_token_index=(
                        int(item.left_token_index) if item.left_token_index is not None else None
                    ),
                    right_token_index=(
                        int(item.right_token_index) if item.right_token_index is not None else None
                    ),
                    attach_mode=str(item.attach_mode),
                    raw_text=normalized_text,
                    normalized_text=normalized_text,
                    punct_class=str(item.punct_class),
                    source=source,
                    priority=100 if str(item.punct_class) == "sentence_end" else 10,
                    is_sentence_end=str(item.punct_class) == "sentence_end",
                    metadata={
                        "group_id": item.group_id,
                        "confidence": float(item.confidence),
                        "boundary_weight": float(item.boundary_weight),
                        "render_default": bool(item.render_default),
                        **dict(item.metadata or {}),
                    },
                )
            )
        return tuple(facts)

    @classmethod
    def _build_fallback_punct_projection(
        cls,
        *,
        token_units: Sequence[AnchoredTokenUnit],
        punctuation_facts: Sequence[PunctuationFact],
    ) -> tuple[str, list[PuncPosition]]:
        clean_text, token_spans = cls._build_synthetic_text(token_units=token_units)
        positions: list[PuncPosition] = []
        for item in punctuation_facts:
            char_index = cls._resolve_fallback_char_index(
                fact=item,
                token_spans=token_spans,
                clean_text=clean_text,
            )
            if char_index is None:
                continue
            positions.append(
                PuncPosition(
                    char_index=int(char_index),
                    punctuation=str(item.normalized_text),
                    confidence=float(item.confidence),
                )
            )
        return clean_text, positions

    @staticmethod
    def _build_synthetic_text(
        *,
        token_units: Sequence[AnchoredTokenUnit],
    ) -> tuple[str, dict[int, tuple[int, int]]]:
        chars: list[str] = []
        token_spans: dict[int, tuple[int, int]] = {}
        previous_char = ""
        for index, token in enumerate(token_units):
            text = str(token.token_text or "").strip()
            if not text:
                continue
            if chars:
                first_char = text[0]
                if (
                    previous_char
                    and previous_char.isascii()
                    and first_char.isascii()
                    and previous_char.isalnum()
                    and first_char.isalnum()
                ):
                    chars.append(" ")
            start = len(chars)
            chars.extend(list(text))
            end = len(chars) - 1
            token_spans[int(index)] = (start, end)
            previous_char = text[-1]
        return "".join(chars), token_spans

    @staticmethod
    def _resolve_fallback_char_index(
        *,
        fact: PunctuationFact,
        token_spans: dict[int, tuple[int, int]],
        clean_text: str,
    ) -> int | None:
        if fact.left_token_index is not None and int(fact.left_token_index) in token_spans:
            return int(token_spans[int(fact.left_token_index)][1])
        if fact.right_token_index is not None and int(fact.right_token_index) in token_spans:
            start = int(token_spans[int(fact.right_token_index)][0])
            return max(start - 1, 0) if clean_text else None
        if clean_text:
            return len(clean_text) - 1
        return None

    @staticmethod
    def _build_speaker_turns(
        *,
        token_units: Sequence[AnchoredTokenUnit],
    ) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for token in token_units:
            speaker = token.speaker_id
            turn = token.turn_id
            if current and current["speaker_id"] == speaker and current["turn_id"] == turn:
                current["end"] = float(token.end)
                continue
            current = {
                "turn_id": turn or "",
                "speaker_id": speaker or "unknown",
                "start": float(token.start),
                "end": float(token.end),
                "source": "decision_ingress_adapter",
                "boundary_confidence": 1.0,
            }
            turns.append(current)
        return turns

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
    def _serialize_boundary_anchor(item: Any) -> dict[str, Any]:
        return {
            "split_idx": int(item.split_idx),
            "decision_time": float(item.event_time),
            "score": float(item.score),
            "reason": str(item.reason),
        }

    @staticmethod
    def _serialize_pair_state(item: PunctuationPairState) -> dict[str, Any]:
        return {
            "group_id": str(item.group_id),
            "pair_kind": str(item.pair_kind),
            "open_fact_id": str(item.open_fact_id),
            "close_fact_id": item.close_fact_id,
            "state": str(item.state),
            "metadata": dict(item.metadata or {}),
        }

    @staticmethod
    def _serialize_cross_chunk_lock(item: CrossChunkLock) -> dict[str, Any]:
        return {
            "lock_id": str(item.lock_id),
            "unit_ids": [str(value) for value in item.unit_ids],
            "hook_ids": [str(value) for value in item.hook_ids],
            "reason": str(item.reason),
            "source_chunk_ids": [str(value) for value in item.source_chunk_ids],
            "source_chunk_indices": [int(value) for value in item.source_chunk_indices],
        }
