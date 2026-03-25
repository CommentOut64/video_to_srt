"""DecisionIngressPackage -> DecisionLayerInput 适配器。"""

from __future__ import annotations

from dataclasses import dataclass
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
    CrossChunkLock,
    DecisionIngressPackage,
    DecisionToken,
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

    _SOFT_CUT_THRESHOLD = 0.75

    def build(
        self,
        *,
        package: DecisionIngressPackage,
        speaker_id: str | None = None,
        turn_id: str | None = None,
    ) -> DecisionIngressAdapterResult:
        tokens = tuple(package.tokens)
        annotated_words = [
            AnnotatedWord(
                word=str(token.text_core),
                start=float(token.start),
                end=float(token.end),
                confidence=float(token.metadata.get("match_confidence", 1.0) or 1.0),
                confidence_source="aligned",
                is_pseudo=False,
                speaker_id=token.speaker_id or speaker_id,
                turn_id=token.turn_id or turn_id,
                track_id="decision_ingress",
            )
            for token in tokens
        ]
        boundary_evidences = self._build_boundary_evidences(
            boundary_hints=package.boundary_hints,
            tokens=tokens,
        )
        canonical_punctuation_facts = self._build_canonical_punctuation_facts(
            punctuation_facts=package.punctuation_facts,
        )
        fallback_clean_text, fallback_positions = self._build_fallback_punct_projection(
            tokens=tokens,
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
            projection_chunk_ids=(str(package.owner_chunk_id),),
            metadata={
                "source_chunk_indices": [int(item) for item in package.source_chunk_indices],
                "cross_chunk_locks": [
                    self._serialize_cross_chunk_lock(item) for item in package.cross_chunk_locks
                ],
                "punctuation_pair_states": [
                    self._serialize_pair_state(item) for item in package.punctuation_pair_states
                ],
                "quality_metrics": dict(package.quality_metrics or {}),
                "should_fallback": bool(package.should_fallback),
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
            speaker_turns=self._build_speaker_turns(tokens=tokens),
            fast_draft_cuts=[
                float(item.event_time)
                for item in boundary_evidences
                if bool(item.hard_flag) or float(item.score) >= self._SOFT_CUT_THRESHOLD
            ],
            time_axis_version="anchor_mount_window_first",
            time_mappings=[
                {
                    "slot_index": int(token.slot_index),
                    "slot_id": str(token.metadata.get("slot_id", token.token_id) or token.token_id),
                    "start": float(token.start),
                    "end": float(token.end),
                    "mapping_quality": str(
                        token.metadata.get("mount_status", "aligned") or "aligned"
                    ),
                    "source_chunk_ids": list(token.source_chunk_ids),
                    "source_chunk_indices": list(token.source_chunk_indices),
                }
                for token in tokens
            ],
        )
        fused_evidence = FusedEvidence(
            speaker_changes=[
                self._serialize_boundary_anchor(item)
                for item in package.boundary_hints
                if str(item.reason) == "speaker_change" and not bool(item.blocked_by_lock)
            ],
            pause_anchors=[
                self._serialize_boundary_anchor(item)
                for item in package.boundary_hints
                if "pause" in str(item.reason) and not bool(item.blocked_by_lock)
            ],
            semantic_anchors=[
                self._serialize_boundary_anchor(item)
                for item in package.boundary_hints
                if str(item.reason) in {"lexical_boundary", "anchor_block_close"}
                and not bool(item.blocked_by_lock)
            ],
            punctuation_anchors=[
                {
                    "fact_id": item.fact_id,
                    "left_slot_index": item.left_slot_index,
                    "right_slot_index": item.right_slot_index,
                    "punct_class": item.punct_class,
                    "normalized_text": item.normalized_text,
                    "boundary_weight": float(item.boundary_weight),
                }
                for item in package.punctuation_facts
                if str(item.punct_class) == "sentence_end"
            ],
            evidence_report={
                "builder": "decision_ingress_adapter",
                "boundary_hint_count": len(package.boundary_hints),
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
            canonical_candidate_boundaries=tuple(boundary_evidences),
            policy_snapshot=package.policy_snapshot,
            allow_fast_draft_fallback=True,
            ingress_context=ingress_context,
        )
        return DecisionIngressAdapterResult(
            decision_input=decision_input,
            stream_id=f"timeanchored:{package.window_id}",
            chunk_index=int(package.owner_chunk_index),
            ingress_context=ingress_context,
            compat_report={
                "fallback_clean_text_length": len(fallback_clean_text),
                "fallback_punctuation_position_count": len(fallback_positions),
                "canonical_punctuation_fact_count": len(canonical_punctuation_facts),
                "canonical_boundary_count": len(boundary_evidences),
            },
        )

    @classmethod
    def _build_boundary_evidences(
        cls,
        *,
        boundary_hints: Sequence[Any],
        tokens: Sequence[DecisionToken],
    ) -> tuple[BoundaryEvidence, ...]:
        token_by_slot_index = {int(token.slot_index): token for token in tokens}
        slot_index_by_slot_id = {
            str(token.metadata.get("slot_id", token.token_id) or token.token_id): int(token.slot_index)
            for token in tokens
        }
        evidences: list[BoundaryEvidence] = []
        for hint in boundary_hints:
            if bool(hint.blocked_by_lock):
                continue
            slot_index = slot_index_by_slot_id.get(str(hint.split_after_slot_id))
            if slot_index is None:
                continue
            left_token = token_by_slot_index.get(slot_index)
            if left_token is None:
                continue
            right_token = token_by_slot_index.get(slot_index + 1, left_token)
            evidences.append(
                BoundaryEvidence(
                    split_idx=slot_index,
                    event_time=float(hint.decision_time),
                    left_end=float(left_token.end),
                    right_start=float(right_token.start),
                    reason=str(hint.reason),
                    score=float(hint.score),
                    hard_flag=bool(hint.hard_flag),
                    metadata={"blocked_by_lock": bool(hint.blocked_by_lock)},
                )
            )
        return tuple(evidences)

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
                        int(item.left_slot_index) if item.left_slot_index is not None else None
                    ),
                    right_token_index=(
                        int(item.right_slot_index) if item.right_slot_index is not None else None
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
        tokens: Sequence[DecisionToken],
        punctuation_facts: Sequence[PunctuationFact],
    ) -> tuple[str, list[PuncPosition]]:
        clean_text, token_spans = cls._build_synthetic_text(tokens=tokens)
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
        tokens: Sequence[DecisionToken],
    ) -> tuple[str, dict[int, tuple[int, int]]]:
        chars: list[str] = []
        token_spans: dict[int, tuple[int, int]] = {}
        previous_char = ""
        for token in tokens:
            text = str(token.display_text or token.text_core or "").strip()
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
            token_spans[int(token.slot_index)] = (start, end)
            previous_char = text[-1]
        return "".join(chars), token_spans

    @staticmethod
    def _resolve_fallback_char_index(
        *,
        fact: PunctuationFact,
        token_spans: dict[int, tuple[int, int]],
        clean_text: str,
    ) -> int | None:
        if fact.left_slot_index is not None and int(fact.left_slot_index) in token_spans:
            return int(token_spans[int(fact.left_slot_index)][1])
        if fact.right_slot_index is not None and int(fact.right_slot_index) in token_spans:
            start = int(token_spans[int(fact.right_slot_index)][0])
            return max(start - 1, 0) if clean_text else None
        if clean_text:
            return len(clean_text) - 1
        return None

    @staticmethod
    def _build_speaker_turns(
        *,
        tokens: Sequence[DecisionToken],
    ) -> list[dict[str, Any]]:
        turns: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        for token in tokens:
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
            "slot_id": str(item.split_after_slot_id),
            "decision_time": float(item.decision_time),
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
            "slot_ids": [str(value) for value in item.slot_ids],
            "hook_ids": [str(value) for value in item.hook_ids],
            "reason": str(item.reason),
            "source_chunk_ids": [str(value) for value in item.source_chunk_ids],
            "source_chunk_indices": [int(value) for value in item.source_chunk_indices],
        }
