"""读取 decoder 路径并投影为 AlignmentPath 系列对象。"""

from __future__ import annotations

from app.services.timeanchored_alignment.contracts import (
    AlignedToken,
    BoundaryCandidate,
    LowConfidenceSpan,
)
from app.services.timeanchored_alignment.decoder.contracts import DecoderPath
from app.services.timeanchored_alignment.preparation.contracts import PreparationBundle


class AlignmentPathReader:
    """把 decoder path 解释成正式 AlignmentPath 组成件。"""

    _SENTENCE_END_MARKS = {"。", "！", "？", ".", "!", "?"}

    def read(
        self,
        *,
        preparation: PreparationBundle,
        decode_path: DecoderPath,
    ) -> tuple[tuple[AlignedToken, ...], tuple[BoundaryCandidate, ...], tuple[LowConfidenceSpan, ...], dict[str, object]]:
        tokens = tuple(preparation.canonical_sequence.tokens)
        aligned_tokens: list[AlignedToken] = []
        for step in decode_path.steps:
            token = tokens[step.token_index]
            metadata = dict(step.metadata or {})
            start = float(metadata.get("resolved_start", metadata.get("slice_start", 0.0)) or 0.0)
            end = float(metadata.get("resolved_end", metadata.get("slice_end", start)) or start)
            aligned_tokens.append(
                AlignedToken(
                    token_id=str(token.token_id),
                    text=str(token.text),
                    start=start,
                    end=max(end, start),
                    source_chunk_ids=tuple(str(item) for item in token.source_chunk_ids),
                    confidence=max(0.0, min(1.0, float(step.confidence))),
                    trace={
                        "token_index": int(step.token_index),
                        "slice_index": int(step.slice_index),
                        "score": float(step.score),
                        "lexical_exact": bool(step.lexical_exact),
                        "synthetic": bool(step.synthetic),
                        **metadata,
                    },
                )
            )
        boundary_candidates = self._build_boundary_candidates(
            preparation=preparation,
            aligned_tokens=tuple(aligned_tokens),
            decode_path=decode_path,
        )
        low_confidence_spans = self._build_low_confidence_spans(decode_path=decode_path)
        diagnostics = {
            "aligned_token_count": len(aligned_tokens),
            "boundary_candidate_count": len(boundary_candidates),
            "low_confidence_span_count": len(low_confidence_spans),
        }
        return tuple(aligned_tokens), boundary_candidates, low_confidence_spans, diagnostics

    def _build_boundary_candidates(
        self,
        *,
        preparation: PreparationBundle,
        aligned_tokens: tuple[AlignedToken, ...],
        decode_path: DecoderPath,
    ) -> tuple[BoundaryCandidate, ...]:
        candidates: dict[tuple[int, str], BoundaryCandidate] = {}
        for index in range(len(aligned_tokens) - 1):
            left = aligned_tokens[index]
            right = aligned_tokens[index + 1]
            gap = max(float(right.start) - float(left.end), 0.0)
            if gap > 0.08:
                candidates[(index, "gap_pause")] = BoundaryCandidate(
                    split_token_index=index,
                    event_time=float(left.end + gap / 2.0),
                    reason="gap_pause",
                    score=max(0.0, min(1.0, gap / 0.5)),
                    hard_boundary=False,
                    source_chunk_ids=tuple(
                        dict.fromkeys(
                            list(left.source_chunk_ids) + list(right.source_chunk_ids)
                        ).keys()
                    ),
                    metadata={"gap": float(gap)},
                )
            blank_support = max(
                float((decode_path.steps[index].metadata or {}).get("blank_support", 0.0) or 0.0),
                float((decode_path.steps[index + 1].metadata or {}).get("blank_support", 0.0) or 0.0),
            )
            if blank_support >= 0.55:
                candidates[(index, "blank_boundary_support")] = BoundaryCandidate(
                    split_token_index=index,
                    event_time=float(left.end),
                    reason="blank_boundary_support",
                    score=max(0.0, min(1.0, blank_support)),
                    hard_boundary=False,
                    source_chunk_ids=tuple(
                        dict.fromkeys(
                            list(left.source_chunk_ids) + list(right.source_chunk_ids)
                        ).keys()
                    ),
                    metadata={"blank_support": float(blank_support)},
                )

        for evidence in (preparation.external_stable_facts.punctuation_facts or ()):
            mark = str(getattr(evidence, "mark", "") or "")
            if mark not in self._SENTENCE_END_MARKS:
                continue
            char_index = int(getattr(evidence, "source_char_index", -1) or -1)
            split_index = self._resolve_split_index(
                tokens=tuple(preparation.canonical_sequence.tokens),
                char_index=char_index,
            )
            if split_index is None or split_index >= len(aligned_tokens):
                continue
            candidates[(split_index, "punctuation_sentence_end")] = BoundaryCandidate(
                split_token_index=int(split_index),
                event_time=float(aligned_tokens[split_index].end),
                reason="punctuation_sentence_end",
                score=0.92,
                hard_boundary=True,
                source_chunk_ids=tuple(aligned_tokens[split_index].source_chunk_ids),
                metadata={"mark": mark, "char_index": int(char_index)},
            )
        return tuple(candidates[key] for key in sorted(candidates.keys()))

    @staticmethod
    def _resolve_split_index(*, tokens: tuple[object, ...], char_index: int) -> int | None:
        for index, token in enumerate(tokens):
            token_end = int(getattr(token, "char_end", 0) or 0)
            if token_end - 1 <= char_index:
                return index
        return None

    @staticmethod
    def _build_low_confidence_spans(
        *,
        decode_path: DecoderPath,
    ) -> tuple[LowConfidenceSpan, ...]:
        spans: list[LowConfidenceSpan] = []
        start_index: int | None = None
        confidences: list[float] = []
        reasons: set[str] = set()
        for step in decode_path.steps:
            low_conf = AlignmentPathReader._is_low_confidence_step(step=step)
            if low_conf and start_index is None:
                start_index = int(step.token_index)
            if low_conf:
                confidences.append(float(step.confidence))
                reasons.add(AlignmentPathReader._resolve_low_confidence_reason(step=step))
                continue
            if start_index is not None:
                spans.append(
                    LowConfidenceSpan(
                        start_token_index=int(start_index),
                        end_token_index=int(step.token_index - 1),
                        reason="|".join(sorted(reasons)) or "low_confidence",
                        score=max(0.0, min(1.0, 1.0 - (sum(confidences) / max(len(confidences), 1)))),
                    )
                )
                start_index = None
                confidences = []
                reasons = set()
        if start_index is not None:
            spans.append(
                LowConfidenceSpan(
                    start_token_index=int(start_index),
                    end_token_index=int(decode_path.steps[-1].token_index),
                    reason="|".join(sorted(reasons)) or "low_confidence",
                    score=max(0.0, min(1.0, 1.0 - (sum(confidences) / max(len(confidences), 1)))),
                )
            )
        return tuple(spans)

    @staticmethod
    def _is_low_confidence_step(*, step: DecoderStep) -> bool:
        metadata = dict(step.metadata or {})
        if (
            bool(step.synthetic)
            and bool(step.lexical_exact)
            and str(metadata.get("synthetic_reason", "") or "") == "canonical_only_token"
        ):
            return False
        return bool(step.synthetic) or float(step.confidence) < 0.55

    @staticmethod
    def _resolve_low_confidence_reason(*, step: DecoderStep) -> str:
        metadata = dict(step.metadata or {})
        synthetic_reason = str(metadata.get("synthetic_reason", "") or "")
        if step.synthetic:
            return synthetic_reason or "synthetic"
        return "low_confidence"
