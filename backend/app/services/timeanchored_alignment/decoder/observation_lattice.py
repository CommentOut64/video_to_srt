"""从 PreparationBundle 构建 observation lattice。"""

from __future__ import annotations

import unicodedata

from app.services.timeanchored_alignment.decoder.contracts import (
    LatticeCandidate,
    ObservationLattice,
)
from app.services.timeanchored_alignment.preparation.contracts import PreparationBundle


def _normalize_token(value: str) -> str:
    return "".join(
        char
        for char in str(value or "").strip().lower()
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


class ObservationLatticeBuilder:
    """把 canonical token、pronunciation graph、observation slices 组织成轻量 lattice。"""

    _NULL_ALIGN_SCORE = 0.16

    def build(self, *, preparation: PreparationBundle) -> ObservationLattice:
        canonical_tokens = tuple(preparation.canonical_sequence.tokens)
        observation_slices = tuple(preparation.acoustic_observation_pack.slices)
        token_nodes = tuple(preparation.pronunciation_graph.token_nodes)
        default_language = str(preparation.canonical_sequence.language_hint or "auto")
        blank_track = tuple(preparation.acoustic_observation_pack.blank_track or ())

        rows: list[tuple[LatticeCandidate, ...]] = []
        for token_index, token in enumerate(canonical_tokens):
            token_node = token_nodes[token_index] if token_index < len(token_nodes) else None
            token_key = _normalize_token(token.normalized_text or token.text)
            pronunciation_keys = {
                _normalize_token(getattr(variant, "reading_key", ""))
                for variant in (getattr(token_node, "variants", ()) or ())
                if _normalize_token(getattr(variant, "reading_key", ""))
            }
            candidate_indices = range(
                max(0, token_index - 1),
                min(len(observation_slices), token_index + 2),
            )
            candidates: list[LatticeCandidate] = []
            for slice_index in candidate_indices:
                observation = observation_slices[slice_index]
                primary_key = _normalize_token(observation.primary_token)
                top_candidate_keys = {
                    _normalize_token(getattr(item, "token", ""))
                    for item in (observation.top_candidates or ())
                    if _normalize_token(getattr(item, "token", ""))
                }
                lexical_exact = bool(token_key) and token_key == primary_key
                pronunciation_match = bool(token_key) and (
                    primary_key in pronunciation_keys or bool(pronunciation_keys & top_candidate_keys)
                )
                top_candidate_match = bool(token_key) and token_key in top_candidate_keys
                language_consistent = True
                blank_support = self._resolve_blank_support(
                    blank_track=blank_track,
                    slice_index=slice_index,
                    fallback=float(observation.blank_score or 0.0),
                )
                synthetic = bool((observation.metadata or {}).get("synthetic"))
                match_kind = self._resolve_match_kind(
                    lexical_exact=lexical_exact,
                    pronunciation_match=pronunciation_match,
                    top_candidate_match=top_candidate_match,
                )
                blocked = False
                score = 0.0
                if lexical_exact:
                    score += 0.58
                if pronunciation_match:
                    score += 0.18
                if top_candidate_match:
                    score += 0.10
                if language_consistent:
                    score += 0.08
                score += min(max(blank_support, 0.0), 1.0) * 0.06
                if synthetic:
                    score -= 0.32
                if match_kind == "weak":
                    score -= 0.06
                if slice_index != token_index:
                    score -= 0.08 * abs(slice_index - token_index)
                confidence = max(min(score, 1.0), 0.0)
                candidates.append(
                    LatticeCandidate(
                        token_index=token_index,
                        slice_index=slice_index,
                        score=confidence,
                        lexical_exact=lexical_exact,
                        pronunciation_match=pronunciation_match,
                        top_candidate_match=top_candidate_match,
                        language_consistent=language_consistent,
                        blank_support=blank_support,
                        synthetic=synthetic,
                        blocked=blocked,
                        metadata={
                            "token_text": str(token.text),
                            "slice_primary_token": str(observation.primary_token),
                            "slice_start": float(observation.start),
                            "slice_end": float(observation.end),
                            "blank_support": float(blank_support),
                            "source_chunk_ids": list(token.source_chunk_ids),
                            "source_chunk_indices": list(token.source_chunk_indices),
                            "observation_unit_index": (observation.metadata or {}).get("observation_unit_index"),
                            "observation_unit_text": str((observation.metadata or {}).get("observation_unit_text", "") or ""),
                            "observation_part_index": (observation.metadata or {}).get("observation_part_index"),
                            "observation_part_count": (observation.metadata or {}).get("observation_part_count"),
                            "synthetic": bool(synthetic),
                            "synthetic_reason": str((observation.metadata or {}).get("reason", "") or ""),
                            "match_kind": match_kind,
                        },
                    )
                )
            null_slice_index = min(max(token_index, 0), max(len(observation_slices) - 1, 0))
            candidates.append(
                LatticeCandidate(
                    token_index=token_index,
                    slice_index=int(null_slice_index),
                    score=self._NULL_ALIGN_SCORE,
                    lexical_exact=False,
                    pronunciation_match=False,
                    top_candidate_match=False,
                    language_consistent=True,
                    blank_support=0.0,
                    synthetic=True,
                    blocked=False,
                    metadata={
                        "token_text": str(token.text),
                        "slice_primary_token": "",
                        "slice_start": None,
                        "slice_end": None,
                        "blank_support": 0.0,
                        "source_chunk_ids": list(token.source_chunk_ids),
                        "source_chunk_indices": list(token.source_chunk_indices),
                        "observation_unit_index": None,
                        "observation_unit_text": "",
                        "observation_part_index": None,
                        "observation_part_count": None,
                        "synthetic": True,
                        "synthetic_reason": "null_align",
                        "match_kind": "null",
                    },
                )
            )
            candidates.sort(key=lambda item: item.score, reverse=True)
            rows.append(tuple(candidates))
        return ObservationLattice(
            token_ids=tuple(str(token.token_id) for token in canonical_tokens),
            slice_ids=tuple(str(item.slice_id) for item in observation_slices),
            candidates_by_token=tuple(rows),
            metadata={
                "token_count": len(canonical_tokens),
                "slice_count": len(observation_slices),
                "capability_level": str(preparation.acoustic_observation_pack.capability_level),
            },
        )

    @staticmethod
    def _resolve_blank_support(
        *,
        blank_track: tuple[float, ...],
        slice_index: int,
        fallback: float,
    ) -> float:
        if blank_track:
            left = max(0, slice_index - 1)
            right = min(len(blank_track), slice_index + 2)
            window = [float(value) for value in blank_track[left:right]]
            if window:
                return max(0.0, min(1.0, max(window)))
        return max(0.0, min(1.0, float(fallback or 0.0)))

    @staticmethod
    def _resolve_match_kind(
        *,
        lexical_exact: bool,
        pronunciation_match: bool,
        top_candidate_match: bool,
    ) -> str:
        if lexical_exact:
            return "direct"
        if top_candidate_match:
            return "top_candidate"
        if pronunciation_match:
            return "pronunciation"
        return "weak"
