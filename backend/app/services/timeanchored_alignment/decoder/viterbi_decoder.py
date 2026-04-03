"""Phase 3 最小单调 decoder。"""

from __future__ import annotations

from app.services.timeanchored_alignment.decoder.contracts import (
    DecoderPath,
    DecoderStep,
    ObservationLattice,
)


class ViterbiDecoder:
    """在局部候选上求单调最优路径。"""

    def decode(self, *, lattice: ObservationLattice) -> DecoderPath | None:
        if not lattice.candidates_by_token:
            return None

        states: dict[tuple[int, int], tuple[float, tuple[DecoderStep, ...]]] = {}
        for token_index, candidates in enumerate(lattice.candidates_by_token):
            if not candidates:
                return None
            next_states: dict[tuple[int, int], tuple[float, tuple[DecoderStep, ...]]] = {}
            for candidate in candidates:
                if candidate.blocked:
                    continue
                step = DecoderStep(
                    token_index=token_index,
                    slice_index=int(candidate.slice_index),
                    score=float(candidate.score),
                    confidence=max(0.0, min(1.0, float(candidate.score))),
                    lexical_exact=bool(candidate.lexical_exact),
                    synthetic=bool(candidate.synthetic),
                    metadata=dict(candidate.metadata or {}),
                )
                if token_index == 0:
                    next_states[(token_index, candidate.slice_index)] = (
                        float(candidate.score),
                        (step,),
                    )
                    continue
                for (_, prev_slice_index), (prev_score, prev_steps) in states.items():
                    if candidate.slice_index < prev_slice_index:
                        continue
                    transition_penalty = 0.06 * max(candidate.slice_index - prev_slice_index - 1, 0)
                    jump_penalty = 0.03 * abs(candidate.slice_index - prev_slice_index)
                    total_score = float(prev_score) + float(candidate.score) - transition_penalty - jump_penalty
                    state_key = (token_index, candidate.slice_index)
                    best_score = next_states.get(state_key, (-10.0, tuple()))[0]
                    if total_score <= best_score:
                        continue
                    next_states[state_key] = (total_score, prev_steps + (step,))
            if not next_states:
                return None
            states = next_states

        best_score, best_steps = max(states.values(), key=lambda item: item[0])
        total = len(best_steps)
        matched_token_count = sum(
            1
            for step in best_steps
            if self._counts_as_covered(step=step)
        )
        direct_token_count = sum(1 for step in best_steps if step.lexical_exact)
        return DecoderPath(
            steps=best_steps,
            total_score=float(best_score),
            matched_token_count=matched_token_count,
            direct_token_count=direct_token_count,
            coverage_ratio=float(matched_token_count) / float(max(total, 1)),
            direct_ratio=float(direct_token_count) / float(max(total, 1)),
        )

    @staticmethod
    def _counts_as_covered(*, step: DecoderStep) -> bool:
        if (not step.synthetic) and step.confidence >= 0.45:
            return True
        metadata = dict(step.metadata or {})
        return (
            bool(step.lexical_exact)
            and str(metadata.get("synthetic_reason", "") or "") == "canonical_only_token"
        )
