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
                    state_key = (token_index, candidate.slice_index)
                    best_score = next_states.get(state_key, (-10.0, tuple()))[0]
                    if float(candidate.score) > best_score:
                        next_states[state_key] = (
                            float(candidate.score),
                            (step,),
                        )
                    continue
                for (_, prev_slice_index), (prev_score, prev_steps) in states.items():
                    if candidate.slice_index < prev_slice_index:
                        continue
                    step_metadata = dict(step.metadata or {})
                    prev_metadata = dict(prev_steps[-1].metadata or {})
                    delta = int(candidate.slice_index - prev_slice_index)
                    transition_penalty = 0.06 * max(delta - 1, 0)
                    jump_penalty = 0.03 * abs(delta)
                    if delta > 2:
                        jump_penalty += 0.12 * float(delta - 2)
                    unsupported_gap_penalty = 0.0
                    current_match_kind = str(step_metadata.get("match_kind", "") or "")
                    previous_end = prev_metadata.get("slice_end")
                    current_start = step_metadata.get("slice_start")
                    boundary_support = max(
                        float(prev_metadata.get("blank_support", 0.0) or 0.0),
                        float(step_metadata.get("blank_support", 0.0) or 0.0),
                    )
                    if current_match_kind == "null":
                        unsupported_gap_penalty += 0.04
                    if previous_end is not None and current_start is not None:
                        gap = float(current_start) - float(previous_end)
                        if gap > 0.25 and boundary_support < 0.55:
                            unsupported_gap_penalty += min(float(gap), 4.0) * 0.25
                    total_score = (
                        float(prev_score)
                        + float(candidate.score)
                        - transition_penalty
                        - jump_penalty
                        - unsupported_gap_penalty
                    )
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
