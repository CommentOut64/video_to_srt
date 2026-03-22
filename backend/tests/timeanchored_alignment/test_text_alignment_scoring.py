from __future__ import annotations

from app.services.timeanchored_alignment.contracts import AcousticCandidate
from app.services.timeanchored_alignment.text_alignment_scoring import (
    TextAlignmentScoreConfig,
    TextAlignmentScorer,
)


def test_text_exact_score_has_higher_priority_than_pronunciation_only() -> None:
    scorer = TextAlignmentScorer()
    exact = scorer.score(time_token="你", text_token="你")
    pronunciation_only = scorer.score(
        time_token="甲",
        text_token="乙",
        pronunciation_match=True,
    )

    assert exact.text_exact > 0.0
    assert exact.total_score > pronunciation_only.total_score
    assert pronunciation_only.text_exact == 0.0
    assert pronunciation_only.text_normalized == 0.0
    assert pronunciation_only.pronunciation_prior == 0.0


def test_protected_span_increases_structural_match_score() -> None:
    scorer = TextAlignmentScorer()
    baseline = scorer.score(
        time_token="state-of-the-art",
        text_token="state-of-the-art",
        is_protected_span=False,
    )
    protected = scorer.score(
        time_token="state-of-the-art",
        text_token="state-of-the-art",
        is_protected_span=True,
    )

    assert protected.protected_bonus > 0.0
    assert protected.total_score > baseline.total_score


def test_top_candidates_bonus_only_applies_to_ambiguous_positions() -> None:
    scorer = TextAlignmentScorer()
    ambiguous = scorer.score(
        time_token="你",
        text_token="好",
        top_candidates=(
            AcousticCandidate(text="你", score=0.52),
            AcousticCandidate(text="好", score=0.48),
        ),
    )
    non_ambiguous = scorer.score(
        time_token="你",
        text_token="好",
        top_candidates=(
            AcousticCandidate(text="你", score=0.90),
            AcousticCandidate(text="好", score=0.10),
        ),
    )

    assert ambiguous.is_ambiguous_position is True
    assert ambiguous.top_candidate_bonus > 0.0
    assert non_ambiguous.is_ambiguous_position is False
    assert non_ambiguous.top_candidate_bonus == 0.0


def test_pronunciation_prior_only_as_tie_break_never_overtakes_text_exact() -> None:
    config = TextAlignmentScoreConfig(
        pronunciation_tie_break_bonus=0.2,
        no_text_match_pronunciation_cap=0.02,
    )
    scorer = TextAlignmentScorer(config=config)

    text_exact = scorer.score(time_token="OpenAI", text_token="OpenAI")
    mismatch_with_pron = scorer.score(
        time_token="OpenAI",
        text_token="API",
        pronunciation_match=True,
        top_candidates=(
            AcousticCandidate(text="API", score=0.51),
            AcousticCandidate(text="OpenAI", score=0.49),
        ),
    )

    assert mismatch_with_pron.pronunciation_prior == 0.02
    assert mismatch_with_pron.total_score < text_exact.total_score
