from __future__ import annotations

import pytest

from app.services.language_policy.compiler import build_language_policy_snapshot
from app.services.language_policy.resolver import classify_window_language_distribution


REQUIRED_PHASE3_THRESHOLDS = {
    "text_direct_ratio_min",
    "text_estimated_ratio_max",
    "ctc_low_prob_ratio_max",
    "max_continuous_failure_span",
    "dominant_language_min_ratio",
    "foreign_run_max_ratio",
    "foreign_run_char_max",
    "pronunciation_evidence_min_score",
}


@pytest.mark.parametrize("lang", ["zh", "ja", "en"])
def test_phase3_thresholds_exist_and_compilable(lang: str) -> None:
    snapshot = build_language_policy_snapshot(
        language_hint=lang,
        feature_scope="timeanchored_alignment",
    )
    assert snapshot.language_tag == lang
    assert REQUIRED_PHASE3_THRESHOLDS.issubset(snapshot.thresholds.keys())


def test_mixed_is_explicit_label_in_timeanchored_scope() -> None:
    snapshot = build_language_policy_snapshot(
        language_hint="mixed",
        feature_scope="timeanchored_alignment",
    )
    assert snapshot.language_tag == "mixed"
    assert snapshot.metadata.get("window_kind") == "true_mixed"
    assert snapshot.metadata.get("timeanchored_main_chain_eligible") is False


def test_single_primary_language_can_be_dominant_with_islands() -> None:
    result = classify_window_language_distribution(
        language_char_counts={"zh": 18, "en": 4},
        dominant_language_min_ratio=0.70,
        foreign_run_max_ratio=0.30,
        foreign_run_char_max=8,
    )
    assert result.dominant_language == "zh"
    assert result.window_kind == "dominant_with_islands"
    assert result.can_enter_main_chain is True
    assert result.foreign_run_ratio == pytest.approx(4 / 22, abs=1e-6)


def test_true_mixed_window_cannot_enter_main_chain() -> None:
    result = classify_window_language_distribution(
        language_char_counts={"zh": 7, "en": 6, "ja": 5},
        dominant_language_min_ratio=0.70,
        foreign_run_max_ratio=0.30,
        foreign_run_char_max=8,
    )
    assert result.window_kind == "true_mixed"
    assert result.can_enter_main_chain is False


@pytest.mark.parametrize("lang", ["yue", "ko"])
def test_yue_ko_not_pulled_into_primary_timeanchored_paths(lang: str) -> None:
    snapshot = build_language_policy_snapshot(
        language_hint=lang,
        feature_scope="timeanchored_alignment",
    )
    assert snapshot.language_tag == lang
    assert snapshot.metadata.get("timeanchored_main_chain_eligible") is False
