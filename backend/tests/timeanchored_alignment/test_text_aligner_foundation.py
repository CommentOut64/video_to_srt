from __future__ import annotations

import pytest

from app.services.language_policy.types import (
    WINDOW_KIND_DOMINANT_WITH_ISLANDS,
    WINDOW_KIND_SINGLE_LANGUAGE,
    WINDOW_KIND_TRUE_MIXED,
)
from app.services.timeanchored_alignment.contracts import (
    LanguageRun,
    LanguageRunPackage,
    PhoneUnit,
    PronunciationPackage,
    TextTruthPackage,
    TextTruthQuality,
    TextTruthUnit,
    TimeBasePackage,
    TimeBaseQuality,
    TimeBaseUnit,
    TokenToPhoneSpan,
    TokenUnit,
)
from app.services.timeanchored_alignment.text_aligner import TextAligner


def _build_time_base(tokens: list[str], *, language: str = "zh") -> TimeBasePackage:
    raw_units = tuple(
        TimeBaseUnit(
            text=token,
            start=idx * 0.1,
            end=idx * 0.1 + 0.08,
            confidence=0.92,
            token_type="raw",
        )
        for idx, token in enumerate(tokens)
    )
    word_units = tuple(
        TimeBaseUnit(
            text=token,
            start=idx * 0.1,
            end=idx * 0.1 + 0.08,
            confidence=0.92,
            token_type="word",
        )
        for idx, token in enumerate(tokens)
    )
    return TimeBasePackage(
        raw_units=raw_units,
        word_units=word_units,
        quality=TimeBaseQuality(blank_ratio=0.2, avg_max_prob=0.8, low_prob_ratio=0.1),
        language=language,
    )


def _build_text_truth(tokens: list[str], *, language: str = "zh") -> TextTruthPackage:
    units = tuple(
        TextTruthUnit(
            text=token,
            normalized_text=token.lower(),
            confidence=0.95,
            language=language,
        )
        for token in tokens
    )
    return TextTruthPackage(
        units=units,
        quality=TextTruthQuality(hallucination_risk=0.0, repetition_ratio=0.0, length_ratio=1.0),
        language=language,
        raw_text=" ".join(tokens),
        normalized_text=" ".join(token.lower() for token in tokens),
    )


def _build_language_runs(
    *,
    window_kind: str,
    dominant_language: str,
    source_text: str,
) -> LanguageRunPackage:
    if window_kind == WINDOW_KIND_TRUE_MIXED:
        runs = (
            LanguageRun(run_text="Hello", run_language="en", char_start=0, char_end=5),
            LanguageRun(run_text="你", run_language="zh", char_start=6, char_end=7),
            LanguageRun(run_text="こんにちは", run_language="ja", char_start=8, char_end=13),
        )
        return LanguageRunPackage(
            runs=runs,
            dominant_language="mixed",
            window_kind=WINDOW_KIND_TRUE_MIXED,
            foreign_run_ratio=0.66,
            source_text=source_text,
        )

    runs = (LanguageRun(run_text=source_text, run_language=dominant_language, char_start=0, char_end=len(source_text)),)
    foreign_ratio = 0.2 if window_kind == WINDOW_KIND_DOMINANT_WITH_ISLANDS else 0.0
    return LanguageRunPackage(
        runs=runs,
        dominant_language=dominant_language,
        window_kind=window_kind,
        foreign_run_ratio=foreign_ratio,
        source_text=source_text,
    )


def _build_pronunciation(token_texts: list[str], *, language: str = "zh") -> PronunciationPackage:
    token_units = []
    phone_units = []
    spans = []
    cursor = 0
    for token in token_texts:
        token_units.append(
            TokenUnit(
                token_text=token,
                language=language,
                char_start=cursor,
                char_end=cursor + len(token),
            )
        )
        phone_start = len(phone_units)
        phone_units.append(PhoneUnit(phone_text=f"p-{token}", language=language))
        phone_end = len(phone_units) - 1
        spans.append(TokenToPhoneSpan(token_index=len(token_units) - 1, phone_start=phone_start, phone_end=phone_end))
        cursor += len(token)

    return PronunciationPackage(
        token_units=tuple(token_units),
        phone_units=tuple(phone_units),
        token_to_phone_spans=tuple(spans),
        frontend_source="test",
        dependency_mode={language: "rule_fallback"},
        language=language,
    )


@pytest.mark.parametrize(
    "window_kind",
    [WINDOW_KIND_SINGLE_LANGUAGE, WINDOW_KIND_DOMINANT_WITH_ISLANDS],
)
def test_text_aligner_accepts_single_language_and_island_window(window_kind: str) -> None:
    aligner = TextAligner()
    result = aligner.align_window(
        time_base=_build_time_base(["你", "好", "世", "界"]),
        text_truth=_build_text_truth(["你", "好", "世", "界"]),
        language_runs=_build_language_runs(
            window_kind=window_kind,
            dominant_language="zh",
            source_text="你好世界",
        ),
        pronunciation=_build_pronunciation(["你", "好", "世", "界"]),
    )
    assert result.route == "text"
    assert result.error_code is None
    assert result.metrics.coverage == pytest.approx(1.0, abs=1e-6)


def test_text_aligner_explicitly_rejects_true_mixed_window() -> None:
    aligner = TextAligner()
    result = aligner.align_window(
        time_base=_build_time_base(["你", "好", "世", "界"]),
        text_truth=_build_text_truth(["你", "好", "世", "界"]),
        language_runs=_build_language_runs(
            window_kind=WINDOW_KIND_TRUE_MIXED,
            dominant_language="mixed",
            source_text="Hello 你 こんにちは",
        ),
        pronunciation=_build_pronunciation(["你", "好", "世", "界"]),
    )
    assert result.route == "mixed"
    assert result.error_code == "TRUE_MIXED_WINDOW"
    assert result.items == ()


def test_text_aligner_success_condition_is_threshold_driven() -> None:
    time_base = _build_time_base(["你", "好", "啊"])
    text_truth = _build_text_truth(["你", "好", "哈"])
    language_runs = _build_language_runs(
        window_kind=WINDOW_KIND_SINGLE_LANGUAGE,
        dominant_language="zh",
        source_text="你好哈",
    )
    pronunciation = _build_pronunciation(["你", "好", "哈"])

    permissive = TextAligner(
        threshold_overrides={
            "text_direct_ratio_min": 0.60,
            "text_estimated_ratio_max": 0.50,
        }
    )
    strict = TextAligner(
        threshold_overrides={
            "text_direct_ratio_min": 0.90,
            "text_estimated_ratio_max": 0.50,
        }
    )

    permissive_result = permissive.align_window(
        time_base=time_base,
        text_truth=text_truth,
        language_runs=language_runs,
        pronunciation=pronunciation,
    )
    strict_result = strict.align_window(
        time_base=time_base,
        text_truth=text_truth,
        language_runs=language_runs,
        pronunciation=pronunciation,
    )

    assert permissive_result.route == "text"
    assert strict_result.route == "error"
    assert strict_result.error_code == "TEXT_DIRECT_RATIO_LOW"


def test_text_exact_success_cannot_be_overridden_by_pronunciation_evidence() -> None:
    aligner = TextAligner(
        threshold_overrides={
            "text_direct_ratio_min": 0.9,
            "text_estimated_ratio_max": 0.1,
        }
    )
    result = aligner.align_window(
        time_base=_build_time_base(["OpenAI", "API"], language="en"),
        text_truth=_build_text_truth(["OpenAI", "API"], language="en"),
        language_runs=_build_language_runs(
            window_kind=WINDOW_KIND_SINGLE_LANGUAGE,
            dominant_language="en",
            source_text="OpenAI API",
        ),
        # 人为构造“冲突”读音证据，验证不会反向覆盖文本 exact 成功。
        pronunciation=_build_pronunciation(["zzz", "yyy"], language="en"),
    )
    assert result.route == "text"
    assert [item.status for item in result.items] == ["direct", "direct"]
