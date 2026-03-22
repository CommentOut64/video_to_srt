from __future__ import annotations

from app.services.homophone.tokenizers import HomophoneTokenizer
from app.services.language_policy.types import WINDOW_KIND_SINGLE_LANGUAGE
from app.services.timeanchored_alignment.contracts import (
    AlignmentItem,
    AlignmentMetrics,
    FinalAlignmentResult,
    LanguageRun,
    LanguageRunPackage,
    PhoneUnit,
    PronunciationPackage,
    ProtectedSpan,
    TextTruthPackage,
    TextTruthQuality,
    TextTruthUnit,
    TimeBasePackage,
    TimeBaseQuality,
    TimeBaseUnit,
    TokenToPhoneSpan,
    TokenUnit,
)
from app.services.timeanchored_alignment.phonetic_aligner import PhoneticAligner


def _time_base(tokens: list[str]) -> TimeBasePackage:
    units = tuple(
        TimeBaseUnit(
            text=token,
            start=idx * 0.1,
            end=idx * 0.1 + 0.08,
            confidence=0.9,
        )
        for idx, token in enumerate(tokens)
    )
    return TimeBasePackage(
        raw_units=units,
        word_units=units,
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.85, low_prob_ratio=0.1),
        language="zh",
    )


def _text_truth(tokens: list[str], *, protected_spans: tuple[ProtectedSpan, ...] = ()) -> TextTruthPackage:
    units = tuple(
        TextTruthUnit(text=token, normalized_text=token, confidence=0.95, language="zh")
        for token in tokens
    )
    text = "".join(tokens)
    return TextTruthPackage(
        units=units,
        quality=TextTruthQuality(hallucination_risk=0.0, repetition_ratio=0.0, length_ratio=1.0),
        language="zh",
        raw_text=text,
        normalized_text=text,
        protected_spans=protected_spans,
    )


def _pron(tokens: list[str]) -> PronunciationPackage:
    token_units = []
    phone_units = []
    spans = []
    cursor = 0
    for token in tokens:
        token_units.append(TokenUnit(token_text=token, language="zh", char_start=cursor, char_end=cursor + len(token)))
        phone_units.append(PhoneUnit(phone_text=f"zh-{token}", language="zh"))
        spans.append(
            TokenToPhoneSpan(
                token_index=len(token_units) - 1,
                phone_start=len(phone_units) - 1,
                phone_end=len(phone_units) - 1,
            )
        )
        cursor += len(token)
    return PronunciationPackage(
        token_units=tuple(token_units),
        phone_units=tuple(phone_units),
        token_to_phone_spans=tuple(spans),
        frontend_source="test",
        dependency_mode={"zh": "pypinyin"},
        language="zh",
    )


def _runs(text: str) -> LanguageRunPackage:
    return LanguageRunPackage(
        runs=(LanguageRun(run_text=text, run_language="zh", char_start=0, char_end=len(text)),),
        dominant_language="zh",
        window_kind=WINDOW_KIND_SINGLE_LANGUAGE,
        foreign_run_ratio=0.0,
        source_text=text,
    )


def _text_result(tokens: list[str], statuses: list[str]) -> FinalAlignmentResult:
    items = []
    for idx, (token, status) in enumerate(zip(tokens, statuses)):
        items.append(
            AlignmentItem(
                text=token,
                start=idx * 0.1,
                end=idx * 0.1 + 0.08,
                status=status,
                source="text_aligner",
                confidence=0.4 if status != "direct" else 0.95,
                reason="seed",
            )
        )
    failed_count = sum(1 for s in statuses if s == "failed")
    return FinalAlignmentResult(
        items=tuple(items),
        route="error" if failed_count > 0 else "text",
        metrics=AlignmentMetrics(coverage=(len(tokens) - failed_count) / len(tokens), failed_count=failed_count),
        error_code="TEXT_ALIGNMENT_FAILED" if failed_count > 0 else None,
    )


def test_phonetic_aligner_zh_homophone_rescue() -> None:
    aligner = PhoneticAligner()
    result = aligner.rescue_window(
        text_alignment=_text_result(["编"], ["failed"]),
        time_base=_time_base(["边"]),
        text_truth=_text_truth(["编"]),
        language_runs=_runs("编"),
        pronunciation=_pron(["编"]),
    )
    assert result.alignment.route == "phonetic"
    assert result.alignment.items[0].status == "phonetic"
    assert result.report["rescued_count"] == 1


def test_phonetic_aligner_zh_different_pronunciation_not_rescued() -> None:
    aligner = PhoneticAligner()
    result = aligner.rescue_window(
        text_alignment=_text_result(["偏"], ["failed"]),
        time_base=_time_base(["边"]),
        text_truth=_text_truth(["偏"]),
        language_runs=_runs("偏"),
        pronunciation=_pron(["偏"]),
    )
    assert result.alignment.items[0].status == "failed"
    assert result.report["rescued_count"] == 0


def test_phonetic_aligner_zh_handles_single_gap_without_breaking_monotonicity() -> None:
    aligner = PhoneticAligner()
    result = aligner.rescue_window(
        text_alignment=_text_result(["编", "好", "吗"], ["failed", "direct", "failed"]),
        time_base=_time_base(["边", "好"]),
        text_truth=_text_truth(["编", "好", "吗"]),
        language_runs=_runs("编好吗"),
        pronunciation=_pron(["编", "好", "吗"]),
    )
    assert result.alignment.items[0].status == "phonetic"
    assert result.alignment.items[1].status == "direct"
    assert result.alignment.items[2].start >= result.alignment.items[1].end


def test_phonetic_aligner_zh_skips_protected_span_first() -> None:
    aligner = PhoneticAligner()
    result = aligner.rescue_window(
        text_alignment=_text_result(["3.14", "编"], ["failed", "failed"]),
        time_base=_time_base(["3.14", "边"]),
        text_truth=_text_truth(
            ["3.14", "编"],
            protected_spans=(ProtectedSpan(start=0, end=4, kind="decimal", text="3.14"),),
        ),
        language_runs=_runs("3.14编"),
        pronunciation=_pron(["3.14", "编"]),
    )
    assert result.alignment.items[0].status == "failed"
    assert result.alignment.items[1].status == "phonetic"
    assert any(trace.reason == "protected_span_skip" for trace in result.traces)


def test_phonetic_aligner_zh_polyphone_override_effective() -> None:
    override_key = HomophoneTokenizer().build_query_key(
        query_text="虫",
        language="zh",
        is_fuzzy=False,
    )
    without_override = PhoneticAligner()
    with_override = PhoneticAligner(zh_reading_overrides={"重": override_key})

    base_input = dict(
        text_alignment=_text_result(["重"], ["failed"]),
        time_base=_time_base(["虫"]),
        text_truth=_text_truth(["重"]),
        language_runs=_runs("重"),
        pronunciation=_pron(["重"]),
    )
    no_result = without_override.rescue_window(**base_input)
    yes_result = with_override.rescue_window(**base_input)

    assert no_result.alignment.items[0].status == "failed"
    assert yes_result.alignment.items[0].status == "phonetic"
