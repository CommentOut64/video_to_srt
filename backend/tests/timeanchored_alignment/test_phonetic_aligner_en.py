from __future__ import annotations

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
            start=idx * 0.2,
            end=idx * 0.2 + 0.16,
            confidence=0.9,
            token_type="word",
        )
        for idx, token in enumerate(tokens)
    )
    return TimeBasePackage(
        raw_units=units,
        word_units=units,
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.86, low_prob_ratio=0.1),
        language="en",
    )


def _text_truth(tokens: list[str], *, protected_spans: tuple[ProtectedSpan, ...] = ()) -> TextTruthPackage:
    units = tuple(
        TextTruthUnit(text=token, normalized_text=token.lower(), confidence=0.95, language="en")
        for token in tokens
    )
    text = " ".join(tokens)
    return TextTruthPackage(
        units=units,
        quality=TextTruthQuality(hallucination_risk=0.0, repetition_ratio=0.0, length_ratio=1.0),
        language="en",
        raw_text=text,
        normalized_text=text.lower(),
        protected_spans=protected_spans,
    )


def _pron(tokens: list[str]) -> PronunciationPackage:
    token_units = []
    phone_units = []
    spans = []
    cursor = 0
    for token in tokens:
        token_units.append(TokenUnit(token_text=token, language="en", char_start=cursor, char_end=cursor + len(token)))
        phone_units.append(PhoneUnit(phone_text=f"en-{token.lower()}", language="en"))
        spans.append(
            TokenToPhoneSpan(
                token_index=len(token_units) - 1,
                phone_start=len(phone_units) - 1,
                phone_end=len(phone_units) - 1,
            )
        )
        cursor += len(token) + 1
    return PronunciationPackage(
        token_units=tuple(token_units),
        phone_units=tuple(phone_units),
        token_to_phone_spans=tuple(spans),
        frontend_source="test",
        dependency_mode={"en": "cmudict"},
        language="en",
    )


def _runs(text: str) -> LanguageRunPackage:
    return LanguageRunPackage(
        runs=(LanguageRun(run_text=text, run_language="en", char_start=0, char_end=len(text)),),
        dominant_language="en",
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
                start=idx * 0.2,
                end=idx * 0.2 + 0.16,
                status=status,
                source="text_aligner",
                confidence=0.4 if status != "direct" else 0.96,
                reason="seed",
            )
        )
    failed_count = sum(1 for item in statuses if item == "failed")
    return FinalAlignmentResult(
        items=tuple(items),
        route="error" if failed_count > 0 else "text",
        metrics=AlignmentMetrics(coverage=(len(tokens) - failed_count) / len(tokens), failed_count=failed_count),
        error_code="TEXT_ALIGNMENT_FAILED" if failed_count > 0 else None,
    )


def test_phonetic_aligner_en_homophone_rescue() -> None:
    aligner = PhoneticAligner()
    result = aligner.rescue_window(
        text_alignment=_text_result(["write"], ["failed"]),
        time_base=_time_base(["right"]),
        text_truth=_text_truth(["write"]),
        language_runs=_runs("write"),
        pronunciation=_pron(["write"]),
    )
    assert result.alignment.items[0].status == "phonetic"
    assert result.alignment.route == "phonetic"


def test_phonetic_aligner_en_contraction_rescue() -> None:
    aligner = PhoneticAligner()
    result = aligner.rescue_window(
        text_alignment=_text_result(["dont"], ["failed"]),
        time_base=_time_base(["don't"]),
        text_truth=_text_truth(["dont"]),
        language_runs=_runs("dont"),
        pronunciation=_pron(["dont"]),
    )
    assert result.alignment.items[0].status == "phonetic"


def test_phonetic_aligner_en_protected_structure_skip() -> None:
    aligner = PhoneticAligner()
    result = aligner.rescue_window(
        text_alignment=_text_result(["U.S.", "OpenAI's"], ["failed", "failed"]),
        time_base=_time_base(["US", "OpenAI"]),
        text_truth=_text_truth(
            ["U.S.", "OpenAI's"],
            protected_spans=(
                ProtectedSpan(start=0, end=4, kind="acronym", text="U.S."),
                ProtectedSpan(start=5, end=13, kind="possessive", text="OpenAI's"),
            ),
        ),
        language_runs=_runs("U.S. OpenAI's"),
        pronunciation=_pron(["U.S.", "OpenAI's"]),
    )
    assert result.alignment.items[0].status == "failed"
    assert result.alignment.items[1].status == "failed"
    assert sum(1 for trace in result.traces if trace.reason == "protected_span_skip") == 2


def test_phonetic_aligner_en_rejects_subword_input() -> None:
    aligner = PhoneticAligner()
    result = aligner.rescue_window(
        text_alignment=_text_result(["hello"], ["failed"]),
        time_base=_time_base(["▁hel", "lo"]),
        text_truth=_text_truth(["hello"]),
        language_runs=_runs("hello"),
        pronunciation=_pron(["hello"]),
    )
    assert result.alignment.items[0].status == "failed"
    assert result.report["fallback_reason"] == "en_subword_input_forbidden"


def test_phonetic_aligner_en_does_not_take_over_text_exact_success() -> None:
    aligner = PhoneticAligner()
    result = aligner.rescue_window(
        text_alignment=_text_result(["OpenAI"], ["direct"]),
        time_base=_time_base(["OpenAI"]),
        text_truth=_text_truth(["OpenAI"]),
        language_runs=_runs("OpenAI"),
        pronunciation=_pron(["OpenAI"]),
    )
    assert result.alignment.items[0].status == "direct"
    assert result.alignment.route == "text"

