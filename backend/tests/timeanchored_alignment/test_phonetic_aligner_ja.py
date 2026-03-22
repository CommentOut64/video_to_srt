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
            start=idx * 0.15,
            end=idx * 0.15 + 0.12,
            confidence=0.9,
        )
        for idx, token in enumerate(tokens)
    )
    return TimeBasePackage(
        raw_units=units,
        word_units=units,
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.85, low_prob_ratio=0.1),
        language="ja",
    )


def _text_truth(tokens: list[str]) -> TextTruthPackage:
    units = tuple(
        TextTruthUnit(text=token, normalized_text=token, confidence=0.95, language="ja")
        for token in tokens
    )
    text = "".join(tokens)
    return TextTruthPackage(
        units=units,
        quality=TextTruthQuality(hallucination_risk=0.0, repetition_ratio=0.0, length_ratio=1.0),
        language="ja",
        raw_text=text,
        normalized_text=text,
    )


def _pron(tokens: list[str], *, mode: str) -> PronunciationPackage:
    token_units = []
    phone_units = []
    spans = []
    cursor = 0
    for token in tokens:
        token_units.append(TokenUnit(token_text=token, language="ja", char_start=cursor, char_end=cursor + len(token)))
        phone_units.append(PhoneUnit(phone_text=f"ja-{token}", language="ja"))
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
        dependency_mode={"ja": mode},
        language="ja",
    )


def _runs(text: str) -> LanguageRunPackage:
    return LanguageRunPackage(
        runs=(LanguageRun(run_text=text, run_language="ja", char_start=0, char_end=len(text)),),
        dominant_language="ja",
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
                start=idx * 0.15,
                end=idx * 0.15 + 0.12,
                status=status,
                source="text_aligner",
                confidence=0.4 if status != "direct" else 0.95,
                reason="seed",
            )
        )
    failed_count = sum(1 for status in statuses if status == "failed")
    return FinalAlignmentResult(
        items=tuple(items),
        route="error" if failed_count > 0 else "text",
        metrics=AlignmentMetrics(coverage=(len(tokens) - failed_count) / len(tokens), failed_count=failed_count),
        error_code="TEXT_ALIGNMENT_FAILED" if failed_count > 0 else None,
    )


def test_phonetic_aligner_ja_kouen_homophone_rescue() -> None:
    aligner = PhoneticAligner()
    result = aligner.rescue_window(
        text_alignment=_text_result(["公園"], ["failed"]),
        time_base=_time_base(["講演"]),
        text_truth=_text_truth(["公園"]),
        language_runs=_runs("公園"),
        pronunciation=_pron(["こうえん"], mode="sudachi"),
    )
    assert result.alignment.items[0].status == "phonetic"
    assert result.alignment.route == "phonetic"


def test_phonetic_aligner_ja_prefers_sudachi_mode() -> None:
    aligner = PhoneticAligner()
    result = aligner.rescue_window(
        text_alignment=_text_result(["公園"], ["failed"]),
        time_base=_time_base(["講演"]),
        text_truth=_text_truth(["公園"]),
        language_runs=_runs("公園"),
        pronunciation=_pron(["こうえん"], mode="sudachi"),
    )
    assert result.report["fallback_reason"] == ""


def test_phonetic_aligner_ja_lexicon_fallback_is_explicit() -> None:
    aligner = PhoneticAligner()
    result = aligner.rescue_window(
        text_alignment=_text_result(["公園"], ["failed"]),
        time_base=_time_base(["講演"]),
        text_truth=_text_truth(["公園"]),
        language_runs=_runs("公園"),
        pronunciation=_pron(["こうえん"], mode="lexicon_fallback"),
    )
    assert result.report["fallback_reason"] == "ja_lexicon_fallback"


def test_phonetic_aligner_ja_does_not_apply_zh_fuzzy_logic() -> None:
    aligner = PhoneticAligner()
    result = aligner.rescue_window(
        text_alignment=_text_result(["はし"], ["failed"]),
        time_base=_time_base(["ばし"]),
        text_truth=_text_truth(["はし"]),
        language_runs=_runs("はし"),
        pronunciation=_pron(["はし"], mode="sudachi"),
    )
    assert result.alignment.items[0].status == "failed"

