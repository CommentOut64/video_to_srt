from __future__ import annotations

from app.services.language_policy.types import WINDOW_KIND_DOMINANT_WITH_ISLANDS
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


def _time_base(tokens: list[str], *, language: str) -> TimeBasePackage:
    units = tuple(
        TimeBaseUnit(
            text=token,
            start=idx * 0.12,
            end=idx * 0.12 + 0.1,
            confidence=0.91,
        )
        for idx, token in enumerate(tokens)
    )
    return TimeBasePackage(
        raw_units=units,
        word_units=units,
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.85, low_prob_ratio=0.1),
        language=language,
    )


def _text_truth(tokens: list[str], *, language: str) -> TextTruthPackage:
    units = tuple(
        TextTruthUnit(
            text=token,
            normalized_text=token.lower(),
            confidence=0.94,
            language=language,
        )
        for token in tokens
    )
    raw_text = " ".join(tokens)
    return TextTruthPackage(
        units=units,
        quality=TextTruthQuality(hallucination_risk=0.0, repetition_ratio=0.0, length_ratio=1.0),
        language=language,
        raw_text=raw_text,
        normalized_text=raw_text.lower(),
    )


def _pron(tokens: list[str], *, language: str) -> PronunciationPackage:
    token_units = []
    phone_units = []
    spans = []
    cursor = 0
    for token in tokens:
        token_units.append(TokenUnit(token_text=token, language=language, char_start=cursor, char_end=cursor + len(token)))
        phone_units.append(PhoneUnit(phone_text=f"ph-{token.lower()}", language=language))
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
        dependency_mode={language: "rule_fallback"},
        language=language,
    )


def test_foreign_run_brand_version_api_in_zh_main_window() -> None:
    aligner = TextAligner()
    tokens = ["今天", "发布", "OpenAI", "v3.3.0", "API"]
    result = aligner.align_window(
        time_base=_time_base(tokens, language="zh"),
        text_truth=_text_truth(tokens, language="zh"),
        language_runs=LanguageRunPackage(
            runs=(
                LanguageRun(run_text="今天发布", run_language="zh", char_start=0, char_end=4),
                LanguageRun(run_text="OpenAI", run_language="en", char_start=5, char_end=11, is_foreign_island=True),
                LanguageRun(run_text="v3.3.0", run_language="en", char_start=12, char_end=18, is_protected=True, is_foreign_island=True),
                LanguageRun(run_text="API", run_language="en", char_start=19, char_end=22, is_foreign_island=True),
            ),
            dominant_language="zh",
            window_kind=WINDOW_KIND_DOMINANT_WITH_ISLANDS,
            foreign_run_ratio=0.36,
            source_text="今天发布 OpenAI v3.3.0 API",
        ),
        pronunciation=_pron(tokens, language="zh"),
    )
    assert result.route == "text"
    assert all(item.status in {"direct", "estimated"} for item in result.items)


def test_foreign_run_english_loanword_in_ja_main_window() -> None:
    aligner = TextAligner()
    tokens = ["今日は", "インフラ", "update", "する"]
    result = aligner.align_window(
        time_base=_time_base(tokens, language="ja"),
        text_truth=_text_truth(tokens, language="ja"),
        language_runs=LanguageRunPackage(
            runs=(
                LanguageRun(run_text="今日はインフラ", run_language="ja", char_start=0, char_end=7),
                LanguageRun(run_text="update", run_language="en", char_start=8, char_end=14, is_foreign_island=True),
                LanguageRun(run_text="する", run_language="ja", char_start=15, char_end=17),
            ),
            dominant_language="ja",
            window_kind=WINDOW_KIND_DOMINANT_WITH_ISLANDS,
            foreign_run_ratio=0.28,
            source_text="今日はインフラ update する",
        ),
        pronunciation=_pron(tokens, language="ja"),
    )
    assert result.route == "text"


def test_foreign_run_ratio_too_high_falls_back_to_mixed() -> None:
    aligner = TextAligner()
    tokens = ["你好", "OpenAI", "API", "hello"]
    result = aligner.align_window(
        time_base=_time_base(tokens, language="zh"),
        text_truth=_text_truth(tokens, language="zh"),
        language_runs=LanguageRunPackage(
            runs=(
                LanguageRun(run_text="你好", run_language="zh", char_start=0, char_end=2),
                LanguageRun(run_text="OpenAI", run_language="en", char_start=3, char_end=9, is_foreign_island=True),
                LanguageRun(run_text="API", run_language="en", char_start=10, char_end=13, is_foreign_island=True),
                LanguageRun(run_text="hello", run_language="en", char_start=14, char_end=19, is_foreign_island=True),
            ),
            dominant_language="zh",
            window_kind=WINDOW_KIND_DOMINANT_WITH_ISLANDS,
            foreign_run_ratio=0.8,
            source_text="你好 OpenAI API hello",
        ),
        pronunciation=_pron(tokens, language="zh"),
    )
    assert result.route == "mixed"
    assert result.error_code == "FOREIGN_RUN_RATIO_EXCEEDED"

