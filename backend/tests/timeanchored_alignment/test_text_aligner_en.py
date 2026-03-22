from __future__ import annotations

from app.services.language_policy.types import (
    WINDOW_KIND_DOMINANT_WITH_ISLANDS,
    WINDOW_KIND_SINGLE_LANGUAGE,
)
from app.services.timeanchored_alignment.contracts import (
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
from app.services.timeanchored_alignment.text_aligner import TextAligner


def _build_time_base_for_en(*, raw_tokens: list[str], word_tokens: list[str]) -> TimeBasePackage:
    raw_units = tuple(
        TimeBaseUnit(
            text=token,
            start=idx * 0.2,
            end=idx * 0.2 + 0.16,
            confidence=0.9,
            token_type="raw",
        )
        for idx, token in enumerate(raw_tokens)
    )
    word_units = tuple(
        TimeBaseUnit(
            text=token,
            start=idx * 0.2,
            end=idx * 0.2 + 0.16,
            confidence=0.93,
            token_type="word",
        )
        for idx, token in enumerate(word_tokens)
    )
    return TimeBasePackage(
        raw_units=raw_units,
        word_units=word_units,
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.86, low_prob_ratio=0.1),
        language="en",
    )


def _build_text_truth_en(tokens: list[str], *, protected_spans: tuple[ProtectedSpan, ...] = ()) -> TextTruthPackage:
    units = tuple(
        TextTruthUnit(
            text=token,
            normalized_text=token.lower(),
            confidence=0.95,
            language="en",
        )
        for token in tokens
    )
    raw_text = " ".join(tokens)
    return TextTruthPackage(
        units=units,
        quality=TextTruthQuality(hallucination_risk=0.0, repetition_ratio=0.0, length_ratio=1.0),
        language="en",
        raw_text=raw_text,
        normalized_text=raw_text.lower(),
        protected_spans=protected_spans,
    )


def _pronunciation(tokens: list[str], *, language: str = "en") -> PronunciationPackage:
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
        dependency_mode={"en": "cmudict"},
        language=language,
    )


def test_text_aligner_en_word_units_exact_match() -> None:
    aligner = TextAligner()
    result = aligner.align_window(
        time_base=_build_time_base_for_en(raw_tokens=["rawA", "rawB"], word_tokens=["OpenAI", "API"]),
        text_truth=_build_text_truth_en(["OpenAI", "API"]),
        language_runs=LanguageRunPackage(
            runs=(LanguageRun(run_text="OpenAI API", run_language="en", char_start=0, char_end=10),),
            dominant_language="en",
            window_kind=WINDOW_KIND_SINGLE_LANGUAGE,
            foreign_run_ratio=0.0,
            source_text="OpenAI API",
        ),
        pronunciation=_pronunciation(["OpenAI", "API"]),
    )
    assert result.route == "text"
    assert [item.status for item in result.items] == ["direct", "direct"]


def test_text_aligner_en_contraction_expansion_normalized_match() -> None:
    aligner = TextAligner(
        threshold_overrides={
            "text_direct_ratio_min": 0.5,
            "text_estimated_ratio_max": 0.5,
        }
    )
    result = aligner.align_window(
        time_base=_build_time_base_for_en(raw_tokens=["r1", "r2"], word_tokens=["don't", "panic"]),
        text_truth=_build_text_truth_en(["dont", "panic"]),
        language_runs=LanguageRunPackage(
            runs=(LanguageRun(run_text="dont panic", run_language="en", char_start=0, char_end=10),),
            dominant_language="en",
            window_kind=WINDOW_KIND_SINGLE_LANGUAGE,
            foreign_run_ratio=0.0,
            source_text="dont panic",
        ),
        pronunciation=_pronunciation(["dont", "panic"]),
    )
    assert result.route == "text"
    assert result.items[0].status == "estimated"
    assert result.items[1].status == "direct"


def test_text_aligner_en_protected_structures_remain_stable() -> None:
    tokens = ["U.S.", "don't", "state-of-the-art"]
    raw_text = " ".join(tokens)
    protected = (
        ProtectedSpan(start=0, end=4, kind="acronym", text="U.S."),
        ProtectedSpan(start=5, end=10, kind="apostrophe", text="don't"),
        ProtectedSpan(start=11, end=len(raw_text), kind="hyphen", text="state-of-the-art"),
    )
    aligner = TextAligner()
    result = aligner.align_window(
        time_base=_build_time_base_for_en(raw_tokens=["r1", "r2", "r3"], word_tokens=tokens),
        text_truth=_build_text_truth_en(tokens, protected_spans=protected),
        language_runs=LanguageRunPackage(
            runs=(LanguageRun(run_text=raw_text, run_language="en", char_start=0, char_end=len(raw_text)),),
            dominant_language="en",
            window_kind=WINDOW_KIND_SINGLE_LANGUAGE,
            foreign_run_ratio=0.0,
            source_text=raw_text,
        ),
        pronunciation=_pronunciation(tokens),
    )
    assert result.route == "text"
    assert [item.status for item in result.items] == ["direct", "direct", "direct"]


def test_text_aligner_allows_english_island_inside_chinese_window() -> None:
    aligner = TextAligner()
    result = aligner.align_window(
        time_base=_build_time_base_for_en(
            raw_tokens=["今天", "OpenAI", "发布"],
            word_tokens=["今天", "OpenAI", "发布"],
        ),
        text_truth=TextTruthPackage(
            units=(
                TextTruthUnit(text="今天", normalized_text="今天", confidence=0.9, language="zh"),
                TextTruthUnit(text="OpenAI", normalized_text="openai", confidence=0.9, language="en"),
                TextTruthUnit(text="发布", normalized_text="发布", confidence=0.9, language="zh"),
            ),
            quality=TextTruthQuality(hallucination_risk=0.0, repetition_ratio=0.0, length_ratio=1.0),
            language="zh",
            raw_text="今天 OpenAI 发布",
            normalized_text="今天 openai 发布",
        ),
        language_runs=LanguageRunPackage(
            runs=(
                LanguageRun(run_text="今天", run_language="zh", char_start=0, char_end=2),
                LanguageRun(run_text="OpenAI", run_language="en", char_start=3, char_end=9, is_foreign_island=True),
                LanguageRun(run_text="发布", run_language="zh", char_start=10, char_end=12),
            ),
            dominant_language="zh",
            window_kind=WINDOW_KIND_DOMINANT_WITH_ISLANDS,
            foreign_run_ratio=0.25,
            source_text="今天 OpenAI 发布",
        ),
        pronunciation=_pronunciation(["今天", "OpenAI", "发布"], language="zh"),
    )
    assert result.route == "text"
    assert all(item.status in {"direct", "estimated"} for item in result.items)


def test_text_aligner_en_rejects_subword_main_path() -> None:
    aligner = TextAligner()
    result = aligner.align_window(
        time_base=_build_time_base_for_en(
            raw_tokens=["hello", "world"],
            word_tokens=["▁hel", "lo", "▁world"],
        ),
        text_truth=_build_text_truth_en(["hello", "world"]),
        language_runs=LanguageRunPackage(
            runs=(LanguageRun(run_text="hello world", run_language="en", char_start=0, char_end=11),),
            dominant_language="en",
            window_kind=WINDOW_KIND_SINGLE_LANGUAGE,
            foreign_run_ratio=0.0,
            source_text="hello world",
        ),
        pronunciation=_pronunciation(["hello", "world"]),
    )
    assert result.route == "error"
    assert result.error_code == "EN_SUBWORD_MAIN_PATH_FORBIDDEN"

