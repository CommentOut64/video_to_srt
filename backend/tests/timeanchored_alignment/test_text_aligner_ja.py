from __future__ import annotations

from app.services.language_policy.types import WINDOW_KIND_SINGLE_LANGUAGE
from app.services.timeanchored_alignment.contracts import (
    AcousticCandidate,
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


def _time_base(tokens: list[str]) -> TimeBasePackage:
    units = tuple(
        TimeBaseUnit(
            text=token,
            start=idx * 0.12,
            end=idx * 0.12 + 0.1,
            confidence=0.9,
            token_type="raw",
        )
        for idx, token in enumerate(tokens)
    )
    return TimeBasePackage(
        raw_units=units,
        word_units=units,
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.84, low_prob_ratio=0.1),
        language="ja",
    )


def _text_truth(tokens: list[str], *, protected_spans: tuple[ProtectedSpan, ...] = ()) -> TextTruthPackage:
    units = tuple(
        TextTruthUnit(
            text=token,
            normalized_text=token,
            confidence=0.95,
            language="ja",
        )
        for token in tokens
    )
    raw = "".join(tokens)
    return TextTruthPackage(
        units=units,
        quality=TextTruthQuality(hallucination_risk=0.0, repetition_ratio=0.0, length_ratio=1.0),
        language="ja",
        raw_text=raw,
        normalized_text=raw,
        protected_spans=protected_spans,
    )


def _runs(text: str) -> LanguageRunPackage:
    return LanguageRunPackage(
        runs=(LanguageRun(run_text=text, run_language="ja", char_start=0, char_end=len(text)),),
        dominant_language="ja",
        window_kind=WINDOW_KIND_SINGLE_LANGUAGE,
        foreign_run_ratio=0.0,
        source_text=text,
    )


def _pron(tokens: list[str]) -> PronunciationPackage:
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
        dependency_mode={"ja": "sudachi"},
        language="ja",
    )


def test_text_aligner_ja_script_aware_normalization() -> None:
    aligner = TextAligner(
        threshold_overrides={
            "text_direct_ratio_min": 0.0,
            "text_estimated_ratio_max": 1.0,
        }
    )
    result = aligner.align_window(
        time_base=_time_base(["カタカナ"]),
        text_truth=_text_truth(["かたかな"]),
        language_runs=_runs("かたかな"),
        pronunciation=_pron(["かたかな"]),
    )
    assert result.route == "text"
    assert result.items[0].status == "estimated"


def test_text_aligner_ja_keeps_middle_dot_protected_structure() -> None:
    aligner = TextAligner()
    token = "OpenAI・API"
    result = aligner.align_window(
        time_base=_time_base([token]),
        text_truth=_text_truth(
            [token],
            protected_spans=(ProtectedSpan(start=0, end=len(token), kind="middle_dot", text=token),),
        ),
        language_runs=_runs(token),
        pronunciation=_pron([token]),
    )
    assert result.route == "text"
    assert result.items[0].status == "direct"


def test_text_aligner_ja_kanji_kana_variant_behavior() -> None:
    aligner = TextAligner(
        threshold_overrides={
            "text_direct_ratio_min": 0.0,
            "text_estimated_ratio_max": 1.0,
        }
    )
    result = aligner.align_window(
        time_base=_time_base(["公園"]),
        text_truth=_text_truth(["こうえん"]),
        language_runs=_runs("こうえん"),
        pronunciation=_pron(["こうえん"]),
    )
    assert result.route == "text"
    assert result.items[0].status == "estimated"


def test_text_aligner_ja_pronunciation_tie_break_for_kouen_ambiguity() -> None:
    aligner = TextAligner(
        threshold_overrides={
            "text_direct_ratio_min": 0.0,
            "text_estimated_ratio_max": 1.0,
        }
    )
    ambiguous_unit = TimeBaseUnit(
        text="講演",
        start=0.0,
        end=0.1,
        confidence=0.9,
        top_candidates=(
            AcousticCandidate(text="講演", score=0.51),
            AcousticCandidate(text="公園", score=0.49),
        ),
    )
    base = TimeBasePackage(
        raw_units=(ambiguous_unit,),
        word_units=(ambiguous_unit,),
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.84, low_prob_ratio=0.1),
        language="ja",
    )
    truth = _text_truth(["公園"])
    runs = _runs("公園")

    with_pron = aligner.align_window(
        time_base=base,
        text_truth=truth,
        language_runs=runs,
        pronunciation=_pron(["公園"]),
    )
    no_pron = aligner.align_window(
        time_base=base,
        text_truth=truth,
        language_runs=runs,
        pronunciation=_pron(["別語"]),
    )

    assert with_pron.items[0].status == "estimated"
    assert no_pron.items[0].status == "estimated"
    assert (with_pron.items[0].confidence or 0.0) > (no_pron.items[0].confidence or 0.0)


def test_text_aligner_ja_explicitly_degrades_when_not_using_sudachi() -> None:
    aligner = TextAligner(
        threshold_overrides={
            "text_direct_ratio_min": 0.0,
            "text_estimated_ratio_max": 1.0,
        }
    )
    ambiguous_unit = TimeBaseUnit(
        text="講演",
        start=0.0,
        end=0.1,
        confidence=0.9,
        top_candidates=(
            AcousticCandidate(text="講演", score=0.51),
            AcousticCandidate(text="公園", score=0.49),
        ),
    )
    base = TimeBasePackage(
        raw_units=(ambiguous_unit,),
        word_units=(ambiguous_unit,),
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.84, low_prob_ratio=0.1),
        language="ja",
    )
    truth = _text_truth(["公園"])
    runs = _runs("公園")

    sudachi_result = aligner.align_window(
        time_base=base,
        text_truth=truth,
        language_runs=runs,
        pronunciation=_pron(["公園"]),
    )
    fallback_result = aligner.align_window(
        time_base=base,
        text_truth=truth,
        language_runs=runs,
        pronunciation=PronunciationPackage(
            token_units=(TokenUnit(token_text="公園", language="ja", char_start=0, char_end=2),),
            phone_units=(PhoneUnit(phone_text="ja-こうえん", language="ja"),),
            token_to_phone_spans=(TokenToPhoneSpan(token_index=0, phone_start=0, phone_end=0),),
            frontend_source="test",
            dependency_mode={"ja": "lexicon_fallback"},
            language="ja",
        ),
    )

    assert fallback_result.items[0].status == "estimated"
    assert "ja_pron_fallback" in fallback_result.items[0].reason
    assert (sudachi_result.items[0].confidence or 0.0) > (fallback_result.items[0].confidence or 0.0)
