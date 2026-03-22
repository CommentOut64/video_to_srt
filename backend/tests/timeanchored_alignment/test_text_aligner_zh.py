from __future__ import annotations

from app.services.language_policy.types import WINDOW_KIND_SINGLE_LANGUAGE
from app.services.timeanchored_alignment.contracts import (
    AcousticCandidate,
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


def _time_base(tokens: list[str]) -> TimeBasePackage:
    raw_units = []
    for idx, token in enumerate(tokens):
        raw_units.append(
            TimeBaseUnit(
                text=token,
                start=idx * 0.1,
                end=idx * 0.1 + 0.08,
                confidence=0.9,
                token_type="raw",
            )
        )
    return TimeBasePackage(
        raw_units=tuple(raw_units),
        word_units=tuple(raw_units),
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.85, low_prob_ratio=0.1),
        language="zh",
    )


def _text_truth(tokens: list[str], *, raw_text: str | None = None) -> TextTruthPackage:
    units = tuple(
        TextTruthUnit(text=token, normalized_text=token, confidence=0.95, language="zh")
        for token in tokens
    )
    raw = raw_text if raw_text is not None else "".join(tokens)
    return TextTruthPackage(
        units=units,
        quality=TextTruthQuality(hallucination_risk=0.0, repetition_ratio=0.0, length_ratio=1.0),
        language="zh",
        raw_text=raw,
        normalized_text=raw,
    )


def _language_runs(source_text: str) -> LanguageRunPackage:
    return LanguageRunPackage(
        runs=(LanguageRun(run_text=source_text, run_language="zh", char_start=0, char_end=len(source_text)),),
        dominant_language="zh",
        window_kind=WINDOW_KIND_SINGLE_LANGUAGE,
        foreign_run_ratio=0.0,
        source_text=source_text,
    )


def _pronunciation(tokens: list[str]) -> PronunciationPackage:
    token_units = []
    phone_units = []
    spans = []
    cursor = 0
    for token in tokens:
        token_units.append(TokenUnit(token_text=token, language="zh", char_start=cursor, char_end=cursor + len(token)))
        phone_units.append(PhoneUnit(phone_text=f"p-{token}", language="zh"))
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


def test_text_aligner_zh_exact_match() -> None:
    aligner = TextAligner()
    result = aligner.align_window(
        time_base=_time_base(["你", "好", "世", "界"]),
        text_truth=_text_truth(["你", "好", "世", "界"]),
        language_runs=_language_runs("你好世界"),
        pronunciation=_pronunciation(["你", "好", "世", "界"]),
    )
    assert result.route == "text"
    assert [item.status for item in result.items] == ["direct", "direct", "direct", "direct"]


def test_text_aligner_zh_merge_split_can_be_corrected() -> None:
    aligner = TextAligner(
        threshold_overrides={
            "text_direct_ratio_min": 0.5,
            "text_estimated_ratio_max": 0.6,
        }
    )
    result = aligner.align_window(
        time_base=_time_base(["今天", "发布", "了"]),
        text_truth=_text_truth(["今", "天", "发布", "了"], raw_text="今天发布了"),
        language_runs=_language_runs("今天发布了"),
        pronunciation=_pronunciation(["今", "天", "发布", "了"]),
    )
    assert result.route == "text"
    assert any(item.status in {"estimated", "interpolated"} for item in result.items)
    assert all(item.status != "failed" for item in result.items)


def test_text_aligner_zh_top_candidates_works_on_local_ambiguity() -> None:
    aligner = TextAligner(
        threshold_overrides={
            "text_direct_ratio_min": 0.0,
            "text_estimated_ratio_max": 1.0,
        }
    )
    ambiguous_unit = TimeBaseUnit(
        text="呀",
        start=0.0,
        end=0.08,
        confidence=0.9,
        top_candidates=(
            AcousticCandidate(text="呀", score=0.51),
            AcousticCandidate(text="啊", score=0.49),
        ),
    )
    no_candidate_unit = TimeBaseUnit(
        text="呀",
        start=0.0,
        end=0.08,
        confidence=0.9,
    )
    base_with_ambiguity = TimeBasePackage(
        raw_units=(ambiguous_unit,),
        word_units=(ambiguous_unit,),
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.8, low_prob_ratio=0.1),
        language="zh",
    )
    base_plain = TimeBasePackage(
        raw_units=(no_candidate_unit,),
        word_units=(no_candidate_unit,),
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.8, low_prob_ratio=0.1),
        language="zh",
    )
    truth = _text_truth(["啊"])
    runs = _language_runs("啊")
    pron = _pronunciation(["啊"])

    with_amb = aligner.align_window(time_base=base_with_ambiguity, text_truth=truth, language_runs=runs, pronunciation=pron)
    plain = aligner.align_window(time_base=base_plain, text_truth=truth, language_runs=runs, pronunciation=pron)

    assert with_amb.items[0].status == "estimated"
    assert plain.items[0].status == "estimated"
    assert (with_amb.items[0].confidence or 0.0) > (plain.items[0].confidence or 0.0)


def test_text_aligner_zh_polyphone_ambiguity_uses_pron_tie_break_only() -> None:
    aligner = TextAligner(
        threshold_overrides={
            "text_direct_ratio_min": 0.0,
            "text_estimated_ratio_max": 1.0,
        }
    )
    ambiguous_unit = TimeBaseUnit(
        text="行",
        start=0.0,
        end=0.08,
        confidence=0.9,
        top_candidates=(
            AcousticCandidate(text="行", score=0.51),
            AcousticCandidate(text="型", score=0.49),
        ),
    )
    base = TimeBasePackage(
        raw_units=(ambiguous_unit,),
        word_units=(ambiguous_unit,),
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.8, low_prob_ratio=0.1),
        language="zh",
    )
    truth = _text_truth(["型"])
    runs = _language_runs("型")

    with_pron = aligner.align_window(
        time_base=base,
        text_truth=truth,
        language_runs=runs,
        pronunciation=_pronunciation(["型"]),
    )
    no_pron = aligner.align_window(
        time_base=base,
        text_truth=truth,
        language_runs=runs,
        pronunciation=_pronunciation(["别的词"]),
    )

    assert with_pron.items[0].status == "estimated"
    assert no_pron.items[0].status == "estimated"
    assert (with_pron.items[0].confidence or 0.0) > (no_pron.items[0].confidence or 0.0)


def test_text_aligner_zh_fails_on_large_text_mismatch() -> None:
    aligner = TextAligner()
    result = aligner.align_window(
        time_base=_time_base(["你", "好", "世", "界"]),
        text_truth=_text_truth(["甲", "乙", "丙", "丁"]),
        language_runs=_language_runs("甲乙丙丁"),
        pronunciation=_pronunciation(["甲", "乙", "丙", "丁"]),
    )
    assert result.route == "error"
    assert result.error_code in {"TEXT_DIRECT_RATIO_LOW", "TEXT_ESTIMATED_RATIO_HIGH", "TEXT_ALIGNMENT_FAILED"}
