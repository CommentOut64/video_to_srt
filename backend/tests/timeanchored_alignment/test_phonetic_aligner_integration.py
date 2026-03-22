from __future__ import annotations

from dataclasses import dataclass

from app.services.language_policy.types import (
    WINDOW_KIND_DOMINANT_WITH_ISLANDS,
    WINDOW_KIND_SINGLE_LANGUAGE,
    WINDOW_KIND_TRUE_MIXED,
)
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
from app.services.timeanchored_alignment.phonetic_aligner import (
    PhoneticAligner,
    PhoneticRescueResult,
)
from app.services.timeanchored_alignment.text_aligner import TextAligner


def _time_base(tokens: list[str], *, language: str) -> TimeBasePackage:
    units = tuple(
        TimeBaseUnit(text=token, start=idx * 0.1, end=idx * 0.1 + 0.08, confidence=0.9)
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
        TextTruthUnit(text=token, normalized_text=token.lower(), confidence=0.95, language=language)
        for token in tokens
    )
    joined = " ".join(tokens)
    return TextTruthPackage(
        units=units,
        quality=TextTruthQuality(hallucination_risk=0.0, repetition_ratio=0.0, length_ratio=1.0),
        language=language,
        raw_text=joined,
        normalized_text=joined.lower(),
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


@dataclass
class _SpyPhoneticAligner:
    called: int = 0

    def rescue_window(self, **kwargs) -> PhoneticRescueResult:
        self.called += 1
        src: FinalAlignmentResult = kwargs["text_alignment"]
        replaced_items = []
        for item in src.items:
            replaced_items.append(
                AlignmentItem(
                    text=item.text,
                    start=item.start,
                    end=item.end,
                    status="phonetic" if item.status != "direct" else "direct",
                    source="spy_phonetic",
                    confidence=item.confidence,
                    reason="spy_rescue",
                )
            )
        rescued = FinalAlignmentResult(
            items=tuple(replaced_items),
            route="phonetic",
            metrics=AlignmentMetrics(
                coverage=1.0,
                duration_ratio=1.0,
                failed_count=0,
                route_confidence=0.8,
            ),
            error_code=None,
        )
        return PhoneticRescueResult(
            alignment=rescued,
            traces=tuple(),
            report={"rescued_count": 1},
        )


def test_text_exact_success_does_not_trigger_phonetic_takeover() -> None:
    spy = _SpyPhoneticAligner()
    aligner = TextAligner(phonetic_aligner=spy)
    result = aligner.align_window(
        time_base=_time_base(["OpenAI", "API"], language="en"),
        text_truth=_text_truth(["OpenAI", "API"], language="en"),
        language_runs=LanguageRunPackage(
            runs=(LanguageRun(run_text="OpenAI API", run_language="en", char_start=0, char_end=10),),
            dominant_language="en",
            window_kind=WINDOW_KIND_SINGLE_LANGUAGE,
            foreign_run_ratio=0.0,
            source_text="OpenAI API",
        ),
        pronunciation=_pron(["OpenAI", "API"], language="en"),
    )
    assert spy.called == 0
    assert result.route == "text"
    assert [item.status for item in result.items] == ["direct", "direct"]


def test_unresolved_span_triggers_phonetic_layer() -> None:
    spy = _SpyPhoneticAligner()
    aligner = TextAligner(
        phonetic_aligner=spy,
        threshold_overrides={
            "text_direct_ratio_min": 0.9,
            "text_estimated_ratio_max": 1.0,
        },
    )
    result = aligner.align_window(
        time_base=_time_base(["甲"], language="zh"),
        text_truth=_text_truth(["乙"], language="zh"),
        language_runs=LanguageRunPackage(
            runs=(LanguageRun(run_text="乙", run_language="zh", char_start=0, char_end=1),),
            dominant_language="zh",
            window_kind=WINDOW_KIND_SINGLE_LANGUAGE,
            foreign_run_ratio=0.0,
            source_text="乙",
        ),
        pronunciation=_pron(["乙"], language="zh"),
    )
    assert spy.called == 1
    assert result.route == "phonetic"


def test_foreign_island_only_rescued_in_its_own_run() -> None:
    aligner = TextAligner(
        phonetic_aligner=PhoneticAligner(),
        threshold_overrides={
            "text_direct_ratio_min": 0.0,
            "text_estimated_ratio_max": 1.0,
        },
    )
    result = aligner.align_window(
        time_base=_time_base(["甲", "right"], language="zh"),
        text_truth=TextTruthPackage(
            units=(
                TextTruthUnit(text="乙", normalized_text="乙", confidence=0.95, language="zh"),
                TextTruthUnit(text="write", normalized_text="write", confidence=0.95, language="en"),
            ),
            quality=TextTruthQuality(hallucination_risk=0.0, repetition_ratio=0.0, length_ratio=1.0),
            language="zh",
            raw_text="乙 write",
            normalized_text="乙 write",
        ),
        language_runs=LanguageRunPackage(
            runs=(
                LanguageRun(run_text="乙", run_language="zh", char_start=0, char_end=1),
                LanguageRun(run_text="write", run_language="en", char_start=2, char_end=7, is_foreign_island=True),
            ),
            dominant_language="zh",
            window_kind=WINDOW_KIND_DOMINANT_WITH_ISLANDS,
            foreign_run_ratio=0.4,
            source_text="乙 write",
        ),
        pronunciation=PronunciationPackage(
            token_units=(
                TokenUnit(token_text="write", language="en", char_start=2, char_end=7, is_foreign_island=True),
            ),
            phone_units=(PhoneUnit(phone_text="rayt", language="en"),),
            token_to_phone_spans=(TokenToPhoneSpan(token_index=0, phone_start=0, phone_end=0),),
            frontend_source="test",
            dependency_mode={"en": "cmudict"},
            language="zh",
        ),
    )
    assert result.items[0].status != "phonetic"
    assert result.items[1].status == "phonetic"


def test_true_mixed_window_does_not_enter_phonetic_layer() -> None:
    spy = _SpyPhoneticAligner()
    aligner = TextAligner(phonetic_aligner=spy)
    result = aligner.align_window(
        time_base=_time_base(["你", "Hello"], language="zh"),
        text_truth=_text_truth(["你", "Hello"], language="zh"),
        language_runs=LanguageRunPackage(
            runs=(
                LanguageRun(run_text="你", run_language="zh", char_start=0, char_end=1),
                LanguageRun(run_text="Hello", run_language="en", char_start=2, char_end=7),
            ),
            dominant_language="mixed",
            window_kind=WINDOW_KIND_TRUE_MIXED,
            foreign_run_ratio=0.8,
            source_text="你 Hello",
        ),
        pronunciation=_pron(["你", "Hello"], language="zh"),
    )
    assert spy.called == 0
    assert result.route == "mixed"


def test_text_aligner_surfaces_phonetic_report_and_traces() -> None:
    aligner = TextAligner(
        phonetic_aligner=PhoneticAligner(),
        threshold_overrides={
            "text_direct_ratio_min": 0.9,
            "text_estimated_ratio_max": 1.0,
        },
    )
    result = aligner.align_window(
        time_base=_time_base(["边"], language="zh"),
        text_truth=_text_truth(["编"], language="zh"),
        language_runs=LanguageRunPackage(
            runs=(LanguageRun(run_text="编", run_language="zh", char_start=0, char_end=1),),
            dominant_language="zh",
            window_kind=WINDOW_KIND_SINGLE_LANGUAGE,
            foreign_run_ratio=0.0,
            source_text="编",
        ),
        pronunciation=_pron(["编"], language="zh"),
    )
    assert result.route == "phonetic"
    assert aligner.last_phonetic_report["rescued_count"] == 1
    assert len(aligner.last_phonetic_traces) == 1
    assert aligner.last_phonetic_traces[0].rescued is True
