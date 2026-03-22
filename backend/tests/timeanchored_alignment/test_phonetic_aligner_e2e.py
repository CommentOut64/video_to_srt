from __future__ import annotations

from dataclasses import dataclass

from app.services.homophone.tokenizers import TokenReading
from app.services.timeanchored_alignment.contracts import (
    TextTruthPackage,
    TextTruthQuality,
    TextTruthUnit,
    TimeBasePackage,
    TimeBaseQuality,
    TimeBaseUnit,
)
from app.services.timeanchored_alignment.language_run_frontend import LanguageRunFrontend
from app.services.timeanchored_alignment.phonetic_aligner import PhoneticAligner
from app.services.timeanchored_alignment.pronunciation_frontend import PronunciationFrontend
from app.services.timeanchored_alignment.text_aligner import TextAligner


@dataclass
class _FakeTokenizer:
    def tokenize(self, text: str, language: str) -> list[TokenReading]:
        token = str(text or "").strip()
        if not token:
            return []
        key = token.lower()
        return [
            TokenReading(
                token_text=token,
                reading_key=key,
                reading_key_fuzzy=key,
                reading_key_no_punct=key,
                reading_key_fuzzy_no_punct=key,
                char_start=0,
                char_end=len(token),
            )
        ]


def _time_base(tokens: list[str], *, language: str) -> TimeBasePackage:
    units = tuple(
        TimeBaseUnit(text=token, start=idx * 0.12, end=idx * 0.12 + 0.1, confidence=0.9)
        for idx, token in enumerate(tokens)
    )
    return TimeBasePackage(
        raw_units=units,
        word_units=units,
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.84, low_prob_ratio=0.1),
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


def test_phonetic_aligner_e2e_zh() -> None:
    language_frontend = LanguageRunFrontend()
    pronunciation_frontend = PronunciationFrontend(tokenizer=_FakeTokenizer())
    text_aligner = TextAligner(phonetic_aligner=PhoneticAligner())

    text = "编"
    runs = language_frontend.build_runs(text=text, language_hint="zh")
    pronunciation = pronunciation_frontend.build_package(
        text=text,
        language_runs=runs.runs,
        dominant_language=runs.dominant_language,
    )
    result = text_aligner.align_window(
        time_base=_time_base(["边"], language="zh"),
        text_truth=_text_truth(["编"], language="zh"),
        language_runs=runs,
        pronunciation=pronunciation,
    )
    assert result.route == "phonetic"
    assert result.items[0].status == "phonetic"


def test_phonetic_aligner_e2e_en() -> None:
    language_frontend = LanguageRunFrontend()
    pronunciation_frontend = PronunciationFrontend(tokenizer=_FakeTokenizer())
    text_aligner = TextAligner(phonetic_aligner=PhoneticAligner())

    text = "write"
    runs = language_frontend.build_runs(text=text, language_hint="en")
    pronunciation = pronunciation_frontend.build_package(
        text=text,
        language_runs=runs.runs,
        dominant_language=runs.dominant_language,
    )
    result = text_aligner.align_window(
        time_base=_time_base(["right"], language="en"),
        text_truth=_text_truth(["write"], language="en"),
        language_runs=runs,
        pronunciation=pronunciation,
    )
    assert result.route == "phonetic"
    assert result.items[0].status == "phonetic"


def test_phonetic_aligner_e2e_ja() -> None:
    language_frontend = LanguageRunFrontend()
    pronunciation_frontend = PronunciationFrontend(tokenizer=_FakeTokenizer())
    text_aligner = TextAligner(phonetic_aligner=PhoneticAligner())

    text = "公園"
    runs = language_frontend.build_runs(text=text, language_hint="ja")
    pronunciation = pronunciation_frontend.build_package(
        text=text,
        language_runs=runs.runs,
        dominant_language=runs.dominant_language,
    )
    result = text_aligner.align_window(
        time_base=_time_base(["講演"], language="ja"),
        text_truth=_text_truth(["公園"], language="ja"),
        language_runs=runs,
        pronunciation=pronunciation,
    )
    assert result.route == "phonetic"
    assert result.items[0].status == "phonetic"

