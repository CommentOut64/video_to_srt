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
from app.services.timeanchored_alignment.pronunciation_frontend import PronunciationFrontend
from app.services.timeanchored_alignment.text_aligner import TextAligner


@dataclass
class _FakeTokenizer:
    def tokenize(self, text: str, language: str) -> list[TokenReading]:
        token = str(text or "")
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


def _build_time_base(tokens: list[str], *, language: str) -> TimeBasePackage:
    units = tuple(
        TimeBaseUnit(
            text=token,
            start=idx * 0.12,
            end=idx * 0.12 + 0.1,
            confidence=0.92,
            token_type="raw",
        )
        for idx, token in enumerate(tokens)
    )
    return TimeBasePackage(
        raw_units=units,
        word_units=units,
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.84, low_prob_ratio=0.1),
        language=language,
    )


def _build_text_truth(tokens: list[str], *, language: str) -> TextTruthPackage:
    units = tuple(
        TextTruthUnit(
            text=token,
            normalized_text=token.lower(),
            confidence=0.95,
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


def test_text_aligner_e2e_zh_pipeline() -> None:
    language_frontend = LanguageRunFrontend()
    pronunciation_frontend = PronunciationFrontend(tokenizer=_FakeTokenizer())
    aligner = TextAligner()

    text = "今天 发布 OpenAI"
    runs = language_frontend.build_runs(text=text, language_hint="zh")
    pronunciation = pronunciation_frontend.build_package(
        text=text,
        language_runs=runs.runs,
        dominant_language=runs.dominant_language,
    )
    result = aligner.align_window(
        time_base=_build_time_base(["今天", "发布", "OpenAI"], language="zh"),
        text_truth=_build_text_truth(["今天", "发布", "OpenAI"], language="zh"),
        language_runs=runs,
        pronunciation=pronunciation,
    )
    assert runs.window_kind in {"single_language", "dominant_with_islands"}
    assert result.route == "text"


def test_text_aligner_e2e_en_pipeline() -> None:
    language_frontend = LanguageRunFrontend()
    pronunciation_frontend = PronunciationFrontend(tokenizer=_FakeTokenizer())
    aligner = TextAligner()

    tokens = ["U.S.", "don't", "split", "state-of-the-art"]
    text = " ".join(tokens)
    runs = language_frontend.build_runs(text=text, language_hint="en")
    pronunciation = pronunciation_frontend.build_package(
        text=text,
        language_runs=runs.runs,
        dominant_language=runs.dominant_language,
    )
    result = aligner.align_window(
        time_base=_build_time_base(tokens, language="en"),
        text_truth=_build_text_truth(tokens, language="en"),
        language_runs=runs,
        pronunciation=pronunciation,
    )
    assert runs.dominant_language == "en"
    assert result.route == "text"
    assert [item.status for item in result.items] == ["direct", "direct", "direct", "direct"]


def test_text_aligner_e2e_ja_pipeline() -> None:
    language_frontend = LanguageRunFrontend()
    pronunciation_frontend = PronunciationFrontend(tokenizer=_FakeTokenizer())
    aligner = TextAligner()

    tokens = ["今日は", "OpenAI", "・", "API", "を", "使う"]
    text = " ".join(tokens)
    runs = language_frontend.build_runs(text=text, language_hint="ja")
    pronunciation = pronunciation_frontend.build_package(
        text=text,
        language_runs=runs.runs,
        dominant_language=runs.dominant_language,
    )
    result = aligner.align_window(
        time_base=_build_time_base(tokens, language="ja"),
        text_truth=_build_text_truth(tokens, language="ja"),
        language_runs=runs,
        pronunciation=pronunciation,
    )
    assert runs.dominant_language == "ja"
    assert result.route == "text"

