from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.services.homophone.tokenizers import TokenReading
from app.services.timeanchored_alignment.contracts import LanguageRun
from app.services.timeanchored_alignment.pronunciation_frontend import PronunciationFrontend


@dataclass
class _FakeTokenizer:
    calls: list[tuple[str, str]]

    def tokenize(self, text: str, language: str) -> list[TokenReading]:
        self.calls.append((language, text))
        if language == "zh":
            return [
                TokenReading("你", "ni3", "ni", "ni3", "ni", 0, 1),
                TokenReading("好", "hao3", "hao", "hao3", "hao", 1, 2),
            ]
        if language == "en":
            return [
                TokenReading("OpenAI", "ow|p|en|ai", "ow|p|en|ai", "ow|p|en|ai", "ow|p|en|ai", 0, 6),
            ]
        if language == "ja":
            return [
                TokenReading("東京", "とうきょう", "とうきょう", "とうきょう", "とうきょう", 0, 2),
            ]
        return []


def test_pronunciation_frontend_supports_zh_with_en_island() -> None:
    tokenizer = _FakeTokenizer(calls=[])
    frontend = PronunciationFrontend(
        tokenizer=tokenizer,
        dependency_mode_overrides={"zh": "pypinyin", "en": "cmudict"},
    )
    package = frontend.build_package(
        text="你好OpenAI",
        language_runs=[
            LanguageRun("你好", "zh", 0, 2),
            LanguageRun("OpenAI", "en", 2, 8, is_foreign_island=True),
        ],
        dominant_language="zh",
    )

    assert ("zh", "你好") in tokenizer.calls
    assert ("en", "OpenAI") in tokenizer.calls
    assert package.frontend_source == "homophone_tokenizer"
    assert package.dependency_mode["zh"] == "pypinyin"
    assert package.dependency_mode["en"] == "cmudict"
    assert len(package.token_units) == 3
    assert len(package.phone_units) >= 3
    assert len(package.token_to_phone_spans) == 3


def test_pronunciation_frontend_ja_supports_recommended_and_fallback_modes() -> None:
    tokenizer = _FakeTokenizer(calls=[])
    recommended = PronunciationFrontend(
        tokenizer=tokenizer,
        dependency_mode_overrides={"ja": "sudachi"},
    ).build_package(
        text="東京",
        language_runs=[LanguageRun("東京", "ja", 0, 2)],
        dominant_language="ja",
    )
    fallback = PronunciationFrontend(
        tokenizer=tokenizer,
        dependency_mode_overrides={"ja": "lexicon_fallback"},
    ).build_package(
        text="東京",
        language_runs=[LanguageRun("東京", "ja", 0, 2)],
        dominant_language="ja",
    )

    assert recommended.dependency_mode["ja"] == "sudachi"
    assert fallback.dependency_mode["ja"] == "lexicon_fallback"
    assert recommended.phone_units
    assert fallback.phone_units


def test_pronunciation_frontend_has_no_forbidden_heavy_dependencies() -> None:
    source_path = Path(__file__).resolve().parents[2] / "app" / "services" / "timeanchored_alignment" / "pronunciation_frontend.py"
    source = source_path.read_text(encoding="utf-8")
    lowered = source.lower()
    assert "bert" not in lowered
    assert "g2pw" not in lowered
    assert "pyopenjtalk" not in lowered
