from __future__ import annotations

from dataclasses import dataclass

from app.services.timeanchored_alignment.adapters.slow.whisper_text_adapter import (
    WhisperTextAdapter,
)


@dataclass
class _DummySanitizer:
    cleaned_text: str

    def sanitize_minimal(self, text: str, prompt: str | None = None) -> str:
        return self.cleaned_text


@dataclass
class _DummyHallucinationDetector:
    result: bool

    def is_hallucination(self, result: dict, prompt: str | None = None) -> bool:
        return self.result


def test_whisper_text_adapter_builds_text_truth_package() -> None:
    adapter = WhisperTextAdapter(
        sanitizer=_DummySanitizer(cleaned_text="v3.3.0 don't split U.S."),
        hallucination_detector=_DummyHallucinationDetector(result=False),
    )
    package = adapter.build_text_truth_package(
        whisper_result={
            "raw_text": "  v3.3.0 don't split U.S.  ",
            "text": "v3.3.0 don't split U.S.",
            "language": "en",
            "confidence": 0.91,
            "prompt": "Context prompt",
        }
    )

    assert package.language == "en"
    assert package.raw_text == "v3.3.0 don't split U.S."
    assert package.normalized_text == "v3.3.0 don't split U.S."
    assert package.is_hallucination is False
    assert package.quality_signals["confidence"] == 0.91
    assert package.source_metadata["prompt"] == "Context prompt"
    assert package.units[0].text == "v3.3.0 don't split U.S."
    assert package.units[0].normalized_text == "v3.3.0 don't split U.S."


def test_whisper_text_adapter_preserves_explicit_hallucination_flag() -> None:
    adapter = WhisperTextAdapter(
        sanitizer=_DummySanitizer(cleaned_text="clean text"),
        hallucination_detector=_DummyHallucinationDetector(result=False),
    )
    package = adapter.build_text_truth_package(
        whisper_result={
            "raw_text": "clean text",
            "language": "zh",
            "is_hallucination": True,
            "confidence": 0.80,
        }
    )

    assert package.is_hallucination is True
    assert package.quality.hallucination_risk == 1.0


def test_whisper_text_adapter_uses_detector_when_flag_absent() -> None:
    adapter = WhisperTextAdapter(
        sanitizer=_DummySanitizer(cleaned_text="clean text"),
        hallucination_detector=_DummyHallucinationDetector(result=True),
    )
    package = adapter.build_text_truth_package(
        whisper_result={
            "raw_text": "clean text",
            "language": "ja",
            "confidence": 0.62,
        }
    )

    assert package.language == "ja"
    assert package.is_hallucination is True
    assert package.quality_signals["length_ratio"] >= 1.0
