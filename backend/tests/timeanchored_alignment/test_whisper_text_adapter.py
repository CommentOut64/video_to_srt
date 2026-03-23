from __future__ import annotations

from dataclasses import dataclass

from app.services.timeanchored_alignment.adapters.slow.whisper_text_adapter import (
    WhisperTextAdapter,
)


@dataclass
class _DummySanitizer:
    cleaned_text: str | None = None

    def sanitize_minimal(self, text: str, prompt: str | None = None) -> str:
        if self.cleaned_text is None:
            return text
        return self.cleaned_text


@dataclass
class _DummyHallucinationDetector:
    result: bool

    def is_hallucination(self, result: dict, prompt: str | None = None) -> bool:
        return self.result


def test_whisper_text_adapter_builds_text_truth_package() -> None:
    adapter = WhisperTextAdapter(
        sanitizer=_DummySanitizer(cleaned_text=None),
        hallucination_detector=_DummyHallucinationDetector(result=False),
    )
    package = adapter.build_text_truth_package(
        whisper_result={
            "raw_text": "  v3.3.0 don't split U.S.  ",
            "text": "v3.3.0 don't split U.S.",
            "language": "en",
            "confidence": 0.91,
            "prompt": "Context prompt",
            "raw_result": {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 0.8,
                        "text": "v3.3.0 don't split U.S.",
                        "words": [
                            {"word": "v3.3.0", "start": 0.0, "end": 0.2, "probability": 0.9},
                            {"word": "don't", "start": 0.2, "end": 0.4, "probability": 0.92},
                            {"word": "split", "start": 0.4, "end": 0.6, "probability": 0.91},
                            {"word": "U.S.", "start": 0.6, "end": 0.8, "probability": 0.9},
                        ],
                    }
                ],
            },
        }
    )

    assert package.language == "en"
    assert package.raw_text == "v3.3.0 don't split U.S."
    assert package.normalized_text == "v3.3.0 don't split U.S."
    assert package.is_hallucination is False
    assert package.quality_signals["confidence"] == 0.91
    assert package.source_metadata["prompt"] == "Context prompt"
    assert package.source_metadata["timestamp_mode"] == "word"
    assert len(package.units) == 4
    assert package.units[0].text == "v3.3.0"
    assert package.units[0].start == 0.0
    assert package.units[-1].text == "U.S."
    assert package.units[-1].end == 0.8


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


def test_whisper_text_adapter_falls_back_to_segment_splitting_when_no_word_timestamps() -> None:
    adapter = WhisperTextAdapter(
        sanitizer=_DummySanitizer(cleaned_text=None),
        hallucination_detector=_DummyHallucinationDetector(result=False),
    )
    package = adapter.build_text_truth_package(
        whisper_result={
            "raw_text": "你好世界",
            "text": "你好世界",
            "language": "zh",
            "confidence": 0.88,
            "raw_result": {
                "segments": [
                    {"start": 1.0, "end": 1.4, "text": "你好"},
                    {"start": 1.4, "end": 1.8, "text": "世界"},
                ],
            },
        }
    )

    assert package.source_metadata["timestamp_mode"] == "segment"
    assert len(package.units) == 4
    assert package.units[0].text == "你"
    assert package.units[0].start == 1.0
    assert package.units[-1].text == "界"
    assert package.units[-1].end == 1.8
