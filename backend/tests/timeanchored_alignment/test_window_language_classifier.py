from __future__ import annotations

from app.models.sensevoice_models import SentenceSegment
from app.services.punctuation.semantic_buffer import SemanticChunk
from app.services.timeanchored_alignment.window_language_classifier import (
    MIXED_DECISION_DOMAINS,
    WindowLanguageClassifier,
)


def _build_chunk(
    *,
    chunk_id: str,
    text: str,
    language: str,
    start: float,
    end: float,
    speaker_id: str = "spk-1",
) -> SemanticChunk:
    return SemanticChunk(
        chunk_id=chunk_id,
        text=text,
        sentences=[SentenceSegment(text=text, text_clean=text, start=start, end=end)],
        punctuation_result=None,
        punctuation_decision=None,
        pending_tail="",
        audio_range=(start, end),
        language=language,
        source_chunks=[f"chunk-{chunk_id}"],
        speaker_id=speaker_id,
    )


def test_window_language_classifier_detects_zh_primary() -> None:
    classifier = WindowLanguageClassifier()
    chunks = [
        _build_chunk(chunk_id="1", text="这是中文句子。", language="zh", start=0.0, end=2.0),
        _build_chunk(chunk_id="2", text="继续中文内容。", language="zh", start=2.0, end=4.0),
    ]

    decision = classifier.classify(chunks, slow_language_hint="zh")

    assert decision.primary_language == "zh"
    assert decision.is_mixed_window is False
    assert decision.can_enter_main_chain is True


def test_window_language_classifier_detects_ja_primary() -> None:
    classifier = WindowLanguageClassifier()
    chunks = [
        _build_chunk(chunk_id="1", text="これは日本語です。", language="ja", start=0.0, end=2.0),
        _build_chunk(chunk_id="2", text="次の文も日本語です。", language="ja", start=2.0, end=4.0),
    ]

    decision = classifier.classify(chunks, slow_language_hint="ja")

    assert decision.primary_language == "ja"
    assert decision.is_mixed_window is False
    assert decision.can_enter_main_chain is True


def test_window_language_classifier_detects_en_primary() -> None:
    classifier = WindowLanguageClassifier()
    chunks = [
        _build_chunk(chunk_id="1", text="This is an English sentence.", language="en", start=0.0, end=2.0),
        _build_chunk(chunk_id="2", text="Another English sentence follows.", language="en", start=2.0, end=4.0),
    ]

    decision = classifier.classify(chunks, slow_language_hint="en")

    assert decision.primary_language == "en"
    assert decision.is_mixed_window is False
    assert decision.can_enter_main_chain is True


def test_window_language_classifier_detects_mixed_window() -> None:
    classifier = WindowLanguageClassifier()
    chunks = [
        _build_chunk(chunk_id="1", text="这是中文片段。", language="zh", start=0.0, end=2.0),
        _build_chunk(chunk_id="2", text="This is English content.", language="en", start=2.0, end=4.0),
    ]

    decision = classifier.classify(chunks, slow_language_hint="auto")

    assert decision.primary_language == "mixed"
    assert decision.is_mixed_window is True
    assert decision.can_enter_main_chain is False


def test_mixed_window_only_supports_flush_split_fallback() -> None:
    classifier = WindowLanguageClassifier()
    chunks = [
        _build_chunk(chunk_id="1", text="你好 world", language="mixed", start=0.0, end=2.0),
        _build_chunk(chunk_id="2", text="さようなら goodbye", language="mixed", start=2.0, end=4.0),
    ]

    decision = classifier.classify(chunks, slow_language_hint="auto")

    assert decision.is_mixed_window is True
    assert decision.decision_domains == MIXED_DECISION_DOMAINS
