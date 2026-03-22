from __future__ import annotations

from app.services.timeanchored_alignment.contracts import AlignmentItem
from app.services.timeanchored_alignment.sentence_segmenter import (
    SentenceSegmenter,
    SentenceSegmenterConfig,
)


def _item(
    text: str,
    *,
    start: float,
    end: float,
    source: str = "text_aligner",
) -> AlignmentItem:
    return AlignmentItem(
        text=text,
        start=start,
        end=end,
        status="direct",
        source=source,
        confidence=0.9,
    )


def test_long_pause_boundary() -> None:
    segmenter = SentenceSegmenter(config=SentenceSegmenterConfig(long_pause_gap_sec=0.5))
    stream = (
        _item("你", start=0.0, end=0.2),
        _item("好", start=0.2, end=0.4),
        _item("世", start=1.2, end=1.4),
        _item("界", start=1.4, end=1.6),
    )
    segments = segmenter.segment(stream=stream, language="zh")
    assert len(segments) == 2
    assert [seg.text for seg in segments] == ["你好", "世界"]


def test_speaker_change_boundary() -> None:
    segmenter = SentenceSegmenter(config=SentenceSegmenterConfig(long_pause_gap_sec=5.0))
    stream = (
        _item("我", start=0.0, end=0.2),
        _item("来", start=0.2, end=0.4),
        _item("了", start=0.4, end=0.6),
    )
    segments = segmenter.segment(
        stream=stream,
        language="zh",
        speaker_boundaries=(0.4,),
    )
    assert len(segments) == 2
    assert [seg.text for seg in segments] == ["我来", "了"]


def test_blank_valley_boundary() -> None:
    segmenter = SentenceSegmenter(config=SentenceSegmenterConfig(long_pause_gap_sec=5.0))
    stream = (
        _item("A", start=0.0, end=0.2),
        _item("B", start=0.2, end=0.4),
        _item("C", start=0.4, end=0.6),
    )
    segments = segmenter.segment(
        stream=stream,
        language="en",
        blank_valley_boundaries=(0.4,),
    )
    assert len(segments) == 2
    assert [seg.text for seg in segments] == ["AB", "C"]


def test_protected_span_not_split() -> None:
    segmenter = SentenceSegmenter(config=SentenceSegmenterConfig(long_pause_gap_sec=5.0))
    stream = (
        _item("v1.", start=0.0, end=0.2),
        _item("2", start=0.2, end=0.35),
        _item("发布", start=0.35, end=0.7),
    )
    segments = segmenter.segment(stream=stream, language="en")
    assert len(segments) == 1
    assert segments[0].text == "v1.2发布"
