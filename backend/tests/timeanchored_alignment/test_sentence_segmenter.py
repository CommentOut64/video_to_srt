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
    assert [seg.text for seg in segments] == ["A B", "C"]


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


def test_english_spacing_preserved_for_timeanchored_sentence_text() -> None:
    segmenter = SentenceSegmenter(config=SentenceSegmenterConfig(long_pause_gap_sec=5.0))
    stream = (
        _item("It's", start=0.0, end=0.2),
        _item("still", start=0.2, end=0.4),
        _item("only", start=0.4, end=0.6),
        _item("7", start=0.6, end=0.7),
        _item(":28", start=0.7, end=0.8),
        _item("PM,", start=0.8, end=0.9),
        _item("there's", start=0.9, end=1.1),
        _item("plenty", start=1.1, end=1.3),
        _item("of", start=1.3, end=1.4),
        _item("time", start=1.4, end=1.6),
        _item("left", start=1.6, end=1.8),
        _item("to", start=1.8, end=1.9),
        _item("go", start=1.9, end=2.0),
        _item("and", start=2.0, end=2.2),
        _item("hunt", start=2.2, end=2.4),
        _item("evil", start=2.4, end=2.6),
        _item("down.", start=2.6, end=2.8),
    )
    segments = segmenter.segment(stream=stream, language="en")
    assert len(segments) == 1
    assert segments[0].text == "It's still only 7:28 PM, there's plenty of time left to go and hunt evil down."


def test_japanese_keeps_no_space_between_tokens() -> None:
    segmenter = SentenceSegmenter(config=SentenceSegmenterConfig(long_pause_gap_sec=5.0))
    stream = (
        _item("明日", start=0.0, end=0.2),
        _item("も", start=0.2, end=0.3),
        _item("行く", start=0.3, end=0.5),
        _item("。", start=0.5, end=0.6),
    )
    segments = segmenter.segment(stream=stream, language="ja")
    assert len(segments) == 1
    assert segments[0].text == "明日も行く。"


def test_japanese_region_tag_keeps_no_space_between_tokens() -> None:
    segmenter = SentenceSegmenter(config=SentenceSegmenterConfig(long_pause_gap_sec=5.0))
    stream = (
        _item("明日", start=0.0, end=0.2),
        _item("も", start=0.2, end=0.3),
        _item("行く", start=0.3, end=0.5),
        _item("。", start=0.5, end=0.6),
    )
    segments = segmenter.segment(stream=stream, language="ja-JP")
    assert len(segments) == 1
    assert segments[0].text == "明日も行く。"


def test_english_region_tag_keeps_spacing_rules() -> None:
    segmenter = SentenceSegmenter(config=SentenceSegmenterConfig(long_pause_gap_sec=5.0))
    stream = (
        _item("A", start=0.0, end=0.2),
        _item("B", start=0.2, end=0.4),
        _item(",", start=0.4, end=0.5),
        _item("C", start=0.5, end=0.7),
    )
    segments = segmenter.segment(stream=stream, language="en-US")
    assert len(segments) == 1
    assert segments[0].text == "A B, C"
