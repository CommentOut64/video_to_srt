from __future__ import annotations

"""p20260323 回归门禁。

历史 job 目录里的 checkpoint/SRT 产物来自旧链路，适合作为问题来源样本，不再适合作为当前主链真源。
这里改为使用同一批样本文本构造“活回归”，直接守住当前切分层与输出层的关键行为。
"""

from app.services.alignment.types import CharMapping, TextTrack, TextTrackBundle
from app.services.punctuation.base import PuncPosition
from app.services.streaming_subtitle import StreamingSubtitleManager
from app.services.textflow.canonical_text_stream_adapter import CanonicalTextStreamAdapter
from app.services.textflow.contracts import (
    CanonicalTextStream,
    ConsumedBoundaryPunct,
    CoreToken,
    PunctuationFact,
    SegmentPlan,
    SegmentationResult,
    SubtitleBatch,
    SubtitleItem,
)
from app.services.textflow.render_core import RenderCore


def _track(
    *,
    text: str,
    source: str,
    punct_positions: list[PuncPosition],
) -> TextTrack:
    char_mapping = [CharMapping(raw_idx=index, clean_idx=index) for index in range(len(text))]
    clean_to_word: list[int | None] = []
    word_index = 0
    for char in text:
        if char in "，。！？；：,.!?;: ":
            clean_to_word.append(None)
            continue
        clean_to_word.append(word_index)
        word_index += 1
    return TextTrack(
        raw_text=text,
        text_itn_raw=text,
        text_clean=text,
        char_mapping=char_mapping,
        raw_to_clean=list(range(len(text))),
        clean_to_raw=list(range(len(text))),
        clean_to_word=clean_to_word,
        language="zh",
        source=source,
        punct_positions=punct_positions,
        mapping_coverage=1.0 if text else 0.0,
    )


def _segment(
    *,
    segment_id: str,
    token_start: int,
    token_end: int,
    start: float,
    end: float,
    consumed_boundary_punct: ConsumedBoundaryPunct | None = None,
) -> SegmentPlan:
    return SegmentPlan(
        segment_id=segment_id,
        token_start=token_start,
        token_end=token_end,
        start=start,
        end=end,
        boundary_reason="punctuation",
        boundary_score=0.9,
        consumed_boundary_punct=consumed_boundary_punct,
    )


def _render(stream: CanonicalTextStream, *segments: SegmentPlan) -> list[str]:
    result = RenderCore().render(
        canonical_stream=stream,
        segmentation_result=SegmentationResult(segments=tuple(segments)),
    )
    return [item.text_display for item in result.subtitles]


def test_job_p20260323_chunk_8_replace_chunk_batch_does_not_accumulate_repeated_sentences() -> None:
    manager = StreamingSubtitleManager("job-p20260323-repeat")
    batch = SubtitleBatch(
        chunk_id="chunk-8",
        items=(
            SubtitleItem(
                segment_id="chunk-8-seg-0",
                chunk_id="chunk-8",
                start=62.879,
                end=65.339,
                text="给了六个同事里面年龄最小的一个说我不喜欢喝可乐送给你喝吧",
            ),
            SubtitleItem(
                segment_id="chunk-8-seg-1",
                chunk_id="chunk-8",
                start=65.339,
                end=67.739,
                text="没想到这个女孩的好心举动",
            ),
        ),
    )

    manager.replace_chunk_batch(batch)
    manager.replace_chunk_batch(batch)
    manager.replace_chunk_batch(batch)

    snapshot = manager.to_checkpoint_data()
    items = [item for item in snapshot["subtitle_items_snapshot"] if item["chunk_id"] == "chunk-8"]

    assert len(manager.chunk_sentences[8]) == 2
    assert len(items) == 2
    assert [item["text"] for item in items] == [
        "给了六个同事里面年龄最小的一个说我不喜欢喝可乐送给你喝吧",
        "没想到这个女孩的好心举动",
    ]


def test_job_p20260323_entry_77_render_path_deduplicates_double_weak_punct() -> None:
    text = "没想到回来以后，警察已经来了，幸运的他"
    first_comma = text.index("，")
    second_comma = text.index("，", first_comma + 1)
    adapter = CanonicalTextStreamAdapter()
    stream = adapter.build(
        stream_id="p20260323",
        chunk_ref="chunk-34",
        tracks=TextTrackBundle(
            sv_track=_track(
                text=text,
                source="sensevoice",
                punct_positions=[
                    PuncPosition(char_index=first_comma, punctuation="，", confidence=0.95),
                    PuncPosition(char_index=second_comma, punctuation="，", confidence=0.95),
                ],
            ),
            whisper_track=_track(
                text=text,
                source="whisper",
                punct_positions=[
                    PuncPosition(char_index=first_comma, punctuation="，", confidence=0.92),
                    PuncPosition(char_index=second_comma, punctuation="，", confidence=0.92),
                ],
            ),
            chosen_track=_track(text=text, source="sensevoice", punct_positions=[]),
        ),
        text_source="fast",
        language="zh",
    )

    texts = _render(
        stream,
        _segment(
            segment_id="chunk-34-seg-0",
            token_start=0,
            token_end=len(stream.tokens) - 1,
            start=256.921,
            end=259.801,
        ),
    )

    assert texts == [text]
    assert "，，" not in texts[0]


def test_job_p20260323_entry_79_render_path_preserves_decimal_points() -> None:
    stream = CanonicalTextStream(
        stream_id="p20260323",
        chunk_ref="chunk-35",
        language="zh",
        text_source="slow",
        tokens=(
            CoreToken("t0", 0, "氰化钠的致死剂量为", "氰化钠的致死剂量为", 261.492, 263.100, source="slow"),
            CoreToken("t1", 1, "0.15", "0.15", 263.100, 263.700, source="slow"),
            CoreToken("t2", 2, "到", "到", 263.700, 263.900, source="slow"),
            CoreToken("t3", 3, "0.2", "0.2", 263.900, 264.300, source="slow"),
            CoreToken("t4", 4, "克", "克", 264.300, 264.500, source="slow"),
            CoreToken("t5", 5, "警方一共发现了四瓶毒可乐", "警方一共发现了四瓶毒可乐", 264.500, 266.000, source="slow"),
            CoreToken("t6", 6, "分别对他们进行了化验", "分别对他们进行了化验", 266.000, 267.600, source="slow"),
            CoreToken("t7", 7, "结果显示", "结果显示", 267.600, 268.392, source="slow"),
        ),
        punctuation_facts=(
            PunctuationFact(
                fact_id="pf-comma-0",
                left_token_index=4,
                right_token_index=5,
                attach_mode="between",
                raw_text="，",
                normalized_text="，",
                punct_class="weak",
                source="slow",
            ),
            PunctuationFact(
                fact_id="pf-comma-1",
                left_token_index=5,
                right_token_index=6,
                attach_mode="between",
                raw_text="，",
                normalized_text="，",
                punct_class="weak",
                source="slow",
            ),
            PunctuationFact(
                fact_id="pf-comma-2",
                left_token_index=6,
                right_token_index=7,
                attach_mode="between",
                raw_text="，",
                normalized_text="，",
                punct_class="weak",
                source="slow",
            ),
        ),
    )

    text = _render(
        stream,
        _segment(
            segment_id="chunk-35-seg-0",
            token_start=0,
            token_end=7,
            start=261.492,
            end=268.392,
        ),
    )[0]

    assert "0.15" in text
    assert "0.2" in text
    assert "，，" not in text


def test_job_p20260323_entry_81_render_path_keeps_third_bottle_as_next_sentence_start() -> None:
    stream = CanonicalTextStream(
        stream_id="p20260323",
        chunk_ref="chunk-37-38",
        language="zh",
        text_source="slow",
        tokens=(
            CoreToken("t0", 0, "第二瓶导致中年男子死亡的可乐中的氰化钠总含量为", "第二瓶导致中年男子死亡的可乐中的氰化钠总含量为", 274.027, 277.800, source="slow"),
            CoreToken("t1", 1, "1.9", "1.9", 277.800, 278.100, source="slow"),
            CoreToken("t2", 2, "克", "克", 278.100, 278.300, source="slow"),
            CoreToken("t3", 3, "第三瓶在路灯下发现的可乐中的氰化钠总含量为", "第三瓶在路灯下发现的可乐中的氰化钠总含量为", 278.300, 282.800, source="slow"),
            CoreToken("t4", 4, "3", "3", 282.800, 283.000, source="slow"),
            CoreToken("t5", 5, "克", "克", 283.000, 283.200, source="slow"),
        ),
        punctuation_facts=(
            PunctuationFact(
                fact_id="pf-period-third-bottle",
                left_token_index=2,
                right_token_index=3,
                attach_mode="between",
                raw_text="。",
                normalized_text="。",
                punct_class="sentence_end",
                source="slow",
            ),
        ),
    )

    texts = _render(
        stream,
        _segment(
            segment_id="chunk-37-seg-0",
            token_start=0,
            token_end=2,
            start=274.027,
            end=278.300,
            consumed_boundary_punct=ConsumedBoundaryPunct(
                fact_id="pf-period-third-bottle",
                raw_text="。",
                normalized_text="。",
                punct_class="sentence_end",
                source="slow",
                render_hint="drop_period_default",
            ),
        ),
        _segment(
            segment_id="chunk-38-seg-0",
            token_start=3,
            token_end=5,
            start=278.300,
            end=283.200,
        ),
    )

    assert "第三瓶" not in texts[0]
    assert texts[0].endswith("1.9克")
    assert texts[1].startswith("第三瓶在路灯下发现的可乐中的氰化钠总含量为3克")
