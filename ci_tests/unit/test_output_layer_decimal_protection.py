"""输出层小数点保护测试。"""

from __future__ import annotations

from app.models.sensevoice_models import SentenceSegment
from app.services.alignment.types import OutputLayerInput
from app.services.textflow.output_layer import OutputLayerProcessor


class _DummySubtitleManager:
    def replace_chunk(self, chunk_index: int, sentences: list[SentenceSegment]) -> list[int]:
        return list(range(len(sentences)))


def _build_processor() -> OutputLayerProcessor:
    return OutputLayerProcessor(subtitle_manager=_DummySubtitleManager())


def test_output_layer_cjk_standardization_preserves_decimal_dot() -> None:
    processor = _build_processor()
    sentence = SentenceSegment(
        text="低一瓶中的氰化钠总含量为1.4到1.6克.",
        text_clean="低一瓶中的氰化钠总含量为1.4到1.6克.",
        is_draft=False,
    )
    result = processor.process(
        OutputLayerInput(chunk_index=1, sentence_segments=[sentence], language="zh")
    )

    payload_sentence = result.output_payload["sentence_segments"][0]
    assert payload_sentence["text"] == "低一瓶中的氰化钠总含量为1.4到1.6克。"
    assert payload_sentence["text_clean"] == "低一瓶中的氰化钠总含量为1.4到1.6克。"


def test_output_layer_cjk_standardization_keeps_non_decimal_conversion() -> None:
    processor = _build_processor()
    sentence = SentenceSegment(
        text="这是测试.",
        text_clean="这是测试.",
        is_draft=False,
    )
    result = processor.process(
        OutputLayerInput(chunk_index=2, sentence_segments=[sentence], language="zh")
    )

    payload_sentence = result.output_payload["sentence_segments"][0]
    assert payload_sentence["text"] == "这是测试。"
    assert payload_sentence["text_clean"] == "这是测试。"
