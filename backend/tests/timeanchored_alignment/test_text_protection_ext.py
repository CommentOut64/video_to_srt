from __future__ import annotations

from app.services.text_protection import extract_protected_spans


def test_extract_protected_spans_supports_phase3_rules() -> None:
    text = (
        "3.14 是版本 v3.3.0，地区 U.S.，技术 state-of-the-art，缩写 don't，"
        "中点 A・B，显卡 RTX 4090，模型 OpenAI GPT-4o。"
    )
    spans = extract_protected_spans(text)
    span_map = {text[item.start:item.end]: item.kind for item in spans}

    assert span_map["3.14"] == "decimal"
    assert span_map["v3.3.0"] == "version"
    assert span_map["U.S."] == "abbrev_dot"
    assert span_map["state-of-the-art"] == "hyphen"
    assert span_map["don't"] == "apostrophe"
    assert span_map["・"] == "middle_dot"
    assert span_map["RTX 4090"] == "alnum_mixed"
    assert span_map["OpenAI GPT-4o"] == "alnum_mixed"


def test_extract_protected_spans_output_is_sorted_and_non_overlapping() -> None:
    text = "版本 v3.3.0 和 U.S. 都要保护"
    spans = extract_protected_spans(text)
    assert spans == sorted(spans, key=lambda item: item.start)
    for index in range(1, len(spans)):
        assert spans[index - 1].end <= spans[index].start
