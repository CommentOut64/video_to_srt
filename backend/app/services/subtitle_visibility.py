"""
字幕可见性过滤工具。

V3.2.4+dev.20260222.01
"""
from __future__ import annotations

from typing import Any, Iterable, List, TypeVar

T = TypeVar("T")

_KNOWN_CONFIDENCE_SOURCES = {
    "fast",
    "slow",
    "merged",
    "manual",
    "sensevoice",
    "whisper",
    "whisper_patch",
    "llm_correction",
    "llm_translation",
    "imported",
}


def normalize_confidence_source(value: Any) -> str:
    """规范化置信度来源字符串。"""
    return str(value or "").strip().lower()


def is_unknown_confidence_source(value: Any) -> bool:
    """判断来源是否为 unknown。"""
    return normalize_confidence_source(value) == "unknown"


def is_hidden_unknown_sentence(sentence: Any) -> bool:
    """
    判断字幕是否应被 unknown 可见性规则过滤。

    规则：
    1. 句级 confidence_source=unknown 直接过滤；
    2. 句级来源缺失时，若词级来源全部为 unknown/空，也过滤。
    """
    confidence_source = _read_field(sentence, "confidence_source")
    normalized_source = normalize_confidence_source(confidence_source)
    if normalized_source == "unknown":
        return True
    if normalized_source in _KNOWN_CONFIDENCE_SOURCES:
        return False

    words = _read_field(sentence, "words")
    if not isinstance(words, list) or not words:
        return False

    has_known_source = False
    has_unknown_source = False
    for word in words:
        word_source = normalize_confidence_source(_read_field(word, "confidence_source"))
        if not word_source or word_source == "unknown":
            has_unknown_source = True
            continue
        has_known_source = True
        break
    return has_unknown_source and not has_known_source


def filter_hidden_unknown_sentences(sentences: Iterable[T]) -> List[T]:
    """过滤掉来源为 unknown 的字幕。"""
    return [sentence for sentence in sentences if not is_hidden_unknown_sentence(sentence)]


def _read_field(target: Any, key: str) -> Any:
    if isinstance(target, dict):
        return target.get(key)
    return getattr(target, key, None)


__all__ = [
    "filter_hidden_unknown_sentences",
    "is_hidden_unknown_sentence",
    "is_unknown_confidence_source",
    "normalize_confidence_source",
]
