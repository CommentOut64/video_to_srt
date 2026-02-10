"""
同音读音切分器。

V3.2.0+dev.20260210.02
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Literal

try:
    from pypinyin import Style, pinyin
except ImportError:  # pragma: no cover - 依赖可选时的降级路径
    Style = None  # type: ignore[assignment]
    pinyin = None  # type: ignore[assignment]


Language = Literal["zh", "ja", "en"]


@dataclass(frozen=True)
class TokenReading:
    token_text: str
    reading_key: str
    reading_key_fuzzy: str
    reading_key_no_punct: str
    reading_key_fuzzy_no_punct: str
    char_start: int
    char_end: int


_WORD_RE = re.compile(r"[A-Za-z']+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ALNUM_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fffぁ-んァ-ン]")


def _strip_tone(reading_key: str) -> str:
    return re.sub(r"[1-5]$", "", reading_key)


def _is_effective_symbol(text: str) -> bool:
    return bool(_ALNUM_RE.search(text))


class HomophoneTokenizer:
    """轻量级分词与读音编码器。"""

    def tokenize(self, text: str, language: Language) -> List[TokenReading]:
        if language == "zh":
            return self._tokenize_zh(text)
        if language == "ja":
            return self._tokenize_ja(text)
        return self._tokenize_en(text)

    def build_query_key(self, query_text: str, language: Language, is_fuzzy: bool) -> str:
        query = query_text.strip()
        if not query:
            return ""
        if language == "zh":
            if re.fullmatch(r"[a-z]+[1-5]?", query.lower()):
                return _strip_tone(query.lower()) if is_fuzzy else query.lower()
            if pinyin is not None and Style is not None:
                py_items = pinyin(query, style=Style.TONE3, heteronym=False)
                joined = "|".join(item[0] for item in py_items if item)
            else:
                joined = query
            return _strip_tone(joined) if is_fuzzy else joined
        if language == "ja":
            normalized = query.lower().replace(" ", "")
            return normalized
        normalized_en = query.lower()
        return normalized_en

    def _tokenize_zh(self, text: str) -> List[TokenReading]:
        records: List[TokenReading] = []
        cursor = 0
        for char in text:
            if _CJK_RE.search(char):
                if pinyin is not None and Style is not None:
                    py_items = pinyin(char, style=Style.TONE3, heteronym=False)
                    strict = py_items[0][0] if py_items and py_items[0] else char
                else:
                    strict = char
            else:
                strict = char
            fuzzy = _strip_tone(strict)
            key_no_punct = strict if _is_effective_symbol(char) else ""
            fuzzy_no_punct = fuzzy if _is_effective_symbol(char) else ""
            records.append(
                TokenReading(
                    token_text=char,
                    reading_key=strict,
                    reading_key_fuzzy=fuzzy,
                    reading_key_no_punct=key_no_punct,
                    reading_key_fuzzy_no_punct=fuzzy_no_punct,
                    char_start=cursor,
                    char_end=cursor + len(char),
                )
            )
            cursor += len(char)
        return records

    def _tokenize_ja(self, text: str) -> List[TokenReading]:
        records: List[TokenReading] = []
        cursor = 0
        for char in text:
            # Phase 1: 日语按字符兜底；后续可替换 Sudachi 词级实现。
            strict = char
            fuzzy = char
            key_no_punct = strict if _is_effective_symbol(char) else ""
            fuzzy_no_punct = fuzzy if _is_effective_symbol(char) else ""
            records.append(
                TokenReading(
                    token_text=char,
                    reading_key=strict,
                    reading_key_fuzzy=fuzzy,
                    reading_key_no_punct=key_no_punct,
                    reading_key_fuzzy_no_punct=fuzzy_no_punct,
                    char_start=cursor,
                    char_end=cursor + len(char),
                )
            )
            cursor += len(char)
        return records

    def _tokenize_en(self, text: str) -> List[TokenReading]:
        records: List[TokenReading] = []
        for match in _WORD_RE.finditer(text):
            token = match.group(0)
            strict = token.lower()
            fuzzy = strict
            records.append(
                TokenReading(
                    token_text=token,
                    reading_key=strict,
                    reading_key_fuzzy=fuzzy,
                    reading_key_no_punct=strict,
                    reading_key_fuzzy_no_punct=fuzzy,
                    char_start=match.start(),
                    char_end=match.end(),
                )
            )
        return records
