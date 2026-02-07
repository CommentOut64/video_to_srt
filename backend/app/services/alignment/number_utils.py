"""
数字规范化兜底工具（用于 WeText 不可用时的最小转换）。
V3.2.0+dev.20260202.03
"""
from __future__ import annotations

import re
from typing import Dict, Optional


_CHINESE_DIGITS: Dict[str, int] = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

_CHINESE_UNITS: Dict[str, int] = {
    "十": 10,
    "百": 100,
    "千": 1000,
    "万": 10_000,
    "亿": 100_000_000,
}

_CHINESE_NUM_PATTERN = re.compile(r"[零〇一二两三四五六七八九十百千万亿]+")


def normalize_numbers(text: str, lang: Optional[str] = None) -> str:
    """将可识别的中文数字转换为阿拉伯数字（兜底）。"""
    if not text:
        return ""

    lang_key = (lang or "auto").lower()
    if lang_key not in {"zh", "yue", "ja", "auto"} and not _CHINESE_NUM_PATTERN.search(text):
        return text

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        converted = _convert_chinese_sequence(token)
        return converted if converted is not None else token

    return _CHINESE_NUM_PATTERN.sub(_replace, text)


def _convert_chinese_sequence(token: str) -> Optional[str]:
    if not token:
        return None
    has_unit = any(ch in _CHINESE_UNITS for ch in token)
    if not has_unit:
        if len(token) <= 1:
            return None
        digits = []
        for ch in token:
            if ch not in _CHINESE_DIGITS:
                return None
            digits.append(str(_CHINESE_DIGITS[ch]))
        return "".join(digits)
    value = _chinese_to_arabic(token)
    return str(value) if value is not None else None


def _chinese_to_arabic(token: str) -> Optional[int]:
    total = 0
    section = 0
    number = 0

    for ch in token:
        if ch in _CHINESE_DIGITS:
            number = _CHINESE_DIGITS[ch]
            continue
        unit = _CHINESE_UNITS.get(ch)
        if unit is None:
            return None
        if unit >= 10_000:
            section = (section + (number or 0)) * unit
            total += section
            section = 0
            number = 0
            continue
        if number == 0:
            number = 1
        section += number * unit
        number = 0

    return total + section + number

