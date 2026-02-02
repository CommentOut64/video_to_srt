"""
统一规范化层（WeText ITN + 安全标点标记 + 清洗）。
V3.2.0+dev.20260202.03
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set

from app.services.alignment.number_utils import normalize_numbers
from app.services.alignment.types import CharMapping, NormalizationResult


_PUNCTUATION_SET = set(",.!?;:\"()[]{}，。！？；：、（）【】《》“”‘’「」『』")
_DECIMAL_DOT_CHARS = {".", "。", "．"}
_HYPHEN_CHARS = {"-", "‐", "‑", "–", "—"}
_JAPANESE_MIDDLE_DOT = "・"
_EN_ABBREV_PATTERN = re.compile(r"(?:\b[A-Za-z]\.){2,}[A-Za-z]\.?")


class TextNormalizer:
    """统一规范化器（单例模式：复用 WeText 实例，避免重复初始化）。"""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._itn_cache: Dict[str, object] = {}

    def normalize(self, text: str, lang: str) -> NormalizationResult:
        """统一规范化：ITN → 安全标点标记 → 清洗，输出分轨文本。"""
        if not text:
            return NormalizationResult(
                text_itn_raw="",
                text_clean="",
                char_mapping=[],
                raw_to_clean=[],
                clean_to_raw=[],
            )

        text_itn_raw = self._apply_itn(text, lang)
        text_clean, mapping, raw_to_clean, clean_to_raw = self._clean_punctuation(
            text_itn_raw,
            lang,
        )
        return NormalizationResult(
            text_itn_raw=text_itn_raw,
            text_clean=text_clean,
            char_mapping=mapping,
            raw_to_clean=raw_to_clean,
            clean_to_raw=clean_to_raw,
        )

    def _apply_itn(self, text: str, lang: str) -> str:
        if not text:
            return ""
        try:
            itn = self._get_itn(lang)
            return itn.normalize(text)
        except Exception as exc:
            self._logger.warning("WeText ITN 失败，回退 number_utils: %s", exc)
            return normalize_numbers(text, lang=lang)

    def _get_itn(self, lang: str):
        if lang not in self._itn_cache:
            try:
                from wetext import Normalizer
            except ImportError as exc:
                raise ImportError("缺少 wetext 依赖，请先安装 wetext>=0.1.2") from exc
            wetext_lang = lang if lang in ("zh", "en", "ja") else "auto"
            self._itn_cache[lang] = Normalizer(
                lang=wetext_lang,
                operator="itn",
                enable_0_to_9=True,
            )
        return self._itn_cache[lang]

    def _clean_punctuation(
        self,
        text: str,
        lang: str,
    ) -> tuple[str, List[CharMapping], List[Optional[int]], List[int]]:
        safe_marks = _mark_safe_punct(text, lang)
        clean_chars: List[str] = []
        mapping: List[CharMapping] = []
        raw_to_clean: List[Optional[int]] = []
        clean_to_raw: List[int] = []
        for idx, ch in enumerate(text):
            if _is_punct(ch) and idx not in safe_marks:
                mapping.append(CharMapping(raw_idx=idx, clean_idx=None, punct=ch))
                raw_to_clean.append(None)
                continue
            clean_chars.append(ch)
            clean_idx = len(clean_chars) - 1
            mapping.append(CharMapping(raw_idx=idx, clean_idx=clean_idx, punct=None))
            raw_to_clean.append(clean_idx)
            clean_to_raw.append(idx)
        return "".join(clean_chars), mapping, raw_to_clean, clean_to_raw


def _is_punct(ch: str) -> bool:
    return ch in _PUNCTUATION_SET


def _mark_safe_punct(text: str, lang: str) -> Set[int]:
    safe: Set[int] = set()
    for match in _EN_ABBREV_PATTERN.finditer(text):
        for idx in range(match.start(), match.end()):
            if text[idx] == ".":
                safe.add(idx)
    for idx, ch in enumerate(text):
        if ch in _DECIMAL_DOT_CHARS and _is_decimal_dot(text, idx):
            safe.add(idx)
            continue
        if ch == "." and _is_english_abbrev_dot(text, idx):
            safe.add(idx)
            continue
        if ch in _HYPHEN_CHARS and _is_english_hyphen(text, idx):
            safe.add(idx)
            continue
        if ch == _JAPANESE_MIDDLE_DOT and _is_japanese_middle_dot(lang):
            safe.add(idx)
    return safe


def _is_decimal_dot(text: str, index: int) -> bool:
    if index <= 0 or index >= len(text) - 1:
        return False
    if text[index] not in _DECIMAL_DOT_CHARS:
        return False
    return text[index - 1].isdigit() and text[index + 1].isdigit()


def _is_english_abbrev_dot(text: str, index: int) -> bool:
    if index <= 0 or index >= len(text) - 1:
        return False
    left = text[index - 1]
    right = text[index + 1]
    return left.isalpha() and right.isalpha()


def _is_english_hyphen(text: str, index: int) -> bool:
    if index <= 0 or index >= len(text) - 1:
        return False
    left = text[index - 1]
    right = text[index + 1]
    return left.isalpha() and right.isalpha()


def _is_japanese_middle_dot(lang: str) -> bool:
    lang_key = (lang or "").lower()
    return lang_key.startswith("ja")


_normalizer_instance: Optional[TextNormalizer] = None


def get_alignment_text_normalizer(logger: Optional[logging.Logger] = None) -> TextNormalizer:
    """获取统一规范化器单例（单例模式：减少 WeText 初始化开销）。"""
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = TextNormalizer(logger=logger)
    return _normalizer_instance
