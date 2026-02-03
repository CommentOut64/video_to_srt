"""
统一规范化层（WeText ITN + 安全标点标记 + 清洗）。
V3.2.0+dev.20260203.04
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
_ABNORMAL_ALNUM_PATTERN = re.compile(r"(?:[A-Za-z]{3,}\d{2,}|\d{2,}[A-Za-z]{3,})")
_CJK_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_SPACED_DIGIT_PATTERN = re.compile(r"(?<=\d)\s+(?=\d)")
_CJK_SINGLE_DIGIT_PATTERN = re.compile(r"(?<=[\u4e00-\u9fff])([0-9])(?=[\u4e00-\u9fff])")
_CJK_DIGIT_MAP = {
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}


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

        text_itn_raw, itn_fallback, itn_reason = self._apply_itn_safely(text, lang)
        text_itn_raw = self._post_itn_cleanup(text_itn_raw, lang)
        if not itn_fallback:
            quality_ok, quality_reason = self._check_itn_quality(text, text_itn_raw)
            if not quality_ok:
                self._logger.warning("ITN 质量不达标，回退 number_utils: %s", quality_reason)
                text_itn_raw = normalize_numbers(text, lang=lang)
                text_itn_raw = self._post_itn_cleanup(text_itn_raw, lang)
                itn_fallback = True
                itn_reason = quality_reason
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
            itn_fallback=itn_fallback,
            itn_fallback_reason=itn_reason,
        )

    def _apply_itn_safely(self, text: str, lang: str) -> tuple[str, bool, Optional[str]]:
        if not text:
            return "", False, None
        try:
            return self._apply_itn(text, lang), False, None
        except Exception as exc:
            self._logger.warning("WeText ITN 失败，回退 number_utils: %s", exc)
            return normalize_numbers(text, lang=lang), True, "itn_exception"

    def _apply_itn(self, text: str, lang: str) -> str:
        if not text:
            return ""
        itn = self._get_itn(lang)
        return itn.normalize(text)

    def _get_itn(self, lang: str):
        """
        获取或创建 ITN 归一化器

        V3.2.0+dev.20260203.07: 修复单个数字转换问题
        - 设置 enable_0_to_9=False（官方推荐）
        - 保持口语化表达："想一想" 不会变成 "想1想"
        - 连续数字仍然转换："六六六" → "666"
        """
        if lang not in self._itn_cache:
            try:
                from wetext import Normalizer
            except ImportError as exc:
                raise ImportError("缺少 wetext 依赖，请先安装 wetext>=0.1.2") from exc
            wetext_lang = lang if lang in ("zh", "en", "ja") else "auto"
            self._itn_cache[lang] = Normalizer(
                lang=wetext_lang,
                operator="itn",
                enable_0_to_9=False,  # 官方推荐：保持单个数字口语化
            )
        return self._itn_cache[lang]

    def _post_itn_cleanup(self, text: str, lang: str) -> str:
        """ITN 后清理（数字空格修正 + 单字数字回转 + 英文修正）。"""
        if not text:
            return text

        normalized = text
        if _has_cjk(normalized):
            normalized = _collapse_spaced_digits(normalized)
            normalized = _convert_single_digit_in_cjk(normalized)

        lang_key = (lang or "").lower()
        if not lang_key.startswith("en"):
            return normalized

        normalized = re.sub(
            r"\b([APap])\s*\.?\s*m\.?m?\.?\b",
            lambda m: f"{m.group(1).upper()}M",
            normalized,
        )
        normalized = re.sub(r"([A-Za-z]{2,})(\d)", r"\1 \2", normalized)

        def _split_digit_word(match: re.Match) -> str:
            suffix = match.group(2)
            if suffix.lower() in {"st", "nd", "rd", "th"}:
                return match.group(0)
            return f"{match.group(1)} {suffix}"

        normalized = re.sub(r"(\d)([A-Za-z]{2,})", _split_digit_word, normalized)
        normalized = re.sub(r"\s{2,}", " ", normalized)
        return normalized.strip()

    @staticmethod
    def _check_itn_quality(original: str, normalized: str) -> tuple[bool, Optional[str]]:
        """检查 ITN 质量，异常返回 False。"""
        if not normalized:
            return False, "empty_itn"
        if not original:
            return True, None
        origin_len = len(original)
        norm_len = len(normalized)
        ratio = norm_len / max(origin_len, 1)
        if origin_len >= 8 and (ratio < 0.3 or ratio > 3.0):
            return False, f"length_ratio_outlier:{ratio:.2f}"
        abnormal = _ABNORMAL_ALNUM_PATTERN.findall(normalized)
        if len(abnormal) >= 2:
            return False, "abnormal_alnum_glue"
        return True, None

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


def _has_cjk(text: str) -> bool:
    return bool(_CJK_CHAR_PATTERN.search(text))


def _collapse_spaced_digits(text: str) -> str:
    if not text:
        return text
    return _SPACED_DIGIT_PATTERN.sub("", text)


def _convert_single_digit_in_cjk(text: str) -> str:
    if not text:
        return text

    def _replace(match: re.Match) -> str:
        return _CJK_DIGIT_MAP.get(match.group(1), match.group(1))

    return _CJK_SINGLE_DIGIT_PATTERN.sub(_replace, text)


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
