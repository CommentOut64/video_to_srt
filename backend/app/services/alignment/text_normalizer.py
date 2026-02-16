"""
统一规范化层（WeText ITN + 安全标点标记 + 清洗）。
V3.2.0+dev.20260204.02
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.logging import resolve_loguru_logger
from app.services.alignment.number_utils import normalize_numbers
from app.services.alignment.types import CharMapping, NormalizationResult
from app.services.text_normalizer import TextNormalizer as MinimalTextNormalizer
from app.services.text_pipeline_config import NormalizationConfig, TextPipelineConfig


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
_DECIMAL_DOT_PLACEHOLDER = "__DECIMAL_DOT__"
_DECIMAL_DOT_PATTERN = re.compile(r"(?<=\d)[\.\u3002\uFF0E](?=\d)")
_PUNCT_TO_FULLWIDTH = {
    ",": "，",
    ".": "。",
    "!": "！",
    "?": "？",
    ";": "；",
    ":": "：",
    "(": "（",
    ")": "）",
    "[": "【",
    "]": "】",
}
_PUNCT_TO_HALFWIDTH = {value: key for key, value in _PUNCT_TO_FULLWIDTH.items()}
_PUNCT_TO_HALFWIDTH.update({"．": "."})


class TextNormalizer:
    """统一规范化器（单例模式：复用 WeText 实例，避免重复初始化）。"""

    def __init__(self, logger: Optional[Any] = None) -> None:
        self._logger = resolve_loguru_logger(
            logger,
            __name__,
            layer="规范层",
            processor_name="text_normalizer",
        )
        self._itn_cache: Dict[str, object] = {}

    def normalize(
        self,
        text: str,
        lang: str,
        *,
        config: Optional[NormalizationConfig] = None,
    ) -> NormalizationResult:
        """统一规范化：最小清洗 → ITN → 后清理 → 标点宽度统一 → 安全标点清洗。"""
        if not text:
            return NormalizationResult(
                text_itn_raw="",
                text_clean="",
                char_mapping=[],
                raw_to_clean=[],
                clean_to_raw=[],
                mapping_coverage=0.0,
            )

        active_config = config or self._load_config()
        base_text = MinimalTextNormalizer.clean(text)
        if not base_text:
            # V3.2.0+dev.20260205.08:
            # 最小清洗后为空通常意味着输入仅包含 <|...|> / ▁ 等模型内部标记。
            # 为避免污染在分层链路中“回流”为 text_clean，这里视为无有效文本，交由上层按质量信号降级/跳过。
            self._logger.warning(
                "最小清洗后为空，视为无有效文本以避免标签/▁污染: raw_len={} has_tag={} has_sp={} lang={}",
                len(text),
                1 if "<|" in text else 0,
                1 if "▁" in text else 0,
                lang or "auto",
            )
            return NormalizationResult(
                text_itn_raw="",
                text_clean="",
                char_mapping=[],
                raw_to_clean=[],
                clean_to_raw=[],
                itn_fallback=False,
                itn_fallback_reason="empty_after_min_clean",
                mapping_coverage=0.0,
            )

        text_itn_raw, itn_fallback, itn_reason = self._apply_itn_pipeline(
            base_text,
            lang,
            active_config,
        )
        text_itn_raw = self._normalize_punctuation_width(text_itn_raw, lang, active_config)
        text_clean, mapping, raw_to_clean, clean_to_raw = self._clean_punctuation(
            text_itn_raw,
            lang,
            active_config,
        )
        mapping_coverage = _calc_mapping_coverage(raw_to_clean)
        self._logger.info(
            "规范层完成 input_len={} clean_len={} itn_fallback={} mapping_cov={:.2f} lang={}",
            len(base_text),
            len(text_clean),
            1 if itn_fallback else 0,
            mapping_coverage,
            lang or "auto",
        )
        return NormalizationResult(
            text_itn_raw=text_itn_raw,
            text_clean=text_clean,
            char_mapping=mapping,
            raw_to_clean=raw_to_clean,
            clean_to_raw=clean_to_raw,
            itn_fallback=itn_fallback,
            itn_fallback_reason=itn_reason,
            mapping_coverage=mapping_coverage,
        )

    @staticmethod
    def _load_config() -> NormalizationConfig:
        return TextPipelineConfig.from_runtime().normalization

    def _apply_itn_pipeline(
        self,
        text: str,
        lang: str,
        config: NormalizationConfig,
    ) -> Tuple[str, bool, Optional[str]]:
        if not config.is_itn_enabled:
            return self._post_itn_cleanup(text, lang, config), False, None
        text_itn_raw, itn_fallback, itn_reason = self._apply_itn_safely(text, lang)
        text_itn_raw = self._post_itn_cleanup(text_itn_raw, lang, config)
        if not itn_fallback:
            quality_ok, quality_reason = self._check_itn_quality(text, text_itn_raw, config)
            if not quality_ok:
                self._logger.warning("ITN 质量不达标，回退 number_utils: {}", quality_reason)
                text_itn_raw = normalize_numbers(text, lang=lang)
                text_itn_raw = self._post_itn_cleanup(text_itn_raw, lang, config)
                itn_fallback = True
                itn_reason = quality_reason
        return text_itn_raw, itn_fallback, itn_reason

    def _apply_itn_safely(self, text: str, lang: str) -> tuple[str, bool, Optional[str]]:
        if not text:
            return "", False, None
        try:
            return self._apply_itn(text, lang), False, None
        except Exception as exc:
            self._logger.warning("WeText ITN 失败，回退 number_utils: {}", exc)
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

    def _post_itn_cleanup(self, text: str, lang: str, config: NormalizationConfig) -> str:
        """ITN 后清理（数字空格修正 + 单字数字回转 + 英文修正）。"""
        if not text:
            return text

        normalized = text
        if _has_cjk(normalized):
            if config.is_cjk_digit_merge:
                normalized = _collapse_spaced_digits(normalized)
            if config.is_cjk_single_digit_to_zh:
                normalized = _convert_single_digit_in_cjk(normalized)

        lang_key = (lang or "").lower()
        if lang_key.startswith("en"):
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

        if config.is_collapse_spaces:
            normalized = re.sub(r"\s{2,}", " ", normalized)
        return normalized.strip()

    @staticmethod
    def _check_itn_quality(
        original: str,
        normalized: str,
        config: NormalizationConfig,
    ) -> tuple[bool, Optional[str]]:
        """检查 ITN 质量，异常返回 False。"""
        if not normalized:
            return False, "empty_itn"
        if not original:
            return True, None
        origin_len = len(original)
        norm_len = len(normalized)
        ratio = norm_len / max(origin_len, 1)
        if origin_len >= 8 and (ratio < config.itn_quality_min_ratio or ratio > config.itn_quality_max_ratio):
            return False, f"length_ratio_outlier:{ratio:.2f}"
        abnormal = _ABNORMAL_ALNUM_PATTERN.findall(normalized)
        if len(abnormal) >= 2:
            return False, "abnormal_alnum_glue"
        return True, None

    def _clean_punctuation(
        self,
        text: str,
        lang: str,
        config: NormalizationConfig,
    ) -> tuple[str, List[CharMapping], List[Optional[int]], List[int]]:
        safe_marks = _mark_safe_punct(text, lang, config)
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

    def _normalize_punctuation_width(
        self,
        text: str,
        lang: str,
        config: NormalizationConfig,
    ) -> str:
        """统一全角/半角标点（小数点保护可选）。"""
        if not text:
            return text
        mode = (config.punct_width or "auto").lower()
        if mode == "full":
            is_to_fullwidth = True
        elif mode == "half":
            is_to_fullwidth = False
        else:
            is_to_fullwidth = _should_use_fullwidth(lang, text)

        normalized = text
        if config.is_decimal_protection_enabled:
            normalized = _DECIMAL_DOT_PATTERN.sub(_DECIMAL_DOT_PLACEHOLDER, normalized)
        mapping = _PUNCT_TO_FULLWIDTH if is_to_fullwidth else _PUNCT_TO_HALFWIDTH
        for old, new in mapping.items():
            normalized = normalized.replace(old, new)
        if config.is_decimal_protection_enabled:
            normalized = normalized.replace(_DECIMAL_DOT_PLACEHOLDER, ".")
        return normalized


def _is_punct(ch: str) -> bool:
    return ch in _PUNCTUATION_SET


def _mark_safe_punct(text: str, lang: str, config: NormalizationConfig) -> Set[int]:
    safe: Set[int] = set()
    if not config.is_safe_punct_enabled:
        return safe
    if config.is_abbrev_dot_protection_enabled:
        for match in _EN_ABBREV_PATTERN.finditer(text):
            for idx in range(match.start(), match.end()):
                if text[idx] == ".":
                    safe.add(idx)
    for idx, ch in enumerate(text):
        if config.is_decimal_protection_enabled and ch in _DECIMAL_DOT_CHARS and _is_decimal_dot(text, idx):
            safe.add(idx)
            continue
        if config.is_abbrev_dot_protection_enabled and ch == "." and _is_english_abbrev_dot(text, idx):
            safe.add(idx)
            continue
        if config.is_hyphen_protection_enabled and ch in _HYPHEN_CHARS and _is_english_hyphen(text, idx):
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


def _should_use_fullwidth(lang: str, text: str) -> bool:
    lang_key = (lang or "").lower()
    if lang_key.startswith(("zh", "ja", "jp", "yue", "ko")):
        return True
    if lang_key.startswith("en"):
        return False
    return _has_cjk(text)


def _calc_mapping_coverage(raw_to_clean: List[Optional[int]]) -> float:
    if not raw_to_clean:
        return 0.0
    mapped = sum(1 for idx in raw_to_clean if idx is not None)
    return mapped / max(len(raw_to_clean), 1)


_normalizer_instance: Optional[TextNormalizer] = None


def get_alignment_text_normalizer(logger: Optional[Any] = None) -> TextNormalizer:
    """获取统一规范化器单例（单例模式：减少 WeText 初始化开销）。"""
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = TextNormalizer(logger=logger)
    return _normalizer_instance
