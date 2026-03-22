"""
语言策略配置与词表加载器。
V3.2.0+dev.20260221.07
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional

import yaml

from app.core.config import config
from app.core.logging import resolve_loguru_logger

_LOGGER = resolve_loguru_logger(
    None,
    __name__,
    layer="语言适配层",
    processor_name="language_policy_loader",
)

_POLICY_CACHE: Optional[Dict[str, Any]] = None
_LEXICON_CACHE: Dict[str, FrozenSet[str]] = {}


def _default_policy_config() -> Dict[str, Any]:
    return {
        "policy_version": "V3.2.0+dev.20260221.07",
        "default_language": "zh",
        "language_aliases": {
            "auto": "zh",
            "mixed": "zh",
            "zh-cn": "zh",
            "zh-hans": "zh",
            "zh-hant": "zh",
            "en-us": "en",
            "en-gb": "en",
            "ja-jp": "ja",
        },
        "languages": {
            "zh": {
                "sentence_end_chars": "。？！.!?",
                "lexicon": {
                    "continuation_words": "continuation_words.txt",
                    "incomplete_endings": "incomplete_endings.txt",
                    "semantic_anchor_words": "semantic_anchor_words.txt",
                },
                "thresholds": {
                    "final_min_mapping_coverage": 0.60,
                    "soft_cut_min_score": 0.35,
                    "soft_cut_min_score_pause": 0.34,
                    "soft_cut_min_score_punctuation": 0.40,
                    "soft_cut_min_score_semantic": 0.35,
                    "soft_cut_min_score_llm": 0.35,
                    "soft_cut_min_score_fast_draft": 0.20,
                    "soft_cut_pause_anchor_trigger_sec": 0.30,
                    "soft_cut_pause_anchor_min_gap_sec": 0.70,
                    "soft_cut_semantic_anchor_confidence": 0.70,
                    "soft_cut_weak_punct_anchor_confidence": 0.60,
                    "text_direct_ratio_min": 0.65,
                    "text_estimated_ratio_max": 0.35,
                    "ctc_low_prob_ratio_max": 0.30,
                    "max_continuous_failure_span": 6.0,
                    "dominant_language_min_ratio": 0.55,
                    "foreign_run_max_ratio": 0.50,
                    "foreign_run_char_max": 24.0,
                    "pronunciation_evidence_min_score": 0.45,
                },
                "carry_rules": {
                    "short_tail": [0.45, 2.20],
                    "singleton_tail": [0.65, 0.95],
                },
                "cross_chunk": {
                    "english_carry_words": ["a", "an", "the"],
                    "cjk_carry_words": ["经", "直", "才", "因", "检", "警", "因为", "以及", "直到", "经过", "检测"],
                    "cjk_max_gap_sec": 0.10,
                    "cjk_max_duration_sec": 0.80,
                },
            },
            "en": {
                "sentence_end_chars": ".!?",
                "lexicon": {
                    "continuation_words": "continuation_words.txt",
                    "incomplete_endings": "incomplete_endings.txt",
                    "semantic_anchor_words": "semantic_anchor_words.txt",
                },
                "thresholds": {
                    "final_min_mapping_coverage": 0.60,
                    "soft_cut_min_score": 0.35,
                    "soft_cut_min_score_pause": 0.35,
                    "soft_cut_min_score_punctuation": 0.35,
                    "soft_cut_min_score_semantic": 0.35,
                    "soft_cut_min_score_llm": 0.35,
                    "soft_cut_min_score_fast_draft": 0.35,
                    "soft_cut_pause_anchor_trigger_sec": 0.40,
                    "soft_cut_pause_anchor_min_gap_sec": 1.00,
                    "soft_cut_semantic_anchor_confidence": 0.30,
                    "soft_cut_weak_punct_anchor_confidence": 0.00,
                    "text_direct_ratio_min": 0.65,
                    "text_estimated_ratio_max": 0.35,
                    "ctc_low_prob_ratio_max": 0.30,
                    "max_continuous_failure_span": 6.0,
                    "dominant_language_min_ratio": 0.55,
                    "foreign_run_max_ratio": 0.50,
                    "foreign_run_char_max": 24.0,
                    "pronunciation_evidence_min_score": 0.45,
                },
                "carry_rules": {
                    "short_tail": [0.45, 2.20],
                    "singleton_tail": [0.60, 0.95],
                },
                "cross_chunk": {
                    "english_carry_words": ["a", "an", "the"],
                    "cjk_carry_words": [],
                    "cjk_max_gap_sec": 0.10,
                    "cjk_max_duration_sec": 0.80,
                },
            },
            "ja": {
                "sentence_end_chars": "。？！.!?",
                "lexicon": {
                    "continuation_words": "continuation_words.txt",
                    "incomplete_endings": "incomplete_endings.txt",
                    "semantic_anchor_words": "semantic_anchor_words.txt",
                },
                "thresholds": {
                    "final_min_mapping_coverage": 0.60,
                    "soft_cut_min_score": 0.35,
                    "soft_cut_min_score_pause": 0.34,
                    "soft_cut_min_score_punctuation": 0.40,
                    "soft_cut_min_score_semantic": 0.35,
                    "soft_cut_min_score_llm": 0.35,
                    "soft_cut_min_score_fast_draft": 0.30,
                    "soft_cut_pause_anchor_trigger_sec": 0.30,
                    "soft_cut_pause_anchor_min_gap_sec": 0.70,
                    "soft_cut_semantic_anchor_confidence": 0.70,
                    "soft_cut_weak_punct_anchor_confidence": 0.60,
                    "text_direct_ratio_min": 0.65,
                    "text_estimated_ratio_max": 0.35,
                    "ctc_low_prob_ratio_max": 0.30,
                    "max_continuous_failure_span": 6.0,
                    "dominant_language_min_ratio": 0.55,
                    "foreign_run_max_ratio": 0.50,
                    "foreign_run_char_max": 24.0,
                    "pronunciation_evidence_min_score": 0.45,
                },
                "carry_rules": {
                    "short_tail": [0.45, 2.20],
                    "singleton_tail": [0.60, 0.95],
                },
                "cross_chunk": {
                    "english_carry_words": ["a", "an", "the"],
                    "cjk_carry_words": [],
                    "cjk_max_gap_sec": 0.10,
                    "cjk_max_duration_sec": 0.80,
                },
            },
        },
    }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _policy_file_path() -> Path:
    return (
        Path(config.BASE_DIR)
        / "backend"
        / "app"
        / "config"
        / "language_policy"
        / "policy.yaml"
    )


def _lexicon_base_dir() -> Path:
    return (
        Path(config.BASE_DIR)
        / "backend"
        / "app"
        / "config"
        / "language_policy"
        / "lexicons"
    )


def load_language_policy_config() -> Dict[str, Any]:
    """加载语言策略配置（带默认值回退）。"""
    path = _policy_file_path()
    defaults = _default_policy_config()
    if not path.exists():
        _LOGGER.warning("未找到语言策略配置文件，使用默认配置: {}", path)
        return defaults
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            _LOGGER.warning("语言策略配置格式异常，使用默认配置: {}", path)
            return defaults
        return _deep_merge(defaults, payload)
    except Exception:
        _LOGGER.exception("加载语言策略配置失败，回退默认配置: {}", path)
        return defaults


def get_language_policy_config() -> Dict[str, Any]:
    global _POLICY_CACHE
    if _POLICY_CACHE is None:
        _POLICY_CACHE = load_language_policy_config()
    return _POLICY_CACHE


def _normalize_lexicon_item(text: str) -> str:
    return str(text or "").strip().lower()


def load_language_lexicon(
    *,
    language_tag: str,
    filename: str,
) -> FrozenSet[str]:
    """
    读取指定语言词表（UTF-8，一行一词，支持 # 注释）。
    """
    normalized_tag = str(language_tag or "").strip().lower()
    normalized_filename = str(filename or "").strip()
    cache_key = f"{normalized_tag}:{normalized_filename}"
    if cache_key in _LEXICON_CACHE:
        return _LEXICON_CACHE[cache_key]

    file_path = _lexicon_base_dir() / normalized_tag / normalized_filename
    if not file_path.exists():
        _LOGGER.warning("词表文件不存在，使用空词表: {}", file_path)
        _LEXICON_CACHE[cache_key] = frozenset()
        return _LEXICON_CACHE[cache_key]

    items = set()
    try:
        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            normalized = _normalize_lexicon_item(line)
            if normalized:
                items.add(normalized)
    except Exception:
        _LOGGER.exception("词表读取失败，使用空词表: {}", file_path)
        items = set()

    _LEXICON_CACHE[cache_key] = frozenset(sorted(items))
    return _LEXICON_CACHE[cache_key]


def reset_language_policy_cache() -> None:
    """重置缓存（测试与热更新使用）。"""
    global _POLICY_CACHE
    _POLICY_CACHE = None
    _LEXICON_CACHE.clear()
