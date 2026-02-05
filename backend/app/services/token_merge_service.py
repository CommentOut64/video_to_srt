"""
Token 合并服务（快流词级/字级合并统一入口）。
V3.2.0+dev.20260131.03
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple


_PUNCTUATION_SET = set(",.!?;:\"()[]{}，。！？；：、（）【】《》“”‘’「」『』")
_LEFT_PUNCT = set("([{“‘（【《「『")
_SPECIAL_TAG_PATTERN = re.compile(r"<\|.*?\|>")

# V3.2.0+dev.20260131.03: SenseVoice 显示置信度校准参数（仅影响 display 口径）
_SV_DISPLAY_RAW_MIN = 0.10
_SV_DISPLAY_RAW_MAX = 0.60


@dataclass
class TokenMergeResult:
    """Token 合并输出结构。"""

    raw_tokens: List[Dict[str, Any]]
    words: List[Dict[str, Any]]


class TokenMergeMode:
    """合并模式枚举（字符串常量）。"""

    TOKEN_BOUNDARY = "token_boundary"
    BY_SCRIPT = "by_script"


def merge_tokens(
    tokens: List[Dict[str, Any]],
    *,
    language: Optional[str] = None,
) -> TokenMergeResult:
    """
    将 CTC token 合并为词级时间戳。

    设计原则：
    - 保留原始 token 列表（raw_tokens）用于调试与回退
    - 统一在合并阶段处理标点 token 的时间戳归并
    - 对无空格语言（中/日/韩等）默认按脚本切分，避免整段合并
    """
    normalized = [_normalize_token(token) for token in tokens if token]
    raw_tokens = [token.copy() for token in normalized]
    cleaned = _merge_punctuation_tokens(normalized)
    mode = _resolve_merge_mode(cleaned, language)
    words = _merge_by_mode(cleaned, mode)
    return TokenMergeResult(raw_tokens=raw_tokens, words=words)


def _normalize_token(token: Dict[str, Any]) -> Dict[str, Any]:
    word = str(token.get("word", "") or "")
    # 兜底清理标签与 SentencePiece 符号，避免污染后续合并
    word = _SPECIAL_TAG_PATTERN.sub("", word)
    word = word.replace("▁", " ")
    start = _safe_float(token.get("start", 0.0))
    end = _safe_float(token.get("end", 0.0))
    confidence = _safe_float(token.get("confidence", 1.0), default=1.0)
    is_pseudo = bool(token.get("is_pseudo", False))
    return {
        **token,
        "word": word,
        "start": start,
        "end": end,
        "confidence": confidence,
        "is_pseudo": is_pseudo,
    }


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def _calibrate_sensevoice_display_confidence(raw_conf: float) -> float:
    """
    SenseVoice 显示口径校准（不影响内部决策置信度）。

    说明：
    - raw_conf 来自 CTC softmax，分布偏高（0.9+ 常见）
    - 校准后用于 confidence_display_raw，供前端显示与高亮
    """
    raw = _clip(_safe_float(raw_conf, default=0.0), 0.0, 1.0)
    return _SV_DISPLAY_RAW_MIN + (_SV_DISPLAY_RAW_MAX - _SV_DISPLAY_RAW_MIN) * raw


def _resolve_merge_mode(tokens: List[Dict[str, Any]], language: Optional[str]) -> str:
    if _has_boundary_marker(tokens):
        return TokenMergeMode.TOKEN_BOUNDARY
    if language and language.lower() in {"zh", "yue", "ja", "ko"}:
        return TokenMergeMode.BY_SCRIPT
    return TokenMergeMode.BY_SCRIPT


def _has_boundary_marker(tokens: List[Dict[str, Any]]) -> bool:
    for token in tokens:
        text = token.get("word", "")
        if text.startswith(" ") or text.startswith("▁"):
            return True
    return False


def _merge_punctuation_tokens(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not tokens:
        return []
    cleaned: List[Dict[str, Any]] = []
    for idx, token in enumerate(tokens):
        text = token.get("word", "") or ""
        stripped = text.strip().lstrip("▁")
        if stripped in _PUNCTUATION_SET:
            merge_to_next = stripped in _LEFT_PUNCT
            if merge_to_next:
                _merge_punct_to_next(tokens, idx)
            else:
                _merge_punct_to_prev(cleaned, token) or _merge_punct_to_next(tokens, idx)
            continue
        if stripped == "" and text.strip() == "":
            # 纯空白 token，保留边界信息但不输出
            continue
        cleaned.append(token)
    return cleaned


def _merge_punct_to_prev(
    cleaned: List[Dict[str, Any]],
    punct: Dict[str, Any],
) -> bool:
    if not cleaned:
        return False
    prev = cleaned[-1]
    prev_end = prev.get("end", 0.0)
    prev["end"] = max(prev_end, _safe_float(punct.get("end", prev_end)))
    return True


def _merge_punct_to_next(tokens: List[Dict[str, Any]], index: int) -> bool:
    for next_token in tokens[index + 1:]:
        text = next_token.get("word", "") or ""
        stripped = text.strip().lstrip("▁")
        if stripped in _PUNCTUATION_SET or stripped == "":
            continue
        next_start = next_token.get("start", 0.0)
        next_token["start"] = min(next_start, _safe_float(tokens[index].get("start", next_start)))
        return True
    return False


def _merge_by_mode(tokens: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if not tokens:
        return []
    words: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    current_conf_min = 1.0
    current_conf_sum = 0.0
    current_conf_weight = 0.0
    current_script = None
    current_is_pseudo = False

    def flush_current() -> None:
        nonlocal current, current_conf_min, current_conf_sum, current_conf_weight, current_script, current_is_pseudo
        if not current:
            return
        display_conf = (
            current_conf_sum / current_conf_weight
            if current_conf_weight > 0.0
            else current_conf_min
        )
        current["confidence"] = current_conf_min
        current["confidence_raw"] = current_conf_min
        current["confidence_display_raw"] = _clip(display_conf, 0.0, 1.0)
        current["is_pseudo"] = current_is_pseudo
        current["token_type"] = "word"
        words.append(current)
        current = None
        current_conf_min = 1.0
        current_conf_sum = 0.0
        current_conf_weight = 0.0
        current_script = None
        current_is_pseudo = False

    for token in tokens:
        raw_text = token.get("word", "") or ""
        boundary, text = _strip_boundary(raw_text)
        if not text:
            if boundary:
                flush_current()
            continue
        token_script = _detect_script_group(text)
        start_new = current is None or boundary
        if not start_new and mode == TokenMergeMode.BY_SCRIPT:
            if token_script in {"han", "kana", "hangul"}:
                start_new = True
            elif current_script != token_script:
                start_new = True
        if start_new:
            flush_current()
            current = {
                "word": text,
                "start": _safe_float(token.get("start", 0.0)),
                "end": _safe_float(token.get("end", 0.0)),
            }
            current_script = token_script
            current_is_pseudo = bool(token.get("is_pseudo", False))
            token_conf = _safe_float(token.get("confidence", 1.0), default=1.0)
            token_display = token.get("confidence_display_raw")
            if token_display is None:
                token_display = _calibrate_sensevoice_display_confidence(token_conf)
            token_display = _safe_float(token_display, default=token_conf)
            current_conf_min = min(current_conf_min, token_conf)
            weight = _confidence_weight(token)
            current_conf_sum += token_display * weight
            current_conf_weight += weight
            continue

        current["word"] += text
        current["end"] = max(current.get("end", 0.0), _safe_float(token.get("end", 0.0)))
        current_is_pseudo = current_is_pseudo or bool(token.get("is_pseudo", False))
        token_conf = _safe_float(token.get("confidence", 1.0), default=1.0)
        token_display = token.get("confidence_display_raw")
        if token_display is None:
            token_display = _calibrate_sensevoice_display_confidence(token_conf)
        token_display = _safe_float(token_display, default=token_conf)
        current_conf_min = min(current_conf_min, token_conf)
        weight = _confidence_weight(token)
        current_conf_sum += token_display * weight
        current_conf_weight += weight

    flush_current()
    return words


def _strip_boundary(text: str) -> Tuple[bool, str]:
    if not text:
        return False, ""
    boundary = text.startswith(" ") or text.startswith("▁")
    cleaned = text.lstrip(" ▁")
    return boundary, cleaned


def _detect_script_group(text: str) -> str:
    for char in text:
        if char.isspace():
            continue
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF:
            return "han"
        if 0x3040 <= code <= 0x30FF:
            return "kana"
        if 0xAC00 <= code <= 0xD7AF:
            return "hangul"
        if char.isdigit():
            return "digit"
        if char.isalpha():
            return "latin"
        return "other"
    return "other"


def _confidence_weight(token: Dict[str, Any]) -> float:
    start = _safe_float(token.get("start", 0.0))
    end = _safe_float(token.get("end", start))
    duration = max(end - start, 0.0)
    return duration if duration > 0.0 else 1.0
