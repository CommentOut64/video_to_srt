"""
文本结构保护规则层。

设计目标：
1. 让“小数点/缩写点/连字符”等保护能力通过规则扩展，而不是散落在业务代码里。
2. 业务层只调用统一入口，不关心具体保护细节。
3. 新增保护时，只需要新增规则并注册到规则集。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Tuple

from app.models.sensevoice_models import WordTimestamp

if TYPE_CHECKING:
    from app.services.timeanchored_alignment.contracts import ProtectedSpan

_DEFAULT_SENTENCE_END_CHARS = ("。", "！", "？", ".", "!", "?")


class BaseTextProtectionRule:
    """文本保护规则基类（策略模式）。"""

    rule_id: str = "base"

    def can_merge_word_tokens(self, left_token: str, right_token: str) -> bool:
        return False

    def resolve_sentence_end(
        self,
        current_token: str,
        next_token: Optional[str],
        sentence_end_chars: Iterable[str],
    ) -> Optional[bool]:
        return None

    def should_preserve_trailing_punct(self, text: str, index: int, char: str) -> bool:
        return False

    def should_skip_raw_punctuation(self, text: str, index: int, char: str) -> bool:
        return False

    def extract_protected_spans(self, text: str) -> Tuple[ProtectedSpan, ...]:
        return tuple()


def _build_protected_span(*, start: int, end: int, kind: str, text: str = "") -> "ProtectedSpan":
    from app.services.timeanchored_alignment.contracts import ProtectedSpan

    return ProtectedSpan(
        start=int(start),
        end=int(end),
        kind=str(kind or ""),
        text=str(text or ""),
    )


class DecimalProtectionRule(BaseTextProtectionRule):
    """
    小数结构保护规则。

    覆盖能力：
    - 词级边界识别（如 0 + . + 15）
    - 句末强标点判定拦截（避免把小数点当句号）
    - 尾部标点清理保护（避免 0. 被误删）
    - 原文标点提取时跳过小数点
    """

    rule_id = "decimal_protection"
    _DECIMAL_DOT_CHARS = {".", "。", "．"}
    _DECIMAL_PATTERN = re.compile(r"(?<!\d)\d+[\.\u3002\uFF0E]\d+(?!\d)")

    @classmethod
    def is_decimal_dot_in_text(cls, text: str, index: int) -> bool:
        if index <= 0 or index >= len(text) - 1:
            return False
        if text[index] not in cls._DECIMAL_DOT_CHARS:
            return False
        return text[index - 1].isdigit() and text[index + 1].isdigit()

    @classmethod
    def _normalize_token(cls, token: str) -> str:
        return str(token or "").replace("▁", " ").strip()

    @classmethod
    def _first_nonspace_char(cls, token: str) -> Optional[str]:
        for ch in token:
            if not ch.isspace():
                return ch
        return None

    @classmethod
    def _last_nonspace_index(cls, token: str) -> int:
        idx = len(token) - 1
        while idx >= 0 and token[idx].isspace():
            idx -= 1
        return idx

    @classmethod
    def _has_digit_before_last_dot(cls, token: str) -> bool:
        idx = cls._last_nonspace_index(token)
        if idx < 0 or token[idx] not in cls._DECIMAL_DOT_CHARS:
            return False
        prev = idx - 1
        while prev >= 0 and token[prev].isspace():
            prev -= 1
        return prev >= 0 and token[prev].isdigit()

    def can_merge_word_tokens(self, left_token: str, right_token: str) -> bool:
        left = self._normalize_token(left_token)
        right = self._normalize_token(right_token)
        if not left or not right:
            return False
        left_idx = self._last_nonspace_index(left)
        if left_idx < 0:
            return False
        left_last = left[left_idx]
        right_first = self._first_nonspace_char(right)
        if right_first is None:
            return False

        # 形态 A: "0" + "." / "0" + "。"
        if left_last.isdigit() and right_first in self._DECIMAL_DOT_CHARS:
            return True
        # 形态 B: "0." + "15"
        if (
            left_last in self._DECIMAL_DOT_CHARS
            and self._has_digit_before_last_dot(left)
            and right_first.isdigit()
        ):
            return True
        return False

    def resolve_sentence_end(
        self,
        current_token: str,
        next_token: Optional[str],
        sentence_end_chars: Iterable[str],
    ) -> Optional[bool]:
        current = self._normalize_token(current_token)
        if not current:
            return None
        end_chars = set(sentence_end_chars)
        idx = self._last_nonspace_index(current)
        if idx < 0:
            return None
        tail = current[idx]
        if tail not in end_chars:
            return None
        if tail not in self._DECIMAL_DOT_CHARS:
            return None
        if not self._has_digit_before_last_dot(current):
            return None
        if not next_token:
            return None
        # 只有在“右侧是数字链”时，才阻断句末边界。
        right_first = self._first_nonspace_char(self._normalize_token(next_token))
        if right_first is None:
            return None
        if right_first.isdigit() or right_first in self._DECIMAL_DOT_CHARS:
            return False
        return None

    def should_preserve_trailing_punct(self, text: str, index: int, char: str) -> bool:
        if char not in self._DECIMAL_DOT_CHARS:
            return False
        if index <= 0:
            return False
        prev = index - 1
        while prev >= 0 and text[prev].isspace():
            prev -= 1
        return prev >= 0 and text[prev].isdigit()

    def should_skip_raw_punctuation(self, text: str, index: int, char: str) -> bool:
        if char not in self._DECIMAL_DOT_CHARS:
            return False
        return self.is_decimal_dot_in_text(text, index)

    def extract_protected_spans(self, text: str) -> Tuple[ProtectedSpan, ...]:
        spans: list[ProtectedSpan] = []
        for matched in self._DECIMAL_PATTERN.finditer(text):
            spans.append(_build_protected_span(
                start=matched.start(),
                end=matched.end(),
                kind="decimal",
                text=matched.group(0),
            ))
        return tuple(spans)


class TimeExpressionProtectionRule(BaseTextProtectionRule):
    """时间表达保护规则（如 7:28 PM / 07:28 p.m. / 7:28:09PM）。"""

    rule_id = "time_expression_protection"
    _TIME_PUNCT_CHARS = {":", "：", "."}
    _TIME_CORE_PATTERN = re.compile(r"^\d{1,2}(?:[:：][0-5]\d){1,2}$")
    _MERIDIEM_PATTERN = re.compile(r"^(?:[AaPp][Mm]|[AaPp]\.[Mm]\.)$")
    _TIME_PATTERN = re.compile(
        r"\b\d{1,2}(?:[:：][0-5]\d){1,2}(?:\s*(?:[AaPp][Mm]|[AaPp]\.[Mm]\.))?(?=$|[\s,，;；:：!?.])"
    )

    @classmethod
    def _normalize_token(cls, token: str) -> str:
        return str(token or "").replace("▁", " ").strip()

    @classmethod
    def _trim_boundary_weak_punct(cls, token: str) -> str:
        return cls._normalize_token(token).strip(",，;；")

    def can_merge_word_tokens(self, left_token: str, right_token: str) -> bool:
        left = self._trim_boundary_weak_punct(left_token)
        right = self._trim_boundary_weak_punct(right_token)
        if not left or not right:
            return False
        return bool(
            self._TIME_CORE_PATTERN.fullmatch(left)
            and self._MERIDIEM_PATTERN.fullmatch(right.replace(" ", ""))
        )

    @classmethod
    def is_time_punctuation_in_text(cls, text: str, index: int) -> bool:
        if index < 0 or index >= len(text):
            return False
        if text[index] not in cls._TIME_PUNCT_CHARS:
            return False
        for matched in cls._TIME_PATTERN.finditer(str(text or "")):
            if matched.start() <= index < matched.end():
                return True
        return False

    def should_skip_raw_punctuation(self, text: str, index: int, char: str) -> bool:
        if char not in self._TIME_PUNCT_CHARS:
            return False
        return self.is_time_punctuation_in_text(text, index)

    def extract_protected_spans(self, text: str) -> Tuple[ProtectedSpan, ...]:
        spans: list[ProtectedSpan] = []
        for matched in self._TIME_PATTERN.finditer(text):
            spans.append(_build_protected_span(
                start=matched.start(),
                end=matched.end(),
                kind="time_expr",
                text=matched.group(0),
            ))
        return tuple(spans)


class RegexSpanProtectionRule(BaseTextProtectionRule):
    """基于正则的保护 span 规则。"""

    span_kind: str = "regex"
    pattern: re.Pattern[str]

    def extract_protected_spans(self, text: str) -> Tuple[ProtectedSpan, ...]:
        spans: list[ProtectedSpan] = []
        for matched in self.pattern.finditer(text):
            spans.append(_build_protected_span(
                start=matched.start(),
                end=matched.end(),
                kind=self.span_kind,
                text=matched.group(0),
            ))
        return tuple(spans)


class VersionProtectionRule(RegexSpanProtectionRule):
    span_kind = "version"
    pattern = re.compile(r"\bv\d+(?:\.\d+)+\b", re.IGNORECASE)


class AbbrevDotProtectionRule(RegexSpanProtectionRule):
    span_kind = "abbrev_dot"
    pattern = re.compile(r"\b(?:[A-Za-z]\.){2,}[A-Za-z]?\.?")


class HyphenProtectionRule(RegexSpanProtectionRule):
    span_kind = "hyphen"
    pattern = re.compile(r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b")


class ApostropheProtectionRule(RegexSpanProtectionRule):
    span_kind = "apostrophe"
    pattern = re.compile(r"\b[A-Za-z]+(?:'[A-Za-z]+)+\b")


class MiddleDotProtectionRule(RegexSpanProtectionRule):
    span_kind = "middle_dot"
    pattern = re.compile(r"・")


class AlnumMixedProtectionRule(BaseTextProtectionRule):
    """字母数字混合片段保护（如 RTX 4090 / OpenAI GPT-4o）。"""

    _PATTERNS = (
        re.compile(r"\b[A-Za-z]{2,}\s+\d+[A-Za-z0-9-]*\b"),
        re.compile(r"\b[A-Za-z]{2,}\s+[A-Za-z]+-[A-Za-z0-9]+\b"),
    )

    def extract_protected_spans(self, text: str) -> Tuple[ProtectedSpan, ...]:
        spans: list[ProtectedSpan] = []
        for pattern in self._PATTERNS:
            for matched in pattern.finditer(text):
                spans.append(_build_protected_span(
                    start=matched.start(),
                    end=matched.end(),
                    kind="alnum_mixed",
                    text=matched.group(0),
                ))
        return tuple(spans)


@dataclass(frozen=True)
class TextProtectionRuleSet:
    """规则集合（组合模式）。"""

    rules: Tuple[BaseTextProtectionRule, ...]

    def can_merge_word_tokens(self, left_token: str, right_token: str) -> bool:
        return any(rule.can_merge_word_tokens(left_token, right_token) for rule in self.rules)

    def resolve_sentence_end(
        self,
        current_token: str,
        next_token: Optional[str],
        sentence_end_chars: Iterable[str],
        default_value: bool,
    ) -> bool:
        for rule in self.rules:
            decision = rule.resolve_sentence_end(current_token, next_token, sentence_end_chars)
            if decision is not None:
                return bool(decision)
        return default_value

    def should_preserve_trailing_punct(self, text: str, index: int, char: str) -> bool:
        return any(rule.should_preserve_trailing_punct(text, index, char) for rule in self.rules)

    def should_skip_raw_punctuation(self, text: str, index: int, char: str) -> bool:
        return any(rule.should_skip_raw_punctuation(text, index, char) for rule in self.rules)

    def extract_protected_spans(self, text: str) -> Tuple[ProtectedSpan, ...]:
        if not text:
            return tuple()
        collected: list[ProtectedSpan] = []
        for rule in self.rules:
            collected.extend(rule.extract_protected_spans(text))
        if not collected:
            return tuple()
        collected.sort(
            key=lambda item: (
                item.start,
                -self._span_priority(item.kind),
                -(item.end - item.start),
            )
        )

        accepted: list[ProtectedSpan] = []
        for item in collected:
            if accepted and item.start < accepted[-1].end:
                previous = accepted[-1]
                if (
                    self._span_priority(item.kind) > self._span_priority(previous.kind)
                    and (item.end - item.start) >= (previous.end - previous.start)
                ):
                    accepted[-1] = item
                continue
            accepted.append(item)
        return tuple(accepted)

    @staticmethod
    def _span_priority(kind: str) -> int:
        priority_map = {
            "time_expr": 100,
            "decimal": 90,
            "version": 80,
            "abbrev_dot": 70,
            "hyphen": 60,
            "apostrophe": 50,
            "middle_dot": 40,
            "alnum_mixed": 10,
        }
        return int(priority_map.get(str(kind or ""), 0))


def build_default_rule_set() -> TextProtectionRuleSet:
    """默认规则集入口；后续新增规则只需在这里注册。"""
    return TextProtectionRuleSet(
        rules=(
            DecimalProtectionRule(),
            TimeExpressionProtectionRule(),
            VersionProtectionRule(),
            AbbrevDotProtectionRule(),
            HyphenProtectionRule(),
            ApostropheProtectionRule(),
            MiddleDotProtectionRule(),
            AlnumMixedProtectionRule(),
        )
    )


_DEFAULT_RULE_SET = build_default_rule_set()


def can_merge_word_tokens(
    left_token: str,
    right_token: str,
    *,
    rule_set: Optional[TextProtectionRuleSet] = None,
) -> bool:
    active = rule_set or _DEFAULT_RULE_SET
    return active.can_merge_word_tokens(left_token, right_token)


def is_sentence_end_punct(
    current_token: str,
    next_token: Optional[str] = None,
    *,
    sentence_end_chars: Sequence[str] = _DEFAULT_SENTENCE_END_CHARS,
    rule_set: Optional[TextProtectionRuleSet] = None,
) -> bool:
    token = str(current_token or "").replace("▁", " ").strip()
    if not token:
        return False
    idx = len(token) - 1
    while idx >= 0 and token[idx].isspace():
        idx -= 1
    if idx < 0:
        return False
    default_value = token[idx] in set(sentence_end_chars)
    active = rule_set or _DEFAULT_RULE_SET
    return active.resolve_sentence_end(
        current_token=token,
        next_token=next_token,
        sentence_end_chars=sentence_end_chars,
        default_value=default_value,
    )


def should_preserve_trailing_punct(
    text: str,
    index: int,
    *,
    rule_set: Optional[TextProtectionRuleSet] = None,
) -> bool:
    if index < 0 or index >= len(text):
        return False
    active = rule_set or _DEFAULT_RULE_SET
    return active.should_preserve_trailing_punct(text, index, text[index])


def should_skip_raw_punctuation(
    text: str,
    index: int,
    char: str,
    *,
    rule_set: Optional[TextProtectionRuleSet] = None,
) -> bool:
    active = rule_set or _DEFAULT_RULE_SET
    return active.should_skip_raw_punctuation(text, index, char)


def is_decimal_dot_in_text(text: str, index: int) -> bool:
    """兼容入口：供外层按需直接查询小数点判定。"""
    return DecimalProtectionRule.is_decimal_dot_in_text(text, index)


def is_time_punctuation_in_text(text: str, index: int) -> bool:
    """兼容入口：供外层按需直接查询时间表达内标点判定。"""
    return TimeExpressionProtectionRule.is_time_punctuation_in_text(text, index)


def extract_protected_spans(
    text: str,
    *,
    rule_set: Optional[TextProtectionRuleSet] = None,
) -> list[ProtectedSpan]:
    active = rule_set or _DEFAULT_RULE_SET
    return list(active.extract_protected_spans(text))


def merge_protected_word_tokens(
    words: Sequence[WordTimestamp],
    *,
    rule_set: Optional[TextProtectionRuleSet] = None,
) -> list[WordTimestamp]:
    """
    将受保护结构合并为不可拆词级 token。

    当前默认规则会把 `0 + . + 15` 合并为 `0.15`，并保留时间戳包络：
    - start: 首 token.start
    - end: 末 token.end
    """
    active = rule_set or _DEFAULT_RULE_SET
    if not words:
        return []
    merged: list[WordTimestamp] = [_clone_word(words[0])]
    for item in words[1:]:
        current = _clone_word(item)
        previous = merged[-1]
        if active.can_merge_word_tokens(previous.word, current.word):
            merged[-1] = _merge_word_pair(previous, current)
            continue
        merged.append(current)
    return merged


def _clone_word(word: WordTimestamp) -> WordTimestamp:
    cloned = WordTimestamp(
        word=str(word.word or ""),
        start=float(word.start or 0.0),
        end=float(word.end or 0.0),
        confidence=word.confidence,
        confidence_raw=word.confidence_raw,
        confidence_display_raw=word.confidence_display_raw,
        confidence_source=word.confidence_source,
        token_type=word.token_type,
        is_pseudo=bool(word.is_pseudo),
        warning_type=word.warning_type,
        perplexity=word.perplexity,
    )
    for attr in ("speaker_id", "turn_id", "speaker_label", "speaker_color_key"):
        if hasattr(word, attr):
            setattr(cloned, attr, getattr(word, attr))
    return cloned


def _merge_word_pair(left: WordTimestamp, right: WordTimestamp) -> WordTimestamp:
    left_duration = max(float(left.end or 0.0) - float(left.start or 0.0), 0.0)
    right_duration = max(float(right.end or 0.0) - float(right.start or 0.0), 0.0)
    left_weight = left_duration if left_duration > 0 else max(len(str(left.word or "").strip()), 1)
    right_weight = right_duration if right_duration > 0 else max(len(str(right.word or "").strip()), 1)

    left_warning_value = getattr(left.warning_type, "value", str(left.warning_type))
    merged = WordTimestamp(
        word=f"{left.word}{right.word}",
        start=min(float(left.start or 0.0), float(right.start or 0.0)),
        end=max(float(left.end or 0.0), float(right.end or 0.0)),
        confidence=_weighted_optional(left.confidence, right.confidence, left_weight, right_weight),
        confidence_raw=_weighted_optional(left.confidence_raw, right.confidence_raw, left_weight, right_weight),
        confidence_display_raw=_weighted_optional(
            left.confidence_display_raw,
            right.confidence_display_raw,
            left_weight,
            right_weight,
        ),
        confidence_source=_merge_confidence_source(left.confidence_source, right.confidence_source),
        token_type=left.token_type if left.token_type == right.token_type else left.token_type or right.token_type,
        is_pseudo=bool(left.is_pseudo) and bool(right.is_pseudo),
        warning_type=left.warning_type if left_warning_value != "none" else right.warning_type,
        perplexity=max(
            [v for v in (left.perplexity, right.perplexity) if v is not None],
            default=None,
        ),
    )
    for attr in ("speaker_id", "turn_id", "speaker_label", "speaker_color_key"):
        left_value = getattr(left, attr, None)
        right_value = getattr(right, attr, None)
        setattr(merged, attr, left_value if left_value is not None else right_value)
    return merged


def _weighted_optional(
    left: Optional[float],
    right: Optional[float],
    left_weight: float,
    right_weight: float,
) -> Optional[float]:
    values: list[tuple[float, float]] = []
    if left is not None:
        values.append((float(left), max(left_weight, 1e-6)))
    if right is not None:
        values.append((float(right), max(right_weight, 1e-6)))
    if not values:
        return None
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return values[-1][0]
    return sum(value * weight for value, weight in values) / total_weight


def _merge_confidence_source(left: Optional[str], right: Optional[str]) -> Optional[str]:
    left_norm = str(left or "").strip().lower()
    right_norm = str(right or "").strip().lower()
    if not left_norm:
        return right
    if not right_norm:
        return left
    if left_norm == right_norm:
        return left
    return "merged"
