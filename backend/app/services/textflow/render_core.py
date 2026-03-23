"""RenderCore（Phase 4）。"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from app.services.text_protection import is_decimal_dot_in_text
from app.services.textflow.contracts import (
    CanonicalTextStream,
    PunctuationFact,
    RenderPolicy,
    RenderedSubtitle,
    RenderResult,
    SegmentationResult,
)


class RenderCore:
    """基于切分计划重建展示文本。"""

    _HALF_TO_FULL_PUNCT = {
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
        "{": "｛",
        "}": "｝",
    }
    _FULL_TO_HALF_PUNCT = {value: key for key, value in _HALF_TO_FULL_PUNCT.items()}
    _FULL_TO_HALF_PUNCT.update(
        {
            "．": ".",
            "、": ",",
            "“": "\"",
            "”": "\"",
            "‘": "'",
            "’": "'",
        }
    )
    _EN_RIGHT_PUNCT = {",", ".", "!", "?", ";", ":", ")", "]", "}"}
    _EN_LEFT_PUNCT = {"(", "[", "{"}

    def render(
        self,
        *,
        canonical_stream: CanonicalTextStream,
        segmentation_result: SegmentationResult,
        policy: Optional[RenderPolicy] = None,
    ) -> RenderResult:
        render_policy = policy or RenderPolicy()
        language = self._resolve_language_tag(
            language=canonical_stream.language,
            text="".join(token.text_core for token in canonical_stream.tokens),
        )
        subtitles: List[RenderedSubtitle] = []
        dropped_punct_facts: List[Dict[str, Any]] = []
        rendered_terminal_count = 0

        facts = list(canonical_stream.punctuation_facts or [])
        tokens = list(canonical_stream.tokens or [])
        token_by_index = {token.index: token for token in tokens}

        for segment in segmentation_result.segments:
            segment_tokens = [
                token_by_index[idx]
                for idx in range(int(segment.token_start), int(segment.token_end) + 1)
                if idx in token_by_index
            ]
            if not segment_tokens:
                continue

            inner_text = self._render_inner_text(
                segment_tokens=segment_tokens,
                facts=facts,
                segment_token_start=int(segment.token_start),
                segment_token_end=int(segment.token_end),
                language=language,
                policy=render_policy,
                dropped_punct_facts=dropped_punct_facts,
            )
            terminal_punct = self._resolve_terminal_punct(
                consumed=segment.consumed_boundary_punct,
                language=language,
                policy=render_policy,
            )
            if terminal_punct:
                rendered_terminal_count += 1
                text_display_raw = f"{inner_text}{terminal_punct}"
            else:
                text_display_raw = inner_text

            text_display = self._standardize_text_by_language(text=text_display_raw, language=language)
            text_core_joined = self._join_token_cores(
                token_texts=[token.text_core for token in segment_tokens],
                language=language,
            )
            subtitles.append(
                RenderedSubtitle(
                    segment_id=segment.segment_id,
                    start=float(segment.start),
                    end=float(segment.end),
                    text_display=text_display,
                    text_core_joined=text_core_joined,
                    terminal_punct=terminal_punct or None,
                    speaker_id=segment_tokens[0].speaker_id,
                    turn_id=segment_tokens[0].turn_id,
                    text_source=canonical_stream.text_source,
                    trace=dict(segment.trace or {}),
                )
            )

        render_report = {
            "segment_count": len(subtitles),
            "input_segment_count": len(segmentation_result.segments),
            "rendered_terminal_count": rendered_terminal_count,
            "dropped_punct_fact_count": len(dropped_punct_facts),
            "dropped_punct_facts": dropped_punct_facts,
        }
        output_trace = tuple(
            {
                "segment_id": item.segment_id,
                "start": item.start,
                "end": item.end,
                "text_display": item.text_display,
                "terminal_punct": item.terminal_punct,
            }
            for item in subtitles
        )
        return RenderResult(
            subtitles=tuple(subtitles),
            render_report=render_report,
            output_trace=output_trace,
        )

    def _render_inner_text(
        self,
        *,
        segment_tokens: Sequence[Any],
        facts: Sequence[PunctuationFact],
        segment_token_start: int,
        segment_token_end: int,
        language: str,
        policy: RenderPolicy,
        dropped_punct_facts: List[Dict[str, Any]],
    ) -> str:
        prefix_map: Dict[int, List[str]] = {}
        suffix_map: Dict[int, List[str]] = {}

        for fact in facts:
            if fact.punct_class == "sentence_end":
                continue
            if not policy.show_inner_punctuation:
                dropped_punct_facts.append({"fact_id": fact.fact_id, "reason": "inner_hidden_by_policy"})
                continue
            if not self._is_fact_in_segment(
                fact=fact,
                segment_token_start=segment_token_start,
                segment_token_end=segment_token_end,
            ):
                continue
            if fact.attach_mode in {"trailing", "between"} and fact.left_token_index is not None:
                suffix_map.setdefault(int(fact.left_token_index), []).append(fact.normalized_text)
            elif fact.attach_mode == "leading" and fact.right_token_index is not None:
                prefix_map.setdefault(int(fact.right_token_index), []).append(fact.normalized_text)
            else:
                dropped_punct_facts.append({"fact_id": fact.fact_id, "reason": "unsupported_attach_mode"})

        parts: List[str] = []
        for idx, token in enumerate(segment_tokens):
            token_idx = int(token.index)
            part = f"{''.join(prefix_map.get(token_idx, []))}{token.text_core}{''.join(suffix_map.get(token_idx, []))}"
            parts.append(part)
            if idx >= len(segment_tokens) - 1:
                continue
            if language == "en":
                parts.append(" ")
        return "".join(parts)

    @staticmethod
    def _is_fact_in_segment(
        *,
        fact: PunctuationFact,
        segment_token_start: int,
        segment_token_end: int,
    ) -> bool:
        anchors = [fact.left_token_index, fact.right_token_index]
        for anchor in anchors:
            if anchor is None:
                continue
            if segment_token_start <= int(anchor) <= segment_token_end:
                return True
        return False

    def _resolve_terminal_punct(
        self,
        *,
        consumed: Any,
        language: str,
        policy: RenderPolicy,
    ) -> Optional[str]:
        if consumed is None:
            return None
        punct = str(getattr(consumed, "normalized_text", "") or "")
        if not punct:
            return None
        if punct in {".", "。"} and not policy.show_terminal_period:
            return None
        if punct not in {".", "。"} and not policy.show_terminal_non_period_punct:
            return None
        return self._standardize_terminal_punct(punct=punct, language=language)

    def _standardize_terminal_punct(self, *, punct: str, language: str) -> str:
        if language == "en":
            return self._to_halfwidth_punctuation(punct)
        if language in {"zh", "ja"}:
            return self._to_fullwidth_punctuation(punct)
        return punct

    @staticmethod
    def _join_token_cores(*, token_texts: Sequence[str], language: str) -> str:
        if language == "en":
            return " ".join(token_texts)
        return "".join(token_texts)

    def _standardize_text_by_language(self, *, text: str, language: str) -> str:
        if not text:
            return text
        if language == "en":
            return self._standardize_english_punctuation(text)
        if language in {"zh", "ja"}:
            return self._standardize_cjk_punctuation(text)
        return text

    @staticmethod
    def _resolve_language_tag(*, language: str, text: str) -> str:
        tag = str(language or "auto").lower()
        if tag.startswith("en"):
            return "en"
        if tag.startswith("zh") or tag.startswith("yue"):
            return "zh"
        if tag.startswith("ja") or tag.startswith("jp"):
            return "ja"

        has_cjk = bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff]", text))
        has_latin = bool(re.search(r"[A-Za-z]", text))
        if has_cjk and not has_latin:
            return "zh"
        if has_latin and not has_cjk:
            return "en"
        return "auto"

    def _standardize_english_punctuation(self, text: str) -> str:
        normalized = self._to_halfwidth_punctuation(text)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"\s+([,.;:!?\)\]\}])", r"\1", normalized)
        normalized = re.sub(r"([\(\[\{])\s+", r"\1", normalized)

        chars = list(normalized)
        out: list[str] = []
        length = len(chars)
        for index, char in enumerate(chars):
            out.append(char)
            if index >= length - 1:
                continue
            next_char = chars[index + 1]
            if next_char.isspace():
                continue
            if char in {",", ";", ":", "!", "?"} and next_char not in self._EN_RIGHT_PUNCT:
                out.append(" ")
            elif (
                char == "."
                and self._should_insert_space_after_dot(
                    prev_char=chars[index - 1] if index > 0 else "",
                    next_char=next_char,
                )
            ):
                out.append(" ")

        normalized = "".join(out)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        normalized = re.sub(r"\s+([,.;:!?\)\]\}])", r"\1", normalized)
        normalized = re.sub(r"([\(\[\{])\s+", r"\1", normalized)
        return normalized

    @staticmethod
    def _should_insert_space_after_dot(*, prev_char: str, next_char: str) -> bool:
        if not prev_char or not next_char:
            return False
        if prev_char.isdigit() and next_char.isdigit():
            return False
        if prev_char.isalpha() and next_char.isalpha():
            if prev_char.isupper() and next_char.isupper():
                return False
            return prev_char.islower() and next_char.isupper()
        return next_char.isalnum()

    def _standardize_cjk_punctuation(self, text: str) -> str:
        normalized = self._to_fullwidth_punctuation(text)
        normalized = re.sub(r"\s*([，。！？；：、）】》」』｝])\s*", r"\1", normalized)
        normalized = re.sub(r"\s*([（【《「『｛])\s*", r"\1", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _to_halfwidth_punctuation(self, text: str) -> str:
        normalized = text
        for source, target in self._FULL_TO_HALF_PUNCT.items():
            normalized = normalized.replace(source, target)
        return normalized

    def _to_fullwidth_punctuation(self, text: str) -> str:
        chars: list[str] = []
        for index, char in enumerate(text):
            if char == "." and is_decimal_dot_in_text(text, index):
                chars.append(char)
                continue
            chars.append(self._HALF_TO_FULL_PUNCT.get(char, char))
        return "".join(chars)

