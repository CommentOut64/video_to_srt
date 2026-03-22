"""慢流窗口提示词构建器。"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence, Tuple

from app.services.punctuation.semantic_buffer import SemanticChunk


@dataclass(frozen=True)
class HintBuilder:
    """仅产出术语 hints 与短尾上下文，避免整段正文拼接。"""

    max_hint_items: int = 8
    max_terms: int = 6
    tail_context_chars: int = 48
    max_tail_chunks: int = 2

    _LATIN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9'_-]{2,31}")
    _CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,6}")

    def build_hints(
        self,
        chunks: Sequence[SemanticChunk],
        *,
        primary_language: str,
    ) -> Tuple[str, ...]:
        terms = self._collect_terms(chunks)
        hints = list(terms[: self.max_terms])
        tail_context = self._build_tail_context(chunks, primary_language=primary_language)
        if tail_context:
            hints.append(tail_context)
        return tuple(self._dedupe_and_trim(hints))

    def _collect_terms(self, chunks: Sequence[SemanticChunk]) -> list[str]:
        counter: Counter[str] = Counter()
        for chunk in chunks:
            text = str(getattr(chunk, "text", "") or "")
            for token in self._LATIN_PATTERN.findall(text):
                counter[token] += 1
            for token in self._CJK_PATTERN.findall(text):
                counter[token] += 1
        ordered = [token for token, _ in counter.most_common()]
        return ordered

    def _build_tail_context(
        self,
        chunks: Sequence[SemanticChunk],
        *,
        primary_language: str,
    ) -> str:
        recent_chunks = list(chunks[-self.max_tail_chunks :])
        merged = " ".join(str(getattr(chunk, "text", "") or "").strip() for chunk in recent_chunks).strip()
        if not merged:
            return ""
        normalized = " ".join(merged.split())
        if len(normalized) > self.tail_context_chars:
            normalized = normalized[-self.tail_context_chars :].lstrip()
        return f"ctx:{primary_language}:{normalized}"

    def _dedupe_and_trim(self, hints: Sequence[str]) -> list[str]:
        deduped: list[str] = []
        seen = set()
        for hint in hints:
            normalized = str(hint or "").strip()
            if not normalized:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
            if len(deduped) >= self.max_hint_items:
                break
        return deduped
