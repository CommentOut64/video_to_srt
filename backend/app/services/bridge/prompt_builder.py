"""
Bridge Prompt 构建器。
V3.2.0+dev.20260201.04
"""
from __future__ import annotations

from typing import List, Optional

from app.models.sensevoice_models import SentenceSegment


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _join_sentences(sentences: List[SentenceSegment]) -> str:
    parts = [sentence.text for sentence in sentences if sentence and sentence.text]
    if not parts:
        return ""
    if _has_cjk("".join(parts)):
        return "".join(parts)
    return " ".join(part.strip() for part in parts if part.strip())


class PromptBuilder:
    """Prompt 构建器（建造者模式）：集中处理上下文拼接与尾句截取。"""

    def build_tail(self, sentences: List[SentenceSegment], max_sentences: int = 2) -> str:
        if not sentences:
            return ""
        tail = sentences[-max(1, int(max_sentences)) :]
        return _join_sentences(tail)

    def build_prompt(
        self,
        sentences: List[SentenceSegment],
        *,
        history_context: Optional[str] = None,
        tail_context: str = "",
        max_context_sentences: int = 6,
    ) -> str:
        context_sentences = sentences[-max(1, int(max_context_sentences)) :]
        context_text = _join_sentences(context_sentences)
        segments = [seg for seg in [tail_context, history_context, context_text] if seg]
        if not segments:
            return ""
        return " ".join(segment.strip() for segment in segments if segment.strip())
