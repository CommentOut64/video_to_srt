"""
Bridge Prompt 构建器。
V3.2.0+dev.20260205.04

更新日志：
- V3.2.0+dev.20260205.04: 使用空格分隔关键词，避免逗号密集导致 Whisper 标点感染
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

from app.models.sensevoice_models import SentenceSegment

_EN_STOPWORDS = {
    "the", "and", "but", "this", "that", "there", "here",
    "when", "what", "who", "where", "why", "how",
    "then", "now", "so", "or", "if", "as", "at",
    "it", "its", "i", "you", "he", "she", "we", "they",
}
_CJK_STOPWORDS = {"这个", "那个", "什么", "怎么", "为什么", "因为", "所以", "但是", "然后"}
_EN_WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
_CJK_WORD_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,4}")


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _join_sentences(sentences: List[SentenceSegment]) -> str:
    parts = [sentence.text for sentence in sentences if sentence and sentence.text]
    if not parts:
        return ""
    if _has_cjk("".join(parts)):
        return "".join(parts)
    return " ".join(part.strip() for part in parts if part.strip())


def _extract_keywords(text: str, *, min_word_length: int = 3) -> Counter:
    counter: Counter = Counter()
    if not text:
        return counter

    for word in _EN_WORD_PATTERN.findall(text):
        normalized = word.lower()
        if normalized in _EN_STOPWORDS:
            continue
        if len(normalized) < min_word_length:
            continue
        counter[word] += 1

    for word in _CJK_WORD_PATTERN.findall(text):
        if word in _CJK_STOPWORDS:
            continue
        counter[word] += 1
    return counter


class PromptBuilder:
    """Prompt 构建器（建造者模式）：输出受限的 Glossary，避免正文回显。
    
    V3.2.0+dev.20260205.04: 使用空格分隔关键词，避免逗号密集导致标点感染。
    """

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
        max_keywords: int = 20,
        max_prompt_length: int = 160,
    ) -> str:
        context_sentences = sentences[-max(1, int(max_context_sentences)) :]
        context_text = _join_sentences(context_sentences)
        segments = [
            segment.strip()
            for segment in [tail_context, history_context, context_text]
            if segment and segment.strip()
        ]
        if not segments:
            return ""
        source_text = " ".join(segments)
        keywords = _extract_keywords(source_text)
        if not keywords:
            return ""

        sorted_keywords = sorted(
            keywords.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )
        selected: List[str] = []
        current_length = len("Glossary: .")
        for word, _ in sorted_keywords:
            if len(selected) >= max(1, int(max_keywords)):
                break
            # V3.2.0+dev.20260205.04: 改为空格分隔，避免逗号感染
            additional = len(word) + (1 if selected else 0)  # 空格长度
            if current_length + additional > max_prompt_length:
                break
            selected.append(word)
            current_length += additional

        if not selected:
            return ""
        # V3.2.0+dev.20260205.04: 使用空格分隔，避免密集逗号
        return f"Glossary: {' '.join(selected)}."
