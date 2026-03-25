"""Whisper prompt seed 生成。"""

from __future__ import annotations

from app.services.timeanchored_alignment.slow_window.contracts import PromptSeed, WindowSourceUnit


class PromptSeedBuilder:
    """从 source units 汇总 prompt seed。"""

    def build(self, source_units: tuple[WindowSourceUnit, ...]) -> PromptSeed:
        texts = [unit.text.strip() for unit in source_units if unit.text.strip()]
        keywords: list[str] = []
        seen: set[str] = set()
        for text in texts:
            for token in text.replace("。", " ").replace("，", " ").split():
                normalized = token.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                keywords.append(normalized)
                if len(keywords) >= 8:
                    break
            if len(keywords) >= 8:
                break
        return PromptSeed(text=" ".join(texts).strip(), keywords=tuple(keywords))
