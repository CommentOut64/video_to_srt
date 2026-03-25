"""语言画像推断。"""

from __future__ import annotations

from collections import Counter

from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowLanguageProfile,
    WindowSourceUnit,
)


class LanguageHintResolver:
    """根据 source units 冻结语言画像。"""

    def resolve(self, source_units: tuple[WindowSourceUnit, ...]) -> WindowLanguageProfile:
        languages = [unit.language for unit in source_units if unit.language]
        if not languages:
            return WindowLanguageProfile(
                primary_language="auto",
                language_mix_state="single_language",
                decision_domains=("auto",),
                should_bypass_whisper=False,
            )

        counts = Counter(languages)
        primary_language, primary_count = counts.most_common(1)[0]
        ratio = primary_count / max(len(languages), 1)
        if len(counts) == 1:
            mix_state = "single_language"
        elif ratio >= 0.6:
            mix_state = "dominant_mixed"
        else:
            mix_state = "true_mixed"

        return WindowLanguageProfile(
            primary_language=primary_language,
            language_mix_state=mix_state,
            decision_domains=tuple(sorted(counts.keys())),
            should_bypass_whisper=False,
        )
