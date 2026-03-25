"""候选切点跟踪。"""

from __future__ import annotations

from app.services.timeanchored_alignment.slow_window.contracts import SlowWindowIngressUnit


class CutCandidateTracker:
    """只记录候选切点，不直接 flush。"""

    def __init__(self, *, long_pause_sec: float = 1.8) -> None:
        self._long_pause_sec = float(long_pause_sec)

    def evaluate(
        self,
        previous: SlowWindowIngressUnit | None,
        current: SlowWindowIngressUnit,
    ) -> tuple[str, ...]:
        if previous is None:
            return ()

        reasons: list[str] = []
        pause_gap = max(0.0, float(current.audio_range[0]) - float(previous.audio_range[1]))
        if pause_gap >= self._long_pause_sec:
            reasons.append("long_pause")
        if current.language and previous.language and current.language != previous.language:
            reasons.append("language_shift")
        if current.speaker_id and previous.speaker_id and current.speaker_id != previous.speaker_id:
            reasons.append("speaker_change")
        return tuple(reasons)
