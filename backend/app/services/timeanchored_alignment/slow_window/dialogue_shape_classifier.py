"""对话形态分类。"""

from __future__ import annotations

from collections import Counter

from app.services.timeanchored_alignment.slow_window.contracts import (
    DialogueShapeSnapshot,
    SlowWindowIngressUnit,
)


class DialogueShapeClassifier:
    """按 speaker/turn 分布给出轻量对话形态结论。"""

    def classify(self, ingress_units: tuple[SlowWindowIngressUnit, ...]) -> DialogueShapeSnapshot:
        if not ingress_units:
            return DialogueShapeSnapshot(
                shape="empty",
                speaker_count=0,
                dominant_speaker_id=None,
                dominant_speaker_ratio=0.0,
                speaker_switch_count=0,
                speaker_switch_density=0.0,
                turn_count=0,
                avg_turn_duration_sec=0.0,
            )

        speakers = [unit.speaker_id for unit in ingress_units]
        counts = Counter(speakers)
        dominant_speaker, dominant_count = counts.most_common(1)[0]
        speaker_switch_count = 0
        previous = None
        for speaker in speakers:
            if previous is not None and speaker != previous:
                speaker_switch_count += 1
            previous = speaker

        total_duration = sum(max(float(unit.audio_range[1]) - float(unit.audio_range[0]), 0.0) for unit in ingress_units)
        speaker_count = len(counts)
        dominant_ratio = dominant_count / max(len(ingress_units), 1)
        if speaker_count <= 1:
            shape = "single_speaker"
        elif dominant_ratio >= 0.75:
            shape = "dominant_plus_backchannel"
        elif speaker_switch_count <= max(1, len(ingress_units) // 2):
            shape = "two_party_stable"
        else:
            shape = "ping_pong_fragmented"

        return DialogueShapeSnapshot(
            shape=shape,
            speaker_count=speaker_count,
            dominant_speaker_id=dominant_speaker,
            dominant_speaker_ratio=dominant_ratio,
            speaker_switch_count=speaker_switch_count,
            speaker_switch_density=speaker_switch_count / max(total_duration, 1e-6),
            turn_count=len({unit.turn_id for unit in ingress_units if unit.turn_id}),
            avg_turn_duration_sec=total_duration / max(len(ingress_units), 1),
        )
