"""
Timeline 轮次构建器（Phase 2）。

设计模式：Builder Pattern。
原因：将“聚类轨迹 + segmentation 边界”融合逻辑集中封装，
避免在编排层散落 turn 边界判定细节。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models.speaker_timeline_models import SpeakerTurn, TimelineSource


@dataclass(frozen=True)
class TimelineChunkAssignment:
    """单个 chunk 的聚类与状态信息。"""

    chunk_id: str
    start: float
    end: float
    speaker_id: str
    speaker_status: str
    is_overlap: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class TurnBuilderConfig:
    """轮次构建参数。"""

    segmentation_boundary_tolerance_sec: float = 0.25
    long_pause_cut_sec: float = 1.8
    min_turn_duration_sec: float = 0.15


class TurnBuilder:
    """融合聚类轨迹和 segmentation 边界，输出 `SpeakerTurn`。"""

    def __init__(self, config: TurnBuilderConfig | None = None) -> None:
        self.config = config or TurnBuilderConfig()

    def build_turns(
        self,
        *,
        block_id: str,
        assignments: list[TimelineChunkAssignment],
        segmentation_boundaries: list[float],
    ) -> list[SpeakerTurn]:
        """构建块级轮次。"""
        if not assignments:
            return []

        ordered = sorted(assignments, key=lambda item: (item.start, item.end))
        sorted_boundaries = sorted(segmentation_boundaries)

        turns: list[SpeakerTurn] = []
        current_start = ordered[0].start
        current_end = ordered[0].end
        current_speaker = self._resolve_turn_speaker(
            candidate_speaker_id=ordered[0].speaker_id,
            speaker_status=ordered[0].speaker_status,
            previous_turn_speaker_id=None,
        )
        has_overlap = ordered[0].is_overlap

        for idx in range(1, len(ordered)):
            prev_item = ordered[idx - 1]
            item = ordered[idx]

            prev_speaker = self._resolve_turn_speaker(
                candidate_speaker_id=prev_item.speaker_id,
                speaker_status=prev_item.speaker_status,
                previous_turn_speaker_id=current_speaker,
            )
            next_speaker = self._resolve_turn_speaker(
                candidate_speaker_id=item.speaker_id,
                speaker_status=item.speaker_status,
                previous_turn_speaker_id=current_speaker,
            )

            is_speaker_changed = next_speaker != prev_speaker
            pause_gap = max(0.0, item.start - prev_item.end)
            has_long_pause = pause_gap >= self.config.long_pause_cut_sec

            split_time, source, boundary_confidence = self._pick_split_boundary(
                previous_end=prev_item.end,
                next_start=item.start,
                is_speaker_changed=is_speaker_changed,
                has_long_pause=has_long_pause,
                segmentation_boundaries=sorted_boundaries,
            )

            if split_time is None:
                current_end = max(current_end, item.end)
                has_overlap = has_overlap or item.is_overlap
                continue

            if split_time - current_start >= self.config.min_turn_duration_sec:
                turns.append(
                    self._build_turn(
                        block_id=block_id,
                        turn_index=len(turns),
                        speaker_id=current_speaker,
                        start=current_start,
                        end=split_time,
                        boundary_confidence=boundary_confidence,
                        is_overlap=has_overlap,
                        source=source,
                    )
                )

            current_start = split_time
            current_end = max(item.end, split_time)
            current_speaker = next_speaker
            has_overlap = item.is_overlap

        if current_end - current_start >= self.config.min_turn_duration_sec:
            turns.append(
                self._build_turn(
                    block_id=block_id,
                    turn_index=len(turns),
                    speaker_id=current_speaker,
                    start=current_start,
                    end=current_end,
                    boundary_confidence=1.0,
                    is_overlap=has_overlap,
                    source="merged" if sorted_boundaries else "cluster",
                )
            )

        return turns

    def _resolve_turn_speaker(
        self,
        *,
        candidate_speaker_id: str,
        speaker_status: str,
        previous_turn_speaker_id: str | None,
    ) -> str:
        """候选态 speaker 不参与硬切分。"""
        if speaker_status == "candidate":
            if previous_turn_speaker_id:
                return previous_turn_speaker_id
            return "unknown"
        return candidate_speaker_id or "unknown"

    def _pick_split_boundary(
        self,
        *,
        previous_end: float,
        next_start: float,
        is_speaker_changed: bool,
        has_long_pause: bool,
        segmentation_boundaries: list[float],
    ) -> tuple[float | None, TimelineSource, float]:
        """选择分割点并估算边界置信度。"""
        is_need_split = is_speaker_changed or has_long_pause
        if not is_need_split:
            return None, "cluster", 0.0

        midpoint = (previous_end + next_start) / 2.0
        matched_boundary = self._find_matched_segmentation_boundary(
            midpoint=midpoint,
            previous_end=previous_end,
            next_start=next_start,
            segmentation_boundaries=segmentation_boundaries,
        )

        if matched_boundary is not None:
            source: TimelineSource = "merged"
            split_time = matched_boundary
            base_confidence = 0.90 if is_speaker_changed else 0.82
        else:
            source = "cluster"
            split_time = midpoint
            base_confidence = 0.72 if is_speaker_changed else 0.64

        if has_long_pause:
            base_confidence += 0.06
        confidence = max(0.0, min(1.0, base_confidence))
        return split_time, source, confidence

    def _find_matched_segmentation_boundary(
        self,
        *,
        midpoint: float,
        previous_end: float,
        next_start: float,
        segmentation_boundaries: list[float],
    ) -> float | None:
        lower = previous_end - self.config.segmentation_boundary_tolerance_sec
        upper = next_start + self.config.segmentation_boundary_tolerance_sec

        candidates = [
            value
            for value in segmentation_boundaries
            if lower <= value <= upper
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: abs(item - midpoint))

    @staticmethod
    def _build_turn(
        *,
        block_id: str,
        turn_index: int,
        speaker_id: str,
        start: float,
        end: float,
        boundary_confidence: float,
        is_overlap: bool,
        source: TimelineSource,
    ) -> SpeakerTurn:
        return SpeakerTurn(
            turn_id=f"{block_id}-turn-{turn_index:04d}",
            block_id=block_id,
            speaker_id=speaker_id,
            start=float(start),
            end=float(end),
            boundary_confidence=float(boundary_confidence),
            is_overlap=bool(is_overlap),
            source=source,
        )

