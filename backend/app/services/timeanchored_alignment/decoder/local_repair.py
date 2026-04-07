"""局部低置信 token 的时间修补。"""

from __future__ import annotations

from dataclasses import replace

from app.services.timeanchored_alignment.decoder.contracts import DecoderPath, DecoderStep
from app.services.timeanchored_alignment.preparation.contracts import PreparationBundle


class LocalRepair:
    """只做最小时间修补，不改变正式输出链。"""

    def repair(
        self,
        *,
        preparation: PreparationBundle,
        decode_path: DecoderPath,
    ) -> DecoderPath:
        frame_stride = float(preparation.acoustic_observation_pack.control_meta.get("frame_stride", 0.06) or 0.06)
        repaired_steps: list[DecoderStep] = []
        previous_end = float(preparation.scope.absolute_time_range[0] or 0.0)
        steps = tuple(decode_path.steps)
        index = 0
        while index < len(steps):
            step = steps[index]
            metadata = dict(step.metadata or {})
            start = metadata.get("slice_start")
            end = metadata.get("slice_end")
            if self._needs_local_repair(
                step=step,
                start=start,
                end=end,
                metadata=metadata,
            ):
                span_end_index = index
                while span_end_index + 1 < len(steps):
                    next_step = steps[span_end_index + 1]
                    next_metadata = dict(next_step.metadata or {})
                    if not self._needs_local_repair(
                        step=next_step,
                        start=next_metadata.get("slice_start"),
                        end=next_metadata.get("slice_end"),
                        metadata=next_metadata,
                    ):
                        break
                    span_end_index += 1
                next_anchor = self._resolve_next_anchor(
                    steps=steps,
                    start_index=span_end_index + 1,
                )
                if next_anchor is not None and next_anchor > previous_end:
                    span_count = span_end_index - index + 1
                    span_width = float(next_anchor - previous_end) / float(max(span_count, 1))
                    for span_offset in range(span_count):
                        span_step = steps[index + span_offset]
                        span_metadata = dict(span_step.metadata or {})
                        resolved_start = previous_end + float(span_offset) * span_width
                        resolved_end = previous_end + float(span_offset + 1) * span_width
                        span_metadata["synthetic"] = True
                        span_metadata["repair_reason"] = "bounded_span_interpolation"
                        span_metadata["resolved_start"] = float(resolved_start)
                        span_metadata["resolved_end"] = float(resolved_end)
                        repaired_steps.append(replace(span_step, metadata=span_metadata))
                    previous_end = float(next_anchor)
                    index = span_end_index + 1
                    continue
                start = previous_end
                end = float(start) + max(frame_stride, 0.03)
                metadata["synthetic"] = True
                metadata["repair_reason"] = "local_interpolation"
            start = max(float(start), previous_end)
            end = max(float(end), start)
            metadata["resolved_start"] = float(start)
            metadata["resolved_end"] = float(end)
            previous_end = float(end)
            repaired_steps.append(replace(step, metadata=metadata))
            index += 1
        return replace(decode_path, steps=tuple(repaired_steps))

    @staticmethod
    def _resolve_next_anchor(
        *,
        steps: tuple[DecoderStep, ...],
        start_index: int,
    ) -> float | None:
        for index in range(max(0, int(start_index)), len(steps)):
            metadata = dict(steps[index].metadata or {})
            start = metadata.get("slice_start")
            end = metadata.get("slice_end")
            if LocalRepair._needs_local_repair(
                step=steps[index],
                start=start,
                end=end,
                metadata=metadata,
            ):
                continue
            try:
                resolved_start = float(start)
                resolved_end = float(end)
            except (TypeError, ValueError):
                continue
            if resolved_end > resolved_start:
                return resolved_start
        return None

    @staticmethod
    def _needs_local_repair(
        *,
        step: DecoderStep,
        start: object,
        end: object,
        metadata: dict[str, object],
    ) -> bool:
        if start is None or end is None:
            return True
        try:
            if float(end) <= float(start):
                return True
        except (TypeError, ValueError):
            return True
        synthetic_reason = str(metadata.get("synthetic_reason", "") or "")
        match_kind = str(metadata.get("match_kind", "") or "")
        if synthetic_reason in {"null_align", "canonical_only_token", "estimated"}:
            return True
        if match_kind in {"null", "estimated"}:
            return True
        return bool(step.synthetic or metadata.get("synthetic"))
