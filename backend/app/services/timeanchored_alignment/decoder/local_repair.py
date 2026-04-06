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
        for step in decode_path.steps:
            metadata = dict(step.metadata or {})
            start = metadata.get("slice_start")
            end = metadata.get("slice_end")
            if self._needs_local_repair(
                step=step,
                start=start,
                end=end,
                metadata=metadata,
            ):
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
        return replace(decode_path, steps=tuple(repaired_steps))

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
