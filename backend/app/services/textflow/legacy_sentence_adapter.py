"""LegacySentenceAdapter（Phase 5）。"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from app.models.sensevoice_models import SentenceSegment, TextSource
from app.services.alignment.types import OutputTrace
from app.services.textflow.contracts import RenderResult, RenderedSubtitle


class LegacySentenceAdapter:
    """在南向兼容边界把 RenderResult 转回 SentenceSegment。"""

    def to_legacy(
        self,
        *,
        render_result: RenderResult,
    ) -> Tuple[List[SentenceSegment], List[OutputTrace]]:
        sentences: List[SentenceSegment] = []
        traces: List[OutputTrace] = []
        subtitles: Sequence[RenderedSubtitle] = list(render_result.subtitles or ())

        for idx, subtitle in enumerate(subtitles):
            trace_data = dict(subtitle.trace or {})
            split_reason = str(trace_data.get("split_reason", "") or "")
            split_risk = str(trace_data.get("split_risk", "") or "")
            window_id = str(trace_data.get("window_id", "") or "")
            mapped_cut_time = trace_data.get("mapped_cut_time", subtitle.end)
            mapping_quality = str(trace_data.get("mapping_quality", "") or "")
            mapping_reason = str(trace_data.get("mapping_reason", "") or "")
            pyannote_frame_time = trace_data.get("pyannote_frame_time")

            sentence = SentenceSegment(
                text=str(subtitle.text_display or ""),
                text_clean=str(subtitle.text_display or ""),
                start=float(subtitle.start),
                end=float(subtitle.end),
                source=self._map_text_source(str(subtitle.text_source or "")),
                is_draft=False,
                is_finalized=True,
                speaker_id=subtitle.speaker_id,
                turn_id=subtitle.turn_id,
                split_reason=split_reason,
                split_risk=split_risk,
                window_id=window_id,
                pyannote_frame_time=pyannote_frame_time,
                mapped_cut_time=mapped_cut_time,
                mapping_quality=mapping_quality,
                mapping_reason=mapping_reason,
                segment_id=subtitle.segment_id,
                sentence_uid=subtitle.segment_id,
            )
            sentences.append(sentence)
            traces.append(
                OutputTrace(
                    sentence_index=idx,
                    split_reason=split_reason,
                    split_risk=split_risk,
                    window_id=window_id,
                    pyannote_frame_time=pyannote_frame_time,
                    mapped_cut_time=mapped_cut_time,
                    mapping_quality=mapping_quality,
                    mapping_reason=mapping_reason,
                    sentence_start=float(subtitle.start),
                    sentence_end=float(subtitle.end),
                )
            )
        return sentences, traces

    @staticmethod
    def _map_text_source(text_source: str) -> TextSource:
        normalized = str(text_source or "").strip().lower()
        if normalized == "fast":
            return TextSource.SENSEVOICE
        if normalized in {"slow", "aligned"}:
            return TextSource.WHISPER_PATCH
        return TextSource.WHISPER_PATCH

