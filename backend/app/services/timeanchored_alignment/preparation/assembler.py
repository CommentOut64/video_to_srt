"""AlignmentPreparation assembler。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence
import unicodedata

from app.core.logging import resolve_loguru_logger
from app.services.alignment.types import PunctTrack
from app.services.timeanchored_alignment.chunk_projector import ChunkWindow
from app.services.timeanchored_alignment.contracts import (
    ProtectedSpan,
    TextTruthPackage,
    TextTruthUnit,
)
from app.services.timeanchored_alignment.preparation.contracts import (
    AlignmentPreparationCompat,
    AlignmentPreparationPackage,
    PreparedSlowText,
    PunctuationEvidence,
)
from app.services.timeanchored_alignment.preparation.display_projection_builder import (
    DisplayProjectionBuilder,
)
from app.services.timeanchored_alignment.preparation.hook_selector import HookSelector
from app.services.timeanchored_alignment.preparation.language_profile_builder import (
    LanguageProfileBuilder,
)
from app.services.timeanchored_alignment.preparation.pronunciation_hint_builder import (
    PronunciationHintBuilder,
)
from app.services.timeanchored_alignment.preparation.punctuation_evidence_builder import (
    PunctuationEvidenceBuilder,
)
from app.services.timeanchored_alignment.preparation.safe_pre_normalizer import (
    SafePreNormalizer,
)
from app.services.timeanchored_alignment.preparation.slow_text_normalizer import (
    SlowTextNormalizer,
)
from app.services.timeanchored_alignment.preparation.source_attribution_binder import (
    SourceAttributionBinder,
)
from app.services.timeanchored_alignment.preparation.structure_protector import (
    StructureProtector,
)
from app.services.timeanchored_alignment.slow_window.contracts import ReadySlowWindow
from app.services.timeanchored_alignment.window_time_base_assembler import WindowTimeBasePackage


class AlignmentPreparationAssembler:
    """把慢流 window + time base 收口为 AlignmentPreparationPackage。"""

    def __init__(
        self,
        *,
        logger: Any | None = None,
        sanitizer: Any | None = None,
        hallucination_detector: Any | None = None,
        slow_text_normalizer: SlowTextNormalizer | None = None,
        safe_pre_normalizer: SafePreNormalizer | None = None,
        structure_protector: StructureProtector | None = None,
        display_projection_builder: DisplayProjectionBuilder | None = None,
        punctuation_evidence_builder: PunctuationEvidenceBuilder | None = None,
        source_attribution_binder: SourceAttributionBinder | None = None,
        hook_selector: HookSelector | None = None,
        language_profile_builder: LanguageProfileBuilder | None = None,
        pronunciation_hint_builder: PronunciationHintBuilder | None = None,
    ) -> None:
        self._logger = resolve_loguru_logger(
            logger,
            __name__,
            layer="准备层",
            component="alignment_preparation",
        )
        self._slow_text_normalizer = slow_text_normalizer or SlowTextNormalizer(
            sanitizer=sanitizer,
            hallucination_detector=hallucination_detector,
        )
        self._safe_pre_normalizer = safe_pre_normalizer or SafePreNormalizer()
        self._structure_protector = structure_protector or StructureProtector()
        self._display_projection_builder = display_projection_builder or DisplayProjectionBuilder()
        self._punctuation_evidence_builder = punctuation_evidence_builder or PunctuationEvidenceBuilder()
        self._source_attribution_binder = source_attribution_binder or SourceAttributionBinder()
        self._hook_selector = hook_selector or HookSelector()
        self._language_profile_builder = language_profile_builder or LanguageProfileBuilder()
        self._pronunciation_hint_builder = pronunciation_hint_builder or PronunciationHintBuilder()

    def prepare(
        self,
        *,
        ready_window: ReadySlowWindow,
        window_time_base: WindowTimeBasePackage,
        whisper_result: Mapping[str, Any],
        default_language: str,
        fallback_text: str = "",
        chunk_window: ChunkWindow | None = None,
        external_punct_track: PunctTrack | None = None,
    ) -> AlignmentPreparationPackage:
        normalized = self._slow_text_normalizer.normalize(
            whisper_result=whisper_result,
            default_language=default_language,
            fallback_text=fallback_text,
        )
        self._logger.debug(
            "AlignmentPreparation 开始 window_id={} owner_chunk_id={} source_chunk_count={} source_unit_count={} raw_unit_count={} word_unit_count={} whisper_segment_count={} normalized_text_len={} fallback_text_len={} timed_unit_count={}",
            ready_window.window_id,
            ready_window.owner_chunk_id,
            len(ready_window.source_chunk_ids),
            len(ready_window.source_units),
            len(window_time_base.raw_units),
            len(window_time_base.word_units),
            len(tuple(whisper_result.get("segments") or ())),
            len(str(normalized.source_text or "")),
            len(str(fallback_text or "")),
            self._count_timed_units(normalized.text_truth.units),
        )
        safe_text = self._safe_pre_normalizer.normalize(
            text=normalized.source_text,
            language=normalized.language,
        )
        protected = self._structure_protector.protect(text=safe_text.normalized_text)
        projection = self._display_projection_builder.build(
            source_text=safe_text.normalized_text,
            protected_units=protected.protected_units,
            source_language=safe_text.language,
        )
        lexical_punctuation_evidences = self._punctuation_evidence_builder.build(
            source_text=safe_text.normalized_text,
            source_char_to_text_index=projection.source_char_to_text_index,
            protected_units=protected.protected_units,
        )
        external_punctuation_evidences = tuple()
        if external_punct_track is not None and getattr(external_punct_track, "positions", None):
            external_punctuation_evidences = self._punctuation_evidence_builder.build_from_punct_track(
                track_clean_text_ref=str(getattr(external_punct_track, "clean_text_ref", "") or ""),
                window_text=projection.window_text.text,
                positions=tuple(getattr(external_punct_track, "positions", ()) or ()),
                evidence_source="punct_track",
                remap_mode="tolerant",
            )
        punctuation_evidences = self._merge_punctuation_evidences(
            lexical_evidences=lexical_punctuation_evidences,
            external_evidences=external_punctuation_evidences,
        )
        language_profile = self._language_profile_builder.build(
            text=projection.window_text.text,
            language_hint=safe_text.language,
        )
        pronunciation = self._pronunciation_hint_builder.build(
            text=projection.window_text.text,
            language_hint=safe_text.language,
            language_runs=language_profile.package.runs,
            dominant_language=language_profile.package.dominant_language,
        )
        slots = self._source_attribution_binder.bind(
            text=projection.window_text.text,
            source_units=ready_window.source_units,
            punctuation_evidences=punctuation_evidences,
            pronunciation_hints=pronunciation.hints,
        )

        compat_protected_spans = tuple(
            ProtectedSpan(
                start=unit.start,
                end=unit.end,
                kind=unit.kind,
                text=unit.text,
            )
            for unit in projection.protected_units
        )
        compat_text_truth = self._build_compat_text_truth(
            base_truth=normalized.text_truth,
            lexical_text=projection.window_text.text,
            protected_spans=compat_protected_spans,
        )
        resolved_chunk_window = chunk_window or self._build_owner_chunk_window(ready_window=ready_window)
        compat = AlignmentPreparationCompat(
            time_base=window_time_base,
            text_truth=compat_text_truth,
            protected_spans=compat_protected_spans,
            language_runs=replace(
                language_profile.package,
                source_text=projection.window_text.text,
            ),
            pronunciation=pronunciation.package,
            pronunciation_report={
                **pronunciation.report,
                "window_kind": str(language_profile.package.window_kind),
                "foreign_run_ratio": float(language_profile.package.foreign_run_ratio),
            },
            chunk_window=resolved_chunk_window,
        )
        package = AlignmentPreparationPackage(
            window_id=ready_window.window_id,
            owner_chunk_id=ready_window.owner_chunk_id,
            owner_chunk_index=ready_window.owner_chunk_index,
            source_chunk_ids=ready_window.source_chunk_ids,
            source_chunk_indices=ready_window.source_chunk_indices,
            slow_text=PreparedSlowText(
                window_text=projection.window_text,
                slots=slots,
                punctuation_evidences=punctuation_evidences,
                protected_units=projection.protected_units,
                language_runs=tuple(language_profile.package.runs),
                pronunciation_hints=pronunciation.hints,
            ),
            fast_hooks=self._hook_selector.select(window_time_base=window_time_base),
            coverage=ready_window.coverage,
            compat=compat,
        )
        self._logger.debug(
            "AlignmentPreparation 完成 window_id={} window_text_len={} slot_count={} punctuation_count={} lexical_punctuation_count={} external_punctuation_count={} protected_unit_count={} language_run_count={} pronunciation_hint_count={} fast_hook_count={} compat_text_truth_unit_count={} compat_timed_unit_count={}",
            package.window_id,
            len(package.slow_text.window_text.text),
            len(package.slow_text.slots),
            len(package.slow_text.punctuation_evidences),
            len(lexical_punctuation_evidences),
            len(external_punctuation_evidences),
            len(package.slow_text.protected_units),
            len(package.slow_text.language_runs),
            len(package.slow_text.pronunciation_hints),
            len(package.fast_hooks),
            len(package.compat.text_truth.units),
            self._count_timed_units(package.compat.text_truth.units),
        )
        return package

    @staticmethod
    def _merge_punctuation_evidences(
        *,
        lexical_evidences: tuple[PunctuationEvidence, ...],
        external_evidences: tuple[PunctuationEvidence, ...],
    ) -> tuple[PunctuationEvidence, ...]:
        dedup: dict[tuple[str, int, str], PunctuationEvidence] = {}
        for item in lexical_evidences:
            dedup[(str(item.mark), int(item.source_char_index), str(item.attach_side))] = item
        for item in external_evidences:
            key = (str(item.mark), int(item.source_char_index), str(item.attach_side))
            if key in dedup:
                continue
            dedup[key] = item
        return tuple(dedup.values())

    @staticmethod
    def _build_owner_chunk_window(*, ready_window: ReadySlowWindow) -> ChunkWindow:
        for binding in ready_window.coverage.chunk_bindings:
            if binding.is_owner or binding.role == "owner":
                return ChunkWindow(
                    chunk_ref=int(binding.chunk_index),
                    start=float(binding.chunk_start),
                    end=float(binding.chunk_end),
                )
        start = float(ready_window.audio_segments[0][0]) if ready_window.audio_segments else 0.0
        end = float(ready_window.audio_segments[-1][1]) if ready_window.audio_segments else max(start, 0.01)
        return ChunkWindow(
            chunk_ref=int(ready_window.owner_chunk_index),
            start=start,
            end=end,
        )

    @staticmethod
    def _build_compat_text_truth(
        *,
        base_truth: TextTruthPackage,
        lexical_text: str,
        protected_spans: Sequence[ProtectedSpan],
    ) -> TextTruthPackage:
        confidence = 0.0
        if base_truth.units:
            confidence = sum(float(unit.confidence) for unit in base_truth.units) / len(base_truth.units)
        units = AlignmentPreparationAssembler._try_preserve_timed_units(
            base_truth=base_truth,
            lexical_text=lexical_text,
        )
        if units is None:
            units = AlignmentPreparationAssembler._tokenize_text_truth_units(
                text=lexical_text,
                language=str(base_truth.language or "auto"),
                default_confidence=confidence or 0.8,
            )
        return replace(
            base_truth,
            units=units,
            raw_text=lexical_text,
            normalized_text=lexical_text,
            protected_spans=tuple(protected_spans),
        )

    @staticmethod
    def _try_preserve_timed_units(
        *,
        base_truth: TextTruthPackage,
        lexical_text: str,
    ) -> tuple[TextTruthUnit, ...] | None:
        preserved_units: list[TextTruthUnit] = []
        for unit in base_truth.units:
            normalized_token = AlignmentPreparationAssembler._strip_punctuation(
                str(unit.normalized_text or unit.text or "")
            )
            if not normalized_token:
                continue
            preserved_units.append(
                TextTruthUnit(
                    text=normalized_token,
                    normalized_text=normalized_token,
                    confidence=float(unit.confidence),
                    language=str(unit.language),
                    start=unit.start,
                    end=unit.end,
                    source=str(unit.source),
                )
            )
        if not preserved_units:
            return None
        preserved_joined = "".join(item.normalized_text for item in preserved_units)
        if AlignmentPreparationAssembler._normalize_match_key(
            preserved_joined
        ) != AlignmentPreparationAssembler._normalize_match_key(lexical_text):
            return None
        return tuple(preserved_units)

    @staticmethod
    def _tokenize_text_truth_units(
        *,
        text: str,
        language: str,
        default_confidence: float,
    ) -> tuple[TextTruthUnit, ...]:
        source_text = str(text or "")
        if not source_text:
            return tuple()
        if any(char.isspace() for char in source_text):
            tokens = [item for item in source_text.split(" ") if item]
        else:
            tokens = [char for char in source_text if char]
        return tuple(
            TextTruthUnit(
                text=token,
                normalized_text=token,
                confidence=float(default_confidence),
                language=language,
            )
            for token in tokens
        )

    @staticmethod
    def _strip_punctuation(text: str) -> str:
        return "".join(
            char
            for char in str(text or "")
            if not unicodedata.category(char).startswith("P")
        ).strip()

    @staticmethod
    def _normalize_match_key(text: str) -> str:
        return "".join(
            char
            for char in str(text or "")
            if not char.isspace() and not unicodedata.category(char).startswith("P")
        )

    @staticmethod
    def _count_timed_units(units: Sequence[TextTruthUnit]) -> int:
        return sum(
            1
            for unit in units
            if getattr(unit, "start", None) is not None and getattr(unit, "end", None) is not None
        )
