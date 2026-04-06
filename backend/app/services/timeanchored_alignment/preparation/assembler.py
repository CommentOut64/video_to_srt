"""AlignmentPreparation assembler。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence
import unicodedata

from app.core.logging import resolve_loguru_logger
from app.services.alignment.types import PunctTrack
from app.services.timeanchored_alignment.chunk_projector import ChunkWindow
from app.services.timeanchored_alignment.contracts import (
    AcousticObservationPack,
    AcousticObservationQuality,
    AcousticObservationSlice,
    AcousticObservationTokenCandidate,
    LayerSummary,
    OBSERVATION_CAPABILITY_FRAME_POSTERIOR,
    OBSERVATION_CAPABILITY_TIMESTAMP_ONLY,
    OBSERVATION_CAPABILITY_TOKEN_TOPK,
    ProtectedSpan,
    SelectedTextTruth,
    TextTruthPackage,
    TextTruthUnit,
)
from app.services.timeanchored_alignment.preparation.contracts import (
    AlignmentPreparationCompat,
    CanonicalSequence,
    CanonicalToken,
    ExternalStableFacts,
    PreparationBundle,
    PreparationProvenance,
    PreparationReport,
    PreparationScope,
    PreparedSlowText,
    PunctuationEvidence,
    PronunciationEdge,
    PronunciationGraph,
    PronunciationStateNode,
    PronunciationTokenNode,
    PronunciationVariant,
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
from app.services.timeanchored_alignment.preparation.token_provenance_binder import (
    TokenProvenanceBinder,
)
from app.services.timeanchored_alignment.preparation.structure_protector import (
    StructureProtector,
)
from app.services.timeanchored_alignment.slow_window.contracts import ReadySlowWindow
from app.services.timeanchored_alignment.window_time_base_assembler import WindowTimeBasePackage


class AlignmentPreparationAssembler:
    """把选择层文本真相与 window 观测收口为 PreparationBundle。"""

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
        token_provenance_binder: TokenProvenanceBinder | None = None,
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
        self._token_provenance_binder = token_provenance_binder or TokenProvenanceBinder()
        self._hook_selector = hook_selector or HookSelector()
        self._language_profile_builder = language_profile_builder or LanguageProfileBuilder()
        self._pronunciation_hint_builder = pronunciation_hint_builder or PronunciationHintBuilder()

    def prepare(
        self,
        *,
        ready_window: ReadySlowWindow,
        window_time_base: WindowTimeBasePackage,
        selected_text_truth: SelectedTextTruth,
        default_language: str,
        whisper_result: Mapping[str, Any] | None = None,
        fallback_text: str = "",
        chunk_window: ChunkWindow | None = None,
        external_punct_track: PunctTrack | None = None,
        external_speaker_turns: Sequence[Any] | None = None,
    ) -> PreparationBundle:
        selected_payload = self._build_selected_text_payload(
            selected_text_truth=selected_text_truth,
            whisper_result=whisper_result,
            default_language=default_language,
            fallback_text=fallback_text,
        )
        normalized = self._slow_text_normalizer.normalize(
            whisper_result=selected_payload,
            default_language=str(selected_text_truth.language_hint or default_language or "auto"),
            fallback_text=str(selected_text_truth.text or fallback_text),
        )
        self._logger.debug(
            "AlignmentPreparation 开始 window_id={} owner_chunk_id={} source_chunk_count={} source_unit_count={} raw_unit_count={} word_unit_count={} selected_text_source={} normalized_text_len={} fallback_text_len={} timed_unit_count={}",
            ready_window.window_id,
            ready_window.owner_chunk_id,
            len(ready_window.source_chunk_ids),
            len(ready_window.source_units),
            len(window_time_base.raw_units),
            len(window_time_base.word_units),
            selected_text_truth.text_source,
            len(str(normalized.source_text or "")),
            len(str(selected_text_truth.text or fallback_text or "")),
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
        token_units = self._token_provenance_binder.bind(
            text=projection.window_text.text,
            token_units=tuple(pronunciation.package.token_units),
            source_units=ready_window.source_units,
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
        fast_hooks = self._hook_selector.select(window_time_base=window_time_base)
        slow_text = PreparedSlowText(
            window_text=projection.window_text,
            token_units=token_units,
            punctuation_evidences=punctuation_evidences,
            protected_units=projection.protected_units,
            language_runs=tuple(language_profile.package.runs),
            pronunciation_hints=pronunciation.hints,
        )
        canonical_sequence = self._build_canonical_sequence(
            slow_text=slow_text,
            protected_spans=compat_protected_spans,
            selected_text_truth=selected_text_truth,
        )
        pronunciation_graph = self._build_pronunciation_graph(
            token_units=token_units,
            language_runs=tuple(language_profile.package.runs),
            pronunciation_hints=pronunciation.hints,
        )
        acoustic_observation_pack = self._build_acoustic_observation_pack(
            canonical_sequence=canonical_sequence,
            window_time_base=window_time_base
        )
        scope = self._build_scope(
            ready_window=ready_window,
            window_time_base=window_time_base,
        )
        external_stable_facts = self._build_external_stable_facts(
            ready_window=ready_window,
            punctuation_evidences=punctuation_evidences,
            external_punct_track=external_punct_track,
            external_speaker_turns=external_speaker_turns,
        )
        provenance = self._build_provenance(
            selected_text_truth=selected_text_truth,
            window_time_base=window_time_base,
            external_stable_facts=external_stable_facts,
        )
        report = self._build_preparation_report(
            canonical_sequence=canonical_sequence,
            pronunciation_graph=pronunciation_graph,
            acoustic_observation_pack=acoustic_observation_pack,
            fast_hook_count=len(fast_hooks),
        )
        package = PreparationBundle(
            window_id=ready_window.window_id,
            owner_chunk_id=ready_window.owner_chunk_id,
            owner_chunk_index=ready_window.owner_chunk_index,
            source_chunk_ids=ready_window.source_chunk_ids,
            source_chunk_indices=ready_window.source_chunk_indices,
            slow_text=slow_text,
            fast_hooks=fast_hooks,
            coverage=ready_window.coverage,
            compat=compat,
            canonical_sequence=canonical_sequence,
            pronunciation_graph=pronunciation_graph,
            acoustic_observation_pack=acoustic_observation_pack,
            scope=scope,
            provenance=provenance,
            external_stable_facts=external_stable_facts,
            report=report,
            debug_refs={
                "window_id": str(ready_window.window_id),
                "owner_chunk_id": str(ready_window.owner_chunk_id),
                "selected_text_source": str(selected_text_truth.text_source),
                "observation_capability": str(acoustic_observation_pack.capability_level),
            },
        )
        self._logger.debug(
            "AlignmentPreparation 完成 window_id={} window_text_len={} token_unit_count={} punctuation_count={} lexical_punctuation_count={} external_punctuation_count={} protected_unit_count={} language_run_count={} pronunciation_hint_count={} fast_hook_count={} observation_slice_count={} compat_text_truth_unit_count={} compat_timed_unit_count={}",
            package.window_id,
            len(package.slow_text.window_text.text),
            len(package.slow_text.token_units),
            len(package.slow_text.punctuation_evidences),
            len(lexical_punctuation_evidences),
            len(external_punctuation_evidences),
            len(package.slow_text.protected_units),
            len(package.slow_text.language_runs),
            len(package.slow_text.pronunciation_hints),
            len(package.fast_hooks),
            len(package.acoustic_observation_pack.slices),
            len(package.compat.text_truth.units),
            self._count_timed_units(package.compat.text_truth.units),
        )
        return package

    @staticmethod
    def _build_selected_text_payload(
        *,
        selected_text_truth: SelectedTextTruth,
        whisper_result: Mapping[str, Any] | None,
        default_language: str,
        fallback_text: str,
    ) -> dict[str, Any]:
        payload = dict(whisper_result or {})
        raw_result = dict(payload.get("raw_result") or {})
        segments = tuple(raw_result.get("segments") or payload.get("segments") or ())
        raw_text = str(selected_text_truth.raw_text or selected_text_truth.text or "").strip()
        clean_text = str(selected_text_truth.text or fallback_text or raw_text).strip()
        payload["text"] = raw_text or clean_text
        payload["text_clean"] = clean_text
        payload["text_itn_raw"] = raw_text or clean_text
        payload["language"] = str(
            selected_text_truth.language_hint
            or payload.get("language")
            or default_language
            or "auto"
        )
        confidence = selected_text_truth.quality.get("confidence")
        if confidence is not None:
            payload["confidence"] = confidence
        payload["raw_result"] = {
            **raw_result,
            "segments": list(segments),
        }
        return payload

    @staticmethod
    def _build_canonical_sequence(
        *,
        slow_text: PreparedSlowText,
        protected_spans: tuple[ProtectedSpan, ...],
        selected_text_truth: SelectedTextTruth,
    ) -> CanonicalSequence:
        language_runs = tuple(slow_text.language_runs)
        tokens = tuple(
            CanonicalToken(
                token_id=str(token.unit_id),
                text=str(token.token_text),
                normalized_text=str(token.normalized_text),
                char_start=int(token.char_start),
                char_end=int(token.char_end),
                language=AlignmentPreparationAssembler._resolve_token_language(
                    char_start=int(token.char_start),
                    char_end=int(token.char_end),
                    language_runs=language_runs,
                    default_language=str(selected_text_truth.language_hint or "auto"),
                ),
                source_chunk_ids=tuple(str(item) for item in token.source_chunk_ids),
                source_chunk_indices=tuple(int(item) for item in token.source_chunk_indices),
                is_protected=AlignmentPreparationAssembler._token_overlaps_protected_span(
                    char_start=int(token.char_start),
                    char_end=int(token.char_end),
                    protected_spans=protected_spans,
                ),
                metadata={"source_unit_ids": list(token.source_unit_ids)},
            )
            for token in slow_text.token_units
        )
        return CanonicalSequence(
            original_text=str(slow_text.window_text.text),
            normalized_text=str(slow_text.window_text.text),
            tokens=tokens,
            protected_spans=tuple(protected_spans),
            language_runs=language_runs,
            frontend_version="preparation_assembler",
            language_hint=str(selected_text_truth.language_hint or "auto"),
            metadata={
                "display_text": str(slow_text.window_text.display_text),
                "text_source": str(selected_text_truth.text_source),
                "source_chunk_ids": list(selected_text_truth.source_chunk_ids),
            },
        )

    @staticmethod
    def _build_pronunciation_graph(
        *,
        token_units: Sequence[Any],
        language_runs: Sequence[Any],
        pronunciation_hints: Sequence[Any],
    ) -> PronunciationGraph:
        token_nodes: list[PronunciationTokenNode] = []
        state_nodes: list[PronunciationStateNode] = []
        edges: list[PronunciationEdge] = []
        for index, token in enumerate(token_units):
            language = AlignmentPreparationAssembler._resolve_token_language(
                char_start=int(getattr(token, "char_start", 0) or 0),
                char_end=int(getattr(token, "char_end", 0) or 0),
                language_runs=language_runs,
                default_language=str(getattr(token, "language", "") or "auto"),
            )
            variants = AlignmentPreparationAssembler._build_pronunciation_variants(
                token=token,
                pronunciation_hints=pronunciation_hints,
            )
            token_node_id = f"token-{index}"
            token_nodes.append(
                PronunciationTokenNode(
                    node_id=token_node_id,
                    token_index=index,
                    token_text=str(getattr(token, "token_text", "") or ""),
                    language=language,
                    variants=variants,
                    metadata={
                        "char_start": int(getattr(token, "char_start", 0) or 0),
                        "char_end": int(getattr(token, "char_end", 0) or 0),
                    },
                )
            )
            for state_index, variant in enumerate(variants):
                state_node_id = f"state-{index}-{state_index}"
                state_nodes.append(
                    PronunciationStateNode(
                        node_id=state_node_id,
                        token_index=index,
                        reading_key=variant.reading_key,
                        language=language,
                        state_index=state_index,
                    )
                )
                edges.append(
                    PronunciationEdge(
                        from_node_id=token_node_id,
                        to_node_id=state_node_id,
                        edge_kind="variant",
                    )
                )
                if index + 1 < len(token_units):
                    edges.append(
                        PronunciationEdge(
                            from_node_id=state_node_id,
                            to_node_id=f"token-{index + 1}",
                            edge_kind="advance",
                        )
                    )
        return PronunciationGraph(
            token_nodes=tuple(token_nodes),
            state_nodes=tuple(state_nodes),
            edges=tuple(edges),
            metadata={"token_count": len(token_nodes), "state_count": len(state_nodes)},
        )

    @staticmethod
    def _build_pronunciation_variants(
        *,
        token: Any,
        pronunciation_hints: Sequence[Any],
    ) -> tuple[PronunciationVariant, ...]:
        token_text = str(getattr(token, "normalized_text", None) or getattr(token, "token_text", "") or "")
        token_start = int(getattr(token, "char_start", 0) or 0)
        token_end = int(getattr(token, "char_end", 0) or 0)
        variants: list[PronunciationVariant] = []
        seen_keys: set[str] = set()
        for hint in pronunciation_hints:
            hint_start = int(getattr(hint, "char_start", 0) or 0)
            hint_end = int(getattr(hint, "char_end", 0) or 0)
            if hint_end <= token_start or hint_start >= token_end:
                continue
            reading_key = str(getattr(hint, "reading_key", "") or "").strip()
            if not reading_key or reading_key in seen_keys:
                continue
            seen_keys.add(reading_key)
            variants.append(
                PronunciationVariant(
                    reading_key=reading_key,
                    source=str(getattr(hint, "source", "") or "pronunciation_frontend"),
                )
            )
        if token_text and token_text not in seen_keys:
            variants.insert(
                0,
                PronunciationVariant(reading_key=token_text, source="token_text"),
            )
        return tuple(variants)

    @staticmethod
    def _build_acoustic_observation_pack(
        *,
        canonical_sequence: CanonicalSequence,
        window_time_base: WindowTimeBasePackage,
    ) -> AcousticObservationPack:
        char_units = AlignmentPreparationAssembler._explode_observation_chars(
            units=tuple(window_time_base.word_units or window_time_base.raw_units)
        )
        slices = AlignmentPreparationAssembler._project_observation_slices_to_canonical_tokens(
            canonical_sequence=canonical_sequence,
            char_units=char_units,
        )
        blank_track = tuple(
            float(item)
            for item in (window_time_base.metadata.get("blank_track", ()) or ())
        )
        if blank_track:
            capability_level = OBSERVATION_CAPABILITY_FRAME_POSTERIOR
        elif any(getattr(unit, "top_candidates", ()) for unit in window_time_base.word_units):
            capability_level = OBSERVATION_CAPABILITY_TOKEN_TOPK
        else:
            capability_level = OBSERVATION_CAPABILITY_TIMESTAMP_ONLY
        topk_limit = max(
            (len(getattr(unit, "top_candidates", ()) or ()) for unit in window_time_base.word_units),
            default=0,
        )
        return AcousticObservationPack(
            capability_level=capability_level,
            adapter_type=str(window_time_base.source or "sensevoice_window"),
            source_chunk_ids=tuple(str(item) for item in window_time_base.source_chunk_ids),
            source_chunk_indices=tuple(int(item) for item in window_time_base.source_chunk_indices),
            absolute_time_range=AlignmentPreparationAssembler._resolve_absolute_time_range(
                window_time_base=window_time_base
            ),
            slices=slices,
            quality=AcousticObservationQuality(
                slice_count=len(slices),
                blank_coverage=1.0 if blank_track else 0.0,
                topk_coverage=(
                    sum(
                        1
                        for unit in window_time_base.word_units
                        if getattr(unit, "top_candidates", ())
                    )
                    / max(len(window_time_base.word_units), 1)
                ),
                timestamp_coverage=1.0 if slices else 0.0,
                capability_degraded=False,
                notes=tuple() if capability_level != OBSERVATION_CAPABILITY_TIMESTAMP_ONLY else ("timestamp_only",),
            ),
            blank_track=blank_track,
            topk_limit=topk_limit or None,
            control_meta={
                "frame_stride": float(window_time_base.frame_stride),
                "raw_unit_count": len(window_time_base.raw_units),
                "word_unit_count": len(window_time_base.word_units),
            },
        )

    @staticmethod
    def _build_scope(
        *,
        ready_window: ReadySlowWindow,
        window_time_base: WindowTimeBasePackage,
    ) -> PreparationScope:
        return PreparationScope(
            window_id=str(ready_window.window_id),
            source_chunk_ids=tuple(str(item) for item in ready_window.source_chunk_ids),
            source_chunk_indices=tuple(int(item) for item in ready_window.source_chunk_indices),
            absolute_time_range=AlignmentPreparationAssembler._resolve_absolute_time_range(
                ready_window=ready_window,
                window_time_base=window_time_base,
            ),
        )

    @staticmethod
    def _build_external_stable_facts(
        *,
        ready_window: ReadySlowWindow,
        punctuation_evidences: Sequence[PunctuationEvidence],
        external_punct_track: PunctTrack | None,
        external_speaker_turns: Sequence[Any] | None = None,
    ) -> ExternalStableFacts:
        speaker_turn_rows: list[dict[str, Any]] = []
        if external_speaker_turns:
            for item in tuple(external_speaker_turns or ()):
                serialized = AlignmentPreparationAssembler._serialize_external_speaker_turn(item)
                if serialized is not None:
                    speaker_turn_rows.append(serialized)
        if speaker_turn_rows:
            speaker_turns = tuple(speaker_turn_rows)
            speaker_turn_source = "timeline_turns"
        else:
            speaker_turns = tuple(
                {
                    "unit_id": str(unit.unit_id),
                    "speaker_id": str(unit.speaker_id or ""),
                    "turn_id": str(unit.turn_id or ""),
                    "audio_start": float(unit.audio_start),
                    "audio_end": float(unit.audio_end),
                    "source_chunk_ids": list(unit.source_chunk_ids),
                    "source_chunk_indices": list(unit.source_chunk_indices),
                }
                for unit in ready_window.source_units
            )
            speaker_turn_source = "window_source_units"
        pause_facts = tuple(
            {
                "pause_start": float(left.audio_end),
                "pause_end": float(right.audio_start),
                "duration": max(float(right.audio_start) - float(left.audio_end), 0.0),
            }
            for left, right in zip(ready_window.source_units, ready_window.source_units[1:])
            if float(right.audio_start) > float(left.audio_end)
        )
        return ExternalStableFacts(
            punctuation_facts=tuple(punctuation_evidences),
            speaker_turns=speaker_turns,
            pause_facts=pause_facts,
            metadata={
                "has_external_punct_track": bool(
                    external_punct_track is not None
                    and getattr(external_punct_track, "positions", None)
                ),
                "source_unit_count": len(ready_window.source_units),
                "speaker_turn_source": speaker_turn_source,
            },
        )

    @staticmethod
    def _build_provenance(
        *,
        selected_text_truth: SelectedTextTruth,
        window_time_base: WindowTimeBasePackage,
        external_stable_facts: ExternalStableFacts,
    ) -> PreparationProvenance:
        stable_fact_sources = ["slow_text_punctuation"]
        if external_stable_facts.metadata.get("has_external_punct_track"):
            stable_fact_sources.append("punct_track")
        if external_stable_facts.speaker_turns:
            stable_fact_sources.append(
                str(external_stable_facts.metadata.get("speaker_turn_source") or "window_source_units")
            )
        return PreparationProvenance(
            text_source=str(selected_text_truth.text_source),
            observation_source=str(window_time_base.source or "sensevoice_window"),
            stable_fact_sources=tuple(stable_fact_sources),
            metadata={
                "selected_source_chunk_ids": list(selected_text_truth.source_chunk_ids),
            },
        )

    @staticmethod
    def _serialize_external_speaker_turn(item: Any) -> dict[str, Any] | None:
        start = float(getattr(item, "start", getattr(item, "audio_start", 0.0)) or 0.0)
        end = float(getattr(item, "end", getattr(item, "audio_end", start)) or start)
        if end <= start:
            return None
        source_chunk_ids = tuple(
            str(value)
            for value in (
                getattr(item, "source_chunk_ids", ())
                or getattr(item, "chunk_ids", ())
                or ()
            )
            if str(value).strip()
        )
        source_chunk_indices = tuple(
            int(value)
            for value in (
                getattr(item, "source_chunk_indices", ())
                or getattr(item, "chunk_indices", ())
                or ()
            )
        )
        return {
            "unit_id": str(getattr(item, "unit_id", getattr(item, "block_id", "")) or ""),
            "speaker_id": str(getattr(item, "speaker_id", "") or ""),
            "turn_id": str(getattr(item, "turn_id", getattr(item, "unit_id", "")) or ""),
            "audio_start": start,
            "audio_end": end,
            "source_chunk_ids": list(source_chunk_ids),
            "source_chunk_indices": list(source_chunk_indices),
            "boundary_confidence": float(getattr(item, "boundary_confidence", 0.0) or 0.0),
            "source": str(getattr(item, "source", "timeline_turns") or "timeline_turns"),
        }

    @staticmethod
    def _explode_observation_chars(
        *,
        units: Sequence[Any],
    ) -> tuple[dict[str, Any], ...]:
        exploded: list[dict[str, Any]] = []
        for unit_index, unit in enumerate(units):
            token_text = str(getattr(unit, "text", "") or "")
            if not token_text:
                continue
            unit_start = float(getattr(unit, "start", 0.0) or 0.0)
            unit_end = float(getattr(unit, "end", unit_start) or unit_start)
            token_count = max(len(token_text), 1)
            unit_duration = max(unit_end - unit_start, 0.0)
            for char_index, char in enumerate(token_text):
                char_start = unit_start + unit_duration * (char_index / token_count)
                char_end = unit_start + unit_duration * ((char_index + 1) / token_count)
                exploded.append(
                    {
                        "char": char,
                        "start": float(char_start),
                        "end": float(max(char_end, char_start)),
                        "confidence": float(getattr(unit, "confidence", 0.0) or 0.0),
                        "top_candidates": tuple(getattr(unit, "top_candidates", ()) or ()),
                        "token_type": str(getattr(unit, "token_type", "word") or "word"),
                        "unit_index": unit_index,
                    }
                )
        return tuple(exploded)

    @staticmethod
    def _project_observation_slices_to_canonical_tokens(
        *,
        canonical_sequence: CanonicalSequence,
        char_units: Sequence[dict[str, Any]],
    ) -> tuple[AcousticObservationSlice, ...]:
        slices: list[AcousticObservationSlice] = []
        cursor = 0
        current_time = float(char_units[0]["start"]) if char_units else 0.0
        for index, token in enumerate(canonical_sequence.tokens):
            token_text = str(token.text or "")
            token_chars = [char for char in token_text]
            matched: list[dict[str, Any]] = []
            if token_chars:
                if cursor + len(token_chars) <= len(char_units):
                    candidate = list(char_units[cursor: cursor + len(token_chars)])
                    if "".join(item["char"] for item in candidate) == token_text:
                        matched = candidate
                        cursor += len(token_chars)
                if not matched and token_text and not all(
                    unicodedata.category(char).startswith("P") for char in token_chars
                ):
                    joined = "".join(item["char"] for item in char_units[cursor:])
                    token_offset = joined.find(token_text)
                    if token_offset >= 0:
                        matched = list(
                            char_units[cursor + token_offset: cursor + token_offset + len(token_chars)]
                        )
                        cursor = cursor + token_offset + len(token_chars)
            if matched:
                current_time = float(matched[-1]["end"])
                top_candidates = tuple(
                    AcousticObservationTokenCandidate(
                        token=str(candidate.text),
                        score=float(candidate.score),
                        token_id=candidate.token_id,
                    )
                    for candidate in matched[0]["top_candidates"]
                )
                confidence = sum(float(item["confidence"]) for item in matched) / len(matched)
                metadata = {
                    "matched_char_count": len(matched),
                    "matched_unit_indices": [int(item["unit_index"]) for item in matched],
                    "token_type": str(matched[0]["token_type"]),
                }
                slices.append(
                    AcousticObservationSlice(
                        slice_id=f"slice-{index}",
                        start=float(matched[0]["start"]),
                        end=float(max(matched[-1]["end"], matched[0]["start"])),
                        primary_token=token_text,
                        top_candidates=top_candidates,
                        confidence=float(confidence),
                        metadata=metadata,
                    )
                )
                continue
            slices.append(
                AcousticObservationSlice(
                    slice_id=f"slice-{index}",
                    start=float(current_time),
                    end=float(current_time),
                    primary_token=token_text,
                    confidence=None,
                    metadata={"synthetic": True, "reason": "canonical_only_token"},
                )
            )
        return tuple(slices)

    @staticmethod
    def _build_preparation_report(
        *,
        canonical_sequence: CanonicalSequence,
        pronunciation_graph: PronunciationGraph,
        acoustic_observation_pack: AcousticObservationPack,
        fast_hook_count: int,
    ) -> PreparationReport:
        return PreparationReport(
            summary=LayerSummary(
                layer="preparation",
                counters={
                    "token_count": len(canonical_sequence.tokens),
                    "protected_span_count": len(canonical_sequence.protected_spans),
                    "language_run_count": len(canonical_sequence.language_runs),
                    "pronunciation_token_count": len(pronunciation_graph.token_nodes),
                    "observation_slice_count": len(acoustic_observation_pack.slices),
                },
            ),
            canonical_version=canonical_sequence.frontend_version,
            observation_capability=acoustic_observation_pack.capability_level,
            metadata={"fast_hook_count": int(fast_hook_count)},
        )

    @staticmethod
    def _resolve_absolute_time_range(
        *,
        ready_window: ReadySlowWindow | None = None,
        window_time_base: WindowTimeBasePackage | None = None,
    ) -> tuple[float, float]:
        if ready_window is not None and ready_window.audio_segments:
            return (
                float(ready_window.audio_segments[0][0]),
                float(ready_window.audio_segments[-1][1]),
            )
        units = tuple(
            getattr(window_time_base, "word_units", ()) or getattr(window_time_base, "raw_units", ()) or ()
        )
        if units:
            return (float(units[0].start), float(units[-1].end))
        return (0.0, 0.01)

    @staticmethod
    def _resolve_token_language(
        *,
        char_start: int,
        char_end: int,
        language_runs: Sequence[Any],
        default_language: str,
    ) -> str:
        for run in language_runs:
            run_start = int(getattr(run, "char_start", 0) or 0)
            run_end = int(getattr(run, "char_end", 0) or 0)
            if run_end <= char_start or run_start >= char_end:
                continue
            return str(getattr(run, "run_language", "") or default_language or "auto")
        return str(default_language or "auto")

    @staticmethod
    def _token_overlaps_protected_span(
        *,
        char_start: int,
        char_end: int,
        protected_spans: Sequence[ProtectedSpan],
    ) -> bool:
        for span in protected_spans:
            if int(span.end) <= char_start or int(span.start) >= char_end:
                continue
            return True
        return False

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
