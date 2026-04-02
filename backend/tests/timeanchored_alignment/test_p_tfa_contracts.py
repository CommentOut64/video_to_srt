from __future__ import annotations

from dataclasses import fields

from app.services.textflow.contracts import ChunkSentenceIndex, Sentence, SentenceRecord
from app.services.timeanchored_alignment.contracts import (
    ACOUSTIC_OBSERVATION_MIN_REQUIRED_FIELDS,
    OBSERVATION_CAPABILITY_FRAME_POSTERIOR,
    OBSERVATION_CAPABILITY_TIMESTAMP_ONLY,
    OBSERVATION_CAPABILITY_TOKEN_TOPK,
    AlignmentPath,
    AcousticObservationPack,
    LayerError,
    LayerSummary,
    LayerWarning,
    PhysicalChunk,
    PostprocessDebugConfig,
    PTFA_LAYER_OBJECT_REGISTRY,
)
from app.services.timeanchored_alignment.preparation.contracts import (
    CanonicalSequence,
    PreparationBundle,
)
from app.services.timeanchored_alignment.slow_window.contracts import SlowWindow


def test_phase0_freezes_observation_capability_levels() -> None:
    assert OBSERVATION_CAPABILITY_FRAME_POSTERIOR == "frame_posterior_capable"
    assert OBSERVATION_CAPABILITY_TOKEN_TOPK == "token_topk_capable"
    assert OBSERVATION_CAPABILITY_TIMESTAMP_ONLY == "timestamp_only_capable"


def test_phase0_freezes_layer_object_names() -> None:
    assert PTFA_LAYER_OBJECT_REGISTRY["selection"]["outputs"] == (
        "SelectedTextTruth",
        "SelectionDecision",
        "SelectionReport",
    )
    assert PTFA_LAYER_OBJECT_REGISTRY["preparation"]["outputs"] == (
        "CanonicalSequence",
        "PronunciationGraph",
        "AcousticObservationPack",
        "PreparationBundle",
        "PreparationReport",
    )
    assert PTFA_LAYER_OBJECT_REGISTRY["alignment"]["outputs"] == (
        "AlignmentPath",
        "BoundaryCandidates",
        "LowConfidenceSpans",
        "AlignmentReport",
    )
    assert PTFA_LAYER_OBJECT_REGISTRY["output"]["outputs"] == (
        "SentenceRecord",
        "ChunkSentenceIndex",
        "SubtitleBatchCompat",
        "OutputReport",
    )


def test_phase0_physical_chunk_contract_fields_are_frozen() -> None:
    field_names = {field.name for field in fields(PhysicalChunk)}
    assert field_names >= {
        "chunk_id",
        "chunk_index",
        "start",
        "end",
        "language_hint",
        "observation_ref",
        "metadata",
    }


def test_phase0_slow_window_contract_fields_are_frozen() -> None:
    field_names = {field.name for field in fields(SlowWindow)}
    assert field_names >= {
        "window_id",
        "source_chunk_ids",
        "source_chunk_indices",
        "source_units",
        "coverage",
        "flush_reason",
        "mode",
        "metadata",
    }


def test_phase0_preparation_bundle_contract_fields_are_frozen() -> None:
    field_names = {field.name for field in fields(PreparationBundle)}
    assert field_names >= {
        "canonical_sequence",
        "pronunciation_graph",
        "acoustic_observation_pack",
        "scope",
        "provenance",
        "external_stable_facts",
        "debug_refs",
    }


def test_phase0_canonical_sequence_contract_fields_are_frozen() -> None:
    field_names = {field.name for field in fields(CanonicalSequence)}
    assert field_names >= {
        "original_text",
        "normalized_text",
        "tokens",
        "protected_spans",
        "language_runs",
        "frontend_version",
    }


def test_phase0_acoustic_observation_pack_exposes_min_required_fields() -> None:
    field_names = {field.name for field in fields(AcousticObservationPack)}
    assert tuple(ACOUSTIC_OBSERVATION_MIN_REQUIRED_FIELDS) == (
        "capability_level",
        "adapter_type",
        "source_chunk_ids",
        "source_chunk_indices",
        "absolute_time_range",
        "slices",
        "quality",
    )
    for field_name in ACOUSTIC_OBSERVATION_MIN_REQUIRED_FIELDS:
        assert field_name in field_names


def test_phase0_alignment_path_contract_fields_are_frozen() -> None:
    field_names = {field.name for field in fields(AlignmentPath)}
    assert field_names >= {
        "path_id",
        "aligned_tokens",
        "route_confidence",
        "low_confidence_spans",
        "summary",
        "source_chunk_ids",
    }


def test_phase0_sentence_family_contracts_are_frozen() -> None:
    sentence_fields = {field.name for field in fields(Sentence)}
    sentence_record_fields = {field.name for field in fields(SentenceRecord)}
    chunk_index_fields = {field.name for field in fields(ChunkSentenceIndex)}

    assert sentence_fields >= {
        "sentence_id",
        "text",
        "start",
        "end",
        "token_span",
        "source_chunk_ids",
        "overlap_chunk_ids",
        "replace_scope_chunk_ids",
        "route",
        "trace",
    }
    assert sentence_record_fields >= {
        "sentence_id",
        "text",
        "start",
        "end",
        "source_chunk_ids",
        "overlap_chunk_ids",
        "replace_scope_chunk_ids",
        "route",
        "trace",
    }
    assert chunk_index_fields >= {"chunk_id", "sentence_ids"}


def test_phase0_summary_warning_error_schema_is_frozen() -> None:
    warning_fields = {field.name for field in fields(LayerWarning)}
    error_fields = {field.name for field in fields(LayerError)}
    summary_fields = {field.name for field in fields(LayerSummary)}

    assert warning_fields >= {"code", "message", "layer", "job_id", "window_id", "chunk_id", "details"}
    assert error_fields >= {"code", "message", "layer", "job_id", "window_id", "chunk_id", "details"}
    assert summary_fields >= {"layer", "status", "counters", "warnings", "errors", "debug_enabled"}


def test_phase0_debug_switch_contract_is_frozen() -> None:
    config = PostprocessDebugConfig(enabled=True, level="full")
    assert config.enabled is True
    assert config.level == "full"
