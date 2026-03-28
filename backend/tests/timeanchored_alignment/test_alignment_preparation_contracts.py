from __future__ import annotations

from dataclasses import fields

from app.services.timeanchored_alignment.preparation.contracts import (
    AlignmentPreparationPackage,
    PreparedSlowText,
    PunctuationEvidence,
    SlowWindowTextPackage,
)


def test_alignment_preparation_package_contract_exposes_window_hook_and_coverage_fields() -> None:
    field_names = {field.name for field in fields(AlignmentPreparationPackage)}

    assert "window_id" in field_names
    assert "owner_chunk_id" in field_names
    assert "source_chunk_ids" in field_names
    assert "source_chunk_indices" in field_names
    assert "slow_text" in field_names
    assert "fast_hooks" in field_names
    assert "coverage" in field_names


def test_prepared_slow_text_contract_exposes_text_token_unit_and_fact_fields() -> None:
    field_names = {field.name for field in fields(PreparedSlowText)}

    assert "window_text" in field_names
    assert "token_units" in field_names
    assert "slots" not in field_names
    assert "punctuation_evidences" in field_names
    assert "protected_units" in field_names
    assert "language_runs" in field_names
    assert "pronunciation_hints" in field_names


def test_punctuation_evidence_source_char_index_targets_slow_window_text_package_text() -> None:
    window_text = SlowWindowTextPackage(text="你好世界", display_text="你好世界")
    evidence = PunctuationEvidence(mark="，", source_char_index=1)

    assert evidence.source_char_index == 1
    assert window_text.text[evidence.source_char_index] == "好"


def test_alignment_preparation_contracts_do_not_expose_fallback_punctuation_positions() -> None:
    package_fields = {field.name for field in fields(AlignmentPreparationPackage)}
    slow_text_fields = {field.name for field in fields(PreparedSlowText)}
    window_text_fields = {field.name for field in fields(SlowWindowTextPackage)}

    assert "fallback_punctuation_positions" not in package_fields
    assert "fallback_punctuation_positions" not in slow_text_fields
    assert "fallback_punctuation_positions" not in window_text_fields
