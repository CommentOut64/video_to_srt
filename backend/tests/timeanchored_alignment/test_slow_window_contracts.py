from __future__ import annotations

from dataclasses import fields

from app.services.timeanchored_alignment.slow_window.contracts import ReadySlowWindow


def test_ready_slow_window_contract_exposes_owner_source_and_coverage_fields() -> None:
    field_names = {field.name for field in fields(ReadySlowWindow)}

    assert "owner_chunk_id" in field_names
    assert "owner_chunk_index" in field_names
    assert "source_units" in field_names
    assert "coverage" in field_names
    assert "source_chunk_ids" in field_names
    assert "source_chunk_indices" in field_names
