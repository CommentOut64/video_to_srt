from __future__ import annotations

from dataclasses import fields

from app.services.timeanchored_alignment.window_time_base_assembler import WindowTimeBasePackage


def test_window_time_base_package_exposes_source_and_binding_fields() -> None:
    field_names = {field.name for field in fields(WindowTimeBasePackage)}

    assert "source_chunk_ids" in field_names
    assert "source_chunk_indices" in field_names
    assert "chunk_bindings" in field_names
