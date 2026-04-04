from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api.routes import debug_routes


def test_resolve_postprocess_file_allows_graph_and_payload_filenames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(debug_routes.config, "JOBS_DIR", tmp_path)
    base = tmp_path / "job1" / "debug" / "postprocess" / "chunk_0001"
    base.mkdir(parents=True, exist_ok=True)
    decoder_output = base / "21_alignment_decoder.output.json"
    payload = base / "60_output_dispatch.payload.json"
    decoder_output.write_text("{}", encoding="utf-8")
    payload.write_text("{}", encoding="utf-8")

    resolved_decoder_output = debug_routes._resolve_postprocess_file(
        identifier="job1",
        chunk_index=1,
        filename="21_alignment_decoder.output.json",
    )
    resolved_payload = debug_routes._resolve_postprocess_file(
        identifier="job1",
        chunk_index=1,
        filename="60_output_dispatch.payload.json",
    )
    assert resolved_decoder_output == decoder_output
    assert resolved_payload == payload


def test_resolve_postprocess_file_rejects_unsafe_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(debug_routes.config, "JOBS_DIR", tmp_path)
    with pytest.raises(HTTPException):
        debug_routes._resolve_postprocess_file(
            identifier="job1",
            chunk_index=1,
            filename="../escape.json",
        )
