from __future__ import annotations

import json
from pathlib import Path

from app.pipelines.dual_pipeline.services.postprocess_trace_writer import (
    PostprocessTraceWriter,
)


def test_postprocess_trace_writer_writes_stage_file_and_manifest(tmp_path: Path) -> None:
    writer = PostprocessTraceWriter(
        logger=None,
        enabled=True,
        level="summary",
    )
    job_dir = tmp_path / "job-1"
    writer.write_stage(
        job_dir=job_dir,
        chunk_index=3,
        filename="10_preparation.input.json",
        payload={"window_id": "w1", "token_unit_count": 4},
        stage="preparation_input",
    )

    trace_file = job_dir / "debug" / "postprocess" / "chunk_0003" / "10_preparation.input.json"
    assert trace_file.exists()
    with trace_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["window_id"] == "w1"

    manifest_file = job_dir / "debug" / "postprocess" / "manifest.json"
    assert manifest_file.exists()
    with manifest_file.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert "3" in manifest["chunks"]
    chunk_entry = manifest["chunks"]["3"]
    assert "10_preparation.input.json" in chunk_entry["files"]
    assert chunk_entry["files"]["10_preparation.input.json"]["stage"] == "preparation_input"
