from __future__ import annotations

import json
from pathlib import Path

from app.pipelines.dual_pipeline.services.postprocess_trace_writer import (
    PostprocessTraceWriter,
)
from app.services.timeanchored_alignment.contracts import LayerSummary


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


def test_postprocess_trace_writer_supports_empty_layer_summary_schema(tmp_path: Path) -> None:
    writer = PostprocessTraceWriter(
        logger=None,
        enabled=True,
        level="summary",
    )
    job_dir = tmp_path / "job-2"
    writer.write_layer_summary(
        job_dir=job_dir,
        layer_summary=LayerSummary(layer="preparation", status="ok"),
    )

    summary_file = (
        job_dir / "debug" / "postprocess" / "summaries" / "preparation.summary.json"
    )
    assert summary_file.exists()
    with summary_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["layer"] == "preparation"
    assert payload["warnings"] == []
    assert payload["errors"] == []

    manifest_file = job_dir / "debug" / "postprocess" / "manifest.json"
    with manifest_file.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert "preparation" in manifest["summaries"]


def test_postprocess_trace_writer_disabled_mode_does_not_create_files(tmp_path: Path) -> None:
    writer = PostprocessTraceWriter(
        logger=None,
        enabled=False,
        level="full",
    )
    job_dir = tmp_path / "job-disabled"
    writer.write_stage(
        job_dir=job_dir,
        chunk_index=1,
        filename="20_alignment_decoder.input.json",
        payload={"window_id": "w-disabled"},
        stage="alignment_decoder_input",
    )
    writer.write_layer_summary(
        job_dir=job_dir,
        layer_summary=LayerSummary(layer="alignment", status="ok"),
    )

    assert not (job_dir / "debug" / "postprocess").exists()
    assert not (job_dir / "debug" / "postprocess" / "manifest.json").exists()


def test_postprocess_trace_writer_summary_level_skips_full_only_payload(tmp_path: Path) -> None:
    writer = PostprocessTraceWriter(
        logger=None,
        enabled=True,
        level="summary",
    )
    job_dir = tmp_path / "job-summary"
    writer.write_stage(
        job_dir=job_dir,
        chunk_index=2,
        filename="20_alignment_decoder.input.json",
        payload={"window_id": "w-summary"},
        stage="alignment_decoder_input",
        full_only=True,
    )
    writer.write_stage(
        job_dir=job_dir,
        chunk_index=2,
        filename="21_alignment_decoder.output.json",
        payload={"window_id": "w-summary", "status": "ok"},
        stage="alignment_decoder_output",
    )

    full_only_file = (
        job_dir / "debug" / "postprocess" / "chunk_0002" / "20_alignment_decoder.input.json"
    )
    kept_file = (
        job_dir / "debug" / "postprocess" / "chunk_0002" / "21_alignment_decoder.output.json"
    )
    manifest_file = job_dir / "debug" / "postprocess" / "manifest.json"

    assert not full_only_file.exists()
    assert kept_file.exists()

    with manifest_file.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    chunk_entry = manifest["chunks"]["2"]["files"]
    assert "20_alignment_decoder.input.json" not in chunk_entry
    assert chunk_entry["21_alignment_decoder.output.json"]["stage"] == "alignment_decoder_output"
