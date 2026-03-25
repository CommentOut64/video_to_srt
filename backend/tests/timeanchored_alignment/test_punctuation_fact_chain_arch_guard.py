from __future__ import annotations

from pathlib import Path


def test_arch_guard_no_legacy_finalize_wrapper_usage() -> None:
    root = Path(__file__).resolve().parents[2]
    alignment_stage_code = (
        root / "app/pipelines/dual_pipeline/services/alignment_stage_service.py"
    ).read_text(encoding="utf-8")
    assert "host._finalize_timeanchored_stream(" not in alignment_stage_code

    implementation_code = (
        root / "app/pipelines/dual_pipeline/implementation.py"
    ).read_text(encoding="utf-8")
    assert "def _finalize_timeanchored_stream(" not in implementation_code

    facade_code = (
        root / "app/pipelines/dual_pipeline/services/textflow_facade_service.py"
    ).read_text(encoding="utf-8")
    assert "def finalize_timeanchored_stream(" not in facade_code


def test_arch_guard_preparation_accepts_external_punct_track() -> None:
    target = Path(__file__).resolve().parents[2] / "app/services/timeanchored_alignment/preparation/assembler.py"
    code = target.read_text(encoding="utf-8")
    assert "external_punct_track" in code
