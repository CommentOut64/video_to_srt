from __future__ import annotations

from app.pipelines.dual_pipeline.implementation import AsyncDualPipelineKernel
from app.services.model_runtime_config_service import ModelRuntimeConfigService


def test_runtime_defaults_contains_alignment_pipeline_group() -> None:
    defaults = ModelRuntimeConfigService._runtime_defaults()
    group = defaults.get("alignment_pipeline")

    assert isinstance(group, dict)
    assert group.get("version") == "timeanchored"
    assert group.get("mode") == "default"
    assert group.get("shadow_sample_rate") == 0.1
    assert group.get("write_debug_artifacts") is False


def test_alignment_pipeline_flags_respect_explicit_override() -> None:
    runtime = {
        "effective": {
            "alignment_pipeline": {
                "version": "timeanchored",
                "mode": "default",
                "shadow_sample_rate": 0.2,
                "write_debug_artifacts": False,
            },
            "alignment": {
                "dual_time_mode": "off",
            },
        },
        "override": {
            "alignment_pipeline": {
                "version": "timeanchored",
                "mode": "active",
                "shadow_sample_rate": 0.35,
                "write_debug_artifacts": True,
            },
        },
    }

    flags = AsyncDualPipelineKernel._resolve_alignment_pipeline_runtime_flags(
        runtime=runtime,
        legacy_dual_time_mode="off",
    )

    assert flags["version"] == "timeanchored"
    assert flags["mode"] == "active"
    assert flags["shadow_sample_rate"] == 0.35
    assert flags["write_debug_artifacts"] is True


def test_alignment_pipeline_flags_dual_time_shadow_shim_coerces_to_default() -> None:
    runtime = {
        "effective": {
            "alignment": {
                "dual_time_mode": "shadow",
            },
        },
        "override": {},
    }

    flags = AsyncDualPipelineKernel._resolve_alignment_pipeline_runtime_flags(
        runtime=runtime,
        legacy_dual_time_mode="off",
    )

    assert flags["version"] == "timeanchored"
    assert flags["mode"] == "default"


def test_alignment_pipeline_flags_timeanchored_version_promotes_default_mode() -> None:
    runtime = {
        "effective": {
            "alignment_pipeline": {
                "version": "timeanchored",
            },
            "alignment": {
                "dual_time_mode": "off",
            },
        },
        "override": {
            "alignment_pipeline": {
                "version": "timeanchored",
            },
        },
    }

    flags = AsyncDualPipelineKernel._resolve_alignment_pipeline_runtime_flags(
        runtime=runtime,
        legacy_dual_time_mode="off",
    )

    assert flags["version"] == "timeanchored"
    assert flags["mode"] == "default"


def test_alignment_pipeline_flags_invalid_values_fallback_safely() -> None:
    runtime = {
        "effective": {
            "alignment_pipeline": {
                "version": "unknown",
                "mode": "off",
                "shadow_sample_rate": "oops",
            },
            "alignment": {
                "dual_time_mode": "invalid",
            },
        },
        "override": {},
    }

    flags = AsyncDualPipelineKernel._resolve_alignment_pipeline_runtime_flags(
        runtime=runtime,
        legacy_dual_time_mode="off",
    )

    assert flags["version"] == "timeanchored"
    assert flags["mode"] == "default"
    assert flags["shadow_sample_rate"] == 0.1
    assert flags["write_debug_artifacts"] is False
