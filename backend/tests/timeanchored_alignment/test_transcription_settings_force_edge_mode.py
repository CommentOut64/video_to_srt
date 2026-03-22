from __future__ import annotations

from app.models.job_models import JobSettings, TranscriptionConfig


def test_transcription_settings_api_default_edge_selection_mode_is_auto() -> None:
    payload = TranscriptionConfig()
    assert payload.edge_selection_mode == "auto"


def test_job_settings_accepts_force_edge_selection_mode() -> None:
    settings = JobSettings.from_dict(
        {
            "transcription": {
                "edge_selection_mode": "force_fast",
            }
        }
    )

    assert settings.transcription.edge_selection_mode == "force_fast"
    assert settings.to_dict()["transcription"]["edge_selection_mode"] == "force_fast"


def test_job_settings_invalid_edge_selection_mode_falls_back_to_auto() -> None:
    settings = JobSettings.from_dict(
        {
            "transcription": {
                "edge_selection_mode": "invalid_mode",
            }
        }
    )

    assert settings.transcription.edge_selection_mode == "auto"
