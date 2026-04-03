from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from app.services.timeanchored_alignment.contracts import (
    SelectedTextTruth,
    TimeBaseQuality,
    TimeBaseUnit,
)
from app.services.timeanchored_alignment.decoder.service import AlignmentDecoderService
from app.services.timeanchored_alignment.preparation.assembler import AlignmentPreparationAssembler
from app.services.timeanchored_alignment.slow_window.contracts import (
    DialogueShapeSnapshot,
    PromptSeed,
    ReadySlowWindow,
    WindowBatchHint,
    WindowChunkBinding,
    WindowCoverage,
    WindowLanguageProfile,
    WindowSourceUnit,
)
from app.services.timeanchored_alignment.window_time_base_assembler import WindowTimeBasePackage


CASE_ROOT = Path(__file__).resolve().parent / "p_tfa_cases"
MANIFEST_PATH = CASE_ROOT / "manifest.yaml"


def _build_preparation_from_case(case_payload: dict[str, object]):
    observation_tokens = list(case_payload["observation_tokens"])
    chunk_ids = tuple(f"chunk-{index}" for index in range(len(observation_tokens))) or ("chunk-0",)
    chunk_indices = tuple(range(len(observation_tokens))) or (0,)
    language_profile = str(case_payload["language_profile"])
    dominant_language = "en" if language_profile == "en" else "zh"
    bindings = []
    source_units = []
    time_units = []
    cursor = 0.0
    for index, token in enumerate(observation_tokens):
        token_text = str(token)
        start = float(cursor)
        end = float(start + max(0.2, 0.08 * len(token_text)))
        bindings.append(
            WindowChunkBinding(
                chunk_id=chunk_ids[index],
                chunk_index=chunk_indices[index],
                chunk_start=start,
                chunk_end=end,
                overlap_ratio=1.0,
                role="owner" if index == 0 else "core",
                is_owner=index == 0,
            )
        )
        source_units.append(
            WindowSourceUnit(
                unit_id=f"unit-{index}",
                semantic_chunk_id=f"sem-{index}",
                text=token_text,
                audio_start=start,
                audio_end=end,
                source_chunk_ids=(chunk_ids[index],),
                source_chunk_indices=(chunk_indices[index],),
                speaker_id="speaker-a",
                turn_id="turn-a",
                language=dominant_language,
                arrived_at=end,
            )
        )
        time_units.append(
            TimeBaseUnit(
                text=token_text,
                start=start,
                end=end,
                confidence=0.95,
                token_type="word",
            )
        )
        cursor = end

    language_hint = "mixed" if language_profile.startswith("true_mixed") else "zh"
    if language_profile == "en":
        language_hint = "en"
    ready_window = ReadySlowWindow(
        window_id=str(case_payload["case_id"]),
        owner_chunk_id=chunk_ids[0],
        owner_chunk_index=chunk_indices[0],
        window_mode="steady",
        flush_reason="phase3_eval",
        audio_segments=tuple((unit.audio_start, unit.audio_end) for unit in source_units),
        coverage=WindowCoverage(
            core_segments=((0.0, float(cursor)),),
            left_guard_sec=0.0,
            right_guard_sec=0.0,
            chunk_bindings=tuple(bindings),
        ),
        source_semantic_chunk_ids=tuple(f"sem-{index}" for index in range(len(observation_tokens))),
        source_chunk_ids=chunk_ids,
        source_chunk_indices=chunk_indices,
        source_units=tuple(source_units),
        dialogue_shape=DialogueShapeSnapshot(
            shape="single_party",
            speaker_count=1,
            dominant_speaker_id="speaker-a",
            dominant_speaker_ratio=1.0,
            speaker_switch_count=0,
            speaker_switch_density=0.0,
            turn_count=1,
            avg_turn_duration_sec=max(float(cursor), 0.1),
        ),
        language_profile=WindowLanguageProfile(
            primary_language=dominant_language,
            language_mix_state="true_mixed" if language_hint == "mixed" else "single_language",
            decision_domains=("timeanchored_alignment",),
            should_bypass_whisper=False,
        ),
        prompt_seed=PromptSeed(text=str(case_payload["selected_text"])),
        batch_hint=WindowBatchHint(
            duration_bucket="short",
            token_estimate=len(observation_tokens),
            acoustic_density_hint="medium",
            queue_priority=1,
        ),
        created_at=float(cursor),
    )
    window_time_base = WindowTimeBasePackage(
        window_id=ready_window.window_id,
        language=dominant_language,
        raw_units=tuple(time_units),
        word_units=tuple(time_units),
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.9, low_prob_ratio=0.05),
        source_chunk_ids=chunk_ids,
        source_chunk_indices=chunk_indices,
        chunk_bindings=tuple(bindings),
    )
    selected_source = "fast" if str(case_payload["expected_route"]) == "selection_reject_slow" else "slow"
    selected_text_truth = SelectedTextTruth(
        text=str(case_payload["selected_text"]),
        text_source=selected_source,
        language_hint=language_hint,
        source_chunk_ids=chunk_ids,
        quality={"confidence": 0.9},
        metadata={"raw_text": str(case_payload["selected_text"])},
    )
    preparation = AlignmentPreparationAssembler().prepare(
        ready_window=ready_window,
        window_time_base=window_time_base,
        selected_text_truth=selected_text_truth,
        whisper_result={
            "text": str(case_payload["selected_text"]),
            "text_clean": str(case_payload["selected_text"]),
            "text_itn_raw": str(case_payload["selected_text"]),
            "confidence": 0.9,
            "language": dominant_language,
            "raw_result": {"segments": [{"avg_logprob": -0.1}]},
        },
        default_language=dominant_language,
    )
    ambiguous_indices = {
        int(index)
        for index in case_payload.get("ambiguous_token_indices", [])
    }
    if not ambiguous_indices:
        return preparation
    token_nodes = []
    for index, node in enumerate(preparation.pronunciation_graph.token_nodes):
        metadata = dict(node.metadata or {})
        if index in ambiguous_indices:
            metadata["pronunciation_ambiguous"] = True
        token_nodes.append(replace(node, metadata=metadata))
    return replace(
        preparation,
        pronunciation_graph=replace(
            preparation.pronunciation_graph,
            token_nodes=tuple(token_nodes),
        ),
    )


def _load_manifest_cases() -> list[dict[str, object]]:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    return list(manifest["cases"])


@pytest.mark.parametrize("manifest_entry", _load_manifest_cases(), ids=lambda item: str(item["case_id"]))
def test_phase3_decoder_matches_manifest_routes(manifest_entry: dict[str, object]) -> None:
    case_payload = json.loads((CASE_ROOT / str(manifest_entry["data_ref"])).read_text(encoding="utf-8"))
    preparation = _build_preparation_from_case(case_payload)

    result = AlignmentDecoderService().execute(preparation=preparation)

    assert result.alignment_report.route == str(case_payload["expected_route"])
    assert result.alignment_report.failure_semantic == str(case_payload["expected_failure_semantic"])
