from __future__ import annotations

import json
from pathlib import Path

import yaml


CASE_ROOT = Path(__file__).resolve().parent / "p_tfa_cases"
MANIFEST_PATH = CASE_ROOT / "manifest.yaml"
REQUIRED_CASE_TYPES = {
    "duplicate_token",
    "one_to_many",
    "many_to_one",
    "heteronym",
    "dominant_island",
    "true_mixed",
    "protected_structure",
    "slow_hallucination",
}


def test_phase0_manifest_exists_and_contains_required_fields() -> None:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    cases = manifest["cases"]
    assert isinstance(cases, list)
    assert cases

    for case in cases:
        assert case["case_id"]
        assert case["case_type"]
        assert case["language_profile"]
        assert case["expected_route"]
        assert "expected_failure_semantic" in case
        assert case["data_ref"].endswith(".json")


def test_phase0_manifest_covers_required_case_types() -> None:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    observed_types = {case["case_type"] for case in manifest["cases"]}
    assert REQUIRED_CASE_TYPES <= observed_types


def test_phase0_case_payloads_exist_and_match_manifest() -> None:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        payload_path = CASE_ROOT / case["data_ref"]
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        assert payload["case_id"] == case["case_id"]
        assert payload["case_type"] == case["case_type"]
        assert payload["expected_route"] == case["expected_route"]
        assert payload["expected_failure_semantic"] == case["expected_failure_semantic"]
        assert isinstance(payload["selected_text"], str)
        assert isinstance(payload["observation_tokens"], list)
