from __future__ import annotations

from pathlib import Path

from app.schemas.pipeline_context import ProcessingContext
from app.services.timeanchored_alignment.debug_alignment_inspector import (
    analyze_commit_strategies_from_job,
    analyze_commit_strategies_from_preparation,
    build_analysis_run_dir,
    inspect_trace_job,
    prepare_replay_run,
    recover_alignment_window_preparation_from_job,
    replay_alignment_window_from_job,
    replay_alignment_window_from_preparation,
)
from app.services.timeanchored_alignment.preparation.contracts import (
    AlignmentPreparationCompat,
    AlignmentPreparationPackage,
    FastHook,
    PreparedSlowText,
    PreparedTokenUnit,
    PunctuationEvidence,
    SlowWindowTextPackage,
)
from app.services.timeanchored_alignment.contracts import (
    LanguageRun,
    LanguageRunPackage,
    PronunciationPackage,
    TimeBasePackage,
    TextTruthPackage,
    TextTruthQuality,
    TextTruthUnit,
    TimeBaseQuality,
    TimeBaseUnit,
)
from app.services.timeanchored_alignment.slow_window.contracts import (
    WindowChunkBinding,
    WindowCoverage,
)
from app.services.timeanchored_alignment.window_time_base_assembler import WindowTimeBasePackage
from scripts.analyze_timeanchored_alignment import parse_args


def _fixture_job_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "jobs" / "p-20260329-233522-tr-test-en-1-hplj"


def _build_stub_job_source_contexts(seed: object) -> tuple[ProcessingContext, ...]:
    chunk_tokens = {
        7: [
            "Okay",
            "well",
            "I",
            "honestly",
            "can't",
            "see",
            "So",
            "I",
            "just",
            "freeze",
            "Is",
            "that",
            "your",
            "excuse",
            "for",
            "running",
            "into",
            "five",
            "people",
        ],
        8: [
            "No",
            "no",
            "that's",
            "different",
            "I",
            "actually",
            "couldn't",
            "see",
            "lame",
            "excuse",
            "I",
            "know",
        ],
    }

    contexts: list[ProcessingContext] = []
    ready_window = getattr(seed, "ready_window")
    for binding in ready_window.coverage.chunk_bindings:
        chunk_index = int(binding.chunk_index)
        tokens = chunk_tokens[chunk_index]
        units = tuple(
            TimeBaseUnit(
                text=token,
                start=idx * 0.22,
                end=idx * 0.22 + 0.18,
                confidence=0.92,
                token_type="word",
            )
            for idx, token in enumerate(tokens)
        )
        ctx = ProcessingContext(
            job_id="job-debug-inspector",
            chunk_index=chunk_index,
            audio_chunk=None,
        )
        ctx.time_base_chunk = TimeBasePackage(
            raw_units=units,
            word_units=units,
            quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.88, low_prob_ratio=0.05),
            language="en",
        )
        contexts.append(ctx)
    return tuple(contexts)


def _build_replay_preparation() -> AlignmentPreparationPackage:
    coverage = WindowCoverage(
        core_segments=((0.0, 0.9),),
        left_guard_sec=0.0,
        right_guard_sec=0.0,
        chunk_bindings=(
            WindowChunkBinding(
                chunk_id="chunk-1",
                chunk_index=1,
                chunk_start=0.0,
                chunk_end=0.9,
                overlap_ratio=1.0,
                role="owner",
                is_owner=True,
            ),
        ),
    )
    time_base = WindowTimeBasePackage(
        window_id="window-replay-001",
        language="en",
        raw_units=(
            TimeBaseUnit(text="alpha", start=0.0, end=0.2, confidence=0.9, token_type="word"),
            TimeBaseUnit(text="beta", start=0.3, end=0.5, confidence=0.9, token_type="word"),
            TimeBaseUnit(text="gamma", start=0.6, end=0.9, confidence=0.9, token_type="word"),
        ),
        word_units=(
            TimeBaseUnit(text="alpha", start=0.0, end=0.2, confidence=0.9, token_type="word"),
            TimeBaseUnit(text="beta", start=0.3, end=0.5, confidence=0.9, token_type="word"),
            TimeBaseUnit(text="gamma", start=0.6, end=0.9, confidence=0.9, token_type="word"),
        ),
        quality=TimeBaseQuality(blank_ratio=0.1, avg_max_prob=0.9, low_prob_ratio=0.05),
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        chunk_bindings=coverage.chunk_bindings,
    )
    return AlignmentPreparationPackage(
        window_id="window-replay-001",
        owner_chunk_id="chunk-1",
        owner_chunk_index=1,
        source_chunk_ids=("chunk-1",),
        source_chunk_indices=(1,),
        slow_text=PreparedSlowText(
            window_text=SlowWindowTextPackage(
                text="alphabeta.gamma",
                display_text="alpha beta. gamma",
                source_language="en",
            ),
            token_units=(
                PreparedTokenUnit(
                    unit_id="unit-0",
                    token_text="alpha",
                    normalized_text="alpha",
                    char_start=0,
                    char_end=5,
                    speaker_id="speaker-a",
                    turn_id="turn-a",
                    source_chunk_ids=("chunk-1",),
                    source_chunk_indices=(1,),
                ),
                PreparedTokenUnit(
                    unit_id="unit-1",
                    token_text="beta",
                    normalized_text="beta",
                    char_start=5,
                    char_end=9,
                    speaker_id="speaker-a",
                    turn_id="turn-a",
                    source_chunk_ids=("chunk-1",),
                    source_chunk_indices=(1,),
                ),
                PreparedTokenUnit(
                    unit_id="unit-2",
                    token_text="gamma",
                    normalized_text="gamma",
                    char_start=10,
                    char_end=15,
                    speaker_id="speaker-a",
                    turn_id="turn-a",
                    source_chunk_ids=("chunk-1",),
                    source_chunk_indices=(1,),
                ),
            ),
            punctuation_evidences=(PunctuationEvidence(mark=".", source_char_index=8),),
            protected_units=tuple(),
            language_runs=(
                LanguageRun(
                    run_text="alphabeta.gamma",
                    run_language="en",
                    char_start=0,
                    char_end=15,
                ),
            ),
            pronunciation_hints=tuple(),
        ),
        fast_hooks=(
            FastHook(
                hook_text="alpha",
                start=0.0,
                end=0.2,
                confidence=0.9,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
            ),
            FastHook(
                hook_text="beta",
                start=0.3,
                end=0.5,
                confidence=0.9,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
            ),
            FastHook(
                hook_text="gamma",
                start=0.6,
                end=0.9,
                confidence=0.9,
                source_chunk_id="chunk-1",
                source_chunk_index=1,
            ),
        ),
        coverage=coverage,
        compat=AlignmentPreparationCompat(
            time_base=time_base,
            text_truth=TextTruthPackage(
                raw_text="alpha beta gamma",
                normalized_text="alpha beta gamma",
                units=(
                    TextTruthUnit(
                        text="alpha",
                        normalized_text="alpha",
                        confidence=0.9,
                        language="en",
                        source="slow",
                    ),
                    TextTruthUnit(
                        text="beta",
                        normalized_text="beta",
                        confidence=0.9,
                        language="en",
                        source="slow",
                    ),
                    TextTruthUnit(
                        text="gamma",
                        normalized_text="gamma",
                        confidence=0.9,
                        language="en",
                        source="slow",
                    ),
                ),
                protected_spans=tuple(),
                quality=TextTruthQuality(
                    hallucination_risk=0.0,
                    repetition_ratio=0.0,
                    length_ratio=1.0,
                ),
                language="en",
                is_hallucination=False,
            ),
            protected_spans=tuple(),
            language_runs=LanguageRunPackage(
                runs=(
                    LanguageRun(
                        run_text="alpha beta gamma",
                        run_language="en",
                        char_start=0,
                        char_end=16,
                    ),
                ),
                dominant_language="en",
                window_kind="single_language",
                foreign_run_ratio=0.0,
                source_text="alpha beta gamma",
            ),
            pronunciation=PronunciationPackage(
                token_units=tuple(),
                phone_units=tuple(),
                token_to_phone_spans=tuple(),
                frontend_source="test",
                dependency_mode={},
                language="en",
            ),
            pronunciation_report={"source": "test"},
            chunk_window=type("ChunkWindow", (), {"chunk_ref": 1, "start": 0.0, "end": 0.9})(),
        ),
    )


def test_build_analysis_run_dir_uses_job_debug_alignment_inspector(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    run_dir = build_analysis_run_dir(job_dir=job_dir, run_label="demo")
    assert run_dir == job_dir / "debug" / "alignment_inspector" / "demo"


def test_parse_args_supports_inspect_and_replay_modes() -> None:
    inspect_args = parse_args(["inspect", "--job-dir", "F:/jobs/x"])
    assert inspect_args.mode == "inspect"
    assert inspect_args.job_dir == "F:/jobs/x"

    replay_args = parse_args(
        [
            "replay",
            "--job-dir",
            "F:/jobs/x",
            "--audio-path",
            "F:/audio.wav",
        ]
    )
    assert replay_args.mode == "replay"
    assert replay_args.audio_path == "F:/audio.wav"


def test_inspect_trace_job_summarizes_window_metrics() -> None:
    report = inspect_trace_job(job_dir=_fixture_job_dir())

    assert report["window_count"] == 21
    assert report["timeline_validity_counts"] == {
        "fatal": 10,
        "quarantined": 8,
        "repairable": 3,
    }
    assert report["mount_status_counts"] == {
        "anchored": 157,
        "merged": 137,
        "unresolved": 533,
        "inferred": 5,
    }
    assert report["candidate_flow_by_validity"]["fatal"]["avg_primary_candidate_count"] == 23.8
    assert report["candidate_flow_by_validity"]["fatal"]["avg_primary_over_seed_ratio"] == 0.388459


def test_inspect_trace_job_emits_hypothesis_evidence_sections() -> None:
    report = inspect_trace_job(job_dir=_fixture_job_dir())
    evidence = report["hypothesis_evidence"]

    assert set(evidence) == {
        "slow_text_unreliable",
        "anchor_mount_failed",
        "anchor_rejected",
        "chain_or_dp_failed",
    }
    assert evidence["anchor_mount_failed"]["window_count"] == 21
    assert evidence["chain_or_dp_failed"]["zero_gap_rescue_window_count"] == 21
    assert evidence["anchor_rejected"]["windows_with_primary_candidates_but_low_coverage"][0][
        "chunk_name"
    ] == "chunk_0036"
    assert evidence["chain_or_dp_failed"]["worst_graph_sparsity_windows"][0]["chunk_name"] == "chunk_0008"
    assert evidence["chain_or_dp_failed"]["avg_metrics_by_validity"]["fatal"]["edge_per_block"] == 1.88872


def test_inspect_trace_job_detects_decision_ingress_scope_mismatch() -> None:
    report = inspect_trace_job(job_dir=_fixture_job_dir())

    assert report["decision_ingress_scope_mismatch_count"] >= 1
    assert any(
        item["chunk_name"] == "chunk_0008"
        and item["span_selector"] == [0, 5]
        and item["full_window_span"] == [61.8, 76.62]
        for item in report["decision_ingress_scope_mismatches"]
    )


def test_prepare_replay_run_writes_request_under_alignment_inspector(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    result = prepare_replay_run(
        job_dir=job_dir,
        run_label="demo",
        audio_path=tmp_path / "clip.wav",
        clip_start=12.5,
        clip_end=18.0,
    )

    run_dir = job_dir / "debug" / "alignment_inspector" / "demo"
    request_path = run_dir / "request.json"

    assert result["status"] == "prepared"
    assert Path(result["run_dir"]) == run_dir
    assert Path(result["request_path"]) == request_path
    assert request_path.exists()
    assert not (job_dir / "debug" / "postprocess").exists()


def test_replay_alignment_window_from_preparation_returns_stage_artifacts() -> None:
    analysis = replay_alignment_window_from_preparation(
        preparation=_build_replay_preparation(),
        language="en",
    )

    assert analysis.stage_result.anchor_mount_result.metrics["seed_candidate_count"] >= 3
    assert analysis.stage_result.anchor_mount_result.metrics["block_count"] >= 2
    assert analysis.report["window_id"] == "window-replay-001"
    assert analysis.report["compatibility"]["reject_reason_counts"]["hard_boundary_crossed"] >= 1
    assert any(
        item["reason"] == "hard_boundary_crossed"
        for item in analysis.report["compatibility"]["pair_decisions"]
        if not item["compatible"]
    )


def test_recover_alignment_window_preparation_from_job_rebuilds_sw_000005() -> None:
    recovered = recover_alignment_window_preparation_from_job(
        job_dir=_fixture_job_dir(),
        window_id="sw-000005",
        source_context_builder=_build_stub_job_source_contexts,
    )

    assert recovered.preparation.window_id == "sw-000005"
    assert recovered.preparation.source_chunk_ids == ("chunk-7", "chunk-8")
    assert len(recovered.preparation.slow_text.token_units) == 31
    assert len(recovered.preparation.fast_hooks) == 31
    assert recovered.preparation.slow_text.token_units[18].turn_id.endswith("turn-0014")
    assert recovered.preparation.slow_text.token_units[19].turn_id.endswith("turn-0017")
    assert recovered.report["source_chunk_reports"][0]["chunk_id"] == "chunk-7"
    assert recovered.report["source_chunk_reports"][1]["chunk_id"] == "chunk-8"
    assert recovered.report["preparation_punctuation_evidence_count"] >= 10


def test_replay_alignment_window_from_job_returns_job_recovery_report() -> None:
    analysis = replay_alignment_window_from_job(
        job_dir=_fixture_job_dir(),
        window_id="sw-000005",
        source_context_builder=_build_stub_job_source_contexts,
    )

    assert analysis.report["window_id"] == "sw-000005"
    assert analysis.report["job_recovery"]["source_chunk_reports"][0]["turn_id"].endswith("turn-0014")
    assert analysis.report["job_recovery"]["source_chunk_reports"][1]["turn_id"].endswith("turn-0017")
    assert analysis.report["compatibility"]["reject_reason_counts"]["hard_boundary_crossed"] >= 1


def test_analyze_commit_strategies_from_preparation_reports_multi_island_recovery() -> None:
    recovered = recover_alignment_window_preparation_from_job(
        job_dir=_fixture_job_dir(),
        window_id="sw-000005",
        source_context_builder=_build_stub_job_source_contexts,
    )

    analysis = analyze_commit_strategies_from_preparation(
        preparation=recovered.preparation,
        language="en",
        multi_island_block_ids=("block-4", "block-5", "block-9", "block-10"),
    )

    baseline = analysis.report["baseline"]
    no_hard = analysis.report["no_hard_boundary"]
    multi_island = analysis.report["no_hard_boundary_multi_island"]

    assert baseline["metrics"]["coverage_ratio"] < no_hard["metrics"]["coverage_ratio"]
    assert no_hard["metrics"]["coverage_ratio"] < multi_island["metrics"]["coverage_ratio"]
    assert no_hard["plan"]["spans"][1]["span_kind"] == "fallback"
    assert no_hard["plan"]["spans"][1]["token_start"] == 19
    assert no_hard["plan"]["spans"][1]["token_end"] == 31
    assert any(
        span["route"] == "decision" and span["token_start"] == 21 and span["token_end"] == 23
        for span in multi_island["plan"]["spans"]
    )
    assert any(
        span["route"] == "decision" and span["token_start"] == 30 and span["token_end"] == 31
        for span in multi_island["plan"]["spans"]
    )


def test_analyze_commit_strategies_from_job_wraps_job_recovery_report() -> None:
    analysis = analyze_commit_strategies_from_job(
        job_dir=_fixture_job_dir(),
        window_id="sw-000005",
        source_context_builder=_build_stub_job_source_contexts,
        multi_island_block_ids=("block-4", "block-5", "block-9", "block-10"),
    )

    assert analysis.report["job_recovery"]["window_id"] == "sw-000005"
    strategies = analysis.report["strategies"]
    assert strategies["baseline"]["timeline_validity"] == "fatal"
    assert strategies["no_hard_boundary"]["timeline_validity"] == "repairable"
    assert strategies["no_hard_boundary_multi_island"]["plan"]["has_recoverable_spans"] is True


def test_analyze_commit_strategies_auto_multi_island_detects_chunk8_suffix_component() -> None:
    analysis = analyze_commit_strategies_from_job(
        job_dir=_fixture_job_dir(),
        window_id="sw-000005",
        source_context_builder=_build_stub_job_source_contexts,
        auto_multi_island=True,
    )

    strategies = analysis.report["strategies"]
    auto_meta = strategies["_analysis"]
    no_hard = strategies["no_hard_boundary"]
    auto_suffix = strategies["no_hard_boundary_auto_suffix_component"]

    assert auto_meta["auto_selected_multi_island_block_ids"] == [
        "block-4",
        "block-5",
        "block-9",
        "block-10",
    ]
    assert auto_meta["auto_selected_suffix_component"]["selected_turn_ids"] == [
        "timeline-p-20260329-233522-tr-test-en-1-hplj-turn-0017"
    ]
    assert auto_meta["auto_selected_suffix_component"]["selected_source_chunk_ids"] == ["chunk-8"]
    assert no_hard["metrics"]["coverage_ratio"] < auto_suffix["metrics"]["coverage_ratio"]
    assert auto_suffix["metrics"]["coverage_ratio"] == 0.8064516129032258


def test_analyze_commit_strategies_boundary_relaxations_show_turn_is_next_barrier() -> None:
    analysis = analyze_commit_strategies_from_job(
        job_dir=_fixture_job_dir(),
        window_id="sw-000005",
        source_context_builder=_build_stub_job_source_contexts,
        include_boundary_relaxations=True,
    )

    strategies = analysis.report["strategies"]
    no_hard = strategies["no_hard_boundary"]
    no_turn = strategies["no_hard_boundary_no_turn_boundary"]
    no_turn_or_speaker = strategies["no_hard_boundary_no_turn_or_speaker_boundary"]

    assert no_hard["compatibility"]["reject_reason_counts"]["turn_change_blocked"] == 7
    assert no_turn["compatibility"]["reject_reason_counts"].get("turn_change_blocked", 0) == 0
    assert no_turn["metrics"]["anchored_count"] == 29
    assert no_turn["metrics"]["coverage_ratio"] == 0.9354838709677419
    assert no_turn["plan"]["spans"][1]["span_kind"] == "fallback"
    assert no_turn["plan"]["spans"][1]["token_start"] == 19
    assert no_turn["plan"]["spans"][1]["token_end"] == 21
    assert no_turn_or_speaker["metrics"]["coverage_ratio"] == no_turn["metrics"]["coverage_ratio"]
