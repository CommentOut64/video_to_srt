"""Timeanchored 对齐层独立分析工具。"""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from app.services.language_policy import build_language_policy_snapshot
from app.services.timeanchored_alignment.anchor_mount.chain_solver import (
    ChainSolver,
    CompatibilityDecision,
)
from app.services.timeanchored_alignment.anchor_mount.core_pipeline import AnchorMountCorePipeline
from app.services.timeanchored_alignment.anchor_mount.ingress_validator import IngressValidator
from app.services.timeanchored_alignment.anchor_mount.service import (
    AnchorMountAlignmentService,
    AnchorMountStageResult,
)
from app.services.timeanchored_alignment.preparation.assembler import (
    AlignmentPreparationAssembler,
)
from app.services.timeanchored_alignment.preparation.contracts import AlignmentPreparationPackage
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
from app.services.timeanchored_alignment.window_time_base_assembler import (
    WindowTimeBaseAssembler,
    WindowTimeBasePackage,
)
from app.services.timeanchored_alignment.window_recovery_planner import (
    WindowRecoveryPlanner,
)


@dataclass(frozen=True)
class TraceWindowSummary:
    chunk_name: str
    window_id: str
    timeline_validity: str
    unit_count: int
    anchored_count: int
    coverage_ratio: float
    hook_waste_ratio: float
    seed_candidate_count: int
    masked_candidate_count: int
    primary_candidate_count: int
    block_candidate_count: int
    block_count: int
    compatibility_edge_count: int
    avg_in_degree: float
    largest_unresolved_span: int
    unresolved_count: int
    gap_rescue_match_count: int
    duplicate_candidate_hook_count: int


@dataclass(frozen=True)
class ReplayWindowAnalysis:
    stage_result: AnchorMountStageResult
    report: dict[str, Any]


@dataclass(frozen=True)
class ReplayJobWindowSeed:
    job_dir: Path
    trace_dir: Path
    window_id: str
    language: str
    source_text: str
    ready_window: ReadySlowWindow
    source_chunk_reports: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RecoveredWindowPreparation:
    ready_window: ReadySlowWindow
    window_time_base: WindowTimeBasePackage
    preparation: AlignmentPreparationPackage
    report: dict[str, Any]


@dataclass(frozen=True)
class CommitStrategyAnalysis:
    report: dict[str, Any]


def build_analysis_run_dir(*, job_dir: Path, run_label: str) -> Path:
    """为独立分析运行构造 sandbox 目录。"""
    normalized = str(run_label or "").strip() or "run"
    return Path(job_dir) / "debug" / "alignment_inspector" / normalized


def prepare_replay_run(
    *,
    job_dir: Path,
    run_label: str,
    audio_path: Path | None,
    clip_start: float | None = None,
    clip_end: float | None = None,
) -> dict[str, Any]:
    """
    为后续 replay 准备独立 sandbox。

    当前只负责隔离目录与请求落盘，不写入主任务默认 debug/postprocess。
    """

    run_dir = build_analysis_run_dir(job_dir=job_dir, run_label=run_label)
    run_dir.mkdir(parents=True, exist_ok=True)
    request_payload = {
        "mode": "replay",
        "job_dir": str(Path(job_dir)),
        "audio_path": str(audio_path) if audio_path is not None else None,
        "clip_start": None if clip_start is None else float(clip_start),
        "clip_end": None if clip_end is None else float(clip_end),
        "status": "prepared",
        "note": "sandbox 已建立；真实音频重放链待接入独立分析入口",
    }
    request_path = run_dir / "request.json"
    request_path.write_text(
        json.dumps(request_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "status": "prepared",
        "run_dir": str(run_dir),
        "request_path": str(request_path),
        "request": request_payload,
    }


def replay_alignment_window_from_preparation(
    *,
    preparation: AlignmentPreparationPackage,
    language: str,
    policy_snapshot: Any | None = None,
) -> ReplayWindowAnalysis:
    resolved_policy = policy_snapshot
    if resolved_policy is None:
        try:
            resolved_policy = build_language_policy_snapshot(language_hint=language)
        except Exception:
            resolved_policy = None

    ingress_validator = IngressValidator()
    chain_solver = ChainSolver()
    input_view = ingress_validator.validate(
        preparation=preparation,
        language=language,
        policy_snapshot=resolved_policy,
    )
    state = AnchorMountCorePipeline(chain_solver=chain_solver).run(input_view)
    stage_result = AnchorMountAlignmentService(chain_solver=chain_solver).align(
        preparation=preparation,
        language=language,
        policy_snapshot=resolved_policy,
    )
    return ReplayWindowAnalysis(
        stage_result=stage_result,
        report={
            "window_id": preparation.window_id,
            "owner_chunk_id": preparation.owner_chunk_id,
            "owner_chunk_index": preparation.owner_chunk_index,
            "token_unit_count": len(preparation.slow_text.token_units),
            "fast_hook_count": len(preparation.fast_hooks),
            "timeline_validity": stage_result.anchor_mount_result.timeline_validity,
            "should_fallback": bool(stage_result.anchor_mount_result.should_fallback),
            "compatibility": _build_compatibility_report(
                input_view=input_view,
                blocks=tuple(state.local_blocks),
                chain_solver=chain_solver,
            ),
        },
    )


def recover_alignment_window_preparation_from_job(
    *,
    job_dir: Path,
    window_id: str,
    language: str | None = None,
    source_context_builder: Callable[[ReplayJobWindowSeed], tuple[Any, ...]] | None = None,
) -> RecoveredWindowPreparation:
    seed = _recover_job_window_seed(
        job_dir=Path(job_dir),
        window_id=window_id,
        language=language,
    )
    builder = source_context_builder or _build_source_contexts_from_job_audio
    source_contexts = tuple(builder(seed))
    if not source_contexts:
        raise ValueError(f"窗口 {window_id} 未恢复到任何 source context，无法组装 time base")

    window_time_base = WindowTimeBaseAssembler().assemble(
        ready_window=seed.ready_window,
        source_contexts=source_contexts,
    )
    preparation = AlignmentPreparationAssembler().prepare(
        ready_window=seed.ready_window,
        window_time_base=window_time_base,
        whisper_result={
            "text": seed.source_text,
            "text_clean": seed.source_text,
            "min_clean_text": seed.source_text,
            "language": seed.language,
            "confidence": 1.0,
        },
        default_language=seed.language,
        fallback_text=seed.source_text,
    )
    return RecoveredWindowPreparation(
        ready_window=seed.ready_window,
        window_time_base=window_time_base,
        preparation=preparation,
        report={
            "window_id": seed.window_id,
            "trace_dir": str(seed.trace_dir),
            "language": seed.language,
            "source_text_len": len(seed.source_text),
            "source_chunk_reports": list(seed.source_chunk_reports),
            "source_context_count": len(source_contexts),
            "time_base_raw_unit_count": len(window_time_base.raw_units),
            "time_base_word_unit_count": len(window_time_base.word_units),
            "preparation_token_unit_count": len(preparation.slow_text.token_units),
            "preparation_hook_count": len(preparation.fast_hooks),
            "preparation_punctuation_evidence_count": len(preparation.slow_text.punctuation_evidences),
        },
    )


def replay_alignment_window_from_job(
    *,
    job_dir: Path,
    window_id: str,
    language: str | None = None,
    source_context_builder: Callable[[ReplayJobWindowSeed], tuple[Any, ...]] | None = None,
    policy_snapshot: Any | None = None,
) -> ReplayWindowAnalysis:
    recovered = recover_alignment_window_preparation_from_job(
        job_dir=job_dir,
        window_id=window_id,
        language=language,
        source_context_builder=source_context_builder,
    )
    analysis = replay_alignment_window_from_preparation(
        preparation=recovered.preparation,
        language=str(language or recovered.report["language"] or "auto"),
        policy_snapshot=policy_snapshot,
    )
    combined_report = dict(analysis.report)
    combined_report["job_recovery"] = recovered.report
    return ReplayWindowAnalysis(
        stage_result=analysis.stage_result,
        report=combined_report,
    )


def analyze_commit_strategies_from_preparation(
    *,
    preparation: AlignmentPreparationPackage,
    language: str,
    multi_island_block_ids: tuple[str, ...] = (),
    auto_multi_island: bool = False,
    include_boundary_relaxations: bool = False,
    policy_snapshot: Any | None = None,
) -> CommitStrategyAnalysis:
    resolved_policy = policy_snapshot
    if resolved_policy is None:
        try:
            resolved_policy = build_language_policy_snapshot(language_hint=language)
        except Exception:
            resolved_policy = None

    auto_selected_block_ids: tuple[str, ...] = tuple()
    auto_selected_component: dict[str, Any] | None = None
    no_hard_boundary_report: dict[str, Any] | None = None
    strategies: dict[str, ChainSolver] = {
        "baseline": ChainSolver(),
        "no_hard_boundary": _NoHardBoundaryChainSolver(),
    }
    if include_boundary_relaxations:
        strategies["no_hard_boundary_no_turn_boundary"] = _BoundaryRelaxingChainSolver(
            ignore_turn_boundary=True,
        )
        strategies["no_hard_boundary_no_turn_or_speaker_boundary"] = _BoundaryRelaxingChainSolver(
            ignore_turn_boundary=True,
            ignore_speaker_boundary=True,
        )
    if multi_island_block_ids:
        strategies["no_hard_boundary_multi_island"] = _ForcedCommitChainSolver(
            forced_block_ids=multi_island_block_ids,
            ignore_hard_boundaries=True,
        )
    elif auto_multi_island:
        no_hard_boundary_report = _analyze_commit_strategy(
            preparation=preparation,
            language=language,
            chain_solver=_NoHardBoundaryChainSolver(),
            policy_snapshot=resolved_policy,
        )
        selection = (
            dict(no_hard_boundary_report.get("solver_graph", {}) or {}).get("suffix_component_selection")
            or {}
        )
        auto_selected_component = dict(selection)
        auto_selected_block_ids = tuple(
            str(item) for item in tuple(selection.get("selected_block_ids", ()) or ()) if str(item)
        )
        if auto_selected_block_ids:
            strategies["no_hard_boundary_auto_suffix_component"] = _ForcedCommitChainSolver(
                forced_block_ids=auto_selected_block_ids,
                ignore_hard_boundaries=True,
            )

    report: dict[str, Any] = {}
    for name, chain_solver in strategies.items():
        if name == "no_hard_boundary" and no_hard_boundary_report is not None:
            report[name] = dict(no_hard_boundary_report)
            continue
        report[name] = _analyze_commit_strategy(
            preparation=preparation,
            language=language,
            chain_solver=chain_solver,
            policy_snapshot=resolved_policy,
        )
    if auto_selected_block_ids:
        report["_analysis"] = {
            "auto_selected_multi_island_block_ids": list(auto_selected_block_ids),
            "auto_selected_suffix_component": auto_selected_component or {},
        }
    return CommitStrategyAnalysis(report=report)


def analyze_commit_strategies_from_job(
    *,
    job_dir: Path,
    window_id: str,
    language: str | None = None,
    source_context_builder: Callable[[ReplayJobWindowSeed], tuple[Any, ...]] | None = None,
    multi_island_block_ids: tuple[str, ...] = (),
    auto_multi_island: bool = False,
    include_boundary_relaxations: bool = False,
    policy_snapshot: Any | None = None,
) -> CommitStrategyAnalysis:
    recovered = recover_alignment_window_preparation_from_job(
        job_dir=job_dir,
        window_id=window_id,
        language=language,
        source_context_builder=source_context_builder,
    )
    analysis = analyze_commit_strategies_from_preparation(
        preparation=recovered.preparation,
        language=str(language or recovered.report["language"] or "auto"),
        multi_island_block_ids=multi_island_block_ids,
        auto_multi_island=auto_multi_island,
        include_boundary_relaxations=include_boundary_relaxations,
        policy_snapshot=policy_snapshot,
    )
    combined = {
        "job_recovery": recovered.report,
        "strategies": analysis.report,
    }
    return CommitStrategyAnalysis(report=combined)


def inspect_trace_job(*, job_dir: Path) -> dict[str, Any]:
    """读取现有 postprocess trace 并输出结构化取证报告。"""
    job_dir = Path(job_dir)
    postprocess_dir = job_dir / "debug" / "postprocess"
    if not postprocess_dir.exists():
        raise FileNotFoundError(f"未找到 postprocess trace 目录: {postprocess_dir}")

    window_summaries: list[TraceWindowSummary] = []
    mount_status_counts: Counter[str] = Counter()
    validity_counts: Counter[str] = Counter()
    validity_reason_counts: Counter[str] = Counter()
    punctuation_debug_chunk_count = 0
    if (job_dir / "debug" / "punctuation.jsonl").exists():
        punctuation_debug_chunk_count = sum(
            1
            for line in (job_dir / "debug" / "punctuation.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        )

    for graph_path in sorted(postprocess_dir.glob("chunk_*/21_anchor_mount.graph.json")):
        graph = _load_json(graph_path)
        metrics = dict(graph.get("metrics") or {})
        timeline_validity = str(metrics.get("timeline_validity") or "unknown")
        validity_counts[timeline_validity] += 1
        for reason in metrics.get("timeline_validity_reasons") or ():
            validity_reason_counts[str(reason)] += 1
        for mount in graph.get("mounts") or ():
            mount_status_counts[str(mount.get("mount_status") or "unknown")] += 1

        window_summaries.append(
            TraceWindowSummary(
                chunk_name=graph_path.parent.name,
                window_id=str(graph.get("window_id") or ""),
                timeline_validity=timeline_validity,
                unit_count=int(metrics.get("unit_count") or 0),
                anchored_count=int(metrics.get("anchored_count") or 0),
                coverage_ratio=float(metrics.get("coverage_ratio") or 0.0),
                hook_waste_ratio=float(metrics.get("hook_waste_ratio") or 0.0),
                seed_candidate_count=int(metrics.get("seed_candidate_count") or 0),
                masked_candidate_count=int(metrics.get("masked_candidate_count") or 0),
                primary_candidate_count=int(metrics.get("primary_candidate_count") or 0),
                block_candidate_count=int(metrics.get("block_candidate_count") or 0),
                block_count=int(metrics.get("block_count") or 0),
                compatibility_edge_count=int(metrics.get("compatibility_edge_count") or 0),
                avg_in_degree=float(metrics.get("avg_in_degree") or 0.0),
                largest_unresolved_span=int(metrics.get("largest_unresolved_span") or 0),
                unresolved_count=int(metrics.get("unresolved_count") or 0),
                gap_rescue_match_count=int(metrics.get("gap_rescue_match_count") or 0),
                duplicate_candidate_hook_count=int(
                    metrics.get("duplicate_candidate_hook_count") or 0
                ),
            )
        )

    if not window_summaries:
        raise ValueError(f"未在 {postprocess_dir} 找到 21_anchor_mount.graph.json")

    decision_ingress_scope_mismatches = _collect_decision_ingress_scope_mismatches(
        postprocess_dir=postprocess_dir
    )
    report = {
        "job_dir": str(job_dir),
        "window_count": len(window_summaries),
        "timeline_validity_counts": dict(validity_counts),
        "timeline_validity_reason_counts": dict(validity_reason_counts),
        "mount_status_counts": dict(mount_status_counts),
        "windows": [_window_summary_to_dict(item) for item in window_summaries],
        "candidate_flow_by_validity": _build_candidate_flow_by_validity(
            windows=window_summaries,
        ),
        "decision_ingress_scope_mismatch_count": len(decision_ingress_scope_mismatches),
        "decision_ingress_scope_mismatches": decision_ingress_scope_mismatches,
        "hypothesis_evidence": _build_hypothesis_evidence(
            windows=window_summaries,
            punctuation_debug_chunk_count=punctuation_debug_chunk_count,
        ),
    }
    return report


def _build_hypothesis_evidence(
    *,
    windows: list[TraceWindowSummary],
    punctuation_debug_chunk_count: int,
) -> dict[str, Any]:
    grouped: dict[str, list[TraceWindowSummary]] = defaultdict(list)
    for window in windows:
        grouped[window.timeline_validity].append(window)

    zero_gap_rescue_windows = [
        item.chunk_name for item in windows if item.gap_rescue_match_count == 0
    ]
    low_coverage_windows = sorted(windows, key=lambda item: item.coverage_ratio)[:5]
    primary_candidate_survivor_windows = sorted(
        (
            item
            for item in windows
            if item.primary_candidate_count > 0 and item.coverage_ratio < 0.5
        ),
        key=lambda item: (item.coverage_ratio, -item.primary_candidate_count),
    )[:5]
    sparse_graph_windows = sorted(
        windows,
        key=lambda item: (_safe_ratio(item.compatibility_edge_count, item.block_count), item.coverage_ratio),
    )[:5]

    return {
        "slow_text_unreliable": {
            "assessable_from_trace_only": False,
            "reason": "现有 summary trace 不包含音频重放与候选文本-音频逐词校验，不能仅凭 trace 断定慢流文本是否幻觉",
            "punctuation_debug_chunk_count": punctuation_debug_chunk_count,
        },
        "anchor_mount_failed": {
            "window_count": len(windows),
            "worst_coverage_windows": [
                {
                    "chunk_name": item.chunk_name,
                    "window_id": item.window_id,
                    "coverage_ratio": item.coverage_ratio,
                    "unresolved_count": item.unresolved_count,
                    "largest_unresolved_span": item.largest_unresolved_span,
                }
                for item in low_coverage_windows
            ],
        },
        "anchor_rejected": {
            "assessable_from_trace_only": False,
            "reason": "当前落盘只有 summary graph，没有 candidate 级 trust_report / reject_reason，无法仅凭现有 trace 证明“候选存在但被可信度拒绝”",
            "duplicate_candidate_hook_total": sum(
                item.duplicate_candidate_hook_count for item in windows
            ),
            "candidate_flow_by_validity": _build_candidate_flow_by_validity(
                windows=windows,
            ),
            "windows_with_primary_candidates_but_low_coverage": [
                {
                    "chunk_name": item.chunk_name,
                    "window_id": item.window_id,
                    "unit_count": item.unit_count,
                    "anchored_count": item.anchored_count,
                    "coverage_ratio": item.coverage_ratio,
                    "seed_candidate_count": item.seed_candidate_count,
                    "primary_candidate_count": item.primary_candidate_count,
                    "primary_over_seed_ratio": round(
                        _safe_ratio(item.primary_candidate_count, item.seed_candidate_count),
                        6,
                    ),
                }
                for item in primary_candidate_survivor_windows
            ],
        },
        "chain_or_dp_failed": {
            "zero_gap_rescue_window_count": len(zero_gap_rescue_windows),
            "avg_metrics_by_validity": {
                key: {
                    "unit_count": round(mean(item.unit_count for item in items), 6),
                    "anchored_count": round(mean(item.anchored_count for item in items), 6),
                    "coverage_ratio": round(mean(item.coverage_ratio for item in items), 6),
                    "seed_candidate_count": round(
                        mean(item.seed_candidate_count for item in items), 6
                    ),
                    "primary_candidate_count": round(
                        mean(item.primary_candidate_count for item in items), 6
                    ),
                    "block_count": round(mean(item.block_count for item in items), 6),
                    "edge_per_block": round(
                        mean(
                            _safe_ratio(item.compatibility_edge_count, item.block_count)
                            for item in items
                        ),
                        6,
                    ),
                    "hook_waste_ratio": round(
                        mean(item.hook_waste_ratio for item in items), 6
                    ),
                    "compatibility_edge_count": round(
                        mean(item.compatibility_edge_count for item in items), 6
                    ),
                    "avg_in_degree": round(mean(item.avg_in_degree for item in items), 6),
                }
                for key, items in grouped.items()
            },
            "worst_graph_sparsity_windows": [
                {
                    "chunk_name": item.chunk_name,
                    "window_id": item.window_id,
                    "coverage_ratio": item.coverage_ratio,
                    "block_count": item.block_count,
                    "compatibility_edge_count": item.compatibility_edge_count,
                    "edge_per_block": round(
                        _safe_ratio(item.compatibility_edge_count, item.block_count),
                        6,
                    ),
                    "primary_candidate_count": item.primary_candidate_count,
                }
                for item in sparse_graph_windows
            ],
        },
    }


def _recover_job_window_seed(
    *,
    job_dir: Path,
    window_id: str,
    language: str | None,
) -> ReplayJobWindowSeed:
    trace_dir = _find_window_trace_dir(job_dir=job_dir, window_id=window_id)
    prep_input = _load_json(trace_dir / "10_preparation.input.json")
    source_chunk_ids = tuple(str(item) for item in prep_input.get("source_chunk_ids") or ())
    source_chunk_indices = tuple(int(item) for item in prep_input.get("source_chunk_indices") or ())
    if not source_chunk_ids or len(source_chunk_ids) != len(source_chunk_indices):
        raise ValueError(f"窗口 {window_id} 的 source chunk 信息缺失或不完整")

    owner_chunk_id = str(prep_input.get("owner_chunk_id") or source_chunk_ids[-1])
    owner_chunk_index = int(prep_input.get("owner_chunk_index") or source_chunk_indices[-1])
    resolved_language = (
        str(language or prep_input.get("language_hint") or "auto").strip().lower() or "auto"
    )
    chunk_metadata = _load_chunk_metadata_map(job_dir=job_dir)
    punctuation_rows = _load_punctuation_row_map(job_dir=job_dir)
    chunk_provenance = _load_chunk_provenance_from_decision_ingress(trace_dir=trace_dir)

    chunk_bindings: list[WindowChunkBinding] = []
    source_units: list[WindowSourceUnit] = []
    source_chunk_reports: list[dict[str, Any]] = []
    source_text_parts: list[str] = []
    audio_segments: list[tuple[float, float]] = []

    for chunk_id, chunk_index in zip(source_chunk_ids, source_chunk_indices):
        metadata = chunk_metadata.get(int(chunk_index))
        if metadata is None:
            raise ValueError(f"chunk-{chunk_index} 缺少 preprocessing.chunks_metadata")
        punctuation_row = punctuation_rows.get(chunk_id)
        if punctuation_row is None:
            raise ValueError(f"{chunk_id} 缺少 debug/punctuation.jsonl 记录")

        chunk_start = float(metadata.get("start_time") or metadata.get("start") or 0.0)
        chunk_end = float(metadata.get("end_time") or metadata.get("end") or chunk_start)
        punctuated_text = str(
            punctuation_row.get("punctuated_text")
            or punctuation_row.get("original_text")
            or ""
        ).strip()
        if not punctuated_text:
            raise ValueError(f"{chunk_id} 缺少可恢复的 punctuated_text")

        role = "owner" if int(chunk_index) == owner_chunk_index else "core"
        binding = WindowChunkBinding(
            chunk_id=chunk_id,
            chunk_index=int(chunk_index),
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            overlap_ratio=1.0,
            role=role,
            is_owner=role == "owner",
        )
        provenance = chunk_provenance.get(chunk_id, {})
        speaker_id = str(provenance.get("speaker_id") or "unknown")
        turn_id = provenance.get("turn_id")
        normalized_turn_id = None if turn_id in {None, ""} else str(turn_id)

        chunk_bindings.append(binding)
        source_units.append(
            WindowSourceUnit(
                unit_id=f"{window_id}:{chunk_id}",
                semantic_chunk_id=f"{window_id}:{chunk_id}",
                text=punctuated_text,
                audio_start=chunk_start,
                audio_end=chunk_end,
                source_chunk_ids=(chunk_id,),
                source_chunk_indices=(int(chunk_index),),
                speaker_id=speaker_id,
                turn_id=normalized_turn_id,
                language=resolved_language,
                arrived_at=chunk_end,
            )
        )
        source_chunk_reports.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": int(chunk_index),
                "role": role,
                "start": chunk_start,
                "end": chunk_end,
                "speaker_id": speaker_id,
                "turn_id": normalized_turn_id,
                "text_len": len(punctuated_text),
            }
        )
        source_text_parts.append(punctuated_text)
        audio_segments.append((chunk_start, chunk_end))

    source_text = " ".join(part for part in source_text_parts if part).strip()
    if not source_text:
        raise ValueError(f"窗口 {window_id} 无法恢复 source_text")

    coverage = WindowCoverage(
        core_segments=tuple(audio_segments),
        left_guard_sec=0.0,
        right_guard_sec=0.0,
        chunk_bindings=tuple(chunk_bindings),
    )
    ready_window = ReadySlowWindow(
        window_id=window_id,
        owner_chunk_id=owner_chunk_id,
        owner_chunk_index=owner_chunk_index,
        window_mode="steady",
        flush_reason="replay_job_window",
        audio_segments=tuple(audio_segments),
        coverage=coverage,
        source_semantic_chunk_ids=tuple(unit.semantic_chunk_id for unit in source_units),
        source_chunk_ids=source_chunk_ids,
        source_chunk_indices=source_chunk_indices,
        source_units=tuple(source_units),
        dialogue_shape=_build_dialogue_shape(source_units=tuple(source_units)),
        language_profile=WindowLanguageProfile(
            primary_language=resolved_language,
            language_mix_state="single_language",
            decision_domains=("timeanchored_alignment",),
            should_bypass_whisper=False,
        ),
        prompt_seed=PromptSeed(text=source_text),
        batch_hint=WindowBatchHint(
            duration_bucket=f"replay_{len(source_units)}chunk",
            token_estimate=max(len(source_text.split()), 1),
            acoustic_density_hint="unknown",
            queue_priority=0,
        ),
        created_at=max(segment[1] for segment in audio_segments),
    )
    return ReplayJobWindowSeed(
        job_dir=job_dir,
        trace_dir=trace_dir,
        window_id=window_id,
        language=resolved_language,
        source_text=source_text,
        ready_window=ready_window,
        source_chunk_reports=tuple(source_chunk_reports),
    )


def _find_window_trace_dir(*, job_dir: Path, window_id: str) -> Path:
    postprocess_dir = Path(job_dir) / "debug" / "postprocess"
    for prep_input_path in sorted(postprocess_dir.glob("chunk_*/10_preparation.input.json")):
        payload = _load_json(prep_input_path)
        if str(payload.get("window_id") or "") == str(window_id):
            return prep_input_path.parent
    raise FileNotFoundError(f"未找到窗口 {window_id} 对应的 postprocess trace 目录")


def _load_chunk_metadata_map(*, job_dir: Path) -> dict[int, dict[str, Any]]:
    checkpoint = _load_json(Path(job_dir) / "checkpoint.json")
    preprocessing = dict(checkpoint.get("preprocessing") or {})
    chunk_rows = tuple(preprocessing.get("chunks_metadata") or ())
    mapping: dict[int, dict[str, Any]] = {}
    for row in chunk_rows:
        if not isinstance(row, dict):
            continue
        try:
            mapping[int(row.get("index"))] = row
        except (TypeError, ValueError):
            continue
    return mapping


def _load_punctuation_row_map(*, job_dir: Path) -> dict[str, dict[str, Any]]:
    punctuation_path = Path(job_dir) / "debug" / "punctuation.jsonl"
    if not punctuation_path.exists():
        raise FileNotFoundError(f"未找到标点调试文件: {punctuation_path}")
    mapping: dict[str, dict[str, Any]] = {}
    for line in punctuation_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        chunk_id = str(payload.get("chunk_id") or "").strip()
        if chunk_id:
            mapping[chunk_id] = payload
    return mapping


def _load_chunk_provenance_from_decision_ingress(*, trace_dir: Path) -> dict[str, dict[str, Any]]:
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: {
            "speaker": Counter(),
            "turn": Counter(),
        }
    )
    for path in sorted(Path(trace_dir).glob("31_decision_ingress.output*.json")):
        payload = _load_json(path)
        for unit in tuple(payload.get("token_units") or ()):
            if not isinstance(unit, dict):
                continue
            speaker_id = str(unit.get("speaker_id") or "").strip()
            turn_id = str(unit.get("turn_id") or "").strip()
            for chunk_id in tuple(unit.get("source_chunk_ids") or ()):
                normalized_chunk_id = str(chunk_id or "").strip()
                if not normalized_chunk_id:
                    continue
                if speaker_id:
                    counts[normalized_chunk_id]["speaker"][speaker_id] += 1
                if turn_id:
                    counts[normalized_chunk_id]["turn"][turn_id] += 1

    result: dict[str, dict[str, Any]] = {}
    for chunk_id, bucket in counts.items():
        speaker_id = bucket["speaker"].most_common(1)[0][0] if bucket["speaker"] else "unknown"
        turn_id = bucket["turn"].most_common(1)[0][0] if bucket["turn"] else None
        result[chunk_id] = {
            "speaker_id": speaker_id,
            "turn_id": turn_id,
        }
    return result


def _build_dialogue_shape(*, source_units: tuple[WindowSourceUnit, ...]) -> DialogueShapeSnapshot:
    if not source_units:
        return DialogueShapeSnapshot(
            shape="single_speaker",
            speaker_count=0,
            dominant_speaker_id=None,
            dominant_speaker_ratio=0.0,
            speaker_switch_count=0,
            speaker_switch_density=0.0,
            turn_count=0,
            avg_turn_duration_sec=0.01,
        )

    speakers = [str(unit.speaker_id or "unknown") for unit in source_units]
    turns = [str(unit.turn_id or "") for unit in source_units if str(unit.turn_id or "").strip()]
    speaker_counts = Counter(speakers)
    dominant_speaker_id, dominant_count = speaker_counts.most_common(1)[0]
    speaker_switch_count = sum(
        1
        for left, right in zip(speakers, speakers[1:])
        if str(left) != str(right)
    )
    total_duration = sum(max(float(unit.audio_end) - float(unit.audio_start), 0.0) for unit in source_units)
    turn_count = max(len(set(turns)), 1)
    shape = "single_speaker" if len(speaker_counts) <= 1 else "multi_speaker"
    return DialogueShapeSnapshot(
        shape=shape,
        speaker_count=len(speaker_counts),
        dominant_speaker_id=dominant_speaker_id,
        dominant_speaker_ratio=float(dominant_count) / float(len(source_units)),
        speaker_switch_count=speaker_switch_count,
        speaker_switch_density=float(speaker_switch_count) / float(max(len(source_units) - 1, 1)),
        turn_count=turn_count,
        avg_turn_duration_sec=max(total_duration / float(turn_count), 0.01),
    )


class _NoHardBoundaryChainSolver(ChainSolver):
    def _collect_hard_boundaries_after_unit(
        self,
        *,
        input_view: Any,
    ) -> set[int]:
        return set()


class _BoundaryRelaxingChainSolver(_NoHardBoundaryChainSolver):
    def __init__(
        self,
        *,
        ignore_turn_boundary: bool = False,
        ignore_speaker_boundary: bool = False,
    ) -> None:
        super().__init__()
        self._ignore_turn_boundary = bool(ignore_turn_boundary)
        self._ignore_speaker_boundary = bool(ignore_speaker_boundary)

    def _evaluate_compatibility(
        self,
        *,
        input_view: Any,
        previous: Any,
        current: Any,
        hard_boundaries_after_unit: set[int],
    ) -> CompatibilityDecision:
        decision = super()._evaluate_compatibility(
            input_view=input_view,
            previous=previous,
            current=current,
            hard_boundaries_after_unit=hard_boundaries_after_unit,
        )
        if decision.compatible:
            return decision
        if self._ignore_speaker_boundary and decision.reason == "speaker_change_blocked":
            return CompatibilityDecision(
                compatible=True,
                reason="speaker_boundary_ignored",
                diagnostics=dict(decision.diagnostics),
            )
        if self._ignore_turn_boundary and decision.reason == "turn_change_blocked":
            return CompatibilityDecision(
                compatible=True,
                reason="turn_boundary_ignored",
                diagnostics=dict(decision.diagnostics),
            )
        return decision


class _ForcedCommitChainSolver(_NoHardBoundaryChainSolver):
    def __init__(
        self,
        *,
        forced_block_ids: tuple[str, ...],
        ignore_hard_boundaries: bool = True,
    ) -> None:
        super().__init__()
        self._forced_block_ids = tuple(str(item) for item in forced_block_ids if str(item))
        self._ignore_hard_boundaries = bool(ignore_hard_boundaries)

    def solve(
        self,
        *,
        input_view: Any,
        blocks: tuple[Any, ...],
    ) -> Any:
        baseline = super().solve(input_view=input_view, blocks=blocks)
        forced_blocks = {
            str(block.block_id): block
            for block in blocks
            if str(getattr(block, "block_id", "") or "") in self._forced_block_ids
        }
        if not forced_blocks:
            return baseline

        committed = list(getattr(baseline, "committed_blocks", ()) or ())
        seen = {str(getattr(block, "block_id", "") or "") for block in committed}
        for block_id in self._forced_block_ids:
            block = forced_blocks.get(block_id)
            if block is None or block_id in seen:
                continue
            committed.append(block)
            seen.add(block_id)

        committed.sort(
            key=lambda item: (
                min(item.unit_indices),
                min(item.hook_indices),
                max(item.unit_indices),
                max(item.hook_indices),
            )
        )
        return _rebuild_chain_solve_result(
            baseline=baseline,
            input_view=input_view,
            committed_blocks=tuple(committed),
        )


def _rebuild_chain_solve_result(
    *,
    baseline: Any,
    input_view: Any,
    committed_blocks: tuple[Any, ...],
) -> Any:
    from app.services.timeanchored_alignment.anchor_mount.chain_solver import ChainSolveResult

    unit_to_hook_indices: list[tuple[int, ...]] = [tuple() for _ in input_view.token_units]
    unit_anchor_kinds: list[str] = ["none" for _ in input_view.token_units]
    for block in committed_blocks:
        if len(block.unit_indices) == len(block.hook_indices):
            for unit_index, hook_index in zip(block.unit_indices, block.hook_indices, strict=False):
                unit_to_hook_indices[int(unit_index)] = (int(hook_index),)
                unit_anchor_kinds[int(unit_index)] = str(block.anchor_kind)
            continue
        for unit_index in block.unit_indices:
            unit_to_hook_indices[int(unit_index)] = tuple(int(item) for item in block.hook_indices)
            unit_anchor_kinds[int(unit_index)] = str(block.anchor_kind)

    unresolved = tuple(
        index for index, hook_indices in enumerate(unit_to_hook_indices) if not hook_indices
    )
    return ChainSolveResult(
        committed_blocks=committed_blocks,
        unit_to_hook_indices=tuple(unit_to_hook_indices),
        unit_anchor_kinds=tuple(unit_anchor_kinds),
        unresolved_unit_indices=unresolved,
        reseed_count=int(getattr(baseline, "reseed_count", 0) or 0),
        compatibility_edge_count=int(getattr(baseline, "compatibility_edge_count", 0) or 0),
        avg_in_degree=float(getattr(baseline, "avg_in_degree", 0.0) or 0.0),
        max_in_degree=int(getattr(baseline, "max_in_degree", 0) or 0),
        solver_elapsed_ms=float(getattr(baseline, "solver_elapsed_ms", 0.0) or 0.0),
    )


def _build_commit_strategy_report(
    *,
    stage_result: AnchorMountStageResult,
    recovery_plan: Any,
    compatibility_summary: dict[str, Any] | None = None,
    solver_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = dict(getattr(getattr(stage_result, "anchor_mount_result", None), "metrics", {}) or {})
    return {
        "timeline_validity": str(
            getattr(getattr(stage_result, "anchor_mount_result", None), "timeline_validity", "")
            or ""
        ),
        "should_fallback": bool(
            getattr(getattr(stage_result, "anchor_mount_result", None), "should_fallback", False)
        ),
        "metrics": {
            key: metrics.get(key)
            for key in (
                "anchored_count",
                "unresolved_count",
                "coverage_ratio",
                "block_count",
                "compatibility_edge_count",
                "anchor_island_count",
                "open_gap_count",
                "gap_rescue_match_count",
            )
        },
        "compatibility": dict(compatibility_summary or {}),
        "solver_graph": dict(solver_graph or {}),
        "plan": _serialize_recovery_plan(recovery_plan=recovery_plan),
    }


def _serialize_recovery_plan(*, recovery_plan: Any) -> dict[str, Any]:
    spans = tuple(getattr(recovery_plan, "spans", ()) or ())
    return {
        "has_recoverable_spans": bool(getattr(recovery_plan, "has_recoverable_spans", False)),
        "emergency_fallback_only": bool(
            getattr(recovery_plan, "emergency_fallback_only", False)
        ),
        "spans": [
            {
                "span_id": str(getattr(span, "span_id", "") or ""),
                "span_kind": str(getattr(span, "span_kind", "") or ""),
                "route": str(getattr(span, "route", "") or ""),
                "token_start": int(getattr(span, "token_start", -1) or -1),
                "token_end": int(getattr(span, "token_end", -1) or -1),
                "time_start": float(getattr(span, "time_start", 0.0) or 0.0),
                "time_end": float(getattr(span, "time_end", 0.0) or 0.0),
                "reason_codes": [
                    str(item)
                    for item in tuple(getattr(span, "reason_codes", ()) or ())
                ],
            }
            for span in spans
        ],
    }


def _analyze_commit_strategy(
    *,
    preparation: AlignmentPreparationPackage,
    language: str,
    chain_solver: ChainSolver,
    policy_snapshot: Any | None,
) -> dict[str, Any]:
    ingress_validator = IngressValidator()
    input_view = ingress_validator.validate(
        preparation=preparation,
        language=language,
        policy_snapshot=policy_snapshot,
    )
    state = AnchorMountCorePipeline(chain_solver=chain_solver).run(input_view)
    compatibility_report = _build_compatibility_report(
        input_view=input_view,
        blocks=tuple(state.local_blocks),
        chain_solver=chain_solver,
    )
    solver_graph = _build_solver_graph_report(
        input_view=input_view,
        blocks=tuple(state.local_blocks),
        chain_solver=chain_solver,
    )
    stage_result = AnchorMountAlignmentService(chain_solver=chain_solver).align(
        preparation=preparation,
        language=language,
        policy_snapshot=policy_snapshot,
    )
    recovery_plan = WindowRecoveryPlanner().build(stage_result=stage_result)
    return _build_commit_strategy_report(
        stage_result=stage_result,
        recovery_plan=recovery_plan,
        compatibility_summary={
            "reject_reason_counts": dict(compatibility_report.get("reject_reason_counts") or {}),
            "accept_reason_counts": dict(compatibility_report.get("accept_reason_counts") or {}),
            "block_count": int(compatibility_report.get("block_count") or 0),
            "pair_count": len(tuple(compatibility_report.get("pair_decisions") or ())),
        },
        solver_graph=solver_graph,
    )


def _build_solver_graph_report(
    *,
    input_view: Any,
    blocks: tuple[Any, ...],
    chain_solver: ChainSolver,
) -> dict[str, Any]:
    ordered_blocks, incoming_edges, best_scores, parents = _trace_chain_solver(
        input_view=input_view,
        blocks=blocks,
        chain_solver=chain_solver,
    )
    if not ordered_blocks:
        return {
            "block_count": 0,
            "component_count": 0,
            "committed_block_ids": [],
            "components": [],
            "suffix_component_selection": {},
        }
    committed_blocks = tuple(
        chain_solver._reconstruct_best_chain(  # type: ignore[attr-defined]
            ordered_blocks=list(ordered_blocks),
            best_scores=list(best_scores),
            parents=list(parents),
        )
    )
    committed_block_ids = {
        str(getattr(block, "block_id", "") or "") for block in committed_blocks if str(getattr(block, "block_id", "") or "")
    }
    committed_end_unit_index = max(
        (max(int(item) for item in block.unit_indices) for block in committed_blocks),
        default=-1,
    )
    committed_end_hook_index = max(
        (max(int(item) for item in block.hook_indices) for block in committed_blocks),
        default=-1,
    )
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(ordered_blocks))}
    for current_index, previous_indices in incoming_edges.items():
        for previous_index in previous_indices:
            adjacency[current_index].add(previous_index)
            adjacency[previous_index].add(current_index)

    visited: set[int] = set()
    components: list[dict[str, Any]] = []
    for start_index in range(len(ordered_blocks)):
        if start_index in visited:
            continue
        pending = [start_index]
        component_indices: list[int] = []
        visited.add(start_index)
        while pending:
            cursor = pending.pop()
            component_indices.append(cursor)
            for neighbor in adjacency[cursor]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                pending.append(neighbor)
        component_indices.sort()
        component_blocks = [ordered_blocks[index] for index in component_indices]
        component_unit_indices = sorted(
            {int(item) for block in component_blocks for item in tuple(block.unit_indices or ())}
        )
        component_hook_indices = sorted(
            {int(item) for block in component_blocks for item in tuple(block.hook_indices or ())}
        )
        best_terminal_index = max(component_indices, key=lambda index: best_scores[index])
        component_block_ids = [
            str(getattr(ordered_blocks[index], "block_id", "") or "") for index in component_indices
        ]
        component_turn_ids = sorted(
            {
                str(getattr(input_view.token_units[unit_index], "turn_id", "") or "")
                for unit_index in component_unit_indices
                if 0 <= unit_index < len(input_view.token_units)
            }
        )
        component_speaker_ids = sorted(
            {
                str(getattr(input_view.token_units[unit_index], "speaker_id", "") or "")
                for unit_index in component_unit_indices
                if 0 <= unit_index < len(input_view.token_units)
            }
        )
        component_chunk_ids = sorted(
            {
                str(chunk_id)
                for unit_index in component_unit_indices
                if 0 <= unit_index < len(input_view.token_units)
                for chunk_id in tuple(getattr(input_view.token_units[unit_index], "source_chunk_ids", ()) or ())
                if str(chunk_id)
            }
        )
        internal_edge_count = sum(
            1
            for current_index in component_indices
            for previous_index in incoming_edges.get(current_index, [])
            if previous_index in component_indices
        )
        components.append(
            {
                "component_id": f"component-{len(components)}",
                "block_ids": component_block_ids,
                "block_count": len(component_indices),
                "unit_coverage": len(component_unit_indices),
                "hook_coverage": len(component_hook_indices),
                "min_unit_index": component_unit_indices[0] if component_unit_indices else -1,
                "max_unit_index": component_unit_indices[-1] if component_unit_indices else -1,
                "min_hook_index": component_hook_indices[0] if component_hook_indices else -1,
                "max_hook_index": component_hook_indices[-1] if component_hook_indices else -1,
                "best_terminal_block_id": str(
                    getattr(ordered_blocks[best_terminal_index], "block_id", "") or ""
                ),
                "best_terminal_score": round(float(best_scores[best_terminal_index]), 6),
                "has_committed_block": any(block_id in committed_block_ids for block_id in component_block_ids),
                "committed_block_ids": [
                    block_id for block_id in component_block_ids if block_id in committed_block_ids
                ],
                "turn_ids": component_turn_ids,
                "speaker_ids": component_speaker_ids,
                "source_chunk_ids": component_chunk_ids,
                "internal_edge_count": internal_edge_count,
            }
        )

    suffix_component_selection = _select_suffix_component(
        components=tuple(components),
        committed_end_unit_index=committed_end_unit_index,
        committed_end_hook_index=committed_end_hook_index,
    )
    return {
        "block_count": len(ordered_blocks),
        "edge_count": sum(len(items) for items in incoming_edges.values()),
        "component_count": len(components),
        "committed_block_ids": [
            str(getattr(block, "block_id", "") or "") for block in committed_blocks
        ],
        "committed_end_unit_index": committed_end_unit_index,
        "committed_end_hook_index": committed_end_hook_index,
        "components": components,
        "suffix_component_selection": suffix_component_selection,
    }


def _trace_chain_solver(
    *,
    input_view: Any,
    blocks: tuple[Any, ...],
    chain_solver: ChainSolver,
) -> tuple[tuple[Any, ...], dict[int, list[int]], tuple[float, ...], tuple[int, ...]]:
    ordered_blocks = tuple(
        sorted(
            blocks,
            key=lambda item: (
                min(item.unit_indices),
                min(item.hook_indices),
                max(item.unit_indices),
                max(item.hook_indices),
            ),
        )
    )
    incoming_edges = chain_solver._build_sparse_incoming_edges(  # type: ignore[attr-defined]
        input_view=input_view,
        ordered_blocks=list(ordered_blocks),
    )
    best_scores: list[float] = [float("-inf")] * len(ordered_blocks)
    parents: list[int] = [-1] * len(ordered_blocks)
    for index, block in enumerate(ordered_blocks):
        best_scores[index] = float(block.score)
        for previous_index in incoming_edges[index]:
            candidate_score = (
                best_scores[previous_index]
                + chain_solver._transition_bonus(  # type: ignore[attr-defined]
                    previous=ordered_blocks[previous_index],
                    current=block,
                )
                + float(block.score)
            )
            if candidate_score > best_scores[index]:
                best_scores[index] = candidate_score
                parents[index] = previous_index
    return ordered_blocks, incoming_edges, tuple(best_scores), tuple(parents)


def _select_suffix_component(
    *,
    components: tuple[dict[str, Any], ...],
    committed_end_unit_index: int,
    committed_end_hook_index: int,
) -> dict[str, Any]:
    candidates = [
        component
        for component in components
        if not bool(component.get("has_committed_block"))
        and int(component.get("min_unit_index", -1) or -1) > int(committed_end_unit_index)
        and int(component.get("min_hook_index", -1) or -1) > int(committed_end_hook_index)
    ]
    ranked_candidates = sorted(
        candidates,
        key=lambda item: (
            float(item.get("best_terminal_score", 0.0) or 0.0),
            int(item.get("unit_coverage", 0) or 0),
            int(item.get("internal_edge_count", 0) or 0),
            int(item.get("block_count", 0) or 0),
        ),
        reverse=True,
    )
    selected = dict(ranked_candidates[0]) if ranked_candidates else {}
    return {
        "candidate_component_ids": [
            str(item.get("component_id", "") or "") for item in ranked_candidates
        ],
        "selected_component_id": str(selected.get("component_id", "") or ""),
        "selected_block_ids": list(selected.get("block_ids", []) or []),
        "selected_turn_ids": list(selected.get("turn_ids", []) or []),
        "selected_source_chunk_ids": list(selected.get("source_chunk_ids", []) or []),
        "selected_best_terminal_score": float(selected.get("best_terminal_score", 0.0) or 0.0),
    }


def _build_source_contexts_from_job_audio(seed: ReplayJobWindowSeed) -> tuple[Any, ...]:
    import librosa

    from app.engines.sensevoice_engine import SenseVoiceEngine
    from app.pipelines.workers.fast_worker import FastWorker
    from app.schemas.pipeline_context import ProcessingContext
    from app.services.audio.chunk_engine import AudioChunk

    job_id = str(seed.job_dir.name or "job")
    worker = FastWorker(
        job_id=job_id,
        draft_engine=SenseVoiceEngine(language=seed.language),
        sensevoice_language=seed.language,
    )
    contexts: list[ProcessingContext] = []
    for binding in seed.ready_window.coverage.chunk_bindings:
        chunk_index = int(binding.chunk_index)
        audio_path = seed.job_dir / "cache_preprocess" / "separation" / f"chunk_{chunk_index}.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"未找到 chunk 音频: {audio_path}")
        audio_array, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
        chunk = AudioChunk(
            index=chunk_index,
            start=float(binding.chunk_start),
            end=float(binding.chunk_end),
            audio=audio_array,
            sample_rate=int(sample_rate),
            chunk_id=str(binding.chunk_id),
            language=seed.language,
        )
        ctx = ProcessingContext(
            job_id=job_id,
            chunk_index=chunk_index,
            audio_chunk=chunk,
            job_dir=seed.job_dir,
        )
        asyncio.run(worker.process(ctx))
        if ctx.time_base_chunk is None:
            raise ValueError(f"chunk-{chunk_index} 未构建出 time base")
        contexts.append(ctx)
    return tuple(contexts)


def _window_summary_to_dict(item: TraceWindowSummary) -> dict[str, Any]:
    return {
        "chunk_name": item.chunk_name,
        "window_id": item.window_id,
        "timeline_validity": item.timeline_validity,
        "unit_count": item.unit_count,
        "anchored_count": item.anchored_count,
        "coverage_ratio": item.coverage_ratio,
        "hook_waste_ratio": item.hook_waste_ratio,
        "seed_candidate_count": item.seed_candidate_count,
        "masked_candidate_count": item.masked_candidate_count,
        "primary_candidate_count": item.primary_candidate_count,
        "block_candidate_count": item.block_candidate_count,
        "block_count": item.block_count,
        "compatibility_edge_count": item.compatibility_edge_count,
        "avg_in_degree": item.avg_in_degree,
        "largest_unresolved_span": item.largest_unresolved_span,
        "unresolved_count": item.unresolved_count,
        "gap_rescue_match_count": item.gap_rescue_match_count,
        "duplicate_candidate_hook_count": item.duplicate_candidate_hook_count,
    }


def _build_candidate_flow_by_validity(
    *,
    windows: list[TraceWindowSummary],
) -> dict[str, Any]:
    grouped: dict[str, list[TraceWindowSummary]] = defaultdict(list)
    for window in windows:
        grouped[window.timeline_validity].append(window)

    return {
        validity: {
            "window_count": len(items),
            "avg_seed_candidate_count": round(
                mean(item.seed_candidate_count for item in items),
                6,
            ),
            "avg_primary_candidate_count": round(
                mean(item.primary_candidate_count for item in items),
                6,
            ),
            "avg_block_candidate_count": round(
                mean(item.block_candidate_count for item in items),
                6,
            ),
            "avg_block_count": round(mean(item.block_count for item in items), 6),
            "avg_primary_over_seed_ratio": round(
                mean(
                    _safe_ratio(item.primary_candidate_count, item.seed_candidate_count)
                    for item in items
                ),
                6,
            ),
            "avg_block_over_primary_ratio": round(
                mean(
                    _safe_ratio(item.block_candidate_count, item.primary_candidate_count)
                    for item in items
                ),
                6,
            ),
            "avg_anchored_over_primary_ratio": round(
                mean(
                    _safe_ratio(item.anchored_count, item.primary_candidate_count)
                    for item in items
                ),
                6,
            ),
            "avg_coverage_ratio": round(mean(item.coverage_ratio for item in items), 6),
        }
        for validity, items in grouped.items()
    }


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    if float(denominator or 0.0) <= 0.0:
        return 0.0
    return float(numerator or 0.0) / float(denominator)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_compatibility_report(
    *,
    input_view: Any,
    blocks: tuple[Any, ...],
    chain_solver: ChainSolver,
) -> dict[str, Any]:
    pair_decisions: list[dict[str, Any]] = []
    reject_reason_counts: Counter[str] = Counter()
    accept_reason_counts: Counter[str] = Counter()
    for current_index, current in enumerate(blocks):
        for previous_index in range(current_index):
            previous = blocks[previous_index]
            decision = chain_solver.explain_block_compatibility(
                input_view=input_view,
                previous=previous,
                current=current,
            )
            pair_decisions.append(
                {
                    "previous_block_id": str(previous.block_id),
                    "current_block_id": str(current.block_id),
                    "compatible": bool(decision.compatible),
                    "reason": str(decision.reason),
                    "diagnostics": dict(decision.diagnostics),
                }
            )
            if decision.compatible:
                accept_reason_counts[str(decision.reason)] += 1
            else:
                reject_reason_counts[str(decision.reason)] += 1
    return {
        "block_count": len(blocks),
        "blocks": [
            {
                "block_id": str(block.block_id),
                "unit_indices": [int(item) for item in block.unit_indices],
                "hook_indices": [int(item) for item in block.hook_indices],
                "score": float(block.score),
                "block_kind": str(block.block_kind),
                "anchor_kind": str(block.anchor_kind),
                "trust_tier": str(block.trust_tier),
            }
            for block in blocks
        ],
        "pair_decisions": pair_decisions,
        "reject_reason_counts": dict(reject_reason_counts),
        "accept_reason_counts": dict(accept_reason_counts),
    }


def _collect_decision_ingress_scope_mismatches(
    *,
    postprocess_dir: Path,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for payload_path in sorted(postprocess_dir.glob("chunk_*/31_decision_ingress.output*.json")):
        payload = _load_json(payload_path)
        compat_report = payload.get("compat_report") or {}
        speaker_bridge = compat_report.get("speaker_bridge") or {}
        span_selector = compat_report.get("span_selector")
        token_units = payload.get("token_units") or []
        if not (
            isinstance(span_selector, list)
            and len(span_selector) == 2
            and token_units
            and isinstance(speaker_bridge, dict)
        ):
            continue
        token_start, token_end = int(span_selector[0]), int(span_selector[1])
        if not (0 <= token_start < token_end < len(token_units)):
            continue

        try:
            full_start = min(float(item["start"]) for item in token_units)
            full_end = max(float(item["end"]) for item in token_units)
        except (KeyError, TypeError, ValueError):
            continue

        window_spans = speaker_bridge.get("window_spans") or []
        if not window_spans:
            continue
        if not _spans_cover_full_window(
            window_spans=window_spans,
            full_start=full_start,
            full_end=full_end,
        ):
            continue

        mismatches.append(
            {
                "chunk_name": payload_path.parent.name,
                "trace_file": payload_path.name,
                "span_selector": [token_start, token_end],
                "token_count": len(token_units),
                "window_spans": window_spans,
                "full_window_span": [full_start, full_end],
                "selected_turn_count": int(speaker_bridge.get("selected_turn_count") or 0),
                "timeline_turn_count": int(speaker_bridge.get("timeline_turn_count") or 0),
            }
        )
    return mismatches


def _spans_cover_full_window(
    *,
    window_spans: list[Any],
    full_start: float,
    full_end: float,
) -> bool:
    try:
        span_start = min(float(item[0]) for item in window_spans)
        span_end = max(float(item[1]) for item in window_spans)
    except (TypeError, ValueError, IndexError):
        return False
    return abs(span_start - full_start) <= 1e-3 and abs(span_end - full_end) <= 1e-3
