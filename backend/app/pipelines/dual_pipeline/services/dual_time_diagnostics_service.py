"""
双轨实验诊断门面服务。

设计模式：Facade Pattern。
原因：收口 dual-time 对比与诊断落盘逻辑，降低实现类复杂度。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.models.sensevoice_models import SentenceSegment, WordTimestamp
from app.pipelines.dual_pipeline.services.textflow_facade_service import Layer456RunResult
from app.schemas.pipeline_context import ProcessingContext
from app.utils.text_utils import format_srt_timestamp


class DualTimeDiagnosticsService:
    """dual-time 对比、统计与调试落盘门面。"""

    def __init__(self, *, host: Any) -> None:
        self._host = host

    def should_record_m2_stage0_sample(self, *, chunk_index: int) -> bool:
        host = self._host
        if not host._is_m2_enabled:
            return False
        sample_rate = max(0.0, min(1.0, float(host._m2_shadow_sample_rate)))
        if sample_rate <= 0.0:
            return False
        if sample_rate >= 1.0:
            return True
        digest = hashlib.md5(f"{host.job_id}:{int(chunk_index)}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], byteorder="big", signed=False) % 10000
        threshold = int(sample_rate * 10000)
        return bucket < threshold

    @staticmethod
    def build_stage0_run_snapshot(run: Layer456RunResult) -> Dict[str, Any]:
        return {
            "sentence_count": int(len(run.final_sentences)),
            "alignment_score": float(run.alignment_result.alignment_score),
            "gap_ratio": float(run.alignment_result.gap_ratio),
            "alignment_time_source": str(run.alignment_time_source),
            "alignment_time_word_count": int(run.alignment_time_word_count),
        }

    def maybe_record_m2_stage0_sample(
        self,
        *,
        ctx: ProcessingContext,
        selected_variant: str,
        selected_reason: str,
        legacy_run: Layer456RunResult,
        experiment_run: Optional[Layer456RunResult],
        active_run: Layer456RunResult,
        compare_payload: Optional[Dict[str, Any]],
    ) -> None:
        host = self._host
        if not self.should_record_m2_stage0_sample(chunk_index=ctx.chunk_index):
            return

        payload: Dict[str, Any] = {
            "job_id": ctx.job_id,
            "chunk_index": int(ctx.chunk_index),
            "dual_time_mode": str(host._dual_time_mode),
            "selected_variant": str(selected_variant),
            "selected_reason": str(selected_reason),
            "m2_flags": {
                "enable": bool(host._is_m2_enabled),
                "nw_v2_enable": bool(host._is_m2_nw_v2_enabled),
                "time_mapping_enable": bool(host._is_m2_time_mapping_enabled),
                "shadow_sample_rate": float(host._m2_shadow_sample_rate),
                "shadow_provider_class": str(host._m2_shadow_provider_class),
            },
            "active_run": self.build_stage0_run_snapshot(active_run),
            "legacy_run": self.build_stage0_run_snapshot(legacy_run),
        }

        if experiment_run is not None:
            payload["shadow_run"] = self.build_stage0_run_snapshot(experiment_run)
        if compare_payload is not None and isinstance(compare_payload, dict):
            payload["comparison"] = dict(compare_payload.get("comparison") or {})

        host._append_m2_stage0_debug(ctx.job_dir, payload)

    def build_dual_time_compare_payload(
        self,
        *,
        ctx: ProcessingContext,
        legacy_run: Layer456RunResult,
        experiment_run: Layer456RunResult,
    ) -> Dict[str, Any]:
        boundary_stats = self.compute_boundary_metrics(
            legacy_run.final_sentences,
            experiment_run.final_sentences,
            tolerance_sec=self._host._dual_time_boundary_tolerance_sec,
        )
        word_mae = self.compute_word_mae_ms(legacy_run.words_for_split, experiment_run.words_for_split)
        sentence_mae = self.compute_sentence_mae_ms(
            legacy_run.final_sentences,
            experiment_run.final_sentences,
        )
        return {
            "job_id": ctx.job_id,
            "chunk_index": int(ctx.chunk_index),
            "mode": self._host._dual_time_mode,
            "legacy": {
                "sentence_count": len(legacy_run.final_sentences),
                "alignment_score": float(legacy_run.alignment_result.alignment_score),
                "gap_ratio": float(legacy_run.alignment_result.gap_ratio),
                "sentence_texts": [str(sentence.text or "") for sentence in legacy_run.final_sentences],
                "alignment_time_source": legacy_run.alignment_time_source,
                "alignment_time_word_count": int(legacy_run.alignment_time_word_count),
            },
            "experiment": {
                "sentence_count": len(experiment_run.final_sentences),
                "alignment_score": float(experiment_run.alignment_result.alignment_score),
                "gap_ratio": float(experiment_run.alignment_result.gap_ratio),
                "sentence_texts": [str(sentence.text or "") for sentence in experiment_run.final_sentences],
                "alignment_time_source": experiment_run.alignment_time_source,
                "alignment_time_word_count": int(experiment_run.alignment_time_word_count),
            },
            "comparison": {
                **boundary_stats,
                **word_mae,
                **sentence_mae,
            },
        }

    def select_dual_time_variant(
        self,
        *,
        compare_payload: Dict[str, Any],
        experiment_run: Layer456RunResult,
    ) -> Tuple[str, str]:
        if not experiment_run.final_sentences:
            return "legacy", "active_gate_empty_experiment"
        boundary_f1 = float(compare_payload.get("comparison", {}).get("boundary_f1", 0.0) or 0.0)
        if boundary_f1 < self._host._dual_time_active_min_boundary_f1:
            return (
                "legacy",
                f"active_gate_boundary_f1<{self._host._dual_time_active_min_boundary_f1:.2f}",
            )
        return "experiment", "active_gate_pass"

    def update_dual_time_summary(self, ctx: ProcessingContext, payload: Dict[str, Any]) -> None:
        host = self._host
        comparison = payload.get("comparison", {}) if isinstance(payload, dict) else {}
        acc = host._dual_time_compare_accumulator
        acc["chunk_count"] += 1
        acc["boundary_precision_sum"] += float(comparison.get("boundary_precision", 0.0) or 0.0)
        acc["boundary_recall_sum"] += float(comparison.get("boundary_recall", 0.0) or 0.0)
        acc["boundary_f1_sum"] += float(comparison.get("boundary_f1", 0.0) or 0.0)
        acc["word_start_mae_ms_sum"] += float(comparison.get("word_start_mae_ms", 0.0) or 0.0)
        acc["word_end_mae_ms_sum"] += float(comparison.get("word_end_mae_ms", 0.0) or 0.0)
        acc["sentence_start_mae_ms_sum"] += float(comparison.get("sentence_start_mae_ms", 0.0) or 0.0)
        acc["sentence_end_mae_ms_sum"] += float(comparison.get("sentence_end_mae_ms", 0.0) or 0.0)
        if payload.get("selected_variant") == "experiment":
            acc["selected_experiment_count"] += 1

        chunk_count = max(int(acc["chunk_count"]), 1)
        summary = {
            "job_id": ctx.job_id,
            "mode": host._dual_time_mode,
            "chunk_count": int(acc["chunk_count"]),
            "selected_experiment_count": int(acc["selected_experiment_count"]),
            "selected_experiment_ratio": float(acc["selected_experiment_count"] / chunk_count),
            "avg_boundary_precision": float(acc["boundary_precision_sum"] / chunk_count),
            "avg_boundary_recall": float(acc["boundary_recall_sum"] / chunk_count),
            "avg_boundary_f1": float(acc["boundary_f1_sum"] / chunk_count),
            "avg_word_start_mae_ms": float(acc["word_start_mae_ms_sum"] / chunk_count),
            "avg_word_end_mae_ms": float(acc["word_end_mae_ms_sum"] / chunk_count),
            "avg_sentence_start_mae_ms": float(acc["sentence_start_mae_ms_sum"] / chunk_count),
            "avg_sentence_end_mae_ms": float(acc["sentence_end_mae_ms_sum"] / chunk_count),
        }
        host._write_dual_time_summary_debug(ctx.job_dir, summary)

    def write_dual_time_debug_srt(self, job_dir: Optional[Path]) -> None:
        if not job_dir:
            return
        host = self._host
        try:
            debug_dir = job_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            legacy_srt = self.build_srt_from_chunk_map(host._dual_time_legacy_sentences_by_chunk)
            experiment_srt = self.build_srt_from_chunk_map(host._dual_time_experiment_sentences_by_chunk)
            (debug_dir / "dual_time_legacy.srt").write_text(legacy_srt, encoding="utf-8")
            (debug_dir / "dual_time_experiment.srt").write_text(experiment_srt, encoding="utf-8")
        except Exception as exc:
            host.logger.debug("写入双轨实验 SRT 失败（忽略）: %s", exc)

    @staticmethod
    def build_srt_from_chunk_map(chunk_map: Dict[int, List[SentenceSegment]]) -> str:
        rows: List[SentenceSegment] = []
        for chunk_idx in sorted(chunk_map.keys()):
            rows.extend(sorted(chunk_map[chunk_idx], key=lambda item: (item.start, item.end)))
        lines: List[str] = []
        for index, sentence in enumerate(rows, 1):
            text = str(sentence.text or sentence.text_clean or "").strip()
            lines.append(str(index))
            lines.append(
                f"{format_srt_timestamp(float(sentence.start))} --> {format_srt_timestamp(float(sentence.end))}"
            )
            lines.append(text)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def extract_sentence_boundaries(sentences: Sequence[SentenceSegment]) -> List[float]:
        if len(sentences) <= 1:
            return []
        return [float(sentence.end) for sentence in sentences[:-1]]

    @classmethod
    def compute_boundary_metrics(
        cls,
        legacy_sentences: Sequence[SentenceSegment],
        experiment_sentences: Sequence[SentenceSegment],
        *,
        tolerance_sec: float,
    ) -> Dict[str, float]:
        reference = cls.extract_sentence_boundaries(legacy_sentences)
        candidate = cls.extract_sentence_boundaries(experiment_sentences)
        matched = 0
        used = [False] * len(reference)
        for point in candidate:
            best_idx = -1
            best_delta = tolerance_sec + 1.0
            for idx, target in enumerate(reference):
                if used[idx]:
                    continue
                delta = abs(point - target)
                if delta <= tolerance_sec and delta < best_delta:
                    best_idx = idx
                    best_delta = delta
            if best_idx >= 0:
                used[best_idx] = True
                matched += 1

        if candidate:
            precision = matched / len(candidate)
        else:
            precision = 1.0 if not reference else 0.0
        if reference:
            recall = matched / len(reference)
        else:
            recall = 1.0 if not candidate else 0.0
        f1 = 0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall / (precision + recall))
        return {
            "boundary_matched": float(matched),
            "boundary_reference": float(len(reference)),
            "boundary_candidate": float(len(candidate)),
            "boundary_precision": float(precision),
            "boundary_recall": float(recall),
            "boundary_f1": float(f1),
        }

    @staticmethod
    def compute_word_mae_ms(
        legacy_words: Sequence[WordTimestamp],
        experiment_words: Sequence[WordTimestamp],
    ) -> Dict[str, float]:
        overlap = min(len(legacy_words), len(experiment_words))
        if overlap <= 0:
            return {
                "word_overlap": 0.0,
                "word_start_mae_ms": 0.0,
                "word_end_mae_ms": 0.0,
            }
        start_errors = [
            abs(float(legacy_words[i].start) - float(experiment_words[i].start)) * 1000.0
            for i in range(overlap)
        ]
        end_errors = [
            abs(float(legacy_words[i].end) - float(experiment_words[i].end)) * 1000.0
            for i in range(overlap)
        ]
        return {
            "word_overlap": float(overlap),
            "word_start_mae_ms": float(sum(start_errors) / overlap),
            "word_end_mae_ms": float(sum(end_errors) / overlap),
        }

    @staticmethod
    def compute_sentence_mae_ms(
        legacy_sentences: Sequence[SentenceSegment],
        experiment_sentences: Sequence[SentenceSegment],
    ) -> Dict[str, float]:
        overlap = min(len(legacy_sentences), len(experiment_sentences))
        if overlap <= 0:
            return {
                "sentence_overlap": 0.0,
                "sentence_start_mae_ms": 0.0,
                "sentence_end_mae_ms": 0.0,
            }
        start_errors = [
            abs(float(legacy_sentences[i].start) - float(experiment_sentences[i].start)) * 1000.0
            for i in range(overlap)
        ]
        end_errors = [
            abs(float(legacy_sentences[i].end) - float(experiment_sentences[i].end)) * 1000.0
            for i in range(overlap)
        ]
        return {
            "sentence_overlap": float(overlap),
            "sentence_start_mae_ms": float(sum(start_errors) / overlap),
            "sentence_end_mae_ms": float(sum(end_errors) / overlap),
        }
