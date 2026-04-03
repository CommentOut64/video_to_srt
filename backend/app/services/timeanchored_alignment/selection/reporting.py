from __future__ import annotations

from typing import Dict, Iterable, Tuple

from app.services.alignment.types import QualitySignals
from app.services.arbitration.arbiter import ArbitrationResult
from app.services.timeanchored_alignment.contracts import (
    LayerError,
    LayerSummary,
    LayerWarning,
    SelectedTextTruth,
    SelectionDecision,
    SelectionReport,
)
from app.services.timeanchored_alignment.selection.contracts import WindowSelectionScope


_REASON_WARNING_MAP = {
    "hallucination": ("W_SELECTION_REJECT_SLOW_HALLUCINATION", "慢流命中 hallucination 门控，回退 fast"),
    "repetition": ("W_SELECTION_REJECT_SLOW_REPETITION", "慢流命中重复文本门控，回退 fast"),
    "length_ratio_low": ("W_SELECTION_SLOW_LENGTH_LOW", "快慢文本长度比过低，选择 slow"),
    "length_ratio_high": ("W_SELECTION_REJECT_SLOW_LENGTH_HIGH", "快慢文本长度比过高，回退 fast"),
    "low_confidence_slow": ("W_SELECTION_REJECT_SLOW_LOW_CONF", "慢流置信度过低，回退 fast"),
    "fast_tail_guard": ("W_SELECTION_FAST_TAIL_GUARD", "快流补尾保护触发，保留 fast 真相"),
    "forced_fast": ("W_SELECTION_FORCE_FAST", "显式 force_fast 覆盖自动仲裁"),
    "forced_slow": ("W_SELECTION_FORCE_SLOW", "显式 force_slow 覆盖自动仲裁"),
}

_REJECTION_REASON_MAP = {
    "hallucination": "slow_hallucination",
    "repetition": "slow_repetition",
    "length_ratio_high": "slow_length_high",
    "length_ratio_low": "slow_length_low",
    "low_confidence_slow": "slow_low_confidence",
    "fast_tail_guard": "slow_tail_guarded_by_fast",
    "forced_fast": "force_fast",
    "forced_slow": "force_slow",
}


def parse_reason_codes(reason: str) -> Tuple[str, ...]:
    parts = tuple(str(part or "").strip() for part in str(reason or "").split("|"))
    return tuple(part for part in parts if part)


def build_selection_report(
    *,
    scope: WindowSelectionScope,
    quality_signals: QualitySignals,
    arbitration_result: ArbitrationResult,
    selected_text_truth: SelectedTextTruth,
    selection_decision: SelectionDecision,
) -> SelectionReport:
    reason_codes = parse_reason_codes(arbitration_result.reason)
    primary_reason_code = reason_codes[0] if reason_codes else "unknown"
    warnings = tuple(
        _build_warning(
            scope=scope,
            code=warning_code,
            message=warning_message,
            arbitration_result=arbitration_result,
            quality_signals=quality_signals,
        )
        for warning_code, warning_message in _iter_warning_specs(reason_codes)
    )
    errors = tuple(_build_errors(scope=scope, arbitration_result=arbitration_result))
    summary = LayerSummary(
        layer="selection",
        status="error" if errors else "ok",
        counters={
            "chosen_source": selection_decision.chosen_source,
            "decision": selection_decision.decision,
            "coverage": float(arbitration_result.coverage),
            "sv_score": float(arbitration_result.sv_score),
            "wh_score": float(arbitration_result.wh_score),
            "reason_count": len(reason_codes),
        },
        warnings=warnings,
        errors=errors,
        debug_enabled=False,
    )
    return SelectionReport(
        chosen_source=selection_decision.chosen_source,
        primary_reason_code=primary_reason_code,
        decision=selection_decision.decision,
        summary=summary,
        reason_codes=reason_codes,
        warnings=warnings,
        errors=errors,
        metrics={
            "coverage": float(arbitration_result.coverage),
            "sv_score": float(arbitration_result.sv_score),
            "wh_score": float(arbitration_result.wh_score),
            "confidence_fast": float(quality_signals.confidence_fast),
            "confidence_slow": float(quality_signals.confidence_slow),
            "length_ratio": float(quality_signals.length_ratio),
            "mapping_coverage": float(quality_signals.mapping_coverage),
            "is_hallucination": bool(quality_signals.is_hallucination),
            "is_repetition": bool(quality_signals.is_repetition),
            "selected_text_length": len(selected_text_truth.text),
        },
        metadata={
            "job_id": scope.job_id,
            "chunk_index": int(scope.chunk_index),
            "window_id": scope.window_id,
            "edge_selection_mode": scope.edge_selection_mode,
            "source_chunk_ids": list(scope.source_chunk_ids),
            "source_chunk_indices": list(scope.source_chunk_indices),
        },
    )


def build_rejection_reasons(reason_codes: Iterable[str]) -> Tuple[str, ...]:
    rejection_reasons = []
    for reason_code in reason_codes:
        mapped = _REJECTION_REASON_MAP.get(str(reason_code or ""))
        if mapped and mapped not in rejection_reasons:
            rejection_reasons.append(mapped)
    return tuple(rejection_reasons)


def _iter_warning_specs(reason_codes: Iterable[str]) -> Iterable[Tuple[str, str]]:
    for reason_code in reason_codes:
        warning = _REASON_WARNING_MAP.get(str(reason_code or ""))
        if warning is not None:
            yield warning


def _build_warning(
    *,
    scope: WindowSelectionScope,
    code: str,
    message: str,
    arbitration_result: ArbitrationResult,
    quality_signals: QualitySignals,
) -> LayerWarning:
    return LayerWarning(
        code=code,
        message=message,
        layer="selection",
        job_id=scope.job_id,
        chunk_id=str(scope.chunk_index),
        details={
            "chosen_source": arbitration_result.chosen_source,
            "reason": arbitration_result.reason,
            "coverage": float(arbitration_result.coverage),
            "confidence_fast": float(quality_signals.confidence_fast),
            "confidence_slow": float(quality_signals.confidence_slow),
        },
    )


def _build_errors(
    *,
    scope: WindowSelectionScope,
    arbitration_result: ArbitrationResult,
) -> Iterable[LayerError]:
    if not arbitration_result.error_code:
        return ()
    return (
        LayerError(
            code=str(arbitration_result.error_code),
            message="选择层未能拿到可用文本真相",
            layer="selection",
            job_id=scope.job_id,
            chunk_id=str(scope.chunk_index),
            details={
                "chosen_source": arbitration_result.chosen_source,
                "reason": arbitration_result.reason,
                "forced_source": arbitration_result.forced_source,
            },
        ),
    )
