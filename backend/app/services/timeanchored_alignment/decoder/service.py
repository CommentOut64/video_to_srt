"""Phase 3 decoder shadow 服务。"""

from __future__ import annotations

from app.services.language_policy import build_language_policy_snapshot
from app.services.timeanchored_alignment.contracts import (
    OBSERVATION_CAPABILITY_FRAME_POSTERIOR,
    AlignmentPath,
    AlignmentReport,
    LayerSummary,
)
from app.services.timeanchored_alignment.decoder.contracts import DecoderShadowResult
from app.services.timeanchored_alignment.decoder.local_repair import LocalRepair
from app.services.timeanchored_alignment.decoder.observation_lattice import ObservationLatticeBuilder
from app.services.timeanchored_alignment.decoder.path_reader import AlignmentPathReader
from app.services.timeanchored_alignment.decoder.viterbi_decoder import ViterbiDecoder
from app.services.timeanchored_alignment.preparation.contracts import PreparationBundle


class AlignmentDecoderService:
    """只负责 shadow/diff 的 PreparationBundle -> AlignmentPath MVP。"""

    def __init__(
        self,
        *,
        lattice_builder: ObservationLatticeBuilder | None = None,
        decoder: ViterbiDecoder | None = None,
        local_repair: LocalRepair | None = None,
        path_reader: AlignmentPathReader | None = None,
    ) -> None:
        self._lattice_builder = lattice_builder or ObservationLatticeBuilder()
        self._decoder = decoder or ViterbiDecoder()
        self._local_repair = local_repair or LocalRepair()
        self._path_reader = path_reader or AlignmentPathReader()

    def execute(self, *, preparation: PreparationBundle) -> DecoderShadowResult:
        if str(preparation.provenance.text_source or "").strip().lower() != "slow":
            return self._failure(
                preparation=preparation,
                route="selection_reject_slow",
                status="warning",
                diagnostics={"text_source": str(preparation.provenance.text_source or "")},
            )

        snapshot = build_language_policy_snapshot(
            language_hint=str(preparation.canonical_sequence.language_hint or "auto"),
            feature_scope="timeanchored_alignment",
        )
        if not bool(snapshot.metadata.get("timeanchored_main_chain_eligible", True)):
            return self._failure(
                preparation=preparation,
                route="true_mixed_unresolved",
                status="error",
                diagnostics={"window_kind": str(snapshot.metadata.get("window_kind", ""))},
            )

        lattice = self._lattice_builder.build(preparation=preparation)
        decode_path = self._decoder.decode(lattice=lattice)
        if decode_path is None:
            return self._failure(
                preparation=preparation,
                route="alignment_no_path",
                status="error",
                lattice=lattice,
                diagnostics={"reason": "decoder_returned_no_path"},
            )
        repaired_path = self._local_repair.repair(
            preparation=preparation,
            decode_path=decode_path,
        )
        aligned_tokens, boundary_candidates, low_confidence_spans, diagnostics = self._path_reader.read(
            preparation=preparation,
            decode_path=repaired_path,
        )
        coverage_min = self._safe_float(snapshot.thresholds.get("text_direct_ratio_min"), 0.65)
        max_failure_span = max(
            1,
            int(round(self._safe_float(snapshot.thresholds.get("max_continuous_failure_span"), 6.0))),
        )
        ambiguous_pronunciation_token_count = self._count_ambiguous_pronunciation_tokens(
            preparation=preparation,
        )
        longest_low_conf_span = max(
            (
                int(span.end_token_index) - int(span.start_token_index) + 1
                for span in low_confidence_spans
            ),
            default=0,
        )
        route = "alignment_path"
        status = "ok"
        failure_semantic = "none"
        if not aligned_tokens:
            route = "alignment_no_path"
            status = "error"
            failure_semantic = route
        elif (
            float(repaired_path.coverage_ratio) < float(coverage_min)
            or longest_low_conf_span > max_failure_span
        ):
            status = "warning"
            failure_semantic = "alignment_low_confidence"
        elif (
            ambiguous_pronunciation_token_count > 0
            and str(preparation.acoustic_observation_pack.capability_level)
            != OBSERVATION_CAPABILITY_FRAME_POSTERIOR
        ):
            status = "warning"
            failure_semantic = "alignment_low_confidence"

        summary = LayerSummary(
            layer="alignment",
            status=status,
            counters={
                "token_count": len(aligned_tokens),
                "coverage_ratio": float(repaired_path.coverage_ratio),
                "direct_ratio": float(repaired_path.direct_ratio),
                "boundary_candidate_count": len(boundary_candidates),
                "low_confidence_span_count": len(low_confidence_spans),
                "ambiguous_pronunciation_token_count": int(ambiguous_pronunciation_token_count),
            },
        )
        alignment_path = AlignmentPath(
            path_id=f"{preparation.window_id}:shadow",
            aligned_tokens=aligned_tokens,
            route_confidence=max(0.0, min(1.0, float(repaired_path.coverage_ratio))),
            low_confidence_spans=low_confidence_spans,
            summary=summary,
            source_chunk_ids=tuple(preparation.source_chunk_ids),
            boundary_candidates=boundary_candidates,
        )
        report = AlignmentReport(
            summary=summary,
            route=route,
            failure_semantic=failure_semantic,
            metadata={
                "window_id": str(preparation.window_id),
                "coverage_ratio": float(repaired_path.coverage_ratio),
                "direct_ratio": float(repaired_path.direct_ratio),
                "longest_low_conf_span": int(longest_low_conf_span),
                "ambiguous_pronunciation_token_count": int(ambiguous_pronunciation_token_count),
                "thresholds": {
                    "text_direct_ratio_min": float(coverage_min),
                    "max_continuous_failure_span": int(max_failure_span),
                },
                **diagnostics,
            },
        )
        return DecoderShadowResult(
            alignment_path=alignment_path,
            boundary_candidates=boundary_candidates,
            low_confidence_spans=low_confidence_spans,
            alignment_report=report,
            lattice=lattice,
            decode_path=repaired_path,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _failure(
        *,
        preparation: PreparationBundle,
        route: str,
        status: str,
        lattice: object | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> DecoderShadowResult:
        summary = LayerSummary(
            layer="alignment",
            status=status,
            counters={
                "token_count": 0,
                "coverage_ratio": 0.0,
                "direct_ratio": 0.0,
                "boundary_candidate_count": 0,
                "low_confidence_span_count": 0,
            },
        )
        report = AlignmentReport(
            summary=summary,
            route=route,
            failure_semantic=route,
            metadata={
                "window_id": str(preparation.window_id),
                **dict(diagnostics or {}),
            },
        )
        return DecoderShadowResult(
            alignment_path=None,
            boundary_candidates=tuple(),
            low_confidence_spans=tuple(),
            alignment_report=report,
            lattice=lattice,
            diagnostics=dict(diagnostics or {}),
        )

    @staticmethod
    def _safe_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _count_ambiguous_pronunciation_tokens(*, preparation: PreparationBundle) -> int:
        count = 0
        for node in (preparation.pronunciation_graph.token_nodes or ()):
            metadata = dict(getattr(node, "metadata", {}) or {})
            if bool(metadata.get("pronunciation_ambiguous")):
                count += 1
                continue
            non_literal_variants = {
                str(getattr(variant, "reading_key", "") or "").strip()
                for variant in (getattr(node, "variants", ()) or ())
                if str(getattr(variant, "source", "") or "") != "token_text"
                and str(getattr(variant, "reading_key", "") or "").strip()
            }
            if len(non_literal_variants) > 1:
                count += 1
        return count
