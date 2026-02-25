"""
SpectralTriageStage - 音频预检阶段。

职责：
- 为每个 AudioChunk 执行音频预检；
- 产出 `needs_separation/recommended_model`；
- 输出可缓存的预检日志与探针状态。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from app.services.audio.chunk_engine import AudioChunk
from app.services.audio_spectrum_classifier import AudioSpectrumClassifier, get_spectrum_classifier

if TYPE_CHECKING:
    from app.utils.cancellation_token import CancellationToken

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class SpectralTriageStage:
    """音频预检阶段。"""

    TRIAGE_VERSION = "dnsmos_v1"

    def __init__(
        self,
        classifier: Optional[AudioSpectrumClassifier] = None,
        threshold: float = 0.35,
        logger: Optional[logging.Logger] = None,
        cancellation_token: Optional["CancellationToken"] = None,
        use_dnsmos_triage: bool = True,
        use_snr_triage: Optional[bool] = None,
        use_smart_probe: bool = False,
        show_progress: bool = True,
    ) -> None:
        """
        初始化阶段。

        兼容说明：
        - 旧参数 `use_snr_triage` 仍可传入，自动映射到 `use_dnsmos_triage`。
        """
        if use_snr_triage is not None:
            use_dnsmos_triage = bool(use_snr_triage)

        self.classifier = classifier or get_spectrum_classifier(use_dnsmos_strategy=use_dnsmos_triage)
        self.threshold = threshold
        self.logger = logger or logging.getLogger(__name__)
        self.cancellation_token = cancellation_token
        self.use_dnsmos_triage = bool(use_dnsmos_triage)
        # 兼容字段
        self.use_snr_triage = self.use_dnsmos_triage
        self.use_smart_probe = bool(use_smart_probe)
        self.show_progress = show_progress
        self._smart_probe = None
        self._last_triage_log: List[Dict[str, Any]] = []
        self._last_probe_state: Optional[Dict[str, Any]] = None

    @property
    def triage_version(self) -> str:
        return self.TRIAGE_VERSION

    def get_dnsmos_model_hash(self) -> str:
        return self.classifier.get_dnsmos_model_hash()

    def get_dnsmos_threshold_profile_hash(self) -> str:
        return self.classifier.get_dnsmos_threshold_profile_hash()

    def _get_smart_probe(self):
        if self._smart_probe is None and self.use_smart_probe:
            try:
                from app.services.smart_probe_service import get_smart_probe_service

                self._smart_probe = get_smart_probe_service()
                self.logger.info("智能探针服务已加载")
            except Exception as exc:
                self.logger.warning("智能探针服务加载失败: %s", exc)
                self._smart_probe = None
        return self._smart_probe

    def get_smart_probe_params(self) -> Optional[Dict[str, Any]]:
        smart_probe = self._get_smart_probe()
        if smart_probe is None:
            return None
        return {
            "probe_sep_ratio_min": smart_probe.probe_sep_ratio_min,
            "probe_min_coverage": smart_probe.probe_min_coverage,
            "max_step_chunks": smart_probe.max_step_chunks,
        }

    def get_cache_snapshot(self) -> Dict[str, Any]:
        return {
            "triage_log": self._last_triage_log,
            "probe_state": self._last_probe_state,
        }

    async def process(
        self,
        chunks: List[AudioChunk],
        job_dir: Optional[Path] = None,
        diagnosed_indices: Optional[set] = None,
    ) -> List[AudioChunk]:
        if not chunks:
            self.logger.warning("收到空的 chunk 列表，跳过音频预检")
            return chunks

        self.logger.info("开始音频预检，共 %d 个 chunk", len(chunks))
        token = self.cancellation_token
        diagnosed_indices = diagnosed_indices or set()
        sample_rate = chunks[0].sample_rate if chunks else 16000
        triage_log: List[Dict[str, Any]] = []

        if self.use_smart_probe:
            return await self._process_with_smart_probe(
                chunks=chunks,
                job_dir=job_dir,
                diagnosed_indices=diagnosed_indices,
                triage_log=triage_log,
                sample_rate=sample_rate,
                token=token,
            )

        return await self._process_standard(
            chunks=chunks,
            job_dir=job_dir,
            diagnosed_indices=diagnosed_indices,
            triage_log=triage_log,
            sample_rate=sample_rate,
            token=token,
        )

    @staticmethod
    def _build_chunk_log(chunk: AudioChunk, diagnosis) -> Dict[str, Any]:
        return {
            "chunk_index": chunk.index,
            "start_time": round(chunk.start, 3),
            "end_time": round(chunk.end, 3),
            "duration": round(chunk.end - chunk.start, 3),
            "need_separation": diagnosis.need_separation,
            "recommended_model": diagnosis.recommended_model,
            "reason": diagnosis.reason,
            "sig": round(float(diagnosis.sig), 3),
            "bak": round(float(diagnosis.bak), 3),
            "ovrl": round(float(diagnosis.ovrl), 3),
            "p808": round(float(diagnosis.p808), 3),
            "decision_source": diagnosis.decision_source,
            "decision_margin": round(float(diagnosis.decision_margin), 3),
            "triage_layer": diagnosis.triage_layer,
            # 兼容字段
            "snr": diagnosis.snr,
            "c50": diagnosis.c50,
            "snr_level": diagnosis.snr_level,
            "c50_level": diagnosis.c50_level,
        }

    async def _process_standard(
        self,
        chunks: List[AudioChunk],
        job_dir: Optional[Path],
        diagnosed_indices: set,
        triage_log: List[Dict[str, Any]],
        sample_rate: int,
        token,
    ) -> List[AudioChunk]:
        iterator = enumerate(chunks)
        if self.show_progress and TQDM_AVAILABLE:
            iterator = enumerate(
                tqdm(chunks, desc="音频预检", unit="chunk", ncols=80, leave=False)
            )

        for i, chunk in iterator:
            if i in diagnosed_indices:
                self.logger.debug("跳过已诊断 chunk %s", i)
                continue

            diagnosis = self.classifier.diagnose_chunk(audio=chunk.audio, chunk_index=i, sr=sample_rate)
            chunk.needs_separation = diagnosis.need_separation
            chunk.recommended_model = diagnosis.recommended_model
            chunk.spectrum_diagnosis = diagnosis
            triage_log.append(self._build_chunk_log(chunk, diagnosis))

            if token and job_dir and ((i + 1) % 5 == 0 or i == len(chunks) - 1):
                diagnosed_indices.add(i)
                checkpoint_data = {
                    "spectral_triage": {
                        "diagnosed_indices": list(diagnosed_indices),
                        "diagnosed_count": len(diagnosed_indices),
                        "total_chunks": len(chunks),
                    }
                }
                token.check_and_save(checkpoint_data, job_dir)

        self._last_triage_log = triage_log
        self._last_probe_state = None
        return self._finalize_triage(chunks, triage_log, job_dir)

    async def _process_with_smart_probe(
        self,
        chunks: List[AudioChunk],
        job_dir: Optional[Path],
        diagnosed_indices: set,
        triage_log: List[Dict[str, Any]],
        sample_rate: int,
        token,
    ) -> List[AudioChunk]:
        smart_probe = self._get_smart_probe()
        if smart_probe is None:
            self.logger.warning("智能探针不可用，回退标准音频预检")
            return await self._process_standard(
                chunks=chunks,
                job_dir=job_dir,
                diagnosed_indices=diagnosed_indices,
                triage_log=triage_log,
                sample_rate=sample_rate,
                token=token,
            )

        pbar = None
        if self.show_progress and TQDM_AVAILABLE:
            pbar = tqdm(total=len(chunks), desc="智能探针预检", unit="chunk", ncols=80, leave=False)

        def progress_callback(current: int, total: int) -> None:
            if pbar:
                pbar.n = current
                pbar.refresh()

        decision, cache, sequence = smart_probe.run_probe(chunks=chunks, progress_callback=progress_callback)
        if pbar:
            pbar.close()

        self._last_probe_state = {
            "decision": decision,
            "probe_sep_ratio_min": smart_probe.probe_sep_ratio_min,
            "probe_min_coverage": smart_probe.probe_min_coverage,
            "max_step_chunks": smart_probe.max_step_chunks,
            "sequence": sequence,
            "cache": cache,
            "cursor": len(sequence),
        }

        if decision == "SEPARATE_ALL":
            self.logger.info("智能探针判定：全量分离")
            for chunk in chunks:
                chunk.needs_separation = True
                chunk.recommended_model = "htdemucs"
                diag = chunk.spectrum_diagnosis
                if diag is not None:
                    diag.need_separation = True
                    diag.recommended_model = "htdemucs"
                    diag.decision_source = "probe_separate_all"
                triage_log.append(
                    {
                        "chunk_index": chunk.index,
                        "start_time": round(chunk.start, 3),
                        "end_time": round(chunk.end, 3),
                        "duration": round(chunk.end - chunk.start, 3),
                        "need_separation": True,
                        "recommended_model": "htdemucs",
                        "reason": "智能探针判定全量分离",
                        "decision_source": "probe_separate_all",
                        "decision_margin": 0.0,
                        "sig": None,
                        "bak": None,
                        "ovrl": None,
                        "p808": None,
                        "triage_layer": 0,
                    }
                )
            self._last_triage_log = triage_log
            return self._finalize_triage(chunks, triage_log, job_dir)

        if decision == "PASS_ALL":
            self.logger.info("智能探针判定：全量放行")
            for chunk in chunks:
                chunk.needs_separation = False
                chunk.recommended_model = None
                diag = chunk.spectrum_diagnosis
                if diag is not None:
                    diag.need_separation = False
                    diag.recommended_model = None
                    diag.decision_source = "probe_pass_all"
                triage_log.append(
                    {
                        "chunk_index": chunk.index,
                        "start_time": round(chunk.start, 3),
                        "end_time": round(chunk.end, 3),
                        "duration": round(chunk.end - chunk.start, 3),
                        "need_separation": False,
                        "recommended_model": None,
                        "reason": "智能探针判定全量放行",
                        "decision_source": "probe_pass_all",
                        "decision_margin": 0.0,
                        "sig": None,
                        "bak": None,
                        "ovrl": None,
                        "p808": None,
                        "triage_layer": 0,
                    }
                )
            self._last_triage_log = triage_log
            return self._finalize_triage(chunks, triage_log, job_dir)

        self.logger.info("智能探针判定：升级到标准逐 chunk 音频预检")
        return await self._process_standard(
            chunks=chunks,
            job_dir=job_dir,
            diagnosed_indices=diagnosed_indices,
            triage_log=triage_log,
            sample_rate=sample_rate,
            token=token,
        )

    def _finalize_triage(
        self,
        chunks: List[AudioChunk],
        triage_log: List[Dict[str, Any]],
        job_dir: Optional[Path],
    ) -> List[AudioChunk]:
        need_sep_count = sum(1 for chunk in chunks if chunk.needs_separation)
        self.logger.info("音频预检完成: %d/%d 个 chunk 需要分离", need_sep_count, len(chunks))
        if job_dir and triage_log:
            self._save_triage_log(job_dir, triage_log, chunks)
        return chunks

    def _save_triage_log(self, job_dir: Path, triage_log: List[Dict[str, Any]], chunks: List[AudioChunk]) -> None:
        try:
            stats = self.get_statistics(chunks)
            layer_stats = {
                "layer_0_probe": sum(1 for item in triage_log if item.get("triage_layer") == 0),
                "layer_1_dnsmos": sum(1 for item in triage_log if item.get("triage_layer") == 1),
                "layer_2_fallback": sum(1 for item in triage_log if item.get("triage_layer") == 2),
            }
            decision_source_stats: Dict[str, int] = {}
            for item in triage_log:
                source = str(item.get("decision_source") or "unknown")
                decision_source_stats[source] = decision_source_stats.get(source, 0) + 1

            def _build_numeric_stats(key: str) -> Dict[str, float]:
                values = [float(item[key]) for item in triage_log if item.get(key) is not None]
                if not values:
                    return {}
                return {
                    "min": round(min(values), 3),
                    "max": round(max(values), 3),
                    "avg": round(sum(values) / len(values), 3),
                }

            full_log = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_chunks": len(chunks),
                    "triage_version": self.triage_version,
                    "use_dnsmos_triage": self.use_dnsmos_triage,
                    "use_smart_probe": self.use_smart_probe,
                    "dnsmos_model_hash": self.get_dnsmos_model_hash(),
                    "dnsmos_threshold_profile_hash": self.get_dnsmos_threshold_profile_hash(),
                },
                "summary": {
                    **stats,
                    "layer_distribution": layer_stats,
                    "decision_source_distribution": decision_source_stats,
                    "sig_statistics": _build_numeric_stats("sig"),
                    "bak_statistics": _build_numeric_stats("bak"),
                    "ovrl_statistics": _build_numeric_stats("ovrl"),
                    "p808_statistics": _build_numeric_stats("p808"),
                },
                "probe": self._last_probe_state or {},
                "chunks": triage_log,
            }

            log_file = job_dir / "triage_log.json"
            with open(log_file, "w", encoding="utf-8") as file:
                json.dump(full_log, file, indent=2, ensure_ascii=False)
            self.logger.info("音频预检日志已保存: %s", log_file)
        except Exception as exc:
            self.logger.warning("保存音频预检日志失败: %s", exc)

    @staticmethod
    def get_statistics(chunks: List[AudioChunk]) -> dict:
        total = len(chunks)
        need_sep = sum(1 for chunk in chunks if chunk.needs_separation)
        htdemucs = sum(
            1
            for chunk in chunks
            if chunk.needs_separation and chunk.recommended_model == "htdemucs"
        )
        mdx_extra = sum(
            1
            for chunk in chunks
            if chunk.needs_separation and chunk.recommended_model == "mdx_extra"
        )
        return {
            "total_chunks": total,
            "need_separation": need_sep,
            "no_separation": total - need_sep,
            "recommended_htdemucs": htdemucs,
            "recommended_mdx_extra": mdx_extra,
            "separation_ratio": need_sep / total if total > 0 else 0.0,
        }
