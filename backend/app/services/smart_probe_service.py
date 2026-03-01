"""
SmartProbeService - DNSMOS 驱动的智能探针服务。

策略：
- 先中心后扩散抽样；
- 按抽样结果判断 `SEPARATE_ALL / PASS_ALL / ESCALATE_STANDARD`；
- 只做快判，不替代标准逐 chunk 音频预检。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from app.services.runtime_param_resolver import get_dnsmos_runtime_params, get_smart_probe_runtime_params

logger = logging.getLogger(__name__)


class SmartProbeService:
    """DNSMOS 智能探针服务。"""

    def __init__(
        self,
        dnsmos_service,
        probe_sep_ratio_min: float = 0.40,
        probe_min_coverage: float = 0.30,
        max_step_chunks: int = 30,
    ) -> None:
        self.dnsmos = dnsmos_service
        self.probe_sep_ratio_min = probe_sep_ratio_min
        self.probe_min_coverage = probe_min_coverage
        self.max_step_chunks = max_step_chunks

    @staticmethod
    def _fibonacci_generator():
        a, b = 0, 1
        while True:
            yield b
            a, b = b, a + b

    @staticmethod
    def _get_dnsmos_thresholds() -> Dict[str, float]:
        runtime = get_dnsmos_runtime_params()
        return {
            "ovrl_sep_hard": float(runtime.get("ovrl_sep_hard", 2.0)),
            "sig_sep_hard": float(runtime.get("sig_sep_hard", 2.1)),
            "p808_sep_hard": float(runtime.get("p808_sep_hard", 2.0)),
            "ovrl_pass_hard": float(runtime.get("ovrl_pass_hard", 3.1)),
            "sig_pass_hard": float(runtime.get("sig_pass_hard", 3.2)),
            "bak_pass_hard": float(runtime.get("bak_pass_hard", 3.0)),
            "p808_pass_soft": float(runtime.get("p808_pass_soft", 2.9)),
        }

    def _check_chunk(self, chunk) -> Dict[str, object]:
        thresholds = self._get_dnsmos_thresholds()
        result = self.dnsmos.detect(chunk.audio, sr=chunk.sample_rate, chunk_id=chunk.index)
        if not result.is_valid:
            return {
                "sig": None,
                "bak": None,
                "ovrl": None,
                "p808": None,
                "is_hard_separate": False,
                "is_hard_pass": False,
                "probe_decision": "gray",
                "calculated": False,
            }

        is_hard_separate = (
            result.ovrl <= thresholds["ovrl_sep_hard"]
            or result.sig <= thresholds["sig_sep_hard"]
            or result.p808 <= thresholds["p808_sep_hard"]
        )
        is_hard_pass = (
            result.ovrl >= thresholds["ovrl_pass_hard"]
            and result.sig >= thresholds["sig_pass_hard"]
            and result.bak >= thresholds["bak_pass_hard"]
            and result.p808 >= thresholds["p808_pass_soft"]
        )
        if is_hard_separate:
            probe_decision = "separate"
        elif is_hard_pass:
            probe_decision = "pass"
        else:
            probe_decision = "gray"

        return {
            "sig": float(result.sig),
            "bak": float(result.bak),
            "ovrl": float(result.ovrl),
            "p808": float(result.p808),
            "is_hard_separate": is_hard_separate,
            "is_hard_pass": is_hard_pass,
            "probe_decision": probe_decision,
            "calculated": True,
        }

    def _evaluate_state(self, n_chunks: int, cache: Dict[int, Dict[str, object]]) -> str:
        if not cache:
            return "ESCALATE_STANDARD"
        values = list(cache.values())
        separate_count = sum(1 for item in values if bool(item.get("is_hard_separate", False)))
        pass_count = sum(1 for item in values if bool(item.get("is_hard_pass", False)))
        probed = len(values)
        coverage = probed / max(1, n_chunks)
        sep_ratio = separate_count / max(1, probed)

        if sep_ratio >= self.probe_sep_ratio_min:
            return "SEPARATE_ALL"
        if coverage >= self.probe_min_coverage and pass_count == probed:
            return "PASS_ALL"
        return "ESCALATE_STANDARD"

    def run_probe(
        self,
        chunks: List,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[str, Dict[int, Dict[str, object]], List[int]]:
        """
        执行中心扩散探针。

        返回：
        - decision: `SEPARATE_ALL` | `PASS_ALL` | `ESCALATE_STANDARD`
        - cache: 已探测 chunk 结果缓存
        - probe_sequence: 探测顺序
        """
        n_chunks = len(chunks)
        if n_chunks == 0:
            return "PASS_ALL", {}, []

        center = n_chunks // 2
        cache: Dict[int, Dict[str, object]] = {}
        probe_sequence: List[int] = []
        visited = set()

        def _probe(index: int) -> None:
            if index < 0 or index >= n_chunks or index in visited:
                return
            visited.add(index)
            cache[index] = self._check_chunk(chunks[index])
            probe_sequence.append(index)
            if progress_callback:
                progress_callback(len(visited), n_chunks)

        # 1. 中心点
        _probe(center)
        decision = self._evaluate_state(n_chunks=n_chunks, cache=cache)
        if decision in {"SEPARATE_ALL", "PASS_ALL"}:
            return decision, cache, probe_sequence

        # 2. 斐波那契扩散
        fib_gen = self._fibonacci_generator()
        cumulative_radius = 0
        while len(visited) < n_chunks:
            step = min(next(fib_gen), self.max_step_chunks)
            cumulative_radius += step
            left = center - cumulative_radius
            right = center + cumulative_radius
            if left < 0 and right >= n_chunks:
                break
            _probe(left)
            _probe(right)
            decision = self._evaluate_state(n_chunks=n_chunks, cache=cache)
            if decision in {"SEPARATE_ALL", "PASS_ALL"}:
                return decision, cache, probe_sequence

        # 3. 最终判定
        decision = self._evaluate_state(n_chunks=n_chunks, cache=cache)
        if decision not in {"SEPARATE_ALL", "PASS_ALL"}:
            decision = "ESCALATE_STANDARD"
        logger.info(
            "智能探针完成: decision=%s, probed=%d/%d",
            decision,
            len(cache),
            n_chunks,
        )
        return decision, cache, probe_sequence


_smart_probe_instance: Optional[SmartProbeService] = None


def get_smart_probe_service(
    probe_sep_ratio_min: Optional[float] = None,
    probe_min_coverage: Optional[float] = None,
    max_step_chunks: Optional[int] = None,
) -> SmartProbeService:
    """获取智能探针服务单例。"""
    global _smart_probe_instance
    runtime = get_smart_probe_runtime_params()
    effective_sep_ratio = (
        probe_sep_ratio_min
        if probe_sep_ratio_min is not None
        else float(runtime.get("probe_sep_ratio_min", 0.40))
    )
    effective_min_coverage = (
        probe_min_coverage
        if probe_min_coverage is not None
        else float(runtime.get("probe_min_coverage", 0.30))
    )
    effective_max_step = (
        max_step_chunks
        if max_step_chunks is not None
        else int(runtime.get("probe_max_step_chunks", 30))
    )

    if _smart_probe_instance is None:
        from app.services.dnsmos_service import get_dnsmos_service

        dnsmos = get_dnsmos_service()
        _smart_probe_instance = SmartProbeService(
            dnsmos_service=dnsmos,
            probe_sep_ratio_min=effective_sep_ratio,
            probe_min_coverage=effective_min_coverage,
            max_step_chunks=effective_max_step,
        )
    else:
        _smart_probe_instance.probe_sep_ratio_min = effective_sep_ratio
        _smart_probe_instance.probe_min_coverage = effective_min_coverage
        _smart_probe_instance.max_step_chunks = effective_max_step
    return _smart_probe_instance


def reset_smart_probe_service() -> None:
    """重置智能探针服务单例。"""
    global _smart_probe_instance
    _smart_probe_instance = None
