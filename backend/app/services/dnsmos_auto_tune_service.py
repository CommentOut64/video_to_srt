"""
DNSMOS 自动调参服务。

职责：
1. 统一维护 DNSMOS 音频预检自动调参参数分层；
2. 生成 Optuna Trial 的搜索空间；
3. 校验并回灌调参结果到 runtime 配置。

V3.2.4+dev.20260224.02: 新增 DNSMOS 自动调参参数分层与回灌能力。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple

from app.services.model_runtime_config_service import (
    ModelRuntimeConfigService,
    get_model_runtime_config_service,
)

# 核心：必须自动调参
MANDATORY_AUTO_TUNE_PARAMS: Tuple[str, ...] = (
    "ovrl_sep_hard",
    "sig_sep_hard",
    "p808_sep_hard",
    "ovrl_pass_hard",
    "sig_pass_hard",
    "bak_pass_hard",
    "p808_pass_soft",
)

# 建议自动调参（时延/吞吐）
RECOMMENDED_AUTO_TUNE_PARAMS: Tuple[str, ...] = (
    "probe_sep_ratio_min",
    "probe_min_coverage",
)

# 固定参数（当前不作为自动调参重点）
FIXED_PARAMS: Tuple[str, ...] = (
    "yamnet_music_conf_min",
    "yamnet_speech_conf_min",
    "yamnet_music_weak_max",
    "probe_max_step_chunks",
)


class TrialLike(Protocol):
    """最小 Trial 协议，兼容 optuna.trial.Trial。"""

    def suggest_float(self, name: str, low: float, high: float) -> float:
        """建议一个浮点参数。"""


@dataclass(frozen=True)
class DNSMOSAutoTunePlan:
    """DNSMOS 自动调参参数分层快照。"""

    mandatory: Tuple[str, ...] = MANDATORY_AUTO_TUNE_PARAMS
    recommended: Tuple[str, ...] = RECOMMENDED_AUTO_TUNE_PARAMS
    fixed: Tuple[str, ...] = FIXED_PARAMS

    def to_dict(self) -> Dict[str, Tuple[str, ...]]:
        """转字典，供 API 直接输出。"""
        return {
            "mandatory": self.mandatory,
            "recommended": self.recommended,
            "fixed": self.fixed,
        }


@dataclass(frozen=True)
class DNSMOSObjectiveMetrics:
    """自动调参目标函数输入指标。"""

    false_negative: int
    false_positive: int
    avg_latency_ms: float
    fallback_rate: float

    def score(self) -> float:
        """
        计算目标函数值（越小越好）。

        objective = 12*FN + 1*FP + 0.15*AvgLatencyMs + 0.05*FallbackRate
        """
        return (
            12.0 * float(self.false_negative)
            + 1.0 * float(self.false_positive)
            + 0.15 * float(self.avg_latency_ms)
            + 0.05 * float(self.fallback_rate)
        )


class DNSMOSAutoTuneService:
    """DNSMOS 自动调参服务。"""

    _RANGES: Dict[str, Tuple[float, float]] = {
        # 必须自动调参（核心）
        "ovrl_sep_hard": (1.0, 2.4),
        "sig_sep_hard": (1.0, 2.6),
        "p808_sep_hard": (1.0, 2.4),
        "ovrl_pass_hard": (1.2, 3.8),
        "sig_pass_hard": (1.2, 4.2),
        "bak_pass_hard": (1.8, 4.0),
        "p808_pass_soft": (1.2, 3.8),
        # 建议自动调参（时延/吞吐）
        "probe_sep_ratio_min": (0.2, 0.8),
        "probe_min_coverage": (0.15, 0.6),
    }

    def __init__(self, runtime_service: Optional[ModelRuntimeConfigService] = None) -> None:
        self.runtime_service = runtime_service or get_model_runtime_config_service()

    @staticmethod
    def get_plan() -> DNSMOSAutoTunePlan:
        """返回参数分层定义。"""
        return DNSMOSAutoTunePlan()

    def build_trial_params(self, trial: TrialLike, include_recommended: bool = True) -> Dict[str, float]:
        """
        构建一次 Trial 参数。

        说明：
        - `pass_hard` 系列与 `sep_hard` 保持最小间隔 0.2，避免无效搜索空间。
        """
        params: Dict[str, float] = {}
        params["ovrl_sep_hard"] = trial.suggest_float("ovrl_sep_hard", *self._RANGES["ovrl_sep_hard"])
        params["sig_sep_hard"] = trial.suggest_float("sig_sep_hard", *self._RANGES["sig_sep_hard"])
        params["p808_sep_hard"] = trial.suggest_float("p808_sep_hard", *self._RANGES["p808_sep_hard"])

        params["ovrl_pass_hard"] = trial.suggest_float(
            "ovrl_pass_hard",
            max(self._RANGES["ovrl_pass_hard"][0], params["ovrl_sep_hard"] + 0.2),
            self._RANGES["ovrl_pass_hard"][1],
        )
        params["sig_pass_hard"] = trial.suggest_float(
            "sig_pass_hard",
            max(self._RANGES["sig_pass_hard"][0], params["sig_sep_hard"] + 0.2),
            self._RANGES["sig_pass_hard"][1],
        )
        params["bak_pass_hard"] = trial.suggest_float("bak_pass_hard", *self._RANGES["bak_pass_hard"])
        params["p808_pass_soft"] = trial.suggest_float(
            "p808_pass_soft",
            max(self._RANGES["p808_pass_soft"][0], params["p808_sep_hard"] + 0.2),
            self._RANGES["p808_pass_soft"][1],
        )

        if include_recommended:
            params["probe_sep_ratio_min"] = trial.suggest_float(
                "probe_sep_ratio_min",
                *self._RANGES["probe_sep_ratio_min"],
            )
            params["probe_min_coverage"] = trial.suggest_float(
                "probe_min_coverage",
                *self._RANGES["probe_min_coverage"],
            )
        return params

    def validate_candidate(self, params: Dict[str, float], include_recommended: bool = True) -> Dict[str, float]:
        """校验候选参数并返回规范化结果。"""
        normalized: Dict[str, float] = {}
        required = list(MANDATORY_AUTO_TUNE_PARAMS)
        if include_recommended:
            required.extend(RECOMMENDED_AUTO_TUNE_PARAMS)

        for key in required:
            if key not in params:
                raise ValueError(f"缺少自动调参参数: {key}")
            low, high = self._RANGES[key]
            value = float(params[key])
            if value < low or value > high:
                raise ValueError(f"参数超出范围: {key}={value} 不在 [{low}, {high}]")
            normalized[key] = value

        if normalized["ovrl_pass_hard"] <= normalized["ovrl_sep_hard"] + 0.2:
            raise ValueError("ovrl_pass_hard 必须 > ovrl_sep_hard + 0.2")
        if normalized["sig_pass_hard"] <= normalized["sig_sep_hard"] + 0.2:
            raise ValueError("sig_pass_hard 必须 > sig_sep_hard + 0.2")
        if normalized["p808_pass_soft"] <= normalized["p808_sep_hard"] + 0.2:
            raise ValueError("p808_pass_soft 必须 > p808_sep_hard + 0.2")

        for fixed_key in FIXED_PARAMS:
            if fixed_key in params:
                raise ValueError(f"{fixed_key} 当前为固定参数，不应纳入自动调参")
        return normalized

    def build_runtime_payload(
        self,
        params: Dict[str, float],
        include_recommended: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """将调参结果映射为 runtime 更新载荷。"""
        normalized = self.validate_candidate(params=params, include_recommended=include_recommended)
        payload: Dict[str, Dict[str, float]] = {
            "dnsmos": {key: normalized[key] for key in MANDATORY_AUTO_TUNE_PARAMS},
        }
        if include_recommended:
            payload["smart_probe"] = {key: normalized[key] for key in RECOMMENDED_AUTO_TUNE_PARAMS}
        return payload

    def apply_best_params(
        self,
        params: Dict[str, float],
        include_recommended: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """回灌最优参数到 `model_runtime_config.json`。"""
        payload = self.build_runtime_payload(params=params, include_recommended=include_recommended)
        self.runtime_service.update_runtime_global(payload)
        return payload

    def get_fixed_runtime_snapshot(self) -> Dict[str, float | int]:
        """读取当前固定参数快照，便于记录调参上下文。"""
        runtime = self.runtime_service.get_effective_runtime_global().get("effective", {})
        yamnet = dict(runtime.get("yamnet", {}))
        smart_probe = dict(runtime.get("smart_probe", {}))
        return {
            "yamnet_music_conf_min": float(yamnet.get("yamnet_music_conf_min", 0.15)),
            "yamnet_speech_conf_min": float(yamnet.get("yamnet_speech_conf_min", 0.85)),
            "yamnet_music_weak_max": float(yamnet.get("yamnet_music_weak_max", 0.10)),
            "probe_max_step_chunks": int(smart_probe.get("probe_max_step_chunks", 30)),
        }


def get_dnsmos_auto_tune_service(
    runtime_service: Optional[ModelRuntimeConfigService] = None,
) -> DNSMOSAutoTuneService:
    """构造 DNSMOS 自动调参服务。"""
    return DNSMOSAutoTuneService(runtime_service=runtime_service)
