"""
任务生命周期灰度开关配置。

说明：
- Phase 0-5 共用该配置文件；
- 默认全部开启，可通过环境变量按需回退。
"""
from __future__ import annotations

import os


def _read_bool_env(env_name: str, default_value: bool) -> bool:
    """读取布尔环境变量。"""
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default_value
    normalized_value = str(raw_value).strip().lower()
    return normalized_value in {"1", "true", "yes", "on"}


LIFECYCLE_V2_ENABLED: bool = _read_bool_env("LIFECYCLE_V2_ENABLED", True)
STATE_MACHINE_GUARD_ENABLED: bool = _read_bool_env("STATE_MACHINE_GUARD_ENABLED", True)
CANCEL_V2_ENABLED: bool = _read_bool_env("CANCEL_V2_ENABLED", True)
RESUME_MERGER_ENABLED: bool = _read_bool_env("RESUME_MERGER_ENABLED", True)
RUNNER_GATE_ENABLED: bool = _read_bool_env("RUNNER_GATE_ENABLED", True)
RUNNER_DETACH_ON_FORCE_CANCEL_ENABLED: bool = _read_bool_env(
    "RUNNER_DETACH_ON_FORCE_CANCEL_ENABLED",
    True,
)
