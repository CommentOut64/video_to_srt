"""
任务启动模型就绪守卫。

用途：
1. background 自愈模式下，避免“任务一启动就因模型未下载完而 failed”；
2. 在任务启动入口提供两种策略：
   - 返回 409（不阻塞请求）
   - 等待模型就绪（阻塞该请求，带超时）

配置：
- MODEL_TASK_GUARD_MODE=409|wait|off（默认 409）
- MODEL_TASK_GUARD_TIMEOUT_SEC：wait 模式等待超时秒数（默认 600）
- MODEL_TASK_GUARD_POLL_SEC：wait 模式轮询间隔秒数（默认 0.5）
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

if False:  # TYPE_CHECKING（避免运行时循环导入）
    from app.services.model_bootstrap_service import ModelBootstrapService
    from app.services.model_manager_v2 import ModelManagerV2


GuardMode = Literal["409", "wait", "off"]


def resolve_model_task_guard_mode_from_env() -> GuardMode:
    raw = str(os.getenv("MODEL_TASK_GUARD_MODE", "")).strip().lower()
    if raw in {"409", "conflict"}:
        return "409"
    if raw in {"wait", "block"}:
        return "wait"
    if raw in {"off", "0", "false", "no"}:
        return "off"
    return "409"


def resolve_model_task_guard_timeout_sec_from_env(mode: GuardMode) -> float:
    raw = str(os.getenv("MODEL_TASK_GUARD_TIMEOUT_SEC", "")).strip()
    if not raw:
        return 600.0 if mode == "wait" else 0.0
    try:
        value = float(raw)
    except ValueError:
        return 600.0 if mode == "wait" else 0.0
    return max(0.0, value)


def resolve_model_task_guard_poll_sec_from_env() -> float:
    raw = str(os.getenv("MODEL_TASK_GUARD_POLL_SEC", "")).strip()
    if not raw:
        return 0.5
    try:
        value = float(raw)
    except ValueError:
        return 0.5
    return max(0.1, value)


async def check_models_ready(
    model_manager: "ModelManagerV2",
    model_ids: List[str],
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    通过 ModelManagerV2.model_status 做真实文件校验（不触发下载）。

    Returns:
        - is_ready: 是否全部 ready
        - status_by_id: model_status 原样字典
        - not_ready_ids: 非 ready 的模型 ID 列表
    """
    status_by_id: Dict[str, Any] = {}
    not_ready_ids: List[str] = []
    for model_id in model_ids:
        payload = await asyncio.to_thread(model_manager.model_status, model_id)
        status_by_id[model_id] = payload
        if str(payload.get("status", "")) != "ready":
            not_ready_ids.append(model_id)
    return len(not_ready_ids) == 0, status_by_id, not_ready_ids


async def enforce_required_models_ready(
    *,
    model_manager: "ModelManagerV2",
    bootstrap_service: "ModelBootstrapService",
    mode: GuardMode,
    timeout_sec: float,
    poll_sec: float,
    required_model_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    在任务启动前确保“必需模型”就绪。

    - mode=409: 只检查一次，不等待。
    - mode=wait: 若未就绪，则触发后台自愈并等待到 ready 或超时。
    - mode=off: 直接跳过。

    返回值用于 API detail（可直接 JSON 化）。
    """
    if mode == "off":
        return {
            "mode": mode,
            "required_models": [],
            "is_ready": True,
            "not_ready_models": [],
            "models": {},
            "waited_sec": 0.0,
        }

    known_ids = {spec.id for spec in model_manager.registry.list()}
    required = [
        model_id
        for model_id in list(required_model_ids or bootstrap_service.resolve_required_model_ids())
        if model_id in known_ids
    ]
    started_at = time.time()

    is_ready, status_by_id, not_ready = await check_models_ready(model_manager, required)
    if is_ready or mode == "409":
        return {
            "mode": mode,
            "required_models": required,
            "is_ready": bool(is_ready),
            "not_ready_models": not_ready,
            "models": status_by_id,
            "waited_sec": round(time.time() - started_at, 3),
        }

    # mode == "wait"
    # 先触发一次后台自愈，避免用户手动重试。
    bootstrap_service.start_non_blocking(
        model_ids=required,
        is_force=False,
        is_required_on_boot=True,
    )

    deadline = started_at + max(0.0, float(timeout_sec or 0.0))
    while True:
        now = time.time()
        if timeout_sec > 0.0 and now >= deadline:
            break
        await asyncio.sleep(poll_sec)
        is_ready, status_by_id, not_ready = await check_models_ready(model_manager, required)
        if is_ready:
            break

    return {
        "mode": mode,
        "required_models": required,
        "is_ready": bool(is_ready),
        "not_ready_models": not_ready,
        "models": status_by_id,
        "waited_sec": round(time.time() - started_at, 3),
    }


def resolve_required_model_ids_for_job_settings(
    *,
    model_manager: "ModelManagerV2",
    bootstrap_service: "ModelBootstrapService",
    job_settings: Optional[Any],
) -> List[str]:
    """
    按任务配置推断“真正会用到的模型”，避免把所有 asr(v1) 都当必需模型。
    """
    known_ids = {spec.id for spec in model_manager.registry.list()}

    # 1) 先纳入“预置必需模型”（backend/models/pretrained 对应的集合）。
    required: List[str] = list(bootstrap_service.resolve_required_model_ids())

    # 2) 任务级增量：Whisper 仅在 patch/dual 模式需要；Demucs/DNSMOS/speaker 等如果用户关了则可不强制。
    preprocessing = getattr(job_settings, "preprocessing", None) if job_settings is not None else None
    demucs_strategy = str(getattr(preprocessing, "demucs_strategy", "auto") or "auto").strip().lower()
    if demucs_strategy == "off":
        # 移除可能存在的预置 demucs（允许用户显式关闭分离时不阻断任务）
        required = [mid for mid in required if not str(mid).startswith("demucs-")]

    # 音频预检：DNSMOS（开启时）
    enable_spectral_triage = bool(getattr(preprocessing, "enable_spectral_triage", True)) if preprocessing is not None else True
    use_dnsmos_triage = bool(getattr(preprocessing, "use_dnsmos_triage", True)) if preprocessing is not None else True
    if not (enable_spectral_triage and use_dnsmos_triage):
        required = [mid for mid in required if mid != "dnsmos-quality"]

    # 说话人：开启 speaker detection 时，预置 pyannote 模型应视为必需（否则任务中途会失败）
    is_enable_speaker_detection = bool(getattr(preprocessing, "is_enable_speaker_detection", True)) if preprocessing is not None else True
    if not is_enable_speaker_detection:
        required = [
            mid
            for mid in required
            if mid not in {"pyannote-segmentation-3-0", "pyannote-speaker-diarization-community-1"}
        ]

    # Whisper：仅在 patch/dual 模式需要
    transcription = getattr(job_settings, "transcription", None) if job_settings is not None else None
    transcription_profile = str(getattr(transcription, "transcription_profile", "sensevoice_only") or "sensevoice_only").strip()
    if transcription_profile in {"sv_whisper_patch", "sv_whisper_dual"}:
        whisper_model = str(getattr(transcription, "whisper_model", "medium") or "medium").strip()
        whisper_id = f"whisper-{whisper_model.replace('.', '-')}"
        if whisper_id in known_ids:
            required.append(whisper_id)

    # 去重且保持顺序
    deduped: List[str] = []
    for model_id in required:
        if model_id in known_ids and model_id not in deduped:
            deduped.append(model_id)
    return deduped
