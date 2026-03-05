"""
转录任务相关API路由 - v3.5 重构版

支持 1+3 预设模式:
- 顶层: 3个快捷场景宏 (fast/balanced/quality)
- 底层: 4个设置分组 (preprocessing/transcription/refinement/compute)
"""
import os
import re
import shutil
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal, Set
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request, Body
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import json

from app.core.config import config
from app.models.job_models import JobSettings, JobState
from app.models.project_models import infer_task_mode
from app.services.transcription_service import TranscriptionService
from app.services.file_service import FileManagementService
from app.services.sse_service import get_sse_manager
from app.services.job_queue_service import get_queue_service
from app.services.media_prep_service import get_media_prep_service
from app.api.routes.media_routes import _find_video_file
from app.services.transcription_recovery_utils import (
    force_finalize_segments_when_finished,
    force_finalize_snapshot_when_finished,
)
from app.services.model_task_guard_service import (
    enforce_required_models_ready,
    resolve_model_task_guard_mode_from_env,
    resolve_model_task_guard_poll_sec_from_env,
    resolve_model_task_guard_timeout_sec_from_env,
    resolve_required_model_ids_for_job_settings,
)


# ========== v3.5 新版 API 模型 ==========

class PreprocessingSettingsAPI(BaseModel):
    """
    分组一: 预处理与音频设置 API 模型
    """
    # 人声分离策略: off/auto/force_on
    demucs_strategy: str = Field(default="auto", description="人声分离策略")
    # 分离模型: htdemucs/htdemucs_ft/mdx_q/mdx_extra
    demucs_model: str = Field(default="htdemucs", description="Demucs 模型")
    # 分离预测次数: 1-5
    demucs_shifts: int = Field(default=1, ge=1, le=5, description="分离预测次数")
    # 分离模式: global/on_demand
    separation_mode: str = Field(default="on_demand", description="人声分离模式")
    # 是否启用频谱分诊（直通模式应设为 false）
    enable_spectral_triage: bool = Field(default=True, description="是否启用频谱分诊")
    # 分诊灵敏度: 0.0-1.0
    spectrum_threshold: float = Field(default=0.35, ge=0.0, le=1.0, description="分诊灵敏度")
    # 是否启用 DNSMOS 音频预检主链
    use_dnsmos_triage: bool = Field(default=True, description="是否启用 DNSMOS 音频预检")
    # 是否启用中心扩散智能探针（默认开启）
    use_smart_probe: bool = Field(default=True, description="是否启用中心扩散智能探针")
    # 是否启用 ASR 风险前置守卫（命中后强制全局分离）
    enable_asr_risk_guard: bool = Field(default=True, description="是否启用 ASR 风险前置守卫")
    # VAD 静音过滤
    vad_filter: bool = Field(default=True, description="VAD 静音过滤")
    # 是否启用说话人检测（任务级）
    enable_speaker_detection: bool = Field(default=True, description="是否启用说话人检测")
    # 是否启用 speaker 介入切分（任务级）
    enable_speaker_guided_split: bool = Field(default=True, description="是否启用 speaker 介入切分")
    # 手动指定说话人数：0=auto
    speaker_count: int = Field(default=0, ge=0, le=20, description="说话人数，0 表示自动推断")
    # 可选人数范围：0=auto，仅在 speaker_count=0 时生效
    speaker_min_count: int = Field(default=0, ge=0, le=20, description="最小说话人数，0 表示自动")
    speaker_max_count: int = Field(default=0, ge=0, le=20, description="最大说话人数，0 表示自动")


class TranscriptionSettingsAPI(BaseModel):
    """
    分组二: 转录核心设置 API 模型
    """
    # 转录流水线模式: sensevoice_only/sv_whisper_patch/sv_whisper_dual
    transcription_profile: str = Field(default="sensevoice_only", description="转录流水线模式")
    # 主引擎运行设备: auto/cpu
    sensevoice_device: str = Field(default="auto", description="SenseVoice 运行设备")
    # 辅助/复核模型（临时停用任务级覆盖，当前统一读取 .env）
    whisper_model: Optional[str] = Field(default=None, description="Whisper 模型（临时停用任务级覆盖）")
    # 复核触发阈值: 0.0-1.0
    patching_threshold: float = Field(default=0.60, ge=0.0, le=1.0, description="复核触发阈值")


class TaskSubtitleTimeOffsetRequest(BaseModel):
    """任务字幕时间偏移请求"""
    offset: float = Field(
        ...,
        ge=-10.0,
        le=10.0,
        description="偏移量（秒），正值延后，负值提前，范围 -10.0 到 10.0"
    )


class RefinementSettingsAPI(BaseModel):
    """
    分组三: 增强与润色设置 API 模型
    """
    # LLM 任务目标: off/proofread/translate
    llm_task: str = Field(default="off", description="LLM 任务目标")
    # 介入范围: sparse/global
    llm_scope: str = Field(default="sparse", description="LLM 介入范围")
    # 稀疏校对阈值: 0.0-1.0
    sparse_threshold: float = Field(default=0.70, ge=0.0, le=1.0, description="稀疏校对阈值")
    # 目标语言
    target_language: str = Field(default="zh", description="目标语言")
    # 模型提供商: openai_compatible/local_ollama
    llm_provider: str = Field(default="openai_compatible", description="LLM 提供商")
    # 模型名称
    llm_model_name: str = Field(default="gpt-4o-mini", description="LLM 模型名称")


class ComputeSettingsAPI(BaseModel):
    """
    分组四: 计算与系统设置 API 模型
    """
    # 并发调度策略: auto/parallel/serial
    concurrency_strategy: str = Field(default="auto", description="并发调度策略")
    # GPU 选择
    gpu_id: int = Field(default=0, ge=0, description="GPU ID")
    # 输出格式列表
    output_formats: List[str] = Field(default=["srt"], description="输出格式")
    # 临时文件策略: delete_on_complete/keep
    temp_file_policy: str = Field(default="delete_on_complete", description="临时文件策略")


class DebugSettingsAPI(BaseModel):
    """
    调试配置 API 模型
    """
    punctuation_output: bool = Field(default=False, description="标点调试输出（SSE + 文件）")


class TaskConfigAPI(BaseModel):
    """
    v3.5 完整任务配置 API 模型
    整合所有设置分组
    """
    # 选择的宏预设 ID (fast/balanced/quality/custom)
    preset_id: str = Field(default="balanced", description="预设 ID")
    # 四个设置分组
    preprocessing: Optional[PreprocessingSettingsAPI] = None
    transcription: Optional[TranscriptionSettingsAPI] = None
    refinement: Optional[RefinementSettingsAPI] = None
    compute: Optional[ComputeSettingsAPI] = None
    debug: Optional[DebugSettingsAPI] = None


class TranscribeSettings(BaseModel):
    """
    转录设置请求模型 - v3.5 重构版

    仅支持新版 task_config 字段
    """
    # === 新版 1+3 预设配置 ===
    task_config: Optional[TaskConfigAPI] = Field(
        default=None,
        description="v3.5 任务配置 (推荐使用)"
    )


class UploadResponse(BaseModel):
    """上传响应模型"""
    job_id: str
    project_id: str
    filename: str
    original_name: str
    message: str


class CreateJobsBatchRequest(BaseModel):
    """批量创建任务请求。"""

    filenames: List[str] = Field(default_factory=list, description="文件名列表")
    task_config: Optional[TaskConfigAPI] = Field(default=None, description="任务级配置（可选）")


def create_transcription_router(
    transcription_service: TranscriptionService,
    file_service: FileManagementService,
    output_dir: str
):
    """创建转录任务路由"""

    # 创建路由器实例
    router = APIRouter(prefix="/api", tags=["transcription"])

    # 获取SSE管理器
    sse_manager = get_sse_manager()
    from app.services.project_id_resolver import get_project_id_resolver

    def _parse_task_config_form(task_config_raw: Optional[str]) -> Dict[str, Any]:
        """解析 Form 中的 task_config JSON。"""
        if not task_config_raw:
            return {}
        try:
            payload = json.loads(task_config_raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"task_config 不是合法 JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="task_config 必须是对象结构")
        if "task_config" in payload and isinstance(payload["task_config"], dict):
            return dict(payload["task_config"])
        return payload

    async def _guard_models_before_enqueue(job_settings: Optional[JobSettings] = None) -> None:
        """
        任务启动前模型守卫：
        - MODEL_TASK_GUARD_MODE=409: 未就绪直接 409
        - MODEL_TASK_GUARD_MODE=wait: 等待就绪（带超时）
        - MODEL_TASK_GUARD_MODE=off: 不拦截
        """
        guard_mode = resolve_model_task_guard_mode_from_env()
        if guard_mode == "off":
            return

        try:
            from app.services.model_manager_v2 import get_model_manager_v2
            from app.services.model_bootstrap_service import get_model_bootstrap_service
        except Exception:
            # Lite / Full 依赖缺失时，转录入口本就不可用或会走降级，这里不再额外阻断。
            return

        model_manager = get_model_manager_v2()
        bootstrap_service = get_model_bootstrap_service(model_manager=model_manager)
        timeout_sec = resolve_model_task_guard_timeout_sec_from_env(guard_mode)
        poll_sec = resolve_model_task_guard_poll_sec_from_env()
        required_ids = resolve_required_model_ids_for_job_settings(
            model_manager=model_manager,
            bootstrap_service=bootstrap_service,
            job_settings=job_settings,
        )
        decision = await enforce_required_models_ready(
            model_manager=model_manager,
            bootstrap_service=bootstrap_service,
            mode=guard_mode,
            timeout_sec=timeout_sec,
            poll_sec=poll_sec,
            required_model_ids=required_ids,
        )
        if not bool(decision.get("is_ready")):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "MODEL_NOT_READY",
                    "message": "必需模型未就绪，请等待模型下载/修复完成后再启动任务",
                    **decision,
                },
            )

    def _resolve_project_identity(identifier: Optional[str]):
        normalized_identifier = str(identifier or "").strip()
        if not normalized_identifier:
            return None
        try:
            return get_project_id_resolver().resolve_or_fail(normalized_identifier)
        except Exception:
            return None

    def _resolve_project_id_or_404(identifier: str) -> str:
        identity = _resolve_project_identity(identifier)
        if identity is None:
            raise HTTPException(status_code=404, detail="任务未找到")
        return str(identity.project_id)

    def _build_legacy_subtitle_payload(segment: Dict[str, Any], fallback_index: Optional[int] = None) -> Dict[str, Any]:
        raw_index = segment.get("legacy_index", fallback_index)
        try:
            normalized_index = int(raw_index) if raw_index is not None else int(fallback_index or 0)
        except (TypeError, ValueError):
            normalized_index = int(fallback_index or 0)
        return {
            "index": normalized_index,
            "text": segment.get("text", ""),
            "start": segment.get("start", 0.0),
            "end": segment.get("end", 0.0),
            "confidence": None,
            "display_confidence": None,
            "confidence_source": "manual",
            "source": segment.get("source_type", "project_api"),
            "is_modified": bool(segment.get("is_modified", True)),
            "original_text": segment.get("original_text"),
            "segment_id": segment.get("segment_id"),
        }

    def _resolve_runtime_job_id(identifier: str) -> Optional[str]:
        """
        将任意 identifier（project_id / legacy job_id）解析为运行态 job_id。
        """
        normalized_identifier = str(identifier or "").strip()
        if not normalized_identifier:
            return None

        direct_job = transcription_service.get_job(normalized_identifier)
        if direct_job is not None:
            return str(getattr(direct_job, "job_id", "") or normalized_identifier)

        identity = _resolve_project_identity(normalized_identifier)
        if identity is None:
            return None

        # 1) 优先按 project_id 直接命中
        runtime_job = transcription_service.get_job(identity.project_id)
        if runtime_job is not None:
            if str(getattr(runtime_job, "project_id", "") or "").strip() != identity.project_id:
                runtime_job.project_id = identity.project_id
            return str(getattr(runtime_job, "job_id", "") or identity.project_id)

        # 2) 回退 legacy_job_id
        legacy_job_id = str(getattr(identity, "legacy_job_id", "") or "").strip()
        if legacy_job_id:
            runtime_job = transcription_service.get_job(legacy_job_id)
            if runtime_job is not None:
                if str(getattr(runtime_job, "project_id", "") or "").strip() != identity.project_id:
                    runtime_job.project_id = identity.project_id
                return str(getattr(runtime_job, "job_id", "") or legacy_job_id)

        # 3) 最后从内存任务池按 project_id 扫描
        runtime_jobs = getattr(getattr(transcription_service, "job_lifecycle", None), "jobs", None)
        if isinstance(runtime_jobs, dict):
            for runtime_job in runtime_jobs.values():
                if str(getattr(runtime_job, "project_id", "") or "").strip() != identity.project_id:
                    continue
                return str(getattr(runtime_job, "job_id", "") or "")

        return None

    def _resolve_job_workspace(job: Optional[JobState], identifier: str):
        """
        统一解析任务可用目录与媒体 URL 标识。

        返回: (workspace_dir, media_identifier)
        """
        normalized_identifier = str(identifier or "").strip()
        media_identifier = str(getattr(job, "project_id", "") or normalized_identifier)

        if job and str(getattr(job, "dir", "") or "").strip():
            job_dir = Path(job.dir)
            if job_dir.exists():
                return job_dir, media_identifier

        identity = _resolve_project_identity(getattr(job, "project_id", None) or normalized_identifier)
        if identity is not None:
            if job is not None:
                job.project_id = identity.project_id
                job.dir = str(identity.project_dir)
            return identity.project_dir, identity.project_id

        fallback_dir = Path(job.dir) if job and str(getattr(job, "dir", "") or "").strip() else (config.JOBS_DIR / normalized_identifier)
        return fallback_dir, media_identifier

    def _build_job_settings_from_task_config(task_config: Optional[Dict[str, Any]]) -> JobSettings:
        """根据 task_config 生成 JobSettings。"""
        if not task_config:
            return JobSettings()
        # 临时策略：任务级 whisper_model 覆盖停用，统一回退 .env。
        normalized_task_config = dict(task_config)
        transcription_payload = dict(normalized_task_config.get("transcription") or {})
        if "whisper_model" in transcription_payload:
            transcription_payload.pop("whisper_model", None)
            normalized_task_config["transcription"] = transcription_payload

        preset_id = str(normalized_task_config.get("preset_id", "balanced") or "balanced")
        has_custom_groups = any(
            normalized_task_config.get(key)
            for key in ("preprocessing", "transcription", "refinement", "compute", "debug")
        )
        if preset_id != "custom" and not has_custom_groups:
            return JobSettings.from_preset(preset_id)
        try:
            return JobSettings.from_dict(normalized_task_config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _normalize_checkpoint_speaker_settings(
        original_settings: Optional[Dict[str, Any]],
        task_config: Optional[Dict[str, Any]],
        existing_job_settings: Optional[JobSettings],
    ) -> Dict[str, Any]:
        """
        规范化 checkpoint 中的 speaker 相关设置。

        Why:
        - 历史 checkpoint 可能缺少任务级 speaker 字段；
        - /start 收到显式 task_config 时，用户期望 speaker 策略按本次请求生效。
        """
        merged = dict(original_settings or {})
        preprocessing_payload = dict(merged.get("preprocessing") or {})
        request_preprocessing = dict((task_config or {}).get("preprocessing") or {})
        fallback_preprocessing: Dict[str, Any] = {}
        if isinstance(existing_job_settings, JobSettings):
            fallback_preprocessing = (
                existing_job_settings.to_dict().get("preprocessing") or {}
            )

        speaker_keys = (
            "enable_speaker_detection",
            "enable_speaker_guided_split",
            "speaker_count",
            "speaker_min_count",
            "speaker_max_count",
        )
        for key in speaker_keys:
            if key in request_preprocessing:
                preprocessing_payload[key] = request_preprocessing[key]
                continue
            if key not in preprocessing_payload and key in fallback_preprocessing:
                preprocessing_payload[key] = fallback_preprocessing[key]

        merged["preprocessing"] = preprocessing_payload
        return merged

    def _build_task_snapshot(job: JobState) -> Dict[str, Any]:
        """构建前端任务状态快照（包含时间戳，用于版本校验）。"""
        return {
            "id": job.job_id,
            "job_id": job.job_id,
            "project_id": str(getattr(job, "project_id", "") or "").strip() or job.job_id,
            "filename": job.filename,
            "title": job.title,
            "status": job.status,
            "progress": job.progress,
            "phase": job.phase,
            "phase_percent": job.phase_percent,
            "message": job.message,
            "processed": job.processed,
            "total": job.total,
            "language": job.language,
            "updated_at": job.updatedAt,
        }

    @router.get("/stream/{identifier}")
    async def stream_job_progress(identifier: str, request: Request):
        """
        SSE流式端点 - 实时推送转录任务进度

        频道ID格式: project:{project_id}
        事件类型:
        - progress: 进度更新 (包含 percent, phase, message, status等)
        - signal: 关键节点信号 (job_complete, job_failed, job_canceled, job_paused)
        - bgm_detected: BGM检测结果 (level, ratios, max_ratio, recommendation)
        - circuit_breaker_triggered: 熔断触发事件 (triggered, reason, stats, action)
        - segment: 单个段落转录完成 (包含text, start, end等)
        - ping: 心跳
        """
        job_id = identifier
        project_id = _resolve_project_id_or_404(identifier)
        runtime_job_id = (
            _resolve_runtime_job_id(project_id)
            or _resolve_runtime_job_id(job_id)
            or job_id
        )

        # 验证任务是否存在
        job = transcription_service.get_job(runtime_job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        channel_id = f"project:{project_id}"

        # 定义初始状态回调 - 连接时立即发送当前状态
        def get_initial_state():
            current_job = (
                transcription_service.get_job(runtime_job_id)
                or transcription_service.get_job(project_id)
                or transcription_service.get_job(job_id)
            )

            def _build_proxy_state(current_job_state: Optional[JobState]):
                job_dir, media_identifier = _resolve_job_workspace(current_job_state, project_id)
                preview_360p = job_dir / "preview_360p.mp4"
                proxy_720p = job_dir / "proxy_720p.mp4"
                remux_video = job_dir / "remux.mp4"
                source_video = _find_video_file(job_dir) if job_dir.exists() else None

                urls = {
                    "360p": f"/api/media/{media_identifier}/video/preview" if preview_360p.exists() else None,
                    "720p": f"/api/media/{media_identifier}/video" if (proxy_720p.exists() or remux_video.exists()) else None,
                    "source": f"/api/media/{media_identifier}/video" if source_video else None
                }

                media_prep = get_media_prep_service()
                task_status = media_prep.get_full_task_status(media_identifier) if media_prep else None

                state = "idle"
                progress = 0
                error = None
                decision = task_status.get("decision") if task_status else None

                if task_status:
                    state = task_status.get("state", state)
                    progress = task_status.get("progress", progress)
                    error = task_status.get("error")
                else:
                    if proxy_720p.exists() or remux_video.exists():
                        state = "ready_720p"
                        progress = 100
                    elif preview_360p.exists():
                        state = "ready_360p"
                        progress = 100
                    elif source_video:
                        state = "direct_play"
                        progress = 100

                return {
                    "state": state,
                    "progress": progress,
                    "decision": decision,
                    "urls": urls,
                    "error": error,
                    "project_id": media_identifier if media_identifier != project_id else project_id,
                }

            persisted_job = transcription_service.job_lifecycle.state_repo.get_task(runtime_job_id)
            updated_at = persisted_job.updatedAt if persisted_job else None

            if current_job:
                proxy_state = _build_proxy_state(current_job)
                return {
                    "job_id": current_job.job_id,
                    "project_id": project_id,
                    "phase": current_job.phase,
                    "percent": current_job.progress,
                    "message": current_job.message,
                    "status": current_job.status,
                    "processed": current_job.processed,
                    "total": current_job.total,
                    "language": current_job.language or "",
                    "updated_at": updated_at,
                    # 追加当前 Proxy/预览状态，断线重连时立即同步
                    "proxy": proxy_state
                }
            return None

        # 订阅SSE流
        return StreamingResponse(
            sse_manager.subscribe(channel_id, request, initial_state_callback=get_initial_state),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    @router.post("/upload")
    async def upload_file(
        file: UploadFile = File(...),
        task_config: Optional[str] = Form(None),
    ):
        """上传文件并自动创建转录任务（V2.2: 加入队列）"""
        try:
            # 验证文件类型
            if not file_service.is_supported_file(file.filename):
                raise HTTPException(status_code=400, detail="不支持的文件格式")

            # 保存用户原始文件路径信息
            original_filename = file.filename

            # 将文件保存到input目录
            input_path = file_service.get_input_file_path(original_filename)

            # 如果同名文件已存在，添加时间戳
            counter = 1
            base_name, ext = os.path.splitext(original_filename)
            while os.path.exists(input_path):
                new_filename = f"{base_name}_{counter}{ext}"
                input_path = file_service.get_input_file_path(new_filename)
                original_filename = new_filename
                counter += 1

            # 保存文件
            with open(input_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # 创建任务
            parsed_task_config = _parse_task_config_form(task_config)
            settings = _build_job_settings_from_task_config(parsed_task_config)
            job = transcription_service.create_job(original_filename, input_path, settings)
            project_id = str(getattr(job, "project_id", None) or job.job_id)

            # 🔥 新增: 加入队列（而非直接启动）
            queue_service = get_queue_service(transcription_service)
            # 模型未就绪时不要把任务推进队列，避免 Runner 直接失败。
            await _guard_models_before_enqueue(job.settings)
            queue_service.add_job(job)

            return {
                "job_id": project_id,
                "project_id": project_id,
                "filename": original_filename,
                "original_name": file.filename,
                "message": "文件上传成功，已加入转录队列",
                "queue_position": len(queue_service.queue)  # 新增: 队列位置
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"上传文件失败: {str(e)}")

    @router.post("/create-task")
    async def create_job(
        filename: str = Form(...),
        task_config: Optional[str] = Form(None),
    ):
        """为指定文件创建转录任务（本地input模式）"""
        try:
            input_path = file_service.get_input_file_path(filename)
            if not os.path.exists(input_path):
                raise HTTPException(status_code=404, detail="文件不存在")

            if not file_service.is_supported_file(filename):
                raise HTTPException(status_code=400, detail="不支持的文件格式")

            parsed_task_config = _parse_task_config_form(task_config)
            settings = _build_job_settings_from_task_config(parsed_task_config)
            job = transcription_service.create_job(filename, input_path, settings)
            project_id = str(getattr(job, "project_id", None) or job.job_id)

            return {"job_id": project_id, "project_id": project_id, "filename": filename}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")

    @router.post("/create-tasks-batch")
    async def create_jobs_batch(req: CreateJobsBatchRequest):
        """
        批量创建转录任务（从 input 目录选择多个文件）

        Args:
            filenames: 文件名列表

        Returns:
            {
                "success": true,
                "jobs": [{job_id, filename, queue_position}, ...],
                "failed": [{filename, error}, ...],
                "total": int,
                "succeeded": int,
                "failed_count": int
            }
        """
        try:
            queue_service = get_queue_service(transcription_service)
            jobs = []
            failed = []
            filenames = list(req.filenames or [])
            task_config_payload = (
                req.task_config.model_dump(exclude_none=True)
                if req.task_config is not None
                else {}
            )

            for filename in filenames:
                try:
                    # 验证文件存在
                    input_path = file_service.get_input_file_path(filename)
                    if not os.path.exists(input_path):
                        failed.append({"filename": filename, "error": "文件不存在"})
                        continue

                    # 验证文件格式
                    if not file_service.is_supported_file(filename):
                        failed.append({"filename": filename, "error": "不支持的文件格式"})
                        continue

                    # 创建任务
                    settings = _build_job_settings_from_task_config(task_config_payload)
                    job = transcription_service.create_job(filename, input_path, settings)
                    project_id = str(getattr(job, "project_id", None) or job.job_id)

                    # 加入队列
                    await _guard_models_before_enqueue(job.settings)
                    queue_service.add_job(job)

                    jobs.append({
                        "job_id": project_id,
                        "project_id": project_id,
                        "filename": filename,
                        "queue_position": len(queue_service.queue)
                    })

                except Exception as e:
                    failed.append({"filename": filename, "error": str(e)})

            return {
                "success": True,
                "jobs": jobs,
                "failed": failed,
                "total": len(filenames),
                "succeeded": len(jobs),
                "failed_count": len(failed)
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"批量创建任务失败: {str(e)}")

    @router.post("/start")
    async def start_job(job_id: str = Form(...), settings: str = Form(...)):
        """启动转录任务（V2.2: 加入队列而非直接启动）"""
        try:
            from pathlib import Path

            settings_payload = json.loads(settings) if settings else {}
            settings_obj = TranscribeSettings(**settings_payload)

            # 获取队列服务
            queue_service = get_queue_service(transcription_service)
            job = queue_service.get_job(job_id)

            if not job:
                # 如果队列服务中没有，尝试从transcription_service获取
                job = transcription_service.get_job(job_id)

            if not job:
                raise HTTPException(status_code=404, detail="无效 job_id")

            # 检查是否有checkpoint（断点续传场景）
            job_dir = Path(job.dir) if job.dir else None
            checkpoint_path = job_dir / "checkpoint.json" if job_dir else None

            original_settings = None
            if checkpoint_path and checkpoint_path.exists():
                # 有 checkpoint 时优先使用原始设置，避免新旧配置不一致
                try:
                    with open(checkpoint_path, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                    original_settings = checkpoint_data.get("original_settings") or None
                except Exception as e:
                    print(f"读取checkpoint设置失败: {e}")

            task_config = (
                settings_obj.task_config.model_dump(exclude_none=True)
                if settings_obj.task_config else {}
            )

            if original_settings:
                try:
                    normalized_original_settings = _normalize_checkpoint_speaker_settings(
                        original_settings=original_settings,
                        task_config=task_config,
                        existing_job_settings=job.settings,
                    )
                    job.settings = JobSettings.from_dict(normalized_original_settings)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc))
            elif task_config:
                job.settings = _build_job_settings_from_task_config(task_config)
            else:
                if not isinstance(job.settings, JobSettings):
                    job.settings = JobSettings()

            # 在 settings 最终确定之后再做模型守卫（否则可能误判 whisper 需求）。
            await _guard_models_before_enqueue(job.settings)

            # 🔥 关键改动: 如果任务不在队列中，加入队列
            with queue_service.lock:
                if job.status == "paused" or job.status == "failed":
                    # 恢复任务：重新加入队列
                    job.canceled = False
                    job.paused = False
                    job.error = None
                    queue_service.queue.append(job_id)
                    job.status = "queued"
                    job.message = f"已加入队列 (位置: {len(queue_service.queue)})"
                    # 确保任务在jobs字典中
                    queue_service.jobs[job_id] = job
                elif job.status == "uploaded" or job.status == "created":
                    # 新任务：加入队列
                    queue_service.queue.append(job_id)
                    job.status = "queued"
                    job.message = f"已加入队列 (位置: {len(queue_service.queue)})"
                    # 确保任务在jobs字典中
                    queue_service.jobs[job_id] = job
                elif job.status == "queued":
                    # 任务已在队列中
                    queue_position = list(queue_service.queue).index(job_id) + 1 if job_id in queue_service.queue else -1
                    job.message = f"已在队列中 (位置: {queue_position})"

            # 保存队列状态并推送 SSE 通知（修复：之前缺少这一步导致前端收不到状态更新）
            queue_service._save_state()
            queue_service._notify_queue_change()
            queue_service._notify_job_status(job_id, job.status)

            return {
                "job_id": job_id,
                "started": True,
                "queue_position": len(queue_service.queue)
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"启动任务失败: {str(e)}")

    @router.post("/cancel/{identifier}")
    async def cancel_job(identifier: str, delete_data: bool = False):
        """取消转录任务（V2.2: 使用队列服务）"""
        job_id = identifier
        queue_service = get_queue_service(transcription_service)
        result = queue_service.cancel_job(job_id, delete_data=delete_data)
        if not result.success:
            # 占用场景用 423 方便前端弹全局提示；运行中删除用 409 告知稍后再试
            if result.reason_code == "delete_blocked" or "占用" in (result.message or ""):
                status = 423
            elif result.reason_code == "cancel_running" or "取消中" in (result.message or ""):
                status = 409
            elif result.reason_code == "cancel_not_found":
                status = 404
            else:
                status = 400
            raise HTTPException(status_code=status, detail=result.message or "任务未找到")
        job_snapshot = None
        job = transcription_service.job_lifecycle.state_repo.get_task(job_id)
        if job:
            job_snapshot = _build_task_snapshot(job)
        return {
            "job_id": job_id,
            "canceled": result.success,
            "data_deleted": delete_data,
            "success": result.success,
            "status": result.status,
            "reason_code": result.reason_code,
            "message": result.message,
            "pending_delete": result.pending_delete,
            "state_seq": result.state_seq,
            "task": job_snapshot,
        }

    @router.post("/pause/{identifier}")
    async def pause_job(identifier: str):
        """暂停转录任务（V2.2: 使用队列服务）"""
        job_id = identifier
        queue_service = get_queue_service(transcription_service)
        ok = queue_service.pause_job(job_id)
        if not ok:
            raise HTTPException(status_code=404, detail="任务未找到")
        job_snapshot = None
        job = transcription_service.job_lifecycle.state_repo.get_task(job_id)
        if job:
            job_snapshot = _build_task_snapshot(job)
        return {"job_id": job_id, "paused": ok, "task": job_snapshot}

    @router.post("/resume/{identifier}")
    async def resume_job(identifier: str):
        """
        恢复暂停的任务（重新加入队列）

        与 /restore-task 不同：
        - /resume: 恢复暂停的任务，重新加入队列尾部，状态变为 queued
        - /restore-task: 从 checkpoint 断点续传
        """
        job_id = identifier
        queue_service = get_queue_service(transcription_service)
        ok = queue_service.resume_job(job_id)
        if not ok:
            raise HTTPException(status_code=400, detail="无法恢复任务（任务未暂停或不存在）")

        job = queue_service.get_job(job_id)
        queue_position = 0
        if job_id in queue_service.queue:
            queue_position = list(queue_service.queue).index(job_id) + 1
        job_snapshot = None
        persisted_job = transcription_service.job_lifecycle.state_repo.get_task(job_id)
        if persisted_job:
            job_snapshot = _build_task_snapshot(persisted_job)

        return {
            "job_id": job_id,
            "resumed": True,
            "status": job.status if job else "queued",
            "queue_position": queue_position,
            "task": job_snapshot,
        }

    @router.post("/prioritize/{identifier}")
    async def prioritize_job(identifier: str, mode: Optional[str] = None):
        """
        将任务移到队列头部（插队）

        Args:
            job_id: 任务ID
            mode: 插队模式
                - "gentle": 温和插队，放到队列头部，等当前任务完成后执行
                - "force": 强制插队，暂停当前任务A -> 执行B -> B完成后自动恢复A
                - None: 使用默认模式（可通过 /api/queue-settings 配置）
        """
        job_id = identifier
        queue_service = get_queue_service(transcription_service)
        result = queue_service.prioritize_job(job_id, mode=mode)

        if not result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "无法优先此任务")
            )

        return {
            "job_id": job_id,
            "prioritized": True,
            "mode": result.get("mode"),
            "interrupted_job_id": result.get("interrupted_job_id"),
            "queue_position": 1
        }

    @router.get("/queue-settings")
    async def get_queue_settings():
        """
        获取队列设置

        返回:
            - default_prioritize_mode: 默认插队模式 ("gentle" 或 "force")
        """
        queue_service = get_queue_service(transcription_service)
        return queue_service.get_settings()

    @router.post("/queue-settings")
    async def update_queue_settings(
        default_prioritize_mode: Optional[str] = Body(None, embed=True)
    ):
        """
        更新队列设置

        Args:
            default_prioritize_mode: 默认插队模式
                - "gentle": 温和插队（默认）
                - "force": 强制插队
        """
        queue_service = get_queue_service(transcription_service)
        try:
            settings = queue_service.update_settings(
                default_prioritize_mode=default_prioritize_mode
            )
            return {
                "success": True,
                "settings": settings
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/reorder-queue")
    async def reorder_queue(job_ids: list = Body(..., embed=True)):
        """
        重新排序队列

        Args:
            job_ids: 按新顺序排列的任务ID列表
        """
        queue_service = get_queue_service(transcription_service)
        ok = queue_service.reorder_queue(job_ids)

        if not ok:
            raise HTTPException(status_code=400, detail="重排队列失败（任务ID不匹配）")

        return {
            "reordered": True,
            "queue": job_ids
        }

    @router.get("/queue-status")
    async def get_queue_status():
        """获取队列状态摘要"""
        queue_service = get_queue_service(transcription_service)
        return queue_service.get_queue_status()

    @router.get("/events/global")
    async def stream_global_events(request: Request):
        """
        全局SSE流 - 推送所有任务的状态变化 (V3.0)

        事件类型:
        - initial_state: 连接时的初始状态
        - queue_update: 队列顺序变化
        - job_status: 任务状态变化
        - job_progress: 任务进度更新

        注意:
        - initial_state返回所有任务（处理中 + 已完成）
        - 避免客户端连接时漏掉完成任务的实时更新
        """
        queue_service = get_queue_service(transcription_service)

        def get_initial_state():
            """
            返回所有任务列表（第二阶段修复：实时更新）
            包含活跃任务 + 历史完成任务
            """
            lifecycle = transcription_service.job_lifecycle
            queue_state = lifecycle.state_repo.load_queue_state()
            queue_list = queue_state.queue if queue_state else []
            running_id = queue_state.running_job_id if queue_state else None
            interrupted_id = queue_state.interrupted_job_id if queue_state else None
            jobs_summary = lifecycle.list_tasks_summary()

            return {
                "queue": queue_list,
                "running": running_id,
                "interrupted": interrupted_id,
                "queue_updated_at": queue_state.updated_at if queue_state else None,
                "jobs": jobs_summary
            }

        # 订阅SSE流，频道名为 "global"
        return StreamingResponse(
            sse_manager.subscribe("global", request, initial_state_callback=get_initial_state),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    @router.get("/sync-tasks")
    async def sync_tasks():
        """
        同步所有任务（第一阶段修复：数据同步）

        返回所有任务列表（处理中 + 已完成），前端用此接口同步后端实际存在的任务
        此接口为真实源，用于修复幽灵任务问题
        """
        from app.services.project_service import get_project_service

        lifecycle = transcription_service.job_lifecycle
        project_service = get_project_service()
        tasks = lifecycle.list_tasks_summary()
        existing_project_ids: Set[str] = set()
        for task in tasks:
            task_id = str(task.get("id", "") or "").strip()
            if not task_id:
                continue
            resolved_project = None
            task_project_id = str(task.get("project_id", "") or "").strip()
            resolve_seed = task_project_id or task_id
            try:
                resolved_project_id = _resolve_project_id_or_404(resolve_seed)
                resolved_project = project_service.get_project(resolved_project_id)
            except HTTPException:
                alias_project = project_service.find_project_by_alias(resolve_seed)
                if alias_project is not None:
                    alias_project_id = str(getattr(alias_project, "project_id", "") or "").strip()
                    resolved_project_id = alias_project_id or resolve_seed
                    resolved_project = alias_project
                else:
                    resolved_project_id = resolve_seed
            task["project_id"] = resolved_project_id
            subtitle_doc = getattr(resolved_project, "subtitle_doc", None) if resolved_project is not None else None
            subtitle_source_type = (
                str(getattr(subtitle_doc, "source_type", "") or "").strip().lower()
                if subtitle_doc is not None
                else str(task.get("source_type", "") or "").strip().lower()
            )
            task_mode = infer_task_mode(
                raw_task_mode=task.get("task_mode"),
                project_mode=getattr(resolved_project, "mode", None) if resolved_project is not None else None,
                subtitle_source_type=subtitle_source_type,
                project_dir=str(getattr(resolved_project, "dir", "") or ""),
            )
            task["task_mode"] = task_mode
            task["is_project_only"] = task_mode == "subtitle_edit"
            existing_project_ids.add(resolved_project_id)

        for project in project_service.list_projects():
            project_id = str(getattr(project, "project_id", "") or "").strip()
            if not project_id or project_id in existing_project_ids:
                continue
            subtitle_doc = getattr(project, "subtitle_doc", None)
            source_type = str(getattr(subtitle_doc, "source_type", "") or "").strip().lower()
            task_mode = infer_task_mode(
                raw_task_mode=getattr(project, "task_mode", None),
                project_mode=getattr(project, "mode", None),
                subtitle_source_type=source_type,
                project_dir=str(getattr(project, "dir", "") or ""),
            )
            if task_mode != "subtitle_edit":
                continue

            source_type = str(getattr(subtitle_doc, "source_type", "") or "").strip()
            segment_count = int(getattr(subtitle_doc, "segment_count", 0) or 0)
            title = str(getattr(project, "title", "") or "").strip()
            filename = title or (f"{project_id}.srt" if task_mode == "subtitle_edit" else project_id)

            tasks.append(
                {
                    "id": project_id,
                    "project_id": project_id,
                    "filename": filename,
                    "title": title,
                    "status": "finished",
                    "progress": 100.0,
                    "phase_percent": 100.0,
                    "message": "仅编辑项目",
                    "created_time": getattr(project, "created_at", None),
                    "updated_at": getattr(project, "updated_at", None),
                    "phase": "edit_ready",
                    "processed": segment_count,
                    "total": segment_count,
                    "language": None,
                    "source_type": source_type or "import",
                    "task_mode": task_mode,
                    "is_project_only": task_mode == "subtitle_edit",
                }
            )
            existing_project_ids.add(project_id)

        tasks.sort(
            key=lambda item: float(item.get("updated_at") or item.get("created_time") or 0.0),
            reverse=True,
        )
        queue_state = lifecycle.state_repo.load_queue_state()

        return {
            "success": True,
            "tasks": tasks,
            "count": len(tasks),
            "queue": queue_state.queue if queue_state else [],
            "queue_updated_at": queue_state.updated_at if queue_state else None,
            "timestamp": int(time.time() * 1000)
        }

    @router.get("/incomplete-tasks")
    async def get_incomplete_jobs():
        """获取所有未完成的任务"""
        jobs = transcription_service.scan_incomplete_jobs()
        return {"jobs": jobs, "count": len(jobs)}

    @router.post("/restore-task/{identifier}")
    async def restore_job(identifier: str):
        """从检查点恢复任务"""
        job_id = identifier
        job = transcription_service.restore_job_from_checkpoint(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="无法恢复任务，检查点不存在或已损坏")

        return job.to_dict()

    @router.get("/status/{identifier}")
    async def get_job_status(identifier: str, include_media: bool = True):
        """
        获取任务状态（V2.3: 包含队列位置和媒体状态）

        Args:
            job_id: 任务ID
            include_media: 是否包含媒体状态信息（默认True）
        """
        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        # 返回状态（新增queue_position字段）
        result = job.to_dict()
        _, media_identifier = _resolve_job_workspace(job, job_id)
        if media_identifier and media_identifier != job_id:
            result["project_id"] = media_identifier
        elif getattr(job, "project_id", None):
            result["project_id"] = job.project_id

        # 计算队列位置
        queue_state = transcription_service.job_lifecycle.state_repo.load_queue_state()
        if queue_state:
            if job_id in queue_state.queue:
                result["queue_position"] = queue_state.queue.index(job_id) + 1
            elif job_id == queue_state.running_job_id:
                result["queue_position"] = 0
            else:
                result["queue_position"] = -1
        else:
            result["queue_position"] = -1

        # 添加媒体状态信息（用于编辑器）
        if include_media and job.status == "finished":
            job_dir, media_identifier = _resolve_job_workspace(job, job_id)
            if job_dir and job_dir.exists():
                job.update_media_status(str(job_dir))
            else:
                job.media_status = None
            if job.media_status:
                result["media_status"] = {
                    "video_exists": job.media_status.video_exists,
                    "video_format": job.media_status.video_format,
                    "needs_proxy": job.media_status.needs_proxy,
                    "proxy_exists": job.media_status.proxy_exists,
                    "audio_exists": job.media_status.audio_exists,
                    "peaks_ready": job.media_status.peaks_ready,
                    "thumbnails_ready": job.media_status.thumbnails_ready,
                    "srt_exists": job.media_status.srt_exists,
                    # 便捷的URL字段
                    "video_url": f"/api/media/{media_identifier}/video" if job.media_status.video_exists or job.media_status.proxy_exists else None,
                    "audio_url": f"/api/media/{media_identifier}/audio" if job.media_status.audio_exists else None,
                    "peaks_url": f"/api/media/{media_identifier}/peaks" if job.media_status.audio_exists else None,
                    "thumbnails_url": f"/api/media/{media_identifier}/thumbnails" if job.media_status.video_exists else None,
                    "srt_url": f"/api/media/{media_identifier}/srt" if job.media_status.srt_exists else None
                }

        return result

    @router.get("/legacy/tasks/{identifier}/subtitle-time-offset")
    async def get_job_subtitle_time_offset(identifier: str):
        """
        获取任务级字幕时间偏移（无则回退全局）
        """
        from app.services.user_config_service import get_user_config_service

        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        config_service = get_user_config_service()
        global_offset = config_service.get_subtitle_time_offset()
        task_offset = getattr(job, "subtitle_time_offset", None)

        if task_offset is None:
            return {"success": True, "offset": global_offset, "source": "global"}
        return {"success": True, "offset": float(task_offset), "source": "project"}

    @router.post("/legacy/tasks/{identifier}/subtitle-time-offset")
    async def set_job_subtitle_time_offset(identifier: str, req: TaskSubtitleTimeOffsetRequest):
        """
        设置任务级字幕时间偏移（等于全局时不保存）
        """
        from app.services.user_config_service import get_user_config_service

        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        config_service = get_user_config_service()
        global_offset = config_service.get_subtitle_time_offset()
        normalized = round(float(req.offset), 3)
        global_normalized = round(float(global_offset), 3)

        if normalized == global_normalized:
            job.subtitle_time_offset = None
            source = "global"
            effective_offset = global_normalized
        else:
            job.subtitle_time_offset = normalized
            source = "project"
            effective_offset = normalized

        transcription_service.save_job_meta(job)

        return {
            "success": True,
            "offset": effective_offset,
            "source": source
        }

    @router.get("/download/{identifier}")
    async def download_result(identifier: str, copy_to_source: bool = False, auto_repair: bool = True):
        """
        下载转录结果
        V3.1.1+dev.20260106.03: 下载前自动修复时间戳重叠

        Args:
            job_id: 任务ID
            copy_to_source: 是否复制到源文件目录
            auto_repair: 是否自动修复重叠（默认 True）
        """
        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        if not job.srt_path or not os.path.exists(job.srt_path):
            raise HTTPException(status_code=404, detail="字幕文件未生成")

        filename = os.path.basename(job.srt_path)

        # V3.1.1+dev.20260106.03: 读取并修复重叠
        srt_content = None
        if auto_repair:
            try:
                from app.utils.text_utils import repair_srt_overlaps
                with open(job.srt_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                srt_content, repaired_count = repair_srt_overlaps(original_content, gap_ms=1.0)
                if repaired_count > 0:
                    print(f"[{job_id}] 下载时自动修复了 {repaired_count} 处时间戳重叠")
            except Exception as e:
                print(f"[{job_id}] 修复重叠失败: {e}")
                srt_content = None

        # 如果需要复制到源文件目录
        if copy_to_source and job.input_path:
            source_dir = os.path.dirname(job.input_path)
            source_srt_path = os.path.join(source_dir, filename)

            try:
                # 如果有修复后的内容，写入修复后的版本
                if srt_content:
                    with open(source_srt_path, 'w', encoding='utf-8') as f:
                        f.write(srt_content)
                else:
                    shutil.copy2(job.srt_path, source_srt_path)
                print(f"SRT文件已复制到源目录: {source_srt_path}")
            except Exception as e:
                print(f"复制到源目录失败: {e}")

        # 同时复制到输出目录
        output_path = os.path.join(output_dir, filename)
        try:
            # 如果有修复后的内容，写入修复后的版本
            if srt_content:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(srt_content)
            elif not os.path.exists(output_path):
                shutil.copy2(job.srt_path, output_path)

            return FileResponse(
                path=output_path,
                filename=filename,
                media_type='text/plain; charset=utf-8'
            )
        except Exception as e:
            # 如果复制失败，直接返回原文件
            return FileResponse(
                path=job.srt_path,
                filename=filename,
                media_type='text/plain; charset=utf-8'
            )

    @router.post("/copy-result/{identifier}")
    async def copy_result_to_source(identifier: str, auto_repair: bool = True):
        """
        将转录结果复制到源文件目录
        V3.1.1+dev.20260106.03: 复制前自动修复时间戳重叠

        Args:
            job_id: 任务ID
            auto_repair: 是否自动修复重叠（默认 True）
        """
        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        if not job.srt_path or not os.path.exists(job.srt_path):
            raise HTTPException(status_code=404, detail="字幕文件未生成")

        try:
            # 获取原始文件目录
            if job.input_path:
                source_dir = os.path.dirname(job.input_path)
            else:
                # 如果没有input_path，使用input目录
                source_dir = file_service.input_dir

            # 生成目标路径
            srt_filename = os.path.basename(job.srt_path)
            target_path = os.path.join(source_dir, srt_filename)

            # V3.1.1+dev.20260106.03: 读取并修复重叠
            repaired_count = 0
            if auto_repair:
                try:
                    from app.utils.text_utils import repair_srt_overlaps
                    with open(job.srt_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()
                    repaired_content, repaired_count = repair_srt_overlaps(original_content, gap_ms=1.0)
                    if repaired_count > 0:
                        # 写入修复后的内容
                        with open(target_path, 'w', encoding='utf-8') as f:
                            f.write(repaired_content)
                        print(f"[{job_id}] 复制时自动修复了 {repaired_count} 处时间戳重叠")
                    else:
                        # 无需修复，直接复制
                        shutil.copy2(job.srt_path, target_path)
                except Exception as e:
                    print(f"[{job_id}] 修复重叠失败: {e}，直接复制原文件")
                    shutil.copy2(job.srt_path, target_path)
            else:
                # 不修复，直接复制
                shutil.copy2(job.srt_path, target_path)

            message = f"字幕文件已复制到: {target_path}"
            if repaired_count > 0:
                message += f" (自动修复了 {repaired_count} 处重叠)"

            return {
                "success": True,
                "message": message,
                "target_path": target_path,
                "repaired_overlaps": repaired_count
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"复制文件失败: {str(e)}")

    # V3.2.0+dev.20260124.02: 用户编辑字幕接口
    class SubtitleCreateRequest(BaseModel):
        """字幕新增请求模型"""
        text: Optional[str] = ""
        start: float
        end: float

    class SubtitleUpdateRequest(BaseModel):
        """字幕更新请求模型"""
        text: Optional[str] = None
        start: Optional[float] = None
        end: Optional[float] = None

    @router.post("/legacy/tasks/{identifier}/subtitles")
    async def create_subtitle(identifier: str, payload: SubtitleCreateRequest):
        """
        V3.2.0+dev.20260124.02: 用户新增字幕接口
        """
        from app.api.routes import project_routes as project_routes_module
        from app.services.sse_service import push_subtitle_event

        if payload.start < 0 or payload.end <= payload.start:
            raise HTTPException(status_code=400, detail="时间戳不合法")

        job_id = identifier
        project_id = _resolve_project_id_or_404(job_id)
        project_result = await project_routes_module.create_project_subtitle(
            project_id=project_id,
            body=payload,
        )
        segment = dict(project_result.get("data") or {})
        sentence_payload = _build_legacy_subtitle_payload(segment)

        push_subtitle_event(
            sse_manager,
            project_id,
            "added",
            {
                "index": sentence_payload["index"],
                "sentence": sentence_payload,
                "source": "user_add",
                "is_update": True,
            },
        )
        return {"success": True, "data": sentence_payload}

    @router.patch("/legacy/tasks/{identifier}/subtitles/{sentence_index}")
    async def update_subtitle(
        identifier: str,
        sentence_index: int,
        update: SubtitleUpdateRequest
    ):
        """
        V3.2.0+dev.20260124.02: 用户编辑字幕接口

        核心功能：
        1. 接收前端的实时编辑
        2. 标记为 is_modified=True，防止 AI 覆盖
        3. 持久化到编辑落盘文件，必要时同步快照

        Args:
            job_id: 任务 ID
            sentence_index: 句子索引
            update: 更新内容（text/start/end）
        """
        from app.api.routes import project_routes as project_routes_module
        from app.services.sse_service import push_subtitle_event

        update_payload = update.model_dump(exclude_none=True)
        if not update_payload:
            raise HTTPException(status_code=400, detail="更新内容为空")
        if (
            update_payload.get("start") is not None
            and update_payload.get("end") is not None
            and float(update_payload["end"]) < float(update_payload["start"])
        ):
            raise HTTPException(status_code=400, detail="结束时间必须大于等于开始时间")

        job_id = identifier
        project_id = _resolve_project_id_or_404(job_id)
        project_result = await project_routes_module.update_project_subtitle_by_legacy_index(
            project_id=project_id,
            sentence_index=sentence_index,
            body=update,
        )
        segment = dict(project_result.get("data") or {})
        sentence_payload = _build_legacy_subtitle_payload(segment, fallback_index=sentence_index)

        push_subtitle_event(
            sse_manager,
            project_id,
            "edited",
            {
                "index": sentence_payload["index"],
                "sentence": sentence_payload,
                "source": "user_edit",
                "is_update": True,
            },
        )
        return {"success": True, "data": sentence_payload}

    @router.delete("/legacy/tasks/{identifier}/subtitles/{sentence_index}")
    async def delete_subtitle(identifier: str, sentence_index: int):
        """
        V3.2.0+dev.20260124.02: 用户删除字幕接口
        """
        from app.api.routes import project_routes as project_routes_module
        from app.services.sse_service import push_subtitle_event

        job_id = identifier
        project_id = _resolve_project_id_or_404(job_id)
        project_result = await project_routes_module.delete_project_subtitle_by_legacy_index(
            project_id=project_id,
            sentence_index=sentence_index,
        )
        result_data = dict(project_result.get("data") or {})

        push_subtitle_event(
            sse_manager,
            project_id,
            "deleted",
            {
                "index": sentence_index,
                "source": "user_delete",
                "is_update": True,
            },
        )

        return {
            "success": True,
            "data": {
                "index": sentence_index,
                "is_deleted": bool(result_data.get("is_deleted", True)),
                "segment_id": result_data.get("segment_id"),
            },
        }

    # 同音检索路由已迁移到 `homophone_routes.py`，此处不再保留 legacy/job 旧格式入口。

    @router.get("/check-resume/{identifier}")
    async def check_resume(identifier: str):
        """检查任务是否可以断点续传"""
        from pathlib import Path

        job_id = identifier
        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        job_dir = Path(job.dir)
        checkpoint_path = job_dir / "checkpoint.json"

        summary = transcription_service.job_lifecycle.state_repo.get_checkpoint_summary(job_id)
        if not summary and checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception as e:
                return {
                    "can_resume": False,
                    "message": f"检查点文件损坏: {str(e)}"
                }

        if not summary:
            return {
                "can_resume": False,
                "message": "无检查点"
            }

        total_segments = summary.get("total_segments", 0)
        processed_indices = summary.get("processed_indices", [])
        processed_count = len(processed_indices)

        if total_segments > 0:
            progress = (processed_count / total_segments) * 100
        else:
            progress = 0

        return {
            "can_resume": True,
            "progress": round(progress, 2),
            "processed_segments": processed_count,
            "total_segments": total_segments,
            "phase": summary.get("phase", "unknown"),
            "message": f"检测到上次进度 ({progress:.1f}%)，可从断点继续"
        }

    @router.get("/checkpoint-settings/{identifier}")
    async def get_checkpoint_settings(identifier: str):
        """获取checkpoint中保存的原始设置（用于参数校验）"""
        from pathlib import Path

        job_id = identifier
        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        job_dir = Path(job.dir)
        summary = transcription_service.job_lifecycle.state_repo.get_checkpoint_summary(job_id)
        checkpoint_path = None
        if summary and summary.get("_file_path"):
            checkpoint_path = Path(summary["_file_path"])
        if not checkpoint_path:
            checkpoint_path = job_dir / "checkpoint.json"

        if not checkpoint_path.exists():
            return {"has_checkpoint": False}

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return {
                "has_checkpoint": True,
                "original_settings": data.get("original_settings", {}),
                "progress": {
                    "phase": data.get("phase"),
                    "processed": len(data.get("processed_indices", [])),
                    "total": data.get("total_segments", 0)
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取检查点失败: {str(e)}")

    @router.get("/transcription-text/{identifier}")
    async def get_transcription_text(identifier: str):
        """
        从checkpoint中提取已完成的转录文字（未对齐版本）

        用于SSE断线重连后，前端可以调用此API获取当前已转录的所有文字

        返回格式：
        {
            "job_id": "...",
            "has_checkpoint": true,
            "language": "zh",
            "segments": [
                {"id": 0, "start": 10.5, "end": 15.2, "text": "第一句话"},
                {"id": 1, "start": 15.2, "end": 20.0, "text": "第二句话"}
            ],
            "progress": {
                "processed": 50,
                "total": 100,
                "percentage": 50.0
            }
        }
        """
        from pathlib import Path

        job_id = identifier
        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        job_dir = Path(job.dir)
        checkpoint_path = job_dir / "checkpoint.json"
        snapshot_path = job_dir / "transcription_text.json"  # V3.1.2: 完成后保存的精简快照

        # 优先读取 checkpoint，缺失时尝试使用快照文件（任务完成后删除 checkpoint 的兜底）
        data = None
        transcription_data = None
        using_snapshot = False
        logger = logging.getLogger(__name__)
        speaker_store_service = None
        try:
            from app.services.speaker_store import SpeakerStoreService

            speaker_store_service = SpeakerStoreService(job_dir=job_dir)
        except Exception as speaker_exc:
            logger.warning("[%s] SpeakerStoreService 初始化失败，跳过 speaker 合并: %s", job_id, speaker_exc)

        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    transcription_data = data.get("transcription", {})
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"读取 checkpoint 失败: {e}")
        elif snapshot_path.exists():
            try:
                with open(snapshot_path, 'r', encoding='utf-8') as f:
                    transcription_data = json.load(f)
                # 兼容后续统一处理逻辑
                data = {"transcription": transcription_data}
                using_snapshot = True
                logger.info(f"[{job_id}] 使用转录快照恢复字幕（checkpoint 已清理）")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"读取转录快照失败: {e}")
        else:
            return {
                "job_id": job_id,
                "has_checkpoint": False,
                "message": "没有检查点数据"
            }

        try:
            # V3.1.0+: 优先从 transcription.sentences_snapshot 读取（实时字幕快照）
            transcription = transcription_data or {}
            sentences_snapshot = transcription.get("sentences_snapshot", [])
            from app.services.subtitle_edit_store import (
                apply_deletions_to_segments,
                apply_edits_to_segments,
                apply_edits_to_sentences_snapshot,
                build_manual_segments,
                load_deleted_indices,
                load_edits
            )
            from app.services.subtitle_visibility import (
                filter_hidden_unknown_sentences,
                is_hidden_unknown_sentence,
            )
            edits = load_edits(job_dir)
            deleted_indices = load_deleted_indices(job_dir)

            all_segments = []
            detected_language = None
            need_update_checkpoint = False  # V3.1.2: 标记是否需要更新 checkpoint

            if sentences_snapshot:
                # V3.2.0+dev.20260124.02: 叠加用户编辑落盘数据
                if edits:
                    apply_edits_to_sentences_snapshot(sentences_snapshot, edits)
                # 使用新格式（V3.1.0+ 实时字幕快照）
                # V3.1.2+dev.20260111.02: 增加 display_confidence 支持
                from app.core.confidence_mapper import ConfidenceMapper

                for sentence in sentences_snapshot:
                    if is_hidden_unknown_sentence(sentence):
                        continue
                    # V3.1.2: 处理置信度字段
                    # 注意：旧数据可能完全没有 confidence 字段，此时不应显示虚假的准确率
                    raw_conf = sentence.get("confidence")  # 可能为 None
                    source = sentence.get("source", "sensevoice")
                    is_draft = bool(sentence.get("_is_draft", False))
                    is_finalized = sentence.get("_is_finalized")
                    if is_finalized is None:
                        is_finalized = not is_draft

                    # V3.1.2: 检查是否有 display_confidence，没有则计算并标记需要更新
                    display_conf = sentence.get("display_confidence")
                    confidence_source = sentence.get("confidence_source")

                    if display_conf is None and raw_conf is not None:
                        # 有原始置信度但无映射值：实时计算 display_confidence
                        display_conf = ConfidenceMapper.map(raw_conf, source)
                        confidence_source = source
                        # 更新原始数据，稍后写回 checkpoint
                        sentence["display_confidence"] = display_conf
                        sentence["confidence_source"] = confidence_source
                        need_update_checkpoint = True
                    # 如果 raw_conf 为 None，display_conf 也保持 None，前端不显示准确率

                    all_segments.append({
                        "id": sentence.get("_index", 0),
                        "start": sentence.get("start", 0),
                        "end": sentence.get("end", 0),
                        "text": sentence.get("text", ""),
                        "confidence": raw_conf,  # 可能为 None
                        "display_confidence": display_conf,  # 可能为 None（旧数据无置信度）
                        "confidence_source": confidence_source,  # 可能为 None
                        "source": source,
                        "is_modified": sentence.get("is_modified", False),
                        "original_text": sentence.get("original_text"),
                        "is_draft": is_draft,
                        "is_finalized": bool(is_finalized),
                    })

                # 终态强制收口 snapshot，避免取消/完成后残留 draft。
                forced_snapshot_updates = force_finalize_snapshot_when_finished(
                    job_status=job.status,
                    sentences_snapshot=sentences_snapshot,
                )
                if forced_snapshot_updates > 0:
                    need_update_checkpoint = True
                    logger.warning(
                        "[%s] 终态修正快照草稿标记: status=%s, count=%s",
                        job_id,
                        job.status,
                        forced_snapshot_updates,
                    )

                # 按 _index 排序（已经是正确顺序，但保险起见）
                all_segments.sort(key=lambda x: x.get('id', 0))

                # 语言信息从 transcription 或 job 获取
                detected_language = transcription.get("language") or job.language

                # 进度信息从 transcription 获取
                processed_count = transcription.get("processed_count", 0)
                total_chunks = transcription.get("total_chunks", 0)

            else:
                # 回退到旧格式（unaligned_results）
                unaligned_results = data.get("unaligned_results", [])

                for result in unaligned_results:
                    if not detected_language and 'language' in result:
                        detected_language = result['language']
                    all_segments.extend(result.get('segments', []))

                # 按时间排序
                all_segments.sort(key=lambda x: x.get('start', 0))

                # 重新编号
                for idx, seg in enumerate(all_segments):
                    seg['id'] = idx

                # 进度信息从旧格式获取
                processed_count = len(data.get("processed_indices", []))
                total_chunks = data.get("total_segments", 0)

                # V3.2.0+dev.20260124.02: 旧格式下同样叠加用户编辑
                if edits:
                    apply_edits_to_segments(all_segments, edits)

            # V3.2.0+dev.20260124.02: 过滤用户删除的字幕
            if deleted_indices:
                all_segments = apply_deletions_to_segments(all_segments, deleted_indices)

            # V3.2.0+dev.20260124.02: 追加用户新增字幕
            manual_segments = build_manual_segments(edits, deleted_indices)
            if manual_segments:
                all_segments.extend(manual_segments)
                all_segments.sort(key=lambda x: x.get('start', 0))
            all_segments = filter_hidden_unknown_sentences(all_segments)

            # V3.2.0+dev.20260202.06: 统一按时间排序，避免前端展示/导出乱序
            all_segments.sort(
                key=lambda x: (
                    x.get("start", 0),
                    x.get("end", 0),
                    x.get("id", 0),
                )
            )

            # 统一补齐草稿/定稿标记，避免前后端语义漂移
            for seg in all_segments:
                is_draft = bool(seg.get("is_draft", False))
                is_finalized = seg.get("is_finalized")
                if is_finalized is None:
                    is_finalized = not is_draft
                seg["is_draft"] = is_draft
                seg["is_finalized"] = bool(is_finalized)

            forced_segments = force_finalize_segments_when_finished(
                job_status=job.status,
                segments=all_segments,
            )
            if forced_segments > 0:
                logger.warning(
                    "[%s] 终态收口返回段落草稿标记: status=%s, count=%s",
                    job_id,
                    job.status,
                    forced_segments,
                )

            # 定稿段合并 speaker 信息；草稿段强制不携带 speaker 标签
            finalized_sentence_indices: List[int] = []
            for seg in all_segments:
                if seg.get("id") is None or not bool(seg.get("is_finalized")):
                    continue
                try:
                    finalized_sentence_indices.append(int(seg["id"]))
                except (TypeError, ValueError):
                    continue
            speaker_links_map: Dict[int, Dict[str, Any]] = {}
            if speaker_store_service and finalized_sentence_indices:
                try:
                    speaker_links_map = speaker_store_service.get_subtitle_speaker_map(
                        sentence_indices=finalized_sentence_indices
                    )
                except Exception as speaker_query_exc:
                    logger.warning(
                        "[%s] 查询 speaker 链接失败，返回默认 speaker 字段: %s",
                        job_id,
                        speaker_query_exc,
                    )

            for seg in all_segments:
                if bool(seg.get("is_draft")):
                    seg.pop("speaker_id", None)
                    seg.pop("turn_id", None)
                    seg.pop("speaker_label", None)
                    seg.pop("speaker_color_key", None)
                    seg.pop("binding_source", None)
                    continue

                if not bool(seg.get("is_finalized")):
                    continue

                sentence_index = seg.get("id")
                resolved_sentence_index = None
                if sentence_index is not None:
                    try:
                        resolved_sentence_index = int(sentence_index)
                    except (TypeError, ValueError):
                        resolved_sentence_index = None
                link_row = (
                    speaker_links_map.get(resolved_sentence_index)
                    if resolved_sentence_index is not None
                    else None
                )
                speaker_id = str((link_row or {}).get("speaker_id") or "unknown")
                seg["speaker_id"] = speaker_id
                seg["turn_id"] = (link_row or {}).get("turn_id")
                seg["speaker_label"] = str((link_row or {}).get("speaker_label") or speaker_id)
                seg["speaker_color_key"] = str((link_row or {}).get("speaker_color_key") or "speaker-01")
                seg["binding_source"] = str((link_row or {}).get("binding_source") or "auto")

            # 快照模式下补充进度信息，避免 percentage 为 0
            if using_snapshot:
                if not processed_count:
                    processed_count = len(sentences_snapshot)
                if not total_chunks:
                    total_chunks = job.total or len(sentences_snapshot)

            # V3.1.2: 统一在返回前回写迁移字段（display_confidence/finished收口等）
            if need_update_checkpoint:
                try:
                    target_path = checkpoint_path if checkpoint_path.exists() else snapshot_path
                    dump_obj = data if not using_snapshot else transcription
                    with open(target_path, 'w', encoding='utf-8') as f:
                        json.dump(dump_obj, f, ensure_ascii=False, indent=2)
                    logger.info(f"[{job_id}] 已迁移字幕数据并完成 finished 态收口")
                except Exception as e:
                    logger.warning(f"[{job_id}] 迁移 checkpoint 失败: {e}")

            return {
                "job_id": job_id,
                "has_checkpoint": checkpoint_path.exists(),
                "has_snapshot": using_snapshot,
                "language": detected_language or "unknown",
                "segments": all_segments,
                "sentence_count": len(all_segments),
                "progress": {
                    "processed": processed_count,
                    "total": total_chunks,
                    "percentage": round(
                        processed_count / max(1, total_chunks) * 100,
                        2
                    ) if total_chunks > 0 else 0
                }
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"读取转录文字失败: {str(e)}")

    @router.post("/validate-resume-settings")
    async def validate_resume_settings(
        job_id: str = Form(...),
        new_settings: str = Form(...)
    ):
        """
        校验恢复任务时的参数修改

        返回：
        - valid: bool - 是否可以使用新参数
        - warnings: list - 警告信息
        - errors: list - 错误信息（禁止修改的参数）
        - force_original: dict - 必须强制使用的原始参数
        """
        from pathlib import Path

        job = transcription_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务未找到")

        job_dir = Path(job.dir)
        checkpoint_path = job_dir / "checkpoint.json"

        if not checkpoint_path.exists():
            return {
                "valid": True,
                "warnings": [],
                "errors": [],
                "force_original": {},
                "message": "无检查点，可以使用任意参数"
            }

        try:
            # 加载checkpoint
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            original_settings = checkpoint_data.get("original_settings", {})
            if not original_settings:
                return {
                    "valid": True,
                    "warnings": [],
                    "errors": [],
                    "force_original": {},
                    "message": "检查点未包含原始配置，可直接继续"
                }

            # 解析新设置
            new_settings_obj = json.loads(new_settings)

            warnings = []
            errors = []
            force_original = {}

            task_config = new_settings_obj.get("task_config") or {}
            if task_config and task_config != original_settings:
                warnings.append({
                    "param": "task_config",
                    "level": "medium",
                    "reason": "恢复任务应使用与检查点一致的配置",
                    "impact": "中等",
                    "original": original_settings,
                    "new": task_config,
                    "suggestion": "建议使用检查点原始配置继续"
                })
                force_original["task_config"] = original_settings

            return {
                "valid": len(errors) == 0,
                "warnings": warnings,
                "errors": errors,
                "force_original": force_original,
                "message": "参数校验完成" if len(errors) == 0 else "检测到不兼容的参数修改"
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"参数校验失败: {str(e)}")

    @router.post("/rename-task/{identifier}")
    async def rename_job(identifier: str, title: str = Body(..., embed=True)):
        """
        重命名任务

        Args:
            job_id: 任务ID
            title: 新的任务名称（为空时恢复使用 filename）

        Returns:
            {
                "success": bool,
                "job_id": str,
                "title": str,
                "message": str
            }
        """
        try:
            normalized_identifier = str(identifier or "").strip()
            if not normalized_identifier:
                raise HTTPException(status_code=404, detail="任务未找到")

            # 先解析运行态任务ID，兼容 project_id / legacy job_id
            resolved_runtime_job_id = _resolve_runtime_job_id(normalized_identifier)
            job_id = str(resolved_runtime_job_id or normalized_identifier)
            queue_service = get_queue_service(transcription_service)
            job = queue_service.get_job(job_id)

            if not job:
                # 如果队列服务中没有，尝试从状态仓库恢复
                job = transcription_service.get_job(job_id)

            if not job:
                identity = _resolve_project_identity(normalized_identifier)
                if identity is None:
                    raise HTTPException(status_code=404, detail="任务未找到")

                # 兼容仅编辑项目或已脱离运行态任务的项目重命名
                from app.services.project_service import get_project_service
                project_service = get_project_service()
                updated = project_service.update_title(identity.project_id, title.strip() if title else "")
                if not updated:
                    raise HTTPException(status_code=404, detail="任务未找到")
                project = project_service.get_project(identity.project_id)
                updated_at = getattr(project, "updated_at", None) if project else None
                if updated_at is None:
                    raise HTTPException(status_code=500, detail="任务重命名更新时间戳缺失")
                resolved_title = str(getattr(project, "title", "") or "").strip() if project else (title.strip() if title else "")
                return {
                    "success": True,
                    "job_id": str(resolved_runtime_job_id or identity.project_id),
                    "project_id": identity.project_id,
                    "title": resolved_title,
                    "message": "任务重命名成功",
                    "task": None,
                    "updated_at": updated_at
                }

            canonical_job_id = str(getattr(job, "job_id", "") or "").strip() or job_id

            # 更新 title 字段
            job.title = title.strip() if title else ""
            saved = transcription_service.job_lifecycle.save_job_meta(job)
            if not saved:
                raise HTTPException(status_code=500, detail="任务重命名保存失败")

            # 保存任务状态到文件
            if job.dir:
                from pathlib import Path
                job_dir = Path(job.dir)
                state_file = job_dir / "state.json"

                try:
                    state_data = job.to_dict()
                    with open(state_file, 'w', encoding='utf-8') as f:
                        json.dump(state_data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"保存任务状态失败: {e}")

            persisted_job = transcription_service.job_lifecycle.state_repo.get_task(canonical_job_id)
            if persisted_job is None and canonical_job_id != job_id:
                persisted_job = transcription_service.job_lifecycle.state_repo.get_task(job_id)
            if not persisted_job or persisted_job.updatedAt is None:
                raise HTTPException(status_code=500, detail="任务重命名更新时间戳缺失")
            updated_at = persisted_job.updatedAt
            job_snapshot = _build_task_snapshot(persisted_job)
            # 通知 SSE 订阅者任务信息已更新
            sse_manager.broadcast_sync(
                "global",
                "job_renamed",
                {
                    "job_id": canonical_job_id,
                    "title": job.title,
                    "filename": job.filename,
                    "updated_at": updated_at,
                    "project_id": str(getattr(persisted_job, "project_id", "") or "")
                }
            )

            return {
                "success": True,
                "job_id": canonical_job_id,
                "project_id": str(getattr(persisted_job, "project_id", "") or ""),
                "title": job.title,
                "message": "任务重命名成功",
                "task": job_snapshot,
                "updated_at": updated_at
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"重命名任务失败: {str(e)}")

    return router
