"""
Project 语义任务控制路由。

设计说明：
- 采用“Project 主语义 + 运行态 job 执行”的兼容模式；
- 路由入口统一接收 project_id（或可解析标识），内部通过 ProjectIdResolver 映射到运行态 job_id；
- 返回体保留兼容字段 job_id，同时以 project_id 作为主字段。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from app.models.project_models import infer_task_mode


def _get_queue_service(transcription_service):
    from app.services.job_queue_service import get_queue_service

    return get_queue_service(transcription_service)

def _get_project_service():
    from app.services.project_service import get_project_service

    return get_project_service()


def create_project_task_router(transcription_service) -> APIRouter:
    """创建 Project 语义任务控制路由。"""
    router = APIRouter(prefix="/api/projects", tags=["project-tasks"])

    def _resolve_project_identity(identifier: str):
        from app.services.project_id_resolver import get_project_id_resolver

        normalized_identifier = str(identifier or "").strip()
        if not normalized_identifier:
            raise HTTPException(status_code=404, detail="任务未找到")
        try:
            return get_project_id_resolver().resolve_or_fail(normalized_identifier)
        except Exception as exc:
            raise HTTPException(status_code=404, detail="任务未找到") from exc

    def _find_runtime_job_for_project(
        project_id: str,
        *,
        legacy_job_id: Optional[str] = None,
    ) -> Optional[Any]:
        runtime_job = transcription_service.get_job(project_id)
        if runtime_job is not None:
            return runtime_job

        normalized_legacy_job_id = str(legacy_job_id or "").strip()
        if normalized_legacy_job_id:
            runtime_job = transcription_service.get_job(normalized_legacy_job_id)
            if runtime_job is not None:
                return runtime_job

        job_lifecycle = getattr(transcription_service, "job_lifecycle", None)
        runtime_jobs = getattr(job_lifecycle, "jobs", None)
        if isinstance(runtime_jobs, dict):
            for candidate in runtime_jobs.values():
                if str(getattr(candidate, "project_id", "") or "").strip() == project_id:
                    return candidate
        return None

    def _resolve_runtime_job_or_404(project_identifier: str):
        identity = _resolve_project_identity(project_identifier)
        runtime_job = _find_runtime_job_for_project(
            identity.project_id,
            legacy_job_id=identity.legacy_job_id,
        )
        if runtime_job is None:
            raise HTTPException(status_code=404, detail="任务未找到")
        runtime_job.project_id = identity.project_id
        if str(getattr(runtime_job, "dir", "") or "").strip() != str(identity.project_dir):
            runtime_job.dir = str(identity.project_dir)
        return identity, runtime_job

    def _resolve_job_workspace(runtime_job: Any, identity: Any) -> Path:
        runtime_dir = Path(getattr(runtime_job, "dir", "") or "")
        if runtime_dir.exists():
            return runtime_dir
        return Path(identity.project_dir)

    def _is_repo_task_match(candidate: Any, identifiers: set[str]) -> bool:
        normalized_identifiers = {item for item in identifiers if item}
        if not normalized_identifiers:
            return False
        candidate_job_id = str(getattr(candidate, "job_id", "") or "").strip()
        candidate_project_id = str(getattr(candidate, "project_id", "") or "").strip()
        candidate_dir = str(getattr(candidate, "dir", "") or "").strip()
        candidate_workspace = Path(candidate_dir).name if candidate_dir else ""
        return (
            candidate_job_id in normalized_identifiers
            or candidate_project_id in normalized_identifiers
            or candidate_workspace in normalized_identifiers
        )

    def _find_repo_task_for_identifiers(identifiers: set[str]) -> Optional[Any]:
        lifecycle = getattr(transcription_service, "job_lifecycle", None)
        state_repo = getattr(lifecycle, "state_repo", None)
        if state_repo is None:
            return None
        try:
            candidates = []
            for item in state_repo.list_tasks():
                if _is_repo_task_match(item, identifiers):
                    candidates.append(item)
            if not candidates:
                return None
            candidates.sort(
                key=lambda task: float(getattr(task, "updatedAt", 0) or 0.0),
                reverse=True,
            )
            return candidates[0]
        except Exception:
            return None

    def _build_task_snapshot(job: Any) -> Dict[str, Any]:
        project_id = str(getattr(job, "project_id", "") or "").strip() or str(
            getattr(job, "job_id", "") or ""
        ).strip()
        resolved_project_mode: Optional[str] = None
        if project_id:
            try:
                resolved_project = _get_project_service().get_project(project_id)
            except Exception:
                resolved_project = None
            if resolved_project is not None:
                resolved_project_mode = _resolve_project_task_mode(resolved_project)

        task_mode = infer_task_mode(
            raw_task_mode=resolved_project_mode or getattr(job, "task_mode", None),
            project_mode=None,
            subtitle_source_type=None,
            project_dir=str(getattr(job, "dir", "") or ""),
        )
        return {
            "id": getattr(job, "job_id", ""),
            "project_id": project_id or None,
            "filename": getattr(job, "filename", ""),
            "title": getattr(job, "title", ""),
            "status": getattr(job, "status", ""),
            "progress": getattr(job, "progress", 0),
            "phase": getattr(job, "phase", ""),
            "phase_percent": getattr(job, "phase_percent", 0),
            "message": getattr(job, "message", ""),
            "processed": getattr(job, "processed", 0),
            "total": getattr(job, "total", 0),
            "language": getattr(job, "language", None),
            "updated_at": getattr(job, "updatedAt", None),
            "task_mode": task_mode,
            "is_project_only": task_mode == "subtitle_edit",
        }

    def _map_cancel_error_status(reason_code: str, message: str) -> int:
        normalized_reason = str(reason_code or "").strip()
        normalized_message = str(message or "")
        if normalized_reason == "delete_blocked" or "占用" in normalized_message:
            return 423
        if normalized_reason == "cancel_running" or "取消中" in normalized_message:
            return 409
        if normalized_reason == "cancel_not_found":
            return 404
        return 400

    def _build_project_only_task_snapshot(project: Any) -> Dict[str, Any]:
        """
        为“仅编辑项目（无运行态任务）”构建任务卡片快照。
        """
        project_id = str(getattr(project, "project_id", "") or "").strip()
        title = str(getattr(project, "title", "") or "").strip()
        subtitle_doc = getattr(project, "subtitle_doc", None)
        source_type = str(getattr(subtitle_doc, "source_type", "") or "").strip()
        task_mode = _resolve_project_task_mode(project)
        segment_count = int(getattr(subtitle_doc, "segment_count", 0) or 0)
        created_at = getattr(project, "created_at", None)
        updated_at = getattr(project, "updated_at", None)

        if title:
            filename = title
        elif task_mode == "subtitle_edit":
            filename = f"{project_id}.srt"
        else:
            filename = project_id

        return {
            "id": project_id,
            "project_id": project_id,
            "filename": filename,
            "title": title,
            "status": "finished",
            "progress": 100.0,
            "phase": "edit_ready",
            "phase_percent": 100.0,
            "message": "仅编辑项目",
            "processed": segment_count,
            "total": segment_count,
            "language": None,
            "created_time": created_at,
            "updated_at": updated_at,
            "source_type": source_type or "import",
            "task_mode": task_mode,
            "is_project_only": task_mode == "subtitle_edit",
        }

    def _resolve_project_task_mode(project: Any) -> str:
        subtitle_doc = getattr(project, "subtitle_doc", None)
        source_type = str(getattr(subtitle_doc, "source_type", "") or "").strip().lower()
        return infer_task_mode(
            raw_task_mode=getattr(project, "task_mode", None),
            project_mode=getattr(project, "mode", None),
            subtitle_source_type=source_type,
            project_dir=str(getattr(project, "dir", "") or ""),
        )

    def _is_project_only_project(project: Any) -> bool:
        return _resolve_project_task_mode(project) == "subtitle_edit"

    def _build_media_status_from_workspace(project_id: str, workspace_dir: Path) -> Dict[str, Any]:
        video_exts = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".flv", ".m4v"}
        audio_exts = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}

        video_exists = any(
            item.is_file() and item.suffix.lower() in video_exts
            for item in workspace_dir.iterdir()
        )
        audio_exists = any(
            item.is_file() and item.suffix.lower() in audio_exts
            for item in workspace_dir.iterdir()
        )
        proxy_exists = (workspace_dir / "proxy_720p.mp4").exists() or (workspace_dir / "preview_360p.mp4").exists()
        peaks_ready = (workspace_dir / "peaks.json").exists()
        thumbnails_ready = (
            (workspace_dir / "thumbnail_sprite.jpg").exists()
            or (workspace_dir / "thumbnails").exists()
        )
        srt_exists = any(item.is_file() and item.suffix.lower() == ".srt" for item in workspace_dir.iterdir())

        return {
            "video_exists": bool(video_exists),
            "video_format": ".mp4" if video_exists else None,
            "needs_proxy": False,
            "proxy_exists": bool(proxy_exists),
            "audio_exists": bool(audio_exists),
            "peaks_ready": bool(peaks_ready),
            "thumbnails_ready": bool(thumbnails_ready),
            "srt_exists": bool(srt_exists),
            "video_url": f"/api/media/{project_id}/video" if (video_exists or proxy_exists) else None,
            "audio_url": f"/api/media/{project_id}/audio" if audio_exists else None,
            "peaks_url": f"/api/media/{project_id}/peaks" if audio_exists else None,
            "thumbnails_url": f"/api/media/{project_id}/thumbnails" if video_exists else None,
            "srt_url": f"/api/media/{project_id}/srt" if srt_exists else None,
        }

    @router.get("/{project_id}/tasks/status")
    async def get_project_task_status(project_id: str, include_media: bool = True):
        """获取 Project 任务状态（兼容返回 job 字段）。"""
        identity = _resolve_project_identity(project_id)
        runtime_job = _find_runtime_job_for_project(
            identity.project_id,
            legacy_job_id=identity.legacy_job_id,
        )
        if runtime_job is None:
            project_service = _get_project_service()
            project = project_service.get_project(identity.project_id)
            if project is None or not _is_project_only_project(project):
                raise HTTPException(status_code=404, detail="任务未找到")

            result = _build_project_only_task_snapshot(project)
            result["job_id"] = str(getattr(project, "job_id", "") or "") or identity.project_id
            result["project_id"] = identity.project_id
            result["queue_position"] = -1
            if include_media:
                workspace_dir = Path(identity.project_dir)
                if workspace_dir.exists():
                    result["media_status"] = _build_media_status_from_workspace(
                        project_id=identity.project_id,
                        workspace_dir=workspace_dir,
                    )
            return result

        runtime_job.project_id = identity.project_id
        if str(getattr(runtime_job, "dir", "") or "").strip() != str(identity.project_dir):
            runtime_job.dir = str(identity.project_dir)
        runtime_job_id = str(getattr(runtime_job, "job_id", "") or "")
        result = runtime_job.to_dict() if hasattr(runtime_job, "to_dict") else {}
        result["job_id"] = runtime_job_id
        result["project_id"] = identity.project_id
        runtime_task_mode = infer_task_mode(
            raw_task_mode=result.get("task_mode", getattr(runtime_job, "task_mode", None)),
            project_mode=None,
            subtitle_source_type=None,
            project_dir=str(getattr(runtime_job, "dir", "") or str(identity.project_dir)),
        )
        result["task_mode"] = runtime_task_mode
        result["is_project_only"] = runtime_task_mode == "subtitle_edit"

        queue_state = transcription_service.job_lifecycle.state_repo.load_queue_state()
        if queue_state:
            if runtime_job_id in queue_state.queue:
                result["queue_position"] = queue_state.queue.index(runtime_job_id) + 1
            elif runtime_job_id == queue_state.running_job_id:
                result["queue_position"] = 0
            else:
                result["queue_position"] = -1
        else:
            result["queue_position"] = -1

        if include_media and str(getattr(runtime_job, "status", "") or "") == "finished":
            workspace_dir = _resolve_job_workspace(runtime_job, identity)
            if workspace_dir.exists() and hasattr(runtime_job, "update_media_status"):
                runtime_job.update_media_status(str(workspace_dir))
            media_status = getattr(runtime_job, "media_status", None)
            if media_status is not None:
                result["media_status"] = {
                    "video_exists": bool(getattr(media_status, "video_exists", False)),
                    "video_format": getattr(media_status, "video_format", None),
                    "needs_proxy": bool(getattr(media_status, "needs_proxy", False)),
                    "proxy_exists": bool(getattr(media_status, "proxy_exists", False)),
                    "audio_exists": bool(getattr(media_status, "audio_exists", False)),
                    "peaks_ready": bool(getattr(media_status, "peaks_ready", False)),
                    "thumbnails_ready": bool(getattr(media_status, "thumbnails_ready", False)),
                    "srt_exists": bool(getattr(media_status, "srt_exists", False)),
                    "video_url": (
                        f"/api/media/{identity.project_id}/video"
                        if bool(getattr(media_status, "video_exists", False))
                        or bool(getattr(media_status, "proxy_exists", False))
                        else None
                    ),
                    "audio_url": (
                        f"/api/media/{identity.project_id}/audio"
                        if bool(getattr(media_status, "audio_exists", False))
                        else None
                    ),
                    "peaks_url": (
                        f"/api/media/{identity.project_id}/peaks"
                        if bool(getattr(media_status, "audio_exists", False))
                        else None
                    ),
                    "thumbnails_url": (
                        f"/api/media/{identity.project_id}/thumbnails"
                        if bool(getattr(media_status, "video_exists", False))
                        else None
                    ),
                    "srt_url": (
                        f"/api/media/{identity.project_id}/srt"
                        if bool(getattr(media_status, "srt_exists", False))
                        else None
                    ),
                }
        return result

    @router.post("/{project_id}/tasks/pause")
    async def pause_project_task(project_id: str):
        """暂停任务。"""
        identity, runtime_job = _resolve_runtime_job_or_404(project_id)
        runtime_job_id = str(getattr(runtime_job, "job_id", "") or "")
        queue_service = _get_queue_service(transcription_service)
        ok = queue_service.pause_job(runtime_job_id)
        if not ok:
            raise HTTPException(status_code=404, detail="任务未找到")

        job_snapshot = None
        persisted_job = transcription_service.job_lifecycle.state_repo.get_task(runtime_job_id)
        if persisted_job is not None:
            persisted_job.project_id = identity.project_id
            job_snapshot = _build_task_snapshot(persisted_job)
        return {
            "project_id": identity.project_id,
            "job_id": runtime_job_id,
            "paused": bool(ok),
            "task": job_snapshot,
        }

    @router.post("/{project_id}/tasks/resume")
    async def resume_project_task(project_id: str):
        """恢复任务。"""
        identity, runtime_job = _resolve_runtime_job_or_404(project_id)
        runtime_job_id = str(getattr(runtime_job, "job_id", "") or "")
        queue_service = _get_queue_service(transcription_service)
        ok = queue_service.resume_job(runtime_job_id)
        if not ok:
            raise HTTPException(status_code=400, detail="无法恢复任务（任务未暂停或不存在）")

        queue_position = 0
        if runtime_job_id in queue_service.queue:
            queue_position = list(queue_service.queue).index(runtime_job_id) + 1
        job_snapshot = None
        persisted_job = transcription_service.job_lifecycle.state_repo.get_task(runtime_job_id)
        if persisted_job is not None:
            persisted_job.project_id = identity.project_id
            job_snapshot = _build_task_snapshot(persisted_job)
        return {
            "project_id": identity.project_id,
            "job_id": runtime_job_id,
            "resumed": True,
            "status": getattr(runtime_job, "status", "queued"),
            "queue_position": queue_position,
            "task": job_snapshot,
        }

    @router.post("/{project_id}/tasks/cancel")
    async def cancel_project_task(project_id: str, delete_data: bool = False):
        """取消任务。"""
        requested_identifier = str(project_id or "").strip()
        if not requested_identifier:
            raise HTTPException(status_code=404, detail="任务未找到")

        identity = None
        runtime_job = None
        target_project_id = requested_identifier
        try:
            identity = _resolve_project_identity(project_id)
            target_project_id = identity.project_id
            runtime_job = _find_runtime_job_for_project(
                identity.project_id,
                legacy_job_id=identity.legacy_job_id,
            )
            if runtime_job is not None:
                runtime_job.project_id = identity.project_id
                if str(getattr(runtime_job, "dir", "") or "").strip() != str(identity.project_dir):
                    runtime_job.dir = str(identity.project_dir)
        except HTTPException:
            identity = None
            runtime_job = None

        runtime_job_id = str(getattr(runtime_job, "job_id", "") or "").strip()
        if not runtime_job_id:
            if not delete_data:
                raise HTTPException(status_code=404, detail="任务未找到")
            fallback_identifiers = {requested_identifier, target_project_id}
            repo_task = _find_repo_task_for_identifiers(fallback_identifiers)
            if repo_task is not None:
                runtime_job_id = str(getattr(repo_task, "job_id", "") or "").strip()
                repo_project_id = str(getattr(repo_task, "project_id", "") or "").strip()
                if repo_project_id:
                    target_project_id = repo_project_id
            if not runtime_job_id:
                # 兜底到请求标识，复用 queue_service.cancel_job(delete_data=True) 的幂等删除能力。
                runtime_job_id = requested_identifier

        queue_service = _get_queue_service(transcription_service)
        result = queue_service.cancel_job(runtime_job_id, delete_data=delete_data)
        if not result.success:
            status_code = _map_cancel_error_status(
                reason_code=getattr(result, "reason_code", ""),
                message=getattr(result, "message", ""),
            )
            raise HTTPException(
                status_code=status_code,
                detail=getattr(result, "message", "任务未找到"),
            )

        job_snapshot = None
        persisted_job = transcription_service.job_lifecycle.state_repo.get_task(runtime_job_id)
        if persisted_job is not None:
            persisted_project_id = str(getattr(persisted_job, "project_id", "") or "").strip()
            if persisted_project_id:
                target_project_id = persisted_project_id
            persisted_job.project_id = target_project_id
            job_snapshot = _build_task_snapshot(persisted_job)
        return {
            "project_id": target_project_id,
            "job_id": runtime_job_id,
            "canceled": bool(result.success),
            "data_deleted": bool(delete_data),
            "success": bool(result.success),
            "status": getattr(result, "status", ""),
            "reason_code": getattr(result, "reason_code", ""),
            "message": getattr(result, "message", ""),
            "pending_delete": bool(getattr(result, "pending_delete", False)),
            "state_seq": int(getattr(result, "state_seq", 0) or 0),
            "task": job_snapshot,
        }

    @router.post("/{project_id}/tasks/prioritize")
    async def prioritize_project_task(project_id: str, mode: Optional[str] = None):
        """插队任务。"""
        identity, runtime_job = _resolve_runtime_job_or_404(project_id)
        runtime_job_id = str(getattr(runtime_job, "job_id", "") or "")
        queue_service = _get_queue_service(transcription_service)
        result = queue_service.prioritize_job(runtime_job_id, mode=mode)
        if not bool(result.get("success")):
            raise HTTPException(status_code=400, detail=result.get("error", "无法优先此任务"))

        interrupted_job_id = str(result.get("interrupted_job_id") or "").strip()
        interrupted_project_id = None
        if interrupted_job_id:
            try:
                interrupted_identity = _resolve_project_identity(interrupted_job_id)
                interrupted_project_id = interrupted_identity.project_id
            except HTTPException:
                interrupted_project_id = None

        return {
            "project_id": identity.project_id,
            "job_id": runtime_job_id,
            "prioritized": True,
            "mode": result.get("mode"),
            "interrupted_job_id": interrupted_job_id or None,
            "interrupted_project_id": interrupted_project_id,
            "queue_position": 1,
        }

    @router.get("/tasks/queue-status")
    async def get_project_task_queue_status():
        """获取队列状态（增加 project 语义镜像字段）。"""
        queue_service = _get_queue_service(transcription_service)
        queue_status = queue_service.get_queue_status()

        queue_job_ids = list(queue_status.get("queue", []))
        queue_project_ids = []
        for queued_job_id in queue_job_ids:
            try:
                queue_project_ids.append(_resolve_project_identity(str(queued_job_id)).project_id)
            except HTTPException:
                queue_project_ids.append(str(queued_job_id))

        running_job_id = queue_status.get("running")
        running_project_id = None
        if running_job_id:
            try:
                running_project_id = _resolve_project_identity(str(running_job_id)).project_id
            except HTTPException:
                running_project_id = None

        queue_status["queue_project_ids"] = queue_project_ids
        queue_status["running_project_id"] = running_project_id
        return queue_status

    @router.get("/tasks/sync")
    async def sync_project_tasks():
        """
        同步任务列表（project 语义镜像）。
        """
        def _is_snapshot_preferred(incoming: Dict[str, Any], existing: Dict[str, Any]) -> bool:
            active_statuses = {"processing", "queued", "paused", "canceling", "pausing", "created", "uploaded"}
            incoming_status = str(incoming.get("status", "") or "").strip().lower()
            existing_status = str(existing.get("status", "") or "").strip().lower()
            incoming_active = incoming_status in active_statuses
            existing_active = existing_status in active_statuses
            if incoming_active != existing_active:
                return incoming_active

            incoming_updated_at = float(incoming.get("updated_at") or incoming.get("created_time") or 0.0)
            existing_updated_at = float(existing.get("updated_at") or existing.get("created_time") or 0.0)
            if incoming_updated_at != existing_updated_at:
                return incoming_updated_at > existing_updated_at

            incoming_seq = int(incoming.get("state_seq") or 0)
            existing_seq = int(existing.get("state_seq") or 0)
            if incoming_seq != existing_seq:
                return incoming_seq > existing_seq

            incoming_id = str(incoming.get("id", "") or "").strip()
            existing_id = str(existing.get("id", "") or "").strip()
            incoming_project_id = str(incoming.get("project_id", "") or "").strip()
            existing_project_id = str(existing.get("project_id", "") or "").strip()
            incoming_is_canonical = incoming_id == incoming_project_id and bool(incoming_project_id)
            existing_is_canonical = existing_id == existing_project_id and bool(existing_project_id)
            if incoming_is_canonical != existing_is_canonical:
                return incoming_is_canonical
            return incoming_id > existing_id

        lifecycle = transcription_service.job_lifecycle
        project_service = _get_project_service()
        tasks = lifecycle.list_tasks_summary()
        existing_project_ids: set[str] = set()
        for task in tasks:
            task_id = str(task.get("id", "") or "").strip()
            if not task_id:
                continue
            resolved_project = None
            task_project_id = str(task.get("project_id", "") or "").strip()
            resolve_seed = task_project_id or task_id
            try:
                resolved_project_id = _resolve_project_identity(resolve_seed).project_id
                task["project_id"] = resolved_project_id
                existing_project_ids.add(resolved_project_id)
                resolved_project = project_service.get_project(resolved_project_id)
            except HTTPException:
                alias_project = project_service.find_project_by_alias(resolve_seed)
                if alias_project is not None:
                    alias_project_id = str(getattr(alias_project, "project_id", "") or "").strip()
                    if alias_project_id:
                        task["project_id"] = alias_project_id
                        existing_project_ids.add(alias_project_id)
                        resolved_project = alias_project
                    else:
                        task["project_id"] = resolve_seed
                        existing_project_ids.add(resolve_seed)
                else:
                    task["project_id"] = resolve_seed
                    existing_project_ids.add(resolve_seed)

            subtitle_doc = getattr(resolved_project, "subtitle_doc", None) if resolved_project is not None else None
            subtitle_source_type = (
                str(getattr(subtitle_doc, "source_type", "") or "").strip().lower()
                if subtitle_doc is not None
                else str(task.get("source_type", "") or "").strip().lower()
            )
            if resolved_project is not None:
                resolved_task_mode = _resolve_project_task_mode(resolved_project)
            else:
                resolved_task_mode = infer_task_mode(
                    raw_task_mode=task.get("task_mode"),
                    project_mode=None,
                    subtitle_source_type=subtitle_source_type,
                    project_dir=str(task.get("dir", "") or ""),
                )
            task["task_mode"] = resolved_task_mode
            task["is_project_only"] = resolved_task_mode == "subtitle_edit"

        # 合并“仅编辑项目”（存在 project_meta，但无运行态 task_state）
        for project in project_service.list_projects():
            project_id = str(getattr(project, "project_id", "") or "").strip()
            if not project_id or project_id in existing_project_ids:
                continue
            if not _is_project_only_project(project):
                continue
            tasks.append(_build_project_only_task_snapshot(project))
            existing_project_ids.add(project_id)

        deduped_tasks_by_identity: Dict[str, Dict[str, Any]] = {}
        for task in tasks:
            project_id = str(task.get("project_id", "") or task.get("id", "") or "").strip()
            if not project_id:
                continue
            task_mode = str(task.get("task_mode", "") or "transcribe").strip().lower()
            if task_mode not in {"transcribe", "subtitle_edit"}:
                task_mode = "transcribe"
            dedupe_key = f"{project_id}::{task_mode}"
            existing = deduped_tasks_by_identity.get(dedupe_key)
            if existing is None or _is_snapshot_preferred(task, existing):
                deduped_tasks_by_identity[dedupe_key] = task
        tasks = list(deduped_tasks_by_identity.values())

        tasks.sort(
            key=lambda item: float(item.get("updated_at") or item.get("created_time") or 0.0),
            reverse=True,
        )

        queue_state = lifecycle.state_repo.load_queue_state()
        queue_job_ids = list(queue_state.queue) if queue_state else []
        queue_project_ids = []
        for queued_job_id in queue_job_ids:
            try:
                queue_project_ids.append(_resolve_project_identity(str(queued_job_id)).project_id)
            except HTTPException:
                queue_project_ids.append(str(queued_job_id))

        running_job_id = queue_state.running_job_id if queue_state else None
        running_project_id = None
        if running_job_id:
            try:
                running_project_id = _resolve_project_identity(str(running_job_id)).project_id
            except HTTPException:
                running_project_id = None

        return {
            "success": True,
            "tasks": tasks,
            "count": len(tasks),
            "queue": queue_job_ids,
            "queue_project_ids": queue_project_ids,
            "running_job_id": running_job_id,
            "running_project_id": running_project_id,
            "queue_updated_at": queue_state.updated_at if queue_state else None,
        }

    return router
