"""
任务生命周期服务

职责：
1. 任务创建与元信息持久化
2. 任务状态恢复与重启纠偏
3. 断点检查与检查点写入
4. 任务暂停/取消与资源清理
"""
from __future__ import annotations

import gc
import json
import logging
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

from app.models.job_models import JobState, JobSettings
from app.models.project_models import generate_project_id, infer_task_mode
from app.models.task_state_machine import get_state_guard
from app.config.lifecycle_config import STATE_MACHINE_GUARD_ENABLED
from app.services.job_index_service import JobIndexService, get_job_index_service
from app.services.sse_service import get_sse_manager
from app.services.task_event_bus import TaskEventBus
from app.services.task_heartbeat import TaskHeartbeatService
from app.services.task_state_repository import TaskStateRepository
from app.services.checkpoint import RuntimeCheckpointService


class JobLifecycleService:
    """
    任务生命周期服务

    统一管理任务元信息、断点与重启恢复。
    """

    _TERMINAL_STATUSES = {"finished", "failed", "canceled", "force_canceled", "removed"}
    _NON_TERMINAL_STATUSES = {
        "created",
        "uploaded",
        "queued",
        "processing",
        "paused",
        "pausing",
        "canceling",
        "transcribing",
        "running",
    }
    _WORKSPACE_PROJECT_PATTERN = re.compile(
        r"^p-\d{8}-\d{6}-(tr|im|lg)-[a-z0-9-]+-[0-9a-z]{4}$"
    )
    _SOURCE_MEDIA_EXTS = {
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".flv",
        ".wmv",
        ".webm",
        ".m4v",
        ".mp3",
        ".wav",
        ".m4a",
        ".aac",
        ".flac",
        ".ogg",
        ".opus",
        ".wma",
    }
    _VIDEO_MEDIA_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm", ".m4v"}
    _AUDIO_MEDIA_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
    _GENERATED_MEDIA_PREFIXES = ("preview_", "proxy_")
    _GENERATED_MEDIA_FILENAMES = {"remux.mp4", "audio.wav"}

    def __init__(
        self,
        jobs_root: Path,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.jobs_root = Path(jobs_root)
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.state_guard = get_state_guard()

        self.jobs: Dict[str, JobState] = {}
        self.lock = threading.Lock()

        self.job_index: JobIndexService = get_job_index_service(str(self.jobs_root))
        self.job_index.cleanup_invalid_mappings()

        # V3.2.0+dev.20260120.05: 状态仓库/事件总线/心跳服务
        self.state_repo = TaskStateRepository(self.jobs_root / "task_state.db", logger=self.logger)
        self.event_bus = TaskEventBus(
            self.state_repo,
            sse_manager=get_sse_manager(),
            logger=self.logger
        )
        self.heartbeat_service = TaskHeartbeatService(self.state_repo, logger=self.logger)

        # 启动时加载已有任务
        self.load_all_jobs_from_disk()

    def load_all_jobs_from_disk(self) -> None:
        """
        启动时从状态仓库加载任务到内存。
        """
        try:
            jobs = self.state_repo.list_tasks()
            jobs = self._prune_import_only_tasks_from_repo(jobs)
            jobs = self._prune_missing_workspace_tasks_from_repo(jobs)
            jobs = self._deduplicate_project_mode_tasks_from_repo(jobs)
            if not jobs:
                self._import_legacy_tasks_from_disk()
                jobs = self.state_repo.list_tasks()
                jobs = self._prune_import_only_tasks_from_repo(jobs)
                jobs = self._prune_missing_workspace_tasks_from_repo(jobs)
                jobs = self._deduplicate_project_mode_tasks_from_repo(jobs)

            loaded_count = 0
            for job in jobs:
                from_status = job.status
                self.state_guard.sync_seq(job.job_id, job.state_seq)
                if self._normalize_task_media_identity(job):
                    self.state_repo.upsert_task(job)
                inferred_project_id = self._infer_project_id_for_loaded_job(job)
                if inferred_project_id and inferred_project_id != str(getattr(job, "project_id", "") or "").strip():
                    job.project_id = inferred_project_id
                    self.state_repo.upsert_task(job)
                if self._apply_restart_pause(job):
                    self._persist_job_state(job, from_status=from_status, reason="system_restart")

                with self.lock:
                    if job.job_id in self.jobs:
                        continue
                    self.jobs[job.job_id] = job
                loaded_count += 1

            if loaded_count > 0:
                self.logger.info(f"启动时已加载 {loaded_count} 个历史任务到内存")
        except Exception as exc:
            self.logger.error(f"加载历史任务失败: {exc}")

    def create_job(
        self,
        filename: str,
        src_path: str,
        settings: JobSettings,
        job_id: Optional[str] = None,
        job_dir_name: Optional[str] = None,
    ) -> JobState:
        """
        创建转录任务
        """
        # 纯 Project 语义：新建任务默认直接使用 project_id。
        job_id = str(job_id or "").strip() or generate_project_id()
        normalized_dir_name = str(job_dir_name or "").strip() or job_id
        job_dir = self.jobs_root / normalized_dir_name
        job_dir.mkdir(parents=True, exist_ok=True)

        dest_path = job_dir / filename

        # V3.1.1+dev.20260106.01: 使用硬链接替代复制，节省磁盘空间
        if os.path.abspath(src_path) != os.path.abspath(dest_path):
            try:
                os.link(src_path, dest_path)
                self.logger.debug(f"硬链接创建成功: {src_path} -> {dest_path}")
            except (OSError, NotImplementedError) as exc:
                self.logger.warning(f"硬链接创建失败，回退到复制: {exc}")
                try:
                    shutil.copyfile(src_path, dest_path)
                    self.logger.debug(f"文件已复制: {src_path} -> {dest_path}")
                except Exception as copy_err:
                    self.logger.warning(f"文件复制失败: {copy_err}")

        job = JobState(
            job_id=job_id,
            filename=filename,
            dir=str(job_dir),
            input_path=src_path,
            settings=settings,
            status="uploaded",
            phase="pending",
            message="文件已上传",
            project_id=job_id,
        )

        with self.lock:
            self.jobs[job_id] = job

        self.job_index.add_mapping(src_path, job_id)
        self._persist_job_state(job, from_status=None, reason="created")

        self._schedule_waveform_audio_extract(job_id, dest_path)

        self.logger.info(f"任务已创建: {job_id} - {filename}")
        return job

    def save_job_meta(self, job: JobState) -> bool:
        """
        保存任务元信息到状态仓库（SQLite）
        """
        try:
            existing = self.state_repo.get_task(job.job_id)
            from_status = existing.status if existing else None
            self._persist_job_state(job, from_status=from_status, reason=None)
            self.logger.debug(f"任务元信息已保存: {job.job_id}")
            return True
        except Exception as exc:
            self.logger.error(f"保存任务元信息失败 {job.job_id}: {exc}")
            return False

    def load_job_meta(self, job_id: str) -> Optional[JobState]:
        """
        从状态仓库加载任务元信息
        """
        try:
            job = self.state_repo.get_task(job_id)
            if job:
                self.logger.debug(f"从状态仓库加载任务: {job_id}")
                return job

            job_dir = self.jobs_root / job_id
            if not job_dir.exists():
                return None
            if self._should_skip_legacy_import_dir(job_dir):
                self.logger.debug("跳过仅编辑项目的任务元信息加载: %s", job_id)
                return None

            legacy_job = self._load_job_meta_from_file(job_id, job_dir)
            if legacy_job is None:
                legacy_job = self._load_job_from_disk_legacy(job_id, job_dir)

            if legacy_job:
                self._persist_job_state(legacy_job, from_status=None, reason="legacy_import")
                self.logger.info(f"已迁移旧任务到仓库: {job_id}")
                return legacy_job
            return None
        except Exception as exc:
            self.logger.error(f"加载任务元信息失败 {job_id}: {exc}")
            return None

    def get_job(self, job_id: str) -> Optional[JobState]:
        """
        获取任务状态
        """
        with self.lock:
            if job_id in self.jobs:
                return self.jobs[job_id]

        job = self.load_job_meta(job_id)
        if job:
            with self.lock:
                self.jobs[job_id] = job
            self.logger.info(f"从状态仓库恢复任务: {job_id}")
            return job

        return None

    def scan_incomplete_jobs(self) -> List[Dict[str, Any]]:
        """
        扫描所有未完成的任务（以状态仓库为准）
        """
        incomplete_jobs: List[Dict[str, Any]] = []

        try:
            jobs = self.state_repo.list_tasks()
            for job in jobs:
                if job.status in self._TERMINAL_STATUSES:
                    continue

                checkpoint_summary = self.state_repo.get_checkpoint_summary(job.job_id) or {}
                total_segments = checkpoint_summary.get("total_segments", 0)
                processed_indices = checkpoint_summary.get("processed_indices", [])
                processed_count = len(processed_indices)
                progress = (
                    (processed_count / total_segments) * 100
                    if total_segments > 0 else job.progress
                )
                file_path = self.job_index.get_file_path(job.job_id)
                filename = os.path.basename(file_path) if file_path else job.filename

                incomplete_jobs.append({
                    "job_id": job.job_id,
                    "filename": filename,
                    "file_path": file_path,
                    "progress": round(progress, 2),
                    "processed_segments": processed_count,
                    "total_segments": total_segments,
                    "phase": checkpoint_summary.get("phase", job.phase),
                    "dir": job.dir,
                    "status": job.status,
                })

            self.logger.info(f"扫描到 {len(incomplete_jobs)} 个未完成任务")
            return incomplete_jobs
        except Exception as exc:
            self.logger.error(f"扫描未完成任务失败: {exc}")
            return []

    def list_tasks_summary(self) -> List[Dict[str, Any]]:
        """
        返回任务摘要列表（用于前端同步）
        """
        summaries_by_identity: Dict[str, Dict[str, Any]] = {}
        for job in self.state_repo.list_tasks():
            if job.status == "removed":
                continue
            filename = str(getattr(job, "filename", "") or "").strip()
            input_path = str(getattr(job, "input_path", "") or "").strip()
            job_dir = Path(
                str(getattr(job, "dir", "") or "").strip()
                or str(self.jobs_root / job.job_id)
            )
            if not filename or self._is_generated_media_filename(filename):
                source_file = self._pick_source_media_file(
                    job_dir=job_dir,
                    preferred_path=input_path,
                )
                if source_file is not None:
                    filename = source_file.name
                elif input_path:
                    filename = Path(input_path).name
                elif not filename:
                    filename = "unknown"
            task_mode = self._infer_task_mode_for_workspace(job_dir)
            summary = {
                "id": job.job_id,
                "job_id": job.job_id,
                "project_id": str(getattr(job, "project_id", "") or "").strip() or job.job_id,
                "filename": filename,
                "title": job.title,
                "status": job.status,
                "progress": job.progress,
                "phase_percent": job.phase_percent,
                "message": job.message,
                "created_time": job.createdAt,
                "updated_at": job.updatedAt,
                "phase": job.phase,
                "processed": job.processed,
                "total": job.total,
                "language": job.language,
                "task_mode": task_mode,
                "is_project_only": task_mode == "subtitle_edit",
            }
            dedupe_key = f"{summary['project_id']}::{summary['task_mode']}"
            existing = summaries_by_identity.get(dedupe_key)
            if existing is None or self._is_snapshot_preferred(summary, existing):
                summaries_by_identity[dedupe_key] = summary
        summaries = list(summaries_by_identity.values())
        summaries.sort(
            key=lambda item: float(item.get("updated_at") or item.get("created_time") or 0.0),
            reverse=True,
        )
        return summaries

    def restore_job_from_checkpoint(self, job_id: str) -> Optional[JobState]:
        """
        从检查点恢复任务状态（无 checkpoint 时从头开始）
        """
        job_dir = self.jobs_root / job_id
        if not job_dir.exists():
            return None

        checkpoint = self._load_checkpoint(job_dir)

        try:
            existing_job = self.state_repo.get_task(job_id)
            filename = existing_job.filename if existing_job else "unknown"
            input_path = existing_job.input_path if existing_job else None
            source_file = self._pick_source_media_file(
                job_dir=job_dir,
                preferred_path=input_path,
            )
            if source_file is not None:
                filename = source_file.name
                input_path = str(source_file)

            if not input_path:
                self.logger.warning(f"无法找到任务 {job_id} 的输入文件")
                return None

            if checkpoint:
                phase = checkpoint.get("phase", "pending")
                total_segments = checkpoint.get("total_segments", 0)
                processed_indices = checkpoint.get("processed_indices", [])
                processed = len(processed_indices)
                progress = round((processed / max(1, total_segments)) * 100, 2)
                message = f"已暂停 ({processed}/{total_segments}段)"
                self.logger.info(f"从检查点恢复任务: {job_id}")
            else:
                phase = "pending"
                total_segments = 0
                processed = 0
                progress = 0
                message = "系统重启，任务已暂停"
                self.logger.info(f"无检查点，任务将从头开始: {job_id}")

            settings = existing_job.settings if existing_job else JobSettings()
            job = JobState(
                job_id=job_id,
                filename=filename,
                dir=str(job_dir),
                input_path=input_path,
                settings=settings,
                status="paused",
                phase=phase,
                message=message,
                total=total_segments,
                processed=processed,
                progress=progress,
                paused=True,
                title=existing_job.title if existing_job else "",
                createdAt=existing_job.createdAt if existing_job else None,
            )

            with self.lock:
                self.jobs[job_id] = job

            self._persist_job_state(job, from_status=existing_job.status if existing_job else None, reason="checkpoint_restore")
            return job
        except Exception as exc:
            self.logger.error(f"从检查点恢复任务失败: {exc}")
            return None

    def check_file_checkpoint(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        检查文件是否有可用的断点
        """
        job_id = self.job_index.get_job_id(file_path)
        if not job_id:
            return None

        job_dir = self.jobs_root / job_id
        if not job_dir.exists():
            self.job_index.remove_mapping(file_path)
            return None

        checkpoint = self.state_repo.get_checkpoint_summary(job_id)
        if not checkpoint:
            checkpoint = self._load_checkpoint(job_dir)
        if not checkpoint:
            return None

        total_segments = checkpoint.get("total_segments", 0)
        processed_indices = checkpoint.get("processed_indices", [])
        processed_count = len(processed_indices)
        progress = (processed_count / total_segments) * 100 if total_segments > 0 else 0

        return {
            "job_id": job_id,
            "progress": round(progress, 2),
            "processed_segments": processed_count,
            "total_segments": total_segments,
            "phase": checkpoint.get("phase", "unknown"),
            "can_resume": True,
        }

    def start_job(self, job_id: str) -> None:
        """
        启动转录任务（兼容入口，不创建线程）
        """
        job = self.get_job(job_id)
        if not job:
            self.logger.warning(f"任务未找到: {job_id}")
            return

        if job.status not in ("uploaded", "failed", "paused", "created"):
            self.logger.warning(f"任务无法启动: {job_id}, 状态: {job.status}")
            return

        from_status = job.status
        job.canceled = False
        job.paused = False
        job.error = None
        self._persist_job_state(job, from_status=from_status, reason="start_job")
        self.logger.warning(f"start_job已废弃，请使用队列服务: {job_id}")

    def pause_job(self, job_id: str) -> bool:
        """
        暂停转录任务（保存断点）
        """
        job = self.get_job(job_id)
        if not job:
            return False

        from_status = job.status
        job.paused = True
        if not self._transition_job_status(job, "paused", "pause_request"):
            return False
        job.message = "暂停中..."
        try:
            runtime_service = RuntimeCheckpointService(job_dir=Path(job.dir))
            runtime_service.mark_pause_requested()
        except Exception as exc:
            self.logger.warning("写入 pause_requested 失败: %s", exc)
        self._persist_job_state(job, from_status=from_status, reason="pause_request")
        self.logger.info(f"⏸️ 任务暂停请求: {job_id}")
        return True

    def cancel_job(self, job_id: str, delete_data: bool = False) -> Tuple[bool, Optional[str]]:
        """
        取消转录任务
        """
        job = self.get_job(job_id)
        if not job:
            return False, "任务未找到"

        from_status = job.status
        job.canceled = True
        job.message = "取消中..."
        try:
            runtime_service = RuntimeCheckpointService(job_dir=Path(job.dir))
            runtime_service.mark_cancel_requested()
        except Exception as exc:
            self.logger.warning("写入 cancel_requested 失败: %s", exc)
        self._persist_job_state(job, from_status=from_status, reason="cancel_request")
        self.logger.info(f"🛑 任务取消请求: {job_id}, 删除数据: {delete_data}")

        if delete_data:
            try:
                job_dir = Path(job.dir)

                try:
                    from app.services.media_stream_tracker import get_active_streams
                    active_streams = get_active_streams(job_id)
                except Exception:
                    active_streams = 0

                if active_streams > 0:
                    msg = "当前有进程占用，请稍后再试"
                    self.logger.warning(
                        f"[删除任务] 文件被占用，放弃删除: {job_id}, active_streams={active_streams}"
                    )
                    return False, msg

                try:
                    from app.services.media_prep_service import get_media_prep_service
                    media_prep = get_media_prep_service()
                    killed = media_prep.cancel_tasks(job_id)
                    if killed > 0:
                        self.logger.info(f"[删除任务] 已终止 {killed} 个 MediaPrep 子进程: {job_id}")
                        import time
                        time.sleep(0.2)
                except Exception as exc:
                    self.logger.warning(f"[删除任务] 取消 MediaPrep 任务失败: {exc}")

                with self.lock:
                    if job_id in self.jobs:
                        del self.jobs[job_id]
                        self.logger.info(f"已从内存移除任务: {job_id}")

                if job.input_path:
                    self.job_index.remove_mapping(job.input_path)

                if job_dir.exists():
                    success = self._force_remove_directory(
                        job_dir, job_id, max_retries=1, fast_fail=True
                    )
                    if not success:
                        return False, "当前有进程占用，请稍后再试"
                    self.logger.info(f"已删除任务数据: {job_id}")
                self.state_repo.delete_task(job_id)
            except Exception as exc:
                self.logger.error(f"删除任务数据失败: {exc}")
                return False, str(exc)

        return True, None

    # ====== checkpoint ======

    # V3.2.0+dev.20260120.04: 检查点写入迁移到生命周期服务
    def _save_checkpoint(self, job_dir: Path, data: dict, job: JobState) -> None:
        """
        保存检查点数据（原子写入）
        """
        job_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = job_dir / "checkpoint.json"
        temp_path = checkpoint_path.with_suffix(".tmp")

        try:
            if "original_settings" not in data and job and job.settings:
                data["original_settings"] = job.settings.to_dict()
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, checkpoint_path)

            # 同步保存任务元信息与检查点摘要（用于重启恢复）
            self.save_job_meta(job)
            summary = {
                "job_id": data.get("job_id", job.job_id),
                "phase": data.get("phase"),
                "total_segments": data.get("total_segments"),
                "processed_indices": data.get("processed_indices", []),
                "processed": len(data.get("processed_indices", [])),
                "language": data.get("language"),
            }
            self.state_repo.save_checkpoint_summary(
                job_id=job.job_id,
                summary=summary,
                file_path=str(checkpoint_path),
                checksum=None,
            )
        except Exception as exc:
            self.logger.error(f"保存检查点失败: {exc}", exc_info=True)
            raise

    def _load_checkpoint(self, job_dir: Path) -> Optional[dict]:
        """
        读取检查点数据（Phase 1: runtime_state.db 优先）
        """
        try:
            runtime_service = RuntimeCheckpointService(job_dir=job_dir)
            if runtime_service.has_runtime_state():
                snapshot = runtime_service.load_snapshot()
                return {
                    "runtime_state": {
                        "source": "runtime_state.db",
                        "last_unit_commits": snapshot.last_unit_commits,
                    }
                }
        except Exception as exc:
            self.logger.warning("读取 runtime_state 失败，回退 checkpoint.json: %s", exc)

        checkpoint_path = job_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            self.logger.warning(
                f"检查点文件损坏，将重新开始任务: {checkpoint_path} - {exc}"
            )
            return None

    def _flush_checkpoint_after_split(
        self,
        job: JobState,
        job_dir: Path,
        processing_mode: Any,
        segments: List[dict],
        demucs_state: Optional[dict] = None
    ) -> None:
        """
        分段完成后强制刷新 checkpoint（确保断点续传一致性）
        """
        checkpoint_data: Dict[str, Any] = {
            "job_id": job.job_id,
            "phase": "split_complete",
            "segments": segments,
            "processing_mode": processing_mode.value,
        }
        if demucs_state:
            checkpoint_data["demucs"] = demucs_state

        self._save_checkpoint(job_dir, checkpoint_data, job)

        saved_checkpoint = self._load_checkpoint(job_dir)
        if saved_checkpoint is None:
            raise RuntimeError("checkpoint write verification failed: file not readable")
        if saved_checkpoint.get("phase") != "split_complete":
            raise RuntimeError("checkpoint write verification failed: phase mismatch")
        if len(saved_checkpoint.get("segments", [])) != len(segments):
            raise RuntimeError("checkpoint write verification failed: segments count mismatch")

        self.logger.info(
            f"checkpoint flushed and verified after split (mode: {processing_mode.value}, "
            f"segments: {len(segments)})"
        )

    # ====== internal helpers ======

    def _transition_job_status(self, job: JobState, target_status: str, reason: str) -> bool:
        """
        通过状态守卫执行任务状态迁移。

        关闭守卫开关时回退为直接赋值，保证灰度可回滚。
        """
        normalized_target = self.state_guard.normalize_status(target_status)
        if not STATE_MACHINE_GUARD_ENABLED:
            job.status = normalized_target
            return True

        self.state_guard.sync_seq(job.job_id, job.state_seq)
        normalized_current = self.state_guard.normalize_status(job.status)
        if normalized_current == normalized_target:
            job.status = normalized_target
            return True

        result = self.state_guard.transition(
            job_id=job.job_id,
            current_status=normalized_current,
            target_status=normalized_target,
            reason=reason,
        )
        if not result.success:
            self.logger.error(
                "状态迁移被拒绝: job=%s, %s -> %s, reason=%s",
                job.job_id,
                normalized_current,
                normalized_target,
                reason,
            )
            return False

        job.status = result.to_status
        job.state_seq = result.state_seq
        return True

    def _apply_restart_pause(self, job: JobState) -> bool:
        """
        系统重启纠偏：将非终态任务统一标记为暂停。
        """
        if job.status in self._TERMINAL_STATUSES:
            return False

        changed = (
            job.status != "paused"
            or not job.paused
            or job.message != "系统重启，任务已暂停"
        )
        is_transitioned = self._transition_job_status(job, "paused", "restart_correction")
        if not is_transitioned:
            return False
        job.paused = True
        job.message = "系统重启，任务已暂停"
        return changed

    def _persist_job_state(
        self,
        job: JobState,
        from_status: Optional[str],
        reason: Optional[str]
    ) -> None:
        normalized_from_status = self.state_guard.normalize_status(from_status) if from_status else from_status
        job.status = self.state_guard.normalize_status(job.status)
        with self.state_repo.transaction() as conn:
            self.state_repo.upsert_task(job, conn=conn)
            if normalized_from_status and normalized_from_status != job.status:
                self.event_bus.emit_status_event(
                    job_id=job.job_id,
                    from_status=normalized_from_status,
                    to_status=job.status,
                    reason=reason,
                    state_seq=job.state_seq,
                    conn=conn,
                )
            elif reason:
                self.event_bus.emit_status_event(
                    job_id=job.job_id,
                    from_status=normalized_from_status,
                    to_status=job.status,
                    reason=reason,
                    state_seq=job.state_seq,
                    conn=conn,
                )

    @classmethod
    def _is_workspace_project_identifier(cls, identifier: str) -> bool:
        normalized_identifier = str(identifier or "").strip()
        if not normalized_identifier:
            return False
        return bool(cls._WORKSPACE_PROJECT_PATTERN.fullmatch(normalized_identifier))

    @classmethod
    def _is_generated_media_filename(cls, filename: str) -> bool:
        normalized_filename = str(filename or "").strip().lower()
        if not normalized_filename:
            return False
        if normalized_filename in cls._GENERATED_MEDIA_FILENAMES:
            return True
        return normalized_filename.startswith(cls._GENERATED_MEDIA_PREFIXES)

    @classmethod
    def _is_source_media_file(cls, file_path: Path, include_generated: bool = False) -> bool:
        if not file_path.is_file():
            return False
        if file_path.suffix.lower() not in cls._SOURCE_MEDIA_EXTS:
            return False
        normalized_name = file_path.name.lower()
        if normalized_name.endswith(".tmp"):
            return False
        if not include_generated and cls._is_generated_media_filename(normalized_name):
            return False
        return True

    def _pick_source_media_file(
        self,
        *,
        job_dir: Path,
        preferred_path: Optional[str] = None,
    ) -> Optional[Path]:
        preferred = Path(str(preferred_path or "").strip()) if str(preferred_path or "").strip() else None
        if preferred is not None:
            try:
                if preferred.exists() and self._is_source_media_file(preferred, include_generated=False):
                    return preferred
            except Exception:
                pass

        try:
            candidates = [
                item for item in sorted(job_dir.iterdir(), key=lambda candidate: candidate.name.lower())
                if self._is_source_media_file(item, include_generated=False)
            ]
            # 优先视频文件，避免 audio.wav 或其他派生音频覆盖原始输入身份
            for item in candidates:
                if item.suffix.lower() in self._VIDEO_MEDIA_EXTS:
                    return item
            for item in candidates:
                if item.suffix.lower() in self._AUDIO_MEDIA_EXTS:
                    return item
        except Exception:
            return None

        # 兜底：目录内仅剩 preview/proxy/remux 时保留可恢复能力
        try:
            for item in sorted(job_dir.iterdir(), key=lambda candidate: candidate.name.lower()):
                if self._is_source_media_file(item, include_generated=True):
                    return item
        except Exception:
            return None
        return None

    def _normalize_task_media_identity(self, job: JobState) -> bool:
        """
        启动纠偏：修复历史任务把 preview/proxy/remux 误写入 filename/input_path 的问题。
        """
        has_changed = False
        job_dir_raw = str(getattr(job, "dir", "") or "").strip()
        job_dir = Path(job_dir_raw) if job_dir_raw else (self.jobs_root / job.job_id)
        if not job_dir.exists():
            return False

        source_file = self._pick_source_media_file(
            job_dir=job_dir,
            preferred_path=str(getattr(job, "input_path", "") or "").strip(),
        )
        if source_file is None:
            return False

        current_filename = str(getattr(job, "filename", "") or "").strip()
        if (not current_filename or self._is_generated_media_filename(current_filename)) and current_filename != source_file.name:
            job.filename = source_file.name
            has_changed = True

        current_input_path = str(getattr(job, "input_path", "") or "").strip()
        if (not current_input_path or self._is_generated_media_filename(Path(current_input_path).name)) and current_input_path != str(source_file):
            job.input_path = str(source_file)
            has_changed = True

        return has_changed

    def _infer_project_id_for_loaded_job(self, job: JobState) -> Optional[str]:
        """
        为历史任务补齐 project_id，避免重启后回退到旧标识。
        """
        current_project_id = str(getattr(job, "project_id", "") or "").strip()
        if current_project_id:
            return current_project_id

        job_dir_raw = str(getattr(job, "dir", "") or "").strip()
        if job_dir_raw:
            workspace_dir = Path(job_dir_raw)
            workspace_name = workspace_dir.name
            if self._is_workspace_project_identifier(workspace_name):
                return workspace_name

            payload = self._load_project_meta_payload(workspace_dir)
            if isinstance(payload, dict):
                meta_project_id = str(payload.get("project_id") or "").strip()
                if meta_project_id:
                    return meta_project_id

        if self._is_workspace_project_identifier(job.job_id):
            return job.job_id

        return str(job.job_id or "").strip() or None

    def _prune_import_only_tasks_from_repo(self, jobs: List[JobState]) -> List[JobState]:
        """
        启动兜底：清理历史误导入到 task_state 的仅编辑项目任务。
        """
        filtered_jobs: List[JobState] = []
        for job in jobs:
            job_dir = Path(str(job.dir or "").strip()) if str(job.dir or "").strip() else self.jobs_root / job.job_id
            if not self._should_skip_legacy_import_dir(job_dir):
                filtered_jobs.append(job)
                continue

            try:
                self.state_repo.delete_task(job.job_id)
                if job.input_path:
                    self.job_index.remove_mapping(job.input_path)
                self.logger.info("已清理仅编辑项目的误导入任务状态: %s", job.job_id)
            except Exception as exc:
                self.logger.warning("清理误导入任务状态失败，将跳过加载: %s (%s)", job.job_id, exc)
        return filtered_jobs

    def _resolve_existing_workspace_for_job(self, job: JobState) -> Tuple[Optional[Path], Optional[str]]:
        """
        解析任务可用工作目录：
        1) 任务内 dir；
        2) jobs/{project_id}、jobs/{job_id}；
        3) ProjectIdResolver（可解析时回填 project_id）。
        """
        normalized_job_id = str(job.job_id or "").strip()
        normalized_project_id = str(getattr(job, "project_id", "") or "").strip()

        candidates: List[Path] = []
        raw_dir = str(getattr(job, "dir", "") or "").strip()
        if raw_dir:
            candidates.append(Path(raw_dir))
        if normalized_project_id:
            candidates.append(self.jobs_root / normalized_project_id)
        if normalized_job_id:
            candidates.append(self.jobs_root / normalized_job_id)

        for candidate in candidates:
            try:
                if candidate.exists():
                    return candidate, normalized_project_id or None
            except Exception:
                continue

        try:
            from app.services.project_id_resolver import get_project_id_resolver

            resolver = get_project_id_resolver()
        except Exception:
            resolver = None

        if resolver is None:
            return None, None

        for identifier in (normalized_project_id, normalized_job_id):
            if not identifier:
                continue
            try:
                identity = resolver.resolve_or_fail(identifier)
                return Path(identity.project_dir), str(identity.project_id or "").strip() or None
            except Exception:
                continue
        return None, None

    def _prune_missing_workspace_tasks_from_repo(self, jobs: List[JobState]) -> List[JobState]:
        """
        启动完整性清理：删除后端已无可用目录的孤儿任务记录。
        """
        filtered_jobs: List[JobState] = []
        for job in jobs:
            workspace_dir, resolved_project_id = self._resolve_existing_workspace_for_job(job)
            if workspace_dir is not None and workspace_dir.exists():
                has_changed = False
                normalized_workspace = str(workspace_dir)
                if str(getattr(job, "dir", "") or "").strip() != normalized_workspace:
                    job.dir = normalized_workspace
                    has_changed = True
                if (
                    resolved_project_id
                    and resolved_project_id != str(getattr(job, "project_id", "") or "").strip()
                ):
                    job.project_id = resolved_project_id
                    has_changed = True
                if has_changed:
                    try:
                        self.state_repo.upsert_task(job)
                    except Exception as exc:
                        self.logger.warning("回写任务目录修正失败，将保持内存修正: %s (%s)", job.job_id, exc)
                filtered_jobs.append(job)
                continue

            try:
                self.state_repo.delete_task(job.job_id)
            except Exception as exc:
                self.logger.warning("清理目录缺失任务失败(状态删除): %s (%s)", job.job_id, exc)

            try:
                if job.input_path:
                    self.job_index.remove_mapping(job.input_path)
                fallback_path = self.job_index.get_file_path(job.job_id)
                if fallback_path:
                    self.job_index.remove_mapping(fallback_path)
            except Exception as exc:
                self.logger.warning("清理目录缺失任务失败(索引清理): %s (%s)", job.job_id, exc)

            self.logger.info("已清理目录缺失任务状态: %s", job.job_id)
        return filtered_jobs

    @staticmethod
    def _is_snapshot_preferred(incoming: Dict[str, Any], existing: Dict[str, Any]) -> bool:
        """比较两个任务快照的优先级。"""
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

    @staticmethod
    def _is_job_preferred_for_dedup(incoming: JobState, existing: JobState) -> bool:
        """比较两条重复任务记录，决定保留哪一条。"""
        incoming_snapshot = {
            "id": str(getattr(incoming, "job_id", "") or ""),
            "project_id": str(getattr(incoming, "project_id", "") or ""),
            "status": str(getattr(incoming, "status", "") or ""),
            "updated_at": float((getattr(incoming, "updatedAt", 0) or 0) / 1000.0),
            "created_time": float((getattr(incoming, "createdAt", 0) or 0) / 1000.0),
            "state_seq": int(getattr(incoming, "state_seq", 0) or 0),
        }
        existing_snapshot = {
            "id": str(getattr(existing, "job_id", "") or ""),
            "project_id": str(getattr(existing, "project_id", "") or ""),
            "status": str(getattr(existing, "status", "") or ""),
            "updated_at": float((getattr(existing, "updatedAt", 0) or 0) / 1000.0),
            "created_time": float((getattr(existing, "createdAt", 0) or 0) / 1000.0),
            "state_seq": int(getattr(existing, "state_seq", 0) or 0),
        }
        return JobLifecycleService._is_snapshot_preferred(incoming_snapshot, existing_snapshot)

    def _rewrite_queue_state_aliases(self, alias_to_primary: Dict[str, str]) -> None:
        """将队列状态中的历史别名重写为保留任务 ID。"""
        normalized_alias_map = {
            str(alias or "").strip(): str(primary or "").strip()
            for alias, primary in (alias_to_primary or {}).items()
            if str(alias or "").strip() and str(primary or "").strip()
        }
        if not normalized_alias_map:
            return

        queue_state = self.state_repo.load_queue_state()
        if queue_state is None:
            return

        def _resolve_alias(identifier: Optional[str]) -> Optional[str]:
            normalized_identifier = str(identifier or "").strip()
            if not normalized_identifier:
                return None
            visited: set[str] = set()
            current = normalized_identifier
            while current in normalized_alias_map and current not in visited:
                visited.add(current)
                current = normalized_alias_map[current]
            return current

        rewritten_queue: List[str] = []
        seen_queue_ids: set[str] = set()
        for queue_id in list(queue_state.queue):
            resolved_queue_id = _resolve_alias(queue_id)
            if not resolved_queue_id or resolved_queue_id in seen_queue_ids:
                continue
            rewritten_queue.append(resolved_queue_id)
            seen_queue_ids.add(resolved_queue_id)

        running_job_id = _resolve_alias(queue_state.running_job_id)
        interrupted_job_id = _resolve_alias(queue_state.interrupted_job_id)
        self.state_repo.save_queue_state(
            queue=rewritten_queue,
            running_job_id=running_job_id,
            interrupted_job_id=interrupted_job_id,
        )

    def _deduplicate_project_mode_tasks_from_repo(self, jobs: List[JobState]) -> List[JobState]:
        """
        启动去重：同一 `project_id + task_mode` 仅保留一条任务记录。

        Why:
        - 历史阶段可能残留“同目录不同 job_id”记录，导致前端出现复制卡片
        - 去重后回写 queue_state，避免运行队列继续引用已删除别名
        """
        selected_by_identity: Dict[str, JobState] = {}
        alias_to_primary: Dict[str, str] = {}

        for job in jobs:
            workspace_dir, resolved_project_id = self._resolve_existing_workspace_for_job(job)
            if workspace_dir is not None and workspace_dir.exists():
                normalized_workspace_dir = str(workspace_dir)
                if str(getattr(job, "dir", "") or "").strip() != normalized_workspace_dir:
                    job.dir = normalized_workspace_dir
                    try:
                        self.state_repo.upsert_task(job)
                    except Exception as exc:
                        self.logger.warning("回写任务目录失败，将继续去重: %s (%s)", job.job_id, exc)
            else:
                workspace_dir = Path(str(getattr(job, "dir", "") or "").strip()) if str(getattr(job, "dir", "") or "").strip() else None

            canonical_project_id = (
                str(resolved_project_id or "").strip()
                or str(self._infer_project_id_for_loaded_job(job) or "").strip()
                or str(getattr(job, "job_id", "") or "").strip()
            )
            if canonical_project_id and canonical_project_id != str(getattr(job, "project_id", "") or "").strip():
                job.project_id = canonical_project_id
                try:
                    self.state_repo.upsert_task(job)
                except Exception as exc:
                    self.logger.warning("回写任务 project_id 失败，将继续去重: %s (%s)", job.job_id, exc)

            inferred_task_mode = "transcribe"
            if workspace_dir is not None and workspace_dir.exists():
                inferred_task_mode = self._infer_task_mode_for_workspace(workspace_dir)
            dedupe_key = f"{canonical_project_id}::{inferred_task_mode}"

            existing = selected_by_identity.get(dedupe_key)
            if existing is None:
                selected_by_identity[dedupe_key] = job
                continue

            if self._is_job_preferred_for_dedup(job, existing):
                alias_to_primary[str(getattr(existing, "job_id", "") or "").strip()] = str(getattr(job, "job_id", "") or "").strip()
                selected_by_identity[dedupe_key] = job
            else:
                alias_to_primary[str(getattr(job, "job_id", "") or "").strip()] = str(getattr(existing, "job_id", "") or "").strip()

        removed_task_ids: set[str] = set()
        kept_task_ids = {
            str(getattr(job, "job_id", "") or "").strip()
            for job in selected_by_identity.values()
            if str(getattr(job, "job_id", "") or "").strip()
        }
        for alias_id, primary_id in alias_to_primary.items():
            normalized_alias_id = str(alias_id or "").strip()
            normalized_primary_id = str(primary_id or "").strip()
            if (
                not normalized_alias_id
                or not normalized_primary_id
                or normalized_alias_id == normalized_primary_id
                or normalized_alias_id in kept_task_ids
                or normalized_alias_id in removed_task_ids
            ):
                continue
            try:
                self.state_repo.delete_task(normalized_alias_id)
                removed_task_ids.add(normalized_alias_id)
            except Exception as exc:
                self.logger.warning("删除重复任务行失败，将继续: %s (%s)", normalized_alias_id, exc)

        if removed_task_ids:
            self._rewrite_queue_state_aliases(alias_to_primary)
            self.logger.info(
                "启动去重完成: removed=%s kept=%s",
                len(removed_task_ids),
                len(selected_by_identity),
            )

        return list(selected_by_identity.values())

    def _load_project_meta_payload(self, job_dir: Path) -> Optional[Dict[str, Any]]:
        """
        读取项目元信息；读取失败时返回 None，不阻断主流程。
        """
        project_meta_path = job_dir / "project_meta.json"
        if not project_meta_path.exists():
            return None
        try:
            with open(project_meta_path, "r", encoding="utf-8") as meta_file:
                payload = json.load(meta_file)
            if isinstance(payload, dict):
                return payload
            self.logger.warning("project_meta.json 结构非法，忽略过滤: %s", project_meta_path)
            return None
        except Exception as exc:
            self.logger.warning("读取 project_meta.json 失败，忽略过滤: %s (%s)", project_meta_path, exc)
            return None

    def _should_skip_legacy_import_dir(self, job_dir: Path) -> bool:
        """
        判定目录是否应跳过 legacy 任务导入。

        规则：
        - `project_meta.task_mode=subtitle_edit`
          视为仅编辑项目，不参与任务状态导入。
        """
        payload = self._load_project_meta_payload(job_dir)
        if payload is None:
            return False

        task_mode = self._infer_task_mode_from_payload(payload=payload, project_dir=job_dir)
        if task_mode == "subtitle_edit":
            self.logger.debug("识别为仅编辑项目，跳过 legacy 导入: %s", job_dir)
            return True
        return False

    def _infer_task_mode_for_workspace(self, job_dir: Path) -> str:
        payload = self._load_project_meta_payload(job_dir)
        return self._infer_task_mode_from_payload(payload=payload, project_dir=job_dir)

    @staticmethod
    def _infer_task_mode_from_payload(
        *,
        payload: Optional[Dict[str, Any]],
        project_dir: Path,
    ) -> str:
        if not isinstance(payload, dict):
            return "transcribe"
        subtitle_doc = payload.get("subtitle_doc")
        subtitle_source_type = (
            str(subtitle_doc.get("source_type") or "").strip().lower()
            if isinstance(subtitle_doc, dict)
            else ""
        )
        return infer_task_mode(
            raw_task_mode=payload.get("task_mode"),
            project_mode=payload.get("mode"),
            subtitle_source_type=subtitle_source_type,
            project_dir=str(project_dir),
        )

    def _import_legacy_tasks_from_disk(self) -> None:
        """
        兼容迁移：从旧版 job_meta.json / checkpoint.json 导入任务。
        """
        imported = 0
        for job_dir in self.jobs_root.iterdir():
            if not job_dir.is_dir():
                continue

            job_id = job_dir.name
            if self.state_repo.get_task(job_id):
                continue
            if self._should_skip_legacy_import_dir(job_dir):
                continue

            job = self._load_job_meta_from_file(job_id, job_dir)
            if job is None:
                job = self._load_job_from_disk_legacy(job_id, job_dir)

            if job:
                self.state_repo.upsert_task(job)
                self.state_repo.record_event(
                    job_id=job.job_id,
                    event_type="legacy_import",
                    from_status=None,
                    to_status=job.status,
                    reason="legacy_import",
                    state_seq=job.state_seq,
                    payload={"source": "job_meta_or_checkpoint"},
                )
                if job.input_path:
                    self.job_index.add_mapping(job.input_path, job.job_id)
                imported += 1

        if imported > 0:
            self.logger.info(f"已导入 {imported} 个旧任务到状态仓库")

    def _load_job_meta_from_file(self, job_id: str, job_dir: Path) -> Optional[JobState]:
        """
        旧版兼容：从 job_meta.json 读取任务元信息。
        """
        meta_file = job_dir / "job_meta.json"
        if not meta_file.exists():
            return None
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            job = JobState.from_meta_dict(data)
            job.dir = str(job_dir)
            inferred_project_id = self._infer_project_id_for_loaded_job(job)
            if inferred_project_id:
                job.project_id = inferred_project_id
            self.logger.debug(f"从 job_meta.json 读取任务: {job_id}")
            return job
        except Exception as exc:
            self.logger.warning(f"读取 job_meta.json 失败 {job_id}: {exc}")
            return None

    def _load_job_from_disk_legacy(self, job_id: str, job_dir: Path) -> Optional[JobState]:
        """
        旧版兼容：从目录推断任务信息。
        """
        filename = "未知文件"
        input_path = None

        source_file = self._pick_source_media_file(job_dir=job_dir)
        if source_file is not None:
            filename = source_file.name
            input_path = str(source_file)

        srt_files = list(job_dir.glob("*.srt"))
        is_finished = len(srt_files) > 0

        job = JobState(
            job_id=job_id,
            filename=filename,
            dir=str(job_dir),
            input_path=input_path,
            status="finished" if is_finished else "paused",
            phase="editing" if is_finished else "transcribing",
            progress=100 if is_finished else 0,
            message="已完成" if is_finished else "系统重启，任务已暂停",
            srt_path=str(srt_files[0]) if srt_files else None,
            paused=not is_finished,
        )
        inferred_project_id = self._infer_project_id_for_loaded_job(job)
        if inferred_project_id:
            job.project_id = inferred_project_id

        checkpoint_path = job_dir / "checkpoint.json"
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    checkpoint_data = json.load(f)
                total_segments = checkpoint_data.get("total_segments", 0)
                processed_indices = checkpoint_data.get("processed_indices", [])
                if total_segments > 0:
                    job.progress = min((len(processed_indices) / total_segments) * 100, 100)
                job.phase = checkpoint_data.get("phase", job.phase)
                job.language = checkpoint_data.get("language")
                if "unaligned_results" in checkpoint_data:
                    job.segments = checkpoint_data["unaligned_results"]
            except Exception as exc:
                self.logger.warning(f"读取checkpoint失败 {checkpoint_path}: {exc}")

        return job

    def _schedule_waveform_audio_extract(self, job_id: str, dest_path: Path) -> None:
        """
        异步提取音频供波形图使用。
        """
        audio_path = Path(dest_path).parent / "audio.wav"
        if audio_path.exists():
            self.logger.debug(f"[{job_id}] 音频文件已存在，跳过提取")
            return

        self.logger.info(f"[{job_id}] 启动后台音频提取...")

        def extract_audio_for_waveform() -> None:
            """后台提取音频供波形图使用"""
            try:
                # V3.2.4+dev.20260301.01: 使用 FFmpeg 提取（与导入模式一致，避免波形偏移）
                from app.utils.audio_extractor import audio_extractor
                audio_extractor.extract_fast_sync(Path(dest_path), audio_path)
                self.logger.info(f"[{job_id}] 音频提取完成（FFmpeg PCM_16）: {audio_path}")
            except Exception as exc:
                self.logger.error(f"[{job_id}] 音频提取失败: {exc}")

        threading.Thread(
            target=extract_audio_for_waveform,
            daemon=True,
            name=f"AudioExtract-{job_id[:8]}"
        ).start()

    def _force_remove_directory(
        self,
        directory: Path,
        job_id: str,
        max_retries: int = 3,
        fast_fail: bool = False
    ) -> bool:
        """
        强制删除目录（处理 Windows 文件占用问题）
        """
        import time
        import stat

        gc.collect()
        time.sleep(0.1)

        attempts = 1 if fast_fail else max_retries
        for attempt in range(attempts):
            try:
                shutil.rmtree(directory)
                self.logger.info(f"[强制删除] 成功删除目录: {job_id}, 尝试次数: {attempt + 1}")
                return True
            except PermissionError as exc:
                if fast_fail or attempt >= attempts - 1:
                    self.logger.warning(f"[强制删除] 删除失败 (快速返回): {exc}")
                    return False
                self.logger.warning(
                    f"[强制删除] 删除失败 (尝试 {attempt + 1}/{max_retries}): {exc}, "
                    f"等待 {0.5 * (attempt + 1)}s 后重试"
                )
                time.sleep(0.5 * (attempt + 1))

        failed_files = []
        for root, dirs, files in os.walk(directory, topdown=False):
            for name in files:
                file_path = Path(root) / name
                try:
                    os.chmod(file_path, stat.S_IWRITE)
                    file_path.unlink()
                except Exception as exc:
                    self.logger.warning(f"[强制删除] 无法删除文件: {file_path.name}, {exc}")
                    failed_files.append(str(file_path))

            for name in dirs:
                dir_path = Path(root) / name
                try:
                    dir_path.rmdir()
                except Exception as exc:
                    self.logger.debug(f"[强制删除] 无法删除目录: {dir_path.name}, {exc}")

        try:
            directory.rmdir()
            self.logger.info(f"[强制删除] 逐个删除完成: {job_id}")
            return True
        except Exception as exc:
            if failed_files:
                self.logger.error(
                    f"[强制删除] 部分文件无法删除: {job_id}, "
                    f"失败文件数: {len(failed_files)}, 错误: {exc}"
                )
            else:
                self.logger.warning(f"[强制删除] 根目录删除失败: {job_id}, {exc}")
            return False


_job_lifecycle_service: Optional[JobLifecycleService] = None


def get_job_lifecycle_service(
    jobs_root: Path,
    logger: Optional[logging.Logger] = None
) -> JobLifecycleService:
    """
    获取 JobLifecycleService 单例
    """
    global _job_lifecycle_service
    if _job_lifecycle_service is None:
        _job_lifecycle_service = JobLifecycleService(jobs_root=jobs_root, logger=logger)
    return _job_lifecycle_service
