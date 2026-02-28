"""
Project 工作目录迁移服务。

设计模式：
- Domain Service（领域服务）：集中承载目录命名迁移与元数据回写策略。
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from threading import RLock
from time import sleep, time
from typing import TYPE_CHECKING, Any, Iterable, Optional

from app.core.config import config
from app.services.project_id_resolver import ProjectIdentity, get_project_id_resolver
from app.services.project_naming_service import (
    ProjectNamingService,
    get_project_naming_service,
)
from app.services.project_service import ProjectService, get_project_service

if TYPE_CHECKING:
    from app.services.task_state_repository import TaskStateRepository

logger = logging.getLogger(__name__)


class ProjectDirectoryMigrator:
    """Project 工作目录迁移服务。"""

    _NAME_PATTERN = re.compile(
        r"^p-\d{8}-\d{6}-(tr|im|lg)-[a-z0-9-]+-[0-9a-z]{4}$"
    )
    _ACTIVE_STATUSES = {
        # 方案里定义的跳过状态
        "running",
        "queued",
        "pausing",
        "canceling",
        # 与现有实现保持兼容的运行态别名
        "processing",
        "transcribing",
    }
    _TARGET_RETRY = 8

    def __init__(
        self,
        *,
        project_service: Optional[ProjectService] = None,
        naming_service: Optional[ProjectNamingService] = None,
    ) -> None:
        self._project_service = project_service or get_project_service()
        self._naming_service = naming_service or get_project_naming_service()

    def migrate(
        self,
        *,
        jobs_root: Path,
        state_jobs: Optional[Iterable[Any]] = None,
        state_repo: Optional["TaskStateRepository"] = None,
        dry_run: bool = False,
        allow_legacy_resolution: bool = True,
    ) -> dict[str, Any]:
        """
        执行目录命名迁移。

        Args:
            jobs_root: jobs 根目录。
            state_jobs: 运行态内存任务对象列表（可选）。
            state_repo: 任务状态仓库（可选）。
            dry_run: True 时仅输出计划，不执行改名与回写。
            allow_legacy_resolution: 是否允许在迁移过程中触发 legacy -> project 解析。
        """
        report: dict[str, Any] = {
            "dry_run": bool(dry_run),
            "scanned": 0,
            "renamed": 0,
            "skipped": 0,
            "failed": 0,
            "renamed_items": [],
            "skipped_items": [],
            "failures": [],
            "updated_jobs": [],
            "updated_task_rows": 0,
        }
        jobs_root = Path(jobs_root)
        if not jobs_root.exists():
            return report

        runtime_jobs = list(state_jobs or [])
        active_context = self._build_active_context(
            runtime_jobs=runtime_jobs,
            state_repo=state_repo,
        )
        updated_job_ids: set[int] = set()
        processed_project_ids: set[str] = set()

        candidates = [
            item
            for item in sorted(jobs_root.iterdir(), key=lambda path: path.name.lower())
            if item.is_dir() and not item.name.startswith("_")
        ]
        for workspace_dir in candidates:
            report["scanned"] += 1
            try:
                try:
                    identity = self._resolve_identity(
                        workspace_dir=workspace_dir,
                        allow_legacy_resolution=allow_legacy_resolution,
                    )
                except FileNotFoundError as identity_exc:
                    if not allow_legacy_resolution:
                        report["skipped"] += 1
                        report["skipped_items"].append(
                            {
                                "dir": str(workspace_dir),
                                "reason": "unresolved_workspace",
                                "error": str(identity_exc),
                            }
                        )
                        continue
                    raise
                if identity.project_id in processed_project_ids:
                    report["skipped"] += 1
                    report["skipped_items"].append(
                        {
                            "dir": str(workspace_dir),
                            "project_id": identity.project_id,
                            "reason": "duplicate_project",
                        }
                    )
                    continue
                processed_project_ids.add(identity.project_id)

                source_dir = Path(identity.project_dir)
                if not source_dir.exists():
                    report["failed"] += 1
                    report["failures"].append(
                        {
                            "dir": str(workspace_dir),
                            "project_id": identity.project_id,
                            "error": "project_dir_missing_after_resolve",
                        }
                    )
                    continue

                if self._is_canonical_name(source_dir.name):
                    report["skipped"] += 1
                    report["skipped_items"].append(
                        {
                            "dir": str(source_dir),
                            "project_id": identity.project_id,
                            "reason": "already_canonical",
                        }
                    )
                    continue

                if self._is_project_active(
                    project_id=identity.project_id,
                    legacy_job_id=identity.legacy_job_id,
                    project_dir=source_dir,
                    active_context=active_context,
                ):
                    report["skipped"] += 1
                    report["skipped_items"].append(
                        {
                            "dir": str(source_dir),
                            "project_id": identity.project_id,
                            "reason": "active_task",
                        }
                    )
                    continue

                mode, title, source_filename = self._derive_naming_inputs(
                    project_id=identity.project_id,
                    project_dir=source_dir,
                    runtime_jobs=runtime_jobs,
                )
                target_dir = self._pick_target_dir(
                    jobs_root=jobs_root,
                    source_dir=source_dir,
                    mode=mode,
                    title=title,
                    source_filename=source_filename,
                )

                if dry_run:
                    report["renamed"] += 1
                    report["renamed_items"].append(
                        {
                            "project_id": identity.project_id,
                            "source_dir": str(source_dir),
                            "target_dir": str(target_dir),
                            "mode": mode,
                            "legacy_job_id": identity.legacy_job_id,
                        }
                    )
                    continue

                self._rename_dir_with_retry(source_dir=source_dir, target_dir=target_dir)
                canonical_project_id = self._rewrite_project_meta(
                    project_id=identity.project_id,
                    target_dir=target_dir,
                )
                self._rewrite_job_meta(
                    target_dir=target_dir,
                    project_id=canonical_project_id,
                )
                updated_rows = self._rewrite_task_rows(
                    state_repo=state_repo,
                    source_dir=source_dir,
                    target_dir=target_dir,
                    project_id=canonical_project_id,
                    legacy_job_id=identity.legacy_job_id,
                )
                report["updated_task_rows"] += int(updated_rows)
                self._rewrite_runtime_jobs(
                    runtime_jobs=runtime_jobs,
                    source_dir=source_dir,
                    target_dir=target_dir,
                    project_id=canonical_project_id,
                    legacy_job_id=identity.legacy_job_id,
                    report=report,
                    updated_job_ids=updated_job_ids,
                )
                report["renamed"] += 1
                report["renamed_items"].append(
                    {
                        "project_id": canonical_project_id,
                        "source_dir": str(source_dir),
                        "target_dir": str(target_dir),
                        "mode": mode,
                        "legacy_job_id": identity.legacy_job_id,
                    }
                )
            except Exception as exc:
                report["failed"] += 1
                report["failures"].append(
                    {
                        "dir": str(workspace_dir),
                        "error": str(exc),
                    }
                )

        return report

    def _resolve_identity(
        self,
        *,
        workspace_dir: Path,
        allow_legacy_resolution: bool,
    ) -> ProjectIdentity:
        if allow_legacy_resolution:
            return get_project_id_resolver().resolve_or_fail(workspace_dir.name)

        direct_project = self._project_service._load_project_meta(workspace_dir)  # type: ignore[attr-defined]
        if direct_project is None or not str(direct_project.project_id or "").strip():
            raise FileNotFoundError("workspace 尚未 project 化，dry-run 不触发 legacy 迁移")

        return ProjectIdentity(
            project_id=str(direct_project.project_id),
            project_dir=workspace_dir,
            input_identifier=workspace_dir.name,
            legacy_job_id=str(direct_project.job_id or "").strip() or workspace_dir.name,
        )

    def _derive_naming_inputs(
        self,
        *,
        project_id: str,
        project_dir: Path,
        runtime_jobs: list[Any],
    ) -> tuple[str, str, str]:
        project = self._project_service.get_project(project_id)
        if project is None:
            project = self._project_service._load_project_meta(project_dir)  # type: ignore[attr-defined]
        if project is None:
            raise FileNotFoundError(f"project_meta 丢失: {project_id}")

        source_type = ""
        if project.subtitle_doc is not None:
            source_type = str(project.subtitle_doc.source_type or "").strip().lower()

        mode = "transcribe"
        if str(project.mode or "").strip().lower() == "legacy" or source_type == "legacy":
            mode = "legacy"
        elif source_type == "import":
            mode = "import"

        title = str(project.title or "").strip() or project_id
        source_filename = self._guess_source_filename(
            project_dir=project_dir,
            runtime_jobs=runtime_jobs,
            project_id=project_id,
        )
        return mode, title, source_filename

    def _guess_source_filename(
        self,
        *,
        project_dir: Path,
        runtime_jobs: list[Any],
        project_id: str,
    ) -> str:
        for runtime_job in runtime_jobs:
            runtime_project_id = str(getattr(runtime_job, "project_id", "") or "").strip()
            if runtime_project_id and runtime_project_id != project_id:
                continue
            filename = str(getattr(runtime_job, "filename", "") or "").strip()
            if filename:
                return filename

        job_meta_path = project_dir / "job_meta.json"
        if job_meta_path.exists():
            try:
                payload = json.loads(job_meta_path.read_text(encoding="utf-8"))
                filename = str(payload.get("filename", "") or "").strip()
                if filename:
                    return filename
            except Exception:
                pass

        media_exts = {
            ".mp4",
            ".avi",
            ".mkv",
            ".mov",
            ".wmv",
            ".webm",
            ".flv",
            ".m4v",
            ".wav",
            ".mp3",
            ".m4a",
            ".aac",
            ".flac",
            ".ogg",
            ".opus",
            ".wma",
        }
        for item in project_dir.iterdir():
            if item.is_file() and item.suffix.lower() in media_exts:
                return item.name
        return project_dir.name

    def _pick_target_dir(
        self,
        *,
        jobs_root: Path,
        source_dir: Path,
        mode: str,
        title: str,
        source_filename: str,
    ) -> Path:
        for _ in range(self._TARGET_RETRY):
            dir_name = self._naming_service.generate_workspace_dir_name(
                jobs_root=jobs_root,
                mode=mode,
                title=title,
                source_filename=source_filename,
            )
            target_dir = jobs_root / dir_name
            if self._paths_equal(target_dir, source_dir):
                continue
            if target_dir.exists():
                continue
            return target_dir
        raise RuntimeError(f"目录迁移目标命名冲突重试超限: {source_dir}")

    def _rewrite_project_meta(self, *, project_id: str, target_dir: Path) -> str:
        project = self._project_service.get_project(project_id)
        if project is None:
            project = self._project_service._load_project_meta(target_dir)  # type: ignore[attr-defined]
        if project is None:
            raise FileNotFoundError(f"无法加载 project_meta: {project_id}")

        # 目录命名迁移后，非 legacy 项目以 canonical 目录名作为 project_id，避免重启回退旧标识。
        project_mode = str(getattr(project, "mode", "") or "").strip().lower()
        subtitle_doc = getattr(project, "subtitle_doc", None)
        source_type = str(getattr(subtitle_doc, "source_type", "") or "").strip().lower()
        target_project_id = str(getattr(project, "project_id", "") or "").strip() or project_id
        if self._is_canonical_name(target_dir.name) and project_mode != "legacy":
            previous_project_id = target_project_id
            target_project_id = target_dir.name
            if previous_project_id != target_project_id:
                if not str(getattr(project, "job_id", "") or "").strip() and previous_project_id:
                    project.job_id = previous_project_id
                project.project_id = target_project_id

        if subtitle_doc is not None:
            subtitle_doc.project_id = target_project_id
            doc_id = str(getattr(subtitle_doc, "doc_id", "") or "").strip()
            if not doc_id or doc_id == str(getattr(project, "job_id", "") or "").strip():
                subtitle_doc.doc_id = target_project_id
            subtitle_doc.updated_at = time()

        if source_type == "import" and project_mode != "legacy":
            project.mode = "import"

        project.dir = str(target_dir)
        self._project_service.save_project(project, project_dir=target_dir)
        return str(getattr(project, "project_id", "") or "").strip() or target_project_id

    @staticmethod
    def _rewrite_job_meta(*, target_dir: Path, project_id: str) -> None:
        job_meta_path = target_dir / "job_meta.json"
        if not job_meta_path.exists():
            return
        try:
            payload = json.loads(job_meta_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return
            payload["dir"] = str(target_dir)
            payload["project_id"] = project_id
            temp_path = job_meta_path.with_suffix(job_meta_path.suffix + ".tmp")
            temp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temp_path, job_meta_path)
        except Exception as exc:
            logger.warning("回写 job_meta 失败（忽略，不中断迁移）: %s", exc)

    def _rewrite_task_rows(
        self,
        *,
        state_repo: Optional["TaskStateRepository"],
        source_dir: Path,
        target_dir: Path,
        project_id: str,
        legacy_job_id: Optional[str],
    ) -> int:
        if state_repo is None:
            return 0
        updated = 0
        tasks = state_repo.list_tasks()
        for task in tasks:
            if not self._matches_project(
                job=task,
                project_id=project_id,
                legacy_job_id=legacy_job_id,
                source_dir=source_dir,
            ):
                continue
            changed = False
            if str(getattr(task, "dir", "") or "").strip() != str(target_dir):
                task.dir = str(target_dir)
                changed = True
            if str(getattr(task, "project_id", "") or "").strip() != project_id:
                task.project_id = project_id
                changed = True
            if not changed:
                continue
            state_repo.upsert_task(task)
            updated += 1
        return updated

    def _rewrite_runtime_jobs(
        self,
        *,
        runtime_jobs: list[Any],
        source_dir: Path,
        target_dir: Path,
        project_id: str,
        legacy_job_id: Optional[str],
        report: dict[str, Any],
        updated_job_ids: set[int],
    ) -> None:
        for runtime_job in runtime_jobs:
            if not self._matches_project(
                job=runtime_job,
                project_id=project_id,
                legacy_job_id=legacy_job_id,
                source_dir=source_dir,
            ):
                continue
            changed = False
            if str(getattr(runtime_job, "project_id", "") or "").strip() != project_id:
                runtime_job.project_id = project_id
                changed = True
            if str(getattr(runtime_job, "dir", "") or "").strip() != str(target_dir):
                runtime_job.dir = str(target_dir)
                changed = True
            if not changed:
                continue

            runtime_key = id(runtime_job)
            if runtime_key in updated_job_ids:
                continue
            updated_job_ids.add(runtime_key)
            report["updated_jobs"].append(runtime_job)

    def _build_active_context(
        self,
        *,
        runtime_jobs: list[Any],
        state_repo: Optional["TaskStateRepository"],
    ) -> dict[str, set[str]]:
        active_project_ids: set[str] = set()
        active_job_ids: set[str] = set()
        active_dirs: set[str] = set()

        def register(job_obj: Any) -> None:
            status = str(getattr(job_obj, "status", "") or "").strip().lower()
            if status not in self._ACTIVE_STATUSES:
                return
            project_id = str(getattr(job_obj, "project_id", "") or "").strip()
            job_id = str(getattr(job_obj, "job_id", "") or "").strip()
            job_dir = str(getattr(job_obj, "dir", "") or "").strip()

            if project_id:
                active_project_ids.add(project_id)
            if job_id:
                active_job_ids.add(job_id)
            if job_dir:
                active_dirs.add(job_dir.lower())

        for runtime_job in runtime_jobs:
            register(runtime_job)

        if state_repo is not None:
            try:
                active_rows = state_repo.list_tasks(statuses=sorted(self._ACTIVE_STATUSES))
                for row in active_rows:
                    register(row)
            except Exception as exc:
                logger.warning("读取 task_state 活跃任务失败，继续迁移: %s", exc)

        return {
            "project_ids": active_project_ids,
            "job_ids": active_job_ids,
            "dirs": active_dirs,
        }

    def _is_project_active(
        self,
        *,
        project_id: str,
        legacy_job_id: Optional[str],
        project_dir: Path,
        active_context: dict[str, set[str]],
    ) -> bool:
        if project_id in active_context["project_ids"]:
            return True
        if project_id in active_context["job_ids"]:
            return True
        normalized_legacy_job_id = str(legacy_job_id or "").strip()
        if normalized_legacy_job_id and normalized_legacy_job_id in active_context["job_ids"]:
            return True
        if str(project_dir).lower() in active_context["dirs"]:
            return True
        return False

    def _matches_project(
        self,
        *,
        job: Any,
        project_id: str,
        legacy_job_id: Optional[str],
        source_dir: Path,
    ) -> bool:
        runtime_project_id = str(getattr(job, "project_id", "") or "").strip()
        runtime_job_id = str(getattr(job, "job_id", "") or "").strip()
        runtime_dir = str(getattr(job, "dir", "") or "").strip()
        if runtime_project_id == project_id:
            return True
        if runtime_job_id == project_id:
            return True
        if legacy_job_id and runtime_job_id == str(legacy_job_id):
            return True
        if runtime_dir and runtime_dir.lower() == str(source_dir).lower():
            return True
        return False

    def _rename_dir_with_retry(
        self,
        *,
        source_dir: Path,
        target_dir: Path,
        max_retries: int = 8,
    ) -> None:
        if not source_dir.exists():
            raise FileNotFoundError(f"源目录不存在: {source_dir}")
        if target_dir.exists():
            raise FileExistsError(f"目标目录已存在: {target_dir}")

        last_exc: Optional[BaseException] = None
        for attempt in range(1, max_retries + 1):
            try:
                os.replace(source_dir, target_dir)
                return
            except FileExistsError:
                raise
            except (PermissionError, OSError) as exc:
                if not self._is_windows_busy_error(exc):
                    raise
                last_exc = exc
                wait_seconds = min(1.6, 0.1 * (2 ** (attempt - 1)))
                logger.warning(
                    "目录改名被占用，准备重试: source=%s target=%s attempt=%s/%s wait=%.2fs err=%s",
                    source_dir,
                    target_dir,
                    attempt,
                    max_retries,
                    wait_seconds,
                    exc,
                )
                sleep(wait_seconds)
        raise RuntimeError(f"目录改名失败（被占用）: {source_dir} -> {target_dir}, err={last_exc}") from last_exc

    @staticmethod
    def _is_windows_busy_error(exc: BaseException) -> bool:
        if not isinstance(exc, OSError):
            return False
        return getattr(exc, "winerror", None) in {5, 32}

    @classmethod
    def _is_canonical_name(cls, dir_name: str) -> bool:
        return bool(cls._NAME_PATTERN.fullmatch(str(dir_name or "").strip()))

    @staticmethod
    def _paths_equal(path_a: Path, path_b: Path) -> bool:
        try:
            return path_a.resolve() == path_b.resolve()
        except Exception:
            return path_a == path_b


_project_directory_migrator: Optional[ProjectDirectoryMigrator] = None
_project_directory_migrator_lock = RLock()


def get_project_directory_migrator() -> ProjectDirectoryMigrator:
    """获取 ProjectDirectoryMigrator 单例。"""
    global _project_directory_migrator
    if _project_directory_migrator is not None:
        return _project_directory_migrator
    with _project_directory_migrator_lock:
        if _project_directory_migrator is None:
            _project_directory_migrator = ProjectDirectoryMigrator()
    return _project_directory_migrator
