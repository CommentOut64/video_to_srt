"""
Project 身份解析服务。

设计模式：
- Adapter（适配器模式）：统一接收 project_id / job_id / legacy job_id，输出规范 project 语义。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import time
from typing import Any, Optional

from app.core.config import config
from app.services.legacy_projection_service import get_legacy_projection_service
from app.services.project_service import get_project_service

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectIdentity:
    """统一身份解析结果。"""

    project_id: str
    project_dir: Path
    input_identifier: str
    legacy_job_id: Optional[str] = None


class ProjectIdResolver:
    """统一 `resolve_or_fail(identifier)` 入口。"""
    _WORKSPACE_PROJECT_PATTERN = re.compile(
        r"^p-\d{8}-\d{6}-(tr|im|lg)-[a-z0-9-]+-[0-9a-z]{4}$"
    )

    def __init__(self) -> None:
        self._project_service = get_project_service()
        self._legacy_service = get_legacy_projection_service()
        self._lock = RLock()

    def resolve_or_fail(self, identifier: str) -> ProjectIdentity:
        normalized_identifier = str(identifier or "").strip()
        if not normalized_identifier:
            raise FileNotFoundError("identifier 不能为空")

        with self._lock:
            # 1) 先按 project_id 直查
            project = self._project_service.get_project(normalized_identifier)
            if project:
                identity = self._build_identity_if_possible(project, normalized_identifier)
                if identity is not None:
                    return identity

            # 1.5) 再按 project 元信息别名匹配（job_id / subtitle_doc.project_id / subtitle_doc.doc_id）
            alias_project = self._project_service.find_project_by_alias(normalized_identifier)
            if alias_project:
                alias_workspace_dir = (
                    Path(str(getattr(alias_project, "dir", "") or "").strip())
                    if str(getattr(alias_project, "dir", "") or "").strip()
                    else None
                )
                alias_workspace_name = alias_workspace_dir.name if alias_workspace_dir else ""
                if (
                    alias_workspace_dir is not None
                    and alias_workspace_dir.exists()
                    and self._should_prefer_workspace_identity(
                        workspace_name=alias_workspace_name,
                        project=alias_project,
                    )
                ):
                    alias_project = self._rebind_workspace_identity(
                        workspace_name=alias_workspace_name,
                        workspace_dir=alias_workspace_dir,
                        project=alias_project,
                    )
                identity = self._build_identity_if_possible(alias_project, normalized_identifier)
                if identity is not None:
                    return identity

            # 2) 再按目录直达（兼容旧目录输入）
            direct_dir = config.JOBS_DIR / normalized_identifier
            if direct_dir.exists():
                direct_project = self._project_service._load_project_meta(direct_dir)  # type: ignore[attr-defined]
                if direct_project and direct_project.project_id:
                    # 旧目录命中（目录名 != project_id）时，必须先完成目录规范化，后续仅按 project 语义加载。
                    if normalized_identifier != direct_project.project_id:
                        if self._should_prefer_workspace_identity(
                            workspace_name=normalized_identifier,
                            project=direct_project,
                        ):
                            direct_project = self._rebind_workspace_identity(
                                workspace_name=normalized_identifier,
                                workspace_dir=direct_dir,
                                project=direct_project,
                            )
                            identity = self._build_identity_if_possible(direct_project, normalized_identifier)
                            if identity is not None:
                                return identity
                        project_id, _ = self._legacy_service.resolve(normalized_identifier)
                        migrated_project = self._project_service.get_project(project_id)
                        if migrated_project:
                            identity = self._build_identity_if_possible(migrated_project, normalized_identifier)
                            if identity is not None:
                                return identity
                    else:
                        identity = self._build_identity_if_possible(direct_project, normalized_identifier)
                        if identity is not None:
                            return identity

                # 目录存在但尚未 project 化：按 legacy 流程强制转换为 project 目录语义。
                project_id, _ = self._legacy_service.resolve(normalized_identifier)
                migrated_project = self._project_service.get_project(project_id)
                if migrated_project:
                    identity = self._build_identity_if_possible(migrated_project, normalized_identifier)
                    if identity is not None:
                        return identity

            # 3) 最后按 legacy job_id 映射
            project_id, _ = self._legacy_service.resolve(normalized_identifier)
            migrated_project = self._project_service.get_project(project_id)
            if migrated_project:
                identity = self._build_identity_if_possible(migrated_project, normalized_identifier)
                if identity is not None:
                    return identity

        raise FileNotFoundError(f"任务不存在: {normalized_identifier}")

    def migrate_existing_workspaces(
        self,
        *,
        state_jobs: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        """
        批量规范化现有 workspace 到 project 语义。

        返回统计信息与发生变更的内存任务对象（由调用方决定是否持久化）。
        """
        report: dict[str, Any] = {
            "scanned": 0,
            "resolved": 0,
            "migrated_alias": 0,
            "failed": 0,
            "failures": [],
            "updated_jobs": [],
        }
        state_jobs_list = list(state_jobs or [])

        candidate_identifiers: set[str] = set()
        if config.JOBS_DIR.exists():
            for item in config.JOBS_DIR.iterdir():
                if not item.is_dir():
                    continue
                if item.name.startswith("_"):
                    continue
                candidate_identifiers.add(item.name)

        for job in state_jobs_list:
            job_id = str(getattr(job, "job_id", "") or "").strip()
            project_id = str(getattr(job, "project_id", "") or "").strip()
            if job_id:
                candidate_identifiers.add(job_id)
            if project_id:
                candidate_identifiers.add(project_id)

        for identifier in sorted(candidate_identifiers):
            report["scanned"] += 1
            try:
                identity = self.resolve_or_fail(identifier)
                report["resolved"] += 1
                if identifier != identity.project_id:
                    report["migrated_alias"] += 1
            except Exception as exc:
                report["failed"] += 1
                report["failures"].append(
                    {"identifier": identifier, "error": str(exc)}
                )

        for job in state_jobs_list:
            job_dir_value = str(getattr(job, "dir", "") or "").strip()
            workspace_name = Path(job_dir_value).name if job_dir_value else ""
            anchor_identifier = ""
            if ProjectIdResolver._is_workspace_identifier(workspace_name):
                anchor_identifier = workspace_name
            if not anchor_identifier:
                anchor_identifier = (
                    str(getattr(job, "project_id", "") or "").strip()
                    or str(getattr(job, "job_id", "") or "").strip()
                )
            if not anchor_identifier:
                continue
            try:
                identity = self.resolve_or_fail(anchor_identifier)
            except Exception as exc:
                report["failed"] += 1
                report["failures"].append(
                    {"identifier": anchor_identifier, "error": str(exc)}
                )
                continue

            has_changed = False
            if str(getattr(job, "project_id", "") or "").strip() != identity.project_id:
                job.project_id = identity.project_id
                has_changed = True
            if str(getattr(job, "dir", "") or "").strip() != str(identity.project_dir):
                job.dir = str(identity.project_dir)
                has_changed = True
            if has_changed:
                report["updated_jobs"].append(job)

        return report

    @classmethod
    def _is_workspace_identifier(cls, identifier: str) -> bool:
        normalized_identifier = str(identifier or "").strip()
        if not normalized_identifier:
            return False
        return bool(cls._WORKSPACE_PROJECT_PATTERN.fullmatch(normalized_identifier))

    def _should_prefer_workspace_identity(self, *, workspace_name: str, project) -> bool:
        if not self._is_workspace_identifier(workspace_name):
            return False
        project_mode = str(getattr(project, "mode", "") or "").strip().lower()
        return project_mode != "legacy"

    def _rebind_workspace_identity(self, *, workspace_name: str, workspace_dir: Path, project):
        normalized_workspace_name = str(workspace_name or "").strip()
        if not normalized_workspace_name:
            return project

        previous_project_id = str(getattr(project, "project_id", "") or "").strip()
        if previous_project_id == normalized_workspace_name:
            self._normalize_project_metadata(project=project, project_id=normalized_workspace_name)
            project.dir = str(workspace_dir)
            project.updated_at = time()
            self._project_service.save_project(project, project_dir=workspace_dir)
            return project

        if not str(getattr(project, "job_id", "") or "").strip() and previous_project_id:
            project.job_id = previous_project_id

        project.project_id = normalized_workspace_name
        self._normalize_project_metadata(project=project, project_id=normalized_workspace_name)
        project.dir = str(workspace_dir)
        project.updated_at = time()
        self._project_service.save_project(project, project_dir=workspace_dir)
        logger.info(
            "workspace 身份重绑定: workspace=%s old_project_id=%s new_project_id=%s",
            normalized_workspace_name,
            previous_project_id,
            normalized_workspace_name,
        )
        return project

    def _build_identity_if_possible(
        self,
        project,
        input_identifier: str,
    ) -> Optional[ProjectIdentity]:
        project_dir = Path(project.dir) if str(project.dir or "").strip() else None
        if project_dir is None or not project_dir.exists():
            resolved_dir = self._project_service.get_project_dir(project.project_id)
            if resolved_dir and resolved_dir.exists():
                project_dir = resolved_dir

        if project_dir is None or not project_dir.exists():
            return None

        return ProjectIdentity(
            project_id=project.project_id,
            project_dir=project_dir,
            input_identifier=input_identifier,
            legacy_job_id=self._resolve_legacy_job_id(project_dir, project.job_id),
        )

    @staticmethod
    def _resolve_legacy_job_id(project_dir: Path, meta_job_id: Optional[str]) -> Optional[str]:
        normalized_meta_job_id = str(meta_job_id or "").strip()
        if normalized_meta_job_id:
            return normalized_meta_job_id
        dir_name = str(project_dir.name or "").strip()
        return dir_name or None

    @staticmethod
    def _normalize_project_metadata(*, project, project_id: str) -> None:
        """
        统一修正 project 内嵌元信息，避免重绑后出现 project_id 语义不一致。
        """
        subtitle_doc = getattr(project, "subtitle_doc", None)
        source_type = str(getattr(subtitle_doc, "source_type", "") or "").strip().lower()
        project_mode = str(getattr(project, "mode", "") or "").strip().lower()

        if subtitle_doc is not None:
            subtitle_doc.project_id = project_id
            doc_id = str(getattr(subtitle_doc, "doc_id", "") or "").strip()
            if not doc_id or doc_id == str(getattr(project, "job_id", "") or "").strip():
                subtitle_doc.doc_id = project_id
            subtitle_doc.updated_at = time()

        if source_type == "import" and project_mode != "legacy":
            project.mode = "import"


_project_id_resolver: Optional[ProjectIdResolver] = None
_project_id_resolver_lock = RLock()


def get_project_id_resolver() -> ProjectIdResolver:
    """获取 ProjectIdResolver 单例。"""
    global _project_id_resolver
    if _project_id_resolver is not None:
        return _project_id_resolver
    with _project_id_resolver_lock:
        if _project_id_resolver is None:
            _project_id_resolver = ProjectIdResolver()
    return _project_id_resolver
