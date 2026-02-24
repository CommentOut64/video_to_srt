"""
Project 领域服务。

设计模式：
- Service Layer（服务层模式）：集中承载 Project 聚合的业务操作。
- Repository-like（轻量仓储）：复用文件系统存储，不引入新数据库。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from threading import RLock
from time import time
from typing import Dict, List, Optional

from app.core.config import FLAVOR, config
from app.models.project_models import (
    MediaAssetRef,
    Project,
    ProjectMode,
    SourceType,
    SubtitleDocMeta,
    generate_project_id,
)
from app.services.subtitle_edit_store import load_deleted_indices, load_edits

logger = logging.getLogger(__name__)
PROJECT_META_FILENAME = "project_meta.json"


def _atomic_write_json(path: Path, payload: dict) -> None:
    """原子写入 JSON，避免中断导致文件损坏。"""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


class ProjectService:
    """项目聚合根服务。"""

    def __init__(self) -> None:
        self._projects_cache: Dict[str, Project] = {}
        self._project_dir_cache: Dict[str, Path] = {}
        self._lock = RLock()

    def create_import_project(
        self,
        title: str,
        subtitle_segments: List[dict],
        video_path: Optional[str] = None,
        flavor: str = FLAVOR,
    ) -> Project:
        """从外部字幕导入创建项目。"""
        from app.services.subtitle_doc_service import get_subtitle_doc_service

        project_id = generate_project_id()
        project_dir = config.JOBS_DIR / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        normalized_title = str(title or "").strip() or project_id
        normalized_flavor = "lite" if str(flavor).lower() == "lite" else "full"

        try:
            if video_path:
                self._attach_video_asset(project_dir, video_path)
        except Exception as exc:
            logger.warning("导入项目拷贝视频失败，将继续创建无视频项目: %s", exc)

        subtitle_doc_service = get_subtitle_doc_service()
        subtitle_doc = subtitle_doc_service.import_segments(
            project_dir=project_dir,
            segments=subtitle_segments,
            source_type="import",
        )
        media_assets = self.scan_media_assets(project_dir)

        project = Project(
            project_id=project_id,
            title=normalized_title,
            mode="normal",
            flavor=normalized_flavor,
            subtitle_doc=subtitle_doc,
            media_assets=media_assets,
            dir=str(project_dir),
            created_at=time(),
            updated_at=time(),
        )
        self.save_project(project, project_dir=project_dir)
        return project

    def create_normal_project(
        self,
        job_id: str,
        title: str,
        *,
        flavor: str = FLAVOR,
        mode: ProjectMode = "normal",
        source_type: SourceType = "transcribe",
        project_id: Optional[str] = None,
    ) -> Project:
        """
        从现有任务目录创建项目元信息。

        说明：
        - 目录仍复用 `jobs/{job_id}`；
        - `project_id` 可与目录名不同，读取时通过 project_meta 反查。
        """
        job_dir = config.JOBS_DIR / job_id
        if not job_dir.exists():
            raise FileNotFoundError(f"任务目录不存在: {job_dir}")

        target_project_id = project_id or generate_project_id()
        normalized_title = str(title or "").strip() or job_id
        normalized_flavor = "lite" if str(flavor).lower() == "lite" else "full"
        segment_count = self._estimate_segment_count(job_dir)
        subtitle_doc = SubtitleDocMeta(
            doc_id=target_project_id,
            project_id=target_project_id,
            source_type=source_type,
            segment_count=segment_count,
            version=1,
            created_at=time(),
            updated_at=time(),
        )

        project = Project(
            project_id=target_project_id,
            title=normalized_title,
            mode=mode,
            flavor=normalized_flavor,
            job_id=job_id,
            subtitle_doc=subtitle_doc,
            media_assets=self.scan_media_assets(job_dir),
            dir=str(job_dir),
            created_at=time(),
            updated_at=time(),
        )
        self.save_project(project, project_dir=job_dir)
        return project

    def get_project(self, project_id: str) -> Optional[Project]:
        """获取项目。"""
        normalized_project_id = str(project_id or "").strip()
        if not normalized_project_id:
            return None

        with self._lock:
            cached = self._projects_cache.get(normalized_project_id)
            if cached is not None:
                return cached

        direct_dir = config.JOBS_DIR / normalized_project_id
        project = self._load_project_meta(direct_dir)
        if project is not None and project.project_id == normalized_project_id:
            self._cache_project(project, direct_dir)
            return project

        scanned_dir = self._find_project_dir_by_project_id(normalized_project_id)
        if scanned_dir is None:
            return None

        project = self._load_project_meta(scanned_dir)
        if project is None:
            return None
        self._cache_project(project, scanned_dir)
        return project

    def list_projects(self, flavor: Optional[str] = None) -> List[Project]:
        """列出所有项目。"""
        target_flavor = str(flavor).lower().strip() if flavor else ""
        projects: List[Project] = []
        if not config.JOBS_DIR.exists():
            return projects

        for item in config.JOBS_DIR.iterdir():
            if not item.is_dir():
                continue
            meta_path = item / PROJECT_META_FILENAME
            if not meta_path.exists():
                continue
            project = self._load_project_meta(item)
            if project is None:
                continue
            if target_flavor and project.flavor != target_flavor:
                continue
            projects.append(project)
            self._cache_project(project, item)

        projects.sort(key=lambda p: p.updated_at, reverse=True)
        return projects

    def update_title(self, project_id: str, title: str) -> bool:
        """更新项目标题。"""
        project = self.get_project(project_id)
        if project is None:
            return False
        project.title = str(title or "").strip() or project.title
        project.updated_at = time()
        self.save_project(project)
        return True

    def refresh_media_assets(self, project_id: str) -> List[MediaAssetRef]:
        """重新扫描媒体资源。"""
        project = self.get_project(project_id)
        if project is None:
            return []
        project_dir = self.get_project_dir(project.project_id)
        if project_dir is None:
            return []
        project.media_assets = self.scan_media_assets(project_dir)
        project.updated_at = time()
        self.save_project(project, project_dir=project_dir)
        return project.media_assets

    def get_project_dir(self, project_id: str) -> Optional[Path]:
        """根据 project_id 获取目录。"""
        normalized_project_id = str(project_id or "").strip()
        if not normalized_project_id:
            return None

        with self._lock:
            cached_dir = self._project_dir_cache.get(normalized_project_id)
            if cached_dir and cached_dir.exists():
                return cached_dir

        direct_dir = config.JOBS_DIR / normalized_project_id
        if (direct_dir / PROJECT_META_FILENAME).exists():
            with self._lock:
                self._project_dir_cache[normalized_project_id] = direct_dir
            return direct_dir

        scanned_dir = self._find_project_dir_by_project_id(normalized_project_id)
        if scanned_dir:
            with self._lock:
                self._project_dir_cache[normalized_project_id] = scanned_dir
        return scanned_dir

    def save_project(self, project: Project, project_dir: Optional[Path] = None) -> None:
        """持久化项目元信息。"""
        target_dir = project_dir or self.get_project_dir(project.project_id)
        if target_dir is None:
            if project.dir:
                target_dir = Path(project.dir)
            else:
                target_dir = config.JOBS_DIR / project.project_id
        target_dir.mkdir(parents=True, exist_ok=True)

        project.dir = str(target_dir)
        project.updated_at = time()
        self._save_project_meta(project, target_dir)
        self._cache_project(project, target_dir)

    def scan_media_assets(self, project_dir: Path) -> List[MediaAssetRef]:
        """扫描项目目录中的媒体资产。"""
        if not project_dir.exists():
            return []

        assets: List[MediaAssetRef] = []
        video_file = self._find_video_file(project_dir)
        if video_file is not None:
            assets.append(self._build_asset("video", video_file, project_dir))

        named_assets = {
            "audio": project_dir / "audio.wav",
            "peaks": project_dir / "peaks.json",
            "proxy_720p": project_dir / "proxy_720p.mp4",
            "preview_360p": project_dir / "preview_360p.mp4",
            "thumbnail": project_dir / "thumbnail_sprite.jpg",
        }
        for asset_type, path in named_assets.items():
            if path.exists():
                assets.append(self._build_asset(asset_type, path, project_dir))

        return assets

    def _save_project_meta(self, project: Project, project_dir: Path) -> None:
        meta_path = project_dir / PROJECT_META_FILENAME
        payload = project.to_dict()
        _atomic_write_json(meta_path, payload)

    def _load_project_meta(self, project_dir: Path) -> Optional[Project]:
        meta_path = project_dir / PROJECT_META_FILENAME
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            return Project.from_dict(payload, project_dir=str(project_dir))
        except Exception as exc:
            logger.warning("读取 project_meta 失败: %s (%s)", project_dir, exc)
            return None

    def _cache_project(self, project: Project, project_dir: Path) -> None:
        with self._lock:
            self._projects_cache[project.project_id] = project
            self._project_dir_cache[project.project_id] = project_dir

    def _find_project_dir_by_project_id(self, project_id: str) -> Optional[Path]:
        if not config.JOBS_DIR.exists():
            return None
        for item in config.JOBS_DIR.iterdir():
            if not item.is_dir():
                continue
            meta_path = item / PROJECT_META_FILENAME
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as file:
                    payload = json.load(file)
                if payload.get("project_id") == project_id:
                    return item
            except Exception:
                continue
        return None

    def _estimate_segment_count(self, project_dir: Path) -> int:
        """
        粗略统计字幕数量（只读）。

        说明：仅用于 `project_meta.subtitle_doc.segment_count`，
        不触发 `_segment_map` 写入。
        """
        try:
            edits = load_edits(project_dir)
            deleted = load_deleted_indices(project_dir)
            return len([idx for idx in edits.keys() if idx not in deleted])
        except Exception:
            return 0

    def _attach_video_asset(self, project_dir: Path, video_path: str) -> None:
        """将外部视频附加到项目目录（优先硬链接，失败回退复制）。"""
        source_path = Path(video_path)
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"视频文件不存在: {source_path}")

        target_path = project_dir / source_path.name
        if target_path.exists():
            return

        try:
            os.link(source_path, target_path)
            return
        except Exception:
            pass

        shutil.copy2(source_path, target_path)

    @staticmethod
    def _find_video_file(project_dir: Path) -> Optional[Path]:
        video_exts = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".flv", ".m4v"}
        for item in project_dir.iterdir():
            if item.is_file() and item.suffix.lower() in video_exts:
                return item
        return None

    @staticmethod
    def _build_asset(asset_type: str, asset_path: Path, project_dir: Path) -> MediaAssetRef:
        try:
            relative_path = str(asset_path.relative_to(project_dir))
        except Exception:
            relative_path = asset_path.name

        size_bytes = None
        try:
            size_bytes = int(asset_path.stat().st_size)
        except Exception:
            size_bytes = None

        return MediaAssetRef(
            asset_type=asset_type,  # type: ignore[arg-type]
            path=relative_path,
            is_available=asset_path.exists(),
            size_bytes=size_bytes,
        )


_project_service: Optional[ProjectService] = None
_project_service_lock = RLock()


def get_project_service() -> ProjectService:
    """获取 ProjectService 单例。"""
    global _project_service
    if _project_service is not None:
        return _project_service
    with _project_service_lock:
        if _project_service is None:
            _project_service = ProjectService()
    return _project_service
