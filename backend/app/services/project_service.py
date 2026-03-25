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
    TaskMode,
    derive_compat_project_mode,
    generate_project_id,
    infer_task_mode,
    normalize_task_mode,
)
from app.services.project_naming_service import get_project_naming_service
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
        self._project_naming_service = get_project_naming_service()

    def create_import_project(
        self,
        title: str,
        subtitle_segments: List[dict],
        video_path: Optional[str] = None,
        flavor: str = FLAVOR,
    ) -> Project:
        """从外部字幕导入创建项目。"""
        from app.services.subtitle_doc_service import get_subtitle_doc_service

        workspace_dir_name = self._project_naming_service.generate_workspace_dir_name(
            jobs_root=config.JOBS_DIR,
            mode="import",
            title=title,
            source_filename=video_path or title,
        )
        # 仅编辑项目从创建起即使用 canonical workspace 标识，避免重启后二次漂移。
        project_id = workspace_dir_name
        project_dir = config.JOBS_DIR / workspace_dir_name
        project_dir.mkdir(parents=True, exist_ok=True)

        # V3.2.4+dev.20260228.01: 用有意义的默认名称替代 UUID 回退
        normalized_title = str(title or "").strip() or "导入字幕"
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
        subtitle_doc.doc_id = project_id
        subtitle_doc.project_id = project_id
        subtitle_doc.updated_at = time()
        media_assets = self.scan_media_assets(project_dir)

        project = Project(
            project_id=project_id,
            title=normalized_title,
            mode="import",
            task_mode="subtitle_edit",
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
        task_mode: TaskMode = "transcribe",
        source_type: SourceType = "transcribe",
        project_id: Optional[str] = None,
        project_dir: Optional[Path] = None,
    ) -> Project:
        """
        从现有任务目录创建项目元信息。

        说明：
        - 优先复用可用目录（`jobs/{job_id}` / `jobs/{project_id}` / 目录缓存反查）；
        - `project_id` 可与目录名不同，读取时通过 project_meta 反查。
        """
        target_project_id = project_id or generate_project_id()
        candidate_dirs: List[Path] = []
        explicit_project_dir = Path(project_dir) if project_dir is not None else None
        if explicit_project_dir is not None:
            candidate_dirs.append(explicit_project_dir)
        normalized_job_id = str(job_id or "").strip()
        if normalized_job_id:
            candidate_dirs.append(config.JOBS_DIR / normalized_job_id)
        candidate_dirs.append(config.JOBS_DIR / target_project_id)
        resolved_dir = self.get_project_dir(target_project_id)
        if resolved_dir is not None:
            candidate_dirs.append(resolved_dir)

        job_dir: Optional[Path] = None
        for candidate in candidate_dirs:
            if candidate.exists():
                job_dir = candidate
                break

        if job_dir is None:
            raise FileNotFoundError(
                f"任务目录不存在: job_id={normalized_job_id}, project_id={target_project_id}"
            )

        normalized_title = str(title or "").strip() or target_project_id
        normalized_flavor = "lite" if str(flavor).lower() == "lite" else "full"
        normalized_task_mode = infer_task_mode(
            raw_task_mode=task_mode,
            project_mode=mode,
            subtitle_source_type=source_type,
            project_dir=str(job_dir),
        )
        normalized_mode = derive_compat_project_mode(
            task_mode=normalized_task_mode,
            existing_mode=mode,
        )
        normalized_source_type: SourceType = (
            "import" if normalized_task_mode == "subtitle_edit" else source_type
        )
        segment_count = self._estimate_segment_count(job_dir)
        subtitle_doc = SubtitleDocMeta(
            doc_id=target_project_id,
            project_id=target_project_id,
            source_type=normalized_source_type,
            segment_count=segment_count,
            version=1,
            created_at=time(),
            updated_at=time(),
        )

        project = Project(
            project_id=target_project_id,
            title=normalized_title,
            mode=normalized_mode,
            task_mode=normalized_task_mode,
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
                cached_dir = Path(cached.dir) if str(cached.dir or "").strip() else None
                if cached_dir and cached_dir.exists():
                    return cached
                # 缓存目录失效时必须回退到磁盘扫描，避免 project_id 命中脏缓存后误判 404。
                self._projects_cache.pop(normalized_project_id, None)
                self._project_dir_cache.pop(normalized_project_id, None)

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

    def find_project_by_alias(self, identifier: str) -> Optional[Project]:
        """
        按兼容别名查找项目。

        支持别名：
        - `project.job_id`
        - `project.subtitle_doc.project_id`
        - `project.subtitle_doc.doc_id`
        """
        normalized_identifier = str(identifier or "").strip()
        if not normalized_identifier:
            return None

        project = self.get_project(normalized_identifier)
        if project is not None:
            return project

        if not config.JOBS_DIR.exists():
            return None

        for item in config.JOBS_DIR.iterdir():
            if not item.is_dir():
                continue
            meta_path = item / PROJECT_META_FILENAME
            if not meta_path.exists():
                continue

            project = self._load_project_meta(item)
            if project is None:
                continue
            if not self._matches_alias(project, normalized_identifier):
                continue
            self._cache_project(project, item)
            return project

        return None

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
        self._normalize_project_metadata(project=project, project_dir=target_dir)
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

        # 优先使用标准抽取产物 audio.wav；若不存在，则回退到源音频文件（mp3/m4a/...）。
        audio_file = project_dir / "audio.wav"
        if audio_file.exists():
            assets.append(self._build_asset("audio", audio_file, project_dir))
        else:
            source_audio = self._find_audio_file(project_dir)
            if source_audio is not None:
                assets.append(self._build_asset("audio", source_audio, project_dir))

        named_assets = {
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
            has_explicit_task_mode = (
                isinstance(payload, dict)
                and normalize_task_mode(payload.get("task_mode")) is not None
            )
            project = Project.from_dict(payload, project_dir=str(project_dir))
            if self._normalize_project_metadata(
                project=project,
                project_dir=project_dir,
                force_task_mode_write=not has_explicit_task_mode,
            ):
                # 启动或读取时自动补齐 task_mode 与兼容字段，确保元数据始终完整一致。
                self._save_project_meta(project, project_dir)
            return project
        except Exception as exc:
            logger.warning("读取 project_meta 失败: %s (%s)", project_dir, exc)
            return None

    def _normalize_project_metadata(
        self,
        *,
        project: Project,
        project_dir: Path,
        force_task_mode_write: bool = False,
    ) -> bool:
        """
        统一补齐并校正任务模式元数据。
        """
        has_changed = False
        subtitle_doc = getattr(project, "subtitle_doc", None)
        raw_source_type = getattr(subtitle_doc, "source_type", "") if subtitle_doc else ""
        inferred_task_mode = infer_task_mode(
            raw_task_mode=getattr(project, "task_mode", None),
            project_mode=getattr(project, "mode", None),
            subtitle_source_type=raw_source_type,
            project_dir=str(project_dir),
        )
        current_task_mode = normalize_task_mode(getattr(project, "task_mode", None))
        if current_task_mode != inferred_task_mode:
            project.task_mode = inferred_task_mode
            has_changed = True
        elif force_task_mode_write:
            # `Project.from_dict` 可能已经推断出 task_mode，但原始 JSON 字段缺失；
            # 启动/读取时需要回写，保证后续链路只依赖显式元数据。
            project.task_mode = inferred_task_mode
            has_changed = True

        normalized_mode = derive_compat_project_mode(
            task_mode=inferred_task_mode,
            existing_mode=getattr(project, "mode", "normal"),
        )
        if getattr(project, "mode", None) != normalized_mode:
            project.mode = normalized_mode
            has_changed = True

        if subtitle_doc is not None:
            expected_source_type: SourceType = (
                "import" if inferred_task_mode == "subtitle_edit" else "transcribe"
            )
            normalized_source_type = str(getattr(subtitle_doc, "source_type", "") or "").strip().lower()
            if normalized_source_type not in {"import", "legacy", "transcribe"}:
                subtitle_doc.source_type = expected_source_type
                has_changed = True
            elif normalized_source_type == "import" and inferred_task_mode == "transcribe":
                subtitle_doc.source_type = expected_source_type
                has_changed = True
            elif normalized_source_type == "transcribe" and inferred_task_mode == "subtitle_edit":
                subtitle_doc.source_type = expected_source_type
                has_changed = True

            if str(getattr(subtitle_doc, "project_id", "") or "").strip() != project.project_id:
                subtitle_doc.project_id = project.project_id
                has_changed = True

        return has_changed

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

    @staticmethod
    def _matches_alias(project: Project, identifier: str) -> bool:
        normalized_identifier = str(identifier or "").strip()
        if not normalized_identifier:
            return False

        if str(project.job_id or "").strip() == normalized_identifier:
            return True

        subtitle_doc = project.subtitle_doc
        if subtitle_doc is None:
            return False
        if str(subtitle_doc.project_id or "").strip() == normalized_identifier:
            return True
        if str(subtitle_doc.doc_id or "").strip() == normalized_identifier:
            return True
        return False

    def _estimate_segment_count(self, project_dir: Path) -> int:
        """
        粗略统计字幕数量（只读）。

        说明：仅用于 `project_meta.subtitle_doc.segment_count`，
        不触发 `_segment_map` 写入。
        """
        try:
            from app.services.checkpoint import RuntimeCheckpointService

            runtime_service = RuntimeCheckpointService(job_dir=project_dir)
            runtime_payload = runtime_service.load_subtitle_runtime()
            if runtime_payload:
                raw_items = runtime_payload.get("subtitle_items_snapshot", [])
                if isinstance(raw_items, list) and raw_items:
                    return len([item for item in raw_items if isinstance(item, dict)])
                raw_snapshot = runtime_payload.get("sentences_snapshot", [])
                if isinstance(raw_snapshot, list):
                    return len([item for item in raw_snapshot if isinstance(item, dict)])
        except Exception:
            pass
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
        source_candidates = []
        generated_candidates = []
        for item in sorted(project_dir.iterdir(), key=lambda candidate: candidate.name.lower()):
            if not item.is_file() or item.suffix.lower() not in video_exts:
                continue
            normalized_name = item.name.lower()
            if normalized_name.endswith(".tmp"):
                continue
            if normalized_name.startswith(("preview_", "proxy_")) or normalized_name == "remux.mp4":
                generated_candidates.append(item)
                continue
            source_candidates.append(item)
        if source_candidates:
            return source_candidates[0]
        if generated_candidates:
            return generated_candidates[0]
        return None

    @staticmethod
    def _find_audio_file(project_dir: Path) -> Optional[Path]:
        audio_exts = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
        for item in project_dir.iterdir():
            if item.is_file() and item.suffix.lower() in audio_exts:
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
