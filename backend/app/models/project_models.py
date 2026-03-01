"""
Task6 领域模型定义。

设计说明：
- 本文件仅包含数据模型与 ID 生成函数，不承载业务逻辑。
- 采用“贫血模型”模式，便于与现有 JobState 风格保持一致。
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from time import time
from typing import List, Literal, Optional
import uuid

ProjectMode = Literal["normal", "legacy", "import"]
FlavorType = Literal["full", "lite"]
SourceType = Literal["transcribe", "import", "legacy"]
TaskMode = Literal["transcribe", "subtitle_edit"]
AssetType = Literal[
    "video",
    "audio",
    "peaks",
    "proxy_720p",
    "preview_360p",
    "thumbnail",
]
_WORKSPACE_MODE_PATTERN = re.compile(
    r"^p-\d{8}-\d{6}-(tr|im|lg)-[a-z0-9-]+-[0-9a-z]{4}$"
)


def generate_project_id() -> str:
    """生成 project_id。"""
    return f"proj_{uuid.uuid4().hex[:12]}"


def generate_legacy_project_id(job_id: str) -> str:
    """基于 job_id 派生 legacy project_id，保证幂等。"""
    return f"proj_leg_{job_id[:12]}"


def generate_segment_id() -> str:
    """生成 segment_id。"""
    return f"seg_{uuid.uuid4().hex[:16]}"


def normalize_task_mode(raw_value: object) -> Optional[TaskMode]:
    """标准化任务模式，无法识别时返回 None。"""
    normalized_value = str(raw_value or "").strip().lower()
    if normalized_value in {"transcribe", "subtitle_edit"}:
        return normalized_value  # type: ignore[return-value]
    return None


def _infer_task_mode_from_workspace_name(workspace_name: str) -> Optional[TaskMode]:
    """
    根据 canonical workspace 名称推断任务模式。

    约定：
    - `...-tr-...` / `...-lg-...` => `transcribe`
    - `...-im-...` => `subtitle_edit`
    """
    normalized_name = str(workspace_name or "").strip().lower()
    if not normalized_name:
        return None
    match = _WORKSPACE_MODE_PATTERN.fullmatch(normalized_name)
    if match is None:
        return None
    mode_code = match.group(1)
    if mode_code == "im":
        return "subtitle_edit"
    return "transcribe"


def infer_task_mode(
    *,
    raw_task_mode: object = None,
    project_mode: object = None,
    subtitle_source_type: object = None,
    project_dir: str = "",
) -> TaskMode:
    """
    推断任务模式（转录 / 字幕编辑）。

    兼容规则：
    1. 显式 `task_mode` 优先
    2. canonical workspace 命名次之（`-tr-/-im-/-lg-`）
    3. 目录运行态产物优先判定转录任务
    4. 历史 `mode/source_type` 再次之
    5. 最后按目录产物兜底
    6. 无法判定时默认 `transcribe`
    """
    normalized_task_mode = normalize_task_mode(raw_task_mode)
    if normalized_task_mode is not None:
        return normalized_task_mode

    workspace_dir = Path(project_dir) if str(project_dir or "").strip() else None
    workspace_mode_hint = (
        _infer_task_mode_from_workspace_name(workspace_dir.name)
        if workspace_dir is not None
        else None
    )
    if workspace_mode_hint is not None:
        return workspace_mode_hint

    has_runtime_artifact = False
    has_subtitle_edit_artifact = False
    if workspace_dir is not None and workspace_dir.exists():
        has_runtime_artifact = any(
            (workspace_dir / file_name).exists()
            for file_name in ("job_meta.json", "checkpoint.json", "runtime_state.db")
        )
        if has_runtime_artifact:
            return "transcribe"
        has_subtitle_edit_artifact = (workspace_dir / "subtitle_edits.json").exists()

    normalized_project_mode = str(project_mode or "").strip().lower()
    normalized_source_type = str(subtitle_source_type or "").strip().lower()
    if normalized_project_mode == "legacy" or normalized_source_type == "legacy":
        return "transcribe"
    if normalized_project_mode == "import" or normalized_source_type == "import":
        return "subtitle_edit"

    if has_subtitle_edit_artifact:
        return "subtitle_edit"

    return "transcribe"


def derive_compat_project_mode(
    *,
    task_mode: TaskMode,
    existing_mode: object = "normal",
) -> ProjectMode:
    """
    为兼容历史字段 `mode` 生成稳定值。
    """
    normalized_existing_mode = str(existing_mode or "").strip().lower()
    if normalized_existing_mode == "legacy":
        return "legacy"
    if task_mode == "subtitle_edit":
        return "import"
    return "normal"


@dataclass
class MediaAssetRef:
    """媒体资产引用。"""

    asset_type: AssetType
    path: str
    is_available: bool = False
    size_bytes: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "type": self.asset_type,
            "path": self.path,
            "exists": self.is_available,
            "size_bytes": self.size_bytes,
        }

    @staticmethod
    def from_dict(data: dict) -> "MediaAssetRef":
        return MediaAssetRef(
            asset_type=data.get("type", "video"),
            path=data.get("path", ""),
            is_available=bool(data.get("exists", False)),
            size_bytes=data.get("size_bytes"),
        )


@dataclass
class SubtitleDocMeta:
    """字幕文档元信息。"""

    doc_id: str
    project_id: str
    source_type: SourceType
    segment_count: int = 0
    version: int = 1
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "project_id": self.project_id,
            "source_type": self.source_type,
            "segment_count": int(self.segment_count),
            "version": int(self.version),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
        }

    @staticmethod
    def from_dict(data: dict) -> "SubtitleDocMeta":
        return SubtitleDocMeta(
            doc_id=data.get("doc_id", ""),
            project_id=data.get("project_id", ""),
            source_type=data.get("source_type", "transcribe"),
            segment_count=int(data.get("segment_count", 0)),
            version=int(data.get("version", 1)),
            created_at=float(data.get("created_at", time())),
            updated_at=float(data.get("updated_at", time())),
        )


@dataclass
class Project:
    """Project 聚合根数据模型。"""

    project_id: str
    title: str
    mode: ProjectMode = "normal"
    task_mode: TaskMode = "transcribe"
    flavor: FlavorType = "full"

    job_id: Optional[str] = None
    subtitle_doc: Optional[SubtitleDocMeta] = None
    media_assets: List[MediaAssetRef] = field(default_factory=list)
    capability_snapshot: List[str] = field(
        default_factory=lambda: ["subtitle.import_export"]
    )

    dir: str = ""
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "title": self.title,
            "mode": self.mode,
            "task_mode": self.task_mode,
            "flavor": self.flavor,
            "job_id": self.job_id,
            "subtitle_doc": self.subtitle_doc.to_dict() if self.subtitle_doc else None,
            "media_assets": [asset.to_dict() for asset in self.media_assets],
            "capability_snapshot": list(self.capability_snapshot),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
        }

    @staticmethod
    def from_dict(data: dict, project_dir: str = "") -> "Project":
        subtitle_doc_data = data.get("subtitle_doc")
        inferred_task_mode = infer_task_mode(
            raw_task_mode=data.get("task_mode"),
            project_mode=data.get("mode"),
            subtitle_source_type=(
                subtitle_doc_data.get("source_type")
                if isinstance(subtitle_doc_data, dict)
                else None
            ),
            project_dir=project_dir,
        )
        default_source_type: SourceType = (
            "import" if inferred_task_mode == "subtitle_edit" else "transcribe"
        )
        subtitle_doc = None
        if isinstance(subtitle_doc_data, dict):
            subtitle_doc = SubtitleDocMeta.from_dict(
                {
                    "doc_id": subtitle_doc_data.get("doc_id", data.get("project_id", "")),
                    "project_id": data.get("project_id", ""),
                    "source_type": subtitle_doc_data.get("source_type", default_source_type),
                    "segment_count": subtitle_doc_data.get("segment_count", 0),
                    "version": subtitle_doc_data.get("version", 1),
                    "created_at": subtitle_doc_data.get("created_at", time()),
                    "updated_at": subtitle_doc_data.get("updated_at", time()),
                }
            )

        raw_media_assets = data.get("media_assets", [])
        media_assets: List[MediaAssetRef] = []
        if isinstance(raw_media_assets, list):
            for raw_asset in raw_media_assets:
                if isinstance(raw_asset, dict):
                    media_assets.append(MediaAssetRef.from_dict(raw_asset))

        raw_capability_snapshot = data.get("capability_snapshot")
        capability_snapshot = (
            list(raw_capability_snapshot)
            if isinstance(raw_capability_snapshot, list)
            else ["subtitle.import_export"]
        )

        normalized_mode = derive_compat_project_mode(
            task_mode=inferred_task_mode,
            existing_mode=data.get("mode", "normal"),
        )

        return Project(
            project_id=data.get("project_id", ""),
            title=data.get("title", ""),
            mode=normalized_mode,
            task_mode=inferred_task_mode,
            flavor=data.get("flavor", "full"),
            job_id=data.get("job_id"),
            subtitle_doc=subtitle_doc,
            media_assets=media_assets,
            capability_snapshot=capability_snapshot,
            dir=project_dir,
            created_at=float(data.get("created_at", time())),
            updated_at=float(data.get("updated_at", time())),
        )
