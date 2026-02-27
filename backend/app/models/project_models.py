"""
Task6 领域模型定义。

设计说明：
- 本文件仅包含数据模型与 ID 生成函数，不承载业务逻辑。
- 采用“贫血模型”模式，便于与现有 JobState 风格保持一致。
"""

from dataclasses import dataclass, field
from time import time
from typing import List, Literal, Optional
import uuid

ProjectMode = Literal["normal", "legacy"]
FlavorType = Literal["full", "lite"]
SourceType = Literal["transcribe", "import", "legacy"]
AssetType = Literal[
    "video",
    "audio",
    "peaks",
    "proxy_720p",
    "preview_360p",
    "thumbnail",
]


def generate_project_id() -> str:
    """生成 project_id。"""
    return f"proj_{uuid.uuid4().hex[:12]}"


def generate_legacy_project_id(job_id: str) -> str:
    """基于 job_id 派生 legacy project_id，保证幂等。"""
    return f"proj_leg_{job_id[:12]}"


def generate_segment_id() -> str:
    """生成 segment_id。"""
    return f"seg_{uuid.uuid4().hex[:16]}"


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
            source_type=data.get("source_type", "import"),
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
        subtitle_doc = None
        if isinstance(subtitle_doc_data, dict):
            subtitle_doc = SubtitleDocMeta.from_dict(
                {
                    "doc_id": subtitle_doc_data.get("doc_id", data.get("project_id", "")),
                    "project_id": data.get("project_id", ""),
                    "source_type": subtitle_doc_data.get("source_type", "import"),
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

        return Project(
            project_id=data.get("project_id", ""),
            title=data.get("title", ""),
            mode=data.get("mode", "normal"),
            flavor=data.get("flavor", "full"),
            job_id=data.get("job_id"),
            subtitle_doc=subtitle_doc,
            media_assets=media_assets,
            capability_snapshot=capability_snapshot,
            dir=project_dir,
            created_at=float(data.get("created_at", time())),
            updated_at=float(data.get("updated_at", time())),
        )
