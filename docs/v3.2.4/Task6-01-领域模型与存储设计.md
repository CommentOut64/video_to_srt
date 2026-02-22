# Task 6-01: 领域模型与存储设计

> Type: Architecture | Status: Active
> Version: V3.2.4+dev.20260222.01
> 上游文档: `Task6-00-总纲-转录与编辑解耦可执行方案.md`

## 1. 设计原则

1. **新模型为语义层，存储复用**: Project/SubtitleDoc 是内存中的领域对象，底层存储仍使用 `{JOBS_DIR}/{id}/` 目录和 `subtitle_edits.json` 文件
2. **project_id 为权威 ID**: 所有新 API、前端路由以 project_id 为主键，job_id 仅在兼容层出现
3. **segment_id 替代 sentence_index**: 新创建的字幕使用 UUID segment_id，旧字幕通过桥接层转换
4. **幂等映射**: legacy_projection_service 的 resolve 操作必须幂等（同一 job_id 始终映射到同一 project_id）

## 2. ID 体系

### 2.1 ID 生成规则

| ID | 格式 | 生成时机 | 唯一性保证 |
|---|---|---|---|
| `project_id` | `proj_{uuid_hex[:12]}` | 创建项目时 | UUID v4 |
| `job_id` | `{uuid_hex}` | 创建转录任务时（现有逻辑） | UUID v4 |
| `segment_id` | `seg_{uuid_hex[:16]}` | 创建/导入字幕段时 | UUID v4 |
| `legacy_project_id` | `proj_leg_{job_id[:12]}` | legacy 投影时 | 确定性派生，保证幂等 |

### 2.2 ID 映射关系

```
Project
  ├── project_id (主键)
  ├── job_id (可选，mode=normal/legacy 时有值)
  ├── subtitle_doc_id = project_id (1:1 关系，直接复用)
  └── media_assets[] (引用)

SubtitleDoc
  ├── doc_id = project_id
  ├── segments[]
  │   ├── segment_id (新主键)
  │   └── legacy_index (旧 sentence_index，仅 legacy 模式有值)
  └── source_type: transcribe | import | legacy
```

## 3. 领域模型定义

### 3.1 Project 模型

文件: `backend/app/models/project_models.py`

```python
from dataclasses import dataclass, field
from typing import Optional, List, Literal
from time import time
import uuid

# --- 枚举与常量 ---
ProjectMode = Literal["normal", "legacy"]
Flavor = Literal["full", "lite"]
SourceType = Literal["transcribe", "import", "legacy"]


def generate_project_id() -> str:
    return f"proj_{uuid.uuid4().hex[:12]}"


def generate_legacy_project_id(job_id: str) -> str:
    """确定性派生，保证幂等"""
    return f"proj_leg_{job_id[:12]}"


def generate_segment_id() -> str:
    return f"seg_{uuid.uuid4().hex[:16]}"


# --- 核心数据类 ---
@dataclass
class MediaAssetRef:
    """媒体资产引用"""
    asset_type: Literal["video", "audio", "peaks", "proxy_720p", "preview_360p", "thumbnail"]
    path: str  # 相对于项目目录的路径
    exists: bool = False
    size_bytes: Optional[int] = None


@dataclass
class SubtitleDocMeta:
    """字幕文档元信息（不包含具体字幕数据）"""
    doc_id: str
    project_id: str
    source_type: SourceType
    segment_count: int = 0
    version: int = 1
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)


@dataclass
class Project:
    """项目聚合根"""
    project_id: str
    title: str
    mode: ProjectMode = "normal"
    flavor: Flavor = "full"

    # 关联引用
    job_id: Optional[str] = None  # normal 模式自动关联，legacy 模式指向旧 job
    subtitle_doc: Optional[SubtitleDocMeta] = None
    media_assets: List[MediaAssetRef] = field(default_factory=list)

    # 能力快照（Task 6-Full 扩展）
    capability_snapshot: List[str] = field(default_factory=lambda: ["subtitle.import_export"])

    # 元数据
    dir: str = ""  # 项目目录绝对路径
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "title": self.title,
            "mode": self.mode,
            "flavor": self.flavor,
            "job_id": self.job_id,
            "subtitle_doc": {
                "doc_id": self.subtitle_doc.doc_id,
                "source_type": self.subtitle_doc.source_type,
                "segment_count": self.subtitle_doc.segment_count,
                "version": self.subtitle_doc.version,
            } if self.subtitle_doc else None,
            "media_assets": [
                {"type": a.asset_type, "path": a.path, "exists": a.exists}
                for a in self.media_assets
            ],
            "capability_snapshot": self.capability_snapshot,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: dict, project_dir: str = "") -> "Project":
        subtitle_doc = None
        if data.get("subtitle_doc"):
            sd = data["subtitle_doc"]
            subtitle_doc = SubtitleDocMeta(
                doc_id=sd["doc_id"],
                project_id=data["project_id"],
                source_type=sd.get("source_type", "import"),
                segment_count=sd.get("segment_count", 0),
                version=sd.get("version", 1),
            )
        return Project(
            project_id=data["project_id"],
            title=data.get("title", ""),
            mode=data.get("mode", "normal"),
            flavor=data.get("flavor", "full"),
            job_id=data.get("job_id"),
            subtitle_doc=subtitle_doc,
            media_assets=[
                MediaAssetRef(
                    asset_type=a["type"],
                    path=a["path"],
                    exists=a.get("exists", False),
                )
                for a in data.get("media_assets", [])
            ],
            capability_snapshot=data.get("capability_snapshot", ["subtitle.import_export"]),
            dir=project_dir,
            created_at=data.get("created_at", 0),
            updated_at=data.get("updated_at", 0),
        )
```

### 3.2 SubtitleSegment 扩展

字幕段不单独建模为新文件，而是在 `subtitle_doc_service.py` 中使用已有 `SentenceSegment` 并扩展：

```python
# 在 subtitle_doc_service.py 中扩展
@dataclass
class SubtitleSegment:
    """新字幕段，包装现有 SentenceSegment 并增加 segment_id"""
    segment_id: str
    text: str
    start: float
    end: float
    original_text: Optional[str] = None
    is_modified: bool = False
    is_deleted: bool = False
    legacy_index: Optional[int] = None  # 旧 sentence_index，仅兼容层使用
    source_type: SourceType = "import"
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)
```

## 4. 存储策略

### 4.1 目录结构复用

**关键决策**: 不创建新的存储目录体系，完全复用 `{JOBS_DIR}/{id}/` 结构。

```
{JOBS_DIR}/
  ├── {job_id}/                    # 现有转录任务目录（不变）
  │   ├── job_meta.json
  │   ├── subtitle_edits.json
  │   ├── audio.wav, peaks.json, *.mp4 ...
  │   └── checkpoint/ ...
  │
  ├── {project_id}/                # 新建的导入项目目录（同结构）
  │   ├── project_meta.json        # 项目元信息（替代 job_meta.json）
  │   ├── subtitle_edits.json      # 复用现有格式
  │   ├── audio.wav (可选)
  │   └── peaks.json (可选)
  │
  └── _legacy_map.json             # 全局映射索引
```

### 4.2 project_meta.json 格式

```json
{
    "project_id": "proj_a1b2c3d4e5f6",
    "title": "我的字幕项目",
    "mode": "normal",
    "flavor": "full",
    "job_id": null,
    "subtitle_doc": {
        "doc_id": "proj_a1b2c3d4e5f6",
        "source_type": "import",
        "segment_count": 42,
        "version": 1
    },
    "media_assets": [
        {"type": "video", "path": "source.mp4", "exists": true}
    ],
    "capability_snapshot": ["subtitle.import_export"],
    "created_at": 1708892345.123,
    "updated_at": 1708892400.456
}
```

### 4.3 subtitle_edits.json 完全复用

不改动 `subtitle_edit_store.py` 的文件格式。新增的 `segment_id` 通过内存映射维护：

```json
{
    "version": "3.2.4",
    "updated_at": 1708892345.123,
    "edits": {
        "0": {"text": "第一句", "start": 0.0, "end": 2.5},
        "1": {"text": "第二句", "start": 2.5, "end": 5.0}
    },
    "deleted_indices": [],
    "next_manual_index": -1,
    "_segment_map": {
        "seg_a1b2c3d4e5f6g7h8": 0,
        "seg_i9j0k1l2m3n4o5p6": 1
    }
}
```

**说明**: `_segment_map` 字段是新增的可选字段，`subtitle_edit_store.py` 的 `load_edits/save_edit` 函数会忽略它（不识别的字段自动保留）。`SubtitleDocService` 负责在加载时读取并维护此映射。

### 4.4 _legacy_map.json 格式

```json
{
    "version": 1,
    "mappings": {
        "abc123def456": {
            "project_id": "proj_leg_abc123def456",
            "migrated_at": 1708892345.123,
            "status": "migrated"
        }
    }
}
```

## 5. Flavor 体系

### 5.1 后端 flavor 配置

在 `backend/app/core/config.py` 中新增：

```python
# flavor 统一解析
_raw_flavor = os.environ.get("ANCHORFLUX_FLAVOR", "").lower()
_raw_lite = os.environ.get("ANCHORFLUX_LITE", "").lower()

if _raw_flavor in ("full", "lite"):
    FLAVOR: Literal["full", "lite"] = _raw_flavor
elif _raw_lite in ("true", "1", "yes"):
    FLAVOR = "lite"
else:
    FLAVOR = "full"

# 启动后固定，进程内不可变
IS_LITE: bool = (FLAVOR == "lite")
```

### 5.2 前端 flavor 配置

文件: `frontend/src/config/flavor.js`

```javascript
// 编译时注入，支持 tree-shaking
export const FLAVOR = import.meta.env.VITE_APP_FLAVOR || 'full'
export const IS_LITE = FLAVOR === 'lite' || import.meta.env.VITE_LITE_MODE === 'true'

// 能力门控（UI 层，后端为最终边界）
export const CAPABILITIES = {
    canTranscribe: !IS_LITE,
    canSeparateVocal: !IS_LITE,
    canSpectralTriage: !IS_LITE,
    canImportSubtitle: true,
    canExportSubtitle: true,
    canEditSubtitle: true,
    canPreviewMedia: true,
}
```

### 5.3 能力矩阵（后端强约束）

| capability | full | lite | 守卫位置 |
|---|---|---|---|
| `transcribe.*` | 允许 | 禁用(422) | `main.py` 不注册路由 |
| `audio.spectral_triage` | 允许 | 禁用(422) | `main.py` 不注册路由 |
| `audio.vocal_separation` | 允许 | 禁用(422) | `main.py` 不注册路由 |
| `media.prepare_preview` | 允许 | 允许 | 始终注册 |
| `subtitle.import_export` | 允许 | 允许 | 始终注册 |

## 6. 与 Task 6-Full 的演进路径

本方案设计时刻意保留了以下扩展点，确保 Task 6-Full 不需要推倒重来：

1. **Project.capability_snapshot**: 当前默认 `["subtitle.import_export"]`，Task 6-Full 扩展为完整能力列表
2. **SubtitleDocMeta.source_type**: 当前只有 `transcribe/import/legacy`，Task 6-Full 不需要修改
3. **存储层**: Task 6-Full 可引入 `projects.db` SQLite 作为索引层，目录结构不变
4. **Run 模型**: Task 6-Full 新增 `run_models.py` 和 `run_service.py`，与 Project 通过 `project_id` 关联
5. **SSE 通道**: 当前使用别名双发，Task 6-Full 可独立出 `run:{id}` 通道
