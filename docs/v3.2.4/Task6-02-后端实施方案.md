# Task 6-02: 后端实施方案

> Type: Architecture | Status: Active
> Version: V3.2.4+dev.20260222.01
> 上游文档: `Task6-00-总纲`, `Task6-01-领域模型与存储设计`

## 1. 实施顺序

```
Step 1: config.py 改造（flavor 基础设施）
Step 2: project_models.py（领域模型）
Step 3: project_service.py（项目服务）
Step 4: subtitle_doc_service.py（字幕文档服务）
Step 5: legacy_projection_service.py（旧任务投影）
Step 6: project_routes.py（项目 API）
Step 7: legacy_compat_routes.py（旧 API 桥接）
Step 8: main.py 改造（条件注册）
Step 9: transcription_routes.py 桥接
Step 10: streaming_subtitle.py 频道扩展
Step 11: media_routes.py 兼容
Step 12: job_models.py 扩展
```

## 2. Step 1: config.py 改造

**文件**: `backend/app/core/config.py`
**当前行数**: 489
**修改量**: ~30 行新增

### 改动位置

在 `ProjectConfig` 类或模块顶部常量区域新增 flavor 解析：

```python
# --- flavor 统一解析 ---
# 优先读取 ANCHORFLUX_FLAVOR，兼容旧变量 ANCHORFLUX_LITE
_raw_flavor = os.environ.get("ANCHORFLUX_FLAVOR", "").strip().lower()
_raw_lite = os.environ.get("ANCHORFLUX_LITE", "").strip().lower()

if _raw_flavor in ("full", "lite"):
    FLAVOR = _raw_flavor
elif _raw_lite in ("true", "1", "yes"):
    FLAVOR = "lite"
else:
    FLAVOR = "full"

IS_LITE = (FLAVOR == "lite")

# 启动时打印 flavor 确认
import logging
logging.getLogger(__name__).info(f"AnchorFlux flavor: {FLAVOR} (IS_LITE={IS_LITE})")
```

### 与现有代码的关系

- 不修改任何现有配置项
- 新增的 `FLAVOR` 和 `IS_LITE` 作为模块级常量导出
- 其他模块通过 `from backend.app.core.config import IS_LITE, FLAVOR` 使用

## 3. Step 2: project_models.py

**文件**: `backend/app/models/project_models.py`（新增）
**估算行数**: ~250

**内容**: 见 `Task6-01-领域模型与存储设计.md` 第 3.1 节完整代码

**关键设计决策**:
- 纯数据类，不包含业务逻辑
- `to_dict()`/`from_dict()` 遵循 JobState 的同名方法风格
- `generate_legacy_project_id()` 使用确定性派生保证幂等

## 4. Step 3: project_service.py

**文件**: `backend/app/services/project_service.py`（新增）
**估算行数**: ~400

### 核心接口

```python
from pathlib import Path
from typing import Optional, List
from backend.app.models.project_models import (
    Project, SubtitleDocMeta, MediaAssetRef,
    generate_project_id, generate_segment_id,
)
from backend.app.core.config import ProjectConfig as config


class ProjectService:
    """项目聚合根服务"""

    def __init__(self):
        self._projects_cache: dict[str, Project] = {}  # 内存缓存

    # --- 创建 ---

    def create_import_project(
        self,
        title: str,
        subtitle_segments: list[dict],
        video_path: Optional[str] = None,
        flavor: str = "full",
    ) -> Project:
        """从字幕文件导入创建项目

        参数:
            title: 项目标题
            subtitle_segments: 已解析的字幕列表 [{text, start, end}, ...]
            video_path: 可选视频路径
            flavor: full 或 lite

        返回:
            创建的 Project 对象

        核心流程:
            1. generate_project_id()
            2. 创建 {JOBS_DIR}/{project_id}/ 目录
            3. 调用 subtitle_doc_service.import_segments() 写入 subtitle_edits.json
            4. 如有视频，扫描媒体资产
            5. 写入 project_meta.json
            6. 返回 Project
        """

    def create_normal_project(self, job_id: str, title: str) -> Project:
        """从转录任务创建项目（Full 模式，转录完成后调用）

        核心流程:
            1. generate_project_id()
            2. project_dir = config.JOBS_DIR / job_id（复用已有任务目录）
            3. 读取现有 subtitle_edits.json
            4. 扫描现有媒体资产
            5. 写入 project_meta.json（与 job_meta.json 并存）
            6. 返回 Project(mode=normal, job_id=job_id)
        """

    # --- 查询 ---

    def get_project(self, project_id: str) -> Optional[Project]:
        """获取项目

        查找顺序:
            1. 内存缓存
            2. {JOBS_DIR}/{project_id}/project_meta.json
            3. 返回 None
        """

    def list_projects(self, flavor: Optional[str] = None) -> List[Project]:
        """列出所有项目

        扫描 JOBS_DIR 下所有含 project_meta.json 的目录
        """

    # --- 更新 ---

    def update_title(self, project_id: str, title: str) -> bool:
        """更新项目标题"""

    def refresh_media_assets(self, project_id: str) -> List[MediaAssetRef]:
        """重新扫描媒体资产状态"""

    # --- 持久化 ---

    def _save_project_meta(self, project: Project) -> None:
        """写入 project_meta.json"""

    def _load_project_meta(self, project_dir: Path) -> Optional[Project]:
        """读取 project_meta.json"""

    def _scan_media_assets(self, project_dir: Path) -> List[MediaAssetRef]:
        """扫描目录中的媒体文件，生成 MediaAssetRef 列表

        检测文件:
            - *.mp4 / *.mkv / *.avi → video
            - audio.wav → audio
            - peaks.json → peaks
            - proxy_720p.mp4 → proxy_720p
            - preview_360p.mp4 → preview_360p
            - thumbnail_sprite.jpg → thumbnail
        """
```

### 复用策略

- 目录结构完全复用 `config.JOBS_DIR`
- 文件命名约定与现有 `job_meta.json` 并行（`project_meta.json`）
- 媒体文件扫描逻辑可参考 `media_routes.py` 中的文件查找模式

## 5. Step 4: subtitle_doc_service.py

**文件**: `backend/app/services/subtitle_doc_service.py`（新增）
**估算行数**: ~350

### 核心接口

```python
from pathlib import Path
from typing import Optional, List
from backend.app.services.subtitle_edit_store import (
    load_edits, save_edit, load_deleted_indices,
    add_deletion, create_manual_entry, get_edit_store_path,
)
from backend.app.models.project_models import generate_segment_id, SubtitleDocMeta


class SubtitleDocService:
    """字幕文档服务 - 封装 subtitle_edit_store，添加 segment_id 支持

    设计原则:
        - subtitle_edit_store.py 零修改，仅作为底层存储层
        - 本服务在其上添加 segment_id 映射和领域语义
        - sentence_index 仅在与底层交互时使用
    """

    # --- 导入 ---

    def import_segments(
        self,
        project_dir: Path,
        segments: list[dict],
        source_type: str = "import",
    ) -> SubtitleDocMeta:
        """导入字幕段列表

        参数:
            project_dir: 项目目录
            segments: [{text, start, end}, ...]
            source_type: transcribe / import / legacy

        核心流程:
            1. 为每个 segment 生成 segment_id
            2. 按 sentence_index 顺序写入 subtitle_edits.json（调用 save_edit）
            3. 写入 _segment_map 到 subtitle_edits.json
            4. 返回 SubtitleDocMeta
        """

    # --- 加载 ---

    def load_segments(self, project_dir: Path) -> list[dict]:
        """加载完整字幕列表

        核心流程:
            1. 调用 load_edits(project_dir) 获取 {index: edit_data}
            2. 调用 load_deleted_indices(project_dir) 获取已删除索引
            3. 加载 _segment_map（如果存在）
            4. 合并为 SubtitleSegment 列表，按时间排序
            5. 没有 segment_id 的旧条目自动补充
        """

    def get_segment(self, project_dir: Path, segment_id: str) -> Optional[dict]:
        """根据 segment_id 获取单个字幕段"""

    # --- 编辑 ---

    def update_segment(
        self,
        project_dir: Path,
        segment_id: str,
        update: dict,
    ) -> bool:
        """更新字幕段

        核心流程:
            1. 通过 _segment_map 将 segment_id 映射为 sentence_index
            2. 调用 save_edit(project_dir, sentence_index, update, original_text)
            3. 返回成功/失败
        """

    def create_segment(
        self,
        project_dir: Path,
        text: str,
        start: float,
        end: float,
    ) -> dict:
        """创建新字幕段

        核心流程:
            1. 调用 create_manual_entry(project_dir, text, start, end)
            2. 为返回的条目分配 segment_id
            3. 更新 _segment_map
            4. 返回新段数据
        """

    def delete_segment(self, project_dir: Path, segment_id: str) -> bool:
        """删除字幕段

        核心流程:
            1. 通过 _segment_map 获取 sentence_index
            2. 调用 add_deletion(project_dir, sentence_index)
            3. 从 _segment_map 移除
        """

    # --- segment_map 管理 ---

    def _load_segment_map(self, project_dir: Path) -> dict[str, int]:
        """从 subtitle_edits.json 中加载 _segment_map 字段"""

    def _save_segment_map(self, project_dir: Path, seg_map: dict[str, int]) -> None:
        """将 _segment_map 回写到 subtitle_edits.json"""

    def _ensure_segment_ids(self, project_dir: Path, edits: dict) -> dict[str, int]:
        """确保所有 edit 条目都有对应的 segment_id，没有的自动补充"""

    # --- 导出 ---

    def export_srt(self, project_dir: Path) -> str:
        """导出 SRT 格式字符串（复用现有 SRT 生成逻辑）"""

    def export_ass(self, project_dir: Path) -> str:
        """导出 ASS 格式字符串"""

    # --- 字幕文件解析 ---

    @staticmethod
    def parse_srt(content: str) -> list[dict]:
        """解析 SRT 文件内容为 [{text, start, end}, ...]"""

    @staticmethod
    def parse_ass(content: str) -> list[dict]:
        """解析 ASS 文件内容"""

    @staticmethod
    def parse_vtt(content: str) -> list[dict]:
        """解析 VTT 文件内容"""
```

### 与 subtitle_edit_store 的关系

```
SubtitleDocService（新增，语义层）
    │
    ├── load_segments()  →  load_edits() + load_deleted_indices()
    ├── update_segment() →  save_edit()
    ├── create_segment() →  create_manual_entry()
    ├── delete_segment() →  add_deletion()
    │
    └── _segment_map     →  subtitle_edits.json 的可选字段
                            （subtitle_edit_store 不感知此字段）
```

**subtitle_edit_store.py 零改动**：这是本方案最大的复用点。所有现有的字幕编辑逻辑（防抖保存、并发锁、手动索引递减等）完全保留。

## 6. Step 5: legacy_projection_service.py

**文件**: `backend/app/services/legacy_projection_service.py`（新增）
**估算行数**: ~300

### 核心接口

```python
from pathlib import Path
from typing import Optional, Tuple
import json
from backend.app.models.project_models import (
    Project, generate_legacy_project_id, SubtitleDocMeta,
)
from backend.app.core.config import ProjectConfig as config


class LegacyProjectionService:
    """旧任务投影服务 - 将历史 job_id 映射为 Project

    核心策略:
        - 懒迁移：首次 resolve 时创建 Project(mode=legacy)
        - 幂等映射：同一 job_id 始终产生同一 project_id
        - 非破坏性：不修改旧任务的任何文件
    """

    MAP_FILE = "_legacy_map.json"

    def __init__(self, project_service, subtitle_doc_service):
        self._project_service = project_service
        self._subtitle_doc_service = subtitle_doc_service
        self._map_path = config.JOBS_DIR / self.MAP_FILE
        self._map: dict = self._load_map()

    # --- 核心接口 ---

    def resolve(self, job_id: str) -> Tuple[str, bool]:
        """解析 job_id 为 project_id

        返回:
            (project_id, is_newly_migrated)

        流程:
            1. 查 _legacy_map，已映射直接返回
            2. 检查 {JOBS_DIR}/{job_id}/ 是否存在
            3. 存在 → 触发 _migrate(job_id)
            4. 不存在 → 抛出 404
        """

    def _migrate(self, job_id: str) -> str:
        """首次迁移旧任务

        流程:
            1. project_id = generate_legacy_project_id(job_id)
            2. project_dir = config.JOBS_DIR / job_id（原地创建，不复制文件）
            3. 恢复字幕 → SubtitleDoc
            4. 扫描媒体资产
            5. 创建 Project(mode=legacy, job_id=job_id)
            6. 写入 project_meta.json 到原 job_dir
            7. 记录映射到 _legacy_map
            8. 返回 project_id
        """

    # --- 字幕恢复 ---

    def _restore_subtitles(self, job_dir: Path) -> list[dict]:
        """按优先级恢复字幕

        优先级:
            1. subtitle_edits.json + checkpoint/transcription_text（最完整）
            2. checkpoint/transcription_text（次优）
            3. *.srt 文件（兜底）

        恢复时执行:
            - 合并 is_modified/original_text
            - 应用 deleted_indices
            - 为无 segment_id 的旧句生成稳定 ID
            - 标记 source_type=legacy 与 origin_sentence_index
        """

    def _restore_from_edits_and_checkpoint(self, job_dir: Path) -> Optional[list[dict]]:
        """策略 1: subtitle_edits.json + checkpoint"""

    def _restore_from_checkpoint(self, job_dir: Path) -> Optional[list[dict]]:
        """策略 2: 仅 checkpoint"""

    def _restore_from_srt(self, job_dir: Path) -> Optional[list[dict]]:
        """策略 3: 扫描 .srt 文件"""

    # --- 映射管理 ---

    def _load_map(self) -> dict:
        """加载 _legacy_map.json"""

    def _save_map(self) -> None:
        """保存 _legacy_map.json"""

    def is_legacy_job(self, job_id: str) -> bool:
        """检查 job_id 是否已有映射"""

    def get_all_mappings(self) -> dict:
        """返回所有映射（诊断用）"""
```

### 迁移策略要点

1. **原地创建**: 不复制任何文件。在旧 `{JOBS_DIR}/{job_id}/` 目录中直接写入 `project_meta.json`，与现有 `job_meta.json` 并存
2. **幂等保证**: `generate_legacy_project_id(job_id)` 使用确定性派生，重复调用返回相同 ID
3. **非阻塞**: resolve 过程不超过 100ms（文件 IO 为主，无网络调用）
4. **回滚安全**: 删除 `project_meta.json` 和 `_legacy_map.json` 中的映射即可回滚

## 7. Step 6: project_routes.py

**文件**: `backend/app/api/routes/project_routes.py`（新增）
**估算行数**: ~500

### API 端点

```python
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

router = APIRouter(prefix="/api/projects", tags=["projects"])


# --- 导入与创建 ---

@router.post("/import")
async def import_project(
    subtitle_file: UploadFile = File(...),
    format: str = Form("srt"),           # srt | ass | vtt
    video_path: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
):
    """导入字幕文件创建项目

    请求: multipart/form-data
    响应: {project_id, title, subtitle_count, media_assets}

    流程:
        1. 读取上传文件内容
        2. SubtitleDocService.parse_{format}() 解析
        3. ProjectService.create_import_project() 创建项目
        4. 如有 video_path，挂载媒体（提取音频 + peaks）
        5. 返回项目信息
    """


# --- 查询 ---

@router.get("/")
async def list_projects(flavor: Optional[str] = None):
    """列出所有项目

    响应: [{project_id, title, mode, flavor, subtitle_count, created_at}, ...]
    """


@router.get("/{project_id}")
async def get_project(project_id: str):
    """获取单个项目详情

    响应: Project.to_dict()
    404: 项目不存在
    """


# --- 字幕操作 ---

@router.get("/{project_id}/subtitles")
async def get_subtitles(project_id: str):
    """获取项目字幕列表

    响应: {
        segments: [{segment_id, text, start, end, is_modified, ...}, ...],
        doc_meta: {source_type, segment_count, version}
    }
    """


@router.put("/{project_id}/subtitles/{segment_id}")
async def update_subtitle(project_id: str, segment_id: str, body: dict):
    """更新单条字幕

    请求体: {text?, start?, end?}
    响应: {success: true, segment: {...}}
    """


@router.post("/{project_id}/subtitles")
async def create_subtitle(project_id: str, body: dict):
    """新增字幕

    请求体: {text, start, end}
    响应: {segment_id, text, start, end}
    """


@router.delete("/{project_id}/subtitles/{segment_id}")
async def delete_subtitle(project_id: str, segment_id: str):
    """删除字幕

    响应: {success: true}
    """


# --- 导出 ---

@router.get("/{project_id}/export")
async def export_subtitles(project_id: str, format: str = "srt"):
    """导出字幕文件

    参数: format = srt | ass | vtt
    响应: FileResponse 或 StreamingResponse
    """


# --- 媒体 ---

@router.get("/{project_id}/media/{asset_type}")
async def get_media_asset(project_id: str, asset_type: str):
    """获取媒体资产

    asset_type: video | audio | peaks | proxy_720p | preview_360p | thumbnail

    复用逻辑:
        查找 {JOBS_DIR}/{project_id}/ 或 {JOBS_DIR}/{job_id}/ 中的文件
        调用 media_routes 中同名处理函数的核心逻辑
    """
```

### 与现有路由的关系

- `/api/projects/import` 是全新端点
- 字幕操作端点的底层逻辑与 `transcription_routes.py` 中的 `/jobs/{job_id}/subtitles` 共享 `SubtitleDocService`
- 媒体端点复用 `media_routes.py` 的文件查找和流式响应逻辑

## 8. Step 7: legacy_compat_routes.py

**文件**: `backend/app/api/routes/legacy_compat_routes.py`（新增）
**估算行数**: ~200

```python
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

router = APIRouter(prefix="/api/legacy", tags=["legacy"])


@router.get("/tasks/{job_id}/resolve")
async def resolve_legacy_task(job_id: str):
    """解析旧任务 ID 为项目 ID

    响应: {
        project_id: str,
        is_newly_migrated: bool,
        redirect_url: "/editor/project/{project_id}"
    }
    404: 任务不存在
    """


@router.get("/tasks/{job_id}/status")
async def get_legacy_status(job_id: str):
    """获取旧任务状态（兼容旧前端）

    先 resolve 为 project，再返回兼容格式的状态
    """
```

## 9. Step 8: main.py 改造

**文件**: `backend/app/main.py`
**当前行数**: 746
**修改量**: ~60 行

### 改动要点

```python
# 在路由注册区域（约 L229-370）增加条件逻辑：

from backend.app.core.config import IS_LITE, FLAVOR

# 1. 始终注册的路由
app.include_router(media_routes.router)
app.include_router(system_routes.router)
app.include_router(config_routes.router)
app.include_router(file_routes.router)
app.include_router(hardware_routes.router)
app.include_router(debug_routes.router)

# 2. 新增：项目路由（始终注册）
from backend.app.api.routes.project_routes import router as project_router
from backend.app.api.routes.legacy_compat_routes import router as legacy_router
app.include_router(project_router)
app.include_router(legacy_router)

# 3. 条件注册：仅 Full 模式
if not IS_LITE:
    app.include_router(transcription_routes.router)
    app.include_router(demucs_router)
    app.include_router(speaker_routes.router)
    app.include_router(model_routes.router)
    app.include_router(model_runtime_routes.router)

# 4. 启动事件中条件初始化服务
@app.on_event("startup")
async def startup_event():
    # 始终初始化
    init_sse_manager()
    init_media_prep_service()  # 简化版

    if not IS_LITE:
        # 仅 Full 模式初始化
        init_model_manager()
        init_queue_service()
        init_transcription_service()
```

### 注意事项

- 不删除任何现有代码，仅用 `if not IS_LITE:` 包裹
- 服务初始化的条件判断必须在启动事件内部
- 路由注册的条件判断在应用创建阶段

## 10. Step 9: transcription_routes.py 桥接

**文件**: `backend/app/api/routes/transcription_routes.py`
**当前行数**: 2445
**修改量**: ~80 行

### 改动要点

在字幕编辑相关端点中添加可选的 `SubtitleDocService` 转发：

```python
# 在 /jobs/{job_id}/subtitles PUT 端点中：
@router.put("/jobs/{job_id}/subtitles")
async def update_subtitles(job_id: str, body: dict):
    # 新增：检查是否已有 project 映射
    legacy_service = get_legacy_projection_service()
    if legacy_service and legacy_service.is_legacy_job(job_id):
        project_id, _ = legacy_service.resolve(job_id)
        # 转发到 SubtitleDocService
        subtitle_doc_service = get_subtitle_doc_service()
        # ... 使用 segment_id 或 sentence_index 桥接
        return result

    # 原有逻辑保持不变
    # ...
```

### 最小改动原则

- 仅在字幕编辑的 3 个端点（更新、创建、删除）添加桥接检查
- 桥接检查失败时静默回退到原有逻辑
- 不修改任何非字幕编辑的端点

## 11. Step 10: streaming_subtitle.py 频道扩展

**文件**: `backend/app/services/streaming_subtitle.py`
**当前行数**: 1026
**修改量**: ~40 行

### 改动要点

在 SSE 事件推送函数中添加 project 频道别名：

```python
def push_subtitle_event(sse_manager, job_id, event_type, data, project_id=None):
    """推送字幕事件

    改动：增加可选的 project_id 参数
    - 始终推送到 job:{job_id}（向后兼容）
    - 如果提供 project_id，同时推送到 project:{project_id}
    """
    # 原有推送逻辑不变
    sse_manager.publish(f"job:{job_id}", event_type, data)

    # 新增：project 频道别名推送
    if project_id:
        sse_manager.publish(f"project:{project_id}", event_type, data)
```

### StreamingSubtitleManager 扩展

```python
class StreamingSubtitleManager:
    def __init__(self, job_id: str, project_id: Optional[str] = None):
        self.job_id = job_id
        self.project_id = project_id  # 新增：可选 project_id
        # ... 其余不变
```

## 12. Step 11: media_routes.py 兼容

**文件**: `backend/app/api/routes/media_routes.py`
**当前行数**: 2148
**修改量**: ~30 行

### 改动要点

在核心媒体端点中添加 project_id 查找逻辑：

```python
def _resolve_media_dir(identifier: str) -> Path:
    """统一解析媒体目录

    identifier 可以是 job_id 或 project_id
    查找顺序:
        1. {JOBS_DIR}/{identifier}/ 直接存在 → 返回
        2. 查询 legacy_map 获取对应的 job_dir → 返回
        3. 抛出 404
    """
    direct_path = config.JOBS_DIR / identifier
    if direct_path.exists():
        return direct_path
    # legacy 查找...
    raise HTTPException(404)
```

此函数提取自现有代码中的路径构造逻辑，减少重复。

## 13. Step 12: job_models.py 扩展

**文件**: `backend/app/models/job_models.py`
**当前行数**: 653
**修改量**: ~20 行

### 改动要点

```python
@dataclass
class JobSettings:
    # 新增字段
    mode: str = "transcribe"  # "transcribe" | "edit_only"
    # ... 其余不变


@dataclass
class JobState:
    # 新增字段
    project_id: Optional[str] = None  # 关联的 project_id
    # ... 其余不变

    def to_dict(self) -> dict:
        d = {
            # ... 现有字段
            "project_id": self.project_id,  # 新增
        }
        return d
```

## 14. 服务依赖注册

### 工厂函数（在 main.py 或独立的 di.py 中）

```python
# 服务单例
_project_service: Optional[ProjectService] = None
_subtitle_doc_service: Optional[SubtitleDocService] = None
_legacy_projection_service: Optional[LegacyProjectionService] = None


def get_project_service() -> ProjectService:
    global _project_service
    if _project_service is None:
        _project_service = ProjectService()
    return _project_service


def get_subtitle_doc_service() -> SubtitleDocService:
    global _subtitle_doc_service
    if _subtitle_doc_service is None:
        _subtitle_doc_service = SubtitleDocService()
    return _subtitle_doc_service


def get_legacy_projection_service() -> LegacyProjectionService:
    global _legacy_projection_service
    if _legacy_projection_service is None:
        _legacy_projection_service = LegacyProjectionService(
            project_service=get_project_service(),
            subtitle_doc_service=get_subtitle_doc_service(),
        )
    return _legacy_projection_service
```

遵循项目现有的全局单例模式（`get_xxx_service()`），保持一致性。
