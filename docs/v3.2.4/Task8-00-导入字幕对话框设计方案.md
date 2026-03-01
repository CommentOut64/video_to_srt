# Task8: 导入字幕对话框（ImportDialog）设计方案

> 版本: V3.2.4+dev.20260226.01
> 状态: 草案
> 涉及范围: 前端 ImportDialog 新增 + Lite 统一到 task 界面 + 后端字幕文件列表/本地导入 API

---

## 1. 背景与目标

### 1.1 现状问题

1. **字幕导入与视频上传风格割裂**：视频上传使用 TaskCreateDialog（el-dialog 弹窗 + 双 Tab），而字幕导入使用独立的 ImportView.vue 全页面（Tailwind 布局），两者视觉体验不统一。
2. **Lite 版本使用独立界面**：Lite 模式默认路由到 `/import`（ImportView），与 Full 模式的 `/tasks`（TaskListView）是两套完全不同的界面，维护成本高。
3. **ImportView 功能单一**：仅支持直接上传字幕文件，不支持从本地 input 目录选择，与视频上传的双模式体验不对称。

### 1.2 目标

1. **新建 ImportDialog**：模仿 TaskCreateDialog 的 el-dialog 风格，提供「直接上传」和「从本地目录选择」两种 Tab，将字幕导入逻辑全部迁移到该对话框。
2. **Lite 统一到 task 界面**：Lite 版本使用与 Full 相同的 TaskListView，仅隐藏"上传视频"按钮，保留"导入字幕"入口。
3. **废弃 ImportView**：`/import` 路由重定向到 `/tasks?action=import`，删除 ImportView.vue。

### 1.3 核心约束

- 字幕文件导入是**必选**的（.srt / .ass / .vtt）
- 视频/音频文件关联是**可选**的
- 不需要预设、前处理、转录、后处理等 SettingsTabs
- 每次导入一个字幕文件，创建一个项目

---

## 2. 整体布局

### 2.1 直接上传模式

```
+---------------------------------------------------+
|  导入字幕                                       [X] |  <- dialog header
+---------------------------------------------------+
|                                                     |
|  [直接上传]  [从本地目录选择]                         |  <- 字幕选择 tabs
|  ┌───────────────────────────────────────────────┐  |
|  │                                               │  |
|  │     拖拽字幕文件到此处，或 点击选择             │  |  <- el-upload 拖拽区
|  │        支持 SRT、ASS、VTT 格式                 │  |
|  │                                               │  |
|  └───────────────────────────────────────────────┘  |
|  [x] my_subtitle.srt                                |  <- 已选文件 tag
|                                                     |
|  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  |
|                                                     |
|  关联媒体（可选）                                     |
|  ┌───────────────────────────────────────────────┐  |
|  │  + 选择视频或音频文件                           │  |  <- 紧凑选择区
|  │    关联媒体以启用波形和视频预览                  │  |
|  └───────────────────────────────────────────────┘  |
|                                                     |
|  项目名称                                            |
|  ┌───────────────────────────────────────────────┐  |
|  │  my_subtitle                                   │  |  <- el-input (自动填充)
|  └───────────────────────────────────────────────┘  |
|                                                     |
+---------------------------------------------------+
|  [取消]                              [开始导入]      |  <- footer
+---------------------------------------------------+
```

### 2.2 从本地目录选择模式

```
+---------------------------------------------------+
|  导入字幕                                       [X] |
+---------------------------------------------------+
|                                                     |
|  [直接上传]  [从本地目录选择]        [打开input目录]  |
|  ┌───────────────────────────────────────────────┐  |
|  │ [x] 文件名              大小      修改时间     │  |  <- el-table (字幕)
|  │ ─────────────────────────────────────────────  │  |
|  │ ( ) my_video.srt        12 KB    2026-02-26   │  |     单选 radio
|  │ (*) meeting_notes.ass   8 KB     2026-02-25   │  |
|  │ ( ) interview.vtt       15 KB    2026-02-24   │  |
|  └───────────────────────────────────────────────┘  |
|                                                     |
|  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  |
|                                                     |
|  关联媒体（可选）                                     |
|  ┌───────────────────────────────────────────────┐  |
|  │ [ ] demo.mp4            520 MB   2026-02-26   │  |  <- el-table (媒体)
|  │ [ ] audio.wav           150 MB   2026-02-25   │  |     单选 radio
|  └───────────────────────────────────────────────┘  |
|                                                     |
|  项目名称                                            |
|  ┌───────────────────────────────────────────────┐  |
|  │  meeting_notes                                 │  |
|  └───────────────────────────────────────────────┘  |
|                                                     |
+---------------------------------------------------+
|  [取消]                              [开始导入]      |
+---------------------------------------------------+
```

### 2.3 高度预算

**目标**: dialog 总高度 <= 640px，适配 768p 笔记本。

```
区域                              高度(px)    说明
─────────────────────────────────────────────────────
el-dialog header                  54          title + padding
el-dialog body padding-top        24
── 字幕 tabs header               40          el-tabs header
── 字幕 tabs margin-bottom        16          与 TaskCreateDialog 一致
── 字幕 tab 内容区                 180         直接上传: dragger(140)+tip/tag(30+10)
                                              本地选择: el-table(max-height:180)
── 分隔间距                         12
── 媒体区标题                      20          "关联媒体（可选）"
── 媒体区内容                       64          直接上传: 点击选择区(64px)
                                  120         本地选择: el-table(max-height:120)
── 间距                             12
── 项目名称标题                    20          "项目名称" label
── 项目名称输入框                   36          el-input
el-dialog body padding-bottom     20
el-dialog footer                  52          按钮行 + padding
─────────────────────────────────────────────────────
总计 (直接上传模式)               ~550px
总计 (本地选择模式)               ~610px      均低于 640px 限制
```

### 2.4 关键约束

| 项目 | 规格 | 说明 |
|------|------|------|
| dialog 宽度 | 700px | 与 TaskCreateDialog 一致 |
| 字幕 Tab 内容区 | 180px | 直接上传: dragger + tag; 本地选择: el-table |
| 媒体区（直接上传） | 64px | 紧凑点击选择区 |
| 媒体区（本地选择） | 120px | el-table 较小高度（媒体是可选配角） |
| dialog 总高度 | <= 640px | 适配 768p 笔记本 |
| footer | 始终可见 | 取消 + 开始导入 |

---

## 3. 各区域详细设计

### 3.1 Tab 1: 直接上传

与 TaskCreateDialog 的上传 Tab 风格完全一致。

#### 3.1.1 字幕上传区

- el-upload 拖拽区域，`accept=".srt,.ass,.vtt"`
- **单文件限制**（`limit="1"`），区别于视频上传的 `multiple`
- 选中后下方显示 file-tag（带关闭按钮，可删除后重新选择）
- 自动从文件名提取项目名称（去掉 `.srt/.ass/.vtt` 后缀）
- 自动检测字幕格式（根据扩展名）

```vue
<el-upload
  drag
  :auto-upload="false"
  :show-file-list="false"
  :limit="1"
  accept=".srt,.ass,.vtt"
  @change="handleSubtitleFileChange"
>
  <el-icon class="el-icon--upload"><Document /></el-icon>
  <div class="el-upload__text">
    拖拽字幕文件到此处，或 <em>点击选择</em>
  </div>
</el-upload>
<div v-if="!subtitleFile" class="upload-tip">
  支持 SRT、ASS、VTT 格式
</div>
<div v-else class="selected-files-tags">
  <div class="tags-container">
    <span class="file-tag">
      <el-icon class="tag-close" @click.stop="removeSubtitleFile"><Close /></el-icon>
      <span class="tag-name">{{ subtitleFile.name }}</span>
    </span>
  </div>
</div>
```

#### 3.1.2 媒体选择区

采用比字幕区更紧凑的设计（媒体是可选配角）。不使用 el-upload 拖拽区，使用虚线边框的点击选择区。

```vue
<div class="section-label">关联媒体 <span class="optional">（可选）</span></div>
<div
  class="media-select-zone"
  :class="{ 'has-file': mediaFile }"
  @click="triggerMediaFileInput"
>
  <template v-if="!mediaFile">
    <el-icon><FolderAdd /></el-icon>
    <span>选择视频或音频文件</span>
    <span class="media-hint">关联媒体以启用波形和视频预览</span>
  </template>
  <template v-else>
    <span class="file-tag">
      <el-icon class="tag-close" @click.stop="removeMediaFile"><Close /></el-icon>
      <span class="tag-name">{{ mediaFile.name }}</span>
      <span class="tag-size">{{ formatFileSize(mediaFile.size) }}</span>
    </span>
    <span class="media-hint">点击更换媒体文件</span>
  </template>
</div>
<input ref="mediaInput" type="file" accept="video/*,audio/*" style="display:none" @change="handleMediaFileChange" />
```

#### 3.1.3 导入 API

复用现有 `POST /api/projects/import`（FormData 上传），无需后端变更。

### 3.2 Tab 2: 从本地目录选择

与 TaskCreateDialog 的本地选择 Tab 布局一致，但分为两个表格。

#### 3.2.1 字幕文件表格

- 数据来源：**新增** `GET /api/files/subtitles`（扫描 input 目录中的 .srt/.ass/.vtt）
- **单选 radio**（一次只导入一个字幕文件）
- 列：文件名 / 大小 / 修改时间
- max-height: 180px
- 选中后自动从文件名提取项目名称

```vue
<div class="file-list-container subtitle-table">
  <div v-if="loadingSubtitles" class="loading-files">
    <el-icon class="is-loading"><Loading /></el-icon>
    <span>加载文件列表中...</span>
  </div>
  <div v-else-if="subtitleFiles.length === 0" class="empty-files">
    <p>input 目录中没有字幕文件</p>
    <p class="hint">请先将 .srt / .ass / .vtt 文件放入 input 目录</p>
  </div>
  <div v-else>
    <el-table
      :data="subtitleFiles"
      highlight-current-row
      max-height="180"
      @current-change="handleSubtitleSelect"
    >
      <el-table-column width="55">
        <template #default="{ row }">
          <el-radio :model-value="selectedSubtitle" :value="row.name" />
        </template>
      </el-table-column>
      <el-table-column prop="name" label="文件名" min-width="200" />
      <el-table-column prop="size" label="大小" width="100" />
      <el-table-column prop="modified" label="修改时间" width="160" />
    </el-table>
  </div>
</div>
```

#### 3.2.2 媒体文件表格

- 标题："关联媒体（可选）"
- 数据来源：**复用现有** `GET /api/files`（input 目录中的视频/音频文件）
- **单选 radio**，可不选
- max-height: 120px（视觉权重低于字幕表格）

```vue
<div class="section-label">关联媒体 <span class="optional">（可选）</span></div>
<div class="file-list-container media-table">
  <div v-if="loadingMedias" class="loading-files">...</div>
  <div v-else-if="mediaFiles.length === 0" class="empty-files">
    <p>input 目录中没有媒体文件</p>
  </div>
  <div v-else>
    <el-table
      :data="mediaFiles"
      highlight-current-row
      max-height="120"
      @current-change="handleMediaSelect"
    >
      <el-table-column width="55">
        <template #default="{ row }">
          <el-radio :model-value="selectedMedia" :value="row.name" />
        </template>
      </el-table-column>
      <el-table-column prop="name" label="文件名" min-width="200" />
      <el-table-column prop="size" label="大小" width="100" />
      <el-table-column prop="modified" label="修改时间" width="160" />
    </el-table>
  </div>
</div>
```

#### 3.2.3 "打开input目录"按钮

位于 Tab header 右侧，与 TaskCreateDialog 位置一致。

```vue
<el-button
  v-if="importMode === 'select'"
  text
  size="small"
  @click="handleOpenInputFolder"
  class="open-folder-btn"
>
  打开input目录
</el-button>
```

调用现有 `POST /api/files/open-input-folder`。

#### 3.2.4 导入 API

**新增** `POST /api/projects/import-local`（从 input 目录路径导入，无需文件上传）。

### 3.3 项目名称区

两种模式共享。

```vue
<div class="section-label">项目名称</div>
<el-input
  v-model="projectTitle"
  placeholder="自动从字幕文件名提取"
  clearable
/>
```

- 选择字幕文件后自动填充（去掉扩展名）
- 用户可手动修改
- 为空时后端使用字幕文件名作为默认值

### 3.4 Dialog Footer

```
+---------------------------------------------------+
|  [取消]                              [开始导入]      |
+---------------------------------------------------+
```

- **取消**：关闭 dialog，清空状态
- **开始导入**：
  - `disabled` 条件：未选择字幕文件 **或** 正在导入中
  - 导入中文案：`"导入中..."`
  - 正常文案：`"开始导入"`
- **无"保存预设"按钮**：导入不涉及预设

---

## 4. 后端变更

### 4.1 新增：列出 input 目录中的字幕文件

**文件**：`backend/app/services/file_service.py`

```python
SUBTITLE_EXTENSIONS = {'.srt', '.ass', '.vtt'}

def is_supported_subtitle(self, filename: str) -> bool:
    """检查是否为支持的字幕文件"""
    ext = os.path.splitext(filename.lower())[1]
    return ext in self.SUBTITLE_EXTENSIONS

def list_input_subtitle_files(self) -> List[Dict]:
    """获取输入目录中的所有字幕文件"""
    files = []
    if os.path.exists(self.input_dir):
        for filename in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, filename)
            if os.path.isfile(file_path) and self.is_supported_subtitle(filename):
                stat = os.stat(file_path)
                files.append({
                    'name': filename,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(
                        stat.st_mtime
                    ).strftime('%Y-%m-%d %H:%M:%S'),
                    'modified_timestamp': int(stat.st_mtime * 1000),
                    'path': file_path,
                })
    files.sort(key=lambda x: x['modified_timestamp'], reverse=True)
    return files
```

**文件**：`backend/app/api/routes/file_routes.py` 新增端点

```python
@router.get("/files/subtitles")
async def list_subtitle_files():
    """获取输入目录中的所有字幕文件"""
    try:
        files = file_service.list_input_subtitle_files()
        return {"files": files, "input_dir": file_service.input_dir}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取字幕文件列表失败: {str(e)}"
        )
```

### 4.2 新增：从本地路径导入项目

**文件**：`backend/app/api/routes/project_routes.py` 新增端点

```python
@router.post("/api/projects/import-local")
async def import_project_local(
    subtitle_filename: str = Form(...),
    media_filename: Optional[str] = Form(None),
    title: str = Form(""),
    flavor: str = Form(FLAVOR),
):
    """从 input 目录的本地文件导入项目（无需上传）"""

    # 1. 读取字幕文件
    subtitle_path = file_service.get_input_file_path(subtitle_filename)
    if not os.path.exists(subtitle_path):
        raise HTTPException(404, f"字幕文件不存在: {subtitle_filename}")

    content = _read_local_file(subtitle_path)
    detected_format = _detect_subtitle_format(subtitle_filename, None)
    segments = _parse_subtitle(content, detected_format)

    # 2. 可选：获取媒体文件路径
    media_path = None
    if media_filename:
        media_path = file_service.get_input_file_path(media_filename)
        if not os.path.exists(media_path):
            raise HTTPException(404, f"媒体文件不存在: {media_filename}")

    # 3. 创建项目（媒体文件通过硬链接关联到项目目录）
    project = project_service.create_import_project(
        title=title or subtitle_filename,
        subtitle_segments=segments,
        video_path=media_path,
        flavor=flavor,
    )

    return {
        "success": True,
        "data": project.to_dict(),
        "redirect_url": f"/editor/project/{project.project_id}",
    }
```

**辅助函数**（可复用 `import_project` 中已有的）：

```python
def _read_local_file(path: str) -> str:
    """读取本地文件内容，自动检测编码"""
    raw_bytes = Path(path).read_bytes()
    return _decode_upload_content(raw_bytes)

def _parse_subtitle(content: str, fmt: str) -> List[dict]:
    """根据格式调用对应解析器"""
    sds = get_subtitle_doc_service()
    if fmt == "ass":
        return sds.parse_ass(content)
    elif fmt == "vtt":
        return sds.parse_vtt(content)
    return sds.parse_srt(content)
```

### 4.3 后端 API 总结

| 方法 | 路径 | 说明 | 状态 |
|------|------|------|------|
| GET | `/api/files` | 列出 input 目录中的媒体文件 | 现有，复用 |
| GET | `/api/files/subtitles` | 列出 input 目录中的字幕文件 | **新增** |
| POST | `/api/files/open-input-folder` | 打开 input 目录 | 现有，复用 |
| POST | `/api/projects/import` | 上传字幕+媒体文件创建项目 | 现有，复用 |
| POST | `/api/projects/import-local` | 从 input 目录本地路径导入 | **新增** |

---

## 5. 前端 API 客户端扩展

### 5.1 fileApi.js 新增

```javascript
/**
 * 获取 input 目录中的所有字幕文件
 * @returns {Promise<{files: Array, input_dir: string}>}
 */
async listSubtitleFiles() {
  return apiClient.get('/api/files/subtitles')
}
```

### 5.2 projectApi.js 新增

```javascript
/**
 * 从 input 目录本地路径导入项目（无需文件上传）
 * @param {string} subtitleFilename - input 目录中的字幕文件名
 * @param {string|null} mediaFilename - input 目录中的媒体文件名（可选）
 * @param {string} title - 项目名称
 * @returns {Promise<Object>} 项目数据
 */
async importProjectLocal(subtitleFilename, mediaFilename = null, title = '') {
  const formData = new FormData()
  formData.append('subtitle_filename', subtitleFilename)
  formData.append('flavor', FLAVOR)
  if (title) formData.append('title', title)
  if (mediaFilename) formData.append('media_filename', mediaFilename)

  const response = await apiClient.post('/api/projects/import-local', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return unwrapEnvelope(response, null)
}
```

---

## 6. Lite 版本统一策略

### 6.1 路由变更

**文件**：`frontend/src/router/index.js`

```diff
  {
    path: '/',
-   redirect: IS_LITE ? '/import' : '/tasks',
+   redirect: '/tasks',
  },
  {
    path: '/tasks',
    name: 'TaskList',
    component: () => import('@/views/TaskListView.vue'),
    meta: { title: '任务列表' },
-   beforeEnter: (_, __, next) => {
-     if (!ROUTE_VISIBILITY.taskList) {
-       next('/import')
-       return
-     }
-     next()
-   },
  },
  {
    path: '/import',
    name: 'Import',
-   component: () => import('@/views/ImportView.vue'),
-   meta: { title: '导入字幕' },
+   redirect: { path: '/tasks', query: { action: 'import' } },
  },
```

### 6.2 Flavor 配置变更

**文件**：`frontend/src/config/flavor.js`

```diff
  export const ROUTE_VISIBILITY = Object.freeze({
-   taskList: !IS_LITE,
+   taskList: true,         // Lite 也使用任务列表
    taskCreate: !IS_LITE,   // Lite 不显示"上传视频"
    importPage: true,
    editor: true,
  })
```

### 6.3 TaskListHeader 条件渲染

**文件**：`frontend/src/components/task/TaskListHeader.vue`

```vue
<div class="header-right">
  <el-button @click="$emit('open-import')">
    <el-icon><Document /></el-icon>
    导入字幕
  </el-button>
  <!-- Full 模式才显示上传视频 -->
  <el-button v-if="canTranscribe" type="primary" @click="$emit('open-upload')">
    <el-icon><Upload /></el-icon>
    上传视频
  </el-button>
  <el-button type="primary" @click="$emit('exit-system')">
    退出系统
  </el-button>
</div>

<script setup>
import { Upload } from '@element-plus/icons-vue'
import { Document } from '@element-plus/icons-vue'
import { CAPABILITIES } from '@/config/flavor'

const { canTranscribe } = CAPABILITIES

defineEmits(['open-about', 'open-upload', 'open-import', 'exit-system'])
</script>
```

**效果对比**：

| 模式 | Header 按钮 | 默认路由 |
|------|------------|---------|
| Full | `[导入字幕]` `[上传视频]` `[退出系统]` | `/tasks` |
| Lite | `[导入字幕]` `[退出系统]` | `/tasks`（原 `/import` 重定向） |

### 6.4 TaskListView 集成

**文件**：`frontend/src/views/TaskListView.vue`

```javascript
import ImportDialog from '@/components/task/ImportDialog.vue'
import { CAPABILITIES } from '@/config/flavor'

const route = useRoute()
const showImportDialog = ref(false)

onMounted(() => {
  // Lite 模式或 ?action=import 时自动弹出导入窗口
  if (!CAPABILITIES.canTranscribe || route.query.action === 'import') {
    showImportDialog.value = true
  }
})

function handleImportSuccess(projectId) {
  showImportDialog.value = false
  router.push(`/editor/project/${projectId}`)
}
```

```vue
<ImportDialog
  v-model:show-dialog="showImportDialog"
  @close="showImportDialog = false"
  @import-success="handleImportSuccess"
/>
```

### 6.5 TaskCardGrid 空状态适配

**文件**：`frontend/src/components/task/TaskCardGrid.vue`

```vue
<div v-if="tasks.length === 0" class="empty-state">
  <p>暂无任务</p>
  <div class="empty-actions">
    <el-button @click="$emit('open-import')">导入字幕</el-button>
    <el-button v-if="canTranscribe" type="primary" @click="$emit('open-upload')">
      上传视频开始转录
    </el-button>
  </div>
</div>
```

Lite 模式下只显示「导入字幕」，Full 模式下两个按钮并列。

---

## 7. 组件架构

### 7.1 文件结构

```
前端变更:
  frontend/src/
    components/task/
      ImportDialog.vue              # 新增：导入字幕 dialog
      TaskListHeader.vue            # 修改：新增 open-import, 条件隐藏上传视频
      TaskCardGrid.vue              # 修改：空状态新增导入入口
    composables/task-list/
      useTaskImport.js              # 新增：导入逻辑 composable
    views/
      TaskListView.vue              # 修改：集成 ImportDialog + Lite 自动弹出
      ImportView.vue                # 废弃（路由重定向）
    config/
      flavor.js                     # 修改：taskList 始终 true
    router/
      index.js                      # 修改：统一 /tasks, /import 重定向
    services/api/
      fileApi.js                    # 修改：新增 listSubtitleFiles()
      projectApi.js                 # 修改：新增 importProjectLocal()

后端变更:
  backend/app/
    services/
      file_service.py               # 修改：新增 list_input_subtitle_files()
    api/routes/
      file_routes.py                # 修改：新增 GET /api/files/subtitles
      project_routes.py             # 修改：新增 POST /api/projects/import-local
```

### 7.2 数据流

```
TaskListView.vue
  ├── showImportDialog: boolean
  │
  └── ImportDialog.vue
        ├── props: showDialog
        ├── emits: close, import-success(projectId)
        │
        └── useTaskImport composable
              │
              ├── 直接上传模式
              │   ├── subtitleFile: File | null
              │   ├── mediaFile: File | null
              │   └── handleImport() → projectApi.importProject()
              │
              └── 本地选择模式
                  ├── subtitleFiles: []  ← GET /api/files/subtitles
                  ├── mediaFiles: []     ← GET /api/files
                  ├── selectedSubtitle: string | null
                  ├── selectedMedia: string | null
                  └── handleImport() → projectApi.importProjectLocal()
```

### 7.3 ImportDialog 组件接口

```javascript
// Props
defineProps({
  showDialog: { type: Boolean, default: false },
})

// Emits
defineEmits([
  'update:show-dialog',
  'close',
  'import-success',  // 参数: projectId
])
```

### 7.4 useTaskImport composable

```javascript
export function useTaskImport({ router }) {
  // === 共享状态 ===
  const importMode = ref('upload')    // 'upload' | 'select'
  const projectTitle = ref('')
  const isImporting = ref(false)

  // === 直接上传模式 ===
  const subtitleFile = ref(null)      // File 对象
  const mediaFile = ref(null)         // File 对象

  // === 本地选择模式 ===
  const subtitleFiles = ref([])       // input 目录中的字幕文件列表
  const mediaFiles = ref([])          // input 目录中的媒体文件列表
  const selectedSubtitle = ref(null)  // 选中的字幕文件名 (string)
  const selectedMedia = ref(null)     // 选中的媒体文件名 (string)
  const loadingSubtitles = ref(false)
  const loadingMedias = ref(false)

  // === 方法 ===
  function handleSubtitleFileChange(file) { ... }
  function handleMediaFileChange(file) { ... }
  function removeSubtitleFile() { ... }
  function removeMediaFile() { ... }
  async function loadSubtitleFiles() { ... }     // GET /api/files/subtitles
  async function loadMediaFiles() { ... }        // GET /api/files
  async function handleOpenInputFolder() { ... } // POST /api/files/open-input-folder
  function handleSubtitleSelect(row) { ... }     // 字幕表格行选中
  function handleMediaSelect(row) { ... }        // 媒体表格行选中

  async function handleImport() {
    if (importMode.value === 'upload') {
      // 走上传路径: POST /api/projects/import (FormData)
      return projectApi.importProject(subtitleFile, format, mediaFile, title)
    } else {
      // 走本地路径: POST /api/projects/import-local
      return projectApi.importProjectLocal(selectedSubtitle, selectedMedia, title)
    }
  }

  function resetState() { ... }

  // importMode 切换为 'select' 时自动加载文件列表
  watch(importMode, (mode) => {
    if (mode === 'select') {
      loadSubtitleFiles()
      loadMediaFiles()
    }
  })

  return { ... }
}
```

---

## 8. 样式规范

### 8.1 全局 Dialog 样式复用

ImportDialog 与 TaskCreateDialog 共用 `element-override.css` 中的 `.upload-dialog` 全局样式类，确保两个 dialog 外观一致（背景色、圆角、阴影、header/footer 样式）。

### 8.2 字幕 Tab 样式

直接复用 TaskCreateDialog 中 el-tabs 的样式规则（`.tabs-container` 相关），保证 Tab header、active-bar、hover 效果一致。

### 8.3 el-table 样式

直接复用 TaskCreateDialog 中 el-table 的 scoped 样式（`--el-table-*` 变量覆盖 + header/row/hover/scrollbar 定制），确保文件列表表格视觉一致。

### 8.4 区域标签样式

```css
.section-label {
  margin-bottom: 8px;
  color: var(--af-text-primary);
  font-size: 13px;
  font-weight: 500;
}

.section-label .optional {
  color: var(--af-text-muted);
  font-size: 12px;
  font-weight: 400;
}
```

### 8.5 媒体选择区样式（直接上传模式）

```css
.media-select-zone {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  height: 64px;
  padding: 12px;
  background: var(--af-bg-tertiary);
  border: 1px dashed var(--af-border-default);
  border-radius: var(--af-radius-md);
  cursor: pointer;
  transition: all var(--af-transition-fast);
}

.media-select-zone:hover {
  background: var(--af-bg-elevated);
  border-color: var(--af-accent-primary);
}

.media-select-zone.has-file {
  flex-direction: row;
  justify-content: space-between;
  border-style: solid;
}
```

### 8.6 分隔线

```css
.section-divider {
  margin: 12px 0;
  border: none;
  border-top: 1px dashed var(--af-border-light);
}
```

---

## 9. 设计决策

| 编号 | 决策 | 理由 |
|------|------|------|
| D1 | 字幕文件表格用**单选 radio** | 每次只导入一个字幕文件创建一个项目，区别于视频上传的批量多选 checkbox |
| D2 | 媒体文件表格用**单选 radio** | 一个项目最多关联一个媒体文件 |
| D3 | 两个表格分开，字幕表格更大 | 字幕是主角（必选），媒体是配角（可选），视觉主次分明 |
| D4 | 新增 `GET /api/files/subtitles` 独立端点 | 职责清晰，避免改动现有 `GET /api/files` 媒体文件列表的契约 |
| D5 | 新增 `POST /api/projects/import-local` 独立端点 | 本地路径导入参数是文件名而非 File 上传，语义与 `/import` 不同 |
| D6 | Lite 统一使用 `/tasks` | 减少两套界面的维护成本 |
| D7 | `/import` 路由保留为重定向 | 兼容可能存在的外部链接或书签 |
| D8 | 直接上传模式的媒体区用紧凑点击区而非 el-upload | 媒体是可选配角，不需要与字幕同等的拖拽体验，节省纵向空间 |
| D9 | 不需要预设和 SettingsTabs | 导入是纯"文件 -> 编辑项目"操作，不涉及转录管线 |

---

## 10. 实施计划

### Phase 1: 后端扩展

1. `file_service.py` 新增 `is_supported_subtitle()` 和 `list_input_subtitle_files()` 方法
2. `file_routes.py` 新增 `GET /api/files/subtitles` 端点
3. `project_routes.py` 新增 `POST /api/projects/import-local` 端点，提取辅助函数 `_read_local_file()`、`_parse_subtitle()`

### Phase 2: 前端 API 层

1. `fileApi.js` 新增 `listSubtitleFiles()`
2. `projectApi.js` 新增 `importProjectLocal()`

### Phase 3: ImportDialog 组件

1. 新建 `composables/task-list/useTaskImport.js` — 双模式导入逻辑
2. 新建 `components/task/ImportDialog.vue` — 双 Tab + 媒体区 + 项目名称 + footer

### Phase 4: 入口集成

1. `TaskListHeader.vue` — 新增"导入字幕"按钮 + `open-import` emit，条件隐藏"上传视频"
2. `TaskListView.vue` — 引入 ImportDialog，处理 `open-import` 事件，Lite 自动弹出
3. `TaskCardGrid.vue` — 空状态新增"导入字幕"按钮 + `open-import` emit

### Phase 5: Lite 统一与清理

1. `flavor.js` — `taskList: true`
2. `router/index.js` — 删除 Lite 重定向到 `/import` 的逻辑，`/import` 改为重定向到 `/tasks?action=import`
3. 废弃 `ImportView.vue`（确认无其他引用后可删除）
4. 编译测试 `npm run build`

---

## 11. 风险与注意事项

### 11.1 字幕文件编码

后端 `_decode_upload_content()` 已支持 UTF-8 / UTF-8-SIG / GB18030 三种编码自动检测。`import-local` 端点复用同一个解码函数，无额外风险。

### 11.2 硬链接兼容性

本地选择模式的媒体关联复用 `project_service._attach_video_asset()`，已有硬链接 → 复制的自动回退机制。跨文件系统场景已覆盖。

### 11.3 Lite 模式向后兼容

- `/import` 路由保留为重定向，不会出现 404
- `CAPABILITIES.canImportSubtitle` 始终为 true，不受 flavor 影响
- 编辑器页面不受影响（`/editor/project/{id}` 路由不变）

### 11.4 单选 radio 实现

el-table 没有内置 radio 列类型，使用 `highlight-current-row` + 自定义 radio 列实现。需要注意：
- 点击行任意位置即选中（`@current-change`）
- radio 的 `model-value` 绑定到选中的文件名
- 可通过再次点击同一行取消选中（媒体表格需要此能力，因为媒体是可选的）

### 11.5 与 TaskCreateDialog 的样式隔离

两个 dialog 共用 `.upload-dialog` 全局样式，但各自的 scoped 样式互不干扰。需要确保：
- ImportDialog 不引入 SettingsTabs 相关样式
- 两者的 el-tabs、el-table 样式保持视觉一致

---

## 12. 验收标准

1. "导入字幕"按钮在 Full 和 Lite 模式下均可见
2. "上传视频"按钮仅在 Full 模式下可见
3. Lite 模式进入 `/tasks` 后自动弹出 ImportDialog
4. `/import` 路由正确重定向到 `/tasks?action=import` 并弹出 ImportDialog
5. 直接上传模式：拖拽/点击选择字幕 -> 可选媒体 -> 导入成功跳转编辑器
6. 本地选择模式：字幕表格单选 -> 可选媒体单选 -> 导入成功跳转编辑器
7. 两种模式的项目名称均自动从字幕文件名提取
8. "从本地目录选择"Tab 右上角有"打开input目录"按钮
9. input 目录无字幕文件时显示空状态提示
10. Dialog 高度 <= 640px，不触发主界面滚动
11. 所有颜色使用 `--af-*` Token，无硬编码色值
12. `npm run build` 编译通过
