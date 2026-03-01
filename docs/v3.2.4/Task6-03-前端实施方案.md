# Task 6-03: 前端实施方案

> Type: Architecture | Status: Active
> Version: V3.2.4+dev.20260224.03
> 上游文档: `Task6-00-总纲`, `Task6-01-领域模型与存储设计`

## 1. 实施顺序

```
Step 1: vite.config.js（编译时注入）
Step 2: flavor.js（前端 flavor 单一真值）
Step 3: projectApi.js / legacyApi.js（API 客户端）
Step 4: router/index.js（双路由 + 导入入口）
Step 5: ImportView.vue（Lite 字幕导入页面）
Step 6: EditorView.vue（双入口 + Lite 适配）
Step 7: projectStore.js（projectId 支持）
Step 8: sseChannelManager.js（project 频道）
Step 9: 编辑器子组件适配
```

**前置条件**: Task 9（前端状态管理重构）完成后开始前端改造。

## 2. Step 1: vite.config.js

**文件**: `frontend/vite.config.js`
**当前行数**: 43
**修改量**: ~10 行

### 改动

```javascript
// 在 defineConfig 中新增 define 配置
export default defineConfig({
  // ... 现有配置

  define: {
    // 编译时常量注入，支持 tree-shaking
    '__APP_FLAVOR__': JSON.stringify(process.env.VITE_APP_FLAVOR || 'full'),
    '__IS_LITE__': process.env.VITE_LITE_MODE === 'true'
      || process.env.VITE_APP_FLAVOR === 'lite',
  },
})
```

### 说明

- `__APP_FLAVOR__` 和 `__IS_LITE__` 是编译时常量
- 生产构建时 Vite 会将 `if (__IS_LITE__)` 中的死代码分支完全移除
- 同时保留 `import.meta.env.VITE_LITE_MODE` 作为运行时访问方式

## 3. Step 2: flavor.js

**文件**: `frontend/src/config/flavor.js`（新增）
**估算行数**: ~30

```javascript
/**
 * 前端 flavor 单一真值
 *
 * 使用方式:
 *   import { IS_LITE, CAPABILITIES } from '@/config/flavor'
 *
 * 设计原则:
 *   - 此文件是 flavor 判断的唯一入口
 *   - 禁止在组件中直接读取 import.meta.env.VITE_LITE_MODE
 *   - 能力门控仅做 UI 体验收敛，后端为最终安全边界
 */

// 编译时常量（支持 tree-shaking）
export const FLAVOR = __APP_FLAVOR__ || 'full'
export const IS_LITE = __IS_LITE__ || false

// 能力门控映射
export const CAPABILITIES = Object.freeze({
  // 转录相关
  canTranscribe: !IS_LITE,
  canSeparateVocal: !IS_LITE,
  canSpectralTriage: !IS_LITE,
  canManageModels: !IS_LITE,

  // 编辑相关（始终可用）
  canImportSubtitle: true,
  canExportSubtitle: true,
  canEditSubtitle: true,
  canPreviewMedia: true,
})

// 路由可见性
export const ROUTE_VISIBILITY = Object.freeze({
  taskList: !IS_LITE,
  taskCreate: !IS_LITE,
  importPage: true,       // 始终可用
  editor: true,           // 始终可用
})
```

## 4. Step 3: API 客户端

### 4.1 projectApi.js

**文件**: `frontend/src/services/api/projectApi.js`（新增）
**估算行数**: ~150

```javascript
/**
 * 项目 API 客户端
 *
 * 所有新的项目相关请求通过此文件发出
 * 遵循 transcriptionApi.js 的风格约定
 */
import { apiClient } from './client'

const projectApi = {
  // --- 导入与创建 ---

  /**
   * 导入字幕文件创建项目
   * @param {File} subtitleFile - 字幕文件
   * @param {string} format - srt | ass | vtt
   * @param {string} [videoPath] - 可选视频路径
   * @param {string} [title] - 项目标题
   * @returns {Promise<{project_id, title, subtitle_count, media_assets}>}
   */
  async importProject(subtitleFile, format = 'srt', videoPath = null, title = null) {
    const formData = new FormData()
    formData.append('subtitle_file', subtitleFile)
    formData.append('format', format)
    if (videoPath) formData.append('video_path', videoPath)
    if (title) formData.append('title', title)

    const response = await apiClient.post('/api/projects/import', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  },

  // --- 查询 ---

  /**
   * 列出所有项目
   * @param {string} [flavor] - 过滤 flavor
   * @returns {Promise<Array>}
   */
  async listProjects(flavor = null) {
    const params = flavor ? { flavor } : {}
    const response = await apiClient.get('/api/projects/', { params })
    return response.data
  },

  /**
   * 获取单个项目
   * @param {string} projectId
   * @returns {Promise<Object>}
   */
  async getProject(projectId) {
    const response = await apiClient.get(`/api/projects/${projectId}`)
    return response.data
  },

  // --- 字幕操作 ---

  /**
   * 获取项目字幕列表
   * @param {string} projectId
   * @returns {Promise<{segments: Array, doc_meta: Object}>}
   */
  async getSubtitles(projectId) {
    const response = await apiClient.get(`/api/projects/${projectId}/subtitles`)
    return response.data
  },

  /**
   * 更新单条字幕
   * @param {string} projectId
   * @param {string} segmentId
   * @param {Object} update - {text?, start?, end?}
   * @returns {Promise<Object>}
   */
  async updateSubtitle(projectId, segmentId, update) {
    const response = await apiClient.put(
      `/api/projects/${projectId}/subtitles/${segmentId}`,
      update,
    )
    return response.data
  },

  /**
   * 新增字幕
   * @param {string} projectId
   * @param {Object} data - {text, start, end}
   * @returns {Promise<Object>}
   */
  async createSubtitle(projectId, data) {
    const response = await apiClient.post(
      `/api/projects/${projectId}/subtitles`,
      data,
    )
    return response.data
  },

  /**
   * 删除字幕
   * @param {string} projectId
   * @param {string} segmentId
   * @returns {Promise<Object>}
   */
  async deleteSubtitle(projectId, segmentId) {
    const response = await apiClient.delete(
      `/api/projects/${projectId}/subtitles/${segmentId}`,
    )
    return response.data
  },

  // --- 导出 ---

  /**
   * 导出字幕
   * @param {string} projectId
   * @param {string} format - srt | ass | vtt
   * @returns {Promise<Blob>}
   */
  async exportSubtitles(projectId, format = 'srt') {
    const response = await apiClient.get(
      `/api/projects/${projectId}/export`,
      { params: { format }, responseType: 'blob' },
    )
    return response.data
  },

  // --- 媒体 ---

  /**
   * 获取媒体资产 URL
   * @param {string} projectId
   * @param {string} assetType - video | audio | peaks
   * @returns {string}
   */
  getMediaUrl(projectId, assetType) {
    return `/api/projects/${projectId}/media/${assetType}`
  },
}

export default projectApi
```

### 4.2 legacyApi.js

**文件**: `frontend/src/services/api/legacyApi.js`（新增）
**估算行数**: ~50

```javascript
/**
 * 旧任务兼容 API 客户端
 */
import { apiClient } from './client'

const legacyApi = {
  /**
   * 解析旧任务 ID 为项目 ID
   * @param {string} jobId
   * @returns {Promise<{project_id, is_newly_migrated, redirect_url}>}
   */
  async resolveTask(jobId) {
    const response = await apiClient.get(`/api/legacy/tasks/${jobId}/resolve`)
    return response.data
  },
}

export default legacyApi
```

## 5. Step 4: router/index.js

**文件**: `frontend/src/router/index.js`
**当前行数**: 43
**修改量**: ~30 行（接近重写）

### 改造后完整结构

```javascript
import { createRouter, createWebHistory } from 'vue-router'
import { IS_LITE, ROUTE_VISIBILITY } from '@/config/flavor'

const routes = [
  // 根路由：Lite 模式去 /import，Full 模式去 /tasks
  {
    path: '/',
    redirect: IS_LITE ? '/import' : '/tasks',
  },

  // 任务列表（仅 Full 模式）
  {
    path: '/tasks',
    name: 'TaskList',
    component: () => import('@/views/TaskListView.vue'),
    beforeEnter: (to, from, next) => {
      if (!ROUTE_VISIBILITY.taskList) {
        next('/import')
      } else {
        next()
      }
    },
  },

  // 字幕导入页面（始终可用，Lite 模式的主入口）
  {
    path: '/import',
    name: 'Import',
    component: () => import('@/views/ImportView.vue'),
  },

  // 编辑器 - project_id 入口（新主路由）
  {
    path: '/editor/project/:projectId',
    name: 'ProjectEditor',
    component: () => import('@/views/EditorView.vue'),
    props: (route) => ({
      projectId: route.params.projectId,
      jobId: null,
    }),
  },

  // 编辑器 - job_id 入口（兼容旧路由）
  {
    path: '/editor/:jobId',
    name: 'Editor',
    component: () => import('@/views/EditorView.vue'),
    props: (route) => ({
      projectId: null,
      jobId: route.params.jobId,
    }),
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
```

### 说明

- `/editor/project/:projectId` 是新的主入口
- `/editor/:jobId` 保留向后兼容，进入后 EditorView 会 resolve 并重定向
- Lite 模式下 `/tasks` 会被守卫重定向到 `/import`
- 路由懒加载保持不变

## 6. Step 5: ImportView.vue

**文件**: `frontend/src/views/ImportView.vue`（新增）
**估算行数**: ~200

### 功能描述

Lite 模式的主入口页面，提供字幕文件导入功能。

```vue
<template>
  <div class="import-view">
    <!-- 页面标题 -->
    <header class="import-header">
      <h1>字幕编辑器</h1>
      <p>导入字幕文件开始编辑</p>
    </header>

    <!-- 导入区域 -->
    <div class="import-zone">
      <!-- 字幕文件选择 -->
      <div class="file-selector">
        <label>字幕文件</label>
        <input type="file" accept=".srt,.ass,.vtt" @change="onSubtitleFileChange" />
        <span class="hint">支持 SRT、ASS、VTT 格式</span>
      </div>

      <!-- 视频文件选择（可选） -->
      <div class="file-selector">
        <label>视频文件（可选）</label>
        <input type="file" accept="video/*" @change="onVideoFileChange" />
        <span class="hint">关联视频以启用波形和预览</span>
      </div>

      <!-- 项目名称 -->
      <div class="title-input">
        <label>项目名称</label>
        <input v-model="projectTitle" placeholder="我的字幕项目" />
      </div>

      <!-- 导入按钮 -->
      <button
        class="btn-import"
        :disabled="!subtitleFile || isImporting"
        @click="handleImport"
      >
        {{ isImporting ? '导入中...' : '开始导入' }}
      </button>
    </div>

    <!-- 已有项目列表 -->
    <div v-if="projects.length > 0" class="project-list">
      <h2>已有项目</h2>
      <div
        v-for="proj in projects"
        :key="proj.project_id"
        class="project-item"
        @click="openProject(proj.project_id)"
      >
        <span class="project-title">{{ proj.title }}</span>
        <span class="project-info">{{ proj.subtitle_doc?.segment_count }} 条字幕</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import projectApi from '@/services/api/projectApi'

const router = useRouter()

const subtitleFile = ref(null)
const videoFile = ref(null)
const projectTitle = ref('')
const isImporting = ref(false)
const projects = ref([])

onMounted(async () => {
  // 加载已有项目列表
  try {
    projects.value = await projectApi.listProjects()
  } catch (e) {
    console.warn('加载项目列表失败:', e)
  }
})

function onSubtitleFileChange(event) {
  subtitleFile.value = event.target.files[0]
  if (!projectTitle.value && subtitleFile.value) {
    // 自动从文件名提取标题
    projectTitle.value = subtitleFile.value.name.replace(/\.(srt|ass|vtt)$/i, '')
  }
}

function onVideoFileChange(event) {
  videoFile.value = event.target.files[0]
}

async function handleImport() {
  if (!subtitleFile.value) return
  isImporting.value = true

  try {
    // 检测格式
    const ext = subtitleFile.value.name.split('.').pop().toLowerCase()
    const format = ['srt', 'ass', 'vtt'].includes(ext) ? ext : 'srt'

    // 视频路径（Electron 环境下可获取本地路径）
    const videoPath = videoFile.value?.path || null

    const result = await projectApi.importProject(
      subtitleFile.value,
      format,
      videoPath,
      projectTitle.value || undefined,
    )

    // 跳转到编辑器
    router.push(`/editor/project/${result.project_id}`)
  } catch (error) {
    console.error('导入失败:', error)
    // 展示错误提示
  } finally {
    isImporting.value = false
  }
}

function openProject(projectId) {
  router.push(`/editor/project/${projectId}`)
}
</script>
```

### 样式说明

- 使用 Tailwind 类名处理布局（遵循 Modern Rules）
- 颜色使用语义变量（text-text-primary, bg-bg-base）
- 不使用 .el-xxx 覆盖

## 7. Step 6: EditorView.vue 适配

**文件**: `frontend/src/views/EditorView.vue`
**当前行数**: 2496
**修改量**: ~120 行

### 7.1 Props 改造

```javascript
// 原有（约 L259）
// jobId: { type: String, required: true }

// 改为
const props = defineProps({
  projectId: { type: String, default: null },
  jobId: { type: String, default: null },
})
```

### 7.2 初始化逻辑改造

在 `onMounted` 或 `setup` 顶部新增 resolve 逻辑：

```javascript
import { IS_LITE } from '@/config/flavor'
import legacyApi from '@/services/api/legacyApi'
import projectApi from '@/services/api/projectApi'

// 解析出的权威 ID
const resolvedProjectId = ref(null)
const resolvedJobId = ref(null)

async function resolveIdentity() {
  if (props.projectId) {
    // 新路由入口：project_id 已知
    resolvedProjectId.value = props.projectId
    // 尝试获取关联的 job_id（用于 SSE 兼容）
    try {
      const project = await projectApi.getProject(props.projectId)
      resolvedJobId.value = project.job_id || null
    } catch (e) {
      // project 可能没有关联 job_id（纯导入项目）
    }
  } else if (props.jobId) {
    // 旧路由入口：需要 resolve
    resolvedJobId.value = props.jobId
    try {
      const result = await legacyApi.resolveTask(props.jobId)
      resolvedProjectId.value = result.project_id
      // 重定向到新路由（不影响浏览器历史）
      router.replace(`/editor/project/${result.project_id}`)
    } catch (e) {
      // resolve 失败，退回直接使用 job_id
      resolvedProjectId.value = props.jobId
    }
  }
}
```

### 7.3 Lite 模式适配

```javascript
async function loadProject() {
  await resolveIdentity()

  if (IS_LITE || !resolvedJobId.value) {
    // Lite 模式或纯导入项目：直接加载字幕，跳过转录状态检查
    await loadProjectSubtitles()
    return
  }

  // Full 模式：保持现有逻辑
  const jobStatus = await transcriptionApi.getJobStatus(resolvedJobId.value, true)
  // ... 现有的状态分支处理
}

async function loadProjectSubtitles() {
  // 通过 projectApi 加载字幕
  const data = await projectApi.getSubtitles(resolvedProjectId.value)
  projectStore.loadFromProjectData(data)
}
```

### 7.4 SSE 订阅适配

```javascript
function subscribeSSE() {
  if (IS_LITE && !resolvedJobId.value) {
    // 纯编辑模式，不需要 SSE
    // 可选：订阅 project:{projectId} 频道接收编辑同步事件
    return
  }

  // 现有 SSE 订阅逻辑保持不变
  sseUnsubscribe = sseChannelManager.subscribeJob(resolvedJobId.value, {
    // ... 现有处理器
  })
}
```

### 7.5 修改范围控制

**不修改**的区域：
- 子组件模板（VideoStage, WaveformTimeline, SubtitleList 等）
- 播放控制逻辑
- 字幕编辑交互逻辑
- 导出功能

**仅修改**的区域：
- Props 定义（2 行）
- 初始化 resolve 逻辑（~40 行新增）
- loadProject 分支（~30 行）
- SSE 订阅条件（~10 行）
- 部分 computed 属性增加 IS_LITE 分支（~20 行）

## 8. Step 7: projectStore.js 适配

**文件**: `frontend/src/stores/projectStore.js`
**当前行数**: 1598
**修改量**: ~80 行

### 8.1 meta 扩展

```javascript
const meta = ref({
  jobId: null,      // 保留
  projectId: null,  // 新增：权威 ID
  title: '',
  mode: 'normal',   // normal | legacy
  flavor: 'full',   // full | lite
  // ... 其余不变
})
```

### 8.2 新增方法

```javascript
/**
 * 从 Project API 数据加载字幕
 * @param {Object} data - projectApi.getSubtitles() 的返回值
 */
function loadFromProjectData(data) {
  const { segments, doc_meta } = data

  // 设置 meta
  meta.value.projectId = doc_meta.project_id || meta.value.projectId

  // 转换字幕格式
  const newSubtitles = segments.map((seg, index) => ({
    index,
    segment_id: seg.segment_id,
    text: seg.text,
    start: seg.start,
    end: seg.end,
    original_text: seg.original_text || seg.text,
    is_modified: seg.is_modified || false,
    // 保持与现有格式兼容
  }))

  subtitles.value = newSubtitles
}

/**
 * 获取当前的权威 ID（优先 projectId）
 */
const primaryId = computed(() => meta.value.projectId || meta.value.jobId)
```

### 8.3 缓存键适配

```javascript
// 原有缓存键使用 jobId
// 改为优先使用 projectId
watch(
  [subtitles, meta],
  () => {
    const cacheKey = meta.value.projectId || meta.value.jobId
    if (!cacheKey) return

    memoryCache.set(cacheKey, {
      subtitles: toRaw(subtitles.value),
      meta: toRaw(meta.value),
    })
    // ... 其余不变
  },
  { deep: true },
)
```

## 9. Step 8: sseChannelManager.js

**文件**: `frontend/src/services/sseChannelManager.js`
**当前行数**: 780
**修改量**: ~30 行

### 新增方法

```javascript
/**
 * 订阅项目频道
 * @param {string} projectId
 * @param {Object} handlers - 事件处理器
 * @returns {Function} 取消订阅函数
 */
subscribeProject(projectId, handlers = {}) {
  const channelId = `project:${projectId}`
  const url = `${this.baseURL}/api/stream/project/${projectId}`

  return this._subscribe(channelId, url, {
    onSubtitleUpdated: handlers.onSubtitleUpdated,
    onSubtitleAdded: handlers.onSubtitleAdded,
    onSubtitleDeleted: handlers.onSubtitleDeleted,
    onMediaUpdated: handlers.onMediaUpdated,
  })
}
```

### 说明

- 这是一个简单的新增方法，不修改现有 `subscribeJob`
- 后端 SSE 需要对应的 `/api/stream/project/{projectId}` 端点（或复用现有端点通过参数区分）
- 初始阶段可不实现此方法，由 EditorView 根据是否有 jobId 决定使用哪种订阅

## 10. Step 9: 编辑器子组件适配

### 10.1 需要适配的组件

大部分子组件不需要修改，因为它们通过 props 或 projectStore 获取数据，不直接依赖 jobId。

| 组件 | 是否需要修改 | 原因 |
|------|:-----------:|------|
| `SubtitleList/index.vue` | 否 | 数据来源是 projectStore.subtitles |
| `SubtitleItem.vue` | 否 | 编辑通过 emit 向上传递 |
| `WaveformTimeline/index.vue` | 视情况 | 如果直接使用 jobId prop 获取 peaks/audio |
| `PlaybackControls/index.vue` | 否 | 操作 wavesurfer 实例 |
| `VideoStage` | 视情况 | 如果直接使用 jobId 构造视频 URL |
| `EditorHeader` | 否 | 标题来自 projectStore |

### 10.2 VideoStage 和 WaveformTimeline 适配

如果这些组件直接使用 `jobId` prop 构造媒体 URL：

```javascript
// 原有
const videoUrl = computed(() => `/api/media/${props.jobId}/video`)

// 改为（通过 projectStore 获取）
import { useProjectStore } from '@/stores/projectStore'
const projectStore = useProjectStore()
const primaryId = computed(() => projectStore.primaryId)
const videoUrl = computed(() => `/api/media/${primaryId.value}/video`)
```

或者由 EditorView 统一传递 `primaryId` prop，子组件只使用此 prop。

### 10.3 useSubtitleSync.js 适配

**文件**: `frontend/src/composables/useSubtitleSync.js`
**当前行数**: 256
**修改量**: ~10 行

```javascript
// 原有：接收 jobIdRef
export function useSubtitleSync(jobIdRef) {

// 改为：接收 identityRef（可以是 jobId 或 projectId）
export function useSubtitleSync(identityRef) {
  // 内部逻辑不变，identityRef 作为 API 调用的标识符
```

### 10.4 useWaveformCursorDrag.js 适配

**文件**: `frontend/src/composables/useWaveformCursorDrag.js`
**当前行数**: 426
**修改量**: ~15 行

```javascript
// 改造点 1：参数语义从 isVideoReady 升级为 isMediaReady（音频或视频可用）
const canEditTimeline = computed(() => isMediaReady.value)
if (!canEditTimeline.value) {
  console.warn('[WaveformCursorDrag] 时间轴不可编辑')
  return
}

// 改造点 2：导出统一时间换算函数，避免 WaveformTimeline 与右键菜单重复实现
return {
  getTimeFromClientX,
  handleUpperZonePointerDown,
  // ...
}
```

### 10.5 WaveformContextMenu / WaveformTimeline 契约统一

**文件**:
- `frontend/src/composables/useWaveformContextMenu.js`
- `frontend/src/components/editor/WaveformTimeline/index.vue`

```javascript
// WaveformTimeline 内部统一传入时间换算函数
onContextMenu(e, getTimeFromClientX)

// useWaveformContextMenu 增加参数类型防护
if (typeof getTimeFromClientX !== 'function') {
  return
}
```

### 10.6 useProxyVideo.js 适配

**文件**: `frontend/src/composables/useProxyVideo.js`
**当前行数**: 555
**修改量**: ~15 行

```javascript
import { IS_LITE } from '@/config/flavor'

export function useProxyVideo(identityRef) {
  // Lite 模式或无视频时，直接返回 ready 状态
  if (IS_LITE) {
    return {
      state: ref('IDLE'),
      isReady: ref(true),  // Lite 不以 proxy 为前置条件
      progress: ref(0),
      urls: ref({}),
      subscribe: () => {},
      unsubscribe: () => {},
    }
  }

  // Full 模式：现有逻辑完全保留
  // ...
}
```

### 10.7 旧实现清理收敛（project-first）

本阶段在不破坏旧任务兼容的前提下，做以下收敛：

1. **字幕写回优先级统一为 `projectId > jobId`**
   - 涉及文件：
     - `frontend/src/composables/useSubtitleSync.js`
     - `frontend/src/components/editor/SubtitleList/index.vue`
     - `frontend/src/components/editor/SubtitleList/SubtitleItem.vue`
     - `frontend/src/composables/useWaveformContextMenu.js`
   - 规则：
     - 只要存在 `projectId`，编辑写回统一走 `projectApi`
     - 仅在缺失 `projectId` 时回退 `transcriptionApi`

2. **媒体标识命名统一为 `mediaId`**
   - 涉及文件：
     - `frontend/src/components/editor/VideoStage/index.vue`
     - `frontend/src/components/editor/WaveformTimeline/index.vue`
     - `frontend/src/views/EditorView.vue`
   - 目的：
     - 避免 `jobId` 命名向组件内部泄漏旧语义
     - 明确媒体访问依赖的是“媒体身份”，可由 `projectId` 或兼容 `jobId` 提供

3. **移除 `useProxyVideo` 内部旧 SSE 订阅分支**
   - 文件：`frontend/src/composables/useProxyVideo.js`
   - 规则：
     - `useProxyVideo` 仅负责状态机、HTTP 刷新与事件处理器
     - SSE 连接统一由 `EditorView + sseChannelManager` 管理并转发到 handlers

4. **移除视频区“不可用鼠标样式”旧提示**
   - 文件：`frontend/src/components/editor/VideoStage/index.vue`
   - 行为：
     - 不再显示 `not-allowed` 光标
     - 纯音频黑屏占位保持与有视频时一致的点击播放/暂停交互

### 10.8 调试噪音清理（保留错误信号）

为保证 Task6 解耦路径可长期维护，编辑链路内只保留 `warn/error` 级别信号，移除无业务副作用的观测日志：

1. **Editor 主视图**
   - 文件：`frontend/src/views/EditorView.vue`
   - 处理：
     - 删除仅用于开发期观测的 `console.log`
     - 保留 `console.warn` / `console.error` 以支撑故障诊断

2. **媒体与播放链路**
   - 文件：
     - `frontend/src/components/editor/VideoStage/index.vue`
     - `frontend/src/composables/useProxyVideo.js`
   - 处理：
     - 删除转码状态/源切换过程中的调试输出
     - 保留错误上报与失败降级逻辑，确保纯音频与有视频路径行为一致

3. **字幕切分链路**
   - 文件：
     - `frontend/src/components/editor/SubtitleList/SubtitleItem.vue`
     - `frontend/src/composables/useWaveformContextMenu.js`
     - `frontend/src/composables/useSubtitleSync.js`
   - 处理：
     - 删除右键切分与同步过程的调试输出
     - 保留切分失败、同步失败等异常告警

## 11. 实施约束

### 11.1 Modern Rules 遵循

所有新增的 `.vue` 文件必须 100% 遵循 Modern Rules：
- 布局使用 Tailwind class（禁止 `<style>` 中写 margin/padding/flex）
- 颜色使用语义变量（禁止 Hex）
- 不使用 `.el-xxx` 覆盖

### 11.2 修改 Legacy 文件的规则

修改 `EditorView.vue` 等 Legacy 文件时遵循"Boy Scout Rule"：
- 仅修改本任务涉及的代码
- 修改行局部用 Tailwind 替换（如果安全）
- 不尝试全量重构文件

### 11.3 禁止散落的 isLite 判断

- 所有 flavor 判断必须通过 `@/config/flavor` 导入
- 禁止在组件中直接 `import.meta.env.VITE_LITE_MODE`
- 禁止在模板中内联 `v-if="isLite"` 散落判断（应收敛到 CAPABILITIES 或 ROUTE_VISIBILITY）
