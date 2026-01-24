<template>
  <div class="editor-view">
    <!-- 顶部导航栏 - 使用新的 EditorHeader 组件 -->
    <EditorHeader
      :job-id="jobId"
      :task-name="projectName"
      :current-task-status="taskStatus"
      :current-task-phase="taskPhase"
      :current-task-progress="taskProgress"
      :progress-detail="progressDetail"
      :queue-completed="queueCompleted"
      :queue-total="queueTotal"
      :can-undo="canUndo"
      :can-redo="canRedo"
      :active-tasks="activeTasks"
      :last-saved="lastSaved"
      :dual-stream-progress="dualStreamProgress"
      @undo="undo"
      @redo="redo"
      @pause="pauseTranscription"
      @resume="resumeTranscription"
      @cancel="cancelTranscription"
    />

    <!-- 加载状态 -->
    <div v-if="isLoading" class="loading-overlay">
      <div class="loading-spinner"></div>
      <span>加载项目中...</span>
    </div>

    <!-- 错误状态 -->
    <div v-else-if="loadError" class="error-overlay">
      <svg viewBox="0 0 24 24" fill="currentColor" class="error-icon">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
      </svg>
      <h3>加载失败</h3>
      <p>{{ loadError }}</p>
      <button class="retry-btn" @click="loadProject">重试</button>
      <router-link to="/tasks" class="back-link">返回任务列表</router-link>
    </div>

    <!-- 主编辑区域 - Grid 布局 -->
    <main v-else class="workspace-grid" :style="gridStyle">
      <!-- 左侧舞台列 -->
      <section class="stage-column">
        <!-- 视频区域 -->
        <div class="video-wrapper">
          <VideoStage
            ref="videoStageRef"
            :job-id="jobId"
            :show-subtitle="true"
            :enable-keyboard="false"
            :progressive-url="proxyVideo.currentUrl.value"
            :current-resolution="proxyVideo.currentResolution.value"
            :is-upgrading="proxyVideo.isUpgrading.value"
            :upgrade-progress="proxyVideo.progress.value"
            :proxy-state="proxyVideo.state.value"
            :proxy-error="proxyVideo.error.value"
            :auto-trigger-720p="proxyVideo.autoTrigger720p.value"
            @loaded="handleVideoLoaded"
            @error="handleVideoError"
            @check-status="handleCheckVideoStatus"
            @resolution-change="handleResolutionChange"
            @retry="proxyVideo.retry"
            @upgrade-started="handleUpgradeStarted"
            @upgrade-failed="handleUpgradeFailed"
          />
        </div>

        <!-- 播放控制条 - 底座模式 -->
        <div class="controls-wrapper">
          <PlaybackControls :pedestal="true" />
        </div>

        <!-- 波形时间轴 -->
        <div class="waveform-wrapper">
          <WaveformTimeline
            ref="waveformRef"
            :job-id="jobId"
            @ready="handleWaveformReady"
            @region-update="handleRegionUpdate"
            @region-click="handleRegionClick"
          />
        </div>
      </section>

      <!-- 可拖拽分隔条 -->
      <div class="resizer" @mousedown="startResize" :class="{ active: isResizing }"></div>

      <!-- 右侧边栏 -->
      <aside class="sidebar-column">
        <!-- 标签页导航 -->
        <div class="tab-nav">
          <button
            class="tab-btn"
            :class="{ active: activeTab === 'subtitles' }"
            @click="activeTab = 'subtitles'"
          >
            字幕列表
          </button>
          <button
            class="tab-btn"
            :class="{ active: activeTab === 'validation' }"
            @click="activeTab = 'validation'"
          >
            问题检查
            <span v-if="errorCount > 0" class="badge">{{ errorCount }}</span>
          </button>
          <button
            class="tab-btn"
            :class="{ active: activeTab === 'assistant' }"
            @click="activeTab = 'assistant'"
          >
            AI 助手
          </button>
        </div>

        <!-- 标签页内容 -->
        <div class="tab-content">
          <div v-show="activeTab === 'subtitles'" class="tab-pane">
            <SubtitleList
              :auto-scroll="true"
              :editable="true"
              @subtitle-click="handleSubtitleClick"
              @subtitle-edit="handleSubtitleEdit"
            />
          </div>

          <div v-show="activeTab === 'validation'" class="tab-pane">
            <div class="placeholder-panel">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
              </svg>
              <h3>问题检查</h3>
              <p>即将实现</p>
            </div>
          </div>

          <div v-show="activeTab === 'assistant'" class="tab-pane">
            <div class="placeholder-panel">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M21 10.5h-6.5V4h-5v6.5H3v5h6.5V22h5v-6.5H21v-5z"/>
              </svg>
              <h3>AI 助手</h3>
              <p>智能字幕优化功能</p>
              <p class="coming-soon">即将推出</p>
            </div>
          </div>
        </div>
      </aside>
    </main>

    <!-- 底部状态栏 -->
    <footer class="editor-footer">
      <div class="footer-left">
        <span>{{ totalSubtitles }} 条字幕</span>
        <span v-if="currentSubtitle" class="divider">|</span>
        <span v-if="currentSubtitle">当前: #{{ currentSubtitleIndex + 1 }}</span>
      </div>

      <div class="footer-center">
        <span v-if="lastSaved" class="save-time">
          <svg class="icon" viewBox="0 0 24 24" fill="currentColor">
            <path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z"/>
          </svg>
          自动保存于 {{ formatLastSaved(lastSaved) }}
        </span>
      </div>

      <div class="footer-right">
        <span v-if="errorCount > 0" class="error-indicator" @click="activeTab = 'validation'">
          {{ errorCount }} 个问题
        </span>
        <span class="divider">|</span>
        <button class="settings-btn" @click="showSettings" title="设置">
          <svg viewBox="0 0 24 24" fill="currentColor">
            <path d="M19.14,12.94c0.04-0.3,0.06-0.61,0.06-0.94c0-0.32-0.02-0.64-0.07-0.94l2.03-1.58c0.18-0.14,0.23-0.41,0.12-0.61 l-1.92-3.32c-0.12-0.22-0.37-0.29-0.59-0.22l-2.39,0.96c-0.5-0.38-1.03-0.7-1.62-0.94L14.4,2.81c-0.04-0.24-0.24-0.41-0.48-0.41 h-3.84c-0.24,0-0.43,0.17-0.47,0.41L9.25,5.35C8.66,5.59,8.12,5.92,7.63,6.29L5.24,5.33c-0.22-0.08-0.47,0-0.59,0.22L2.74,8.87 C2.62,9.08,2.66,9.34,2.86,9.48l2.03,1.58C4.84,11.36,4.8,11.69,4.8,12s0.02,0.64,0.07,0.94l-2.03,1.58 c-0.18,0.14-0.23,0.41-0.12,0.61l1.92,3.32c0.12,0.22,0.37,0.29,0.59,0.22l2.39-0.96c0.5,0.38,1.03,0.7,1.62,0.94l0.36,2.54 c0.05,0.24,0.24,0.41,0.48,0.41h3.84c0.24,0,0.44-0.17,0.47-0.41l0.36-2.54c0.59-0.24,1.13-0.56,1.62-0.94l2.39,0.96 c0.22,0.08,0.47,0,0.59-0.22l1.92-3.32c0.12-0.22,0.07-0.47-0.12-0.61L19.14,12.94z M12,15.6c-1.98,0-3.6-1.62-3.6-3.6 s1.62-3.6,3.6-3.6s3.6,1.62,3.6,3.6S13.98,15.6,12,15.6z"/>
          </svg>
        </button>
      </div>
    </footer>
  </div>
</template>

<script setup>
/**
 * EditorView - 编辑器主视图
 *
 * 采用现代 NLE 软件的布局逻辑：
 * - Grid 布局实现响应式主工作区
 * - 视频优先，波形图退居辅助
 * - 可拖拽调整侧边栏宽度
 * - 智能进度显示和多任务管理
 */
import { ref, computed, onMounted, onUnmounted, provide, watch, toRef } from 'vue'
import { useRouter, onBeforeRouteLeave } from 'vue-router'
import { useProjectStore } from '@/stores/projectStore'
import { useUnifiedTaskStore } from '@/stores/unifiedTaskStore'
import { useProgressStore } from '@/stores/progressStore'
import { mediaApi, transcriptionApi } from '@/services/api'
import sseChannelManager from '@/services/sseChannelManager'
import { useShortcuts } from '@/hooks/useShortcuts'
import { useProxyVideo } from '@/composables/useProxyVideo'
import { useSubtitleSync } from '@/composables'
import { usePlaybackManager } from '@/services/PlaybackManager'
import { repairSubtitleOverlaps } from '@/utils/subtitleUtils'
import { ElMessage } from 'element-plus'

// 组件导入
import EditorHeader from '@/components/editor/EditorHeader.vue'
import PlaybackControls from '@/components/editor/PlaybackControls/index.vue'
import VideoStage from '@/components/editor/VideoStage/index.vue'
import SubtitleList from '@/components/editor/SubtitleList/index.vue'
import WaveformTimeline from '@/components/editor/WaveformTimeline/index.vue'

// Props
const props = defineProps({
  jobId: { type: String, required: true }
})

// Router & Stores
const router = useRouter()
const projectStore = useProjectStore()
const taskStore = useUnifiedTaskStore()

// 全局播放管理器
const playbackManager = usePlaybackManager()

// 创建响应式 jobId ref，用于监听路由变化
const jobIdRef = toRef(props, 'jobId')
const { forceSyncNow } = useSubtitleSync(jobIdRef)

// Proxy 视频加载状态（新重构版本）
const proxyVideo = useProxyVideo(jobIdRef)

// Refs
const videoStageRef = ref(null)
const waveformRef = ref(null)
const activeTab = ref('subtitles')
const saving = ref(false)
const lastSaved = ref(null)

// 布局状态
const sidebarWidth = ref(350)
const isResizing = ref(false)

// 加载状态
const isLoading = ref(true)
const loadError = ref(null)

// 统一进度状态
const progressStore = useProgressStore()
// 修复：直接使用 getRawState 获取响应式状态对象，避免 computed 嵌套导致响应式丢失
const jobProgress = computed(() => progressStore.getRawState(jobIdRef.value))
const taskStatus = computed(() => jobProgress.value.status || 'idle')
const taskPhase = computed(() => jobProgress.value.phase || 'pending')
const taskProgress = computed(() => jobProgress.value.percent || 0)
const progressDetail = computed(() => jobProgress.value.detail || null)
const isTranscribing = computed(() =>
  ['processing', 'queued'].includes(taskStatus.value)
)
const dualStreamProgress = computed(() => {
  const ds = jobProgress.value.dualStream
  if (ds && ds.mode !== 'unknown') {
    return ds
  }
  return projectStore.dualStreamProgress
})
let sseUnsubscribe = null
let progressPollTimer = null
let proxyPollTimer = null

// Provide 编辑器上下文
// ========== Provide 上下文 ==========

// 提供编辑器上下文给子组件
provide('editorContext', {
  jobId: jobIdRef,
  saving,
  // 视频就绪状态（用于播放控制拦截）
  isVideoReady: computed(() => proxyVideo.isReady.value)
})

// 调试：监听 proxyVideo 状态变化
watch(() => proxyVideo.currentUrl.value, (newUrl, oldUrl) => {
  console.log('[EditorView] proxyVideo.currentUrl 变化:', {
    oldUrl,
    newUrl,
    state: proxyVideo.state.value,
    isReady: proxyVideo.isReady.value,
    urls: {
      preview360p: proxyVideo.urls.value.preview360p,
      proxy720p: proxyVideo.urls.value.proxy720p,
      source: proxyVideo.urls.value.source
    }
  })
})

watch(() => proxyVideo.state.value, (newState, oldState) => {
  console.log('[EditorView] proxyVideo.state 变化:', {
    oldState,
    newState,
    currentUrl: proxyVideo.currentUrl.value
  })
})

// 将 projectStore 的估算双流进度同步到统一状态，供 UI 兜底展示
watch(
  () => projectStore.dualStreamProgress,
  (progress) => {
    if (!progress) return
    progressStore.applyDualStreamEstimate(jobIdRef.value, progress)
  },
  { deep: true }
)

// 监听 jobId 变化，重新加载项目（支持任务切换）
watch(jobIdRef, async (newJobId, oldJobId) => {
  if (newJobId === oldJobId) return

  console.log('[EditorView] jobId 变化，重新加载项目:', { oldJobId, newJobId })

  // 取消旧的 SSE 订阅
  cleanupSSE()

  // 重置项目状态
  projectStore.resetProject()

  // 重新加载项目
  await loadProject()
})

// 监听保存时间用于同步 task-meta 的显示
watch(
  () => projectStore.meta.lastSaved,
  (value) => {
    if (!projectStore.meta.jobId) {
      lastSaved.value = null
      return
    }
    lastSaved.value = value || null
  },
  { immediate: true }
)

// ========== 计算属性 ==========

// 项目名称 - 优先显示 title，否则显示 filename（去除扩展名）
// 【修复】过滤掉看起来像 UUID/16进制 的名称
const projectName = computed(() => {
  // 检查是否看起来像 UUID（16进制字符串）
  // 匹配: 纯16进制字符串（8-36位），或带连字符的UUID格式
  const isUuidLike = (str) => {
    if (!str) return true
    const cleaned = str.replace(/-/g, '')
    // 必须是纯16进制字符，且长度在8-36位之间
    return /^[0-9a-f]{8,36}$/i.test(cleaned)
  }

  // 从 filename 提取不带扩展名的名称
  const getDisplayName = (filename) => {
    if (!filename) return null
    const lastDotIndex = filename.lastIndexOf('.')
    return lastDotIndex > 0 ? filename.substring(0, lastDotIndex) : filename
  }

  // 1. 优先使用用户自定义的 title（非UUID）
  if (projectStore.meta.title && !isUuidLike(projectStore.meta.title)) {
    return projectStore.meta.title
  }

  // 2. 使用 filename（去除扩展名，非UUID）
  const filename = projectStore.meta.filename
  const displayName = getDisplayName(filename)
  if (displayName && !isUuidLike(displayName)) {
    return displayName
  }

  // 3. 尝试从 taskStore 获取（优先 title，其次 filename）
  const task = taskStore.tasks.find(t => t.job_id === props.jobId)
  if (task) {
    // 优先使用 task.title
    if (task.title && !isUuidLike(task.title)) {
      return task.title
    }
    // 其次使用 task.filename
    const taskDisplayName = getDisplayName(task.filename)
    if (taskDisplayName && !isUuidLike(taskDisplayName)) {
      return taskDisplayName
    }
  }

  // 4. 最后尝试使用 filename 本身（即使看起来像UUID，也比"未命名项目"好）
  // 但如果 filename 就是 job_id（完全匹配），则显示"未命名项目"
  if (displayName && displayName !== props.jobId && displayName !== props.jobId.replace(/-/g, '')) {
    return displayName
  }

  return '未命名项目'
})

// 基础状态
const isDirty = computed(() => projectStore.isDirty)
const totalSubtitles = computed(() => projectStore.totalSubtitles)
const currentSubtitle = computed(() => projectStore.currentSubtitle)
const currentSubtitleIndex = computed(() =>
  currentSubtitle.value
    ? projectStore.subtitles.findIndex(s => s.id === currentSubtitle.value.id)
    : -1
)

// 撤销/重做
const canUndo = computed(() => projectStore.canUndo)
const canRedo = computed(() => projectStore.canRedo)

// 队列进度计算
const queueCompleted = computed(() =>
  taskStore.tasks.filter(t => t.status === 'finished').length
)
const queueTotal = computed(() =>
  taskStore.tasks.filter(t => t.status !== 'created').length
)
const activeTasks = computed(() =>
  taskStore.tasks.filter(t => ['processing', 'queued'].includes(t.status)).length
)

// 问题检查计数（统计有警告的字幕数量）
const errorCount = computed(() =>
  projectStore.subtitles.filter(s => s.warning_type && s.warning_type !== 'none').length
)

// Grid 布局样式
const gridStyle = computed(() => ({
  gridTemplateColumns: `1fr 4px ${sidebarWidth.value}px`
}))

// ========== 数据加载 ==========

// 加载项目数据
async function loadProject() {
  isLoading.value = true
  loadError.value = null

  // 先重置项目状态，确保不同任务数据隔离
  projectStore.resetProject()

  try {
    // 1. 获取任务状态
    console.log('[EditorView] 获取任务状态:', props.jobId)
    const jobStatus = await transcriptionApi.getJobStatus(props.jobId, true)
    console.log('[EditorView] 任务状态:', jobStatus)

    progressStore.applySnapshot(
      props.jobId,
      {
        percent: jobStatus.progress,
        status: jobStatus.status,
        phase: jobStatus.phase,
        phase_percent: jobStatus.phase_percent,
        message: jobStatus.message,
        processed: jobStatus.processed,
        total: jobStatus.total
      },
      'http_init'
    )

    // 设置元数据
    projectStore.meta.jobId = props.jobId
    projectStore.meta.filename = jobStatus.filename || '未知文件'
    projectStore.meta.title = jobStatus.title || ''
    projectStore.meta.videoPath = mediaApi.getVideoUrl(props.jobId)
    projectStore.meta.audioPath = mediaApi.getAudioUrl(props.jobId)
    projectStore.meta.duration = jobStatus.media_status?.video?.duration || 0

    // 先尝试同步 Proxy 状态（便于决定是否需要订阅 SSE）
    try {
      await proxyVideo.refresh()
    } catch (e) {
      console.warn('[EditorView] 刷新 Proxy 状态失败（初始阶段，忽略）:', e)
    }

    // 2. 尝试从本地存储恢复
    // V3.1.2+dev.20260112.01: 恢复策略优化
    // 优先级: memoryCache → SmartSaver → segments API → SRT fallback
    const restored = await projectStore.restoreProject(props.jobId)
    const hasLocalRestore = restored && projectStore.subtitles.length > 0
    if (hasLocalRestore) {
      console.log('[EditorView] 从本地存储恢复成功')

      // V3.1.2: 检查缓存数据格式完整性
      // 必须有 sentenceIndex 字段才能支持 SSE 匹配
      const hasValidFormat = projectStore.subtitles.every(s => s.sentenceIndex !== undefined)
      if (!hasValidFormat) {
        // 缓存数据是旧格式，需要重新从 API 加载以获取 sentenceIndex
        console.log('[EditorView] 缓存数据格式过旧（缺少 sentenceIndex），重新从 API 加载')
        await loadTranscribingSegments()
      }
      // V3.2.0+dev.20260124.02: 后端为唯一真理，后续仍会从 API 刷新覆盖
    }

    // 3. 根据任务状态从后端加载字幕数据
    // V3.1.2+dev.20260111.02: 优先从 segments API 加载（含置信度），fallback 到 SRT
    if (jobStatus.status === 'finished') {
      await loadTranscribingSegments()
      // 如果 segments API 无数据，fallback 到 SRT
      if (projectStore.subtitles.length === 0) {
        console.log('[EditorView] segments API 无数据，尝试从 SRT 加载')
        await loadFromSRT()
      }
      // 转录已完成，但可能仍在转码，若未就绪则订阅 SSE + 轮询
      if (!proxyVideo.isReady.value) {
        subscribeSSE()
        startProxyPolling()
      }
    } else if (['processing', 'queued'].includes(jobStatus.status)) {
      await loadTranscribingSegments()
      // 订阅SSE获取实时更新
      subscribeSSE()
      startProgressPolling()
      startProxyPolling()
    } else if (jobStatus.status === 'paused') {
      // 暂停状态：加载已有的 segments，并订阅 SSE 以便接收恢复信号
      await loadTranscribingSegments()
      // 订阅SSE，以便用户点击恢复后能收到状态变更
      subscribeSSE()
      // V3.1.0: 暂停状态下立即刷新一次进度，不等待 SSE 连接
      refreshTaskProgress()
      startProxyPolling()
    } else if (jobStatus.status === 'created') {
      // 任务刚创建，订阅SSE等待开始
      subscribeSSE()
      startProxyPolling()
    } else if (jobStatus.status === 'failed') {
      await loadTranscribingSegments()
    }

    console.log('[EditorView] 项目加载完成')
  } catch (error) {
    console.error('[EditorView] 加载项目失败:', error)

    if (error.response?.status === 404) {
      console.warn(`[EditorView] 任务已在后端删除: ${props.jobId}`)
      try {
        await taskStore.deleteTask(props.jobId)
        loadError.value = '任务不存在（已被删除），本地记录已清理'
        setTimeout(() => router.push('/tasks'), 2000)
      } catch (deleteError) {
        loadError.value = '任务不存在，且清理本地记录失败，请刷新页面'
      }
    } else {
      loadError.value = error.message || '加载失败'
    }
  } finally {
    isLoading.value = false
  }
}

// 加载转录中的 segments
async function loadTranscribingSegments() {
  try {
    const textData = await transcriptionApi.getTranscriptionText(props.jobId)
    if (textData.segments && textData.segments.length > 0) {
      // 直接导入 segments，保留 sentenceIndex 字段
      projectStore.importSegments(textData.segments, {
        jobId: props.jobId,
        filename: projectStore.meta.filename,
        duration: projectStore.meta.duration,
        videoPath: projectStore.meta.videoPath,
        audioPath: projectStore.meta.audioPath
      })
      progressStore.applySnapshot(
        props.jobId,
        {
          percent: textData.progress?.percentage,
          phase: taskPhase.value,
          status: taskStatus.value,
          phase_percent: textData.progress?.phase_percent
        },
        'http_segments'
      )
    }
  } catch (error) {
    console.warn('[EditorView] 加载转录文字失败:', error)
  }
}

// V3.1.2: 从 SRT 文件加载（fallback，无置信度数据）
async function loadFromSRT() {
  try {
    const srtData = await mediaApi.getSRTContent(props.jobId)
    if (srtData.content) {
      projectStore.importSRT(srtData.content, {
        jobId: props.jobId,
        filename: srtData.filename || projectStore.meta.filename,
        duration: projectStore.meta.duration,
        videoPath: projectStore.meta.videoPath,
        audioPath: projectStore.meta.audioPath
      })
      console.log('[EditorView] 从 SRT 文件恢复成功（无置信度数据）')
    }
  } catch (error) {
    console.warn('[EditorView] 从 SRT 加载失败:', error)
  }
}

// Segments 转 SRT 格式
function segmentsToSRT(segments) {
  if (!segments || segments.length === 0) return ''
  return segments.map((seg, idx) => {
    const start = formatSRTTime(seg.start)
    const end = formatSRTTime(seg.end)
    return `${idx + 1}\n${start} --> ${end}\n${seg.text || ''}\n`
  }).join('\n')
}

// SRT 时间格式化
function formatSRTTime(seconds) {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  const ms = Math.round((seconds % 1) * 1000)
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')},${ms.toString().padStart(3, '0')}`
}

// ========== SSE 实时更新 ==========

function subscribeSSE() {
  // 如果已经订阅，先取消避免重复订阅
  if (sseUnsubscribe) {
    console.log('[EditorView] 已存在SSE订阅，先取消')
    sseUnsubscribe()
    sseUnsubscribe = null
  }

  console.log('[EditorView] 订阅任务 SSE:', props.jobId)

  // 使用 sseChannelManager 订阅单任务频道
  sseUnsubscribe = sseChannelManager.subscribeJob(props.jobId, {
    onInitialState(data) {
      console.log('[EditorView] SSE 初始状态:', data)
      progressStore.applySnapshot(props.jobId, data, 'sse_init')
      // 如果附带 Proxy 状态，直接应用到 proxyVideo，避免断线后进度丢失
      if (data.proxy) {
        console.log('[EditorView] 同步初始 Proxy 状态:', data.proxy)
        proxyVideo.applySnapshot(data.proxy)
      }
    },

    // 新增：被服务器踢出时提示用户并停止当前窗口的 SSE
    force_disconnect(data) {
      const reason = data?.reason || '该任务已在其他窗口打开，当前窗口的 SSE 已断开'
      console.warn('[EditorView] 收到强制断开:', reason)
      taskStore.updateTaskSSEStatus(props.jobId, false, reason)
      cleanupSSE()
      // 简单弹窗提示用户（后续可替换为全局提示组件）
      alert(reason)
    },

    onProgress(data) {
      console.log('[EditorView] SSE 进度更新:', data.percent, data.phase, data.detail)
      console.log('[EditorView] SSE status:', data.status, '| taskStatus:', taskStatus.value)
      progressStore.applySseProgress(props.jobId, data)
      if (data.detail) {
        projectStore.updateDualStreamProgressFromSSE({
          fastStream: Math.round(data.detail.fast || 0),
          slowStream: Math.round(data.detail.slow || 0),
          totalChunks: data.total || projectStore.dualStreamProgress.totalChunks
        })
      }
    },

    async onComplete(data) {
      console.log('[EditorView] 任务完成:', data)
      progressStore.markStatus(props.jobId, 'finished', { percent: 100, phase: 'complete' })
      taskStore.updateTaskStatus(props.jobId, 'finished')

      // V3.1.2+dev.20260111.02: 任务完成处理
      // 优先保留 SSE 推送的字幕（含置信度），无数据时从 API 加载
      if (projectStore.subtitles.length === 0) {
        console.log('[EditorView] 无字幕数据（可能断线），尝试恢复...')
        // 先尝试从 segments API 加载（含置信度）
        await loadTranscribingSegments()

        // 如果还是没有数据，从 SRT 文件加载（无置信度，但至少有字幕）
        if (projectStore.subtitles.length === 0) {
          console.log('[EditorView] segments API 无数据，尝试从 SRT 恢复')
          await loadFromSRT()
        }
      } else {
        console.log(`[EditorView] 保留 SSE 推送的 ${projectStore.subtitles.length} 条字幕（含置信度）`)
        // 将所有草稿标记为定稿
        projectStore.subtitles.forEach(sub => {
          if (sub.isDraft) {
            sub.isDraft = false
          }
        })
      }

      stopProgressPolling()
      startProxyPolling()
      // V3.1.2+dev.20260113.04: 不关闭SSE连接，保持接收Proxy转码事件
      // 转录任务完成后，720p转码可能还在进行，需要继续接收 proxy_complete 事件
      // cleanupSSE()
    },

    onFailed(data) {
      console.log('[EditorView] 任务失败:', data)
      progressStore.markStatus(props.jobId, 'failed', { message: data.message })
      taskStore.updateTaskStatus(props.jobId, 'failed')
      taskStore.updateTaskSSEStatus(props.jobId, true, data.message || '转录失败')
      stopProgressPolling()
      // 任务失败后关闭SSE连接
      cleanupSSE()
    },

    onPaused(data) {
      console.log('[EditorView] 任务已暂停:', data)
      progressStore.markStatus(props.jobId, 'paused')
      taskStore.updateTaskStatus(props.jobId, 'paused')
      // 保持SSE连接和进度显示
    },

    onCanceled(data) {
      console.log('[EditorView] 任务已取消:', data)
      progressStore.markStatus(props.jobId, 'canceled')
      taskStore.updateTaskStatus(props.jobId, 'canceled')
      stopProgressPolling()
      cleanupSSE()
    },

    onResumed(data) {
      // 新增：处理任务恢复信号
      console.log('[EditorView] 任务已恢复:', data)
      progressStore.markStatus(props.jobId, data.status || 'queued')
      taskStore.updateTaskStatus(props.jobId, data.status || 'queued')
      startProgressPolling()
    },

    onConnected() {
      console.log('[EditorView] SSE 连接成功')
      taskStore.updateTaskSSEStatus(props.jobId, true)
      taskStore.updateSSEHeartbeat()
      // 连接成功后，主动刷新一次进度状态
      refreshTaskProgress()
      // 同步刷新 Proxy/预览状态，兜底断线期间的转码进度
      proxyVideo.refresh()
      startProxyPolling()
    },

    onPing() {
      // 心跳同步到全局，避免误判超时
      taskStore.updateSSEHeartbeat()
    },

    // V3.1.2+dev.20260113.01: 恢复 onSubtitleUpdate 回调
    // 原因：专用处理器（onDraft等）处理新架构事件，onSubtitleUpdate 处理旧架构事件
    // 两者互补，不会冲突
    onSubtitleUpdate(data) {
      // 只处理旧架构的字幕事件（sv_sentence, whisper_patch, llm_proof 等）
      // 新架构事件（draft, replace_chunk）由专用处理器处理
      if (data.sentence || data.sentence_index !== undefined) {
        console.log('[EditorView] 收到旧架构字幕更新:', data)
        handleStreamingSubtitle(data)
      }
    },

    // Phase 5: 草稿字幕事件（快流/SenseVoice）
    onDraft(data) {
      console.log('[EditorView] 收到草稿字幕:', data)
      handleDraftSubtitle(data)
    },

    // Phase 5: 替换 Chunk 事件（慢流/Whisper）
    onReplaceChunk(data) {
      console.log('[EditorView] 收到替换 Chunk:', data)
      handleReplaceChunk(data)
    },

    // V3.1.0: 恢复字幕事件（断点续传后恢复）
    onRestored(data) {
      console.log('[EditorView] 收到恢复字幕:', data)
      handleRestoredChunk(data)
    },

    // V3.5: 极速模式定稿事件
    onFinalized(data) {
      console.log('[EditorView] 收到定稿字幕:', data)
      handleFinalizedSubtitle(data)
    },

    onSubtitleAdded(data) {
      console.log('[EditorView] 收到新增字幕:', data)
      handleStreamingSubtitle(data)
    },

    onSubtitleDeleted(data) {
      console.log('[EditorView] 收到删除字幕:', data)
      handleSubtitleDeleted(data)
    },

    // 新增：BGM 检测事件
    onBgmDetected(data) {
      console.log('[EditorView] BGM 检测结果:', data)
    },

    // 新增：分离策略事件
    onSeparationStrategy(data) {
      console.log('[EditorView] 分离策略决策:', data)
    },

    // 新增：模型升级事件
    onModelUpgrade(data) {
      console.log('[EditorView] 模型升级:', data)
    },

    // 新增：熔断事件
    onCircuitBreaker(data) {
      console.log('[EditorView] 熔断触发:', data)
    },

    // === Proxy 视频转码事件（转发到 useProxyVideo）===
    onAnalyzeComplete(data) {
      console.log('[EditorView] 转发 analyze_complete 到 useProxyVideo')
      proxyVideo.handlers.onAnalyzeComplete(data)
    },
    onRemuxProgress(data) {
      console.log('[EditorView] 转发 remux_progress 到 useProxyVideo')
      proxyVideo.handlers.onRemuxProgress(data)
    },
    onRemuxComplete(data) {
      console.log('[EditorView] 转发 remux_complete 到 useProxyVideo')
      proxyVideo.handlers.onRemuxComplete(data)
    },
    onPreview360pProgress(data) {
      console.log('[EditorView] 转发 preview_360p_progress 到 useProxyVideo')
      proxyVideo.handlers.onPreview360pProgress(data)
    },
    onPreview360pComplete(data) {
      console.log('[EditorView] 转发 preview_360p_complete 到 useProxyVideo')
      proxyVideo.handlers.onPreview360pComplete(data)
    },
    onProxyProgress(data) {
      console.log('[EditorView] 转发 proxy_progress 到 useProxyVideo')
      proxyVideo.handlers.onProxyProgress(data)
    },
    onProxyComplete(data) {
      console.log('[EditorView] 转发 proxy_complete 到 useProxyVideo')
      proxyVideo.handlers.onProxyComplete(data)
    },
    onProxyError(data) {
      console.log('[EditorView] 转发 proxy_error 到 useProxyVideo')
      proxyVideo.handlers.onProxyError(data)
    }
  })
}

// 清理SSE连接
function cleanupSSE() {
  if (sseUnsubscribe) {
    console.log('[EditorView] 清理SSE连接:', props.jobId)
    sseUnsubscribe()
    sseUnsubscribe = null
  }
}

// 处理流式字幕更新（旧版兼容，已弃用）
// V3.1.2+dev.20260112.01: 此函数已弃用，保留仅供旧版 SSE 事件兼容
// 新架构使用专用处理器：handleDraftSubtitle、handleReplaceChunk 等
// eslint-disable-next-line no-unused-vars
function handleStreamingSubtitle(data) {
  if (!data) return

  // 解析字幕数据格式
  // 兼容两种格式: 直接字段(sv_sentence) 和 嵌套sentence对象(whisper_patch/llm_proof等)
  const sentence = data.sentence || {}
  const sentenceIndex = data.sentence_index ?? data.index ?? sentence.index
  const text = sentence.text ?? data.text ?? data.content
  const start = sentence.start ?? data.start_time ?? data.start
  const end = sentence.end ?? data.end_time ?? data.end
  const warningType = sentence.warning_type ?? data.warning_type ?? 'none'
  const source = data.source ?? data.event_type ?? 'unknown'
  // V3.1.2+dev.20260111.01: 提取 display_confidence 和 confidence_source
  const confidence = sentence.confidence ?? data.confidence
  const displayConfidence = sentence.display_confidence ?? data.display_confidence
  const confidenceSource = sentence.confidence_source ?? data.confidence_source

  if (sentenceIndex === undefined || !text) {
    console.warn('[EditorView] 无效的字幕数据:', data)
    return
  }

  if (projectStore.isSentenceDeleted(sentenceIndex)) {
    console.log(`[EditorView] 字幕已被用户删除，忽略 SSE: ${sentenceIndex}`)
    return
  }

  // 暂停历史记录，SSE 推送的内容不应被撤销
  projectStore.pauseHistory()

  // 更新或添加字幕到 store
  const existingIndex = projectStore.subtitles.findIndex(
    s => s.sentenceIndex === sentenceIndex
  )

  // V3.1.2+dev.20260112.01: 包含 display_confidence 和 confidence_source
  const subtitleData = {
    sentenceIndex,
    text,
    start: start ?? 0,
    end: end ?? 0,
    confidence,
    display_confidence: displayConfidence,
    confidence_source: confidenceSource,
    warning_type: warningType,
    source,
    isModified: sentence.is_modified ?? false,
    originalText: sentence.original_text ?? null
  }

  if (existingIndex >= 0) {
    // V3.1.2+dev.20260112.01: 修复 - 传入正确的 id 而非索引
    const existingId = projectStore.subtitles[existingIndex].id
    projectStore.updateSubtitle(existingId, subtitleData)
  } else {
    // 添加新字幕 - 按时间顺序找到插入位置
    const insertIndex = projectStore.subtitles.findIndex(s => s.start > (start ?? 0))
    const finalIndex = insertIndex === -1 ? projectStore.subtitles.length : insertIndex
    // V3.1.2: 添加 id 字段给新字幕
    subtitleData.id = `sv_${sentenceIndex}`
    projectStore.addSubtitle(finalIndex, subtitleData)
  }

  // 恢复历史记录
  projectStore.resumeHistory()

  console.log(`[EditorView] 字幕 #${sentenceIndex} 已更新，来源: ${source}`)
}

// Phase 5: 处理草稿字幕（快流/SenseVoice）
function handleDraftSubtitle(data) {
  if (!data) return

  // 后端数据格式: { index, chunk_index, sentence: { text, start, end, confidence, words, ... } }
  const chunkIndex = data.chunk_index
  const sentenceIndex = data.index
  const sentence = data.sentence

  if (!sentence) {
    console.warn('[EditorView] 草稿数据缺少 sentence 字段:', data)
    return
  }

  // V3.1.2+dev.20260111.01: 构建 sentenceData，包含 display_confidence
  const sentenceData = {
    index: sentenceIndex,
    text: sentence.text || '',
    start: sentence.start ?? 0,
    end: sentence.end ?? 0,
    confidence: sentence.confidence ?? null,
    display_confidence: sentence.display_confidence,  // V3.1.2: 映射后准确率
    confidence_source: sentence.confidence_source,    // V3.1.2: 置信度来源
    words: sentence.words || [],
    warning_type: sentence.warning_type || 'none',
    is_modified: sentence.is_modified ?? false,
    original_text: sentence.original_text ?? null
  }

  // 调用 projectStore 的草稿处理方法，传递两个参数
  projectStore.appendOrUpdateDraft(chunkIndex, sentenceData)

  console.log(`[EditorView] 处理草稿字幕，chunk_index: ${chunkIndex}, sentence_index: ${sentenceIndex}`)
}

// Phase 5: 处理替换 Chunk（慢流/Whisper）
function handleReplaceChunk(data) {
  if (!data) return

  // 后端数据格式: { chunk_index, old_indices, new_indices, sentences: [...] }
  const chunkIndex = data.chunk_index
  const sentences = Array.isArray(data.sentences) ? data.sentences : []

  // V3.1.2+dev.20260111.01: 转换为 projectStore 需要的格式，包含 display_confidence
  const formattedSentences = sentences.map((sentence, idx) => ({
    index: data.new_indices?.[idx] ?? idx,
    text: sentence.text || '',
    start: sentence.start ?? 0,
    end: sentence.end ?? 0,
    confidence: sentence.confidence ?? null,
    display_confidence: sentence.display_confidence,  // V3.1.2: 映射后准确率
    confidence_source: sentence.confidence_source,    // V3.1.2: 置信度来源
    words: sentence.words || [],
    warning_type: sentence.warning_type || 'none',
    source: sentence.source || 'whisper',
    is_modified: sentence.is_modified ?? false,
    original_text: sentence.original_text ?? null
  }))

  // 调用 projectStore 的替换方法
  projectStore.replaceChunk(chunkIndex, formattedSentences)

  console.log(`[EditorView] 替换 Chunk ${chunkIndex}，共 ${formattedSentences.length} 条定稿字幕`)
}

/**
 * V3.1.0: 处理恢复的字幕（断点续传后恢复）
 * 后端数据格式: { chunk_index, sentences: [...], is_restore: true }
 */
function handleRestoredChunk(data) {
  if (!data) return

  const chunkIndex = data.chunk_index
  const sentences = Array.isArray(data.sentences) ? data.sentences : []

  if (sentences.length === 0) {
    console.log(`[EditorView] Chunk ${chunkIndex} 恢复的字幕为空，跳过`)
    return
  }

  // V3.1.2+dev.20260111.01: 转换为 projectStore 需要的格式，包含 display_confidence
  const formattedSentences = sentences.map((sentence, idx) => ({
    index: sentence.index ?? idx,
    text: sentence.text || '',
    start: sentence.start ?? 0,
    end: sentence.end ?? 0,
    confidence: sentence.confidence ?? null,
    display_confidence: sentence.display_confidence,  // V3.1.2: 映射后准确率
    confidence_source: sentence.confidence_source,    // V3.1.2: 置信度来源
    words: sentence.words || [],
    warning_type: sentence.warning_type || 'none',
    source: sentence.source || 'restored',
    is_draft: sentence.is_draft ?? false,
    is_finalized: sentence.is_finalized ?? true,
    is_modified: sentence.is_modified ?? false,
    original_text: sentence.original_text ?? null
  }))

  // 调用 projectStore 的恢复方法
  projectStore.restoreChunk(chunkIndex, formattedSentences)

  console.log(`[EditorView] 恢复 Chunk ${chunkIndex}，共 ${formattedSentences.length} 条字幕`)
}

function handleSubtitleDeleted(data) {
  if (!data) return
  const sentenceIndex = data.index ?? data.sentence_index
  if (sentenceIndex === undefined || sentenceIndex === null) return
  projectStore.markSentenceDeleted(sentenceIndex)
  const target = projectStore.subtitles.find(s => s.sentenceIndex === sentenceIndex)
  if (target) {
    projectStore.removeSubtitle(target.id)
  }
}

/**
 * V3.5: 处理极速模式定稿字幕
 * 后端数据格式: { index, chunk_index, sentence: {...}, mode: 'sensevoice_only' }
 */
function handleFinalizedSubtitle(data) {
  if (!data) return

  // 极速模式的定稿字幕处理方式与草稿类似，但标记为定稿
  const sentence = data.sentence || {}
  const chunkIndex = data.chunk_index

  // V3.1.2+dev.20260111.01: 构建 sentenceData，包含 display_confidence
  const sentenceData = {
    index: data.index ?? 0,
    text: sentence.text || '',
    start: sentence.start ?? 0,
    end: sentence.end ?? 0,
    confidence: sentence.confidence ?? null,
    display_confidence: sentence.display_confidence,  // V3.1.2: 映射后准确率
    confidence_source: sentence.confidence_source,    // V3.1.2: 置信度来源
    words: sentence.words || [],
    warning_type: sentence.warning_type || 'none',
    is_modified: sentence.is_modified ?? false,
    original_text: sentence.original_text ?? null
  }

  // 调用草稿方法，但数据已标记为定稿
  projectStore.appendOrUpdateDraft(chunkIndex, sentenceData)

  console.log(`[EditorView] 极速模式定稿 Chunk ${chunkIndex}，句子索引 ${data.index}`)
}

// 刷新任务进度（用于SSE重连后的状态同步）
async function refreshTaskProgress() {
  try {
    const jobStatus = await transcriptionApi.getJobStatus(props.jobId, true)
    console.log('[EditorView] 刷新任务进度:', jobStatus)

    progressStore.applySnapshot(
      props.jobId,
      {
        percent: jobStatus.progress,
        status: jobStatus.status,
        phase: jobStatus.phase,
        phase_percent: jobStatus.phase_percent,
        message: jobStatus.message,
        processed: jobStatus.processed,
        total: jobStatus.total
      },
      'http_refresh'
    )
  } catch (error) {
    console.warn('[EditorView] 刷新任务进度失败:', error)
  }
}

function startProgressPolling() {
  stopProgressPolling()
  // 仅在 SSE 长时间无心跳时触发 HTTP 兜底
  progressPollTimer = setInterval(async () => {
    if (!isTranscribing.value) {
      stopProgressPolling()
      return
    }
    const rawState = progressStore.getRawState(props.jobId)
    if (Date.now() - (rawState.lastSseAt || 0) < 10000) {
      return
    }

    try {
      const snapshot = await transcriptionApi.getJobStatus(props.jobId, true)
      progressStore.applySnapshot(
        props.jobId,
        {
          percent: snapshot.progress,
          status: snapshot.status,
          phase: snapshot.phase,
          phase_percent: snapshot.phase_percent,
          message: snapshot.message,
          processed: snapshot.processed,
          total: snapshot.total
        },
        'http_poll'
      )

      // SSE 断连期间，若 Proxy 仍在处理或未就绪，顺便刷新视频转码状态
      if (proxyVideo.isTranscoding.value || !proxyVideo.isReady.value) {
        await proxyVideo.refresh()
      }
    } catch (e) {
      console.warn('[EditorView] 轮询刷新失败:', e)
    }
  }, 15000)
}

function stopProgressPolling() {
  if (progressPollTimer) {
    clearInterval(progressPollTimer)
    progressPollTimer = null
  }
}

// Proxy 状态兜底轮询（独立于转录进度）
async function pollProxyOnce() {
  try {
    const status = await mediaApi.getProxyStatus(props.jobId)
    proxyVideo.applySnapshot(status.data || status)
    // 就绪或报错则停止轮询
    if (proxyVideo.isReady.value || proxyVideo.hasError?.value) {
      stopProxyPolling()
    }
  } catch (e) {
    console.warn('[EditorView] Proxy 状态轮询失败:', e)
  }
}

function startProxyPolling() {
  stopProxyPolling()
  // 已就绪则不轮询
  if (proxyVideo.isReady.value) return
  proxyPollTimer = setInterval(pollProxyOnce, 10000)
  // 立即跑一次，加快恢复
  pollProxyOnce()
}

function stopProxyPolling() {
  if (proxyPollTimer) {
    clearInterval(proxyPollTimer)
    proxyPollTimer = null
  }
}

// ========== 保存功能 ==========

async function saveProject() {
  if (saving.value) return
  saving.value = true
  try {
    const srtContent = projectStore.generateSRT()
    await mediaApi.saveSRTContent(props.jobId, srtContent)
    await projectStore.saveProject()
    lastSaved.value = Date.now()
    console.log('[EditorView] 项目保存成功')
  } catch (error) {
    console.error('[EditorView] 保存失败:', error)
    alert('保存失败: ' + (error.message || '未知错误'))
  } finally {
    saving.value = false
  }
}

// ========== 任务控制 ==========

async function pauseTranscription() {
  try {
    await transcriptionApi.pauseJob(props.jobId)
    progressStore.markStatus(props.jobId, 'paused')
    // 更新本地store状态
    taskStore.updateTaskStatus(props.jobId, 'paused')
    // 暂停后保留SSE连接，以便恢复时能继续接收更新
    console.log('[EditorView] 任务已暂停，保留SSE连接')
  } catch (error) {
    console.error('暂停任务失败:', error)
  }
}

async function resumeTranscription() {
  try {
    // 使用新的 resumeJob API，恢复暂停的任务（重新加入队列）
    const result = await transcriptionApi.resumeJob(props.jobId)

    // 根据后端返回值设置状态（应该是 queued，而不是 processing）
    progressStore.markStatus(props.jobId, result.status || 'queued')

    // 更新本地store状态
    taskStore.updateTaskStatus(props.jobId, result.status || 'queued')

    console.log('[EditorView] 任务已恢复，状态:', result.status, '队列位置:', result.queue_position)

    await refreshTaskProgress()
    startProgressPolling()
  } catch (error) {
    console.error('恢复任务失败:', error)
  }
}

async function cancelTranscription() {
  if (!confirm('确定要取消当前转录任务吗?')) return
  try {
    await transcriptionApi.cancelJob(props.jobId, false)
    progressStore.markStatus(props.jobId, 'canceled')
    // 更新本地store状态
    taskStore.updateTaskStatus(props.jobId, 'canceled')
    // 取消后关闭SSE连接
    cleanupSSE()
    stopProgressPolling()
  } catch (error) {
    console.error('取消任务失败:', error)
  }
}

// ========== 撤销/重做 ==========

function undo() { if (canUndo.value) projectStore.undo() }
function redo() { if (canRedo.value) projectStore.redo() }

// ========== 导出功能 ==========

// 监听导出事件（从 EditorHeader 触发）
onMounted(() => {
  window.addEventListener('header-export', handleExportEvent)
})

onUnmounted(() => {
  window.removeEventListener('header-export', handleExportEvent)
})

async function handleExportEvent(event) {
  const format = event.detail
  await handleExport(format)
}

async function handleExport(format) {
  const segments = await fetchLatestSegments()
  if (!segments || segments.length === 0) {
    alert('导出失败：后端未返回字幕数据')
    return
  }

  let content = ''
  let filename = projectName.value.replace(/\.[^/.]+$/, '')

  switch (format) {
    case 'srt':
      content = segmentsToSRT(segments)
      filename += '.srt'
      break
    case 'ass':
      await handleASSExport(segments)
      return
    case 'vtt':
      content = generateVTTFromSegments(segments)
      filename += '.vtt'
      break
    case 'txt':
      content = segments.map(s => s.text).join('\n')
      filename += '.txt'
      break
    case 'json':
      content = JSON.stringify(segments, null, 2)
      filename += '.json'
      break
  }

  downloadFile(content, filename)
}

async function fetchLatestSegments() {
  try {
    await forceSyncNow()
    const response = await transcriptionApi.getTranscriptionText(props.jobId)
    return response?.segments || []
  } catch (error) {
    console.error('[EditorView] 获取后端字幕失败:', error)
    return []
  }
}

async function handleASSExport(segments) {
  try {
    console.log('[EditorView] 开始生成 ASS 字幕文件')

    const srtContent = segmentsToSRT(segments)
    await mediaApi.saveSRTContent(props.jobId, srtContent)

    // 调用后端 API 生成 ASS 文件
    await transcriptionApi.generateASS(props.jobId, {
      style_preset: 'default',
      title: projectName.value.replace(/\.[^/.]+$/, ''),
      video_width: 1920,
      video_height: 1080
    })

    // 获取生成的 ASS 文件内容
    const assData = await transcriptionApi.getASSContent(props.jobId)

    // 下载文件
    downloadFile(assData.content, assData.filename)

    console.log('[EditorView] ASS 字幕文件导出成功:', assData.filename)
  } catch (error) {
    console.error('[EditorView] 导出 ASS 文件失败:', error)
    alert('导出 ASS 文件失败: ' + (error.message || '未知错误'))
  }
}

/**
 * V3.1.1+dev.20260106.04: 导出前自动修复时间戳重叠
 */
function generateVTTFromSegments(segments) {
  // 修复时间戳重叠（使用1ms间隔）
  const repairedSubtitles = repairSubtitleOverlaps(segments, 1)

  let vtt = 'WEBVTT\n\n'
  repairedSubtitles.forEach((sub, i) => {
    const start = formatVTTTime(sub.start)
    const end = formatVTTTime(sub.end)
    vtt += `${i + 1}\n${start} --> ${end}\n${sub.text}\n\n`
  })
  return vtt
}

function formatVTTTime(seconds) {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = (seconds % 60).toFixed(3)
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.padStart(6, '0')}`
}

function downloadFile(content, filename) {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

// ========== 拖拽调整宽度 ==========

function startResize(e) {
  isResizing.value = true
  document.addEventListener('mousemove', onResize)
  document.addEventListener('mouseup', stopResize)
}

function onResize(e) {
  if (!isResizing.value) return
  const newWidth = window.innerWidth - e.clientX
  sidebarWidth.value = Math.max(280, Math.min(600, newWidth))
}

function stopResize() {
  isResizing.value = false
  document.removeEventListener('mousemove', onResize)
  document.removeEventListener('mouseup', stopResize)
  // 保存用户偏好
  localStorage.setItem('editor-sidebar-width', sidebarWidth.value.toString())
}

// ========== 事件处理 ==========

function handleVideoLoaded(duration) {
  console.log('视频加载完成:', duration)
}

function handleVideoError(error) {
  console.error('视频加载错误:', error)
  // V3.1.2+dev.20260114.22: 兜底降级，如果720p加载失败则回落到360p
  const message = error?.message || ''
  if (proxyVideo.currentResolution.value === '720p' && proxyVideo.urls.value.preview360p) {
    proxyVideo.fallbackTo360p(message)
    ElMessage.warning('720p 加载失败，已自动降级为 360p')
  } else if (!proxyVideo.urls.value.preview360p) {
    ElMessage.error('视频加载失败，且无可用的预览版本')
  }
}

// V3.1.2+dev.20260113.01: 处理720p升级失败
function handleUpgradeFailed(message) {
  console.error('[EditorView] 720p升级失败:', message)
  ElMessage.error(message || '有任务/转码正在运行，请稍后再试')
}

// V3.1.2+dev.20260113.02: 处理720p升级开始
function handleUpgradeStarted() {
  console.log('[EditorView] 720p后台转码已启动')
  ElMessage.success({
    message: '后台转码已开始，完成后将自动替换',
    duration: 5000
  })
}

async function handleCheckVideoStatus() {
  console.log('[EditorView] 视频加载失败，刷新 Proxy 视频状态...')
  try {
    // 刷新 proxyVideo 状态（会从后端重新获取状态并自动订阅SSE）
    await proxyVideo.refresh()
    console.log('[EditorView] Proxy 视频状态已刷新:', {
      state: proxyVideo.state.value,
      isReady: proxyVideo.isReady.value,
      isTranscoding: proxyVideo.isTranscoding.value
    })
  } catch (error) {
    console.error('[EditorView] 刷新 Proxy 视频状态失败:', error)
  }
}

function handleResolutionChange(resolution) {
  console.log('[EditorView] 视频分辨率变更:', resolution)
  // 更新 projectStore 的视频信息
  projectStore.meta.currentResolution = resolution
}

function handleWaveformReady() {
  console.log('波形加载完成')
}

function handleRegionUpdate(region) {
  console.log('区域更新:', region)
}

function handleRegionClick(region) {
  console.log('区域点击:', region)
}

function handleSubtitleClick(subtitle) {
  console.log('字幕点击:', subtitle)
}

function handleSubtitleEdit(id, field, value) {
  console.log('字幕编辑:', id, field, value)
}

function showSettings() {
  // TODO: 实现设置面板
  console.log('打开设置')
}

function formatLastSaved(timestamp) {
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

// ========== 快捷键操作 ==========

function togglePlay() {
  projectStore.player.isPlaying = !projectStore.player.isPlaying
}

function stepBackward() {
  const frameTime = 1 / 30
  const newTime = Math.max(0, projectStore.player.currentTime - frameTime)
  playbackManager.seekTo(newTime)
}

function stepForward() {
  const frameTime = 1 / 30
  const newTime = Math.min(projectStore.meta.duration, projectStore.player.currentTime + frameTime)
  playbackManager.seekTo(newTime)
}

function seekBackward() {
  const newTime = Math.max(0, projectStore.player.currentTime - 5)
  playbackManager.seekTo(newTime)
}

function seekForward() {
  const newTime = Math.min(projectStore.meta.duration, projectStore.player.currentTime + 5)
  playbackManager.seekTo(newTime)
}

function seekToStart() {
  playbackManager.seekTo(0)
}

function seekToEnd() {
  playbackManager.seekTo(projectStore.meta.duration)
}

function zoomInWave() {
  console.log('波形放大')
}

function zoomOutWave() {
  console.log('波形缩小')
}

function fitWave() {
  console.log('波形适应屏幕')
}

function zoomInVideo() {
  console.log('视频画面放大')
}

function zoomOutVideo() {
  console.log('视频画面缩小')
}

function fitVideo() {
  console.log('画面适应窗口')
}

function fontSizeUp() {
  console.log('字体变大')
}

function fontSizeDown() {
  console.log('字体变小')
}

function splitSubtitle() {
  console.log('分割字幕')
}

function mergeSubtitle() {
  console.log('合并字幕')
}

function exportSubtitle() {
  // 触发导出菜单
}

function openTaskMonitor() {
  console.log('打开任务监控')
}

// 使用快捷键系统
useShortcuts({
  togglePlay,
  stepBackward,
  stepForward,
  seekBackward,
  seekForward,
  seekToStart,
  seekToEnd,
  zoomInWave,
  zoomOutWave,
  fitWave,
  zoomInVideo,
  zoomOutVideo,
  fitVideo,
  fontSizeUp,
  fontSizeDown,
  splitSubtitle,
  mergeSubtitle,
  save: saveProject,
  undo,
  redo,
  export: exportSubtitle,
  openTaskMonitor,
})

// ========== 生命周期 ==========

onMounted(() => {
  // 恢复侧边栏宽度偏好
  const savedWidth = localStorage.getItem('editor-sidebar-width')
  if (savedWidth) {
    sidebarWidth.value = parseInt(savedWidth)
  }

  loadProject()
})

onUnmounted(() => {
  // 注意：不在这里关闭SSE连接，以支持页面切换时保持连接
  // SSE连接会在任务完成/失败时自动关闭，或由sseChannelManager统一管理
  console.log('[EditorView] 组件卸载，保留SSE连接以支持后台任务')

  // 停止轮询（轮询仅是备用方案）
  stopProgressPolling()
  stopProxyPolling()
})

onBeforeRouteLeave(async (to, from) => {
  if (isDirty.value) {
    try {
      await projectStore.saveProject()
      console.log('[EditorView] 离开前自动保存成功')
    } catch (error) {
      console.error('[EditorView] 离开前保存失败:', error)
      const answer = window.confirm('保存失败，确定要离开吗? 未保存的修改可能会丢失。')
      if (!answer) return false
    }
  }
})
</script>

<style lang="scss" scoped>
@use '@/styles/variables' as *;
@use '@/styles/mixins' as *;

.editor-view {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--bg-base);
  color: var(--text-normal);
  overflow: hidden;
}

// 加载状态
.loading-overlay {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  color: var(--text-muted);

  .loading-spinner {
    width: 48px;
    height: 48px;
    border: 3px solid var(--border-default);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
}

// 错误状态
.error-overlay {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  color: var(--text-muted);
  text-align: center;

  .error-icon {
    width: 64px;
    height: 64px;
    color: var(--danger);
  }

  h3 {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  p {
    font-size: 14px;
    max-width: 400px;
    margin: 0;
  }

  .retry-btn {
    padding: 10px 24px;
    background: var(--primary);
    color: white;
    border: none;
    border-radius: var(--radius-md);
    font-size: 14px;
    cursor: pointer;
    transition: background var(--transition-fast);
    &:hover { background: var(--primary-hover); }
  }

  .back-link {
    font-size: 13px;
    color: var(--text-secondary);
    text-decoration: underline;
    &:hover { color: var(--text-primary); }
  }
}

@keyframes spin { to { transform: rotate(360deg); } }

// 主工作区 Grid 布局
.workspace-grid {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 4px 350px;
  height: 100%;
  overflow: hidden;
}

// 舞台列 (三明治结构: 视频 + 控制 + 波形)
.stage-column {
  display: grid;
  grid-template-rows: 1fr 48px 200px;  // 波形区域调整为200px（header+刻度+波形+滚动条）
  background: #000;
  min-width: 0;
  overflow: hidden;

  .video-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: hidden;
  }

  .controls-wrapper {
    background: var(--bg-primary);
    border-top: 1px solid var(--border-default);
  }

  .waveform-wrapper {
    background: var(--bg-secondary);
    border-top: 1px solid var(--border-default);
    overflow: hidden;
  }
}

// 可拖拽分隔条
.resizer {
  width: 4px;
  background: var(--border-default);
  cursor: col-resize;
  transition: background 0.2s;
  position: relative;
  z-index: 10;

  &:hover, &.active {
    background: var(--primary);
  }
}

// 侧边栏
.sidebar-column {
  display: flex;
  flex-direction: column;
  background: var(--bg-primary);
  border-left: 1px solid var(--border-default);
  min-width: 280px;
  max-width: 600px;
  overflow: hidden;
}

// 标签页导航
.tab-nav {
  display: flex;
  padding: 0 12px;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-default);

  .tab-btn {
    position: relative;
    padding: 12px 16px;
    font-size: 13px;
    color: var(--text-secondary);
    background: transparent;
    border: none;
    cursor: pointer;
    transition: color var(--transition-fast);

    &::after {
      content: '';
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      height: 2px;
      background: transparent;
      transition: background var(--transition-fast);
    }

    &:hover { color: var(--text-normal); }

    &.active {
      color: var(--primary);
      &::after { background: var(--primary); }
    }

    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 18px;
      height: 18px;
      margin-left: 6px;
      padding: 0 5px;
      background: var(--danger);
      color: white;
      font-size: 11px;
      border-radius: var(--radius-full);
    }
  }
}

// 标签页内容
.tab-content { flex: 1; overflow: hidden; }
.tab-pane { height: 100%; overflow: auto; }

// 占位面板
.placeholder-panel {
  @include flex-center;
  @include flex-column;
  padding: 48px 24px;
  text-align: center;
  color: var(--text-muted);

  svg {
    width: 48px;
    height: 48px;
    margin-bottom: 16px;
    opacity: 0.5;
  }

  h3 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-normal);
    margin-bottom: 8px;
  }

  p { font-size: 13px; margin-bottom: 4px; }
  .coming-soon { color: var(--primary); font-style: italic; }

  .error-list {
    width: 100%;
    max-width: 400px;
    margin-top: 16px;
    text-align: left;
  }

  .error-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    margin-bottom: 4px;
    background: var(--bg-secondary);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: background var(--transition-fast);

    &:hover { background: var(--bg-tertiary); }
    &.error { border-left: 3px solid var(--danger); }
    &.warning { border-left: 3px solid var(--warning); }

    .error-index {
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--text-muted);
    }

    .error-message {
      font-size: 13px;
      color: var(--text-normal);
    }
  }
}

// 底部状态栏
.editor-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 28px;
  padding: 0 16px;
  background: var(--bg-secondary);
  border-top: 1px solid var(--border-default);
  font-size: 12px;
  color: var(--text-muted);
  flex-shrink: 0;

  .footer-left, .footer-center, .footer-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .save-time {
    display: flex;
    align-items: center;
    gap: 4px;
    color: var(--text-secondary);

    .icon {
      width: 14px;
      height: 14px;
      color: var(--success);
    }
  }

  .error-indicator {
    color: var(--danger);
    cursor: pointer;
    &:hover { text-decoration: underline; }
  }

  .settings-btn {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    background: transparent;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;

    svg { width: 16px; height: 16px; }

    &:hover {
      background: var(--bg-tertiary);
      color: var(--text-normal);
    }
  }

  .divider { color: var(--border-default); }
}
</style>
