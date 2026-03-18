<template>
  <div class="editor-view">
    <!-- 顶部导航栏 - 使用新的 EditorHeader 组件 -->
    <EditorHeader
      :job-id="activeJobId"
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
        <path
          d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"
        />
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
            :key="`video-${mediaIdentityId || 'none'}`"
            ref="videoStageRef"
            :media-id="mediaIdentityId"
            :show-subtitle="true"
            :enable-keyboard="false"
            :progressive-url="resolvedProgressiveUrl"
            :current-resolution="proxyVideo.currentResolution.value"
            :is-upgrading="proxyVideo.isUpgrading.value"
            :upgrade-progress="proxyVideo.progress.value"
            :proxy-state="proxyVideo.state.value"
            :proxy-error="proxyVideo.error.value"
            :auto-trigger-720p="proxyVideo.autoTrigger720p.value"
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
            :key="`wave-${mediaIdentityId || 'none'}`"
            ref="waveformRef"
            :media-id="mediaIdentityId"
          />
        </div>
      </section>

      <!-- 可拖拽分隔条 -->
      <div class="resizer" @mousedown="startResize" :class="{ active: isResizing }"></div>

      <!-- 右侧边栏 -->
      <aside class="sidebar-column" :class="{ 'is-resizing': isResizing }">
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
          <!-- 右侧弹簧 + 视图切换按钮（搜索激活时显示） -->
          <div class="tw-flex-1" />
          <el-popover
            v-if="subtitleListRef?.isSearchActive"
            :content="subtitleListRef?.sortMode === 'grouped' ? '当前：分组模式（点击切换为时间线）' : '当前：时间线模式（点击切换为分组）'"
            placement="top"
            trigger="hover"
            popper-class="hint-popover-compact"
            :show-after="500"
          >
            <template #reference>
              <button
                class="tab-nav-icon-btn"
                @click="subtitleListRef?.toggleSortMode?.()"
              >
                <svg v-if="subtitleListRef?.sortMode === 'grouped'" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M3 15h4v-2H3v2zm0 4h4v-2H3v2zm0-8h4V9H3v2zm4-6v2h14V5H7zm0 10h14v-2H7v2zm0 4h14v-2H7v2z"/>
                </svg>
                <svg v-else viewBox="0 0 24 24" fill="currentColor">
                  <path d="M3 14h4v-4H3v4zm0 5h4v-4H3v4zM3 9h4V5H3v4zm5 5h13v-4H8v4zm0 5h13v-4H8v4zM8 5v4h13V5H8z"/>
                </svg>
              </button>
            </template>
          </el-popover>
        </div>

        <!-- 标签页内容 -->
        <div class="tab-content">
          <div v-show="activeTab === 'subtitles'" class="tab-pane">
            <!-- V3.2.5+dev.20260316.04: Phase F-0 运行时硬切换，仅挂载新字幕列表 -->
            <VirtualSubtitleList
              ref="subtitleListRef"
              :auto-scroll="true"
              :enable-auto-resume-follow="true"
              :is-resizing="isResizing"
            />
          </div>

          <div v-show="activeTab === 'validation'" class="tab-pane">
            <div class="placeholder-panel">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path
                  d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
                />
              </svg>
              <h3>问题检查</h3>
              <p>即将实现</p>
            </div>
          </div>

          <div v-show="activeTab === 'assistant'" class="tab-pane">
            <div class="placeholder-panel">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M21 10.5h-6.5V4h-5v6.5H3v5h6.5V22h5v-6.5H21v-5z" />
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
            <path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z" />
          </svg>
          自动保存于 {{ formatLastSaved(lastSaved) }}
        </span>
      </div>

      <div class="footer-right">
        <span v-if="errorCount > 0" class="error-indicator" @click="activeTab = 'validation'">
          {{ errorCount }} 个问题
        </span>
        <span class="divider">|</span>
        <el-popover
          content="高级设置"
          placement="top"
          trigger="hover"
          popper-class="hint-popover-compact"
          :show-after="500"
        >
          <template #reference>
            <button class="settings-btn" @click="showAdvancedSettings = true">
              <svg viewBox="0 0 24 24" fill="currentColor">
                <path
                  d="M19.14,12.94c0.04-0.3,0.06-0.61,0.06-0.94c0-0.32-0.02-0.64-0.07-0.94l2.03-1.58c0.18-0.14,0.23-0.41,0.12-0.61 l-1.92-3.32c-0.12-0.22-0.37-0.29-0.59-0.22l-2.39,0.96c-0.5-0.38-1.03-0.7-1.62-0.94L14.4,2.81c-0.04-0.24-0.24-0.41-0.48-0.41 h-3.84c-0.24,0-0.43,0.17-0.47,0.41L9.25,5.35C8.66,5.59,8.12,5.92,7.63,6.29L5.24,5.33c-0.22-0.08-0.47,0-0.59,0.22L2.74,8.87 C2.62,9.08,2.66,9.34,2.86,9.48l2.03,1.58C4.84,11.36,4.8,11.69,4.8,12s0.02,0.64,0.07,0.94l-2.03,1.58 c-0.18,0.14-0.23,0.41-0.12,0.61l1.92,3.32c0.12,0.22,0.37,0.29,0.59,0.22l2.39-0.96c0.5,0.38,1.03,0.7,1.62,0.94l0.36,2.54 c0.05,0.24,0.24,0.41,0.48,0.41h3.84c0.24,0,0.44-0.17,0.47-0.41l0.36-2.54c0.59-0.24,1.13-0.56,1.62-0.94l2.39,0.96 c0.22,0.08,0.47,0,0.59-0.22l1.92-3.32c0.12-0.22,0.07-0.47-0.12-0.61L19.14,12.94z M12,15.6c-1.98,0-3.6-1.62-3.6-3.6 s1.62-3.6,3.6-3.6s3.6,1.62,3.6,3.6S13.98,15.6,12,15.6z"
                />
              </svg>
            </button>
          </template>
        </el-popover>
      </div>
    </footer>

    <!-- 高级设置对话框 -->
    <el-dialog
      v-model="showAdvancedSettings"
      title="高级设置"
      width="720px"
      class="advanced-settings-dialog"
      align-center
      :lock-scroll="false"
      :close-on-click-modal="false"
      :close-on-press-escape="true"
      @close="handleCloseAdvancedSettings"
    >
      <AdvancedSettings
        v-model="advancedConfig"
        :enable-shortcut-customization="true"
        @open-about="showAboutDialog = true"
      />
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="handleCancelAdvancedSettings">取消</el-button>
          <el-button type="primary" @click="handleSaveAdvancedSettings">保存设置</el-button>
        </span>
      </template>
    </el-dialog>

    <!-- V3.2.4+dev.20260303.04: 高级设置内"关于"Tab 触发 -->
    <AboutDialog v-model="showAboutDialog" />
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
import { ref, computed, onMounted, onUnmounted, provide, watch } from 'vue'
import { onBeforeRouteLeave, useRouter } from 'vue-router'
import { useProjectStore } from '@/stores/projectStore'
import { usePlaybackStore } from '@/stores/playbackStore'
import { useTaskRuntimeStore } from '@/stores/taskRuntimeStore'
import { useEditorDocumentStore } from '@/stores/editor/editorDocumentStore'
import { useEditorCommandBus } from '@/stores/editor/editorCommandBus'
import { useEditorHistoryStore } from '@/stores/editor/editorHistoryStore'
import { useEditorSessionStore } from '@/stores/editor/editorSessionStore'
import { useEditorSyncEngine } from '@/stores/editor/editorSyncEngine'
import { useEditorProjectionBridge } from '@/stores/editor/editorProjectionBridge'
import { useEditorEventProjector } from '@/stores/editor/editorEventProjector'
import {
  scheduleAuthoritativeReloadForRealtimeEvent,
  shouldScheduleAuthoritativeReloadForProjectModeEvent,
} from '@/stores/editor/editorRealtimeSyncPolicy'
import {
  applySentencePatch,
  deleteServerSegment,
  shouldIgnoreProjectAckEvent,
  upsertServerSegment,
} from '@/stores/editor/editorServerProjection'
import { useSubtitleDocumentStore } from '@/stores/subtitleDocumentStore'
import { useStructuralSyncStore } from '@/stores/structuralSyncStore'
import { legacyApi, mediaApi, projectApi, transcriptionApi } from '@/services/api'
import sseChannelManager from '@/services/sseChannelManager'
import { useShortcuts } from '@/hooks/useShortcuts'
import { useProxyVideo } from '@/composables/useProxyVideo'
import { usePlaybackManager } from '@/services/PlaybackManager'
import { repairSubtitleOverlaps } from '@/utils/subtitleUtils'
import {
  DEFAULT_EDITOR_SHORTCUT_CONFIG,
  detectBrowserReservedShortcutConflicts,
  detectEditorShortcutConflicts,
  getEditorShortcutActionLabel,
  getEditorShortcutComboLabel,
  normalizeEditorShortcutConfig,
} from '@/utils/editorShortcuts'
import { ElMessage } from 'element-plus'
import {
  buildDefaultCapabilitySnapshot,
  selectFlavor,
} from '@/state/capabilities/capabilitySelector'

// 组件导入
import EditorHeader from '@/components/editor/EditorHeader.vue'
import PlaybackControls from '@/components/editor/PlaybackControls/index.vue'
import VideoStage from '@/components/editor/VideoStage/index.vue'
import VirtualSubtitleList from '@/components/editor/VirtualSubtitleList/index.vue'
import WaveformTimeline from '@/components/editor/WaveformTimeline/index.vue'
import AdvancedSettings from '@/components/editor/AdvancedSettings.vue'
import AboutDialog from '@/components/AboutDialog.vue'
import { isFeatureEnabled } from '@/config/featureFlags'
import { initEditorCore } from '@/stores/editor/editorCore'

// Props
const rawProps = defineProps({
  projectId: { type: String, default: null },
  jobId: { type: String, default: null },
})
const resolvedProjectId = ref(null)
const resolvedJobId = ref(null)
const props = new Proxy(rawProps, {
  get(target, key) {
    if (key === 'jobId') return resolvedJobId.value || target.jobId || null
    if (key === 'projectId') return resolvedProjectId.value || target.projectId || null
    return target[key]
  },
})

// Stores
const projectStore = useProjectStore()
const playbackStore = usePlaybackStore()
const taskStore = useTaskRuntimeStore()
const editorDocumentStore = useEditorDocumentStore()
const editorCommandBus = useEditorCommandBus()
const editorHistoryStore = useEditorHistoryStore()
const editorSessionStore = useEditorSessionStore()
const editorSyncEngine = useEditorSyncEngine()
const editorProjectionBridge = useEditorProjectionBridge()
const editorEventProjector = useEditorEventProjector()
const subtitleDocumentStore = useSubtitleDocumentStore()
const structuralSyncStore = useStructuralSyncStore()
const router = useRouter()

// V3.2.5+dev.20260315.01: Feature flag 控制新旧内核
const useEditorV2 = isFeatureEnabled('USE_EDITOR_V2')
console.log('[EditorView] useEditorV2:', useEditorV2)

// 全局播放管理器
const playbackManager = usePlaybackManager()

const activeJobId = computed(() => props.jobId || null)
// 媒体链路统一以 project_id 为主；job_id 仅用于任务控制与进度链路。
const mediaIdentityId = computed(() => props.projectId || props.jobId || null)
const playbackSessionId = computed(() => props.projectId || props.jobId || '')
const identityRef = computed(() => mediaIdentityId.value)
const forceSyncNow = subtitleDocumentStore.forceSyncNow

// Proxy 视频加载状态（新重构版本）
// V3.2.4+dev.20260224.01: project 模式也需要接入媒体状态，避免 progressiveUrl 为空导致视频无法加载
const proxyVideo = useProxyVideo(identityRef)

// Refs
const videoStageRef = ref(null)
const waveformRef = ref(null)
const subtitleListRef = ref(null)
const activeTab = ref('subtitles')
const saving = ref(false)
const lastSaved = ref(null)

// 布局状态
const sidebarWidth = ref(350)
const isResizing = ref(false)
let sidebarResizeFrameId = 0
let pendingSidebarWidth = null

// 加载状态
const isLoading = ref(true)
const loadError = ref(null)

// 高级设置状态
const showAdvancedSettings = ref(false)
const showAboutDialog = ref(false)
const advancedConfig = ref({
  general: {
    global_time_offset: projectStore.subtitleOffset, // 从 store 初始化
    duration_adjust: 0,
    auto_save_interval: 60,
    preview_font_size: 24,
    enable_shortcuts: true,
    subtitle_follow_auto_resume: true,
    hide_timeline_scale: false,
    merge_separator: 'space',
    merge_separator_custom: '',
  },
  preprocessing: {
    demucs_strategy: 'auto',
    demucs_model: 'htdemucs',
    demucs_shifts: 1,
    spectrum_threshold: 0.5,
    vad_filter: true,
  },
  transcription: {
    transcription_profile: 'sv_whisper_dual',
    sensevoice_device: 'cuda',
    whisper_model: 'large-v3',
    patching_threshold: 0.3,
  },
  refinement: {
    llm_task: 'off',
    llm_scope: 'sparse',
    sparse_threshold: 0.5,
    target_language: 'zh',
    llm_model_name: 'gpt-4o-mini',
  },
  compute: {
    concurrency_strategy: 'auto',
    gpu_id: 0,
    temp_file_policy: 'delete_on_complete',
  },
  shortcuts: { ...DEFAULT_EDITOR_SHORTCUT_CONFIG },
  preset_id: 'default',
})
const SUBTITLE_FOLLOW_AUTO_RESUME_PREF_KEY = 'editor-subtitle-follow-auto-resume'
const HIDE_TIMELINE_SCALE_PREF_KEY = 'editor-hide-timeline-scale'
const SHORTCUT_ENABLED_PREF_KEY = 'editor-shortcuts-enabled'
const SHORTCUT_CONFIG_PREF_KEY = 'editor-shortcuts-config'
const subtitleFollowAutoResumeEnabled = ref(true)
const hideTimelineScale = ref(false)
const shortcutEnabled = ref(true)
const shortcutConfig = ref({ ...DEFAULT_EDITOR_SHORTCUT_CONFIG })

// 统一进度状态
const progressStore = taskStore
// 修复：直接使用 getRawState 获取响应式状态对象，避免 computed 嵌套导致响应式丢失
const jobProgress = computed(() => progressStore.getRawState(activeJobId.value))
const taskStatus = computed(() => jobProgress.value.status || 'idle')
const taskPhase = computed(() => jobProgress.value.phase || 'pending')
const taskProgress = computed(() => jobProgress.value.percent || 0)
const progressDetail = computed(() => jobProgress.value.detail || null)
const isTranscribing = computed(() => ['processing', 'queued'].includes(taskStatus.value))
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
const isCancelPending = ref(false)
let missingVideoWarnedIdentity = null
const isVideoAbsenceConfirmed = ref(false)
const CANCEL_TIMEOUT_MS = 30000
const CANCEL_TERMINAL_STATUSES = new Set(['canceled', 'failed', 'finished', 'removed', 'force_canceled'])
let cancelTimeoutTimer = null
// V3.2.4+dev.20260222.15: 定稿事件实时回拉后端真源（单飞+补偿，避免请求风暴）
let isRealtimeFinalSyncInFlight = false
let hasRealtimeFinalSyncPending = false
let realtimeFinalSyncReason = null
let editorCoreSessionKey = null
let isEditorProjectionReloadInFlight = false
let hasEditorProjectionReloadPending = false
let editorProjectionReloadReason = null
let strictSyncInFlightPromise = null

watch(
  () => activeJobId.value,
  (jobId) => {
    subtitleDocumentStore.bindTask(jobId)
  },
  { immediate: true }
)

watch(
  () => identityRef.value,
  (identityId) => {
    subtitleDocumentStore.bindSyncIdentity(identityId)
  },
  { immediate: true }
)

const hasVideoSource = computed(() => {
  return !isVideoAbsenceConfirmed.value && !!proxyVideo.currentUrl.value
})
const isMediaReady = computed(() => {
  return (
    hasVideoSource.value
    || !!projectStore.meta.audioPath
    || proxyVideo.isReady.value
    || subtitleDocumentStore.subtitles.length > 0
  )
})

// Provide 编辑器上下文
// ========== Provide 上下文 ==========

// 提供编辑器上下文给子组件
provide('editorContext', {
  jobId: activeJobId,
  saving,
  // 媒体就绪：音频或视频任一可用即可放开编辑能力。
  isMediaReady,
  // 兼容旧注入名称，避免存量组件行为突变。
  isVideoReady: isMediaReady,
  hasVideoSource,
  hideTimelineScale,
})

// 将 projectStore 的估算双流进度同步到统一状态，供 UI 兜底展示
watch(
  () => projectStore.dualStreamProgress,
  (progress) => {
    if (!progress || !activeJobId.value) return
    progressStore.applyDualStreamEstimate(activeJobId.value, progress)
  },
  { deep: true }
)

// 监听路由身份变化，重新加载项目
watch([() => rawProps.projectId, () => rawProps.jobId], async ([newProjectId, newJobId], [oldProjectId, oldJobId]) => {
  if (newProjectId === oldProjectId && newJobId === oldJobId) return
  const nextSessionId = newProjectId || newJobId || ''
  playbackManager.bindSession(nextSessionId, { force: true, resetPosition: true })

  // 取消旧的 SSE 订阅
  cleanupSSE()
  stopCancelTimeoutPolling()
  isCancelPending.value = false

  // 重置项目状态
  projectStore.resetProject()
  editorCoreSessionKey = null

  // 重新加载项目
  await loadProject()
})

watch(
  () => playbackSessionId.value,
  (sessionId) => {
    playbackManager.bindSession(sessionId, { force: true, resetPosition: true })
  },
  { immediate: true }
)

// 监听保存时间用于同步 task-meta 的显示
watch(
  () => projectStore.meta.lastSaved,
  (value) => {
    if (!projectStore.primaryId) {
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
  const task = taskStore.tasks.find((t) => t.job_id === activeJobId.value)
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
  const normalizedJobId = activeJobId.value || ''
  if (
    displayName
    && displayName !== normalizedJobId
    && displayName !== normalizedJobId.replace(/-/g, '')
  ) {
    return displayName
  }

  return '未命名项目'
})

// 基础状态
const isDirty = computed(() => editorSessionStore.isDirty)
const totalSubtitles = computed(() => editorProjectionBridge.totalSubtitles)
const currentSubtitle = computed(() => editorProjectionBridge.currentSubtitle)
const currentSubtitleIndex = computed(() =>
  currentSubtitle.value
    ? editorProjectionBridge.findSubtitleIndexById(currentSubtitle.value.id)
    : -1
)

// 撤销/重做
const canUndo = computed(() => editorHistoryStore.canUndo)
const canRedo = computed(() => editorHistoryStore.canRedo)

// 队列进度计算
const queueCompleted = computed(() => taskStore.tasks.filter((t) => t.status === 'finished').length)
const queueTotal = computed(() => taskStore.tasks.filter((t) => t.status !== 'created').length)
const activeTasks = computed(
  () => taskStore.tasks.filter((t) => ['processing', 'queued'].includes(t.status)).length
)

// 问题检查计数（统计有警告的字幕数量）
const errorCount = computed(() => editorProjectionBridge.warningCount)

// Grid 布局样式
const gridStyle = computed(() => ({
  gridTemplateColumns: `1fr 4px ${sidebarWidth.value}px`,
}))

const resolvedProgressiveUrl = computed(() => {
  if (!hasVideoSource.value) {
    return undefined
  }
  // 仅使用 proxy 状态返回的可播放 URL，禁止回落到本地 videoPath，避免脏缓存触发 404。
  return proxyVideo.currentUrl.value || undefined
})

function hasProjectMediaAsset(project, assetType) {
  const assets = Array.isArray(project?.media_assets) ? project.media_assets : []
  return assets.some((asset) => {
    const type = String(asset?.type || '').trim()
    const exists = asset?.exists
    return type === assetType && exists !== false
  })
}

function applyMediaPaths({ project = null, mediaStatus = null } = {}) {
  const identity = mediaIdentityId.value
  if (!identity) {
    isVideoAbsenceConfirmed.value = false
    projectStore.setMediaPaths({
      videoPath: null,
      audioPath: null,
    })
    return
  }

  const hasProjectVideo =
    hasProjectMediaAsset(project, 'video')
    || hasProjectMediaAsset(project, 'preview_360p')
    || hasProjectMediaAsset(project, 'proxy_720p')
  const hasProjectAudio = hasProjectMediaAsset(project, 'audio')

  const hasVideoFromStatus = Boolean(
    mediaStatus?.video_exists
    || mediaStatus?.video?.exists
    || mediaStatus?.proxy_exists
    || mediaStatus?.proxy?.exists
    || mediaStatus?.proxy?.ready
  )
  const hasAudioFromStatus = Boolean(mediaStatus?.audio_exists || mediaStatus?.audio_url)

  const hasVideo = hasProjectVideo || hasVideoFromStatus
  let hasAudio = hasProjectAudio || hasAudioFromStatus || hasVideo
  const canDetermineVideoPresence = !!project || !!mediaStatus

  // 兼容老任务：状态未携带媒体信息时，至少保留音频链路可用。
  if (!project && !mediaStatus && activeJobId.value) {
    hasAudio = true
  }

  isVideoAbsenceConfirmed.value = canDetermineVideoPresence && !hasVideo

  projectStore.setMediaPaths({
    videoPath: hasVideo ? mediaApi.getVideoUrl(identity) : null,
    audioPath: hasAudio ? mediaApi.getAudioUrl(identity) : null,
  })
}

async function reloadEditorProjectionFromBackend(projectId, reason = 'unknown') {
  void reason
  if (!useEditorV2 || !projectId) {
    return 0
  }

  const nextSessionKey = `${projectId}::${activeJobId.value || ''}`
  if (editorCoreSessionKey !== nextSessionKey) {
    initEditorCore(projectId, activeJobId.value)
    editorCoreSessionKey = nextSessionKey
  }

  const { loadSubtitlesFromBackend } = await import('@/stores/editor/editorCoreLoader')
  await loadSubtitlesFromBackend(projectId)
  return editorProjectionBridge.totalSubtitles
}

async function scheduleEditorProjectionReload(projectId, reason = 'unknown') {
  if (!useEditorV2 || !projectId) {
    return 0
  }

  editorProjectionReloadReason = reason
  if (isEditorProjectionReloadInFlight) {
    hasEditorProjectionReloadPending = true
    return editorProjectionBridge.totalSubtitles
  }

  isEditorProjectionReloadInFlight = true
  try {
    do {
      const currentReason = editorProjectionReloadReason || reason
      hasEditorProjectionReloadPending = false
      editorProjectionReloadReason = null
      await reloadEditorProjectionFromBackend(projectId, currentReason)
    } while (hasEditorProjectionReloadPending)
  } finally {
    isEditorProjectionReloadInFlight = false
  }

  return editorProjectionBridge.totalSubtitles
}

function scheduleV2RealtimeAuthoritativeReload(reason) {
  scheduleAuthoritativeReloadForRealtimeEvent({
    useEditorV2,
    projectId: props.projectId || projectStore.meta.projectId,
    reason,
    activeJobId: activeJobId.value,
    scheduleReload: (projectId, reloadReason) => {
      void scheduleEditorProjectionReload(projectId, reloadReason)
    },
  })
}

function notifyMissingVideoOnce() {
  const identity = mediaIdentityId.value || props.projectId || activeJobId.value
  if (!identity) {
    return
  }
  if (!isVideoAbsenceConfirmed.value) {
    return
  }
  if (projectStore.meta.videoPath || proxyVideo.currentUrl.value) {
    return
  }
  if (missingVideoWarnedIdentity === identity) {
    return
  }
  missingVideoWarnedIdentity = identity
  ElMessage.warning('未导入视频，已切换为纯音频编辑模式')
}

// ========== 数据加载 ==========

// V3.2.0+dev.20260130.10: 读取任务级字幕时间偏移（无则回退全局）
async function loadSubtitleOffset() {
  if (!activeJobId.value) {
    projectStore.setSubtitleOffset(0)
    return
  }
  try {
    const response = await transcriptionApi.getJobSubtitleTimeOffset(activeJobId.value)
    const offset = response?.offset ?? 0
    projectStore.setSubtitleOffset(offset)
  } catch (error) {
    console.warn('[EditorView] 读取字幕偏移失败，使用默认值 0:', error)
    projectStore.setSubtitleOffset(0)
  }
}

async function resolveIdentity() {
  resolvedProjectId.value = rawProps.projectId || null
  resolvedJobId.value = rawProps.jobId || null

  if (resolvedProjectId.value) {
    try {
      const project = await projectApi.getProject(resolvedProjectId.value)
      const sourceType = String(project?.subtitle_doc?.source_type || '').trim().toLowerCase()
      const mode = String(project?.mode || '').trim().toLowerCase()
      const taskMode = String(project?.task_mode || '').trim().toLowerCase()
      const isProjectOnly = taskMode
        ? taskMode === 'subtitle_edit'
        : (sourceType === 'import' || mode === 'import')
      if (project?.job_id && !isProjectOnly) {
        resolvedJobId.value = project.job_id
      }
    } catch (error) {
      console.warn('[EditorView] 读取项目详情失败，继续使用路由参数:', error)
    }
    return
  }

  if (rawProps.jobId) {
    try {
      const result = await legacyApi.resolveTask(rawProps.jobId)
      const projectId = result?.project_id
      if (projectId) {
        resolvedProjectId.value = projectId
        if (router.currentRoute.value.params.projectId !== projectId) {
          router.replace(`/editor/project/${projectId}`)
        }
        return
      }
      throw new Error(`未返回 project_id（job_id=${rawProps.jobId}）`)
    } catch (error) {
      throw new Error(
        `job_id 转 project_id 失败（job_id=${rawProps.jobId}）：${error?.message || '未知错误'}`
      )
    }
  }
}

// 加载项目数据
async function loadProject() {
  console.log('[loadProject] 开始执行，useEditorV2:', useEditorV2)
  isLoading.value = true
  loadError.value = null

  // 先重置项目状态，确保不同任务数据隔离
  console.log('[loadProject] 重置项目状态')
  projectStore.resetProject()
  structuralSyncStore.$reset()

  try {
    console.log('[loadProject] 开始 resolveIdentity')
    await resolveIdentity()
    console.log('[loadProject] resolveIdentity 完成')
    const projectId = props.projectId
    if (!projectId) {
      throw new Error('缺少 project_id，无法进入编辑器（job 转 project 失败）')
    }
    console.log('[loadProject] projectId:', projectId)

    const initialCapabilitySnapshot = resolveCapabilitySnapshot()
    projectStore.setIdentity({
      projectId,
      jobId: activeJobId.value,
      mode: 'normal',
      taskMode: activeJobId.value ? 'transcribe' : 'subtitle_edit',
      capabilitySnapshot: initialCapabilitySnapshot,
      flavor: selectFlavor(initialCapabilitySnapshot),
    })
    projectStore.setMediaPaths({
      videoPath: null,
      audioPath: null,
    })

    let projectMeta = null
    if (projectId) {
      try {
        projectMeta = await projectApi.getProject(projectId)
      } catch (error) {
        console.warn('[EditorView] 读取项目媒体信息失败，使用状态兜底:', error)
      }
    }
    const projectMetaCapabilitySnapshot = resolveCapabilitySnapshot(
      projectMeta?.capability_snapshot || projectMeta?.capabilitySnapshot || projectStore.meta.capabilitySnapshot
    )
    projectStore.setIdentity({
      capabilitySnapshot: projectMetaCapabilitySnapshot,
      flavor: projectMeta?.flavor || selectFlavor(projectMetaCapabilitySnapshot),
      taskMode: projectMeta?.task_mode || projectStore.meta.taskMode || 'transcribe',
    })
    applyMediaPaths({ project: projectMeta })

    // 纯项目模式（Lite 导入）: 跳过任务状态，改走 project 频道同步
    console.log('[loadProject] 检查模式，activeJobId:', activeJobId.value, 'projectId:', projectId)
    if (!activeJobId.value && projectId) {
      console.log('[loadProject] 进入纯项目模式分支')
      try {
        await proxyVideo.refresh()
      } catch (e) {
        console.warn('[EditorView] 刷新 project 媒体状态失败（初始阶段，忽略）:', e)
      }

      await loadSubtitleOffset()
      const project = projectMeta || (await projectApi.getProject(projectId))
      applyMediaPaths({ project })
      projectStore.setProjectTitle(project?.title || projectStore.meta.title)
      const projectCapabilitySnapshot = resolveCapabilitySnapshot(
        project?.capability_snapshot || project?.capabilitySnapshot || projectStore.meta.capabilitySnapshot
      )
      projectStore.setIdentity({
        capabilitySnapshot: projectCapabilitySnapshot,
        flavor: project?.flavor || selectFlavor(projectCapabilitySnapshot),
        mode: project?.mode || 'normal',
        taskMode: project?.task_mode || 'subtitle_edit',
      })
      await scheduleEditorProjectionReload(projectId, 'load_project_project_mode')
      applyMediaPaths({ project })
      subscribeSSE()

      if (!proxyVideo.isReady.value) {
        startProxyPolling()
      }
      notifyMissingVideoOnce()
      return
    }

    if (!activeJobId.value) {
      throw new Error('缺少任务标识，无法加载转录态项目')
    }

    const jobStatus = await transcriptionApi.getJobStatus(activeJobId.value, true)
    applyMediaPaths({ project: projectMeta, mediaStatus: jobStatus.media_status })

    progressStore.applySnapshot(
      activeJobId.value,
      {
        percent: jobStatus.progress,
        status: jobStatus.status,
        phase: jobStatus.phase,
        phase_percent: jobStatus.phase_percent,
        message: jobStatus.message,
        processed: jobStatus.processed,
        total: jobStatus.total,
      },
      'http_init'
    )

    projectStore.setIdentity({
      jobId: activeJobId.value,
      projectId: props.projectId || projectStore.meta.projectId,
      taskMode: projectMeta?.task_mode || 'transcribe',
    })
    projectStore.patchMeta({
      filename: jobStatus.filename || '未知文件',
    })
    projectStore.setProjectTitle(jobStatus.title || '')
    projectStore.setProjectDuration(jobStatus.media_status?.video?.duration || 0)

    try {
      await proxyVideo.refresh()
    } catch (e) {
      console.warn('[EditorView] 刷新 Proxy 状态失败（初始阶段，忽略）:', e)
    }

    // V2 下不再恢复旧字幕缓存，避免首屏出现旧真源与新真源双装载。
    applyMediaPaths({ project: projectMeta, mediaStatus: jobStatus.media_status })

    await loadSubtitleOffset()

    if (jobStatus.status === 'finished') {
      await loadTranscribingSegments('load_project_finished')
      if (!proxyVideo.isReady.value) {
        subscribeSSE()
        startProxyPolling()
      }
    } else if (['processing', 'queued'].includes(jobStatus.status)) {
      await loadTranscribingSegments(`load_project_${jobStatus.status}`)
      subscribeSSE()
      startProgressPolling()
      startProxyPolling()
    } else if (jobStatus.status === 'paused') {
      await loadTranscribingSegments('load_project_paused')
      subscribeSSE()
      refreshTaskProgress()
      startProxyPolling()
    } else if (jobStatus.status === 'created') {
      subscribeSSE()
      startProxyPolling()
    } else if (['canceled', 'force_canceled'].includes(jobStatus.status)) {
      await syncSegmentsFromBackendAsSource({
        reason: 'load_project_canceled_terminal',
      })
    } else if (jobStatus.status === 'failed') {
      await loadTranscribingSegments('load_project_failed')
    }

    notifyMissingVideoOnce()
  } catch (error) {
    console.error('[EditorView] 加载项目失败:', error)

    if (error.response?.status === 404 && activeJobId.value) {
      console.warn(`[EditorView] 任务已在后端删除: ${activeJobId.value}`)
      try {
        await taskStore.deleteTask(activeJobId.value)
        loadError.value = '任务不存在（已被删除），本地记录已清理'
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
async function loadTranscribingSegments(reason = 'unknown') {
  const projectId = props.projectId || projectStore.meta.projectId
  if (!projectId) {
    throw new Error('缺少 project_id，无法拉取字幕真源')
  }
  return scheduleEditorProjectionReload(projectId, reason)
}

// V3.2.4+dev.20260222.14: 终态同步必须以后端为唯一数据源
async function syncSegmentsFromBackendAsSource({ reason = 'unknown' } = {}) {
  await loadTranscribingSegments(reason)
}

// V3.2.4+dev.20260222.15: 每次定稿事件都立即触发后端回拉，且高频事件合并为“当前1次+补1次”
async function scheduleRealtimeFinalSync(reason = 'subtitle_final_event') {
  if (useEditorV2) {
    scheduleV2RealtimeAuthoritativeReload(reason)
    return
  }

  realtimeFinalSyncReason = reason
  if (isRealtimeFinalSyncInFlight) {
    hasRealtimeFinalSyncPending = true
    return
  }

  isRealtimeFinalSyncInFlight = true
  try {
    do {
      const currentReason = realtimeFinalSyncReason || reason
      hasRealtimeFinalSyncPending = false
      realtimeFinalSyncReason = null
      try {
        await syncSegmentsFromBackendAsSource({
          reason: currentReason,
        })
      } catch (error) {
        console.warn('[EditorView] 实时定稿后端回拉失败:', error)
      }
    } while (hasRealtimeFinalSyncPending)
  } finally {
    isRealtimeFinalSyncInFlight = false
  }
}

// Segments 转 SRT 格式
function segmentsToSRT(segments) {
  if (!segments || segments.length === 0) return ''
  return segments
    .map((seg, idx) => {
      const start = formatSRTTime(seg.start)
      const end = formatSRTTime(seg.end)
      return `${idx + 1}\n${start} --> ${end}\n${seg.text || ''}\n`
    })
    .join('\n')
}

// SRT 时间格式化
function formatSRTTime(seconds) {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  const ms = Math.round((seconds % 1) * 1000)
  return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')},${ms.toString().padStart(3, '0')}`
}

function toSafeMs(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) {
    return 0
  }
  return Math.round(numeric)
}

function toNormalizedSegmentId(value) {
  const normalized = String(value ?? '').trim()
  return normalized || null
}

function assertEditorV2SnapshotConsistency(serverSegments) {
  if (!useEditorV2) {
    return
  }

  const localBySegmentId = new Map()
  for (const localId of editorDocumentStore.order) {
    const entity = editorDocumentStore.getEntity(localId)
    if (!entity || entity.isDeleted) {
      continue
    }
    const segmentId = toNormalizedSegmentId(editorDocumentStore.getCold(localId)?.segmentId)
    if (!segmentId) {
      throw new Error(`本地仍存在未绑定字幕（localId=${localId}），请稍后重试导出`)
    }
    if (localBySegmentId.has(segmentId)) {
      throw new Error(`本地存在重复 segment_id=${segmentId}，请刷新后重试`)
    }
    localBySegmentId.set(segmentId, {
      text: String(entity.text ?? ''),
      startMs: toSafeMs(entity.startMs),
      endMs: toSafeMs(entity.endMs),
    })
  }

  const serverBySegmentId = new Map()
  for (const segment of (Array.isArray(serverSegments) ? serverSegments : [])) {
    const segmentId = toNormalizedSegmentId(segment?.segment_id)
    if (!segmentId) {
      continue
    }
    if (serverBySegmentId.has(segmentId)) {
      throw new Error(`后端返回重复 segment_id=${segmentId}，请稍后重试`)
    }
    serverBySegmentId.set(segmentId, {
      text: String(segment?.text ?? ''),
      startMs: toSafeMs(Number(segment?.start ?? 0) * 1000),
      endMs: toSafeMs(Number(segment?.end ?? segment?.start ?? 0) * 1000),
    })
  }

  const mismatches = []
  if (localBySegmentId.size !== serverBySegmentId.size) {
    mismatches.push(`数量不一致(local=${localBySegmentId.size}, server=${serverBySegmentId.size})`)
  }

  for (const [segmentId, localSnapshot] of localBySegmentId.entries()) {
    const serverSnapshot = serverBySegmentId.get(segmentId)
    if (!serverSnapshot) {
      mismatches.push(`后端缺少 segment_id=${segmentId}`)
      continue
    }
    const isTextMismatch = localSnapshot.text !== serverSnapshot.text
    const isTimingMismatch = (
      Math.abs(localSnapshot.startMs - serverSnapshot.startMs) > 1
      || Math.abs(localSnapshot.endMs - serverSnapshot.endMs) > 1
    )
    if (isTextMismatch || isTimingMismatch) {
      mismatches.push(`segment_id=${segmentId} 文本/时间不一致`)
    }
  }

  for (const segmentId of serverBySegmentId.keys()) {
    if (!localBySegmentId.has(segmentId)) {
      mismatches.push(`本地缺少 segment_id=${segmentId}`)
    }
  }

  if (mismatches.length > 0) {
    const summary = mismatches.slice(0, 3).join('；')
    throw new Error(`前后端字幕快照不一致：${summary}`)
  }
}

async function runEditorSyncStrict() {
  if (!useEditorV2) {
    return true
  }

  if (strictSyncInFlightPromise) {
    return strictSyncInFlightPromise
  }

  strictSyncInFlightPromise = (async () => {
    // 兼容热更新/旧会话：若实例尚未拿到 flushStrict，则降级组合 flush + reconcile + flush。
    if (typeof editorSyncEngine.flushStrict === 'function') {
      return editorSyncEngine.flushStrict()
    }
    if (typeof editorSyncEngine.flush === 'function') {
      const firstPass = await editorSyncEngine.flush()
      if (!firstPass) {
        return false
      }
      if (typeof editorSyncEngine.reconcile === 'function') {
        await editorSyncEngine.reconcile()
        return editorSyncEngine.flush()
      }
      return firstPass
    }
    throw new Error('同步引擎不可用：缺少 flush/flushStrict')
  })()

  try {
    return await strictSyncInFlightPromise
  } finally {
    strictSyncInFlightPromise = null
  }
}

function normalizeProjectSegmentEvent(data) {
  const segment = data?.segment
  const segmentId = data?.segment_id || segment?.segment_id
  if (!segmentId) {
    return null
  }

  const rawStart = Number(segment?.start ?? 0)
  const rawEnd = Number(segment?.end ?? rawStart)
  const safeStart = Number.isFinite(rawStart) ? rawStart : 0
  const safeEnd = Number.isFinite(rawEnd) ? Math.max(safeStart, rawEnd) : safeStart
  const legacyIndexRaw = segment?.legacy_index ?? segment?.sentence_index
  const legacyIndex = Number.isFinite(Number(legacyIndexRaw))
    ? Number(legacyIndexRaw)
    : null

  return {
    segmentId: String(segmentId),
    sentenceIndex: legacyIndex,
    text: String(segment?.text ?? ''),
    start: projectStore.toDisplayTime(safeStart),
    end: projectStore.toDisplayTime(safeEnd),
    isModified: Boolean(segment?.is_modified),
    originalText: segment?.original_text ?? null,
    words: Array.isArray(segment?.words) ? segment.words : [],
    confidence: segment?.confidence ?? null,
    displayConfidence: segment?.display_confidence,
    confidenceSource: segment?.confidence_source,
    warningType: segment?.warning_type || 'none',
    source: segment?.source_type || segment?.source || 'project_stream',
  }
}

function handleProjectSubtitleUpsert(data, eventType = 'subtitle.project_upsert') {
  if (shouldIgnoreProjectAckEvent(data, editorSessionStore.sessionId)) {
    return true
  }

  const rawSegment = data?.segment
  if (!rawSegment || typeof rawSegment !== 'object') {
    throw new Error(`[EditorView] ${eventType} 缺少 segment 载荷`)
  }

  const applied = upsertServerSegment(editorDocumentStore, {
    ...rawSegment,
    segment_id: data?.segment_id || rawSegment.segment_id,
  })
  if (!applied) {
    throw new Error(
      `[EditorView] ${eventType} 无法投影到 editorDocumentStore（segment_id=${data?.segment_id || rawSegment.segment_id || 'unknown'}）`
    )
  }
  if (shouldScheduleAuthoritativeReloadForProjectModeEvent({
    useEditorV2,
    activeJobId: activeJobId.value,
  })) {
    scheduleV2RealtimeAuthoritativeReload(
      `${eventType}_${data?.segment_id || rawSegment.segment_id || 'unknown'}`
    )
  }
  return true
}

function handleProjectSubtitleDelete(data) {
  if (shouldIgnoreProjectAckEvent(data, editorSessionStore.sessionId)) {
    return true
  }

  const segmentId = data?.segment_id ?? data?.segment?.segment_id
  if (!segmentId) {
    throw new Error('[EditorView] subtitle.project_delete 缺少 segment_id')
  }

  const deleted = deleteServerSegment(editorDocumentStore, segmentId)
  if (!deleted) {
    if (data?.source === 'project_api' && data?.is_update === true) {
      return true
    }
    throw new Error(`[EditorView] subtitle.project_delete 未命中本地绑定（segment_id=${segmentId}）`)
  }
  if (shouldScheduleAuthoritativeReloadForProjectModeEvent({
    useEditorV2,
    activeJobId: activeJobId.value,
  })) {
    scheduleV2RealtimeAuthoritativeReload(`subtitle.deleted.project_${segmentId}`)
  }
  return true
}

// ========== SSE 实时更新 ==========

function subscribeSSE() {
  const jobId = activeJobId.value
  const projectId = props.projectId
  if (!jobId && !projectId) {
    return
  }
  // 如果已经订阅，先取消避免重复订阅
  if (sseUnsubscribe) {
    sseUnsubscribe()
    sseUnsubscribe = null
  }

  if (!jobId && projectId) {
    sseUnsubscribe = sseChannelManager.subscribeProject(projectId, {
      onSubtitleUpdated(data) {
        handleProjectSubtitleUpsert(data, 'subtitle.edited.project')
      },
      onSubtitleAdded(data) {
        handleProjectSubtitleUpsert(data, 'subtitle.added.project')
      },
      onSubtitleDeleted(data) {
        handleProjectSubtitleDelete(data)
      },
    })
    return
  }

  // 使用 sseChannelManager 订阅单任务频道
  sseUnsubscribe = sseChannelManager.subscribeJob(jobId, {
    onInitialState(data) {
      progressStore.applySnapshot(jobId, data, 'sse_init')
      // 如果附带 Proxy 状态，直接应用到 proxyVideo，避免断线后进度丢失
      if (data.proxy) {
        proxyVideo.applySnapshot(data.proxy)
      }
    },

    // 新增：被服务器踢出时提示用户并停止当前窗口的 SSE
    force_disconnect(data) {
      const reason = data?.reason || '该任务已在其他窗口打开，当前窗口的 SSE 已断开'
      console.warn('[EditorView] 收到强制断开:', reason)
      taskStore.updateTaskSSEStatus(jobId, false, reason)
      cleanupSSE()
      // 简单弹窗提示用户（后续可替换为全局提示组件）
      alert(reason)
    },

    onProgress(data) {
      progressStore.applySseProgress(jobId, data)
      if (data.detail) {
        projectStore.updateDualStreamProgressFromSSE({
          fastStream: Math.round(data.detail.fast || 0),
          slowStream: Math.round(data.detail.slow || 0),
          totalChunks: data.total || projectStore.dualStreamProgress.totalChunks,
        })
      }
    },

    async onComplete(data) {
      if (isCancelPending.value) {
        isCancelPending.value = false
        stopCancelTimeoutPolling()
      }
      progressStore.markStatus(jobId, 'finished', {
        percent: 100,
        phase: 'complete',
        updated_at: data.updated_at ?? data.timestamp,
        state_seq: data.state_seq,
      })
      taskStore.updateTaskStatus(jobId, 'finished', null, {
        updated_at: data.updated_at ?? data.timestamp,
        state_seq: data.state_seq,
      })

      // V3.2.4+dev.20260222.14: 完成态不再本地“草稿改定稿”，必须立即以后端最终口径覆盖
      await syncSegmentsFromBackendAsSource({
        reason: 'signal_job_complete',
      })

      stopProgressPolling()
      startProxyPolling()
      // V3.1.2+dev.20260113.04: 不关闭SSE连接，保持接收Proxy转码事件
      // 转录任务完成后，720p转码可能还在进行，需要继续接收 proxy_complete 事件
      // cleanupSSE()
    },

    onFailed(data) {
      if (isCancelPending.value) {
        isCancelPending.value = false
        stopCancelTimeoutPolling()
      }
      progressStore.markStatus(jobId, 'failed', {
        message: data.message,
        updated_at: data.updated_at ?? data.timestamp,
        state_seq: data.state_seq,
      })
      taskStore.updateTaskStatus(jobId, 'failed', null, {
        updated_at: data.updated_at ?? data.timestamp,
        state_seq: data.state_seq,
      })
      taskStore.updateTaskSSEStatus(jobId, true, data.message || '转录失败')
      stopProgressPolling()
      // 任务失败后关闭SSE连接
      cleanupSSE()
    },

    onPaused(data) {
      progressStore.markStatus(jobId, 'paused', {
        updated_at: data.updated_at ?? data.timestamp,
        state_seq: data.state_seq,
      })
      taskStore.updateTaskStatus(jobId, 'paused', null, {
        updated_at: data.updated_at ?? data.timestamp,
        state_seq: data.state_seq,
      })
      // 保持SSE连接和进度显示
    },

    onPausePending(data) {
      progressStore.markStatus(jobId, 'pausing', {
        updated_at: data.updated_at ?? data.timestamp,
        state_seq: data.state_seq,
      })
      taskStore.updateTaskStatus(jobId, 'pausing', null, {
        updated_at: data.updated_at ?? data.timestamp,
        state_seq: data.state_seq,
      })
    },

    onCanceling(data) {
      isCancelPending.value = true
      startCancelTimeoutPolling()
      progressStore.markStatus(jobId, 'canceling', {
        updated_at: data.updated_at ?? data.timestamp,
        state_seq: data.state_seq,
      })
      taskStore.updateTaskStatus(jobId, 'canceling', null, {
        updated_at: data.updated_at ?? data.timestamp,
        state_seq: data.state_seq,
      })
    },

    async onCanceled(data) {
      await handleCancelTerminal(data, 'canceled')
    },

    async onForceCanceled(data) {
      await handleCancelTerminal(data, 'force_canceled')
    },

    onResumed(data) {
      // 新增：处理任务恢复信号
      progressStore.markStatus(jobId, data.status || 'queued', {
        updated_at: data.updated_at ?? data.timestamp,
        state_seq: data.state_seq,
      })
      taskStore.updateTaskStatus(jobId, data.status || 'queued', null, {
        updated_at: data.updated_at ?? data.timestamp,
        state_seq: data.state_seq,
      })
      startProgressPolling()
    },

    onConnected() {
      taskStore.updateTaskSSEStatus(jobId, true)
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

    // V3.2.4+dev.20260222.14: onSubtitleUpdate 仅用于旧事件，避免新事件双路径改写导致状态漂移
    onSubtitleUpdate(data) {
      // 新架构事件（draft/replace_chunk/finalized/restored）统一走专用处理器
      if (data?.chunk_index !== undefined && data?.chunk_index !== null) {
        return
      }
      // 仅处理旧架构字幕事件（sv_sentence/whisper_patch/llm_proof 等）
      if (data.sentence || data.sentence_index !== undefined) {
        handleStreamingSubtitle(data)
      }
    },

    // Phase 5: 草稿字幕事件（快流/SenseVoice）
    onDraft(data) {
      handleDraftSubtitle(data)
    },

    // Phase 5: 替换 Chunk 事件（慢流/Whisper）
    onReplaceChunk(data) {
      handleReplaceChunk(data)
    },

    // V3.1.0: 恢复字幕事件（断点续传后恢复）
    onRestored(data) {
      handleRestoredChunk(data)
    },

    // V3.5: 极速模式定稿事件
    onFinalized(data) {
      handleFinalizedSubtitle(data)
    },

    onRevised(data) {
      handleRevisedSubtitle(data)
    },

    onSpeakerProfiles(data) {
      handleSpeakerProfiles(data)
    },

    onSubtitleAdded(data) {
      handleSubtitleAdded(data)
    },

    onSubtitleDeleted(data) {
      handleSubtitleDeleted(data)
    },

    onSubtitleEdited(data) {
      handleSubtitleEdited(data)
    },

    onLangidEvent(eventType, data) {
      if (eventType === 'preprocessing.langid.error') {
        ElMessage.error(data?.message || '语言检测失败')
      }
    },

    onSpeakerEvent(eventType, data) {
      if (eventType === 'preprocessing.speaker.error') {
        ElMessage.error(data?.message || '声纹提取失败')
      }
    },

    // === Proxy 视频转码事件（转发到 useProxyVideo）===
    onAnalyzeComplete(data) {
      proxyVideo.handlers.onAnalyzeComplete(data)
    },
    onRemuxProgress(data) {
      proxyVideo.handlers.onRemuxProgress(data)
    },
    onRemuxComplete(data) {
      proxyVideo.handlers.onRemuxComplete(data)
    },
    onPreview360pProgress(data) {
      proxyVideo.handlers.onPreview360pProgress(data)
    },
    onPreview360pComplete(data) {
      proxyVideo.handlers.onPreview360pComplete(data)
    },
    onProxyProgress(data) {
      proxyVideo.handlers.onProxyProgress(data)
    },
    onProxyComplete(data) {
      proxyVideo.handlers.onProxyComplete(data)
    },
    onProxyError(data) {
      proxyVideo.handlers.onProxyError(data)
    },
  })
}

// 清理SSE连接
function cleanupSSE() {
  stopCancelTimeoutPolling()
  if (sseUnsubscribe) {
    sseUnsubscribe()
    sseUnsubscribe = null
  }
}

function stopCancelTimeoutPolling() {
  if (cancelTimeoutTimer) {
    clearTimeout(cancelTimeoutTimer)
    cancelTimeoutTimer = null
  }
}

function isCancelTerminalStatus(status) {
  return CANCEL_TERMINAL_STATUSES.has(status)
}

async function handleCancelTerminal(data, fallbackStatus = 'canceled') {
  if (!activeJobId.value) {
    return
  }
  const terminalStatus = data?.status || fallbackStatus
  isCancelPending.value = false
  stopCancelTimeoutPolling()
  // V3.2.4+dev.20260222.14: 取消终态同样以后端返回为准，不做本地草稿转定稿
  if (['canceled', 'force_canceled'].includes(terminalStatus)) {
    await syncSegmentsFromBackendAsSource({
      reason: `signal_${terminalStatus}`,
    })
  }
  progressStore.markStatus(activeJobId.value, terminalStatus, {
    updated_at: data?.updated_at ?? data?.timestamp,
    state_seq: data?.state_seq,
  })
  taskStore.updateTaskStatus(activeJobId.value, terminalStatus, null, {
    updated_at: data?.updated_at ?? data?.timestamp,
    state_seq: data?.state_seq,
  })
  stopProgressPolling()
  cleanupSSE()
}

function startCancelTimeoutPolling() {
  if (!isCancelPending.value || !activeJobId.value) return
  stopCancelTimeoutPolling()
  cancelTimeoutTimer = setTimeout(async () => {
    if (!isCancelPending.value) return
    console.warn('[EditorView] canceling 超时，主动拉取状态:', activeJobId.value)
    try {
      const snapshot = await transcriptionApi.getJobStatus(activeJobId.value, true)
      const status = snapshot?.status || ''
      if (isCancelTerminalStatus(status)) {
        await handleCancelTerminal(
          {
            status,
            updated_at: snapshot?.updated_at,
            timestamp: snapshot?.timestamp,
          },
          status
        )
        return
      }
      startCancelTimeoutPolling()
    } catch (error) {
      console.warn('[EditorView] canceling 兜底查询失败，下一轮重试:', error)
      startCancelTimeoutPolling()
    }
  }, CANCEL_TIMEOUT_MS)
}

// 处理流式字幕更新（旧版兼容，已弃用）
// V3.1.2+dev.20260112.01: 此函数已弃用，保留仅供旧版 SSE 事件兼容
// 新架构使用专用处理器：handleDraftSubtitle、handleReplaceChunk 等
// eslint-disable-next-line no-unused-vars
function handleStreamingSubtitle(data) {
  if (!data) return

  const sentence = data.sentence || {}
  const sentenceIndex = data.sentence_index ?? data.index ?? sentence.index
  if (sentenceIndex === undefined || sentenceIndex === null) {
    throw new Error('[EditorView] subtitle.legacy_update 缺少 sentence_index')
  }

  const applied = applySentencePatch(editorDocumentStore, {
    ...sentence,
    index: sentenceIndex,
    sentence_index: sentenceIndex,
    text: sentence.text ?? data.text ?? data.content,
    start: sentence.start ?? data.start_time ?? data.start,
    end: sentence.end ?? data.end_time ?? data.end,
    confidence: sentence.confidence ?? data.confidence,
    display_confidence: sentence.display_confidence ?? data.display_confidence,
    confidence_source: sentence.confidence_source ?? data.confidence_source,
    warning_type: sentence.warning_type ?? data.warning_type ?? 'none',
    source: data.source ?? data.event_type ?? 'unknown',
    is_modified: sentence.is_modified ?? false,
    original_text: sentence.original_text ?? null,
  })
  if (!applied) {
    throw new Error(
      `[EditorView] subtitle.legacy_update 未命中新内核字幕（sentence_index=${sentenceIndex}）`
    )
  }
}

// Phase 5: 处理草稿字幕（快流/SenseVoice）
function handleDraftSubtitle(data) {
  if (!data) return

  if (useEditorV2) {
    editorEventProjector.projectDraft(data)
    return
  }

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
    display_confidence: sentence.display_confidence, // V3.1.2: 映射后准确率
    confidence_source: sentence.confidence_source, // V3.1.2: 置信度来源
    words: sentence.words || [],
    warning_type: sentence.warning_type || 'none',
    is_modified: sentence.is_modified ?? false,
    original_text: sentence.original_text ?? null,
  }

  // 调用 projectStore 的草稿处理方法，传递两个参数
  projectStore.appendOrUpdateDraft(chunkIndex, sentenceData)

}

// Phase 5: 处理替换 Chunk（慢流/Whisper）
function handleReplaceChunk(data) {
  if (!data) return

  const chunkIndex = data.chunk_index
  if (useEditorV2) {
    const applied = editorEventProjector.projectReplaceChunk(data)
    if (!applied) {
      throw new Error(
        `[EditorView] subtitle.replace_chunk 未能应用到新内核（chunk_index=${chunkIndex ?? 'unknown'}）`
      )
    }
  }
  void scheduleRealtimeFinalSync(`subtitle_replace_chunk_${chunkIndex ?? 'unknown'}`)
}

function handleSubtitleAdded(data) {
  if (!data) return

  if (useEditorV2) {
    if (data?.segment && typeof data.segment === 'object') {
      handleProjectSubtitleUpsert(data, 'subtitle.added')
      return
    }

    const applied = upsertServerSegment(editorDocumentStore, data)
    if (!applied) {
      throw new Error('[EditorView] subtitle.added 无法投影到 editorDocumentStore')
    }
    return
  }

  handleStreamingSubtitle(data)
}

/**
 * V3.1.0: 处理恢复的字幕（断点续传后恢复）
 * 后端数据格式: { chunk_index, sentences: [...], is_restore: true }
 */
function handleRestoredChunk(data) {
  if (!data) return

  const applied = editorEventProjector.projectRestored(data)
  if (!applied) {
    throw new Error(
      `[EditorView] subtitle.restored 未能应用到新内核（chunk_index=${data.chunk_index ?? 'unknown'}）`
    )
  }
  scheduleV2RealtimeAuthoritativeReload(`subtitle_restored_${data.chunk_index ?? 'unknown'}`)
}

function handleSubtitleDeleted(data) {
  if (!data) return
  const segmentId = data.segment_id ?? data.segment?.segment_id
  if (!segmentId) {
    throw new Error('[EditorView] subtitle.deleted 缺少 segment_id')
  }
  handleProjectSubtitleDelete(data)
}

function handleSubtitleEdited(data) {
  if (!data) return

  if (data?.segment && typeof data.segment === 'object') {
    handleProjectSubtitleUpsert(data, 'subtitle.edited')
    return
  }

  const sentence = data.sentence || {}
  const sentenceIndex = data.index ?? data.sentence_index ?? sentence.index
  const segmentId = data.segment_id ?? sentence.segment_id
  if ((sentenceIndex === undefined || sentenceIndex === null) && !segmentId) {
    throw new Error('[EditorView] subtitle.edited 缺少 sentence_index/segment_id')
  }

  const applied = applySentencePatch(editorDocumentStore, {
    ...sentence,
    ...(sentenceIndex !== undefined && sentenceIndex !== null
      ? {
          index: sentenceIndex,
          sentence_index: sentenceIndex,
        }
      : {}),
    ...(segmentId ? { segment_id: segmentId } : {}),
  })
  if (!applied) {
    throw new Error(
      `[EditorView] subtitle.edited 未命中新内核字幕（key=${sentenceIndex ?? segmentId ?? 'unknown'}）`
    )
  }
}

/**
 * V3.5: 处理极速模式定稿字幕
 * 后端数据格式: { index, chunk_index, sentence: {...}, mode: 'sensevoice_only' }
 */
function handleFinalizedSubtitle(data) {
  if (!data) return

  const chunkIndex = data.chunk_index
  const sentenceIndex = data.index
  if (useEditorV2) {
    const applied = editorEventProjector.projectFinalized(data)
    if (!applied) {
      throw new Error(
        `[EditorView] subtitle.finalized 未能应用到新内核（key=${chunkIndex ?? sentenceIndex ?? 'unknown'}）`
      )
    }
  }
  void scheduleRealtimeFinalSync(`subtitle_finalized_${chunkIndex ?? sentenceIndex ?? 'unknown'}`)
}

function handleRevisedSubtitle(data) {
  if (!data) return

  if (useEditorV2) {
    let applied = false

    if (data.sentence) {
      applied = applySentencePatch(editorDocumentStore, {
        ...data.sentence,
        index: data.index ?? data.sentence.index ?? data.sentence.sentenceIndex,
      }) || applied
    }

    if (Array.isArray(data.sentences)) {
      data.sentences.forEach((item) => {
        applied = applySentencePatch(editorDocumentStore, {
          ...item,
          index: item.index ?? item.sentenceIndex,
        }) || applied
      })
    }

    if (!applied) {
      throw new Error('[EditorView] subtitle.revised 未能应用到新内核')
    }
    scheduleV2RealtimeAuthoritativeReload('subtitle_revised')
    return
  }

  const revisedPayload = { ...data }
  if (data.sentence) {
    revisedPayload.sentence = {
      ...data.sentence,
      index: data.index ?? data.sentence.index ?? data.sentence.sentenceIndex,
    }
  }
  if (Array.isArray(data.sentences)) {
    revisedPayload.sentences = data.sentences.map((item) => ({
      ...item,
      index: item.index ?? item.sentenceIndex,
    }))
  }
  projectStore.applyRevisedSubtitle(revisedPayload)

}

function handleSpeakerProfiles(data) {
  if (!data) return
  projectStore.applySpeakerProfiles(data)
}

// 刷新任务进度（用于SSE重连后的状态同步）
async function refreshTaskProgress() {
  if (!activeJobId.value) {
    return
  }
  try {
    const jobStatus = await transcriptionApi.getJobStatus(activeJobId.value, true)

    progressStore.applySnapshot(
      activeJobId.value,
      {
        percent: jobStatus.progress,
        status: jobStatus.status,
        phase: jobStatus.phase,
        phase_percent: jobStatus.phase_percent,
        message: jobStatus.message,
        processed: jobStatus.processed,
        total: jobStatus.total,
      },
      'http_refresh'
    )
  } catch (error) {
    console.warn('[EditorView] 刷新任务进度失败:', error)
  }
}

function startProgressPolling() {
  if (!activeJobId.value) {
    return
  }
  stopProgressPolling()
  // 仅在 SSE 长时间无心跳时触发 HTTP 兜底
  progressPollTimer = setInterval(async () => {
    if (!isTranscribing.value) {
      stopProgressPolling()
      return
    }
    const rawState = progressStore.getRawState(activeJobId.value)
    if (Date.now() - (rawState.lastSseAt || 0) < 10000) {
      return
    }

    try {
      const snapshot = await transcriptionApi.getJobStatus(activeJobId.value, true)
      progressStore.applySnapshot(
        activeJobId.value,
        {
          percent: snapshot.progress,
          status: snapshot.status,
          phase: snapshot.phase,
          phase_percent: snapshot.phase_percent,
          message: snapshot.message,
          processed: snapshot.processed,
          total: snapshot.total,
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
  if (!mediaIdentityId.value) {
    return
  }
  try {
    const status = await mediaApi.getProxyStatus(mediaIdentityId.value)
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
  if (!mediaIdentityId.value) {
    return
  }
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
    if (mediaIdentityId.value) {
      const srtSegments = useEditorV2
        ? projectStore.applyOffsetToSegments(await fetchLatestSegments())
        : null
      const srtContent = useEditorV2
        ? segmentsToSRT(srtSegments)
        : projectStore.generateSRT()
      await mediaApi.saveSRTContent(mediaIdentityId.value, srtContent)
    }
    if (!useEditorV2) {
      await projectStore.saveProject()
    } else {
      await runEditorSyncStrict()
    }
    lastSaved.value = Date.now()
  } catch (error) {
    console.error('[EditorView] 保存失败:', error)
    alert('保存失败: ' + (error.message || '未知错误'))
  } finally {
    saving.value = false
  }
}

// ========== 任务控制 ==========

async function pauseTranscription() {
  if (!activeJobId.value) return
  try {
    const result = await transcriptionApi.pauseJob(activeJobId.value)
    if (result?.task) {
      taskStore.applyTaskSnapshot(result.task)
      progressStore.markStatus(activeJobId.value, result.task.status || 'paused', {
        updated_at: result.task.updated_at,
      })
    } else {
      progressStore.markStatus(activeJobId.value, 'paused')
      taskStore.updateTaskStatus(activeJobId.value, 'paused', null, { isServer: false })
    }
  } catch (error) {
    console.error('暂停任务失败:', error)
  }
}

async function resumeTranscription() {
  if (!activeJobId.value) return
  try {
    // 使用新的 resumeJob API，恢复暂停的任务（重新加入队列）
    const result = await transcriptionApi.resumeJob(activeJobId.value)

    if (result?.task) {
      taskStore.applyTaskSnapshot(result.task)
      progressStore.markStatus(activeJobId.value, result.task.status || 'queued', {
        updated_at: result.task.updated_at,
      })
    } else {
      // 根据后端返回值设置状态（应该是 queued，而不是 processing）
      progressStore.markStatus(activeJobId.value, result.status || 'queued')
      taskStore.updateTaskStatus(activeJobId.value, result.status || 'queued', null, { isServer: false })
    }

    await refreshTaskProgress()
    startProgressPolling()
  } catch (error) {
    console.error('恢复任务失败:', error)
  }
}

async function cancelTranscription() {
  if (!activeJobId.value) return
  if (!confirm('确定要取消当前转录任务吗?')) return
  try {
    const result = await transcriptionApi.cancelJob(activeJobId.value, false)
    const apiStatus = result?.task?.status || result?.status || 'canceled'
    if (result?.task) {
      taskStore.applyTaskSnapshot(result.task)
      progressStore.markStatus(activeJobId.value, apiStatus, {
        updated_at: result.task.updated_at,
      })
    } else {
      progressStore.markStatus(activeJobId.value, apiStatus)
      taskStore.updateTaskStatus(activeJobId.value, apiStatus, null, { isServer: false })
    }
    if (isCancelTerminalStatus(apiStatus)) {
      await handleCancelTerminal(
        {
          status: apiStatus,
          updated_at: result?.task?.updated_at,
          timestamp: Date.now(),
          state_seq: result?.state_seq,
        },
        apiStatus
      )
    } else {
      // 兼容旧后端 canceling 语义
      isCancelPending.value = true
      startCancelTimeoutPolling()
    }
  } catch (error) {
    isCancelPending.value = false
    stopCancelTimeoutPolling()
    console.error('取消任务失败:', error)
  }
}

// ========== 撤销/重做 ==========

function undo() {
  subtitleListRef.value?.runWithFlip?.(() => editorCommandBus.undo())
    ?? editorCommandBus.undo()
}
function redo() {
  subtitleListRef.value?.runWithFlip?.(() => editorCommandBus.redo())
    ?? editorCommandBus.redo()
}

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
  let segments = []
  try {
    segments = await fetchLatestSegments()
  } catch (error) {
    const reason = error?.message || '字幕同步未完成，请稍后重试'
    alert(`导出失败：${reason}`)
    return
  }
  const displaySegments = projectStore.applyOffsetToSegments(segments || [])
  if (!displaySegments || displaySegments.length === 0) {
    alert('导出失败：后端未返回字幕数据')
    return
  }

  let content = ''
  let filename = projectName.value.replace(/\.[^/.]+$/, '')

  switch (format) {
    case 'srt':
      content = segmentsToSRT(displaySegments)
      filename += '.srt'
      break
    case 'ass':
      await handleASSExport(displaySegments)
      return
    case 'vtt':
      content = generateVTTFromSegments(displaySegments)
      filename += '.vtt'
      break
    case 'txt':
      content = displaySegments.map((s) => s.text).join('\n')
      filename += '.txt'
      break
    case 'json':
      content = JSON.stringify(displaySegments, null, 2)
      filename += '.json'
      break
  }

  downloadFile(content, filename)
}

async function fetchLatestSegments() {
  let pending = 0
  let syncErrors = null
  let syncErrorCount = 0
  let structuralErrorCount = 0

  if (useEditorV2) {
    await runEditorSyncStrict()
    pending = Array.isArray(editorSyncEngine.pendingCommands)
      ? editorSyncEngine.pendingCommands.length
      : Number(editorSyncEngine.pendingCommands?.value?.length || 0)
  } else {
    // 旧链路仍保留给非 V2 模式；Phase F-0 运行时不再使用旧 undo/redo 同步。
    await forceSyncNow()
    // 等待结构性操作（删除/插入/切分/合并）飞行中请求落地
    await structuralSyncStore.waitAll()
    // 第二轮：结构性操作落地后再次冲洗，覆盖“新增后立刻编辑”的补写场景
    await forceSyncNow()

    pending = subtitleDocumentStore.pendingCount()
    syncErrors = subtitleDocumentStore.syncErrors
    syncErrorCount = Number(syncErrors?.size || 0)
    structuralErrorCount = structuralSyncStore.errorCount
  }

  if (pending > 0 || syncErrorCount > 0 || structuralErrorCount > 0) {
    let firstErrorDetail = ''
    if (syncErrorCount > 0 && typeof syncErrors?.entries === 'function') {
      const firstError = syncErrors.entries().next().value
      if (firstError) {
        firstErrorDetail = `，首条错误: [${firstError[0]}] ${firstError[1]}`
      }
    }
    if (structuralErrorCount > 0 && !firstErrorDetail) {
      const firstStructErr = structuralSyncStore.syncErrors.entries().next().value
      if (firstStructErr) {
        firstErrorDetail = `，首条结构性错误: [${firstStructErr[1].type}] ${firstStructErr[1].error}`
      }
    }
    throw new Error(`仍有未同步修改（待同步 ${pending} 条，错误 ${syncErrorCount + structuralErrorCount} 条${firstErrorDetail}）`)
  }

  const projectId = props.projectId || projectStore.meta.projectId
  if (!projectId) {
    throw new Error('缺少 project_id：无法导出，请从任务列表重新打开并完成任务到项目转换')
  }
  if (!useEditorV2) {
    // 旧模式仍保留本地快照保存；V2 已由同步引擎成为唯一持久化入口。
    await projectStore.saveProject()
  }
  const segments = await projectApi.getSubtitles(projectId)
  if (useEditorV2) {
    assertEditorV2SnapshotConsistency(segments)
  }
  return Array.isArray(segments)
    ? segments.map((segment, index) => ({
        id: segment.legacy_index ?? index,
        start: segment.start,
        end: segment.end,
        text: segment.text,
      }))
    : []
}

async function handleASSExport(segments) {
  try {
    const srtContent = segmentsToSRT(segments)
    const projectId = props.projectId || projectStore.meta.projectId
    if (!projectId) {
      throw new Error('缺少 project_id：无法导出 ASS，请从任务列表重新打开并完成任务到项目转换')
    }
    const exported = await projectApi.exportSubtitles(projectId, 'ass')
    const fileName = `${projectName.value.replace(/\.[^/.]+$/, '')}.ass`
    downloadFile(exported?.content || '', fileName)
    return
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
  e.preventDefault()
  isResizing.value = true
  document.addEventListener('mousemove', onResize)
  document.addEventListener('mouseup', stopResize)
}

function onResize(e) {
  if (!isResizing.value) return
  pendingSidebarWidth = Math.max(280, Math.min(600, window.innerWidth - e.clientX))
  if (sidebarResizeFrameId !== 0) return

  sidebarResizeFrameId = requestAnimationFrame(() => {
    sidebarResizeFrameId = 0
    if (pendingSidebarWidth !== null) {
      sidebarWidth.value = pendingSidebarWidth
      pendingSidebarWidth = null
    }
  })
}

function stopResize() {
  isResizing.value = false
  document.removeEventListener('mousemove', onResize)
  document.removeEventListener('mouseup', stopResize)
  if (sidebarResizeFrameId !== 0) {
    cancelAnimationFrame(sidebarResizeFrameId)
    sidebarResizeFrameId = 0
  }
  if (pendingSidebarWidth !== null) {
    sidebarWidth.value = pendingSidebarWidth
    pendingSidebarWidth = null
  }
  // 保存用户偏好
  localStorage.setItem('editor-sidebar-width', sidebarWidth.value.toString())
}

// ========== 事件处理 ==========

function handleVideoError(error) {
  if (!projectStore.meta.videoPath && !proxyVideo.currentUrl.value) {
    console.warn('[EditorView] 当前为纯音频模式，忽略视频错误:', error)
    return
  }
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
  ElMessage.success({
    message: '后台转码已开始，完成后将自动替换',
    duration: 5000,
  })
}

async function handleCheckVideoStatus() {
  try {
    // 刷新 proxyVideo 状态（从后端重新获取状态）
    await proxyVideo.refresh()
  } catch (error) {
    console.error('[EditorView] 刷新 Proxy 视频状态失败:', error)
  }
}

function handleResolutionChange(resolution) {
  // 更新 projectStore 的视频信息
  projectStore.setCurrentResolution(resolution)
}

function loadEditorInteractionPreferences() {
  // 加载快捷键启用开关（仅保存后生效）
  const savedShortcutEnabled = localStorage.getItem(SHORTCUT_ENABLED_PREF_KEY)
  const resolvedShortcutEnabled = savedShortcutEnabled === null ? true : savedShortcutEnabled === 'true'
  shortcutEnabled.value = resolvedShortcutEnabled
  advancedConfig.value.general.enable_shortcuts = resolvedShortcutEnabled

  // 加载快捷键映射（缺失/异常时回退默认）
  try {
    const savedShortcutConfig = localStorage.getItem(SHORTCUT_CONFIG_PREF_KEY)
    const parsed = savedShortcutConfig ? JSON.parse(savedShortcutConfig) : {}
    const normalized = normalizeEditorShortcutConfig(parsed)
    shortcutConfig.value = normalized
    advancedConfig.value.shortcuts = { ...normalized }
  } catch {
    const fallback = normalizeEditorShortcutConfig({})
    shortcutConfig.value = fallback
    advancedConfig.value.shortcuts = { ...fallback }
  }

  const savedAutoResume = localStorage.getItem(SUBTITLE_FOLLOW_AUTO_RESUME_PREF_KEY)
  const resolved = savedAutoResume === null ? true : savedAutoResume === 'true'
  subtitleFollowAutoResumeEnabled.value = resolved
  advancedConfig.value.general.subtitle_follow_auto_resume = resolved

  // 加载隐藏波形刻度设置
  const savedHideScale = localStorage.getItem(HIDE_TIMELINE_SCALE_PREF_KEY)
  hideTimelineScale.value = savedHideScale === 'true'
  advancedConfig.value.general.hide_timeline_scale = hideTimelineScale.value

  // V3.2.4+dev.20260302.02: 加载合并分隔符设置
  try {
    const savedSeparator = localStorage.getItem('editor-merge-separator')
    if (savedSeparator) {
      const config = JSON.parse(savedSeparator)
      advancedConfig.value.general.merge_separator = config.type || 'space'
      advancedConfig.value.general.merge_separator_custom = config.custom || ''
    }
  } catch {
    // 解析失败使用默认值
  }
}

function resolveCapabilitySnapshot(snapshot = null) {
  if (snapshot && typeof snapshot === 'object') {
    return snapshot
  }
  return buildDefaultCapabilitySnapshot()
}

// ========== 字幕全局偏移设置 ==========
// 同步 projectStore.subtitleOffset 到高级设置
watch(
  () => projectStore.subtitleOffset,
  (value) => {
    advancedConfig.value.general.global_time_offset = value
  }
)

async function handleSaveAdvancedSettings() {
  try {
    // V3.2.0+dev.20260209.04: 保存时规范化空值为 0
    let offset = advancedConfig.value.general.global_time_offset
    if (offset === '' || offset === null || offset === undefined) {
      offset = 0
    }
    offset = Number(offset)

    // 0. 快捷键占用校验：浏览器保留组合直接拦截
    const browserReservedConflicts = detectBrowserReservedShortcutConflicts(advancedConfig.value.shortcuts)
    if (browserReservedConflicts.length > 0) {
      const firstConflict = browserReservedConflicts[0]
      ElMessage.error(
        `快捷键被浏览器占用：${getEditorShortcutActionLabel(firstConflict.action)} 不能使用 ${getEditorShortcutComboLabel(firstConflict.combo)}`
      )
      return
    }

    // 0.1 快捷键冲突校验：同一组合不可绑定多个动作
    const normalizedShortcutConfig = normalizeEditorShortcutConfig(advancedConfig.value.shortcuts)
    const shortcutConflicts = detectEditorShortcutConflicts(normalizedShortcutConfig)
    if (shortcutConflicts.length > 0) {
      const firstConflict = shortcutConflicts[0]
      ElMessage.error(
        `快捷键冲突：${getEditorShortcutActionLabel(firstConflict.firstAction)} 与 ${getEditorShortcutActionLabel(firstConflict.secondAction)} 都使用了 ${getEditorShortcutComboLabel(firstConflict.combo)}`
      )
      return
    }

    // 1. 应用到前端 projectStore
    projectStore.setSubtitleOffset(offset)

    // 1.1 应用并持久化快捷键开关与映射（保存后立即生效）
    const isShortcutOn = advancedConfig.value.general.enable_shortcuts !== false
    shortcutEnabled.value = isShortcutOn
    shortcutConfig.value = normalizedShortcutConfig
    advancedConfig.value.shortcuts = { ...normalizedShortcutConfig }
    localStorage.setItem(SHORTCUT_ENABLED_PREF_KEY, String(isShortcutOn))
    localStorage.setItem(SHORTCUT_CONFIG_PREF_KEY, JSON.stringify(normalizedShortcutConfig))

    // 1.2 应用并持久化”字幕手动滚动后自动恢复跟随”偏好
    const followAutoResume = advancedConfig.value.general.subtitle_follow_auto_resume !== false
    subtitleFollowAutoResumeEnabled.value = followAutoResume
    localStorage.setItem(SUBTITLE_FOLLOW_AUTO_RESUME_PREF_KEY, String(followAutoResume))

    // 1.3 应用并持久化”隐藏波形刻度”偏好
    hideTimelineScale.value = advancedConfig.value.general.hide_timeline_scale === true
    localStorage.setItem(HIDE_TIMELINE_SCALE_PREF_KEY, String(hideTimelineScale.value))

    // 1.4 V3.2.4+dev.20260302.02: 持久化合并分隔符设置
    const separatorConfig = {
      type: advancedConfig.value.general.merge_separator || 'space',
      custom: advancedConfig.value.general.merge_separator_custom || ''
    }
    localStorage.setItem('editor-merge-separator', JSON.stringify(separatorConfig))

    // 2. 保存到后端
    if (projectStore.meta.jobId) {
      await transcriptionApi.setJobSubtitleTimeOffset(projectStore.meta.jobId, offset)
    }

    ElMessage.success('高级设置已保存')
    // V3.2.4+dev.20260303.04: 保存后不关闭窗口，允许用户继续修改
  } catch (error) {
    console.error('[EditorView] 保存高级设置失败:', error)
    ElMessage.error('保存高级设置失败: ' + (error.message || '未知错误'))
  }
}

function handleCancelAdvancedSettings() {
  showAdvancedSettings.value = false
}

function handleCloseAdvancedSettings() {
  showAdvancedSettings.value = false
}

function formatLastSaved(timestamp) {
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

// ========== 快捷键操作 ==========

function togglePlay() {
  if (!isMediaReady.value) {
    console.warn('[EditorView] 媒体未就绪，快捷键播放操作被拦截')
    return
  }
  playbackManager.togglePlay()
}

function stepBackward() {
  const frameTime = 1 / 30
  const newTime = Math.max(0, playbackStore.currentTime - frameTime)
  playbackManager.seekTo(newTime)
}

function stepForward() {
  const frameTime = 1 / 30
  const newTime = Math.min(projectStore.meta.duration, playbackStore.currentTime + frameTime)
  playbackManager.seekTo(newTime)
}

function seekBackward() {
  const newTime = Math.max(0, playbackStore.currentTime - 5)
  playbackManager.seekTo(newTime)
}

function seekForward() {
  const newTime = Math.min(projectStore.meta.duration, playbackStore.currentTime + 5)
  playbackManager.seekTo(newTime)
}

function seekToStart() {
  playbackManager.seekTo(0)
}

function seekToEnd() {
  playbackManager.seekTo(projectStore.meta.duration)
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
  save: saveProject,
  undo,
  redo,
}, {
  isShortcutEnabled: shortcutEnabled,
  shortcutConfig,
})

// ========== 生命周期 ==========

onMounted(() => {
  // 恢复侧边栏宽度偏好
  const savedWidth = localStorage.getItem('editor-sidebar-width')
  if (savedWidth) {
    sidebarWidth.value = parseInt(savedWidth)
  }
  loadEditorInteractionPreferences()

  loadProject()
})

onUnmounted(() => {
  if (sidebarResizeFrameId !== 0) {
    cancelAnimationFrame(sidebarResizeFrameId)
    sidebarResizeFrameId = 0
  }
  playbackManager.pause()
  if (!activeJobId.value && props.projectId) {
    // 纯 project 模式无后台任务续跑需求，卸载时应主动释放连接，避免悬挂订阅。
    cleanupSSE()
  } else {
    // 注意：job 模式不在这里关闭SSE连接，以支持页面切换时保持连接
    // SSE连接会在任务完成/失败时自动关闭，或由sseChannelManager统一管理
  }

  // 停止轮询（轮询仅是备用方案）
  isCancelPending.value = false
  stopCancelTimeoutPolling()
  stopProgressPolling()
  stopProxyPolling()
})

onBeforeRouteLeave(async (to, from) => {
  playbackManager.pause()
  if (isDirty.value) {
    try {
      await saveProject()
    } catch (error) {
      console.error('[EditorView] 离开前保存失败:', error)
      const answer = window.confirm('保存失败，确定要离开吗? 未保存的修改可能会丢失。')
      if (!answer) return false
    }
  }
})
</script>

<style scoped>
/* 注意：SCSS 变量和 mixins 已迁移到主题系统 */

/* 旧的 SCSS 文件位于 @/styles/legacy/ 供参考 */

/* 现在使用 CSS Variables（由主题系统注入） */

.editor-view {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--af-bg-base);
  color: var(--af-text-normal);
  overflow: hidden;
}

/* 加载状态 */
.loading-overlay {
  display: flex;
  flex: 1;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 16px;
  color: var(--af-text-muted);
}

.loading-overlay .loading-spinner {
  width: 48px;
  height: 48px;
  border: 3px solid var(--af-border-default);
  border-top-color: var(--af-accent-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

/* 错误状态 */
.error-overlay {
  display: flex;
  flex: 1;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 16px;
  color: var(--af-text-muted);
  text-align: center;
}

.error-overlay .error-icon {
  width: 64px;
  height: 64px;
  color: var(--af-accent-danger);
}

.error-overlay h3 {
  margin: 0;
  color: var(--af-text-primary);
  font-size: 20px;
  font-weight: 600;
}

.error-overlay p {
  margin: 0;
  font-size: 14px;
  max-width: 400px;
}

.error-overlay .retry-btn {
  padding: 10px 24px;
  background: var(--af-accent-primary);
  border: none;
  border-radius: var(--af-radius-md);
  color: var(--af-text-on-dark);
  font-size: 14px;
  transition: background var(--af-transition-fast);
  cursor: pointer;
}

.error-overlay .retry-btn:hover {
  background: var(--af-accent-primary-hover);
}

.error-overlay .back-link {
  color: var(--af-text-secondary);
  font-size: 13px;
  text-decoration: underline;
}

.error-overlay .back-link:hover {
  color: var(--af-text-primary);
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* 主工作区 Grid 布局 */
.workspace-grid {
  display: grid;
  flex: 1;
  height: 100%;
  grid-template-columns: 1fr 4px 350px;
  overflow: hidden;
}

/* 舞台列 (三明治结构: 视频 + 控制 + 波形) */
.stage-column {
  display: grid;
  grid-template-rows: 1fr 48px 180px; /* 波形区域调整为180px（header+波形+滚动条，刻度已嵌入波形） */
  background: var(--af-video-bg);
  min-width: 0;
  overflow: hidden;
}

.stage-column .video-wrapper {
  display: flex;
  justify-content: center;
  align-items: center;
  overflow: hidden;
}

.stage-column .controls-wrapper {
  background: var(--af-bg-primary);
  border-top: 1px solid var(--af-border-default);
}

.stage-column .waveform-wrapper {
  background: var(--af-bg-secondary);
  border-top: 1px solid var(--af-border-default);
  overflow: hidden;
}

/* 可拖拽分隔条 */
.resizer {
  position: relative;
  z-index: 10;
  width: 4px;
  background: var(--af-border-default);
  transition: background 0.2s;
  cursor: col-resize;
}

.resizer:hover,
.resizer.active {
  background: var(--af-accent-primary);
}

/* 侧边栏 */
.sidebar-column {
  display: flex;
  flex-direction: column;
  background: var(--af-bg-primary);
  border-left: 1px solid var(--af-border-default);
  min-width: 280px;
  max-width: 600px;
  overflow: hidden;
}

.sidebar-column.is-resizing :deep(.subtitle-row),
.sidebar-column.is-resizing :deep(.subtitle-row .action-btn),
.sidebar-column.is-resizing :deep(.subtitle-row .delete-btn) {
  transition: none;
}

/* 标签页导航 */
.tab-nav {
  display: flex;
  align-items: stretch;
  padding: 0 12px;
  background: var(--af-bg-secondary);
  border-bottom: 1px solid var(--af-border-default);
}

.tab-nav .tab-btn {
  position: relative;
  padding: 12px 16px;
  background: transparent;
  border: none;
  color: var(--af-text-secondary);
  font-size: 13px;
  cursor: pointer;
  transition: color var(--af-transition-fast);
}

.tab-nav .tab-btn::after {
  position: absolute;
  right: 0;
  bottom: 0;
  left: 0;
  height: 2px;
  background: transparent;
  transition: background var(--af-transition-fast);
  content: '';
}

.tab-nav .tab-btn:hover {
  color: var(--af-text-normal);
}

.tab-nav .tab-btn.active {
  color: var(--af-accent-primary);
}

.tab-nav .tab-btn.active::after {
  background: var(--af-accent-primary);
}

.tab-nav .tab-btn .badge {
  display: inline-flex;
  justify-content: center;
  align-items: center;
  height: 18px;
  padding: 0 5px;
  background: var(--af-accent-danger);
  border-radius: var(--af-radius-full);
  color: var(--af-text-on-dark);
  font-size: 11px;
  min-width: 18px;
  margin-left: 6px;
}

/* tab-nav 右侧图标按钮（视图切换） */
.tab-nav-icon-btn {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 28px;
  height: 28px;
  border: none;
  background: transparent;
  color: var(--af-text-secondary);
  cursor: pointer;
  border-radius: var(--af-radius-md);
  flex-shrink: 0;
  align-self: center;
  transition: all var(--af-transition-fast);
}

.tab-nav-icon-btn:hover {
  background: var(--af-bg-tertiary);
  color: var(--af-accent-primary);
}

.tab-nav-icon-btn svg {
  width: 18px;
  height: 18px;
}

/* 标签页内容 */
.tab-content {
  flex: 1;
  overflow: hidden;
}

.tab-pane {
  height: 100%;
  overflow: hidden;
}

/* 占位面板 */
.placeholder-panel {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 48px 24px;
  color: var(--af-text-muted);
  text-align: center;
}

.placeholder-panel svg {
  width: 48px;
  height: 48px;
  margin-bottom: 16px;
  opacity: 0.5;
}

.placeholder-panel h3 {
  margin-bottom: 8px;
  color: var(--af-text-normal);
  font-size: 16px;
  font-weight: 600;
}

.placeholder-panel p {
  margin-bottom: 4px;
  font-size: 13px;
}

.placeholder-panel .coming-soon {
  color: var(--af-accent-primary);
  font-style: italic;
}

.placeholder-panel .error-list {
  width: 100%;
  max-width: 400px;
  margin-top: 16px;
  text-align: left;
}

.placeholder-panel .error-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  margin-bottom: 4px;
  background: var(--af-bg-secondary);
  border-radius: var(--af-radius-sm);
  cursor: pointer;
  transition: background var(--af-transition-fast);
}

.placeholder-panel .error-item:hover {
  background: var(--af-bg-tertiary);
}

.placeholder-panel .error-item.error {
  border-left: 3px solid var(--af-accent-danger);
}

.placeholder-panel .error-item.warning {
  border-left: 3px solid var(--af-accent-warning);
}

.placeholder-panel .error-item .error-index {
  color: var(--af-text-muted);
  font-family: var(--af-font-mono);
  font-size: 12px;
}

.placeholder-panel .error-item .error-message {
  color: var(--af-text-normal);
  font-size: 13px;
}

/* 底部状态栏 */
.editor-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 32px;
  padding: 0 16px;
  background: var(--af-bg-secondary);
  color: var(--af-text-muted);
  font-size: 12px;
  flex-shrink: 0;
  border-top: 1px solid var(--af-border-default);
}

.editor-footer .footer-left,
.editor-footer .footer-center,
.editor-footer .footer-right {
  display: flex;
  align-items: center;
  gap: 8px;
}

.editor-footer .save-time {
  display: flex;
  align-items: center;
  gap: 4px;
  color: var(--af-text-secondary);
}

.editor-footer .save-time .icon {
  width: 14px;
  height: 14px;
  color: var(--af-accent-success);
}

.editor-footer .error-indicator {
  color: var(--af-accent-danger);
  cursor: pointer;
}

.editor-footer .error-indicator:hover {
  text-decoration: underline;
}

.editor-footer .settings-btn {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 24px;
  height: 24px;
  background: transparent;
  border: none;
  border-radius: 4px;
  color: var(--af-text-muted);
  transition: all 0.2s;
  cursor: pointer;
}

.editor-footer .settings-btn svg {
  width: 16px;
  height: 16px;
}

.editor-footer .settings-btn:hover {
  background: var(--af-bg-tertiary);
  color: var(--af-text-normal);
}

.editor-footer .divider {
  color: var(--af-border-default);
}
</style>
