import { computed, onMounted, onUnmounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { projectTaskApi } from '@/services/api'
import { useVideoPlaybackDiagnostics } from '@/composables/useVideoPlaybackDiagnostics'

function getShellBridge() {
  if (typeof window === 'undefined') {
    return null
  }
  return window.anchorfluxShell || null
}

function unwrapResponse(response) {
  return response?.data || response || null
}

function safeJsonStringify(payload) {
  try {
    return JSON.stringify(payload, null, 2)
  } catch (error) {
    return JSON.stringify({ error: error?.message || '序列化失败' }, null, 2)
  }
}

async function writeClipboardText(text) {
  if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text)
    return true
  }
  return false
}

export function useShellRuntimeDiagnostics() {
  const shellRuntimeInfo = ref(null)
  const queueRuntimeInfo = ref(null)
  const { playbackSummary } = useVideoPlaybackDiagnostics()
  const isLoading = ref(false)
  const errorMessage = ref('')
  const lastUpdatedAt = ref('')
  const isCopying = ref(false)

  let unsubscribeRuntimeInfo = null

  const shellBridgeAvailable = computed(() => Boolean(getShellBridge()))
  const shellPolicy = computed(() => shellRuntimeInfo.value?.policy || null)
  const gpuFeatureStatus = computed(() => shellRuntimeInfo.value?.gpuFeatureStatus || {})
  const queueDiagnosticsEnabled = computed(() => queueRuntimeInfo.value?.runtime_enabled === true)
  const hasAnyDiagnostics = computed(() => Boolean(shellRuntimeInfo.value || queueRuntimeInfo.value || playbackSummary.value))

  async function refreshDiagnostics(options = {}) {
    const isSilent = options.silent === true
    if (!isSilent) {
      isLoading.value = true
    }
    errorMessage.value = ''

    try {
      const shellBridge = getShellBridge()
      const [shellInfo, queueInfo] = await Promise.all([
        shellBridge?.getRuntimeInfo ? shellBridge.getRuntimeInfo().catch(() => null) : Promise.resolve(null),
        projectTaskApi.getRuntimeDiagnostics().catch(() => null),
      ])
      shellRuntimeInfo.value = shellInfo || shellRuntimeInfo.value || null
      queueRuntimeInfo.value = unwrapResponse(queueInfo)
      lastUpdatedAt.value = new Date().toLocaleString()
    } catch (error) {
      errorMessage.value = error?.message || '获取运行时诊断失败'
    } finally {
      if (!isSilent) {
        isLoading.value = false
      }
    }
  }

  async function copySnapshot() {
    const snapshot = {
      shell_runtime: shellRuntimeInfo.value,
      queue_runtime: queueRuntimeInfo.value,
      playback_runtime: playbackSummary.value,
      copied_at: new Date().toISOString(),
    }
    isCopying.value = true
    try {
      const copied = await writeClipboardText(safeJsonStringify(snapshot))
      if (!copied) {
        throw new Error('当前环境不支持剪贴板写入')
      }
      ElMessage.success('诊断快照已复制')
    } catch (error) {
      ElMessage.error(`复制诊断快照失败：${error?.message || '未知错误'}`)
    } finally {
      isCopying.value = false
    }
  }

  onMounted(() => {
    const shellBridge = getShellBridge()
    if (shellBridge?.onRuntimeInfo) {
      unsubscribeRuntimeInfo = shellBridge.onRuntimeInfo((payload) => {
        shellRuntimeInfo.value = payload || null
        lastUpdatedAt.value = new Date().toLocaleString()
      })
    }
    void refreshDiagnostics()
  })

  onUnmounted(() => {
    if (typeof unsubscribeRuntimeInfo === 'function') {
      unsubscribeRuntimeInfo()
      unsubscribeRuntimeInfo = null
    }
  })

  return {
    shellRuntimeInfo,
    queueRuntimeInfo,
    playbackSummary,
    shellPolicy,
    gpuFeatureStatus,
    shellBridgeAvailable,
    queueDiagnosticsEnabled,
    hasAnyDiagnostics,
    isLoading,
    isCopying,
    errorMessage,
    lastUpdatedAt,
    refreshDiagnostics,
    copySnapshot,
    safeJsonStringify,
  }
}
