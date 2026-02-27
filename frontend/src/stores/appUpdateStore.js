import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import systemApi from '@/services/api/systemApi'

const IGNORED_VERSION_KEY = 'anchorflux_ignored_update_version'
const DELAYED_UPDATE_KEY = 'anchorflux_delayed_update'

function safeGetStorageItem(key) {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function safeSetStorageItem(key, value) {
  try {
    localStorage.setItem(key, value)
  } catch (error) {
    console.warn('[AppUpdateStore] 写入 localStorage 失败:', error)
  }
}

function safeRemoveStorageItem(key) {
  try {
    localStorage.removeItem(key)
  } catch (error) {
    console.warn('[AppUpdateStore] 删除 localStorage 失败:', error)
  }
}

function parseDelayedUpdate(rawValue) {
  if (!rawValue) return null
  try {
    return JSON.parse(rawValue)
  } catch {
    return null
  }
}

export const useAppUpdateStore = defineStore('appUpdate', () => {
  const updateInfo = ref(null)
  const isChecking = ref(false)
  const checkError = ref(null)
  const isDelayedUpdatePending = ref(false)
  const isInitialized = ref(false)

  function getIgnoredVersion() {
    return safeGetStorageItem(IGNORED_VERSION_KEY) || null
  }

  function setIgnoredVersion(version) {
    if (version) {
      safeSetStorageItem(IGNORED_VERSION_KEY, String(version))
      return
    }
    safeRemoveStorageItem(IGNORED_VERSION_KEY)
  }

  function getDelayedUpdate() {
    return parseDelayedUpdate(safeGetStorageItem(DELAYED_UPDATE_KEY))
  }

  function setDelayedUpdate(info) {
    if (info) {
      safeSetStorageItem(DELAYED_UPDATE_KEY, JSON.stringify(info))
      isDelayedUpdatePending.value = true
      return
    }
    safeRemoveStorageItem(DELAYED_UPDATE_KEY)
    isDelayedUpdatePending.value = false
  }

  async function checkForUpdate(ignoreIgnoredVersion = false) {
    if (isChecking.value) {
      return { hasUpdate: false, updateInfo: null }
    }

    isChecking.value = true
    checkError.value = null

    try {
      const response = await systemApi.checkUpdate()
      if (!response?.has_update) {
        updateInfo.value = null
        return { hasUpdate: false, updateInfo: null }
      }

      const ignoredVersion = getIgnoredVersion()
      if (!ignoreIgnoredVersion && ignoredVersion === response.latest_version) {
        updateInfo.value = null
        return { hasUpdate: false, updateInfo: null }
      }

      updateInfo.value = {
        currentVersion: response.current_version,
        latestVersion: response.latest_version,
        changelog: response.changelog || '',
        downloadUrl: response.download_url,
        forceUpdate: response.force_update || false,
      }
      return { hasUpdate: true, updateInfo: updateInfo.value }
    } catch (error) {
      console.error('[AppUpdateStore] 检查更新失败:', error)
      checkError.value = error?.message || '检查更新失败'
      return { hasUpdate: false, updateInfo: null, error: checkError.value }
    } finally {
      isChecking.value = false
    }
  }

  function ignoreCurrentUpdate() {
    if (!updateInfo.value?.latestVersion) return
    setIgnoredVersion(updateInfo.value.latestVersion)
    updateInfo.value = null
  }

  async function triggerImmediateUpdate() {
    if (!updateInfo.value) {
      return { success: false, message: '没有可用的更新' }
    }

    try {
      const response = await systemApi.triggerUpdate({
        download_url: updateInfo.value.downloadUrl,
        version: updateInfo.value.latestVersion,
        changelog: updateInfo.value.changelog,
        delay_mode: false,
      })

      if (response?.success) {
        setIgnoredVersion(null)
        setDelayedUpdate(null)
      }
      return response
    } catch (error) {
      console.error('[AppUpdateStore] 触发立即更新失败:', error)
      return { success: false, message: error?.message || '触发更新失败' }
    }
  }

  async function scheduleDelayedUpdate() {
    if (!updateInfo.value) {
      return { success: false, message: '没有可用的更新' }
    }

    try {
      const response = await systemApi.triggerUpdate({
        download_url: updateInfo.value.downloadUrl,
        version: updateInfo.value.latestVersion,
        changelog: updateInfo.value.changelog,
        delay_mode: true,
      })

      if (response?.success) {
        setDelayedUpdate({
          version: updateInfo.value.latestVersion,
          scheduledAt: new Date().toISOString(),
        })
        setIgnoredVersion(null)
      }
      return response
    } catch (error) {
      console.error('[AppUpdateStore] 安排延迟更新失败:', error)
      return { success: false, message: error?.message || '安排更新失败' }
    }
  }

  function clearDelayedUpdate() {
    setDelayedUpdate(null)
  }

  function initialize() {
    if (isInitialized.value) return
    isDelayedUpdatePending.value = getDelayedUpdate() !== null
    isInitialized.value = true
  }

  const hasDelayedUpdate = computed(() => {
    return isDelayedUpdatePending.value || getDelayedUpdate() !== null
  })

  return {
    updateInfo,
    isChecking,
    checkError,
    hasDelayedUpdate,
    initialize,
    checkForUpdate,
    ignoreCurrentUpdate,
    triggerImmediateUpdate,
    scheduleDelayedUpdate,
    clearDelayedUpdate,
    getIgnoredVersion,
    getDelayedUpdate,
  }
})
