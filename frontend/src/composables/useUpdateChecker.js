/**
 * 在线更新检查 Composable
 * V3.1.1+dev.20260105.01
 *
 * 功能：
 * - 检查是否有新版本可用
 * - 管理忽略版本（localStorage 持久化）
 * - 触发立即更新或延迟更新
 * - 启动时自动检查更新
 */

import { ref, computed, readonly } from 'vue'
import systemApi from '@/services/api/systemApi'

// 忽略版本的 localStorage key
const IGNORED_VERSION_KEY = 'anchorflux_ignored_update_version'
// 延迟更新标记的 localStorage key
const DELAYED_UPDATE_KEY = 'anchorflux_delayed_update'

// 更新信息状态（全局单例）
const updateInfo = ref(null)
const isChecking = ref(false)
const checkError = ref(null)
const isDelayedUpdatePending = ref(false)

/**
 * 使用更新检查器
 */
export function useUpdateChecker() {

  /**
   * 获取当前忽略的版本号
   */
  const getIgnoredVersion = () => {
    try {
      return localStorage.getItem(IGNORED_VERSION_KEY) || null
    } catch {
      return null
    }
  }

  /**
   * 设置忽略的版本号
   */
  const setIgnoredVersion = (version) => {
    try {
      if (version) {
        localStorage.setItem(IGNORED_VERSION_KEY, version)
      } else {
        localStorage.removeItem(IGNORED_VERSION_KEY)
      }
    } catch (e) {
      console.warn('[UpdateChecker] Failed to save ignored version:', e)
    }
  }

  /**
   * 获取延迟更新信息
   */
  const getDelayedUpdate = () => {
    try {
      const data = localStorage.getItem(DELAYED_UPDATE_KEY)
      return data ? JSON.parse(data) : null
    } catch {
      return null
    }
  }

  /**
   * 设置延迟更新信息
   */
  const setDelayedUpdate = (info) => {
    try {
      if (info) {
        localStorage.setItem(DELAYED_UPDATE_KEY, JSON.stringify(info))
        isDelayedUpdatePending.value = true
      } else {
        localStorage.removeItem(DELAYED_UPDATE_KEY)
        isDelayedUpdatePending.value = false
      }
    } catch (e) {
      console.warn('[UpdateChecker] Failed to save delayed update:', e)
    }
  }

  /**
   * 检查是否有新版本可用
   * @param {boolean} ignoreIgnoredVersion - 是否忽略已标记忽略的版本（手动检查时为 true）
   * @returns {Promise<{hasUpdate: boolean, updateInfo: Object|null}>}
   */
  const checkForUpdate = async (ignoreIgnoredVersion = false) => {
    if (isChecking.value) {
      return { hasUpdate: false, updateInfo: null }
    }

    isChecking.value = true
    checkError.value = null

    try {
      const response = await systemApi.checkUpdate()

      if (!response.has_update) {
        updateInfo.value = null
        return { hasUpdate: false, updateInfo: null }
      }

      // 检查是否是被忽略的版本
      const ignoredVersion = getIgnoredVersion()
      if (!ignoreIgnoredVersion && ignoredVersion === response.latest_version) {
        console.log('[UpdateChecker] Version ignored:', response.latest_version)
        updateInfo.value = null
        return { hasUpdate: false, updateInfo: null }
      }

      // 有新版本
      updateInfo.value = {
        currentVersion: response.current_version,
        latestVersion: response.latest_version,
        changelog: response.changelog || '',
        downloadUrl: response.download_url,
        forceUpdate: response.force_update || false
      }

      console.log('[UpdateChecker] New version available:', response.latest_version)
      return { hasUpdate: true, updateInfo: updateInfo.value }

    } catch (e) {
      console.error('[UpdateChecker] Check failed:', e)
      checkError.value = e.message || '检查更新失败'
      return { hasUpdate: false, updateInfo: null, error: checkError.value }
    } finally {
      isChecking.value = false
    }
  }

  /**
   * 忽略当前版本
   */
  const ignoreCurrentUpdate = () => {
    if (updateInfo.value?.latestVersion) {
      setIgnoredVersion(updateInfo.value.latestVersion)
      updateInfo.value = null
      console.log('[UpdateChecker] Version ignored')
    }
  }

  /**
   * 触发立即更新
   * @returns {Promise<{success: boolean, message: string}>}
   */
  const triggerImmediateUpdate = async () => {
    if (!updateInfo.value) {
      return { success: false, message: '没有可用的更新' }
    }

    try {
      const response = await systemApi.triggerUpdate({
        download_url: updateInfo.value.downloadUrl,
        version: updateInfo.value.latestVersion,
        changelog: updateInfo.value.changelog,
        delay_mode: false
      })

      if (response.success) {
        // 清除忽略版本和延迟更新标记
        setIgnoredVersion(null)
        setDelayedUpdate(null)
      }

      return response
    } catch (e) {
      console.error('[UpdateChecker] Trigger update failed:', e)
      return { success: false, message: e.message || '触发更新失败' }
    }
  }

  /**
   * 安排重启时更新（延迟模式）
   * @returns {Promise<{success: boolean, message: string}>}
   */
  const scheduleDelayedUpdate = async () => {
    if (!updateInfo.value) {
      return { success: false, message: '没有可用的更新' }
    }

    try {
      const response = await systemApi.triggerUpdate({
        download_url: updateInfo.value.downloadUrl,
        version: updateInfo.value.latestVersion,
        changelog: updateInfo.value.changelog,
        delay_mode: true
      })

      if (response.success) {
        // 保存延迟更新信息到 localStorage
        setDelayedUpdate({
          version: updateInfo.value.latestVersion,
          scheduledAt: new Date().toISOString()
        })
        // 清除忽略版本
        setIgnoredVersion(null)
      }

      return response
    } catch (e) {
      console.error('[UpdateChecker] Schedule delayed update failed:', e)
      return { success: false, message: e.message || '安排更新失败' }
    }
  }

  /**
   * 清除延迟更新标记
   */
  const clearDelayedUpdate = () => {
    setDelayedUpdate(null)
  }

  /**
   * 检查是否有待执行的延迟更新
   */
  const hasDelayedUpdate = computed(() => {
    return isDelayedUpdatePending.value || getDelayedUpdate() !== null
  })

  /**
   * 初始化：检查是否有延迟更新标记
   */
  const initialize = () => {
    const delayed = getDelayedUpdate()
    isDelayedUpdatePending.value = delayed !== null
    if (delayed) {
      console.log('[UpdateChecker] Delayed update pending for version:', delayed.version)
    }
  }

  // 初始化
  initialize()

  return {
    // 状态（只读）
    updateInfo: readonly(updateInfo),
    isChecking: readonly(isChecking),
    checkError: readonly(checkError),
    hasDelayedUpdate,

    // 方法
    checkForUpdate,
    ignoreCurrentUpdate,
    triggerImmediateUpdate,
    scheduleDelayedUpdate,
    clearDelayedUpdate,
    getIgnoredVersion,
    getDelayedUpdate
  }
}
