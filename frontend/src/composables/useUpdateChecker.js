import { computed, readonly } from 'vue'
import { useAppUpdateStore } from '@/stores/appUpdateStore'

/**
 * 在线更新检查兼容层
 *
 * Phase 4：状态与副作用已迁移到 appUpdateStore。
 * 该 composable 仅作为旧调用方兼容入口，Phase 5 可移除。
 */
export function useUpdateChecker() {
  const updateStore = useAppUpdateStore()
  updateStore.initialize()

  const updateInfo = computed(() => updateStore.updateInfo)
  const isChecking = computed(() => updateStore.isChecking)
  const checkError = computed(() => updateStore.checkError)
  const hasDelayedUpdate = computed(() => updateStore.hasDelayedUpdate)

  return {
    updateInfo: readonly(updateInfo),
    isChecking: readonly(isChecking),
    checkError: readonly(checkError),
    hasDelayedUpdate,
    checkForUpdate: updateStore.checkForUpdate,
    ignoreCurrentUpdate: updateStore.ignoreCurrentUpdate,
    triggerImmediateUpdate: updateStore.triggerImmediateUpdate,
    scheduleDelayedUpdate: updateStore.scheduleDelayedUpdate,
    clearDelayedUpdate: updateStore.clearDelayedUpdate,
    getIgnoredVersion: updateStore.getIgnoredVersion,
    getDelayedUpdate: updateStore.getDelayedUpdate,
  }
}
