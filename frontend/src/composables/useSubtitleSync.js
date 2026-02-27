import { computed, onUnmounted, unref, watch } from 'vue'
import { useProjectStore } from '@/stores/projectStore'
import { useSubtitleDocumentStore } from '@/stores/subtitleDocumentStore'

/**
 * 字幕同步兼容层
 *
 * Phase 4：同步队列与副作用迁移到 subtitleDocumentStore。
 * composable 仅保留旧 API 形态，降低调用方改动风险。
 */
export function useSubtitleSync(identityRef) {
  const projectStore = useProjectStore()
  const subtitleDocumentStore = useSubtitleDocumentStore()

  function getActiveIdentity() {
    return unref(identityRef) || projectStore.primaryId
  }

  const stopWatchIdentity = watch(
    () => getActiveIdentity(),
    async (identityId) => {
      await subtitleDocumentStore.bindSyncIdentity(identityId)
    },
    { immediate: true }
  )

  onUnmounted(() => {
    stopWatchIdentity()
  })

  return {
    onSubtitleEdit: subtitleDocumentStore.onSubtitleEdit,
    forceSyncNow: subtitleDocumentStore.forceSyncNow,
    applyPendingEditsToStore: subtitleDocumentStore.applyPendingEditsToStore,
    isSyncing: computed(() => subtitleDocumentStore.isSyncing),
    syncErrors: computed(() => subtitleDocumentStore.syncErrors),
    pendingCount: subtitleDocumentStore.pendingCount,
  }
}
