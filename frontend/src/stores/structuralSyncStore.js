import { defineStore, storeToRefs } from 'pinia'
import { useSyncCoordinatorStore } from '@/core/sync/syncCoordinator'

export const useStructuralSyncStore = defineStore('structuralSync', () => {
  const syncCoordinator = useSyncCoordinatorStore()
  const {
    structuralSyncErrors,
    inflightStructuralCount,
    structuralErrorCount,
    hasStructuralErrors,
  } = storeToRefs(syncCoordinator)

  function trackOperation(type, promise) {
    return syncCoordinator.trackStructuralOperation(type, promise)
  }

  async function waitAll() {
    await syncCoordinator.waitAllStructuralOperations()
  }

  function clearAllErrors() {
    syncCoordinator.clearAllStructuralErrors()
  }

  function clearError(opId) {
    syncCoordinator.clearStructuralError(opId)
  }

  function $reset() {
    syncCoordinator.resetStructuralState()
  }

  return {
    syncErrors: structuralSyncErrors,
    inflightCount: inflightStructuralCount,
    errorCount: structuralErrorCount,
    hasErrors: hasStructuralErrors,
    trackOperation,
    waitAll,
    clearAllErrors,
    clearError,
    $reset,
  }
})