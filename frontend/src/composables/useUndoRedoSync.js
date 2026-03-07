import { storeToRefs } from 'pinia'
import { useSyncCoordinatorStore } from '@/core/sync/syncCoordinator'

export function useUndoRedoSync() {
  const syncCoordinator = useSyncCoordinatorStore()
  const { isUndoRedoSyncing, undoRedoLastError } = storeToRefs(syncCoordinator)

  return {
    undoWithSync: syncCoordinator.undoWithSync,
    redoWithSync: syncCoordinator.redoWithSync,
    flushSync: syncCoordinator.flushUndoRedoSync,
    isSyncing: isUndoRedoSyncing,
    lastSyncError: undoRedoLastError,
  }
}