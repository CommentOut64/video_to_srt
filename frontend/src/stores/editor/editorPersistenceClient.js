// V3.2.4+dev.20260314.03: 编辑器持久化客户端 - 主线程与 Worker 通信
import { ref } from 'vue'
import { useEditorSessionStore } from './editorSessionStore'
import { useEditorDocumentStore } from './editorDocumentStore'
import { useEditorDraftStore } from './editorDraftStore'
import { useEditorCommandBus } from './editorCommandBus'

const EMERGENCY_BUFFER_SIZE = 50
const SNAPSHOT_COMMAND_THRESHOLD = 100
const SNAPSHOT_IDLE_MS = 30000

let worker = null
let emergencyBuffer = []
let commandsSinceSnapshot = 0
let idleHandle = null

export function useEditorPersistenceClient() {
  const isReady = ref(false)

  function init(projectId, sessionId) {
    worker = new Worker(
      new URL('../../workers/editorPersistence.worker.js', import.meta.url),
      { type: 'module' }
    )

    worker.onmessage = (event) => {
      if (event.data.type === 'ready') {
        isReady.value = true
      } else if (event.data.type === 'error') {
        console.error('[PersistenceClient] Worker error:', event.data)
      }
    }

    worker.postMessage({ type: 'init', projectId, sessionId })
    setupEmergencyHandlers(projectId)
  }

  function appendCommands(commands) {
    if (!worker || !isReady.value) return

    for (const cmd of commands) {
      appendToEmergencyBuffer(cmd)
    }

    worker.postMessage({ type: 'append_commands', commands })
    onCommandAppended(commands.length)
  }

  function appendToEmergencyBuffer(command) {
    emergencyBuffer.push(command)
    if (emergencyBuffer.length > EMERGENCY_BUFFER_SIZE) {
      emergencyBuffer.shift()
    }
  }

  function saveSnapshot(snapshot) {
    if (!worker || !isReady.value) return
    worker.postMessage({ type: 'save_snapshot', snapshot })
  }

  function load(projectId) {
    return new Promise((resolve) => {
      const handler = (event) => {
        if (event.data.type === 'loaded') {
          worker.removeEventListener('message', handler)
          resolve(event.data)
        }
      }
      worker.addEventListener('message', handler)
      worker.postMessage({ type: 'load', projectId })
    })
  }

  function onCommandAppended(count = 1) {
    commandsSinceSnapshot += count
    if (commandsSinceSnapshot >= SNAPSHOT_COMMAND_THRESHOLD) {
      triggerSnapshot()
    } else {
      scheduleIdleSnapshot()
    }
  }

  function scheduleIdleSnapshot() {
    if (typeof requestIdleCallback !== 'undefined') {
      if (idleHandle) cancelIdleCallback(idleHandle)
      idleHandle = requestIdleCallback(() => {
        if (commandsSinceSnapshot > 0) triggerSnapshot()
      }, { timeout: SNAPSHOT_IDLE_MS })
    }
  }

  function triggerSnapshot() {
    const docStore = useEditorDocumentStore()
    const snapshot = docStore.takeSnapshot()
    saveSnapshot(snapshot)
    commandsSinceSnapshot = 0
  }

  function setupEmergencyHandlers(projectId) {
    const handler = () => {
      const sessionStore = useEditorSessionStore()
      const draftStore = useEditorDraftStore()
      const docStore = useEditorDocumentStore()

      const data = {
        projectId,
        sessionId: sessionStore.sessionId,
        commands: emergencyBuffer,
        activeDraft: draftStore.activeTextDraft
          ? { ...draftStore.activeTextDraft }
          : null,
        revision: docStore.revision,
        savedAt: Date.now()
      }

      try {
        localStorage.setItem(
          `editor-emergency:${projectId}`,
          JSON.stringify(data)
        )
      } catch (e) {
        cleanOldestEmergencyBackup()
        try {
          localStorage.setItem(
            `editor-emergency:${projectId}`,
            JSON.stringify(data)
          )
        } catch (_) {
          // 放弃
        }
      }
    }

    window.addEventListener('beforeunload', handler)
    window.addEventListener('pagehide', handler)
    window.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') handler()
    })
  }

  function cleanOldestEmergencyBackup() {
    const keys = []
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i)
      if (key?.startsWith('editor-emergency:')) {
        keys.push(key)
      }
    }
    if (keys.length > 0) {
      localStorage.removeItem(keys[0])
    }
  }

  async function restoreProject(projectId) {
    const sessionStore = useEditorSessionStore()
    const docStore = useEditorDocumentStore()
    const commandBus = useEditorCommandBus()

    // 1. 检查 localStorage 紧急缓冲
    let emergencyData = null
    try {
      const raw = localStorage.getItem(`editor-emergency:${projectId}`)
      if (raw) {
        emergencyData = JSON.parse(raw)
        localStorage.removeItem(`editor-emergency:${projectId}`)
      }
    } catch (e) {
      console.error('[PersistenceClient] Failed to load emergency buffer:', e)
    }

    // 2. 通过 Worker 加载 IndexedDB
    const { snapshot, pendingCommands } = await load(projectId)

    // 3. 合并紧急缓冲和 Worker 日志
    const allCommands = [...pendingCommands]
    if (emergencyData?.commands) {
      for (const cmd of emergencyData.commands) {
        if (!allCommands.find(c => c.commandId === cmd.commandId)) {
          allCommands.push(cmd)
        }
      }
    }
    allCommands.sort((a, b) => a.createdAt - b.createdAt)

    // 4. 恢复文档
    if (snapshot) {
      docStore.restoreFromSnapshot(snapshot)
    }

    // 5. 回放命令日志
    for (const cmd of allCommands) {
      if (snapshot && cmd.createdAt <= snapshot.createdAt) continue
      commandBus.dispatch({ ...cmd, source: 'rehydrate' })
    }

    // 6. 恢复草稿
    if (emergencyData?.activeDraft) {
      const draftStore = useEditorDraftStore()
      draftStore.activeTextDraft = emergencyData.activeDraft
    }

    return { commandsReplayed: allCommands.length }
  }

  function destroy() {
    if (idleHandle) cancelIdleCallback(idleHandle)
    worker?.terminate()
    worker = null
    isReady.value = false
  }

  return {
    isReady,
    init,
    appendCommands,
    saveSnapshot,
    load,
    restoreProject,
    destroy
  }
}
