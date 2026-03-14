// V3.2.5+dev.20260314.01: 编辑器会话元数据管理
import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useEditorSessionStore = defineStore('editorSession', () => {
  // ─── 会话标识 ───
  const projectId = ref(null)
  const jobId = ref(null)
  const sessionId = ref(null)
  const idSeed = ref('')

  // ─── ID 计数器 ───
  const idCounter = ref(0)
  const commandCounter = ref(0)

  // ─── 保存状态 ───
  const ackedRevision = ref(0)
  const isSaving = ref(false)
  const isDirty = ref(false)

  // ─── 会话状态 ───
  const isClosed = ref(false)
  const closedReason = ref(null)

  // ─── ID 生成器（唯一入口） ───
  function nextLocalId() {
    idCounter.value++
    return `sub_${idSeed.value}_${String(idCounter.value).padStart(6, '0')}`
  }

  function nextCommandId() {
    commandCounter.value++
    return `cmd_${idSeed.value}_${String(commandCounter.value).padStart(6, '0')}`
  }

  // ─── 会话生命周期 ───
  function openSession(pid, jid) {
    projectId.value = pid
    jobId.value = jid
    sessionId.value = crypto.randomUUID().slice(0, 8)

    const randomBytes = new Uint8Array(2)
    crypto.getRandomValues(randomBytes)
    idSeed.value = Array.from(randomBytes).map(b => b.toString(36)).join('').slice(0, 3)

    idCounter.value = 0
    commandCounter.value = 0
    ackedRevision.value = 0
    isSaving.value = false
    isDirty.value = false
    isClosed.value = false
    closedReason.value = null
  }

  function closeSession(reason = 'graceful_exit') {
    isClosed.value = true
    closedReason.value = reason
  }

  // ─── 保存状态管理 ───
  function markDirty() {
    isDirty.value = true
  }

  function markSaved(revision) {
    ackedRevision.value = revision
    isDirty.value = false
  }

  async function requestSave() {
    if (isSaving.value) return
    isSaving.value = true
    try {
      // 保存编排逻辑由外部实现
      return { success: true }
    } finally {
      isSaving.value = false
    }
  }

  // ─── 时间转换工具 ───
  function secondsToMs(seconds) {
    return Math.round(seconds * 1000)
  }

  return {
    projectId,
    jobId,
    sessionId,
    idSeed,
    idCounter,
    commandCounter,
    ackedRevision,
    isSaving,
    isDirty,
    isClosed,
    closedReason,
    nextLocalId,
    nextCommandId,
    openSession,
    closeSession,
    markDirty,
    markSaved,
    requestSave,
    secondsToMs,
  }
})
