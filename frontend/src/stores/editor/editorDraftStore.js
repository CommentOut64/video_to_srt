// V3.2.5+dev.20260314.01: 输入草稿缓冲 - 行级隔离
import { defineStore } from 'pinia'
import { shallowRef } from 'vue'
import { useEditorCommandBus } from './editorCommandBus'
import { useEditorSessionStore } from './editorSessionStore'

export const useEditorDraftStore = defineStore('editorDraft', () => {
  const activeTextDraft = shallowRef(null)
  const activeTimingDraft = shallowRef(null)
  let idleTimer = null
  const IDLE_COMMIT_MS = 300

  function startTextEdit(localId, currentText) {
    if (activeTextDraft.value) commitTextDraft()
    activeTextDraft.value = {
      localId,
      text: currentText,
      originalText: currentText,
      isComposing: false,
      selectionStart: 0,
      selectionEnd: 0,
      startedAt: Date.now(),
    }
  }

  function updateTextDraft(text, selectionStart = 0, selectionEnd = 0) {
    if (!activeTextDraft.value) return
    activeTextDraft.value = {
      ...activeTextDraft.value,
      text,
      selectionStart,
      selectionEnd,
    }
    if (!activeTextDraft.value.isComposing) {
      scheduleIdleCommit()
    }
  }

  function setComposing(isComposing) {
    if (!activeTextDraft.value) return
    activeTextDraft.value = { ...activeTextDraft.value, isComposing }
    if (isComposing) {
      cancelIdleCommit()
    } else {
      scheduleIdleCommit()
    }
  }

  function commitTextDraft() {
    if (!activeTextDraft.value) return
    cancelIdleCommit()

    const commandBus = useEditorCommandBus()
    const sessionStore = useEditorSessionStore()

    const command = {
      type: 'update_text',
      commandId: sessionStore.nextCommandId(),
      source: 'user',
      createdAt: Date.now(),
      localId: activeTextDraft.value.localId,
      before: { text: activeTextDraft.value.originalText },
      after: { text: activeTextDraft.value.text },
      mergeKey: `text_${activeTextDraft.value.localId}_${activeTextDraft.value.startedAt}`,
    }

    commandBus.dispatch(command)
    activeTextDraft.value.originalText = activeTextDraft.value.text
  }

  function cancelTextDraft() {
    cancelIdleCommit()
    activeTextDraft.value = null
  }

  function startTimingDrag(localId, startMs, endMs) {
    if (activeTimingDraft.value) commitTimingDraft()
    activeTimingDraft.value = {
      localId,
      startMs,
      endMs,
      isDragging: true,
      dragOrigin: { startMs, endMs },
    }
  }

  function updateTimingDraft(startMs, endMs) {
    if (!activeTimingDraft.value) return
    activeTimingDraft.value = { ...activeTimingDraft.value, startMs, endMs }
  }

  function commitTimingDraft() {
    if (!activeTimingDraft.value) return

    const { localId, startMs, endMs, dragOrigin } = activeTimingDraft.value
    if (startMs === dragOrigin.startMs && endMs === dragOrigin.endMs) {
      activeTimingDraft.value = null
      return
    }

    const commandBus = useEditorCommandBus()
    const sessionStore = useEditorSessionStore()

    const command = {
      type: 'update_timing',
      commandId: sessionStore.nextCommandId(),
      source: 'user',
      createdAt: Date.now(),
      localId,
      before: { startMs: dragOrigin.startMs, endMs: dragOrigin.endMs },
      after: { startMs, endMs },
    }

    commandBus.dispatch(command)
    activeTimingDraft.value = null
  }

  function cancelTimingDraft() {
    activeTimingDraft.value = null
  }

  function scheduleIdleCommit() {
    cancelIdleCommit()
    idleTimer = setTimeout(() => {
      commitTextDraft()
    }, IDLE_COMMIT_MS)
  }

  function cancelIdleCommit() {
    if (idleTimer) {
      clearTimeout(idleTimer)
      idleTimer = null
    }
  }

  function flushAll() {
    if (activeTextDraft.value) commitTextDraft()
    if (activeTimingDraft.value) commitTimingDraft()
    activeTextDraft.value = null
    activeTimingDraft.value = null
  }

  return {
    activeTextDraft,
    activeTimingDraft,
    startTextEdit,
    updateTextDraft,
    setComposing,
    commitTextDraft,
    cancelTextDraft,
    startTimingDrag,
    updateTimingDraft,
    commitTimingDraft,
    cancelTimingDraft,
    flushAll,
  }
})
