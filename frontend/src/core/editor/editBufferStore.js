import { defineStore } from 'pinia'
import { ref } from 'vue'

const TEXT_COMMIT_DEBOUNCE_MS = 250

function normalizeId(rawId) {
  const normalized = String(rawId ?? '').trim()
  return normalized || null
}

function cloneTimeDraftPayload(payload = {}) {
  const draft = {}
  if (payload.start !== undefined) draft.start = Number(payload.start)
  if (payload.end !== undefined) draft.end = Number(payload.end)
  if (payload.mode !== undefined) draft.mode = payload.mode
  return draft
}

export const useEditBufferStore = defineStore('editBuffer', () => {
  const textSessions = ref(new Map())
  const timeDrafts = ref(new Map())
  const textCommitTimers = new Map()

  function ensureTextSession(subtitleId, initialText = '') {
    const normalizedId = normalizeId(subtitleId)
    if (!normalizedId) return null

    let session = textSessions.value.get(normalizedId)
    if (!session) {
      session = {
        originalText: String(initialText ?? ''),
        draftText: String(initialText ?? ''),
        isEditing: false,
        isComposing: false,
      }
      textSessions.value.set(normalizedId, session)
    }
    return session
  }

  function clearTextCommitTimer(subtitleId) {
    const normalizedId = normalizeId(subtitleId)
    if (!normalizedId) return

    const timer = textCommitTimers.get(normalizedId)
    if (!timer) return

    clearTimeout(timer)
    textCommitTimers.delete(normalizedId)
  }

  function beginTextEdit(subtitleId, initialText = '') {
    const session = ensureTextSession(subtitleId, initialText)
    if (!session) return

    session.originalText = String(initialText ?? '')
    session.draftText = String(initialText ?? '')
    session.isEditing = true
    session.isComposing = false
  }

  function endTextEdit(subtitleId) {
    const session = ensureTextSession(subtitleId)
    if (!session) return

    session.isEditing = false
    session.isComposing = false
    session.originalText = session.draftText
  }

  function cancelTextEdit(subtitleId, fallbackText = '') {
    const session = ensureTextSession(subtitleId, fallbackText)
    if (!session) return String(fallbackText ?? '')

    clearTextCommitTimer(subtitleId)
    session.draftText = session.originalText ?? String(fallbackText ?? '')
    session.isEditing = false
    session.isComposing = false
    return session.draftText
  }

  function updateTextDraft(subtitleId, nextText = '') {
    const session = ensureTextSession(subtitleId, nextText)
    if (!session) return

    session.draftText = String(nextText ?? '')
  }

  function syncCommittedText(subtitleId, committedText = '') {
    const session = ensureTextSession(subtitleId, committedText)
    if (!session) return

    if (session.isEditing) {
      return
    }

    const normalizedText = String(committedText ?? '')
    session.originalText = normalizedText
    session.draftText = normalizedText
  }

  function setTextComposing(subtitleId, isComposing) {
    const session = ensureTextSession(subtitleId)
    if (!session) return
    session.isComposing = Boolean(isComposing)
  }

  function scheduleTextCommit(
    subtitleId,
    {
      delay = TEXT_COMMIT_DEBOUNCE_MS,
      getCommittedText = () => '',
      onCommit = null,
    } = {}
  ) {
    const normalizedId = normalizeId(subtitleId)
    if (!normalizedId) return

    clearTextCommitTimer(normalizedId)
    const timer = setTimeout(() => {
      textCommitTimers.delete(normalizedId)
      flushTextCommit(normalizedId, {
        immediate: true,
        getCommittedText,
        onCommit,
      })
    }, delay)
    textCommitTimers.set(normalizedId, timer)
  }

  function flushTextCommit(
    subtitleId,
    {
      immediate = false,
      getCommittedText = () => '',
      onCommit = null,
    } = {}
  ) {
    const session = ensureTextSession(subtitleId)
    if (!session) return null

    clearTextCommitTimer(subtitleId)
    if (!immediate) {
      return session.draftText
    }

    const committedText = String(getCommittedText?.() ?? '')
    if (session.draftText !== committedText) {
      onCommit?.(session.draftText)
    }
    return session.draftText
  }

  function isTextEditing(subtitleId) {
    const normalizedId = normalizeId(subtitleId)
    if (!normalizedId) return false
    return Boolean(textSessions.value.get(normalizedId)?.isEditing)
  }

  function isTextComposing(subtitleId) {
    const normalizedId = normalizeId(subtitleId)
    if (!normalizedId) return false
    return Boolean(textSessions.value.get(normalizedId)?.isComposing)
  }

  function getTextDraft(subtitleId, fallbackText = '') {
    const normalizedId = normalizeId(subtitleId)
    if (!normalizedId) return String(fallbackText ?? '')
    return textSessions.value.get(normalizedId)?.draftText ?? String(fallbackText ?? '')
  }

  function setTimeDraft(subtitleId, payload = {}) {
    const normalizedId = normalizeId(subtitleId)
    if (!normalizedId) return null

    const nextDraft = {
      ...(timeDrafts.value.get(normalizedId) || {}),
      ...cloneTimeDraftPayload(payload),
    }
    timeDrafts.value.set(normalizedId, nextDraft)
    return nextDraft
  }

  function getTimeDraft(subtitleId) {
    const normalizedId = normalizeId(subtitleId)
    if (!normalizedId) return null
    return timeDrafts.value.get(normalizedId) || null
  }

  function getBufferedTime(subtitleId, field, fallbackValue) {
    const draft = getTimeDraft(subtitleId)
    const candidate = Number(draft?.[field])
    return Number.isFinite(candidate) ? candidate : fallbackValue
  }

  function syncCommittedTime(subtitleId, committed = {}) {
    const normalizedId = normalizeId(subtitleId)
    if (!normalizedId) return

    const currentDraft = timeDrafts.value.get(normalizedId)
    if (!currentDraft) {
      return
    }

    const committedStart = Number(committed.start)
    const committedEnd = Number(committed.end)
    const draftStart = Number(currentDraft.start)
    const draftEnd = Number(currentDraft.end)

    const hasCommittedMatch = (
      (!Number.isFinite(draftStart) || Math.abs(draftStart - committedStart) <= 0.0005)
      && (!Number.isFinite(draftEnd) || Math.abs(draftEnd - committedEnd) <= 0.0005)
    )

    if (hasCommittedMatch || currentDraft.mode !== 'drag') {
      timeDrafts.value.delete(normalizedId)
    }
  }

  function clearTimeDraft(subtitleId) {
    const normalizedId = normalizeId(subtitleId)
    if (!normalizedId) return
    timeDrafts.value.delete(normalizedId)
  }

  function clearAllDrafts(subtitleId) {
    const normalizedId = normalizeId(subtitleId)
    if (!normalizedId) return
    clearTextCommitTimer(normalizedId)
    textSessions.value.delete(normalizedId)
    timeDrafts.value.delete(normalizedId)
  }

  return {
    beginTextEdit,
    endTextEdit,
    cancelTextEdit,
    updateTextDraft,
    syncCommittedText,
    setTextComposing,
    scheduleTextCommit,
    flushTextCommit,
    clearTextCommitTimer,
    isTextEditing,
    isTextComposing,
    getTextDraft,
    setTimeDraft,
    getTimeDraft,
    getBufferedTime,
    syncCommittedTime,
    clearTimeDraft,
    clearAllDrafts,
  }
})
