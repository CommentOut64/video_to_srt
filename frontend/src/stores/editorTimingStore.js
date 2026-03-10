import { defineStore } from 'pinia'
import { ref } from 'vue'

function clampSubtitleOffset(value) {
  const num = Number(value) || 0
  return Math.min(10, Math.max(-10, num))
}

function shiftWords(words, delta) {
  if (!Array.isArray(words) || words.length === 0) return []
  return words.map((word) => {
    const start = word?.start
    const end = word?.end
    const nextStart = typeof start === 'number' ? Math.max(0, start + delta) : start
    const nextEnd = typeof end === 'number' ? Math.max(nextStart ?? 0, end + delta) : end
    return {
      ...word,
      start: nextStart,
      end: nextEnd,
    }
  })
}

let applyOffsetDeltaHandler = () => {}
let writeMetaSubtitleOffsetHandler = () => {}

export const useEditorTimingStore = defineStore('editorTiming', () => {
  const subtitleOffset = ref(0)

  function bindRuntime({ applyOffsetDelta, writeMetaSubtitleOffset } = {}) {
    applyOffsetDeltaHandler = typeof applyOffsetDelta === 'function' ? applyOffsetDelta : () => {}
    writeMetaSubtitleOffsetHandler = typeof writeMetaSubtitleOffset === 'function' ? writeMetaSubtitleOffset : () => {}
  }

  function toDisplayTime(baseTime) {
    return (Number(baseTime) || 0) + subtitleOffset.value
  }

  function toBaseTime(displayTime) {
    return (Number(displayTime) || 0) - subtitleOffset.value
  }

  function applyOffsetToSentenceData(sentenceData) {
    if (!sentenceData) return sentenceData
    const delta = subtitleOffset.value
    const start = toDisplayTime(sentenceData.start ?? 0)
    const end = toDisplayTime(sentenceData.end ?? 0)
    return {
      ...sentenceData,
      start: Math.max(0, start),
      end: Math.max(Math.max(0, start), end),
      words: shiftWords(sentenceData.words, delta),
    }
  }

  function applyOffsetToSegments(segments) {
    if (!Array.isArray(segments)) return []
    return segments.map((segment) => {
      const start = toDisplayTime(segment.start ?? 0)
      const end = toDisplayTime(segment.end ?? 0)
      return {
        ...segment,
        start: Math.max(0, start),
        end: Math.max(Math.max(0, start), end),
      }
    })
  }

  function setSubtitleOffset(value, options = {}) {
    const { applyDelta = true, syncMeta = true } = options
    const normalized = Math.round(clampSubtitleOffset(value) * 1000) / 1000
    const previous = subtitleOffset.value
    subtitleOffset.value = normalized
    if (syncMeta) {
      writeMetaSubtitleOffsetHandler(normalized)
    }
    if (applyDelta) {
      applyOffsetDeltaHandler(normalized - previous)
    }
    return normalized
  }

  return {
    subtitleOffset,
    bindRuntime,
    setSubtitleOffset,
    toBaseTime,
    toDisplayTime,
    applyOffsetToSegments,
    applyOffsetToSentenceData,
  }
})
