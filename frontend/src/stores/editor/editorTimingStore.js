// V3.2.5+dev.20260314.01: 时间域状态管理
import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useEditorTimingStore = defineStore('editorTiming', () => {
  const offsetMs = ref(0)

  function setOffsetMs(ms) {
    if (Math.abs(ms) > 10000) {
      console.warn('offsetMs 超出 ±10s 范围，已截断')
      ms = Math.max(-10000, Math.min(10000, ms))
    }
    offsetMs.value = ms
  }

  function toDisplayMs(baseMs) {
    return baseMs + offsetMs.value
  }

  function toBaseMs(displayMs) {
    return displayMs - offsetMs.value
  }

  function secondsToMs(s) {
    return Math.round(s * 1000)
  }

  function msToSeconds(ms) {
    return ms / 1000
  }

  return { offsetMs, setOffsetMs, toDisplayMs, toBaseMs, secondsToMs, msToSeconds }
})
