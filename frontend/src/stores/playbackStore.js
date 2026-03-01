import { defineStore } from 'pinia'
import { computed, shallowRef, watch } from 'vue'
import { useProjectStore } from './projectStore'

function normalizeTime(value) {
  const numericValue = Number(value)
  if (!Number.isFinite(numericValue)) return 0
  return Math.max(0, numericValue)
}

export const usePlaybackStore = defineStore('playback', () => {
  const projectStore = useProjectStore()

  // 高频时间通道：供拖拽/时钟同步使用，避免组件层每次都直接写响应式主状态
  const currentTimeRaw = shallowRef(normalizeTime(projectStore.player.currentTime))

  watch(
    () => projectStore.player.currentTime,
    (nextTime) => {
      currentTimeRaw.value = normalizeTime(nextTime)
    },
    { immediate: true }
  )

  const currentTime = computed(() => projectStore.player.currentTime)
  const isPlaying = computed(() => projectStore.player.isPlaying)
  const playbackRate = computed(() => projectStore.player.playbackRate)
  const volume = computed(() => projectStore.player.volume)
  const isSeeking = computed(() => projectStore.player.isSeeking)

  function updateCurrentTimeRaw(time) {
    currentTimeRaw.value = normalizeTime(time)
  }

  function commitCurrentTime(time = currentTimeRaw.value) {
    const normalizedTime = normalizeTime(time)
    currentTimeRaw.value = normalizedTime
    projectStore.seekTo(normalizedTime)
    return normalizedTime
  }

  function setPlaying(playing) {
    projectStore.setIsPlaying(playing)
  }

  function setPlaybackRate(rate) {
    projectStore.setPlaybackRate(rate)
  }

  function setVolume(nextVolume) {
    projectStore.setPlayerVolume(nextVolume)
  }

  function setSeeking(seeking) {
    projectStore.setPlayerSeeking(seeking)
  }

  function reset() {
    currentTimeRaw.value = 0
    projectStore.seekTo(0)
    projectStore.setIsPlaying(false)
    projectStore.setPlaybackRate(1)
    projectStore.setPlayerVolume(1)
    projectStore.setPlayerSeeking(false)
  }

  return {
    currentTimeRaw,
    currentTime,
    isPlaying,
    playbackRate,
    volume,
    isSeeking,
    updateCurrentTimeRaw,
    commitCurrentTime,
    setPlaying,
    setPlaybackRate,
    setVolume,
    setSeeking,
    reset,
  }
})
