import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { themes, defaultThemeId, getTheme, getAvailableThemes } from '@/theme/tokens'
import { injectTheme, injectBaseTokens } from '@/theme/utils/inject'

const STORAGE_KEY = 'anchorflux-theme'

function loadSavedThemeId() {
  try {
    return localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

function saveThemeId(themeId) {
  try {
    localStorage.setItem(STORAGE_KEY, themeId)
  } catch (error) {
    console.warn('[AppPreferenceStore] 保存主题失败:', error)
  }
}

export const useAppPreferenceStore = defineStore('appPreference', () => {
  const currentThemeId = ref(defaultThemeId)
  const isInitialized = ref(false)

  const currentTheme = computed(() => getTheme(currentThemeId.value))
  const isDark = computed(() => currentThemeId.value === 'dark')
  const availableThemes = computed(() => getAvailableThemes())

  function initialize() {
    if (isInitialized.value) return

    injectBaseTokens()
    const savedTheme = loadSavedThemeId()
    if (savedTheme && themes[savedTheme]) {
      currentThemeId.value = savedTheme
    }

    injectTheme(currentTheme.value)
    isInitialized.value = true
  }

  function setTheme(themeId) {
    const safeThemeId = themes[themeId] ? themeId : defaultThemeId
    currentThemeId.value = safeThemeId
    saveThemeId(safeThemeId)
    injectTheme(getTheme(safeThemeId))
  }

  function toggleDarkMode() {
    setTheme(isDark.value ? 'light' : 'dark')
  }

  return {
    currentThemeId,
    currentTheme,
    isDark,
    availableThemes,
    isInitialized,
    initialize,
    setTheme,
    toggleDarkMode,
  }
})
