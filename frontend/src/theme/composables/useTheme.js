import { computed, readonly } from 'vue'
import { useAppPreferenceStore } from '@/stores/appPreferenceStore'

/**
 * 主题管理兼容层
 *
 * Phase 4：主题状态迁移至 appPreferenceStore。
 * 该 composable 保留原 API，避免一次性修改全部调用方。
 */
export function useTheme() {
  const preferenceStore = useAppPreferenceStore()

  return {
    currentThemeId: readonly(computed(() => preferenceStore.currentThemeId)),
    currentTheme: readonly(computed(() => preferenceStore.currentTheme)),
    isDark: readonly(computed(() => preferenceStore.isDark)),
    availableThemes: readonly(computed(() => preferenceStore.availableThemes)),
    isInitialized: readonly(computed(() => preferenceStore.isInitialized)),
    initialize: preferenceStore.initialize,
    setTheme: preferenceStore.setTheme,
    toggleDarkMode: preferenceStore.toggleDarkMode,
  }
}
