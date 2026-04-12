// V3.2.5+dev.20260314.03: 同音搜索延迟初始化
import { ref } from 'vue'

let homophoneEngine = null

export function useHomophoneSearch() {
  const isActive = ref(false)
  const isLoading = ref(false)

  async function activate() {
    if (isActive.value || isLoading.value) return

    isLoading.value = true

    // 懒加载同音搜索引擎（模拟异步加载）
    await new Promise(resolve => setTimeout(resolve, 100))

    homophoneEngine = {
      initialized: true
    }

    isActive.value = true
    isLoading.value = false
  }

  function search(text) {
    if (!isActive.value || !homophoneEngine) return []
    // 简化实现：返回空数组
    return []
  }

  function deactivate() {
    homophoneEngine = null
    isActive.value = false
  }

  return { isActive, isLoading, activate, search, deactivate }
}
