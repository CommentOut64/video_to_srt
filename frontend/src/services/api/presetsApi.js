import { apiClient } from './client'

/**
 * 自定义预设 API
 */
const presetsApi = {
  /** 获取所有自定义预设 */
  async getCustomPresets() {
    return apiClient.get('/api/presets/custom')
  },

  /** 新增自定义预设 */
  async createCustomPreset(name, config) {
    return apiClient.post('/api/presets/custom', { name, config })
  },

  /** 删除自定义预设 */
  async deleteCustomPreset(presetId) {
    return apiClient.delete(`/api/presets/custom/${presetId}`)
  }
}

export default presetsApi
