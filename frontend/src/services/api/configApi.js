import { apiClient } from './client'

/**
 * 配置相关 API
 */
const configApi = {
  async getSubtitleTimeOffset() {
    return apiClient.get('/api/config/subtitle-time-offset')
  },

  async setSubtitleTimeOffset(offset) {
    return apiClient.post('/api/config/subtitle-time-offset', { offset })
  },

  async resetSubtitleTimeOffset() {
    return apiClient.post('/api/config/subtitle-time-offset', { offset: 0 })
  }
}

export default configApi
