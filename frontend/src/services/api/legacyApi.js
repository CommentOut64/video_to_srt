/**
 * 旧任务兼容 API 客户端。
 */

import { apiClient } from './client'

class LegacyAPI {
  async resolveTask(jobId) {
    return apiClient.get(`/api/legacy/tasks/${jobId}/resolve`)
  }

  async getLegacyStatus(jobId) {
    return apiClient.get(`/api/legacy/tasks/${jobId}/status`)
  }
}

export default new LegacyAPI()

