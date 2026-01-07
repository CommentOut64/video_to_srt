/**
 * 模型管理 API
 *
 * 职责：管理 Whisper 模型的下载、删除
 */

import { apiClient } from './client'

class ModelAPI {
  /**
   * 获取 Whisper 模型列表
   * @returns {Promise<Array<{
   *   id: string,
   *   name: string,
   *   size: number,
   *   status: string,
   *   downloaded: boolean
   * }>>}
   */
  async listWhisperModels() {
    return apiClient.get('/api/models/whisper')
  }

  /**
   * 下载 Whisper 模型
   * @param {string} modelId - 模型ID (tiny, base, small, medium, large-v2, large-v3)
   * @returns {Promise<{success: boolean, message: string}>}
   */
  async downloadWhisperModel(modelId) {
    return apiClient.post(`/api/models/whisper/${modelId}/download`)
  }

  /**
   * 删除 Whisper 模型
   * @param {string} modelId - 模型ID
   * @returns {Promise<{success: boolean, message: string}>}
   */
  async deleteWhisperModel(modelId) {
    return apiClient.delete(`/api/models/whisper/${modelId}`)
  }
}

// 导出单例实例
export default new ModelAPI()
