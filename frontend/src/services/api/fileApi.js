/**
 * 文件管理 API
 *
 * 职责：管理 input 目录中的文件
 */

import { apiClient } from './client'

class FileAPI {
  /**
   * 获取 input 目录中的所有媒体文件
   * @returns {Promise<{
   *   files: Array<{name: string, size: number, modified: string}>,
   *   input_dir: string,
   *   message?: string
   * }>}
   */
  async listFiles() {
    return apiClient.get('/api/files')
  }

  /**
   * 删除 input 目录中的文件
   * @param {string} filename - 文件名
   * @returns {Promise<{success: boolean, message: string}>}
   */
  async deleteFile(filename) {
    return apiClient.delete(`/api/files/${encodeURIComponent(filename)}`)
  }

  /**
   * 批量创建转录任务（从 input 目录选择多个文件）
   * @param {string[]} filenames - 文件名列表
   * @param {Object|null} taskConfig - 任务级配置（可选）
   * @returns {Promise<{success: boolean, jobs: Array, failed: Array, total: number, succeeded: number, failed_count: number}>}
   */
  async createJobsBatch(filenames, taskConfig = null) {
    return apiClient.post('/api/create-jobs-batch', {
      filenames,
      ...(taskConfig ? { task_config: taskConfig } : {}),
    })
  }

  /**
   * 使用系统文件管理器打开 input 目录
   * @returns {Promise<{success: boolean, message: string}>}
   */
  async openInputFolder() {
    return apiClient.post('/api/files/open-input-folder')
  }
}

// 导出单例实例
export default new FileAPI()
