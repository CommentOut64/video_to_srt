/**
 * Project 语义任务控制 API。
 *
 * 说明：
 * - 所有接口以 project_id 为主语义；
 * - 返回体保留 job_id 兼容字段。
 */

import { apiClient } from './client'

class ProjectTaskAPI {
  /**
   * 获取任务状态（project 语义）。
   * @param {string} projectId
   * @param {boolean} includeMedia
   * @returns {Promise<Object>}
   */
  async getTaskStatus(projectId, includeMedia = true) {
    return apiClient.get(`/api/projects/${projectId}/tasks/status`, {
      params: { include_media: includeMedia },
    })
  }

  /**
   * 暂停任务。
   * @param {string} projectId
   * @returns {Promise<Object>}
   */
  async pauseTask(projectId) {
    return apiClient.post(`/api/projects/${projectId}/tasks/pause`)
  }

  /**
   * 恢复任务。
   * @param {string} projectId
   * @returns {Promise<Object>}
   */
  async resumeTask(projectId) {
    return apiClient.post(`/api/projects/${projectId}/tasks/resume`)
  }

  /**
   * 取消任务。
   * @param {string} projectId
   * @param {boolean} deleteData
   * @returns {Promise<Object>}
   */
  async cancelTask(projectId, deleteData = false) {
    return apiClient.post(`/api/projects/${projectId}/tasks/cancel`, null, {
      params: { delete_data: deleteData },
    })
  }

  /**
   * 插队任务。
   * @param {string} projectId
   * @param {string} mode
   * @returns {Promise<Object>}
   */
  async prioritizeTask(projectId, mode = 'gentle') {
    return apiClient.post(`/api/projects/${projectId}/tasks/prioritize`, null, {
      params: { mode },
    })
  }

  /**
   * 获取队列状态（project 语义镜像）。
   * @returns {Promise<Object>}
   */
  async getQueueStatus() {
    return apiClient.get('/api/projects/tasks/queue-status')
  }

  /**
   * 同步任务列表（project 语义镜像）。
   * @returns {Promise<Object>}
   */
  async syncTasks() {
    return apiClient.get('/api/projects/tasks/sync')
  }

  /**
   * 获取任务运行时诊断快照。
   * @returns {Promise<Object>}
   */
  async getRuntimeDiagnostics() {
    return apiClient.get('/api/projects/tasks/runtime-diagnostics')
  }
}

export default new ProjectTaskAPI()
