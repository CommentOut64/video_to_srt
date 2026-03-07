/**
 * 项目域 API 客户端。
 *
 * 说明：
 * - 后端 project 路由统一返回 { success, data } 包装。
 * - 这里统一做解包，保证调用方拿到稳定结构。
 */

import { apiClient } from './client'
import { FLAVOR } from '@/config/flavor'

const DEFAULT_IMPORT_UPLOAD_TIMEOUT_MS = 10 * 60 * 1000
const ENV_IMPORT_UPLOAD_TIMEOUT_MS = Number(import.meta.env.VITE_IMPORT_UPLOAD_TIMEOUT_MS)
const IMPORT_UPLOAD_TIMEOUT_MS =
  Number.isFinite(ENV_IMPORT_UPLOAD_TIMEOUT_MS) && ENV_IMPORT_UPLOAD_TIMEOUT_MS > 0
    ? ENV_IMPORT_UPLOAD_TIMEOUT_MS
    : DEFAULT_IMPORT_UPLOAD_TIMEOUT_MS

function unwrapEnvelope(response, fallback = null) {
  if (response && typeof response === 'object' && 'data' in response) {
    return response.data ?? fallback
  }
  return response ?? fallback
}

class ProjectAPI {
  async importProject(subtitleFile, format = 'srt', videoFile = null, title = '') {
    const formData = new FormData()
    formData.append('subtitle_file', subtitleFile)
    formData.append('subtitle_format', format)
    formData.append('flavor', FLAVOR)
    if (title) {
      formData.append('title', title)
    }
    if (videoFile) {
      formData.append('video_file', videoFile)
    }

    const response = await apiClient.post('/api/projects/import', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: IMPORT_UPLOAD_TIMEOUT_MS,
    })
    return unwrapEnvelope(response, null)
  }

  /**
   * 从 input 目录本地路径导入项目（无需文件上传）
   * @param {string} subtitleFilename - input 目录中的字幕文件名
   * @param {string|null} mediaFilename - input 目录中的媒体文件名（可选）
   * @param {string} title - 项目名称
   * @returns {Promise<Object>} 项目数据
   */
  async importProjectLocal(subtitleFilename, mediaFilename = null, title = '') {
    const formData = new FormData()
    formData.append('subtitle_filename', subtitleFilename)
    formData.append('flavor', FLAVOR)
    if (title) formData.append('title', title)
    if (mediaFilename) formData.append('media_filename', mediaFilename)

    const response = await apiClient.post('/api/projects/import-local', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return unwrapEnvelope(response, null)
  }

  async listProjects(flavor = null) {
    const params = flavor ? { flavor } : undefined
    const response = await apiClient.get('/api/projects', { params })
    return unwrapEnvelope(response, [])
  }

  async getProject(projectId) {
    const response = await apiClient.get(`/api/projects/${projectId}`)
    return unwrapEnvelope(response, null)
  }

  async updateProjectTitle(projectId, title) {
    const response = await apiClient.patch(`/api/projects/${projectId}/title`, { title })
    return unwrapEnvelope(response, null)
  }

  async getSubtitles(projectId) {
    const response = await apiClient.get(`/api/projects/${projectId}/subtitles`)
    return unwrapEnvelope(response, [])
  }

  async updateSubtitle(projectId, segmentId, update) {
    const response = await apiClient.patch(
      `/api/projects/${projectId}/subtitles/${segmentId}`,
      update
    )
    return unwrapEnvelope(response, null)
  }

  async createSubtitle(projectId, payload) {
    const response = await apiClient.post(`/api/projects/${projectId}/subtitles`, payload)
    return unwrapEnvelope(response, null)
  }

  async deleteSubtitle(projectId, segmentId) {
    const response = await apiClient.delete(`/api/projects/${projectId}/subtitles/${segmentId}`)
    return unwrapEnvelope(response, null)
  }

  // V3.2.4+dev.20260303.01: undo/redo 批量同步
  // 不走 unwrapEnvelope，保留完整 {success, data} envelope 供调用方判断业务失败
  async batchSyncSubtitles(projectId, diff) {
    const response = await apiClient.post(
      `/api/projects/${projectId}/subtitles/batch-sync`,
      diff
    )
    return response
  }

  async applyEditorOps(projectId, payload) {
    const response = await apiClient.post(
      `/api/projects/${projectId}/editor-ops:apply`,
      payload
    )
    return response
  }

  async exportSubtitles(projectId, format = 'srt') {
    const response = await apiClient.get(`/api/projects/${projectId}/export`, {
      params: { format },
    })
    return unwrapEnvelope(response, null)
  }

  getMediaUrl(projectId, assetType) {
    return `${apiClient.defaults.baseURL || 'http://localhost:8000'}/api/media/${projectId}/${assetType}`
  }
}

export default new ProjectAPI()
