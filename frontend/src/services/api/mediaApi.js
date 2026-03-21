/**
 * 媒体资源 API
 *
 * 职责：管理编辑器所需的所有媒体资源
 * - 视频、音频文件流
 * - 波形峰值数据
 * - 视频缩略图 / Sprite 图
 * - SRT 字幕内容读写
 * - 媒体信息查询
 */

import { apiClient } from './client'

function readShellMediaCapability() {
  if (typeof window === 'undefined') {
    return {}
  }
  const shell = window.anchorfluxShell || null
  return (
    shell?.getMediaCapabilities?.()
    || shell?.mediaCapabilities
    || {}
  )
}

function buildShellMediaCapabilityParams() {
  const capability = readShellMediaCapability()
  const params = {}

  if (typeof capability.hevcDirectPlay === 'boolean') {
    params.shell_hevc_direct_play = capability.hevcDirectPlay ? '1' : '0'
  }
  if (typeof capability.h264DirectPlay === 'boolean') {
    params.shell_h264_direct_play = capability.h264DirectPlay ? '1' : '0'
  }
  if (capability.activeGpuPreference) {
    params.shell_active_gpu_preference = capability.activeGpuPreference
  }
  return params
}

function appendQueryParams(url, params) {
  const entries = Object.entries(params || {}).filter(([, value]) => value !== undefined && value !== null && value !== '')
  if (entries.length === 0) {
    return url
  }
  const search = new URLSearchParams(entries)
  const separator = url.includes('?') ? '&' : '?'
  return `${url}${separator}${search.toString()}`
}

class MediaAPI {
  /**
   * 获取 API base URL（用于构建媒体资源 URL）
   * @private
   */
  _getBaseURL() {
    return apiClient.defaults.baseURL || 'http://localhost:8000'
  }

  /**
   * 获取视频 URL（支持 Range 请求，自动处理 Proxy）
   * @param {string} identifier - project_id 或兼容 job_id
   * @returns {string} 视频URL
   */
  getVideoUrl(identifier) {
    return appendQueryParams(
      `${this._getBaseURL()}/api/media/${identifier}/video`,
      buildShellMediaCapabilityParams()
    )
  }

  /**
   * 获取音频 URL（支持 Range 请求）
   * @param {string} identifier - project_id 或兼容 job_id
   * @returns {string} 音频URL
   */
  getAudioUrl(identifier) {
    return `${this._getBaseURL()}/api/media/${identifier}/audio`
  }

  /**
   * 获取波形峰值数据
   * @param {string} jobId - 任务ID
   * @param {number} samples - 采样点数（默认2000）
   * @param {string} method - 生成方法 ('auto' | 'ffmpeg' | 'wave')
   * @returns {Promise<{peaks: number[], duration: number, method: string, samples: number}>}
   */
  async getPeaks(jobId, samples = 2000, method = 'auto') {
    return apiClient.get(`/api/media/${jobId}/peaks`, {
      params: { samples, method }
    })
  }

  /**
   * 获取视频缩略图
   * @param {string} jobId - 任务ID
   * @param {number} count - 缩略图数量（默认10）
   * @param {boolean} sprite - 是否使用 Sprite 雪碧图（默认 true）
   * @returns {Promise<Object>} 缩略图数据或Sprite数据
   */
  async getThumbnails(jobId, count = 10, sprite = true) {
    return apiClient.get(`/api/media/${jobId}/thumbnails`, {
      params: { count, sprite }
    })
  }

  /**
   * 获取 Sprite 图片 URL
   * @param {string} jobId - 任务ID
   * @returns {string} Sprite图片URL
   */
  getSpriteUrl(jobId) {
    return `${this._getBaseURL()}/api/media/${jobId}/sprite.jpg`
  }

  /**
   * 获取 SRT 字幕内容
   * @param {string} jobId - 任务ID
   * @returns {Promise<{job_id: string, filename: string, content: string, encoding: string}>}
   */
  async getSRTContent(jobId) {
    return apiClient.get(`/api/media/${jobId}/srt`)
  }

  /**
   * 保存编辑后的 SRT 字幕内容
   * @param {string} jobId - 任务ID
   * @param {string} content - SRT文本内容
   * @returns {Promise<{success: boolean, message: string, filename: string}>}
   */
  async saveSRTContent(jobId, content) {
    return apiClient.post(`/api/media/${jobId}/srt`, {
      content
    })
  }

  /**
   * 获取媒体信息摘要（一次性获取所有资源状态）
   * @param {string} jobId - 任务ID
   * @returns {Promise<{
   *   job_id: string,
   *   video: Object,
   *   audio: Object,
   *   peaks: Object,
   *   thumbnails: Object,
   *   srt: Object
   * }>}
   */
  async getMediaInfo(identifier) {
    return apiClient.get(`/api/media/${identifier}/info`)
  }

  /**
   * 检查 Proxy 视频生成状态
   * @param {string} jobId - 任务ID
   * @returns {Promise<{
   *   exists: boolean,
   *   size: number,
   *   generating: boolean,
   *   progress: number,
   *   status: string,
   *   needs_proxy: boolean
   * }>}
   */
  async getProxyStatus(identifier) {
    return apiClient.get(`/api/media/${identifier}/proxy-status`, {
      params: buildShellMediaCapabilityParams()
    })
  }

  /**
   * 触发转录后处理（预生成编辑器所需的所有资源）
   * @param {string} jobId - 任务ID
   * @returns {Promise<{
   *   peaks: boolean,
   *   thumbnails: boolean,
   *   sprite: boolean,
   *   proxy: boolean,
   *   proxy_needed: boolean
   * }>}
   */
  async postProcess(jobId) {
    return apiClient.post(`/api/media/${jobId}/post-process`)
  }

  /**
   * 轮询检查 Proxy 视频生成状态（直到完成或失败）
   * @param {string} jobId - 任务ID
   * @param {Function} onProgress - 进度回调 (progress) => void
   * @param {number} interval - 轮询间隔（毫秒，默认2000）
   * @returns {Promise<boolean>} 是否生成成功
   */
  async pollProxyStatus(jobId, onProgress = null, interval = 2000) {
    return new Promise((resolve) => {
      const poll = async () => {
        try {
          const status = await this.getProxyStatus(jobId)

          if (onProgress) {
            onProgress(status.progress)
          }

          if (status.exists) {
            resolve(true)
            return
          }

          if (status.status === 'failed') {
            resolve(false)
            return
          }

          if (status.generating) {
            // 继续轮询
            setTimeout(poll, interval)
          } else {
            // 未开始生成
            resolve(false)
          }
        } catch (error) {
          console.error('轮询 Proxy 状态失败:', error)
          resolve(false)
        }
      }

      poll()
    })
  }

  // ========== 渐进式加载 API ==========

  /**
   * 获取 360p 预览视频 URL
   * @param {string} jobId - 任务ID
   * @returns {string} 预览视频URL
   */
  getPreviewVideoUrl(jobId) {
    return appendQueryParams(
      `${this._getBaseURL()}/api/media/${jobId}/video/preview`,
      buildShellMediaCapabilityParams()
    )
  }

  /**
   * 触发 360p 预览视频生成
   * @param {string} jobId - 任务ID
   * @returns {Promise<{ success: boolean, message: string, url?: string, generating?: boolean }>}
   */
  async generatePreview(jobId) {
    return apiClient.post(`/api/media/${jobId}/generate-preview`)
  }
}

// 导出单例实例
export default new MediaAPI()
