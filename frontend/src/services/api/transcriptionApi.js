/**
 * 转录任务 API
 *
 * 职责：管理转录任务的完整生命周期
 * - 上传文件并创建任务
 * - 启动、暂停、取消任务
 * - 获取任务状态和队列信息
 * - 下载转录结果
 */

import { apiClient } from './client'

class TranscriptionAPI {
  /**
   * 上传文件并创建转录任务
   * @param {File} file - 视频文件对象
   * @param {Function} onProgress - 上传进度回调 (percent) => void
   * @param {Object|null} taskConfig - 任务级配置（可选）
   * @returns {Promise<{job_id: string, filename: string, message: string, queue_position: number}>}
   */
  async uploadFile(file, onProgress = null, taskConfig = null) {
    const formData = new FormData();
    formData.append("file", file);
    if (taskConfig && typeof taskConfig === "object") {
      formData.append("task_config", JSON.stringify(taskConfig));
    }

    const config = {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    };

    // 添加上传进度监听
    if (onProgress) {
      config.onUploadProgress = (progressEvent) => {
        const percent = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total,
        );
        onProgress(percent);
      };
    }

    return apiClient.post("/api/upload", formData, config);
  }

  /**
   * 为本地 input 文件创建转录任务
   * @param {string} filename - 文件名
   * @param {Object|null} taskConfig - 任务级配置（可选）
   * @returns {Promise<{job_id: string, filename: string}>}
   */
  async createJob(filename, taskConfig = null) {
    const formData = new FormData();
    formData.append("filename", filename);
    if (taskConfig && typeof taskConfig === "object") {
      formData.append("task_config", JSON.stringify(taskConfig));
    }

    return apiClient.post("/api/create-job", formData);
  }

  /**
   * 启动转录任务（加入队列）
   *
   * v3.5+ 建议仅使用 task_config 字段（前端当前只发送新版配置）
   *
   * @param {string} jobId - 任务ID
   * @param {Object} settings - 转录设置
   *
   * === v3.5 新版配置 (推荐) ===
   * @param {Object} [settings.task_config] - v3.5 任务配置
   * @param {string} [settings.task_config.preset_id] - 预设ID (fast/balanced/quality/custom)
   * @param {Object} [settings.task_config.preprocessing] - 预处理设置
   * @param {string} [settings.task_config.preprocessing.demucs_strategy] - 人声分离策略 (off/auto/force_on)
   * @param {string} [settings.task_config.preprocessing.demucs_model] - Demucs 模型
   * @param {number} [settings.task_config.preprocessing.demucs_shifts] - 分离预测次数 (1-5)
   * @param {number} [settings.task_config.preprocessing.spectrum_threshold] - 分诊灵敏度 (0.0-1.0)
   * @param {boolean} [settings.task_config.preprocessing.vad_filter] - VAD 静音过滤
   * @param {Object} [settings.task_config.transcription] - 转录设置
   * @param {string} [settings.task_config.transcription.transcription_profile] - 转录流水线 (sensevoice_only/sv_whisper_patch/sv_whisper_dual)
   * @param {string} [settings.task_config.transcription.sensevoice_device] - SenseVoice 设备 (auto/cpu)
   * @param {string} [settings.task_config.transcription.whisper_model] - Whisper 模型 (tiny/small/medium/large-v3)
   * @param {number} [settings.task_config.transcription.patching_threshold] - 复核触发阈值 (0.0-1.0)
   * @param {Object} [settings.task_config.refinement] - 增强设置
   * @param {string} [settings.task_config.refinement.llm_task] - LLM 任务 (off/proofread/translate)
   * @param {string} [settings.task_config.refinement.llm_scope] - LLM 范围 (sparse/global)
   * @param {number} [settings.task_config.refinement.sparse_threshold] - 稀疏校对阈值 (0.0-1.0)
   * @param {string} [settings.task_config.refinement.target_language] - 目标语言
   * @param {string} [settings.task_config.refinement.llm_provider] - LLM 提供商
   * @param {string} [settings.task_config.refinement.llm_model_name] - LLM 模型名称
   * @param {Object} [settings.task_config.compute] - 计算设置
   * @param {string} [settings.task_config.compute.concurrency_strategy] - 并发策略 (auto/parallel/serial)
   * @param {number} [settings.task_config.compute.gpu_id] - GPU ID
   * @param {string} [settings.task_config.compute.temp_file_policy] - 临时文件策略
   *
   * @returns {Promise<{job_id: string, started: boolean, queue_position: number}>}
   */
  async startJob(jobId, settings) {
    const formData = new FormData();
    formData.append("job_id", jobId);
    formData.append("settings", JSON.stringify(settings));

    // FormData 会自动设置正确的 Content-Type (multipart/form-data with boundary)
    return apiClient.post("/api/start", formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  }

  /**
   * 取消任务
   * @param {string} jobId - 任务ID
   * @param {boolean} deleteData - 是否删除任务数据
   * @returns {Promise<{job_id: string, canceled: boolean, data_deleted: boolean}>}
   */
  async cancelJob(jobId, deleteData = false) {
    return apiClient.post(`/api/cancel/${jobId}`, null, {
      params: { delete_data: deleteData },
    });
  }

  /**
   * 暂停任务
   * @param {string} jobId - 任务ID
   * @returns {Promise<{job_id: string, paused: boolean}>}
   */
  async pauseJob(jobId) {
    return apiClient.post(`/api/pause/${jobId}`);
  }

  /**
   * 恢复暂停的任务（重新加入队列）
   *
   * 与 restoreJob 不同：
   * - resumeJob: 恢复暂停的任务，重新加入队列尾部，状态变为 queued
   * - restoreJob: 从 checkpoint 断点续传
   *
   * @param {string} jobId - 任务ID
   * @returns {Promise<{job_id: string, resumed: boolean, status: string, queue_position: number}>}
   */
  async resumeJob(jobId) {
    return apiClient.post(`/api/resume/${jobId}`);
  }

  /**
   * 任务插队
   * @param {string} jobId - 任务ID
   * @param {string} mode - 插队模式 ('gentle' | 'force')
   * @returns {Promise<{job_id: string, prioritized: boolean, mode: string, queue_position: number}>}
   */
  async prioritizeJob(jobId, mode = "gentle") {
    return apiClient.post(`/api/prioritize/${jobId}`, null, {
      params: { mode },
    });
  }

  /**
   * 获取任务状态
   * @param {string} jobId - 任务ID
   * @param {boolean} includeMedia - 是否包含媒体状态信息
   * @returns {Promise<Object>} 完整任务对象
   */
  async getJobStatus(jobId, includeMedia = true) {
    return apiClient.get(`/api/status/${jobId}`, {
      params: { include_media: includeMedia },
    });
  }

  /**
   * 获取任务级字幕时间偏移（无则回退全局）
   * @param {string} jobId - 任务ID
   * @returns {Promise<{offset: number, source: string}>}
   */
  async getJobSubtitleTimeOffset(jobId) {
    return apiClient.get(`/api/jobs/${jobId}/subtitle-time-offset`);
  }

  /**
   * 设置任务级字幕时间偏移（等于全局则不保存）
   * @param {string} jobId - 任务ID
   * @param {number} offset - 偏移量（秒）
   * @returns {Promise<{offset: number, source: string}>}
   */
  async setJobSubtitleTimeOffset(jobId, offset) {
    return apiClient.post(`/api/jobs/${jobId}/subtitle-time-offset`, { offset });
  }

  /**
   * 获取队列状态摘要
   * @returns {Promise<{queue: string[], running: string, interrupted: string, jobs: Object}>}
   */
  async getQueueStatus() {
    return apiClient.get("/api/queue-status");
  }

  /**
   * 获取队列设置
   * @returns {Promise<{default_prioritize_mode: string}>}
   */
  async getQueueSettings() {
    return apiClient.get("/api/queue-settings");
  }

  /**
   * 更新队列设置
   * @param {string} defaultPrioritizeMode - 默认插队模式 ('gentle' | 'force')
   * @returns {Promise<{success: boolean, settings: Object}>}
   */
  async updateQueueSettings(defaultPrioritizeMode) {
    return apiClient.post("/api/queue-settings", {
      default_prioritize_mode: defaultPrioritizeMode,
    });
  }

  /**
   * 重新排序队列
   * @param {string[]} jobIds - 按新顺序排列的任务ID列表
   * @returns {Promise<{reordered: boolean, queue: string[]}>}
   */
  async reorderQueue(jobIds) {
    return apiClient.post("/api/reorder-queue", {
      job_ids: jobIds,
    });
  }

  /**
   * 下载 SRT 文件
   * @param {string} jobId - 任务ID
   * @param {boolean} copyToSource - 是否复制到源文件目录
   * @returns {Promise<Blob>} SRT 文件 Blob
   */
  async downloadResult(jobId, copyToSource = false) {
    const response = await apiClient.get(`/api/download/${jobId}`, {
      params: { copy_to_source: copyToSource },
      responseType: "blob",
    });
    return response;
  }

  /**
   * 复制转录结果到源文件目录
   * @param {string} jobId - 任务ID
   * @returns {Promise<{success: boolean, message: string, target_path: string}>}
   */
  async copyResultToSource(jobId) {
    return apiClient.post(`/api/copy-result/${jobId}`);
  }

  /**
   * 生成 ASS 字幕文件
   * @param {string} jobId - 任务ID
   * @param {Object} options - 生成选项
   * @param {string} options.style_preset - 样式预设 (default/movie/news/danmaku)
   * @param {string} options.title - 字幕标题
   * @param {number} options.video_width - 视频宽度
   * @param {number} options.video_height - 视频高度
   * @returns {Promise<{job_id: string, filename: string, message: string}>}
   */
  async generateASS(jobId, options = {}) {
    return apiClient.post(`/api/media/${jobId}/ass/generate`, {
      style_preset: options.style_preset || "default",
      title: options.title || "Untitled",
      video_width: options.video_width || 1920,
      video_height: options.video_height || 1080,
    });
  }

  /**
   * 获取 ASS 字幕文件内容
   * @param {string} jobId - 任务ID
   * @returns {Promise<{job_id: string, filename: string, content: string, encoding: string}>}
   */
  async getASSContent(jobId) {
    return apiClient.get(`/api/media/${jobId}/ass`);
  }

  /**
   * 同步所有任务（第一阶段修复：数据同步）
   *
   * 从后端获取所有实际存在的任务列表（处理中 + 已完成）
   * 用于在应用启动时同步前端的 localStorage 与后端 jobs 目录的一致性
   * 修复幽灵任务问题
   *
   * @returns {Promise<{success: boolean, tasks: Array, count: number, timestamp: number}>}
   */
  async syncTasks() {
    return apiClient.get("/api/sync-tasks");
  }

  /**
   * 获取所有未完成的任务
   * @returns {Promise<{jobs: Object[], count: number}>}
   */
  async getIncompleteJobs() {
    return apiClient.get("/api/incomplete-jobs");
  }

  /**
   * 检查任务是否可以断点续传
   * @param {string} jobId - 任务ID
   * @returns {Promise<{can_resume: boolean, progress: number, message: string}>}
   */
  async checkResume(jobId) {
    return apiClient.get(`/api/check-resume/${jobId}`);
  }

  /**
   * 从检查点恢复任务
   * @param {string} jobId - 任务ID
   * @returns {Promise<Object>} 任务对象
   */
  async restoreJob(jobId) {
    return apiClient.post(`/api/restore-job/${jobId}`);
  }

  /**
   * 获取 checkpoint 中保存的原始设置
   * @param {string} jobId - 任务ID
   * @returns {Promise<{has_checkpoint: boolean, original_settings: Object, progress: Object}>}
   */
  async getCheckpointSettings(jobId) {
    return apiClient.get(`/api/checkpoint-settings/${jobId}`);
  }

  /**
   * 获取已完成的转录文字（从checkpoint）
   * @param {string} jobId - 任务ID
   * @returns {Promise<{job_id: string, segments: Array, progress: Object}>}
   */
  async getTranscriptionText(jobId) {
    return apiClient.get(`/api/transcription-text/${jobId}`);
  }

  /**
   * 校验恢复任务时的参数修改
   * @param {string} jobId - 任务ID
   * @param {Object} newSettings - 新设置
   * @returns {Promise<{valid: boolean, warnings: Array, errors: Array, force_original: Object}>}
   */
  async validateResumeSettings(jobId, newSettings) {
    const formData = new FormData();
    formData.append("job_id", jobId);
    formData.append("new_settings", JSON.stringify(newSettings));

    return apiClient.post("/api/validate-resume-settings", formData);
  }

  /**
   * 获取任务缩略图（任务卡片展示用）
   * @param {string} jobId - 任务ID
   * @returns {Promise<{thumbnail: string|null, message: string}>} Base64编码的JPEG缩略图或null
   */
  async getThumbnail(jobId) {
    return apiClient.get(`/api/media/${jobId}/thumbnail`);
  }

  /**
   * 重命名任务
   * @param {string} jobId - 任务ID
   * @param {string} title - 新的任务名称（为空时恢复使用 filename）
   * @returns {Promise<{success: boolean, job_id: string, title: string, message: string}>}
   */
  async renameJob(jobId, title) {
    return apiClient.post(`/api/rename-job/${jobId}`, {
      title,
    });
  }

  /**
   * V3.2.0+dev.20260124.01: 更新字幕（用户编辑）
   * @param {string} jobId - 任务ID
   * @param {number} sentenceIndex - 句子索引
   * @param {Object} update - 更新内容
   * @param {string} [update.text] - 新文本
   * @param {number} [update.start] - 新开始时间
   * @param {number} [update.end] - 新结束时间
   * @returns {Promise<{success: boolean, data: Object}>}
   */
  async updateSubtitle(jobId, sentenceIndex, update) {
    return apiClient.patch(`/api/jobs/${jobId}/subtitles/${sentenceIndex}`, update);
  }

  /**
   * V3.2.0+dev.20260124.02: 新增字幕（用户手动添加）
   * @param {string} jobId - 任务ID
   * @param {Object} payload - 新字幕内容
   * @param {string} [payload.text] - 文本
   * @param {number} payload.start - 开始时间
   * @param {number} payload.end - 结束时间
   * @returns {Promise<{success: boolean, data: Object}>}
   */
  async createSubtitle(jobId, payload) {
    return apiClient.post(`/api/jobs/${jobId}/subtitles`, payload);
  }

  /**
   * V3.2.0+dev.20260124.02: 删除字幕（用户手动删除）
   * @param {string} jobId - 任务ID
   * @param {number} sentenceIndex - 句子索引
   * @returns {Promise<{success: boolean, data: Object}>}
   */
  async deleteSubtitle(jobId, sentenceIndex) {
    return apiClient.delete(`/api/jobs/${jobId}/subtitles/${sentenceIndex}`);
  }

  /**
   * 获取说话人资料列表
   * @param {string} jobId - 任务ID
   * @returns {Promise<{job_id: string, profiles: Array}>}
   */
  async listSpeakerProfiles(jobId) {
    return apiClient.get(`/api/speakers/${jobId}/profiles`);
  }

  /**
   * 获取指定说话人的字幕绑定列表
   * @param {string} jobId - 任务ID
   * @param {string} speakerId - 说话人ID
   * @returns {Promise<{job_id: string, speaker_id: string, subtitles: Array}>}
   */
  async listSpeakerSubtitles(jobId, speakerId) {
    return apiClient.get(`/api/speakers/${jobId}/profiles/${speakerId}/subtitles`);
  }

  /**
   * 更新说话人资料（名称/颜色/锁定/状态）
   * @param {string} jobId - 任务ID
   * @param {string} speakerId - 说话人ID
   * @param {Object} payload - 更新字段
   * @returns {Promise<{success: boolean, data: Object, revision_id: string, updated_at: number}>}
   */
  async updateSpeakerProfile(jobId, speakerId, payload) {
    return apiClient.patch(`/api/speakers/${jobId}/profiles/${speakerId}`, payload);
  }

  /**
   * 改绑句级 speaker
   * @param {string} jobId - 任务ID
   * @param {number} sentenceIndex - 句子索引
   * @param {string} speakerId - 目标说话人ID
   * @returns {Promise<{success: boolean, data: Object, revision_id: string, updated_at: number}>}
   */
  async rebindSubtitleSpeaker(jobId, sentenceIndex, speakerId) {
    return apiClient.patch(`/api/speakers/${jobId}/subtitles/${sentenceIndex}`, {
      speaker_id: speakerId,
    });
  }

  /**
   * 合并说话人
   * @param {string} jobId - 任务ID
   * @param {string} sourceSpeakerId - 源说话人ID
   * @param {string} targetSpeakerId - 目标说话人ID
   * @returns {Promise<{success: boolean, data: Object, revision_id: string, updated_at: number}>}
   */
  async mergeSpeakerProfiles(jobId, sourceSpeakerId, targetSpeakerId) {
    return apiClient.post(`/api/speakers/${jobId}/profiles/merge`, {
      source_speaker_id: sourceSpeakerId,
      target_speaker_id: targetSpeakerId,
    });
  }
}

// 导出单例实例
export default new TranscriptionAPI()
