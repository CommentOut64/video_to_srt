/**
 * 系统管理 API
 * 包含系统关闭、客户端心跳、在线更新等功能
 */

import { apiClient } from './client'

/**
 * 检查是否有活跃的客户端
 * @returns {Promise<{has_active: boolean, count: number}>}
 */
export async function hasActiveClients() {
  return apiClient.get('/api/system/has-active-clients')
}

/**
 * 客户端注册
 * @param {string} clientId - 客户端ID
 * @param {string} [userAgent] - 客户端User-Agent
 * @returns {Promise<{success: boolean, client_id: string}>}
 */
export async function registerClient(clientId, userAgent = null) {
  return apiClient.post('/api/system/register', {
    client_id: clientId,
    user_agent: userAgent
  })
}

/**
 * 客户端心跳
 * @param {string} clientId - 客户端ID
 * @returns {Promise<{success: boolean, active_clients: number}>}
 */
export async function heartbeat(clientId) {
  return apiClient.post('/api/system/heartbeat', {
    client_id: clientId
  })
}

/**
 * 客户端注销
 * @param {string} clientId - 客户端ID
 * @returns {Promise<{success: boolean}>}
 */
export async function unregisterClient(clientId) {
  return apiClient.post('/api/system/unregister', {
    client_id: clientId
  })
}

/**
 * 关闭系统
 * @param {Object} options - 关闭选项
 * @param {boolean} [options.cleanup_temp=false] - 是否清理临时文件
 * @param {boolean} [options.force=false] - 是否强制关闭
 * @returns {Promise<{success: boolean, message: string, cleanup_report: Object}>}
 */
export async function shutdownSystem(options = {}) {
  return apiClient.post('/api/system/shutdown', {
    cleanup_temp: options.cleanup_temp || false,
    force: options.force || false
  })
}

// Phase 5: 硬件信息获取
/**
 * 获取硬件信息和优化配置
 * @returns {Promise<{success: boolean, hardware: Object, optimization: Object}>}
 */
export async function getHardwareInfo() {
  return apiClient.get('/api/hardware/status')
}

/**
 * 获取基础硬件信息
 * @returns {Promise<{success: boolean, hardware: Object}>}
 */
export async function getHardwareBasic() {
  return apiClient.get('/api/hardware/basic')
}

// ========== 在线更新 API (V3.1.1+dev.20260105.01) ==========

/**
 * 获取当前系统版本
 * @returns {Promise<{success: boolean, version: string, build_date: string}>}
 */
export async function getVersion() {
  return apiClient.get('/api/system/version')
}

/**
 * 检查是否有新版本可用
 * @returns {Promise<{
 *   has_update: boolean,
 *   current_version: string,
 *   latest_version?: string,
 *   changelog?: string,
 *   download_url?: string,
 *   force_update?: boolean
 * }>}
 */
export async function checkUpdate() {
  return apiClient.get('/api/system/check-update')
}

/**
 * 触发更新流程
 * @param {Object} options - 更新选项
 * @param {string} options.download_url - 更新包下载地址
 * @param {string} options.version - 目标版本号
 * @param {string} [options.changelog] - 更新日志
 * @param {boolean} [options.delay_mode=false] - 延迟模式（重启时更新）
 * @returns {Promise<{success: boolean, message: string, signal_file: string, delay_mode: boolean}>}
 */
export async function triggerUpdate(options) {
  return apiClient.post('/api/system/trigger-update', {
    download_url: options.download_url,
    version: options.version,
    changelog: options.changelog || '',
    delay_mode: options.delay_mode || false
  })
}

const systemApi = {
  hasActiveClients,
  registerClient,
  heartbeat,
  unregisterClient,
  shutdownSystem,
  getHardwareInfo,
  getHardwareBasic,
  // 在线更新 API
  getVersion,
  checkUpdate,
  triggerUpdate
}

export default systemApi
