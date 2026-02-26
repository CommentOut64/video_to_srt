/**
 * API 服务总入口
 *
 * 统一导出所有 API 服务模块
 */

// HTTP 客户端和工具
export { apiClient, createCancelToken, isCancelError, APIError, NetworkError } from './client'

// API 服务模块
export { default as transcriptionApi } from './transcriptionApi'
export { default as mediaApi } from './mediaApi'
export { default as modelApi } from './modelApi'
export { default as fileApi } from './fileApi'
export { default as systemApi } from './systemApi'
export { default as configApi } from './configApi'
export { default as projectApi } from './projectApi'
export { default as legacyApi } from './legacyApi'
export { default as presetsApi } from './presetsApi'
