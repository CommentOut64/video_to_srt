/**
 * HTTP 客户端基础配置
 *
 * 职责：
 * - 创建配置好的 Axios 实例
 * - 添加请求/响应拦截器
 * - 统一错误处理和转换
 * - 支持请求取消
 */

import axios from 'axios'

// 自定义错误类
export class APIError extends Error {
  constructor(status, message, data = null) {
    super(message)
    this.name = 'APIError'
    this.status = status
    this.data = data
  }
}

export class NetworkError extends Error {
  constructor(message) {
    super(message)
    this.name = 'NetworkError'
  }
}

// 创建 Axios 实例
export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || window.location.origin,
  timeout: 30000, // 默认 30 秒超时
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    // 可以在这里添加 token 等认证信息
    // const token = localStorage.getItem('token')
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`
    // }

    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器 - 统一错误处理
apiClient.interceptors.response.use(
  (response) => {
    // 成功响应，直接返回 data
    return response.data
  },
  (error) => {
    if (error?.code === 'ECONNABORTED') {
      throw new NetworkError('请求超时，请检查媒体文件体积后重试')
    }

    if (error.response) {
      // HTTP 错误响应 (4xx, 5xx)
      const { status, data } = error.response

      let message = '请求失败'

      // 处理特定的 HTTP 状态码
      if (status === 404) {
        message = data?.detail || '资源未找到'
      } else if (status === 400) {
        message = data?.detail || '请求参数错误'
      } else if (status === 401) {
        message = '未授权，请登录'
      } else if (status === 403) {
        message = '无权限访问'
      } else if (status === 500) {
        message = data?.detail || '服务器内部错误'
      } else if (status === 202) {
        // 特殊处理：Proxy 视频生成中
        if (data?.detail?.proxy_generating) {
          throw new APIError(202, data.detail.message, data.detail)
        }
      } else {
        message = data?.detail || data?.message || `请求失败 (${status})`
      }

      throw new APIError(status, message, data)
    } else if (error.request) {
      // 网络错误 (无响应)
      throw new NetworkError('网络连接失败，请检查网络设置')
    } else {
      // 其他错误
      throw new Error(error.message || '未知错误')
    }
  }
)

/**
 * 创建可取消的请求
 * @returns {{ token: CancelToken, cancel: Function }}
 */
export function createCancelToken() {
  const source = axios.CancelToken.source()
  return {
    token: source.token,
    cancel: source.cancel
  }
}

/**
 * 检查是否为取消错误
 * @param {Error} error
 * @returns {boolean}
 */
export function isCancelError(error) {
  return axios.isCancel(error)
}
