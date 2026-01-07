/**
 * SSE 频道管理器（增强版）
 *
 * 职责：
 * - 管理多个 SSE 连接（全局频道、单任务频道、模型频道）
 * - 自动重连机制
 * - 事件分发和处理
 * - 支持全局事件流和任务事件流
 */

import EventEmitter from 'eventemitter3'

class SSEChannelManager extends EventEmitter {
  constructor() {
    super()

    // 频道连接池 { channelId: EventSource }
    this.channels = new Map()

    // 频道配置 { channelId: config }
    this.channelConfigs = new Map()

    // 重连状态 { channelId: { timer, attempts } }
    this.reconnectState = new Map()

    // [V3.1.0] 延迟关闭定时器 { channelId: timeoutId }
    // 用于在取消订阅时提供短暂缓冲期，避免快速切换页面时频繁重建连接
    this.pendingUnsubscribeTimers = new Map()

    // [V3.1.0] 延迟关闭的缓冲时间（毫秒）
    this.UNSUBSCRIBE_DELAY = 2000

    // 基础 URL
    this.baseURL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

    console.log('[SSEChannelManager] 频道管理器已初始化')
  }

  /**
   * 订阅全局事件流（所有任务状态变化）
   * @param {Object} handlers - 事件处理器
   *   {
   *     onInitialState: (state) => void,
   *     onQueueUpdate: (queue) => void,
   *     onJobStatus: (jobId, status, data) => void,
   *     onJobProgress: (jobId, progress, data) => void
   *   }
   * @returns {Function} 取消订阅函数
   */
  subscribeGlobal(handlers = {}) {
    const channelId = 'global'
    const url = `${this.baseURL}/api/events/global`

    return this._subscribe(channelId, url, {
      initial_state: (data) => {
        console.log('[SSE Global] 初始状态:', data)
        handlers.onInitialState?.(data)
      },
      queue_update: (data) => {
        console.log('[SSE Global] 队列更新:', data)
        handlers.onQueueUpdate?.(data.queue)
      },
      job_status: (data) => {
        // V3.1.0: 状态事件不应该包含进度信息，避免归零
        // 只有当后端明确提供了 percent/progress 时才传递，否则不设置默认值
        const percent = data.percent ?? data.progress
        console.log('[SSE Global] 任务状态:', data)
        // 使用 data.id 而非 data.job_id，兼容全局频道的字段名
        // 只有当 percent 存在时才添加到 data 中
        const statusData = percent !== undefined ? { ...data, percent } : data
        handlers.onJobStatus?.(data.id || data.job_id, data.status, statusData)
      },
      job_progress: (data) => {
        // 兼容处理：优先使用percent，fallback到progress
        const percent = data.percent ?? data.progress ?? 0
        console.log('[SSE Global] 任务进度:', data.id || data.job_id, percent)
        // 使用 data.id 而非 data.job_id，兼容全局频道的字段名
        handlers.onJobProgress?.(data.id || data.job_id, percent, { ...data, percent })
      },
      // [V3.1.0] 新增：任务删除事件，解决幽灵任务问题
      job_removed: (data) => {
        console.log('[SSE Global] 任务删除:', data.job_id)
        handlers.onJobRemoved?.(data.job_id)
      },
      connected: (data) => {
        console.log('[SSE Global] 连接成功:', data)
        handlers.onConnected?.(data)
      },
      ping: () => {
        // 心跳事件，用于保持连接活跃
        handlers.onPing?.()
      }
    })
  }

  /**
   * 订阅单个任务事件流
   * @param {string} jobId - 任务ID
   * @param {Object} handlers - 事件处理器
   *   {
   *     onProgress: (data) => void,
   *     onSignal: (signal, data) => void,
   *     onComplete: (data) => void,
   *     onFailed: (data) => void,
   *     onProxyProgress: (progress) => void,
   *     // Phase 5: 双模态架构新增
   *     onDraft: (data) => void,           // 草稿字幕（快流/SenseVoice）
   *     onReplaceChunk: (data) => void     // 替换 Chunk（慢流/Whisper）
   *   }
   * @returns {Function} 取消订阅函数
   */
  subscribeJob(jobId, handlers = {}) {
    const channelId = `job:${jobId}`
    const url = `${this.baseURL}/api/stream/${jobId}`

    // V3.1.0: 区分总体进度和阶段进度
    // 总体进度处理函数 - 用于更新主进度条
    const handleOverallProgress = (data) => {
      const percent = data.percent ?? data.progress ?? 0
      console.log(`[SSE Job ${jobId}] 总体进度:`, percent, data.detail)
      handlers.onProgress?.({ ...data, percent })
    }

    // 阶段进度处理函数 - 仅用于日志和特定阶段处理，不更新主进度条
    const handlePhaseProgress = (data) => {
      const percent = data.percent ?? data.progress ?? 0
      console.log(`[SSE Job ${jobId}] 阶段进度:`, data.phase || 'unknown', percent)
      // 阶段进度不调用 onProgress，避免覆盖总体进度导致抖动
      handlers.onPhaseProgress?.({ ...data, percent })
    }

    // 统一的信号处理函数
    const handleSignal = (data) => {
      const signal = data.signal || data.code
      console.log(`[SSE Job ${jobId}] 信号:`, signal)

      // 分发特定信号事件
      if (signal === 'job_complete') {
        handlers.onComplete?.(data)
      } else if (signal === 'job_failed') {
        handlers.onFailed?.(data)
      } else if (signal === 'job_paused') {
        handlers.onPaused?.(data)
      } else if (signal === 'job_canceled') {
        handlers.onCanceled?.(data)
      } else if (signal === 'job_resumed') {
        handlers.onResumed?.(data)
      }

      handlers.onSignal?.(signal, data)
    }

    // 统一的片段处理函数
    const handleSegment = (data) => {
      console.log(`[SSE Job ${jobId}] 片段:`, data)
      handlers.onSegment?.(data)
    }

    // 统一的对齐完成处理函数
    const handleAligned = (data) => {
      console.log(`[SSE Job ${jobId}] 对齐完成:`, data)
      handlers.onAligned?.(data)
    }

    // 统一的对齐进度处理函数
    const handleAlignProgress = (data) => {
      console.log(`[SSE Job ${jobId}] 对齐进度:`, data)
      handlers.onAlignProgress?.(data)
    }

    return this._subscribe(channelId, url, {
      // === 初始状态 ===
      initial_state: (data) => {
        console.log(`[SSE Job ${jobId}] 初始状态:`, data)
        handlers.onInitialState?.(data)
      },

      // === 进度事件（新格式带命名空间前缀） ===
      // V3.1.0: 只有 progress.overall 更新主进度条，其他阶段进度单独处理
      'progress.overall': handleOverallProgress,
      'progress.extract': handlePhaseProgress,
      'progress.bgm_detect': handlePhaseProgress,
      'progress.spectrum_analysis': handlePhaseProgress,
      'progress.demucs': handlePhaseProgress,
      'progress.vad': handlePhaseProgress,
      'progress.sensevoice': handlePhaseProgress,
      'progress.whisper': handlePhaseProgress,
      'progress.llm_proof': handlePhaseProgress,
      'progress.llm_trans': handlePhaseProgress,
      'progress.srt': handlePhaseProgress,
      'progress.fast': handlePhaseProgress,   // V3.1.0: 新增 fast/slow 阶段
      'progress.slow': handlePhaseProgress,   // V3.1.0: 新增 fast/slow 阶段
      'progress.preprocess': handlePhaseProgress,  // V3.1.0: 新增预处理阶段
      'progress.align': handleAlignProgress,

      // === 信号事件（新格式带命名空间前缀） ===
      'signal.job_start': handleSignal,
      'signal.job_complete': handleSignal,
      'signal.job_failed': handleSignal,
      'signal.job_paused': handleSignal,
      'signal.job_canceled': handleSignal,
      'signal.job_resumed': handleSignal,
      'signal.phase_start': handleSignal,
      'signal.phase_complete': handleSignal,
      'signal.circuit_breaker': (data) => {
        console.log(`[SSE Job ${jobId}] 熔断事件:`, data)
        handlers.onCircuitBreaker?.(data)
        handleSignal(data)
      },
      'signal.model_upgrade': (data) => {
        console.log(`[SSE Job ${jobId}] 模型升级:`, data)
        handlers.onModelUpgrade?.(data)
        handleSignal(data)
      },
      'signal.bgm_detected': (data) => {
        console.log(`[SSE Job ${jobId}] BGM 检测:`, data)
        handlers.onBgmDetected?.(data)
      },
      'signal.separation_strategy': (data) => {
        console.log(`[SSE Job ${jobId}] 分离策略:`, data)
        handlers.onSeparationStrategy?.(data)
      },

      // === 字幕流式事件（新格式带命名空间前缀） ===
      'subtitle.segment': handleSegment,
      'subtitle.aligned': handleAligned,

      // Phase 5: 双模态架构 - 草稿事件（快流/SenseVoice）
      'subtitle.draft': (data) => {
        console.log(`[SSE Job ${jobId}] 草稿字幕:`, data)
        handlers.onDraft?.(data)
        handlers.onSubtitleUpdate?.(data)
      },

      // Phase 5: 双模态架构 - 替换 Chunk 事件（慢流/Whisper）
      'subtitle.replace_chunk': (data) => {
        console.log(`[SSE Job ${jobId}] 替换 Chunk:`, data)
        handlers.onReplaceChunk?.(data)
        handlers.onSubtitleUpdate?.(data)
      },

      // V3.1.0: 字幕恢复事件（断点续传后恢复字幕）
      'subtitle.restored': (data) => {
        console.log(`[SSE Job ${jobId}] 恢复字幕:`, data)
        handlers.onRestored?.(data)
        handlers.onSubtitleUpdate?.(data)
      },

      // V3.5: 极速模式定稿事件
      'subtitle.finalized': (data) => {
        console.log(`[SSE Job ${jobId}] 定稿字幕:`, data)
        handlers.onFinalized?.(data)
        handlers.onSubtitleUpdate?.(data)
      },

      // 旧版事件 (兼容)
      'subtitle.sv_sentence': (data) => {
        console.log(`[SSE Job ${jobId}] SenseVoice 句子:`, data)
        handlers.onSvSentence?.(data)
        handlers.onSubtitleUpdate?.(data)
      },
      'subtitle.whisper_patch': (data) => {
        console.log(`[SSE Job ${jobId}] Whisper 补刀:`, data)
        handlers.onWhisperPatch?.(data)
        handlers.onSubtitleUpdate?.(data)
      },
      'subtitle.llm_proof': (data) => {
        console.log(`[SSE Job ${jobId}] LLM 校对:`, data)
        handlers.onLlmProof?.(data)
        handlers.onSubtitleUpdate?.(data)
      },
      'subtitle.llm_trans': (data) => {
        console.log(`[SSE Job ${jobId}] LLM 翻译:`, data)
        handlers.onLlmTrans?.(data)
        handlers.onSubtitleUpdate?.(data)
      },
      'subtitle.batch_update': (data) => {
        console.log(`[SSE Job ${jobId}] 批量更新:`, data)
        handlers.onBatchUpdate?.(data)
        handlers.onSubtitleUpdate?.(data)
      },

      // === 视频转码相关事件 ===
      // 分析完成事件（新增：智能转码决策）
      analyze_complete: (data) => {
        console.log(`[SSE Job ${jobId}] 分析完成:`, data.decision)
        handlers.onAnalyzeComplete?.(data)
      },
      // 容器重封装进度（新增：零转码优化）
      remux_progress: (data) => {
        console.log(`[SSE Job ${jobId}] 重封装进度:`, data.progress)
        handlers.onRemuxProgress?.(data)
      },
      // 容器重封装完成（新增）
      remux_complete: (data) => {
        console.log(`[SSE Job ${jobId}] 重封装完成`)
        handlers.onRemuxComplete?.(data)
      },
      // Proxy 错误事件（新增：统一错误处理）
      proxy_error: (data) => {
        console.error(`[SSE Job ${jobId}] Proxy 错误:`, data.message)
        handlers.onProxyError?.(data)
      },
      // 360p 预览进度
      preview_360p_progress: (data) => {
        console.log(`[SSE Job ${jobId}] 360p 预览进度:`, data.progress)
        handlers.onPreview360pProgress?.(data)
      },
      // 360p 预览完成
      preview_360p_complete: (data) => {
        console.log(`[SSE Job ${jobId}] 360p 预览完成:`, data)
        handlers.onPreview360pComplete?.(data)
      },
      // 720p Proxy 进度
      proxy_progress: (data) => {
        console.log(`[SSE Job ${jobId}] Proxy 进度:`, data.progress)
        handlers.onProxyProgress?.(data)
      },
      // 720p Proxy 完成
      proxy_complete: (data) => {
        console.log(`[SSE Job ${jobId}] Proxy 完成:`, data)
        handlers.onProxyComplete?.(data)
      },

      // === 连接和心跳 ===
      connected: (data) => {
        console.log(`[SSE Job ${jobId}] 连接成功`)
        handlers.onConnected?.(data)
      },
      ping: () => {
        handlers.onPing?.()
      }
    })
  }

  /**
   * 订阅模型下载进度
   * @param {Object} handlers - 事件处理器
   *   {
   *     onProgress: (modelId, progress, data) => void,
   *     onComplete: (modelId, data) => void
   *   }
   * @returns {Function} 取消订阅函数
   */
  subscribeModels(handlers = {}) {
    const channelId = 'models'
    const url = `${this.baseURL}/api/models/stream`

    return this._subscribe(channelId, url, {
      model_progress: (data) => {
        console.log('[SSE Models] 下载进度:', data.model_id, data.progress)
        handlers.onProgress?.(data.model_id, data.progress, data)
      },
      model_complete: (data) => {
        console.log('[SSE Models] 下载完成:', data.model_id)
        handlers.onComplete?.(data.model_id, data)
      },
      connected: (data) => {
        console.log('[SSE Models] 连接成功')
        handlers.onConnected?.(data)
      },
      ping: () => {
        // 心跳事件，用于保持连接活跃
        handlers.onPing?.()
      }
    })
  }

  /**
   * 通用订阅方法（内部使用）
   * @private
   */
  _subscribe(channelId, url, eventHandlers) {
    // [V3.1.0] 如果有待执行的延迟关闭，先取消它（用户快速切换回来的场景）
    this._cancelPendingUnsubscribe(channelId)

    // 检查是否已存在连接
    const existingConnection = this.channels.get(channelId)

    // 关键修复：即使连接存在，也需要重建以确保新的事件处理器被绑定
    // 原因：EventSource 的事件监听器是在创建时绑定的，复用连接会导致新组件的回调无法被调用
    if (existingConnection) {
      console.log(`[SSEChannelManager] 频道 ${channelId} 存在旧连接，关闭后重建以更新事件处理器`)
      // 关闭旧连接
      existingConnection.close()
      this.channels.delete(channelId)
      // 清除重连定时器
      const reconnectInfo = this.reconnectState.get(channelId)
      if (reconnectInfo?.timer) {
        clearTimeout(reconnectInfo.timer)
      }
      this.reconnectState.delete(channelId)
    }

    // 保存配置
    this.channelConfigs.set(channelId, { url, eventHandlers })

    // 创建连接
    this._createConnection(channelId, url, eventHandlers)

    // 返回取消订阅函数
    return () => this._requestUnsubscribe(channelId)
  }

  /**
   * 请求取消订阅（延迟执行，避免页面切换时误关闭）
   * [V3.1.0] 修复：实现真正的延迟关闭逻辑，解决 EventSource 连接泄漏问题
   * @private
   */
  _requestUnsubscribe(channelId) {
    // 检查是否已有待执行的关闭定时器
    const existingTimer = this.pendingUnsubscribeTimers.get(channelId)
    if (existingTimer) {
      // 已有待执行的关闭请求，无需重复设置
      console.log(`[SSEChannelManager] 延迟取消订阅已排队: ${channelId}`)
      return
    }

    console.log(`[SSEChannelManager] 延迟取消订阅: ${channelId}（${this.UNSUBSCRIBE_DELAY}ms 后执行）`)

    // 设置延迟关闭定时器
    const timer = setTimeout(() => {
      console.log(`[SSEChannelManager] 执行延迟取消订阅: ${channelId}`)
      // 清除定时器记录
      this.pendingUnsubscribeTimers.delete(channelId)
      // 真正执行关闭
      this.unsubscribe(channelId)
    }, this.UNSUBSCRIBE_DELAY)

    this.pendingUnsubscribeTimers.set(channelId, timer)
  }

  /**
   * 取消待执行的延迟关闭定时器
   * [V3.1.0] 新增：在重新订阅时取消待执行的关闭操作
   * @private
   */
  _cancelPendingUnsubscribe(channelId) {
    const timer = this.pendingUnsubscribeTimers.get(channelId)
    if (timer) {
      console.log(`[SSEChannelManager] 取消延迟关闭: ${channelId}`)
      clearTimeout(timer)
      this.pendingUnsubscribeTimers.delete(channelId)
    }
  }

  /**
   * 创建 EventSource 连接
   * @private
   */
  _createConnection(channelId, url, eventHandlers) {
    try {
      console.log(`[SSEChannelManager] 创建连接: ${channelId} -> ${url}`)

      const eventSource = new EventSource(url, {
        withCredentials: false
      })

      this.channels.set(channelId, eventSource)

      // 监听所有事件类型
      for (const [eventType, handler] of Object.entries(eventHandlers)) {
        eventSource.addEventListener(eventType, (event) => {
          try {
            const data = JSON.parse(event.data)
            handler(data)
          } catch (error) {
            console.error(`[SSE ${channelId}] 解析事件失败:`, eventType, error, event.data)
          }
        })
      }

      // 连接成功
      eventSource.onopen = () => {
        console.log(`[SSE ${channelId}] 连接成功`)
        // 清除重连状态
        this.reconnectState.delete(channelId)
      }

      // 错误处理
      eventSource.onerror = (error) => {
        console.error(`[SSE ${channelId}] 连接错误:`, error)

        // 关键修复：检查HTTP状态码，如果是404等客户端错误，不重连
        // EventSource 在404时会触发error事件，但不会暴露status code
        // 我们可以通过readyState判断，如果是CLOSED则可能是不可恢复错误
        const readyState = eventSource.readyState

        // 对于job频道，如果连接失败，先尝试验证任务是否存在
        if (channelId.startsWith('job:') && readyState === EventSource.CLOSED) {
          const jobId = channelId.split(':')[1]

          // 尝试调用一个轻量级API验证任务是否存在
          fetch(`${this.baseURL}/api/status/${jobId}`)
            .then(res => {
              if (res.status === 404) {
                console.error(`[SSE ${channelId}] 任务不存在，停止重连`)
                // 清理资源，不再重连
                this.channels.delete(channelId)
                this.channelConfigs.delete(channelId)
                this.reconnectState.delete(channelId)
                if (eventSource) eventSource.close()
                return
              }

              // 任务存在但连接失败，进行重连
              if (readyState === EventSource.CLOSED) {
                console.warn(`[SSE ${channelId}] 连接已关闭但任务存在，尝试重连`)
                this._scheduleReconnect(channelId)
              }
            })
            .catch(() => {
              // 验证请求失败，也进行重连（可能是网络问题）
              if (readyState === EventSource.CLOSED) {
                this._scheduleReconnect(channelId)
              }
            })
        } else {
          // 其他频道或readyState不是CLOSED，正常重连
          if (readyState === EventSource.CLOSED) {
            console.warn(`[SSE ${channelId}] 连接已关闭，尝试重连`)
            this._scheduleReconnect(channelId)
          }
        }
      }

    } catch (error) {
      console.error(`[SSE ${channelId}] 创建连接失败:`, error)
      this._scheduleReconnect(channelId)
    }
  }

  /**
   * 计划重连
   * @private
   */
  _scheduleReconnect(channelId) {
    const config = this.channelConfigs.get(channelId)
    if (!config) {
      console.warn(`[SSE ${channelId}] 无配置信息，无法重连`)
      return
    }

    // 获取重连状态
    let reconnectInfo = this.reconnectState.get(channelId)
    if (!reconnectInfo) {
      reconnectInfo = { attempts: 0, timer: null }
      this.reconnectState.set(channelId, reconnectInfo)
    }

    // 清除旧的定时器
    if (reconnectInfo.timer) {
      clearTimeout(reconnectInfo.timer)
    }

    reconnectInfo.attempts++

    // 最大重连次数限制（防止任务不存在时无限重连）
    const MAX_RECONNECT_ATTEMPTS = 5
    if (reconnectInfo.attempts > MAX_RECONNECT_ATTEMPTS) {
      console.warn(`[SSE ${channelId}] 已达到最大重连次数 (${MAX_RECONNECT_ATTEMPTS})，停止重连`)
      // 清理资源
      this.channels.delete(channelId)
      this.channelConfigs.delete(channelId)
      this.reconnectState.delete(channelId)
      return
    }

    // 计算重连延迟（指数退避，最大 30 秒）
    const delay = Math.min(1000 * Math.pow(2, reconnectInfo.attempts - 1), 30000)

    console.log(`[SSE ${channelId}] ${delay}ms 后尝试第 ${reconnectInfo.attempts}/${MAX_RECONNECT_ATTEMPTS} 次重连`)

    reconnectInfo.timer = setTimeout(() => {
      // 先关闭旧连接
      const oldEventSource = this.channels.get(channelId)
      if (oldEventSource) {
        oldEventSource.close()
        this.channels.delete(channelId)
      }

      // 创建新连接
      this._createConnection(channelId, config.url, config.eventHandlers)
    }, delay)
  }

  /**
   * 取消订阅指定频道
   * [V3.1.0] 增强：显式清理事件处理器引用，防止闭包泄漏
   * @param {string} channelId - 频道ID
   */
  unsubscribe(channelId) {
    console.log(`[SSEChannelManager] 取消订阅: ${channelId}`)

    // [V3.1.0] 清除待执行的延迟关闭定时器
    this._cancelPendingUnsubscribe(channelId)

    // 关闭 EventSource
    const eventSource = this.channels.get(channelId)
    if (eventSource) {
      // [V3.1.0] 移除所有事件监听器（防止闭包引用泄漏）
      // EventSource 的 onopen/onerror/onmessage 设为 null
      eventSource.onopen = null
      eventSource.onerror = null
      eventSource.onmessage = null
      eventSource.close()
      this.channels.delete(channelId)
    }

    // 清除重连定时器
    const reconnectInfo = this.reconnectState.get(channelId)
    if (reconnectInfo?.timer) {
      clearTimeout(reconnectInfo.timer)
    }
    this.reconnectState.delete(channelId)

    // [V3.1.0] 删除配置（包含事件处理器闭包引用）
    // 这是关键：channelConfigs 中的 eventHandlers 持有对 store 的引用
    const config = this.channelConfigs.get(channelId)
    if (config) {
      // 显式清空 eventHandlers 对象中的所有函数引用
      if (config.eventHandlers) {
        for (const key of Object.keys(config.eventHandlers)) {
          config.eventHandlers[key] = null
        }
      }
      this.channelConfigs.delete(channelId)
    }
  }

  /**
   * 取消所有订阅
   */
  unsubscribeAll() {
    console.log('[SSEChannelManager] 取消所有订阅')

    // [V3.1.0] 先清除所有待执行的延迟关闭定时器
    for (const timer of this.pendingUnsubscribeTimers.values()) {
      clearTimeout(timer)
    }
    this.pendingUnsubscribeTimers.clear()

    // 关闭所有频道
    for (const channelId of this.channels.keys()) {
      this.unsubscribe(channelId)
    }
  }

  /**
   * 手动重连指定频道
   * @param {string} channelId - 频道ID
   */
  reconnect(channelId) {
    console.log(`[SSEChannelManager] 手动重连: ${channelId}`)

    this.unsubscribe(channelId)

    const config = this.channelConfigs.get(channelId)
    if (config) {
      this._createConnection(channelId, config.url, config.eventHandlers)
    } else {
      console.error(`[SSE ${channelId}] 无配置信息，无法重连`)
    }
  }

  /**
   * 获取所有活跃频道
   * @returns {string[]}
   */
  getActiveChannels() {
    return Array.from(this.channels.keys())
  }

  /**
   * 检查频道是否已连接
   * @param {string} channelId - 频道ID
   * @returns {boolean}
   */
  isChannelActive(channelId) {
    const eventSource = this.channels.get(channelId)
    return eventSource && eventSource.readyState === EventSource.OPEN
  }

  /**
   * 销毁管理器
   */
  destroy() {
    console.log('[SSEChannelManager] 销毁管理器')
    this.unsubscribeAll()
    this.removeAllListeners()
  }
}

// 导出单例
export const sseChannelManager = new SSEChannelManager()
export default sseChannelManager
