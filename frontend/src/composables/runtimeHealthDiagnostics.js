/**
 * 运行时健康诊断（默认关闭）
 *
 * 用法（浏览器控制台）：
 * window.__AF_RUNTIME_HEALTH_DIAG__ = {
 *   enabled: true,
 *   sampleIntervalMs: 10000,
 *   maxSamples: 2000,
 *   logToConsole: false,
 * }
 * window.__AF_RUNTIME_HEALTH__.getState()
 */

const GLOBAL_CONFIG_KEY = '__AF_RUNTIME_HEALTH_DIAG__'
const GLOBAL_STATE_KEY = '__AF_RUNTIME_HEALTH_STATE__'
const GLOBAL_API_KEY = '__AF_RUNTIME_HEALTH__'

const DEFAULT_CONFIG = {
  enabled: false,
  sampleIntervalMs: 60000,
  maxSamples: 1500,
  logToConsole: false,
}

function sanitizeConfig(rawConfig) {
  if (!rawConfig || typeof rawConfig !== 'object') {
    return {}
  }

  const sampleIntervalMs = Number(rawConfig.sampleIntervalMs)
  const maxSamples = Number(rawConfig.maxSamples)
  return {
    enabled: rawConfig.enabled === true,
    sampleIntervalMs: Number.isFinite(sampleIntervalMs)
      ? Math.max(1000, sampleIntervalMs)
      : undefined,
    maxSamples: Number.isFinite(maxSamples)
      ? Math.max(100, Math.floor(maxSamples))
      : undefined,
    logToConsole: rawConfig.logToConsole === true,
  }
}

function getMemoryHeapUsedMb() {
  if (typeof performance === 'undefined') return null
  const mem = performance?.memory
  if (!mem || !Number.isFinite(mem.usedJSHeapSize)) return null
  return Number((mem.usedJSHeapSize / (1024 * 1024)).toFixed(2))
}

function ensureState() {
  if (typeof window === 'undefined') {
    return { samples: [], counters: {}, startedAt: null }
  }
  if (!window[GLOBAL_STATE_KEY]) {
    window[GLOBAL_STATE_KEY] = {
      samples: [],
      counters: {},
      startedAt: null,
    }
  }
  return window[GLOBAL_STATE_KEY]
}

export function getRuntimeHealthDiagnosticsConfig() {
  if (typeof window === 'undefined') {
    return { ...DEFAULT_CONFIG }
  }
  return {
    ...DEFAULT_CONFIG,
    ...sanitizeConfig(window[GLOBAL_CONFIG_KEY]),
  }
}

export function setRuntimeHealthDiagnosticsConfig(nextConfig) {
  if (typeof window === 'undefined') {
    return { ...DEFAULT_CONFIG }
  }
  const mergedConfig = {
    ...getRuntimeHealthDiagnosticsConfig(),
    ...sanitizeConfig(nextConfig),
  }
  window[GLOBAL_CONFIG_KEY] = mergedConfig
  return mergedConfig
}

export function clearRuntimeHealthDiagnosticsState() {
  const state = ensureState()
  state.samples = []
  state.counters = {}
  state.startedAt = null
}

export function getRuntimeHealthDiagnosticsState() {
  const state = ensureState()
  return {
    startedAt: state.startedAt,
    counters: { ...state.counters },
    samples: [...state.samples],
  }
}

export function recordRuntimeHealthCounter(counterKey, delta = 1) {
  const config = getRuntimeHealthDiagnosticsConfig()
  if (!config.enabled) return

  const state = ensureState()
  const key = String(counterKey || '').trim()
  if (!key) return
  const nextDelta = Number.isFinite(Number(delta)) ? Number(delta) : 1
  state.counters[key] = Number(state.counters[key] || 0) + nextDelta
}

export function createRuntimeHealthSampler(source, collectMetrics) {
  const config = getRuntimeHealthDiagnosticsConfig()
  if (!config.enabled || typeof window === 'undefined') {
    return () => {}
  }

  const state = ensureState()
  if (!state.startedAt) {
    state.startedAt = Date.now()
  }

  const sampleOnce = () => {
    const runtimeConfig = getRuntimeHealthDiagnosticsConfig()
    if (!runtimeConfig.enabled) return

    let extraMetrics = {}
    if (typeof collectMetrics === 'function') {
      try {
        const result = collectMetrics()
        if (result && typeof result === 'object') {
          extraMetrics = result
        }
      } catch (error) {
        extraMetrics = { samplerError: String(error) }
      }
    }

    const sample = {
      ts: Date.now(),
      source: String(source || 'unknown'),
      heapUsedMb: getMemoryHeapUsedMb(),
      ...extraMetrics,
    }
    state.samples.push(sample)
    if (state.samples.length > runtimeConfig.maxSamples) {
      state.samples.splice(0, state.samples.length - runtimeConfig.maxSamples)
    }

    if (runtimeConfig.logToConsole) {
      console.log('[RuntimeHealthDiag] sample', sample)
    }
  }

  sampleOnce()
  const timer = window.setInterval(sampleOnce, config.sampleIntervalMs)
  return () => {
    window.clearInterval(timer)
  }
}

if (typeof window !== 'undefined' && !window[GLOBAL_API_KEY]) {
  window[GLOBAL_API_KEY] = {
    getConfig: getRuntimeHealthDiagnosticsConfig,
    setConfig: setRuntimeHealthDiagnosticsConfig,
    getState: getRuntimeHealthDiagnosticsState,
    clear: clearRuntimeHealthDiagnosticsState,
  }
}

