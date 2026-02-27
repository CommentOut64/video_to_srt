import { beforeEach, vi } from 'vitest'

function createMemoryStorage() {
  const memory = new Map()
  return {
    clear() {
      memory.clear()
    },
    getItem(key) {
      return memory.has(key) ? memory.get(key) : null
    },
    setItem(key, value) {
      memory.set(String(key), String(value))
    },
    removeItem(key) {
      memory.delete(String(key))
    },
  }
}

if (!globalThis.localStorage) {
  globalThis.localStorage = createMemoryStorage()
}

if (!globalThis.sessionStorage) {
  globalThis.sessionStorage = createMemoryStorage()
}

if (!globalThis.window) {
  globalThis.window = {
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  }
}

vi.mock('vue-router', () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
  }),
  useRoute: () => ({
    query: {},
    params: {},
  }),
  onBeforeRouteLeave: vi.fn(),
}))

const smartSaverMemory = new Map()
function safeClone(payload) {
  return JSON.parse(JSON.stringify(payload))
}
globalThis.__resetSmartSaverMemoryForTests = () => {
  smartSaverMemory.clear()
}

vi.mock('@/services/SmartSaver', () => ({
  default: {
    onSaveSuccess: null,
    onSaveError: null,
    save(payload) {
      smartSaverMemory.set(payload.jobId, safeClone(payload))
      return true
    },
    async forceSave(payload) {
      smartSaverMemory.set(payload.jobId, safeClone(payload))
      if (typeof this.onSaveSuccess === 'function') {
        this.onSaveSuccess(payload.jobId)
      }
      return true
    },
    async restoreFromBackup(jobId) {
      const hit = smartSaverMemory.get(jobId)
      return hit ? safeClone(hit) : null
    },
  },
}))

beforeEach(() => {
  globalThis.localStorage.clear()
  globalThis.sessionStorage.clear()
  globalThis.__resetSmartSaverMemoryForTests()
})
