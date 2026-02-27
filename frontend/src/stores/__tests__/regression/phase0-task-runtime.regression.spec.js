import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useUnifiedTaskStore } from '@/stores/unifiedTaskStore'
import { useProgressStore } from '@/stores/progressStore'
import { PHASE0_FIXTURES } from './fixtures'

describe('Phase 0 回归 - 任务链路', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('覆盖 上传 -> 转录 -> 暂停/恢复 -> 取消 -> 刷新恢复', () => {
    const taskStore = useUnifiedTaskStore()
    const progressStore = useProgressStore()
    const now = Date.now()
    const { jobId, filename } = PHASE0_FIXTURES

    // 上传
    taskStore.addTask({
      job_id: jobId,
      filename,
      status: 'created',
      progress: 0,
      state_seq: 1,
      updated_at: now,
    })
    expect(taskStore.getTask(jobId)?.status).toBe('created')

    // 转录
    progressStore.applySseProgress(jobId, {
      status: 'processing',
      phase: 'transcribe',
      percent: 12,
      state_seq: 2,
      updated_at: now + 10,
    })
    expect(taskStore.getTask(jobId)?.status).toBe('processing')
    expect(taskStore.getTask(jobId)?.progress).toBe(12)

    // 暂停
    progressStore.markStatus(jobId, 'paused', {
      message: '用户暂停',
      state_seq: 3,
      updated_at: now + 20,
    })
    expect(taskStore.getTask(jobId)?.status).toBe('paused')

    // 恢复
    progressStore.markStatus(jobId, 'queued', {
      message: '恢复排队',
      state_seq: 4,
      updated_at: now + 30,
    })
    progressStore.applySseProgress(jobId, {
      status: 'processing',
      percent: 35,
      state_seq: 5,
      updated_at: now + 40,
    })
    expect(taskStore.getTask(jobId)?.status).toBe('processing')
    expect(taskStore.getTask(jobId)?.progress).toBe(35)

    // 取消
    progressStore.markStatus(jobId, 'canceled', {
      message: '用户取消',
      state_seq: 6,
      updated_at: now + 50,
    })
    expect(taskStore.getTask(jobId)?.status).toBe('canceled')

    // 防倒退：旧序号事件必须被拒绝
    const accepted = taskStore.updateTaskStatus(jobId, 'processing', '旧事件', {
      state_seq: 5,
      updated_at: now + 45,
      isServer: true,
    })
    expect(accepted).toBe(false)
    expect(taskStore.getTask(jobId)?.status).toBe('canceled')

    // 刷新恢复
    taskStore.saveTasks()
    setActivePinia(createPinia())
    const restoredTaskStore = useUnifiedTaskStore()
    const restored = restoredTaskStore.getTask(jobId)
    expect(restored).toBeTruthy()
    expect(restored?.status).toBe('canceled')
    expect(restored?.progress).toBe(35)
    expect(restored?.state_seq).toBe(6)
  })
})
