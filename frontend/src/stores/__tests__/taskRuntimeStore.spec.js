import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useTaskRuntimeStore } from '@/stores/taskRuntimeStore'

describe('TaskRuntimeStore Phase 2', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('统一真源：任务与进度状态共享同一运行时数据', () => {
    const taskStore = useTaskRuntimeStore()
    const progressStore = useTaskRuntimeStore()
    const jobId = 'phase2-job-001'
    const now = Date.now()

    taskStore.addTask({
      job_id: jobId,
      filename: 'phase2-demo.mp4',
      status: 'created',
      progress: 0,
      state_seq: 1,
      updated_at: now,
    })

    progressStore.applySseProgress(jobId, {
      status: 'processing',
      percent: 23,
      phase: 'transcribe',
      state_seq: 2,
      updated_at: now + 10,
    })

    expect(taskStore.$id).toBe('taskRuntime')
    expect(progressStore.$id).toBe('taskRuntime')
    expect(taskStore.getTask(jobId)?.status).toBe('processing')
    expect(taskStore.getTask(jobId)?.progress).toBe(23)
    expect(progressStore.getRawState(jobId).percent).toBe(23)
    expect(progressStore.getRawState(jobId).stateSeq).toBe(2)
  })

  it('防倒退：低序号投影事件被拒绝', () => {
    const taskStore = useTaskRuntimeStore()
    const progressStore = useTaskRuntimeStore()
    const jobId = 'phase2-job-002'
    const now = Date.now()

    taskStore.addTask({
      job_id: jobId,
      filename: 'phase2-regress.mp4',
      status: 'processing',
      progress: 40,
      phase: 'transcribe',
      state_seq: 5,
      updated_at: now,
    })

    progressStore.applySseProgress(jobId, {
      status: 'processing',
      percent: 45,
      state_seq: 6,
      updated_at: now + 10,
    })

    const snapshot = progressStore.applySnapshot(jobId, {
      status: 'queued',
      percent: 10,
      state_seq: 4,
      updated_at: now + 5,
    }, 'http')

    expect(snapshot.lastProjection?.accepted).toBe(false)
    expect(snapshot.lastProjection?.reason).toBe('stale_state_seq')
    expect(taskStore.getTask(jobId)?.status).toBe('processing')
    expect(taskStore.getTask(jobId)?.progress).toBe(45)
  })

  it('同序号同时间戳：低优先级来源不覆盖 SSE 状态', () => {
    const taskStore = useTaskRuntimeStore()
    const progressStore = useTaskRuntimeStore()
    const jobId = 'phase2-job-003'
    const now = Date.now()

    taskStore.addTask({
      job_id: jobId,
      filename: 'phase2-priority.mp4',
      status: 'processing',
      progress: 30,
      state_seq: 6,
      updated_at: now,
    })

    progressStore.applySseProgress(jobId, {
      status: 'processing',
      percent: 50,
      state_seq: 7,
      updated_at: now + 10,
    })

    const snapshot = progressStore.applySnapshot(jobId, {
      status: 'queued',
      percent: 45,
      state_seq: 7,
      updated_at: now + 10,
    }, 'http')

    expect(snapshot.lastProjection?.accepted).toBe(false)
    expect(snapshot.lastProjection?.reason).toBe('same_seq_lower_priority')
    expect(taskStore.getTask(jobId)?.status).toBe('processing')
    expect(taskStore.getTask(jobId)?.progress).toBe(50)
  })
})
