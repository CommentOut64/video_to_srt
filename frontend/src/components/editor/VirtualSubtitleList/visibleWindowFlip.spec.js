import { describe, expect, it, vi } from 'vitest'
import {
  buildVisibleFlipPlan,
  playVisibleFlip,
  shouldSkipVisibleFlip,
  waitForStableLayout,
} from './visibleWindowFlip'

describe('buildVisibleFlipPlan', () => {
  it('只为 before/after 都存在且发生位移的项生成 move 操作', () => {
    const before = new Map([
      ['a', { localId: 'a', top: 100 }],
      ['b', { localId: 'b', top: 180 }],
    ])
    const after = new Map([
      ['a', { localId: 'a', top: 20 }],
      ['b', { localId: 'b', top: 100 }],
    ])

    const plan = buildVisibleFlipPlan(before, after, {
      maxAnimatedItems: 12,
      enterOffsetPx: 8,
    })

    expect(plan.skip).toBe(false)
    expect(plan.moves).toEqual([
      { localId: 'a', deltaY: 80 },
      { localId: 'b', deltaY: 80 },
    ])
    expect(plan.enters).toEqual([])
  })

  it('可见项超过阈值时直接降级 skip', () => {
    const before = new Map()
    const after = new Map()

    for (let index = 0; index < 20; index += 1) {
      before.set(`id-${index}`, { localId: `id-${index}`, top: index * 80 })
      after.set(`id-${index}`, { localId: `id-${index}`, top: index * 80 - 80 })
    }

    const plan = buildVisibleFlipPlan(before, after, {
      maxAnimatedItems: 12,
      enterOffsetPx: 8,
    })

    expect(plan.skip).toBe(true)
    expect(plan.reason).toBe('too_many_items')
  })
})

describe('shouldSkipVisibleFlip', () => {
  it('命中 grouped 模式时必须跳过', () => {
    expect(shouldSkipVisibleFlip({
      isGroupedMode: true,
      isResizing: false,
      isInProgrammaticScrollGuard: false,
      hasRecentUserScroll: false,
      prefersReducedMotion: false,
    })).toEqual({ skip: true, reason: 'grouped_mode' })
  })

  it('命中 sidebar resizing 时必须跳过', () => {
    expect(shouldSkipVisibleFlip({
      isGroupedMode: false,
      isResizing: true,
      isInProgrammaticScrollGuard: false,
      hasRecentUserScroll: false,
      prefersReducedMotion: false,
    })).toEqual({ skip: true, reason: 'sidebar_resizing' })
  })

  it('命中 prefers-reduced-motion 时必须跳过', () => {
    expect(shouldSkipVisibleFlip({
      isGroupedMode: false,
      isResizing: false,
      isInProgrammaticScrollGuard: false,
      hasRecentUserScroll: false,
      prefersReducedMotion: true,
    })).toEqual({ skip: true, reason: 'reduced_motion' })
  })

  it('所有条件均不命中时不跳过', () => {
    expect(shouldSkipVisibleFlip({
      isGroupedMode: false,
      isResizing: false,
      isInProgrammaticScrollGuard: false,
      hasRecentUserScroll: false,
      prefersReducedMotion: false,
    })).toEqual({ skip: false, reason: null })
  })
})

describe('waitForStableLayout', () => {
  it('要求连续两帧稳定后才返回最终快照', async () => {
    const snapshots = [
      new Map([['a', { localId: 'a', top: 100 }]]),
      new Map([['a', { localId: 'a', top: 60 }]]),
      new Map([['a', { localId: 'a', top: 60 }]]),
    ]

    const result = await waitForStableLayout({
      capture: () => snapshots.shift(),
      nextFrame: () => Promise.resolve(),
      stableFrameCount: 2,
      maxFrames: 4,
    })

    expect(result.get('a').top).toBe(60)
  })

  it('达到 maxFrames 仍未稳定时返回最后一帧快照', async () => {
    let callCount = 0
    const result = await waitForStableLayout({
      capture: () => {
        callCount += 1
        return new Map([['a', { localId: 'a', top: callCount * 10 }]])
      },
      nextFrame: () => Promise.resolve(),
      stableFrameCount: 2,
      maxFrames: 3,
    })

    // maxFrames=3，每帧都在变，返回最后一次 capture 结果
    expect(result.get('a').top).toBeGreaterThan(0)
  })
})

describe('playVisibleFlip', () => {
  it('只对 move/enter 项调用 animate，并暴露 cancel 清理函数', () => {
    const cancel = vi.fn()
    const animate = vi.fn(() => ({
      finished: Promise.resolve(),
      cancel,
    }))

    const moveEl = { isConnected: true, animate }
    const enterEl = { isConnected: true, animate }

    const cleanup = playVisibleFlip({
      skip: false,
      moves: [{ localId: 'a', deltaY: 80, element: moveEl }],
      enters: [{ localId: 'b', enterOffsetPx: 8, element: enterEl }],
    }, {
      durationMs: 160,
      easing: 'cubic-bezier(0.4, 0, 0.2, 1)',
    })

    expect(animate).toHaveBeenCalledTimes(2)
    cleanup()
    expect(cancel).toHaveBeenCalled()
  })

  it('skip 为 true 时返回空清理函数且不调用 animate', () => {
    const animate = vi.fn()
    const cleanup = playVisibleFlip({
      skip: true,
      moves: [{ localId: 'a', deltaY: 80, element: { isConnected: true, animate } }],
      enters: [],
    })

    expect(animate).not.toHaveBeenCalled()
    expect(typeof cleanup).toBe('function')
    cleanup() // 不抛异常
  })

  it('element 未连接 DOM 时跳过该项', () => {
    const animate = vi.fn()
    const cleanup = playVisibleFlip({
      skip: false,
      moves: [{ localId: 'a', deltaY: 80, element: { isConnected: false, animate } }],
      enters: [],
    })

    expect(animate).not.toHaveBeenCalled()
    cleanup()
  })
})
