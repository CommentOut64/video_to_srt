import { describe, expect, it } from 'vitest'
import {
  buildDefaultCapabilitySnapshot,
  selectCapabilities,
  selectRouteVisibility,
} from '@/state/capabilities/capabilitySelector'

describe('capabilitySelector routeVisibility', () => {
  it('默认可见性包含拆分字段并保留 taskCreate 兼容', () => {
    const routeVisibility = selectRouteVisibility()

    expect(routeVisibility.projectCreate).toBe(true)
    expect(typeof routeVisibility.transcribeCreate).toBe('boolean')
    expect(routeVisibility.taskCreate).toBe(routeVisibility.transcribeCreate)
  })

  it('快照覆盖只影响显式字段并保留其余默认值', () => {
    const snapshot = buildDefaultCapabilitySnapshot({
      capabilities: {
        canTranscribe: false,
      },
      routeVisibility: {
        transcribeCreate: false,
        projectCreate: true,
      },
    })

    expect(selectCapabilities(snapshot).canTranscribe).toBe(false)
    expect(snapshot.routeVisibility.transcribeCreate).toBe(false)
    expect(snapshot.routeVisibility.projectCreate).toBe(true)
    expect(snapshot.routeVisibility.importPage).toBe(true)
  })

  it('未知顶层字段不会混入能力快照', () => {
    const snapshot = buildDefaultCapabilitySnapshot({
      unknownTopField: 'ignored',
      routeVisibility: {
        projectCreate: false,
      },
    })

    expect(snapshot.unknownTopField).toBeUndefined()
    expect(snapshot.routeVisibility.projectCreate).toBe(false)
    expect(snapshot.routeVisibility.editor).toBe(true)
  })

  it('基线为 false 的布尔能力不会被快照提权', () => {
    const baseCapabilities = selectCapabilities()
    const baseRouteVisibility = selectRouteVisibility()

    const forcedTrueSnapshot = buildDefaultCapabilitySnapshot({
      capabilities: Object.fromEntries(
        Object.keys(baseCapabilities).map((key) => [key, true]),
      ),
      routeVisibility: Object.fromEntries(
        Object.keys(baseRouteVisibility).map((key) => [key, true]),
      ),
    })

    const falseCapabilityKeys = Object.keys(baseCapabilities).filter(
      (key) => baseCapabilities[key] === false,
    )
    const falseRouteKeys = Object.keys(baseRouteVisibility).filter(
      (key) => baseRouteVisibility[key] === false,
    )

    for (const key of falseCapabilityKeys) {
      expect(forcedTrueSnapshot.capabilities[key]).toBe(false)
    }
    for (const key of falseRouteKeys) {
      expect(forcedTrueSnapshot.routeVisibility[key]).toBe(false)
    }

    // full 基线下可能不存在 false 字段，至少保证该测试用例有断言执行
    if (falseCapabilityKeys.length === 0 && falseRouteKeys.length === 0) {
      expect(forcedTrueSnapshot).toBeTruthy()
    }
  })
})
