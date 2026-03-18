import { describe, expect, it } from 'vitest'
import { shouldIgnoreStructuralShortcut } from './keyboardSafety'

describe('shouldIgnoreStructuralShortcut', () => {
  it('忽略来自 textarea 的危险结构快捷键', () => {
    const target = {
      tagName: 'TEXTAREA',
      isContentEditable: false,
      closest: () => null,
    }

    expect(shouldIgnoreStructuralShortcut({
      key: 'Delete',
      target,
      isComposing: false,
    })).toBe(true)
  })

  it('忽略来自 contenteditable 子节点的危险结构快捷键', () => {
    const child = {
      tagName: 'SPAN',
      isContentEditable: false,
      closest: (selector) => (selector.includes('[contenteditable') ? {} : null),
    }

    expect(shouldIgnoreStructuralShortcut({
      key: 'Delete',
      target: child,
      isComposing: false,
    })).toBe(true)
  })

  it('输入法组合态时忽略危险结构快捷键', () => {
    const target = {
      tagName: 'DIV',
      isContentEditable: false,
      closest: () => null,
    }

    expect(shouldIgnoreStructuralShortcut({
      key: 'Delete',
      target,
      isComposing: true,
    })).toBe(true)
  })

  it('普通列表容器上的 Delete 不应被忽略', () => {
    const target = {
      tagName: 'DIV',
      isContentEditable: false,
      closest: () => null,
    }

    expect(shouldIgnoreStructuralShortcut({
      key: 'Delete',
      target,
      isComposing: false,
    })).toBe(false)
  })
})
