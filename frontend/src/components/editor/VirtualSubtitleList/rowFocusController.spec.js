import { describe, expect, it, vi } from 'vitest'
import { focusRowForEditing, resolveDeleteFallbackId } from './rowFocusController'

describe('focusRowForEditing', () => {
  it('编辑态聚焦时应收口到当前行并同步 selectedSubtitleId', () => {
    const selectionStore = {
      selectOnly: vi.fn(),
    }
    const subtitleDocumentStore = {
      setSelectedSubtitleId: vi.fn(),
    }

    focusRowForEditing('row-15', {
      selectionStore,
      subtitleDocumentStore,
    })

    expect(selectionStore.selectOnly).toHaveBeenCalledWith('row-15')
    expect(subtitleDocumentStore.setSelectedSubtitleId).toHaveBeenCalledWith('row-15')
  })

  it('localId 缺失时不应污染选择态', () => {
    const selectionStore = {
      selectOnly: vi.fn(),
    }
    const subtitleDocumentStore = {
      setSelectedSubtitleId: vi.fn(),
    }

    focusRowForEditing('', {
      selectionStore,
      subtitleDocumentStore,
    })

    expect(selectionStore.selectOnly).not.toHaveBeenCalled()
    expect(subtitleDocumentStore.setSelectedSubtitleId).not.toHaveBeenCalled()
  })
})

describe('resolveDeleteFallbackId', () => {
  it('删除中间行时应优先落到相邻后项', () => {
    expect(resolveDeleteFallbackId('row-3', ['row-1', 'row-2', 'row-3', 'row-4'])).toBe('row-4')
  })

  it('删除末尾行时应回退到前一项', () => {
    expect(resolveDeleteFallbackId('row-4', ['row-1', 'row-2', 'row-3', 'row-4'])).toBe('row-3')
  })
})
