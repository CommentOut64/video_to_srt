import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('VirtualSubtitleList FLIP style regression', () => {
  it('禁止给 vue-recycle-scroller item-view 直接加 transform transition', () => {
    const source = readFileSync(
      resolve(import.meta.dirname, './index.vue'),
      'utf8'
    )

    expect(source).not.toContain('.vue-recycle-scroller__item-view')
    expect(source).not.toContain('transition: transform 150ms')
  })

  it('row-shell 间距仍由参与测量的内层承担', () => {
    const source = readFileSync(
      resolve(import.meta.dirname, './index.vue'),
      'utf8'
    )

    expect(source).toContain('.row-shell {')
    expect(source).toContain('padding-bottom: 6px;')
  })
})
