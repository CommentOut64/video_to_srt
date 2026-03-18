/**
 * 同音检索与批量替换 Composable
 *
 * 职责：
 * - 管理搜索模式和排序模式状态
 * - 调用同音 API 进行搜索和替换
 * - 维护匹配结果和选择状态
 * - 提供分组/时间线双视图数据
 */

import { ref, computed, watch } from 'vue'
import transcriptionApi from '@/services/api/transcriptionApi'

// 搜索模式枚举
export const SearchMode = {
  LITERAL: 'literal',           // 文本精确匹配
  REGEX: 'regex',               // 正则表达式
  HOMOPHONE_STRICT: 'homophone_strict',  // 同音严格（仅完全同音）
  HOMOPHONE_FUZZY: 'homophone_fuzzy',    // 同音模糊（包含近音）
}

export function normalizeVisibleSearchMode(mode) {
  return mode === SearchMode.HOMOPHONE_STRICT
    ? SearchMode.HOMOPHONE_FUZZY
    : mode
}

// 排序模式枚举
export const SortMode = {
  GROUPED: 'grouped',   // 按读音簇分组
  TIMELINE: 'timeline', // 按时间线排序
}

// 索引状态枚举
export const IndexStatus = {
  UNKNOWN: 'unknown',
  READY: 'ready',
  BUILDING: 'building',
  FAILED: 'failed',
  MISSING: 'missing',
}

function buildSubtitleMutationSignature(subtitleList) {
  const source = Array.isArray(subtitleList) ? subtitleList : []
  return source
    .map((item) => {
      const sentenceIndex = item?.sentenceIndex ?? item?.legacyIndex ?? item?.id ?? item?.localId ?? ''
      const text = String(item?.text || '')
      return `${sentenceIndex}:${text}`
    })
    .join('\u0001')
}

// 6 色调色板（用于读音簇可视化）
const CLUSTER_PALETTE = [
  '#3b82f6', // blue
  '#ef4444', // red
  '#22c55e', // green
  '#a855f7', // purple
  '#f59e0b', // amber
  '#06b6d4', // cyan
]

/**
 * 根据 clusterId 生成稳定的颜色
 * @param {string} clusterId - 读音簇 ID
 * @returns {string} 十六进制颜色值
 */
function pickClusterColor(clusterId) {
  if (!clusterId) return CLUSTER_PALETTE[0]
  let hash = 0
  for (let i = 0; i < clusterId.length; i++) {
    hash = (hash * 31 + clusterId.charCodeAt(i)) >>> 0
  }
  return CLUSTER_PALETTE[hash % CLUSTER_PALETTE.length]
}

/**
 * 根据搜索词推断语言，保证同音接口必填语言字段可用。
 * 说明：后端当前不支持 auto，因此前端做轻量推断兜底。
 * @param {string} text
 * @returns {'zh' | 'ja' | 'en'}
 */
function detectLanguage(text) {
  const normalized = String(text || '')
  if (/[ぁ-んァ-ン]/.test(normalized)) return 'ja'
  if (/[\u4e00-\u9fff]/.test(normalized)) return 'zh'
  if (/[A-Za-z]/.test(normalized)) return 'en'
  return 'zh'
}

/**
 * 根据当前字幕内容推断主语言。
 * 设计说明：
 * - 近音模式下，用户可能输入拼音/罗马音（纯字母）。
 * - 若仅按查询词判断，会将中文拼音误判为英文，导致后端走错语言索引。
 * @param {Array} subtitleList
 * @returns {'zh' | 'ja' | 'en'}
 */
function detectLanguageFromSubtitles(subtitleList) {
  const source = Array.isArray(subtitleList) ? subtitleList : []
  const sampleText = source
    .slice(0, 200)
    .map((item) => String(item?.text || ''))
    .join('\n')

  if (!sampleText.trim()) return 'zh'

  const jaCount = (sampleText.match(/[ぁ-んァ-ン]/g) || []).length
  const zhCount = (sampleText.match(/[\u4e00-\u9fff]/g) || []).length
  const enCount = (sampleText.match(/[A-Za-z]/g) || []).length

  if (jaCount >= zhCount && jaCount >= enCount && jaCount > 0) return 'ja'
  if (zhCount >= enCount && zhCount > 0) return 'zh'
  if (enCount > 0) return 'en'
  return 'zh'
}

/**
 * 转义正则特殊字符。
 * @param {string} text
 * @returns {string}
 */
function escapeRegExp(text) {
  return String(text).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/**
 * 同音检索与批量替换 Composable
 * @param {Object} options - 配置选项
 * @param {Ref<string>} options.projectId - 当前项目主身份（project_id）
 * @param {Ref<Array>} options.subtitles - 字幕列表引用
 * @returns {Object} 同音搜索相关状态和方法
 */
export function useHomophoneSearch(options = {}) {
  // 兼容旧参数名 jobId，内部统一使用 projectId 语义。
  const projectId = options.projectId ?? options.jobId
  const { subtitles } = options

  // ========================
  // 核心状态
  // ========================

  // 搜索/替换模式
  const rawSearchMode = ref(SearchMode.LITERAL)
  const searchMode = computed({
    get: () => normalizeVisibleSearchMode(rawSearchMode.value),
    set: (value) => {
      rawSearchMode.value = normalizeVisibleSearchMode(value)
    },
  })
  const sortMode = ref(SortMode.TIMELINE)
  const isSearchActive = ref(false)      // 是否处于搜索状态
  const isReplaceMode = ref(false)       // 是否展开替换面板

  // 搜索输入
  const searchText = ref('')
  const searchReading = ref('')          // 仅近音模式可用
  const isIgnorePunctuation = ref(false)
  const replaceText = ref('')

  // 索引状态
  const indexStatus = ref(IndexStatus.UNKNOWN)
  const indexProgress = ref(0)
  const indexMessage = ref('')

  // 搜索状态
  const isSearching = ref(false)
  const searchError = ref(null)

  // 匹配结果
  // matchedSubtitleIds: 匹配到的字幕索引集合
  const matchedSubtitleIds = ref(new Set())
  // selectedSubtitleIds: 用户选中（勾选）的字幕索引集合
  const selectedSubtitleIds = ref(new Set())
  // matchSpansMap: subtitleIndex -> [{start, end, readingKey}]
  const matchSpansMap = ref(new Map())
  // matchedTextSnapshot: subtitleIndex -> text（搜索时的文本快照，用于变更时比对匹配区域是否仍有效）
  const matchedTextSnapshot = ref(new Map())
  // clusterMetaMap: subtitleIndex -> {clusterId, readingLabel, color}
  const clusterMetaMap = ref(new Map())
  // clustersInfo: 后端返回的完整 clusters 信息
  const clustersInfo = ref({})

  const subtitleBySentenceIndexMap = computed(() => {
    const source = subtitles?.value || []
    const indexMap = new Map()
    for (const item of source) {
      const sentenceIndex = item?.sentenceIndex
      if (sentenceIndex === undefined || sentenceIndex === null) continue
      indexMap.set(Number(sentenceIndex), item)
    }
    return indexMap
  })
  const subtitleMutationSignature = computed(() => {
    return buildSubtitleMutationSignature(subtitles?.value || [])
  })

  /**
   * 解析近音查询语言。
   * 规则：
   * 1) 查询词本身若包含中/日字符，优先按查询词判断；
   * 2) 查询词为拼音/罗马音等字母串时，按当前字幕主语言判断。
   * @param {string} queryInput
   * @returns {'zh' | 'ja' | 'en'}
   */
  function resolveHomophoneLanguage(queryInput) {
    const inferredFromQuery = detectLanguage(queryInput)
    if (/[ぁ-んァ-ン\u4e00-\u9fff]/.test(String(queryInput || ''))) {
      return inferredFromQuery
    }
    return detectLanguageFromSubtitles(subtitles?.value || [])
  }

  // ========================
  // 计算属性
  // ========================

  // 是否为近音搜索模式
  const isHomophoneMode = computed(() => {
    return searchMode.value === SearchMode.HOMOPHONE_FUZZY
  })

  // 是否可以执行搜索
  const canSearch = computed(() => {
    if (!searchText.value.trim()) return false
    if (isHomophoneMode.value && ![
      IndexStatus.READY,
      IndexStatus.MISSING,
      IndexStatus.UNKNOWN,
    ].includes(indexStatus.value)) {
      return false
    }
    return true
  })

  // 匹配数量
  const matchCount = computed(() => matchedSubtitleIds.value.size)

  // 选中数量
  const selectedCount = computed(() => selectedSubtitleIds.value.size)

  // 是否全选
  const isAllSelected = computed(() => {
    if (matchedSubtitleIds.value.size === 0) return false
    return selectedSubtitleIds.value.size === matchedSubtitleIds.value.size
  })

  // 是否部分选中（indeterminate 状态）
  const isIndeterminate = computed(() => {
    if (matchedSubtitleIds.value.size === 0) return false
    return selectedSubtitleIds.value.size > 0 &&
           selectedSubtitleIds.value.size < matchedSubtitleIds.value.size
  })

  // 分组视图数据
  const groupedViewItems = computed(() => {
    if (!isSearchActive.value || !subtitles?.value) return []

    // 按 clusterId 分组
    const groups = new Map()

    for (const [subtitleIndex, meta] of clusterMetaMap.value.entries()) {
      const clusterId = meta.clusterId || '_default'
      if (!groups.has(clusterId)) {
        groups.set(clusterId, {
          clusterId,
          readingLabel: meta.readingLabel || clusterId,
          color: meta.color,
          items: [],
        })
      }

      const subtitle = subtitleBySentenceIndexMap.value.get(Number(subtitleIndex))
      if (subtitle) {
        groups.get(clusterId).items.push({
          index: subtitleIndex,
          subtitle,
          matchSpans: matchSpansMap.value.get(subtitleIndex) || [],
          isSelected: selectedSubtitleIds.value.has(subtitleIndex),
          clusterColor: meta.color,
        })
      }
    }

    // 组内按时间排序
    for (const group of groups.values()) {
      group.items.sort((a, b) => a.subtitle.start - b.subtitle.start)
    }

    // 返回分组数组（按组内第一个条目时间排序）
    return Array.from(groups.values()).sort((a, b) => {
      const aFirst = a.items[0]?.subtitle?.start || 0
      const bFirst = b.items[0]?.subtitle?.start || 0
      return aFirst - bFirst
    })
  })

  // 时间线视图数据
  const timelineViewItems = computed(() => {
    if (!isSearchActive.value || !subtitles?.value) return []

    const items = []
    for (const subtitleIndex of matchedSubtitleIds.value) {
      const subtitle = subtitleBySentenceIndexMap.value.get(Number(subtitleIndex))
      if (subtitle) {
        const meta = clusterMetaMap.value.get(subtitleIndex) || {}
        items.push({
          index: subtitleIndex,
          subtitle,
          matchSpans: matchSpansMap.value.get(subtitleIndex) || [],
          isSelected: selectedSubtitleIds.value.has(subtitleIndex),
          clusterColor: meta.color || null,
        })
      }
    }

    // 按时间排序
    return items.sort((a, b) => a.subtitle.start - b.subtitle.start)
  })

  // ========================
  // 方法
  // ========================

  /**
   * 检查索引状态
   */
  async function checkIndexStatus() {
    if (!projectId?.value) return

    try {
      const result = await transcriptionApi.getHomophoneIndexStatus(projectId.value)
      const payload = result?.data || {}
      indexStatus.value = payload.status || IndexStatus.UNKNOWN
      indexProgress.value = 100
      indexMessage.value = payload.status === IndexStatus.READY ? '索引已就绪' : ''
    } catch (error) {
      console.error('检查索引状态失败:', error)
      indexStatus.value = IndexStatus.UNKNOWN
    }
  }

  /**
   * 执行搜索
   */
  async function executeSearch() {
    if (!canSearch.value || !projectId?.value) return

    isSearching.value = true
    searchError.value = null

    try {
      if (isHomophoneMode.value) {
        const queryInput = searchReading.value?.trim() || searchText.value.trim()
        const payload = {
          mode: searchMode.value,
          query_text: queryInput,
          language: resolveHomophoneLanguage(queryInput),
          is_ignore_punctuation: isIgnorePunctuation.value,
          limit: 2000,
        }
        const result = await transcriptionApi.homophoneFind(projectId.value, payload)
        parseHomophoneSearchResult(result?.data || {})
      } else {
        parseTextSearchResult({
          mode: searchMode.value,
          queryText: searchText.value,
        })
      }

      // 激活搜索状态
      isSearchActive.value = true

      // 默认全选所有匹配项
      selectAll()

    } catch (error) {
      console.error('搜索失败:', error)
      searchError.value = error.message || '搜索失败'
      clearSearchResult()
    } finally {
      isSearching.value = false
    }
  }

  /**
   * 解析近音搜索结果（后端结构）。
   * @param {Object} data - 后端返回的 data 字段
   */
  function parseHomophoneSearchResult(data) {
    const { matches = [], index_status: status = IndexStatus.UNKNOWN } = data

    matchedSubtitleIds.value.clear()
    matchSpansMap.value.clear()
    clusterMetaMap.value.clear()
    clustersInfo.value = {}
    indexStatus.value = status || IndexStatus.UNKNOWN

    for (const match of matches) {
      const idx = Number(match.sentence_index)
      const clusterId = String(match.cluster_id || '_default')
      const span = {
        start: Number(match.char_start),
        end: Number(match.char_end),
        readingKey: String(match.reading_label || ''),
      }

      if (Number.isNaN(idx)) continue

      matchedSubtitleIds.value.add(idx)
      const exists = matchSpansMap.value.get(idx) || []
      exists.push(span)
      matchSpansMap.value.set(idx, exists)

      clusterMetaMap.value.set(idx, {
        clusterId,
        readingLabel: String(match.reading_label || clusterId),
        color: pickClusterColor(clusterId),
      })
    }

    // 去重并排序 span
    for (const [idx, spans] of matchSpansMap.value.entries()) {
      const uniqueSpans = []
      const seen = new Set()
      for (const span of spans) {
        const key = `${span.start}:${span.end}`
        if (seen.has(key)) continue
        seen.add(key)
        uniqueSpans.push(span)
      }
      uniqueSpans.sort((a, b) => a.start - b.start)
      matchSpansMap.value.set(idx, uniqueSpans)
    }

    // 记录匹配字幕的文本快照，用于后续变更时判断匹配区域是否仍有效
    captureMatchedTextSnapshot()
  }

  /**
   * 解析文本/正则搜索结果（前端本地匹配）。
   * 说明：后端 find 当前仅支持近音模式，文本/正则在前端本地做筛选与高亮。
   * @param {Object} options
   * @param {'literal'|'regex'} options.mode
   * @param {string} options.queryText
   */
  function parseTextSearchResult({ mode, queryText }) {
    const keyword = String(queryText || '').trim()

    matchedSubtitleIds.value.clear()
    matchSpansMap.value.clear()
    clusterMetaMap.value.clear()
    clustersInfo.value = {}

    if (!keyword || !subtitles?.value?.length) return

    let regex
    if (mode === SearchMode.REGEX) {
      regex = new RegExp(keyword, 'g')
    } else {
      regex = new RegExp(escapeRegExp(keyword), 'g')
    }

    for (const subtitle of subtitles.value) {
      const sentenceIndex = Number(subtitle?.sentenceIndex)
      if (Number.isNaN(sentenceIndex)) continue
      const text = String(subtitle?.text || '')
      if (!text) continue

      const spans = []
      regex.lastIndex = 0
      let matched
      while ((matched = regex.exec(text)) !== null) {
        const raw = String(matched[0] || '')
        if (!raw.length) {
          regex.lastIndex += 1
          continue
        }
        spans.push({
          start: matched.index,
          end: matched.index + raw.length,
          readingKey: '',
        })
      }

      if (!spans.length) continue
      matchedSubtitleIds.value.add(sentenceIndex)
      matchSpansMap.value.set(sentenceIndex, spans)
      clusterMetaMap.value.set(sentenceIndex, {
        clusterId: '_text_match',
        readingLabel: mode === SearchMode.REGEX ? '正则匹配' : '文本匹配',
        color: pickClusterColor('_text_match'),
      })
    }

    captureMatchedTextSnapshot()
  }

  /**
   * 为所有当前匹配的字幕记录文本快照
   */
  function captureMatchedTextSnapshot() {
    matchedTextSnapshot.value.clear()
    for (const idx of matchedSubtitleIds.value) {
      const subtitle = subtitleBySentenceIndexMap.value.get(idx)
      if (subtitle) {
        matchedTextSnapshot.value.set(idx, String(subtitle.text || ''))
      }
    }
  }

  /**
   * 精细失效：保留未受影响的匹配，移除已删除/匹配区域被改动的条目
   * 返回 true 表示仍有存活匹配
   */
  function pruneStaleMatches() {
    const staleIds = []

    for (const idx of matchedSubtitleIds.value) {
      const subtitle = subtitleBySentenceIndexMap.value.get(idx)
      if (!subtitle) {
        // 字幕已被删除
        staleIds.push(idx)
        continue
      }

      const oldText = matchedTextSnapshot.value.get(idx)
      const newText = String(subtitle.text || '')
      if (oldText === newText) {
        // 文本未变，匹配仍有效
        continue
      }

      // 文本变了，逐 span 检查匹配区域的子串是否一致
      const spans = matchSpansMap.value.get(idx)
      if (!spans || spans.length === 0) {
        staleIds.push(idx)
        continue
      }

      const survivingSpans = spans.filter((span) => {
        if (span.end > newText.length) return false
        if (span.end > oldText.length) return false
        return oldText.substring(span.start, span.end) === newText.substring(span.start, span.end)
      })

      if (survivingSpans.length === 0) {
        staleIds.push(idx)
      } else if (survivingSpans.length < spans.length) {
        matchSpansMap.value.set(idx, survivingSpans)
        // 更新快照为当前文本
        matchedTextSnapshot.value.set(idx, newText)
      } else {
        // 所有 span 都存活，仅更新快照
        matchedTextSnapshot.value.set(idx, newText)
      }
    }

    // 清除失效条目
    for (const idx of staleIds) {
      matchedSubtitleIds.value.delete(idx)
      matchSpansMap.value.delete(idx)
      clusterMetaMap.value.delete(idx)
      selectedSubtitleIds.value.delete(idx)
      matchedTextSnapshot.value.delete(idx)
    }

    // 触发响应式更新
    if (staleIds.length > 0) {
      matchedSubtitleIds.value = new Set(matchedSubtitleIds.value)
      selectedSubtitleIds.value = new Set(selectedSubtitleIds.value)
      matchSpansMap.value = new Map(matchSpansMap.value)
      clusterMetaMap.value = new Map(clusterMetaMap.value)
      matchedTextSnapshot.value = new Map(matchedTextSnapshot.value)
    }

    return matchedSubtitleIds.value.size > 0
  }

  /**
   * 清除搜索结果
   */
  function clearSearchResult() {
    matchedSubtitleIds.value.clear()
    selectedSubtitleIds.value.clear()
    matchSpansMap.value.clear()
    matchedTextSnapshot.value.clear()
    clusterMetaMap.value.clear()
    clustersInfo.value = {}
    isSearchActive.value = false
  }

  /**
   * 重置搜索状态（清除输入和结果）
   */
  function resetSearch() {
    searchText.value = ''
    searchReading.value = ''
    replaceText.value = ''
    isReplaceMode.value = false
    clearSearchResult()
  }

  function invalidateSearchForSubtitleMutation() {
    if (!isHomophoneMode.value) {
      // 文本/正则模式：数据全在本地，直接重新匹配
      const keyword = String(searchText.value || '').trim()
      if (!keyword) {
        clearSearchResult()
        return
      }

      // 先清旧匹配再重算，保持选中集与匹配集一致
      matchedSubtitleIds.value.clear()
      matchSpansMap.value.clear()
      clusterMetaMap.value.clear()
      clustersInfo.value = {}

      parseTextSearchResult({
        mode: searchMode.value,
        queryText: searchText.value,
      })

      if (matchedSubtitleIds.value.size > 0) {
        // 仍有匹配：保留搜索态，裁剪选中集（移除已不匹配的项）
        const survivingSelected = new Set()
        for (const idx of selectedSubtitleIds.value) {
          if (matchedSubtitleIds.value.has(idx)) {
            survivingSelected.add(idx)
          }
        }
        selectedSubtitleIds.value = survivingSelected
        isSearchActive.value = true
      } else {
        // 无匹配项：退出搜索
        selectedSubtitleIds.value.clear()
        isSearchActive.value = false
      }
      return
    }

    // 近音模式：精细裁剪，只移除已删除/匹配区域被改动的条目
    const hasRemainingMatches = pruneStaleMatches()
    // 标记索引过期，完整匹配集可能已变（新增字幕可能也匹配）
    indexStatus.value = IndexStatus.MISSING
    indexMessage.value = '字幕已变更，近音索引待刷新'
    if (!hasRemainingMatches) {
      isSearchActive.value = false
    }
  }

  /**
   * 切换单条选择
   * @param {number} subtitleIndex - 字幕索引
   */
  function toggleSelect(subtitleIndex) {
    if (selectedSubtitleIds.value.has(subtitleIndex)) {
      selectedSubtitleIds.value.delete(subtitleIndex)
    } else {
      selectedSubtitleIds.value.add(subtitleIndex)
    }
    // 触发响应式更新
    selectedSubtitleIds.value = new Set(selectedSubtitleIds.value)
  }

  /**
   * 全选匹配项
   */
  function selectAll() {
    selectedSubtitleIds.value = new Set(matchedSubtitleIds.value)
  }

  /**
   * 取消全选
   */
  function deselectAll() {
    selectedSubtitleIds.value.clear()
    selectedSubtitleIds.value = new Set()
  }

  /**
   * 切换全选状态
   */
  function toggleSelectAll() {
    if (isAllSelected.value) {
      deselectAll()
    } else {
      selectAll()
    }
  }

  /**
   * 执行批量替换
   * @returns {Promise<{success: boolean, count: number, indices: number[]}>}
   */
  async function executeBatchReplace() {
    if (!projectId?.value || selectedSubtitleIds.value.size === 0) {
      return { success: false, count: 0, indices: [] }
    }

    if (!replaceText.value) {
      return { success: false, count: 0, indices: [] }
    }

    try {
      const queryInput = isHomophoneMode.value
        ? (searchReading.value?.trim() || searchText.value.trim())
        : searchText.value.trim()
      const result = await transcriptionApi.homophoneBatchReplace(projectId.value, {
        mode: searchMode.value,
        query_text: queryInput,
        replace_text: replaceText.value,
        language: isHomophoneMode.value
          ? resolveHomophoneLanguage(queryInput)
          : detectLanguage(queryInput),
        is_ignore_punctuation: isIgnorePunctuation.value,
        selected_sentence_indices: Array.from(selectedSubtitleIds.value),
      })

      // 替换成功后清除搜索状态
      if (result?.success) {
        clearSearchResult()
        replaceText.value = ''
        isReplaceMode.value = false
      }

      return {
        success: Boolean(result?.success),
        count: Number(result?.data?.updated_count || 0),
        indices: Array.isArray(result?.data?.updated_indices)
          ? result.data.updated_indices.map((item) => Number(item))
          : [],
      }
    } catch (error) {
      console.error('批量替换失败:', error)
      return { success: false, count: 0, indices: [] }
    }
  }

  // ========================
  // 副作用
  // ========================

  // 搜索模式切换时，如果切换到近音模式，检查索引状态
  watch(searchMode, (newMode) => {
    if (newMode === SearchMode.HOMOPHONE_FUZZY) {
      checkIndexStatus()
    }
  })

  // projectId 变化时重置状态
  watch(() => projectId?.value, () => {
    resetSearch()
    indexStatus.value = IndexStatus.UNKNOWN
  })

  watch(subtitleMutationSignature, (newSignature, oldSignature) => {
    if (oldSignature === undefined || newSignature === oldSignature) {
      return
    }
    if (!isHomophoneMode.value && !isSearchActive.value) {
      return
    }
    invalidateSearchForSubtitleMutation()
  })

  // ========================
  // 返回
  // ========================

  return {
    // 枚举
    SearchMode,
    SortMode,
    IndexStatus,

    // 状态
    searchMode,
    sortMode,
    isSearchActive,
    isReplaceMode,
    searchText,
    searchReading,
    isIgnorePunctuation,
    replaceText,
    indexStatus,
    indexProgress,
    indexMessage,
    isSearching,
    searchError,
    matchedSubtitleIds,
    selectedSubtitleIds,
    matchSpansMap,
    clusterMetaMap,
    clustersInfo,

    // 计算属性
    isHomophoneMode,
    canSearch,
    matchCount,
    selectedCount,
    isAllSelected,
    isIndeterminate,
    groupedViewItems,
    timelineViewItems,

    // 方法
    checkIndexStatus,
    executeSearch,
    clearSearchResult,
    resetSearch,
    invalidateSearchForSubtitleMutation,
    toggleSelect,
    selectAll,
    deselectAll,
    toggleSelectAll,
    executeBatchReplace,
    pickClusterColor,
  }
}
