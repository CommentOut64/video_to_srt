// V3.2.5+dev.20260314.01: 字幕实体数据模型

/**
 * 字幕热数据 - 编辑热路径字段
 * 存储在 editorDocumentStore.entities (shallowReactive Map)
 */
export function createSubtitleHotEntity(localId, text, startMs, endMs) {
  return {
    localId,
    text,
    startMs,
    endMs,
    isDraft: false,
    isModified: false,
    isDeleted: false,
    revision: 0,
  }
}

/**
 * 字幕冷数据 - 低频大体积字段
 * 存储在 editorDocumentStore.coldEntities (普通 Map，markRaw)
 */
export function createSubtitleColdEntity(localId) {
  return {
    localId,
    segmentId: null, // 唯一身份主键
    sentenceIndex: null, // 仅保留 legacy 兼容/诊断，不再作为运行时绑定索引
    chunkId: null, // 批替换锚点
    words: null,
    confidence: null,
    displayConfidence: null,
    confidenceSource: null,
    speakerId: null,
    sourceType: null,
    warningType: 'none',
    originalText: null,
  }
}
