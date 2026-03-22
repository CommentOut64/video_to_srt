import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { usePlaybackStore } from '@/stores/playbackStore'
import { useProjectStore } from '@/stores/projectStore'
import { useEditorDocumentStore } from './editorDocumentStore'

function normalizeWord(word, offsetSec) {
  if (!word || typeof word !== 'object') {
    return null
  }

  const startMs = Number(word.startMs)
  const endMs = Number(word.endMs)
  const hasMsTiming = Number.isFinite(startMs) || Number.isFinite(endMs)
  const legacyStart = Number(word.start)
  const legacyEnd = Number(word.end)

  const start = hasMsTiming
    ? Math.max(0, (Number.isFinite(startMs) ? startMs : 0) / 1000 + offsetSec)
    : (Number.isFinite(legacyStart) ? Math.max(0, legacyStart + offsetSec) : undefined)
  const end = hasMsTiming
    ? Math.max(start ?? 0, (Number.isFinite(endMs) ? endMs : startMs) / 1000 + offsetSec)
    : (Number.isFinite(legacyEnd) ? Math.max(start ?? 0, legacyEnd + offsetSec) : undefined)

  return {
    ...word,
    start,
    end,
    word: word.word ?? word.text ?? '',
  }
}

export const useEditorProjectionBridge = defineStore('editorProjectionBridge', () => {
  const docStore = useEditorDocumentStore()
  const playbackStore = usePlaybackStore()
  const projectStore = useProjectStore()

  const selectedSubtitleIdState = ref(null)

  const subtitles = computed(() => {
    // 显式依赖文档修订号，确保 startMs/endMs 等热字段原地更新后投影仍会刷新。
    const documentRevision = docStore.revision
    if (documentRevision) {
      documentRevision.value
    }
    const offsetSec = Number(projectStore.subtitleOffset) || 0

    return docStore.order
      .map((localId) => {
        const hot = docStore.getEntity(localId)
        if (!hot || hot.isDeleted) {
          return null
        }

        const cold = docStore.getCold(localId)
        const sentenceIndex = cold?.sentenceIndex ?? null
        const start = Math.max(0, hot.startMs / 1000 + offsetSec)
        const end = Math.max(start, hot.endMs / 1000 + offsetSec)

        return {
          id: localId,
          localId,
          segment_id: cold?.segmentId ?? null,
          segmentId: cold?.segmentId ?? null,
          sentenceIndex,
          sentence_index: sentenceIndex,
          legacy_index: sentenceIndex,
          text: hot.text ?? '',
          start,
          end,
          startMs: hot.startMs,
          endMs: hot.endMs,
          isDraft: Boolean(hot.isDraft),
          isModified: Boolean(hot.isModified),
          isFinalized: !hot.isDraft,
          isDeleted: Boolean(hot.isDeleted),
          isDirty: Boolean(hot.isModified),
          chunk_id: cold?.chunkId ?? null,
          words: Array.isArray(cold?.words)
            ? cold.words.map((word) => normalizeWord(word, offsetSec)).filter(Boolean)
            : [],
          confidence: cold?.confidence ?? null,
          display_confidence: cold?.displayConfidence ?? null,
          confidence_source: cold?.confidenceSource ?? null,
          warning_type: cold?.warningType || 'none',
          originalText: cold?.originalText ?? null,
          source: cold?.sourceType || 'editor_v2',
          speaker_id: cold?.speakerId ?? null,
        }
      })
      .filter(Boolean)
  })

  const subtitleById = computed(() => {
    return new Map(subtitles.value.map((subtitle, index) => [subtitle.id, { subtitle, index }]))
  })

  const selectedSubtitleId = computed(() => {
    const current = selectedSubtitleIdState.value
    if (!current) {
      return null
    }
    return subtitleById.value.has(current) ? current : null
  })

  const currentSubtitleId = computed(() => {
    // V3.2.5+dev.20260321.03: 使用高频 raw 通道，修复播放期间字幕预览不更新
    const currentTime = Number(playbackStore.currentTimeRaw) || 0
    const matched = subtitles.value.find((subtitle) => {
      return currentTime >= subtitle.start && currentTime < subtitle.end
    })
    return matched?.id ?? null
  })

  const currentSubtitle = computed(() => {
    const currentId = currentSubtitleId.value
    return currentId ? subtitleById.value.get(currentId)?.subtitle ?? null : null
  })

  const totalSubtitles = computed(() => subtitles.value.length)
  const draftCount = computed(() => subtitles.value.filter((subtitle) => subtitle.isDraft).length)
  const warningCount = computed(() => {
    return subtitles.value.filter(
      (subtitle) => subtitle.warning_type && subtitle.warning_type !== 'none'
    ).length
  })

  function setSelectedSubtitleId(localId) {
    selectedSubtitleIdState.value = localId || null
  }

  function findSubtitleById(localId) {
    return subtitleById.value.get(localId)?.subtitle ?? null
  }

  function findSubtitleIndexById(localId) {
    return subtitleById.value.get(localId)?.index ?? -1
  }

  function findSubtitleByTime(timeSec) {
    const currentTime = Number(timeSec)
    if (!Number.isFinite(currentTime)) {
      return null
    }
    return (
      subtitles.value.find((subtitle) => currentTime >= subtitle.start && currentTime < subtitle.end)
      || null
    )
  }

  return {
    subtitles,
    currentSubtitleId,
    currentSubtitle,
    selectedSubtitleId,
    warningCount,
    totalSubtitles,
    draftCount,
    setSelectedSubtitleId,
    findSubtitleById,
    findSubtitleIndexById,
    findSubtitleByTime,
  }
})
