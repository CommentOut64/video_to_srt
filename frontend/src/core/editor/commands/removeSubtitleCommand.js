import { cloneCommandValue, cloneStringArray } from "./commandUtils";

// 命令工厂：删除命令显式维护删除标记与受影响 chunk 映射，避免回滚时依赖全局快照。
export function createRemoveSubtitleCommand({
  index,
  subtitle,
  beforeDeleted = false,
  afterDeleted = true,
  beforeChunkIds = null,
  afterChunkIds = null,
  beforeMetaIsDirty = false,
  afterMetaIsDirty = true,
  insertSubtitleSnapshotAt,
  removeSubtitleById,
  setSentenceDeleted,
  setChunkSubtitleIds,
  setMetaDirty,
}) {
  const normalizedSubtitle = cloneCommandValue(subtitle);
  const targetId = normalizedSubtitle?.id ?? null;
  const sentenceIndex = normalizedSubtitle?.sentenceIndex;
  const chunkId = normalizedSubtitle?.chunk_id ?? null;
  const normalizedBeforeChunkIds = beforeChunkIds === null ? null : cloneStringArray(beforeChunkIds);
  const normalizedAfterChunkIds = afterChunkIds === null ? null : cloneStringArray(afterChunkIds);

  return {
    type: "remove_subtitle",
    targetId,
    removeIndex: index,
    subtitleSnapshot: normalizedSubtitle,
    beforeDeleted: Boolean(beforeDeleted),
    afterDeleted: Boolean(afterDeleted),
    beforeChunkIds: normalizedBeforeChunkIds,
    afterChunkIds: normalizedAfterChunkIds,
    beforeMetaIsDirty: Boolean(beforeMetaIsDirty),
    afterMetaIsDirty: Boolean(afterMetaIsDirty),
    undo() {
      insertSubtitleSnapshotAt(index, cloneCommandValue(normalizedSubtitle));
      if (sentenceIndex !== undefined && sentenceIndex !== null) {
        setSentenceDeleted(sentenceIndex, Boolean(beforeDeleted));
      }
      if (chunkId) {
        setChunkSubtitleIds(chunkId, cloneStringArray(normalizedBeforeChunkIds));
      }
      setMetaDirty(Boolean(beforeMetaIsDirty));
    },
    redo() {
      removeSubtitleById(targetId);
      if (sentenceIndex !== undefined && sentenceIndex !== null) {
        setSentenceDeleted(sentenceIndex, Boolean(afterDeleted));
      }
      if (chunkId) {
        setChunkSubtitleIds(chunkId, cloneStringArray(normalizedAfterChunkIds));
      }
      setMetaDirty(Boolean(afterMetaIsDirty));
    },
  };
}