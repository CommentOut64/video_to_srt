import { cloneCommandValue } from "./commandUtils";

// 命令工厂：用最小前后状态描述单条字幕更新，避免整表快照进入热路径。
export function createUpdateSubtitleCommand({
  targetId,
  beforeSubtitle,
  afterSubtitle,
  beforeMetaIsDirty = false,
  afterMetaIsDirty = true,
  applySubtitleSnapshot,
  setMetaDirty,
  mergeKey = null,
}) {
  const normalizedBeforeSubtitle = cloneCommandValue(beforeSubtitle);
  const normalizedAfterSubtitle = cloneCommandValue(afterSubtitle);
  const normalizedTargetId = targetId
    ?? normalizedAfterSubtitle?.id
    ?? normalizedBeforeSubtitle?.id
    ?? null;

  return {
    type: "update_subtitle",
    targetId: normalizedTargetId,
    mergeKey,
    beforeSubtitleSnapshot: normalizedBeforeSubtitle,
    afterSubtitleSnapshot: normalizedAfterSubtitle,
    beforeMetaIsDirty: Boolean(beforeMetaIsDirty),
    afterMetaIsDirty: Boolean(afterMetaIsDirty),
    undo() {
      applySubtitleSnapshot(
        normalizedTargetId,
        cloneCommandValue(normalizedBeforeSubtitle)
      );
      setMetaDirty(Boolean(beforeMetaIsDirty));
    },
    redo() {
      applySubtitleSnapshot(
        normalizedTargetId,
        cloneCommandValue(normalizedAfterSubtitle)
      );
      setMetaDirty(Boolean(afterMetaIsDirty));
    },
    canMerge(other) {
      return Boolean(
        mergeKey
        && other?.mergeKey
        && mergeKey === other.mergeKey
        && other?.type === "update_subtitle"
        && other?.targetId === normalizedTargetId
      );
    },
    merge(other) {
      return createUpdateSubtitleCommand({
        targetId: normalizedTargetId,
        beforeSubtitle: normalizedBeforeSubtitle,
        afterSubtitle: other?.afterSubtitleSnapshot ?? normalizedAfterSubtitle,
        beforeMetaIsDirty,
        afterMetaIsDirty: other?.afterMetaIsDirty ?? afterMetaIsDirty,
        applySubtitleSnapshot,
        setMetaDirty,
        mergeKey,
      });
    },
  };
}