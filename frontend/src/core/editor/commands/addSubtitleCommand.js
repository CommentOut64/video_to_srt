import { cloneCommandValue } from "./commandUtils";

// 命令工厂：新增命令只保留插入点和新增字幕本身，撤销时直接反向删除。
export function createAddSubtitleCommand({
  insertIndex,
  subtitle,
  beforeMetaIsDirty = false,
  afterMetaIsDirty = true,
  insertSubtitleSnapshotAt,
  removeSubtitleById,
  setMetaDirty,
}) {
  const normalizedSubtitle = cloneCommandValue(subtitle);
  const targetId = normalizedSubtitle?.id ?? null;

  return {
    type: "add_subtitle",
    targetId,
    insertIndex,
    subtitleSnapshot: normalizedSubtitle,
    beforeMetaIsDirty: Boolean(beforeMetaIsDirty),
    afterMetaIsDirty: Boolean(afterMetaIsDirty),
    undo() {
      removeSubtitleById(targetId);
      setMetaDirty(Boolean(beforeMetaIsDirty));
    },
    redo() {
      insertSubtitleSnapshotAt(insertIndex, cloneCommandValue(normalizedSubtitle));
      setMetaDirty(Boolean(afterMetaIsDirty));
    },
  };
}