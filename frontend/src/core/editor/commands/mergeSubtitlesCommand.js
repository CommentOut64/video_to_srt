import { cloneCommandValue } from "./commandUtils";

// 命令工厂：合并命令只记录相邻两条字幕与合并结果，撤销时恢复原区间。
export function createMergeSubtitlesCommand({
  index,
  beforeSubtitles,
  afterSubtitle,
  beforeMetaIsDirty = false,
  afterMetaIsDirty = true,
  replaceSubtitleRange,
}) {
  const normalizedBeforeSubtitles = Array.isArray(beforeSubtitles)
    ? beforeSubtitles.map((subtitle) => cloneCommandValue(subtitle))
    : [];
  const normalizedAfterSubtitle = cloneCommandValue(afterSubtitle);

  return {
    type: "merge_subtitles",
    targetId: normalizedAfterSubtitle?.id ?? normalizedBeforeSubtitles[0]?.id ?? null,
    mergeIndex: index,
    beforeSubtitleSnapshots: normalizedBeforeSubtitles,
    afterSubtitleSnapshot: normalizedAfterSubtitle,
    beforeMetaIsDirty: Boolean(beforeMetaIsDirty),
    afterMetaIsDirty: Boolean(afterMetaIsDirty),
    undo() {
      replaceSubtitleRange(
        index,
        1,
        normalizedBeforeSubtitles.map((subtitle) => cloneCommandValue(subtitle)),
        Boolean(beforeMetaIsDirty)
      );
    },
    redo() {
      replaceSubtitleRange(index, normalizedBeforeSubtitles.length, [
        cloneCommandValue(normalizedAfterSubtitle),
      ], Boolean(afterMetaIsDirty));
    },
  };
}