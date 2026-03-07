import { cloneCommandValue } from "./commandUtils";

// 命令工厂：切分命令用区间替换回放原字幕与左右字幕对，避免整表恢复。
export function createSplitSubtitleCommand({
  index,
  beforeSubtitle,
  afterSubtitles,
  beforeMetaIsDirty = false,
  afterMetaIsDirty = true,
  replaceSubtitleRange,
}) {
  const normalizedBeforeSubtitle = cloneCommandValue(beforeSubtitle);
  const normalizedAfterSubtitles = Array.isArray(afterSubtitles)
    ? afterSubtitles.map((subtitle) => cloneCommandValue(subtitle))
    : [];

  return {
    type: "split_subtitle",
    targetId: normalizedBeforeSubtitle?.id ?? null,
    splitIndex: index,
    beforeSubtitleSnapshot: normalizedBeforeSubtitle,
    afterSubtitleSnapshots: normalizedAfterSubtitles,
    beforeMetaIsDirty: Boolean(beforeMetaIsDirty),
    afterMetaIsDirty: Boolean(afterMetaIsDirty),
    undo() {
      replaceSubtitleRange(index, normalizedAfterSubtitles.length, [
        cloneCommandValue(normalizedBeforeSubtitle),
      ], Boolean(beforeMetaIsDirty));
    },
    redo() {
      replaceSubtitleRange(
        index,
        1,
        normalizedAfterSubtitles.map((subtitle) => cloneCommandValue(subtitle)),
        Boolean(afterMetaIsDirty)
      );
    },
  };
}