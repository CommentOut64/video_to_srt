import { defineStore } from "pinia";
import { ref } from "vue";
import { buildColdSubtitleRecord, buildHotSubtitleRecord } from "./subtitleKernelUtils";

function compareOrder(left, right) {
  if (left.startMs !== right.startMs) {
    return left.startMs - right.startMs;
  }
  return left.localId.localeCompare(right.localId);
}

export const useSubtitleIndexStore = defineStore("subtitleIndex", () => {
  const orderedLocalIds = ref([]);
  const byLocalId = ref(new Map());
  const bySegmentId = ref(new Map());
  const bySentenceIndex = ref(new Map());

  function replaceAll(subtitles = []) {
    const orderEntries = [];
    const nextByLocalId = new Map();
    const nextBySegmentId = new Map();
    const nextBySentenceIndex = new Map();

    subtitles.forEach((subtitle) => {
      const hotRecord = buildHotSubtitleRecord(subtitle);
      const coldRecord = buildColdSubtitleRecord(subtitle);
      if (!hotRecord.localId) return;

      orderEntries.push({ localId: hotRecord.localId, startMs: hotRecord.startMs });
      nextByLocalId.set(hotRecord.localId, hotRecord.localId);
      if (coldRecord.segmentId) {
        nextBySegmentId.set(String(coldRecord.segmentId), hotRecord.localId);
      }
      if (Number.isFinite(Number(coldRecord.sentenceIndex))) {
        nextBySentenceIndex.set(Number(coldRecord.sentenceIndex), hotRecord.localId);
      }
    });

    orderEntries.sort(compareOrder);
    orderedLocalIds.value = orderEntries.map((entry) => entry.localId);
    byLocalId.value = nextByLocalId;
    bySegmentId.value = nextBySegmentId;
    bySentenceIndex.value = nextBySentenceIndex;
  }

  function clear() {
    orderedLocalIds.value = [];
    byLocalId.value = new Map();
    bySegmentId.value = new Map();
    bySentenceIndex.value = new Map();
  }

  return {
    orderedLocalIds,
    byLocalId,
    bySegmentId,
    bySentenceIndex,
    replaceAll,
    clear,
  };
});
