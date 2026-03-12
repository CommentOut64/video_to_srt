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

  // V3.2.4+dev.20260311.03: 重新设计 insert，接收 hotRecords 用于排序
  function insert(localId, startMs, segmentId, sentenceIndex, hotRecords) {
    const newByLocalId = new Map(byLocalId.value);
    const newBySegmentId = new Map(bySegmentId.value);
    const newBySentenceIndex = new Map(bySentenceIndex.value);

    newByLocalId.set(localId, localId);
    if (segmentId) {
      newBySegmentId.set(String(segmentId), localId);
    }
    if (Number.isFinite(Number(sentenceIndex))) {
      newBySentenceIndex.set(Number(sentenceIndex), localId);
    }

    const orderEntries = orderedLocalIds.value.map((id) => {
      const hot = hotRecords?.get(id);
      return { localId: id, startMs: hot?.startMs || 0 };
    });
    orderEntries.push({ localId, startMs: Number(startMs) || 0 });
    orderEntries.sort(compareOrder);

    orderedLocalIds.value = orderEntries.map((e) => e.localId);
    byLocalId.value = newByLocalId;
    bySegmentId.value = newBySegmentId;
    bySentenceIndex.value = newBySentenceIndex;
  }

  function remove(localId) {
    const newByLocalId = new Map(byLocalId.value);
    const newBySegmentId = new Map(bySegmentId.value);
    const newBySentenceIndex = new Map(bySentenceIndex.value);

    newByLocalId.delete(localId);
    for (const [key, value] of newBySegmentId.entries()) {
      if (value === localId) newBySegmentId.delete(key);
    }
    for (const [key, value] of newBySentenceIndex.entries()) {
      if (value === localId) newBySentenceIndex.delete(key);
    }

    orderedLocalIds.value = orderedLocalIds.value.filter((id) => id !== localId);
    byLocalId.value = newByLocalId;
    bySegmentId.value = newBySegmentId;
    bySentenceIndex.value = newBySentenceIndex;
  }

  return {
    orderedLocalIds,
    byLocalId,
    bySegmentId,
    bySentenceIndex,
    replaceAll,
    clear,
    insert,
    remove,
  };
});
