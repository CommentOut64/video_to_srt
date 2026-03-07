import { defineStore } from "pinia";
import { ref } from "vue";
import { buildHotSubtitleRecord } from "./subtitleKernelUtils";

export const useSubtitleHotStore = defineStore("subtitleHot", () => {
  const records = ref(new Map());

  function replaceAll(subtitles = []) {
    const nextRecords = new Map();
    subtitles.forEach((subtitle) => {
      const record = buildHotSubtitleRecord(subtitle);
      if (record.localId) {
        nextRecords.set(record.localId, record);
      }
    });
    records.value = nextRecords;
  }

  function clear() {
    records.value = new Map();
  }

  return {
    records,
    replaceAll,
    clear,
  };
});
