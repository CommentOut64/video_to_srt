import { defineStore } from "pinia";
import { ref } from "vue";
import { buildColdSubtitleRecord } from "./subtitleKernelUtils";

export const useSubtitleColdStore = defineStore("subtitleCold", () => {
  const records = ref(new Map());

  function replaceAll(subtitles = []) {
    const nextRecords = new Map();
    subtitles.forEach((subtitle) => {
      const record = buildColdSubtitleRecord(subtitle);
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
