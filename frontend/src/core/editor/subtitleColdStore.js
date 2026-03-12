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

  // V3.2.4+dev.20260311.01: 添加原子插入/删除方法
  function insert(localId, coldData) {
    const newRecords = new Map(records.value);
    newRecords.set(localId, coldData);
    records.value = newRecords;
  }

  function remove(localId) {
    const newRecords = new Map(records.value);
    newRecords.delete(localId);
    records.value = newRecords;
  }

  function get(localId) {
    return records.value.get(localId) || null;
  }

  return {
    records,
    replaceAll,
    clear,
    insert,
    remove,
    get,
  };
});
