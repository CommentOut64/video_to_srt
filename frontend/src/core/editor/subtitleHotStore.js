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

  function update(localId, patch) {
    const record = records.value.get(localId);
    if (!record) return null;

    const nextRevision = (record.revision || 0) + 1;
    const updated = {
      ...record,
      ...patch,
      revision: nextRevision,
      lastModified: Date.now(),
    };

    const newRecords = new Map(records.value);
    newRecords.set(localId, updated);
    records.value = newRecords;

    return updated;
  }

  function tryApplyServerPatch(localId, serverPatch, serverRevision) {
    const record = records.value.get(localId);
    if (!record) return false;

    if (record.revision > serverRevision) {
      console.warn(`[HotStore] 跳过过期回填: localId=${localId}, client=${record.revision} > server=${serverRevision}`);
      return false;
    }

    const newRecords = new Map(records.value);
    newRecords.set(localId, {
      ...record,
      ...serverPatch,
      serverRevision,
      lastSynced: Date.now(),
    });
    records.value = newRecords;

    return true;
  }

  function clear() {
    records.value = new Map();
  }

  // V3.2.4+dev.20260311.01: 添加原子插入/删除方法，支持命令对象直接操作内核
  function insert(localId, hotData) {
    const normalizedRevision = Number.isFinite(Number(hotData?.revision))
      ? Number(hotData.revision)
      : 0;
    const newRecords = new Map(records.value);
    newRecords.set(localId, {
      ...hotData,
      revision: normalizedRevision,
      lastModified: Date.now(),
    });
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
    update,
    tryApplyServerPatch,
    clear,
    insert,
    remove,
    get,
  };
});
