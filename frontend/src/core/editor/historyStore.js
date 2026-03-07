import { defineStore } from "pinia";
import { computed, ref } from "vue";

const MAX_HISTORY_SIZE = 50;

export const useEditorHistoryStore = defineStore("editorHistory", () => {
  const undoStack = ref([]);
  const redoStack = ref([]);
  const pauseDepth = ref(0);
  const isApplying = ref(false);

  const history = computed(() => undoStack.value);
  const canUndo = computed(() => undoStack.value.length > 0);
  const canRedo = computed(() => redoStack.value.length > 0);
  const isHistoryTracking = computed(() => pauseDepth.value === 0 && !isApplying.value);

  function clearHistory() {
    undoStack.value = [];
    redoStack.value = [];
  }

  function pauseHistory() {
    pauseDepth.value += 1;
  }

  function resumeHistory() {
    pauseDepth.value = Math.max(0, pauseDepth.value - 1);
  }

  function recordCommand(command) {
    if (!command || pauseDepth.value > 0 || isApplying.value) {
      return false;
    }

    const previous = undoStack.value[undoStack.value.length - 1];
    if (previous && typeof previous.canMerge === "function" && previous.canMerge(command)) {
      const merged = previous.merge(command);
      undoStack.value.splice(undoStack.value.length - 1, 1, merged);
      redoStack.value = [];
      return true;
    }

    undoStack.value = [...undoStack.value, command].slice(-MAX_HISTORY_SIZE);
    redoStack.value = [];
    return true;
  }

  function withApplying(callback) {
    isApplying.value = true;
    try {
      return callback();
    } finally {
      isApplying.value = false;
    }
  }

  function undo() {
    if (!canUndo.value) return false;
    const command = undoStack.value[undoStack.value.length - 1];
    undoStack.value = undoStack.value.slice(0, -1);
    withApplying(() => {
      command.undo?.();
    });
    redoStack.value = [...redoStack.value, command].slice(-MAX_HISTORY_SIZE);
    return true;
  }

  function redo() {
    if (!canRedo.value) return false;
    const command = redoStack.value[redoStack.value.length - 1];
    redoStack.value = redoStack.value.slice(0, -1);
    withApplying(() => {
      command.redo?.();
    });
    undoStack.value = [...undoStack.value, command].slice(-MAX_HISTORY_SIZE);
    return true;
  }

  return {
    history,
    canUndo,
    canRedo,
    isHistoryTracking,
    clearHistory,
    pauseHistory,
    resumeHistory,
    recordCommand,
    undo,
    redo,
  };
});
