/**
 * ProjectStore - 项目数据管理
 *
 * 负责管理字幕编辑器的核心数据，包括字幕数据、播放器状态、视图配置等
 * 实现了撤销/重做、自动保存、智能问题检测等功能
 */
import { defineStore } from "pinia";
import { ref, computed, watch, toRaw } from "vue";
import { useRefHistory } from "@vueuse/core";
import localforage from "localforage";
import smartSaver from "@/services/SmartSaver";
import { repairSubtitleOverlaps } from "@/utils/subtitleUtils";

export const useProjectStore = defineStore("project", () => {
  // ========== 1. 项目元数据 ==========
  const meta = ref({
    jobId: null, // 转录任务ID
    projectId: null, // 项目ID（Task6 主键）
    mode: "normal", // normal | legacy
    taskMode: "transcribe", // transcribe | subtitle_edit
    flavor: "full", // full | lite
    videoPath: null, // 视频文件路径
    audioPath: null, // 音频文件路径
    peaksPath: null, // 波形峰值数据路径
    duration: 0, // 视频总时长（秒）
    filename: "", // 源文件名
    title: "", // 用户自定义任务名称
    videoFormat: null, // 视频格式
    hasProxyVideo: false, // 是否有 Proxy 视频
    lastSaved: Date.now(), // 最后保存时间
    isDirty: false, // 是否有未保存修改
    // 渐进式加载相关（状态由 useProxyVideo composable 管理）
    currentResolution: null, // 当前视频分辨率 ('360p', '720p', 'source')
    // Phase 0.5: 能力协商快照留位（本期不参与行为判断）
    capabilitySnapshot: null,
    // V3.2.0+dev.20260130.09: 字幕全局时间偏移（秒）
    subtitleOffset: 0,
  });

  // ========== 2. 字幕数据（Single Source of Truth） ==========
  const subtitles = ref([]);

  // V3.2.0+dev.20260130.09: 字幕全局时间偏移（秒，正值延后，负值提前）
  const subtitleOffset = ref(0);

  // 用户删除的字幕索引集合（用于阻止 SSE 回补）
  const deletedSentenceIndices = ref(new Set());

  // ========== 2.1 双模态架构: Chunk 索引映射 ==========
  // chunk_id -> [subtitle_id_1, subtitle_id_2, ...]
  const chunkSubtitleMap = ref(new Map());

  // 双流进度状态
  const dualStreamProgress = ref({
    fastStream: 0, // 快流(SenseVoice)进度 0-100
    slowStream: 0, // 慢流(Whisper)进度 0-100
    totalChunks: 0, // 总 Chunk 数
    draftChunks: 0, // 草稿 Chunk 数
    finalizedChunks: 0, // 定稿 Chunk 数
  });

  // 说话人 profile 缓存（后端真源的前端镜像）
  const speakerProfiles = ref(new Map());

  // ========== 3. Undo/Redo 历史记录 ==========
  // 【重要】撤销/重做策略：
  // - 历史记录只追踪用户编辑操作
  // - 转录过程中 SSE 推送的字幕变更不应被撤销
  // - 使用 pause/resume 控制历史记录：
  //   - SSE 推送字幕时暂停记录（appendOrUpdateDraft, replaceChunk）
  //   - 用户编辑时正常记录（updateSubtitle, addSubtitle, removeSubtitle）
  // - 以下情况会清除历史记录以建立"基线"：
  //   1. importSRT() - 导入转录结果时
  //   2. restoreProject() - 从缓存/存储恢复项目时
  //   3. resetProject() - 重置项目时
  const {
    history,
    undo,
    redo,
    canUndo,
    canRedo,
    clear: clearHistory,
    pause: pauseHistory,
    resume: resumeHistory,
    isTracking: isHistoryTracking,
  } = useRefHistory(subtitles, {
    deep: true,
    capacity: 50, // 限制历史记录步数
    clone: true, // 深拷贝，确保历史记录独立
    flush: 'sync', // 同步记录，避免一次操作产生多个历史记录
  });

  // ========== 4. 播放器全局状态 ==========
  const player = ref({
    currentTime: 0, // 当前播放时间（秒）
    isPlaying: false, // 是否正在播放
    playbackRate: 1.0, // 播放速度（0.5-4.0）
    volume: 1.0, // 音量（0.0-1.0）
    isSeeking: false, // 全局Seek锁：标记用户是否正在主动跳转（解决进度条拖动循环问题）
  });

  // ========== 5. 视图状态 ==========
  const view = ref({
    theme: "dark", // 'dark' | 'light'
    zoomLevel: 100, // 波形缩放比例（%）
    autoScroll: true, // 列表自动跟随播放
    selectedSubtitleId: null, // 当前选中的字幕ID
  });

  // ========== 6. 计算属性 ==========
  const primaryId = computed(() => meta.value.projectId || meta.value.jobId || null);
  const totalSubtitles = computed(() => subtitles.value.length);

  const currentSubtitle = computed(() => {
    return subtitles.value.find(
      (s) =>
        player.value.currentTime >= s.start && player.value.currentTime < s.end
    );
  });

  // ========== 6.1 字幕偏移辅助函数 ==========
  function clampSubtitleOffset(value) {
    const num = Number(value) || 0;
    return Math.min(10, Math.max(-10, num));
  }

  function toDisplayTime(baseTime) {
    return (Number(baseTime) || 0) + subtitleOffset.value;
  }

  function toBaseTime(displayTime) {
    return (Number(displayTime) || 0) - subtitleOffset.value;
  }

  function shiftWords(words, delta) {
    if (!Array.isArray(words) || words.length === 0) return [];
    return words.map((word) => {
      const start = word?.start;
      const end = word?.end;
      const nextStart = typeof start === 'number' ? Math.max(0, start + delta) : start;
      const nextEnd = typeof end === 'number' ? Math.max(nextStart ?? 0, end + delta) : end;
      return {
        ...word,
        start: nextStart,
        end: nextEnd,
      };
    });
  }

  function applyOffsetDelta(delta) {
    if (!delta) return;
    subtitles.value = subtitles.value.map((subtitle) => {
      const start = (Number(subtitle.start) || 0) + delta;
      const end = (Number(subtitle.end) || 0) + delta;
      const adjustedStart = Math.max(0, start);
      const adjustedEnd = Math.max(adjustedStart, end);
      return {
        ...subtitle,
        start: adjustedStart,
        end: adjustedEnd,
        words: shiftWords(subtitle.words, delta),
      };
    });
  }

  function applyOffsetToSentenceData(sentenceData) {
    if (!sentenceData) return sentenceData;
    const delta = subtitleOffset.value;
    const start = toDisplayTime(sentenceData.start ?? 0);
    const end = toDisplayTime(sentenceData.end ?? 0);
    return {
      ...sentenceData,
      start: Math.max(0, start),
      end: Math.max(Math.max(0, start), end),
      words: shiftWords(sentenceData.words, delta),
    };
  }

  function applyOffsetToSegments(segments) {
    if (!Array.isArray(segments)) return [];
    return segments.map((segment) => {
      const start = toDisplayTime(segment.start ?? 0);
      const end = toDisplayTime(segment.end ?? 0);
      return {
        ...segment,
        start: Math.max(0, start),
        end: Math.max(Math.max(0, start), end),
      };
    });
  }

  function normalizeSpeakerFields(source = {}, options = {}) {
    const { stripSpeaker = false } = options;
    if (stripSpeaker) {
      return {
        speaker_id: null,
        turn_id: null,
        speaker_label: null,
        speaker_color_key: null,
        binding_source: null,
      };
    }
    const speakerId = source.speaker_id ?? source.speakerId ?? null;
    return {
      speaker_id: speakerId,
      turn_id: source.turn_id ?? source.turnId ?? null,
      speaker_label: source.speaker_label ?? source.speakerLabel ?? speakerId,
      speaker_color_key: source.speaker_color_key ?? source.speakerColorKey ?? null,
      binding_source: source.binding_source ?? source.bindingSource ?? null,
    };
  }

  function setSubtitleOffset(value, options = {}) {
    const { applyDelta = true } = options;
    const normalized = Math.round(clampSubtitleOffset(value) * 1000) / 1000;
    const previous = subtitleOffset.value;
    subtitleOffset.value = normalized;
    meta.value.subtitleOffset = normalized;
    if (applyDelta) {
      applyOffsetDelta(normalized - previous);
    }
  }

  function normalizeCapabilitySnapshot(snapshot) {
    if (!snapshot) {
      return null;
    }
    if (Array.isArray(snapshot)) {
      // 后端旧格式 capability_snapshot(List[str]) 不参与能力合并，统一回退运行时基线
      console.debug("[normalizeCapabilitySnapshot] 忽略数组快照格式，回退默认能力");
      return null;
    }
    if (typeof snapshot !== "object") {
      console.warn("[normalizeCapabilitySnapshot] 非对象快照，已忽略:", typeof snapshot);
      return null;
    }
    if (
      snapshot.capabilities
      && typeof snapshot.capabilities === "object"
      && !Array.isArray(snapshot.capabilities)
    ) {
      return snapshot;
    }
    console.warn("[normalizeCapabilitySnapshot] 非标准快照结构，已忽略");
    return null;
  }

  // ========== 6.2 Phase 1: 状态写入口收口 ==========
  function patchMeta(patch = {}) {
    if (!patch || typeof patch !== "object") {
      return;
    }
    meta.value = {
      ...meta.value,
      ...patch,
    };
  }

  function setIdentity(payload = {}) {
    const nextPatch = {};
    if (Object.prototype.hasOwnProperty.call(payload, "projectId")) {
      nextPatch.projectId = payload.projectId || null;
    }
    if (Object.prototype.hasOwnProperty.call(payload, "jobId")) {
      nextPatch.jobId = payload.jobId || null;
    }
    if (Object.prototype.hasOwnProperty.call(payload, "mode")) {
      nextPatch.mode = payload.mode || "normal";
    }
    if (Object.prototype.hasOwnProperty.call(payload, "taskMode")) {
      nextPatch.taskMode = payload.taskMode || "transcribe";
    }
    if (Object.prototype.hasOwnProperty.call(payload, "flavor")) {
      nextPatch.flavor = payload.flavor || "full";
    }
    if (Object.prototype.hasOwnProperty.call(payload, "capabilitySnapshot")) {
      nextPatch.capabilitySnapshot = normalizeCapabilitySnapshot(payload.capabilitySnapshot);
    }
    patchMeta(nextPatch);
  }

  function setMediaPaths(payload = {}) {
    const nextPatch = {};
    if (Object.prototype.hasOwnProperty.call(payload, "videoPath")) {
      nextPatch.videoPath = payload.videoPath || null;
    }
    if (Object.prototype.hasOwnProperty.call(payload, "audioPath")) {
      nextPatch.audioPath = payload.audioPath || null;
    }
    if (Object.prototype.hasOwnProperty.call(payload, "peaksPath")) {
      nextPatch.peaksPath = payload.peaksPath || null;
    }
    patchMeta(nextPatch);
  }

  function setProjectTitle(title) {
    patchMeta({ title: title || "" });
  }

  function setProjectDuration(duration) {
    const normalized = Number(duration);
    patchMeta({ duration: Number.isFinite(normalized) && normalized > 0 ? normalized : 0 });
  }

  function setCurrentResolution(resolution) {
    patchMeta({ currentResolution: resolution || null });
  }

  function setZoomLevel(level) {
    const normalized = Number(level);
    view.value.zoomLevel = Number.isFinite(normalized) ? normalized : view.value.zoomLevel;
  }

  function setSelectedSubtitleId(subtitleId) {
    view.value.selectedSubtitleId = subtitleId || null;
  }

  function setPlayerVolume(volume) {
    const normalized = Number(volume);
    if (!Number.isFinite(normalized)) {
      return;
    }
    player.value.volume = Math.max(0, Math.min(1, normalized));
  }

  function setPlaybackRate(rate) {
    const normalized = Number(rate);
    if (!Number.isFinite(normalized)) {
      return;
    }
    player.value.playbackRate = Math.max(0.25, Math.min(4, normalized));
  }

  function insertSubtitleAt(index, subtitle) {
    const insertIndex = Math.max(0, Math.min(index, subtitles.value.length));
    subtitles.value.splice(insertIndex, 0, subtitle);
    return insertIndex;
  }

  function removeSubtitleAt(index) {
    if (index < 0 || index >= subtitles.value.length) {
      return null;
    }
    const removed = subtitles.value.splice(index, 1);
    return removed[0] || null;
  }

  const isDirty = computed(() => {
    return meta.value.isDirty || subtitles.value.some((s) => s.isDirty);
  });

  // 旧的字幕检查系统已移除，保留空数组以兼容现有引用
  const validationErrors = computed(() => []);

  // ========== 7. 智能保存系统 ==========
  // 内存缓存（热数据）
  const memoryCache = new Map();
  const MAX_MEMORY_CACHE = 10; // 最多缓存10个任务的数据

  // 标记是否正在进行保存后的状态更新（避免循环触发）
  let isUpdatingAfterSave = false;

  // 配置智能保存回调
  smartSaver.onSaveSuccess = (jobId) => {
    console.log("[ProjectStore] 自动保存成功:", jobId);
    // 使用标记避免循环触发 watch
    isUpdatingAfterSave = true;

    // 暂停历史记录追踪，避免 isDirty 重置创建历史记录
    pauseHistory();

    meta.value.lastSaved = Date.now();
    meta.value.isDirty = false;
    // 重置每个字幕的 isDirty 标记
    subtitles.value.forEach((s) => (s.isDirty = false));

    // 恢复历史记录追踪
    resumeHistory();

    isUpdatingAfterSave = false;
  };

  smartSaver.onSaveError = (error, jobId) => {
    console.error("[ProjectStore] 自动保存失败:", jobId, error);
  };

  // 监听数据变化，触发智能保存
  watch(
    [subtitles, meta],
    () => {
      // 跳过保存后的状态更新触发
      if (isUpdatingAfterSave) return;
      const cacheKey = meta.value.projectId || meta.value.jobId;
      if (!cacheKey) return;

      // 更新内存缓存
      memoryCache.set(cacheKey, {
        subtitles: toRaw(subtitles.value),
        meta: toRaw(meta.value),
      });

      // 限制内存缓存大小（LRU淘汰）
      if (memoryCache.size > MAX_MEMORY_CACHE) {
        const firstKey = memoryCache.keys().next().value;
        memoryCache.delete(firstKey);
      }

      // 触发智能保存
      smartSaver.save({
        jobId: cacheKey,
        subtitles: subtitles.value,
        meta: meta.value,
      });
    },
    { deep: true }
  );

  // ========== 8. Actions ==========

  /**
   * 导入SRT字幕（从转录结果加载）
   *
   * V3.1.2+dev.20260112.01: SRT 格式不包含置信度，将 confidence 和 display_confidence 设为 null
   * 前端不显示置信度徽章，避免误导用户
   */
  function importSRT(srtContent, metadata) {
    const parsed = parseSRT(srtContent);
    subtitles.value = parsed.map((item, idx) => ({
      id: `subtitle-${Date.now()}-${idx}`,
      sentenceIndex: idx,  // V3.1.2: 添加 sentenceIndex 以支持 SSE 匹配
      start: toDisplayTime(item.start),
      end: toDisplayTime(item.end),
      text: item.text,
      isDirty: false,
      isModified: false,
      originalText: null,
      // Phase 5: 双模态架构新增字段
      chunk_id: null, // 物理切片ID
      isDraft: false, // 已导入的SRT都是定稿
      isFinalized: true,
      words: [], // 字级置信度数据
      // V3.1.2+dev.20260112.01: SRT 无置信度数据，置空避免显示误导性的 100%
      confidence: null,
      display_confidence: null,
      confidence_source: 'srt_fallback',  // 标记来源用于 UI 判断
      warning_type: "none", // 警告类型: none/low_confidence/high_perplexity/both
      source: "imported", // 来源: sensevoice/whisper/imported
      ...normalizeSpeakerFields({}, { stripSpeaker: true }),
    }));

    meta.value = {
      ...meta.value,
      ...metadata,
      lastSaved: Date.now(),
      isDirty: false,
      capabilitySnapshot: normalizeCapabilitySnapshot(
        metadata?.capabilitySnapshot ?? metadata?.capability_snapshot ?? meta.value.capabilitySnapshot
      ),
      subtitleOffset: subtitleOffset.value,
    };

    // 清除历史记录，避免撤销到空状态
    clearHistory();
    // 清除 Chunk 映射
    chunkSubtitleMap.value.clear();
    speakerProfiles.value = new Map();
    deletedSentenceIndices.value.clear();
  }

  /**
   * 导入 Segments（从 API 加载转录中的字幕）
   *
   * 与 importSRT 的区别：
   * - 直接导入 segments，不经过 SRT 转换
   * - 保留 sentenceIndex 字段，用于 SSE 推送时的字幕匹配
   * - 保留 confidence、source 等元数据
   * - V3.1.2: 保留 display_confidence、confidence_source
   */
  function importSegments(segments, metadata) {
    subtitles.value = segments.map((seg) => ({
      id: `subtitle-${seg.id}`,
      sentenceIndex: seg.id,  // 关键：保留全局句子索引
      start: toDisplayTime(seg.start),
      end: toDisplayTime(seg.end),
      text: seg.text,
      isDirty: false,
      isModified: seg.is_modified ?? false,
      originalText: seg.original_text ?? null,
      chunk_id: null,
      words: [],
      confidence: seg.confidence ?? null,
      display_confidence: seg.display_confidence,  // V3.1.2: 映射后准确率
      confidence_source: seg.confidence_source,    // V3.1.2: 置信度来源
      warning_type: "none",
      source: seg.source || "imported",
      isDraft: seg.is_draft ?? false,
      isFinalized: seg.is_finalized ?? !(seg.is_draft ?? false),
      ...normalizeSpeakerFields(seg, { stripSpeaker: Boolean(seg.is_draft) }),
    }));

    meta.value = {
      ...meta.value,
      ...metadata,
      lastSaved: Date.now(),
      isDirty: false,
      capabilitySnapshot: normalizeCapabilitySnapshot(
        metadata?.capabilitySnapshot ?? metadata?.capability_snapshot ?? meta.value.capabilitySnapshot
      ),
      subtitleOffset: subtitleOffset.value,
    };

    // 清除历史记录，避免撤销到空状态
    clearHistory();
    // 清除 Chunk 映射
    chunkSubtitleMap.value.clear();
    speakerProfiles.value = new Map();
    deletedSentenceIndices.value.clear();
  }

  /**
   * 从 Project API 数据载入字幕（Task6）。
   * @param {Array|Object} payload - 允许传入 segments 数组，或 { segments, doc_meta } 对象
   * @param {Object} metadata - 可选的附加 meta
   */
  function loadFromProjectData(payload, metadata = {}) {
    const segments = Array.isArray(payload) ? payload : payload?.segments || [];
    const docMeta = Array.isArray(payload) ? null : payload?.doc_meta || payload?.meta || null;
    const now = Date.now();
    subtitles.value = segments.map((seg, index) => {
      const segmentId = String(seg.segment_id || seg.id || `seg-${index}`);
      const legacyIndexRaw = seg.legacy_index ?? seg.sentence_index ?? index;
      const sentenceIndex = Number.isFinite(Number(legacyIndexRaw)) ? Number(legacyIndexRaw) : index;
      return {
        id: `subtitle-${segmentId}`,
        segment_id: segmentId,
        sentenceIndex,
        start: toDisplayTime(seg.start ?? 0),
        end: toDisplayTime(seg.end ?? 0),
        text: String(seg.text ?? ""),
        isDirty: false,
        isModified: Boolean(seg.is_modified),
        originalText: seg.original_text ?? null,
        chunk_id: null,
        words: Array.isArray(seg.words) ? seg.words : [],
        confidence: seg.confidence ?? null,
        display_confidence: seg.display_confidence,
        confidence_source: seg.confidence_source,
        warning_type: seg.warning_type || "none",
        source: seg.source_type || seg.source || "imported",
        isDraft: Boolean(seg.is_draft),
        isFinalized: seg.is_finalized ?? !Boolean(seg.is_draft),
        ...normalizeSpeakerFields(seg, { stripSpeaker: Boolean(seg.is_draft) }),
      };
    });

    meta.value = {
      ...meta.value,
      ...metadata,
      projectId: metadata.projectId || docMeta?.project_id || meta.value.projectId,
      mode: metadata.mode || meta.value.mode || "normal",
      taskMode: metadata.taskMode || metadata.task_mode || meta.value.taskMode || "transcribe",
      lastSaved: now,
      isDirty: false,
      capabilitySnapshot: normalizeCapabilitySnapshot(
        metadata?.capabilitySnapshot
          ?? metadata?.capability_snapshot
          ?? docMeta?.capability_snapshot
          ?? meta.value.capabilitySnapshot
      ),
      subtitleOffset: subtitleOffset.value,
    };

    clearHistory();
    chunkSubtitleMap.value.clear();
    speakerProfiles.value = new Map();
    deletedSentenceIndices.value.clear();
  }

  /**
   * 从缓存/存储恢复项目
   */
  async function restoreProject(identityId) {
    if (!identityId) {
      return false;
    }
    try {
      // 优先从内存缓存获取
      if (memoryCache.has(identityId)) {
        const cached = memoryCache.get(identityId);
        subtitles.value = cached.subtitles;
        meta.value = {
          ...cached.meta,
          capabilitySnapshot: normalizeCapabilitySnapshot(cached?.meta?.capabilitySnapshot),
        };
        if (cached?.meta?.subtitleOffset !== undefined) {
          setSubtitleOffset(cached.meta.subtitleOffset, { applyDelta: false });
        }
        // 恢复后清除历史记录，防止撤回到转录期间的状态
        clearHistory();
        console.log("[ProjectStore] 项目已从内存缓存恢复");
        return true;
      }

      // 使用智能保存系统恢复（支持 IndexedDB + localStorage 备份）
      const saved = await smartSaver.restoreFromBackup(identityId);
      if (saved) {
        subtitles.value = saved.subtitles;
        meta.value = {
          ...saved.meta,
          capabilitySnapshot: normalizeCapabilitySnapshot(saved?.meta?.capabilitySnapshot),
        };
        if (saved?.meta?.subtitleOffset !== undefined) {
          setSubtitleOffset(saved.meta.subtitleOffset, { applyDelta: false });
        }
        // 恢复后清除历史记录，防止撤回到转录期间的状态
        clearHistory();
        console.log("[ProjectStore] 项目已从存储恢复");
        return true;
      }
      return false;
    } catch (error) {
      console.error("[ProjectStore] 恢复项目失败:", error);
      return false;
    }
  }

  /**
   * 更新字幕内容
   */
  function updateSubtitle(id, payload, options = {}) {
    const index = subtitles.value.findIndex((s) => s.id === id);
    if (index === -1) return;

    const current = subtitles.value[index];
    const { isUserEdit = false } = options;
    const normalizedPayload = { ...payload };

    if (normalizedPayload.is_modified !== undefined && normalizedPayload.isModified === undefined) {
      normalizedPayload.isModified = normalizedPayload.is_modified;
    }
    if (normalizedPayload.original_text !== undefined && normalizedPayload.originalText === undefined) {
      normalizedPayload.originalText = normalizedPayload.original_text;
    }
    delete normalizedPayload.is_modified;
    delete normalizedPayload.original_text;

    // 如果用户修改了文本，清空模型置信度，避免误导性徽章
    const isTextEdited = isUserEdit && normalizedPayload.text !== undefined && normalizedPayload.text !== current.text;
    const sanitizedPayload = isTextEdited
      ? {
          ...normalizedPayload,
          confidence: null,
          display_confidence: null,
          confidence_source: 'manual',
        }
      : normalizedPayload;

    if (isUserEdit) {
      const hasUserEdit = sanitizedPayload.text !== undefined
        || sanitizedPayload.start !== undefined
        || sanitizedPayload.end !== undefined;
      if (hasUserEdit) {
        sanitizedPayload.isModified = true;
      }
      if (sanitizedPayload.text !== undefined && !current.originalText && !sanitizedPayload.originalText) {
        sanitizedPayload.originalText = current.text;
      }
    }

    subtitles.value[index] = {
      ...current,
      ...sanitizedPayload,
      isDirty: isUserEdit ? true : current.isDirty,
    };
    if (isUserEdit) {
      meta.value.isDirty = true;
    }
  }

  /**
   * 添加字幕
   *
   * V3.1.2+dev.20260112.01: 补齐 sentenceIndex、display_confidence、confidence_source 字段
   */
  function addSubtitle(insertIndex, payload) {
    const newSubtitle = {
      id: `subtitle-${Date.now()}`,
      sentenceIndex: payload.sentenceIndex,  // V3.1.2: 全局句子索引
      start: payload.start || 0,
      end: payload.end || 0,
      text: payload.text || "",
      isDirty: true,
      isModified: payload.isModified ?? false,
      originalText: payload.originalText ?? null,
      // Phase 5: 双模态架构新增字段
      chunk_id: payload.chunk_id || null,
      isDraft: payload.isDraft ?? false,
      words: payload.words || [],
      confidence: payload.confidence ?? null,  // V3.1.2: 默认 null 而非 1.0
      display_confidence: payload.display_confidence,  // V3.1.2: 映射后准确率
      confidence_source: payload.confidence_source ?? 'manual',    // V3.1.2: 置信度来源
      warning_type: payload.warning_type || "none",
      source: payload.source || "manual",
    };
    subtitles.value.splice(insertIndex, 0, newSubtitle);
    meta.value.isDirty = true;
  }

  /**
   * 删除字幕
   */
  function removeSubtitle(id, options = {}) {
    const index = subtitles.value.findIndex((s) => s.id === id);
    if (index !== -1) {
      const subtitle = subtitles.value[index];
      if (options.isUserEdit && subtitle?.sentenceIndex !== undefined) {
        deletedSentenceIndices.value.add(subtitle.sentenceIndex);
      }
      if (subtitle?.chunk_id && chunkSubtitleMap.value.has(subtitle.chunk_id)) {
        const list = chunkSubtitleMap.value.get(subtitle.chunk_id) || [];
        chunkSubtitleMap.value.set(
          subtitle.chunk_id,
          list.filter((item) => item !== subtitle.id)
        );
      }
      subtitles.value.splice(index, 1);
      meta.value.isDirty = true;
    }
  }

  function isSentenceDeleted(sentenceIndex) {
    return deletedSentenceIndices.value.has(sentenceIndex);
  }

  function markSentenceDeleted(sentenceIndex) {
    if (sentenceIndex === undefined || sentenceIndex === null) return;
    deletedSentenceIndices.value.add(sentenceIndex);
  }

  // ========== 字幕切分功能 ==========

  /**
   * 切分字幕（核心方法）
   *
   * 支持两种切分模式：
   * 1. 基于时间点切分（波形图右键）- 传入 splitTime
   * 2. 基于光标位置切分（字幕列表编辑模式）- 传入 cursorPosition
   *
   * 时间戳计算策略（混合策略）：
   * - 优先使用字级时间戳（words 数组）精确切分
   * - 回退到字数比例估算
   *
   * @param {string} id - 要切分的字幕 ID
   * @param {Object} options - 切分选项
   * @param {number} [options.splitTime] - 切分时间点（秒），用于波形图切分
   * @param {number} [options.cursorPosition] - 光标位置（字符索引），用于文本切分
   * @returns {Object|null} 切分结果 { success, leftId, rightId, error }
   */
  function splitSubtitle(id, options = {}) {
    const { splitTime, cursorPosition } = options;

    // 1. 查找目标字幕
    const index = subtitles.value.findIndex((s) => s.id === id);
    if (index === -1) {
      return { success: false, error: '字幕不存在' };
    }

    const subtitle = subtitles.value[index];

    // 2. 前置校验：草稿字幕不允许切分
    if (subtitle.isDraft) {
      return { success: false, error: '草稿字幕不允许切分，请等待转录完成' };
    }

    // 3. 计算切分点
    let splitResult;
    if (splitTime !== undefined) {
      // 基于时间点切分（波形图模式）
      splitResult = _splitByTime(subtitle, splitTime);
    } else if (cursorPosition !== undefined) {
      // 基于光标位置切分（文本编辑模式）
      splitResult = _splitByCursor(subtitle, cursorPosition);
    } else {
      return { success: false, error: '必须提供 splitTime 或 cursorPosition' };
    }

    if (!splitResult.success) {
      return splitResult;
    }

    const { left, right } = splitResult;

    // 4. 生成新字幕 ID
    const timestamp = Date.now();
    const leftId = `${id}-split-L-${timestamp}`;
    const rightId = `${id}-split-R-${timestamp}`;

    // 5. 构建新字幕对象
    const leftSubtitle = {
      ...subtitle,
      id: leftId,
      start: left.start,
      end: left.end,
      text: left.text,
      words: left.words || [],
      isDirty: true,
      isModified: true,
      source: 'split',  // 标记来源为切分
    };

    const rightSubtitle = {
      ...subtitle,
      id: rightId,
      start: right.start,
      end: right.end,
      text: right.text,
      words: right.words || [],
      isDirty: true,
      isModified: true,
      source: 'split',
    };
    rightSubtitle.sentenceIndex = undefined;
    rightSubtitle.segment_id = undefined;
    rightSubtitle.originalText = null;

    // 6. 原子操作：删除旧字幕，插入两个新字幕
    console.log('[ProjectStore] 切分前历史记录数:', history.value.length);
    subtitles.value.splice(index, 1, leftSubtitle, rightSubtitle);
    console.log('[ProjectStore] 切分后历史记录数:', history.value.length);

    // 7. 标记项目已修改
    meta.value.isDirty = true;

    console.log(`[ProjectStore] 字幕切分成功: ${id} -> ${leftId}, ${rightId}`);

    return {
      success: true,
      leftId,
      rightId,
      originalSentenceIndex: subtitle.sentenceIndex,
      leftSubtitle,
      rightSubtitle,
    };
  }

  /**
   * 基于时间点切分（内部方法）
   * 用于波形图右键切分场景
   */
  function _splitByTime(subtitle, splitTime) {
    const { start, end, text, words } = subtitle;

    // 校验时间点是否在字幕范围内
    if (splitTime <= start || splitTime >= end) {
      return { success: false, error: '切分点必须在字幕时间范围内' };
    }

    // 优先使用字级时间戳
    if (words && words.length > 0) {
      // 查找切分点所在的字
      const splitIndex = words.findIndex(w => w.end >= splitTime);

      if (splitIndex > 0) {
        const leftWords = words.slice(0, splitIndex);
        const rightWords = words.slice(splitIndex);

        return {
          success: true,
          left: {
            start: start,
            end: leftWords[leftWords.length - 1].end,
            text: leftWords.map(w => w.word).join(''),
            words: leftWords,
          },
          right: {
            start: rightWords[0].start,
            end: end,
            text: rightWords.map(w => w.word).join(''),
            words: rightWords,
          },
        };
      }
    }

    // 回退：按时间比例估算文本切分点
    const ratio = (splitTime - start) / (end - start);
    const charIndex = Math.max(1, Math.min(text.length - 1, Math.round(text.length * ratio)));

    return {
      success: true,
      left: {
        start: start,
        end: splitTime,
        text: text.slice(0, charIndex),
        words: [],
      },
      right: {
        start: splitTime,
        end: end,
        text: text.slice(charIndex),
        words: [],
      },
    };
  }

  /**
   * 基于光标位置切分（内部方法）
   * 用于字幕列表编辑模式右键切分场景
   */
  function _splitByCursor(subtitle, cursorPosition) {
    const { start, end, text, words } = subtitle;

    // 校验光标位置
    if (cursorPosition <= 0 || cursorPosition >= text.length) {
      return { success: false, error: '光标位置必须在文本中间' };
    }

    const leftText = text.slice(0, cursorPosition);
    const rightText = text.slice(cursorPosition);

    // 优先使用字级时间戳
    if (words && words.length > 0) {
      // 根据文本切分点找字级时间戳边界
      let charCount = 0;
      let splitWordIndex = 0;

      for (let i = 0; i < words.length; i++) {
        charCount += words[i].word.length;
        if (charCount >= cursorPosition) {
          splitWordIndex = i + 1;
          break;
        }
      }

      // 确保至少切分出一个字
      splitWordIndex = Math.max(1, Math.min(words.length - 1, splitWordIndex));

      const leftWords = words.slice(0, splitWordIndex);
      const rightWords = words.slice(splitWordIndex);

      if (leftWords.length > 0 && rightWords.length > 0) {
        return {
          success: true,
          left: {
            start: start,
            end: leftWords[leftWords.length - 1].end,
            text: leftText,
            words: leftWords,
          },
          right: {
            start: rightWords[0].start,
            end: end,
            text: rightText,
            words: rightWords,
          },
        };
      }
    }

    // 回退：按字数比例估算时间
    const ratio = cursorPosition / text.length;
    const splitTime = start + (end - start) * ratio;

    return {
      success: true,
      left: {
        start: start,
        end: splitTime,
        text: leftText,
        words: [],
      },
      right: {
        start: splitTime,
        end: end,
        text: rightText,
        words: [],
      },
    };
  }

  /**
   * V3.2.4+dev.20260302.02: 读取合并分隔符设置
   * 从 localStorage 读取用户配置的字幕合并分隔符
   * @returns {string} 分隔符字符串
   */
  function getMergeSeparator() {
    try {
      const raw = localStorage.getItem('editor-merge-separator');
      if (!raw) return ' ';
      const config = JSON.parse(raw);
      const map = {
        'space': ' ',
        'comma-full': '\uff0c',
        'comma-half': ',',
        'period-full': '\u3002',
        'period-half': '.',
      };
      if (config.type === 'custom') return config.custom || '';
      return map[config.type] ?? ' ';
    } catch {
      return ' ';
    }
  }

  /**
   * 合并相邻字幕（核心方法）
   *
   * 支持两个方向：
   * 1. prev - 与前字幕合并：保留前一条 identity，删除当前条
   * 2. next - 与后字幕合并：保留当前条 identity，删除后一条
   *
   * 合并规则：
   * - 文本：按时间顺序拼接（前 + 分隔符 + 后）
   * - 时间戳：start = min(两者), end = max(两者)
   * - 索引：数组 splice 后 Vue 自动重渲染序号
   *
   * @param {string} id - 当前右键的字幕 ID
   * @param {'prev'|'next'} direction - 合并方向
   * @returns {Object} { success, keptSubtitle, removedSubtitle, error }
   */
  function mergeSubtitles(id, direction) {
    // 1. 查找当前字幕
    const index = subtitles.value.findIndex((s) => s.id === id);
    if (index === -1) {
      return { success: false, error: '字幕不存在' };
    }

    const current = subtitles.value[index];

    // 2. 前置校验
    if (current.isDraft) {
      return { success: false, error: '草稿字幕不允许合并' };
    }

    // 3. 确定目标字幕
    const targetIndex = direction === 'prev' ? index - 1 : index + 1;
    if (targetIndex < 0 || targetIndex >= subtitles.value.length) {
      return { success: false, error: '没有可合并的相邻字幕' };
    }

    const target = subtitles.value[targetIndex];
    if (target.isDraft) {
      return { success: false, error: '目标字幕为草稿，不允许合并' };
    }

    // 4. 确定合并顺序（时间上靠前的 + 靠后的）
    const first = direction === 'prev' ? target : current;
    const second = direction === 'prev' ? current : target;

    // 5. 合并文本和时间戳（使用可配置分隔符）
    const separator = getMergeSeparator();
    const mergedText = first.text + separator + second.text;
    const mergedStart = Math.min(first.start, second.start);
    const mergedEnd = Math.max(first.end, second.end);

    // 6. 确定保留方和删除方
    //    merge-prev: 保留 prev(targetIndex)，删除 current(index)
    //    merge-next: 保留 current(index)，删除 next(targetIndex)
    const keptIndex = direction === 'prev' ? targetIndex : index;
    const removedIndex = direction === 'prev' ? index : targetIndex;
    const kept = subtitles.value[keptIndex];
    const removed = subtitles.value[removedIndex];

    // 保存删除方信息（splice 前）
    const removedSnapshot = { ...removed };

    // 7. 原子操作：单次 splice 替换两条为一条（确保 useRefHistory 只生成一条历史记录）
    //    keptIndex 始终是较小索引，splice(keptIndex, 2, merged) 一次完成
    const mergedSubtitle = {
      ...kept,
      text: mergedText,
      start: mergedStart,
      end: mergedEnd,
      words: [], // 合并后字级时间戳失效
      isDirty: true,
      isModified: true,
    };
    subtitles.value.splice(keptIndex, 2, mergedSubtitle);

    // 8. 标记项目已修改
    meta.value.isDirty = true;

    const keptAfterSplice = subtitles.value.find((s) => s.id === kept.id);
    console.log(`[ProjectStore] 字幕合并成功: ${first.id} + ${second.id} -> ${kept.id}`);

    return {
      success: true,
      keptSubtitle: keptAfterSplice,
      removedSubtitle: removedSnapshot,
    };
  }

  // ========== Phase 5: 双模态架构专用方法 ==========

  /**
   * 添加或更新草稿字幕（快流推送）
   *
   * 当收到 subtitle.draft 事件时调用
   * 按时间顺序插入，保持字幕列表有序
   * 【注意】此操作不记录到历史，用户无法撤销 SSE 推送的内容
   *
   * @param {string} chunk_id - Chunk ID
   * @param {object} sentenceData - 句子数据
   */
  function appendOrUpdateDraft(chunk_id, sentenceData) {
    // 暂停历史记录，SSE 推送的内容不应被撤销
    pauseHistory();

    // V3.1.2+dev.20260111.01: 添加 display_confidence 和 confidence_source
    const {
      index: sentenceIndex,
      text,
      start,
      end,
      confidence = null,
      display_confidence,  // V3.1.2: 映射后准确率
      confidence_source,   // V3.1.2: 置信度来源
      words = [],
      warning_type = "none",
      is_draft,
      is_finalized,
    } = sentenceData;
    const isDraftSentence = is_draft ?? !Boolean(is_finalized);
    const isFinalizedSentence = is_finalized ?? !isDraftSentence;

    if (deletedSentenceIndices.value.has(sentenceIndex)) {
      console.log(`[ProjectStore] 草稿字幕已被用户删除，跳过: ${sentenceIndex}`);
      resumeHistory();
      return;
    }

    // 生成唯一ID
    const subtitleId = `draft-${chunk_id}-${sentenceIndex}`;

    // 查找是否已存在
    const existingIndex = subtitles.value.findIndex((s) => s.id === subtitleId);

    // V3.1.2+dev.20260111.01: 包含 display_confidence 和 confidence_source
    const normalized = applyOffsetToSentenceData({ start, end, words });
    const subtitleData = {
      id: subtitleId,
      start: normalized.start,
      end: normalized.end,
      text,
      isDirty: false,
      isModified: sentenceData.is_modified ?? false,
      originalText: sentenceData.original_text ?? null,
      chunk_id,
      isDraft: isDraftSentence,
      isFinalized: isFinalizedSentence,
      words: normalized.words,
      confidence,
      display_confidence,  // V3.1.2: 映射后准确率
      confidence_source,   // V3.1.2: 置信度来源
      warning_type,
      source: sentenceData.source || (isDraftSentence ? "sensevoice" : "finalized"),
      sentenceIndex, // 保留原始句子索引
      ...normalizeSpeakerFields(sentenceData, { stripSpeaker: Boolean(isDraftSentence) }),
    };

    if (existingIndex >= 0) {
      // 更新现有草稿
      subtitles.value[existingIndex] = subtitleData;
      console.log(`[ProjectStore] 更新草稿字幕: ${subtitleId}`);
    } else {
      // 按时间顺序插入
      const insertIndex = findInsertIndex(normalized.start);
      subtitles.value.splice(insertIndex, 0, subtitleData);

      // 更新 Chunk 映射
      if (!chunkSubtitleMap.value.has(chunk_id)) {
        chunkSubtitleMap.value.set(chunk_id, []);
      }
      chunkSubtitleMap.value.get(chunk_id).push(subtitleId);

      console.log(
        `[ProjectStore] 添加草稿字幕: ${subtitleId}, 位置: ${insertIndex}, ` +
        `display_confidence: ${display_confidence}, confidence: ${confidence}`
      );
    }

    // 更新双流进度
    updateDualStreamProgress();

    // 恢复历史记录
    resumeHistory();
  }

  /**
   * 替换 Chunk 的所有字幕（慢流推送）
   *
   * 当收到 subtitle.replace_chunk 事件时调用
   * 删除旧的草稿字幕，添加新的定稿字幕
   *
   * @param {string} chunk_id - Chunk ID
   * @param {Array} sentences - 定稿句子列表
   */
  function replaceChunk(chunk_id, sentences) {
    // 暂停历史记录，SSE 推送的内容不应被撤销
    pauseHistory();

    // 1. 删除该 Chunk 的所有旧字幕（保留用户编辑）
    const oldSubtitleIds = chunkSubtitleMap.value.get(chunk_id) || [];
    const protectedIds = oldSubtitleIds.filter((id) => {
      const subtitle = subtitles.value.find((s) => s.id === id);
      return subtitle?.isModified;
    });
    subtitles.value = subtitles.value.filter(
      (s) => !oldSubtitleIds.includes(s.id) || protectedIds.includes(s.id)
    );

    console.log(
      `[ProjectStore] 删除 Chunk ${chunk_id} 的 ${oldSubtitleIds.length} 个旧字幕, ` +
      `保护 ${protectedIds.length} 条用户编辑`
    );

    // 2. 添加新的定稿字幕
    const newSubtitleIds = [...protectedIds];
    sentences.forEach((sentence, idx) => {
      if (deletedSentenceIndices.value.has(sentence.index)) {
        return;
      }
      const subtitleId = buildUniqueSubtitleId(`final-${chunk_id}-${idx}`);
      const normalized = applyOffsetToSentenceData({
        start: sentence.start,
        end: sentence.end,
        words: sentence.words || [],
      });
      // V3.1.2+dev.20260111.01: 包含 display_confidence 和 confidence_source
      const subtitleData = {
        id: subtitleId,
        start: normalized.start,
        end: normalized.end,
        text: sentence.text,
        isDirty: false,
        isModified: sentence.is_modified ?? false,
        originalText: sentence.original_text ?? null,
        chunk_id,
        isDraft: false, // 定稿
        isFinalized: true,
        words: normalized.words || [],
        confidence: sentence.confidence ?? null,
        display_confidence: sentence.display_confidence,  // V3.1.2: 映射后准确率
        confidence_source: sentence.confidence_source,    // V3.1.2: 置信度来源
        warning_type: sentence.warning_type || "none",
        source: sentence.source || "whisper",
        sentenceIndex: sentence.index,
        ...normalizeSpeakerFields(sentence, { stripSpeaker: false }),
      };

      // 按时间顺序插入
      const insertIndex = findInsertIndex(normalized.start);
      subtitles.value.splice(insertIndex, 0, subtitleData);
      newSubtitleIds.push(subtitleId);
    });

    // 3. 更新 Chunk 映射
    chunkSubtitleMap.value.set(chunk_id, newSubtitleIds);

    console.log(
      `[ProjectStore] 替换 Chunk ${chunk_id}: 添加 ${sentences.length} 个定稿字幕, ` +
      `首条 display_confidence: ${sentences[0]?.display_confidence}`
    );

    // 更新双流进度
    updateDualStreamProgress();

    // 恢复历史记录
    resumeHistory();
  }

  /**
   * 取消收敛时将已推送草稿转为定稿。
   *
   * 约束：
   * 1. 已有定稿保持不变
   * 2. 草稿转定稿时沿用“同句索引优先去重”策略，避免与既有定稿重复
   * 3. 未推送数据不处理
   */
  async function finalizeDraftSubtitlesOnCancel() {
    pauseHistory();

    const finalizedSentenceKeys = new Set();
    const finalizedChunkKeys = new Set();
    subtitles.value.forEach((subtitle) => {
      if (subtitle.isDraft) return;
      if (subtitle.sentenceIndex !== undefined && subtitle.sentenceIndex !== null) {
        finalizedSentenceKeys.add(String(subtitle.sentenceIndex));
      }
      if (subtitle.chunk_id !== undefined && subtitle.chunk_id !== null) {
        finalizedChunkKeys.add(
          `${subtitle.chunk_id}::${(subtitle.text || "").trim()}::${Number(subtitle.start || 0).toFixed(3)}::${Number(subtitle.end || 0).toFixed(3)}`
        );
      }
    });

    const removedIds = new Set();
    subtitles.value = subtitles.value
      .map((subtitle) => {
        if (!subtitle.isDraft) {
          return subtitle;
        }

        const hasSentenceIndex = subtitle.sentenceIndex !== undefined && subtitle.sentenceIndex !== null;
        const sentenceKey = hasSentenceIndex ? String(subtitle.sentenceIndex) : null;
        const chunkKey = subtitle.chunk_id !== undefined && subtitle.chunk_id !== null
          ? `${subtitle.chunk_id}::${(subtitle.text || "").trim()}::${Number(subtitle.start || 0).toFixed(3)}::${Number(subtitle.end || 0).toFixed(3)}`
          : null;
        const isDuplicated = Boolean(
          (sentenceKey && finalizedSentenceKeys.has(sentenceKey))
          || (chunkKey && finalizedChunkKeys.has(chunkKey))
        );
        if (isDuplicated) {
          removedIds.add(subtitle.id);
          return null;
        }

        if (sentenceKey) {
          finalizedSentenceKeys.add(sentenceKey);
        }
        if (chunkKey) {
          finalizedChunkKeys.add(chunkKey);
        }

        return {
          ...subtitle,
          isDraft: false,
          isFinalized: true,
          source: subtitle.source || "cancel_finalized",
        };
      })
      .filter(Boolean);

    if (removedIds.size > 0) {
      chunkSubtitleMap.value.forEach((subtitleIds, chunkId) => {
        const filtered = (subtitleIds || []).filter((id) => !removedIds.has(id));
        chunkSubtitleMap.value.set(chunkId, filtered);
      });
    }

    subtitles.value.sort((left, right) => {
      const byStart = (left.start || 0) - (right.start || 0);
      if (byStart !== 0) return byStart;
      return (left.end || 0) - (right.end || 0);
    });

    updateDualStreamProgress();
    console.log(
      `[ProjectStore] 取消收敛完成，草稿转定稿并去重: removed=${removedIds.size}`
    );

    // 关键路径使用同步备份 + 立即持久化，避免取消后用户立刻刷新导致数据丢失
    const cacheKey = primaryId.value;
    if (cacheKey) {
      try {
        await smartSaver.forceSaveCritical({
          jobId: cacheKey,
          subtitles: subtitles.value,
          meta: meta.value,
        });
      } catch (error) {
        console.error("[ProjectStore] 取消收敛后关键保存失败:", error);
      }
    }
    resumeHistory();
  }

  /**
   * V3.1.0: 恢复字幕（断点续传后恢复）
   *
   * 当收到 subtitle.restored 事件时调用
   * 将从 Checkpoint 恢复的字幕添加到前端，确保不会与已有字幕冲突
   *
   * @param {string} chunk_id - Chunk ID
   * @param {Array} sentences - 恢复的句子列表
   */
  function restoreChunk(chunk_id, sentences) {
    console.log(`[ProjectStore] restoreChunk 被调用: chunk_id=${chunk_id}, sentences.length=${sentences?.length || 0}`);

    // 参数校验
    if (chunk_id === undefined || chunk_id === null) {
      console.warn('[ProjectStore] restoreChunk: chunk_id 为 undefined/null，使用 "unknown" 作为默认值');
      chunk_id = 'unknown';
    }

    if (!sentences || sentences.length === 0) {
      console.warn('[ProjectStore] restoreChunk: sentences 为空，跳过恢复');
      return;
    }

    // 暂停历史记录，恢复的内容不应被撤销
    pauseHistory();

    // 检查该 Chunk 是否已有字幕（避免重复恢复）
    const existingIds = chunkSubtitleMap.value.get(chunk_id) || [];
    if (existingIds.length > 0) {
      console.log(
        `[ProjectStore] Chunk ${chunk_id} 已有 ${existingIds.length} 个字幕，跳过恢复`
      );
      resumeHistory();
      return;
    }

    // 添加恢复的字幕
    const newSubtitleIds = [];
    const beforeLength = subtitles.value.length;

    sentences.forEach((sentence, idx) => {
      if (deletedSentenceIndices.value.has(sentence.index)) {
        return;
      }
      // 使用 restored 前缀标识恢复的字幕
      const subtitleId = `restored-${chunk_id}-${sentence.index ?? idx}`;
      const normalized = applyOffsetToSentenceData({
        start: sentence.start,
        end: sentence.end,
        words: sentence.words || [],
      });
      // V3.1.2+dev.20260111.01: 包含 display_confidence 和 confidence_source
      const subtitleData = {
        id: subtitleId,
        start: normalized.start,
        end: normalized.end,
        text: sentence.text,
        isDirty: false,
        isModified: sentence.is_modified ?? false,
        originalText: sentence.original_text ?? null,
        chunk_id,
        isDraft: sentence.is_draft ?? false,
        isFinalized: sentence.is_finalized ?? !(sentence.is_draft ?? false),
        isRestored: true, // 标记为恢复的字幕
        words: normalized.words || [],
        confidence: sentence.confidence ?? null,
        display_confidence: sentence.display_confidence,  // V3.1.2: 映射后准确率
        confidence_source: sentence.confidence_source,    // V3.1.2: 置信度来源
        warning_type: sentence.warning_type || "none",
        source: sentence.source || "restored",
        sentenceIndex: sentence.index,
        ...normalizeSpeakerFields(sentence, { stripSpeaker: Boolean(sentence.is_draft ?? false) }),
      };

      // 按时间顺序插入
      const insertIndex = findInsertIndex(normalized.start);
      subtitles.value.splice(insertIndex, 0, subtitleData);
      newSubtitleIds.push(subtitleId);
    });

    // 更新 Chunk 映射
    chunkSubtitleMap.value.set(chunk_id, newSubtitleIds);

    const afterLength = subtitles.value.length;
    console.log(
      `[ProjectStore] 恢复 Chunk ${chunk_id}: 添加 ${sentences.length} 个字幕, ` +
      `subtitles: ${beforeLength} -> ${afterLength}`
    );

    // 更新双流进度
    updateDualStreamProgress();

    // 恢复历史记录
    resumeHistory();
  }

  /**
   * 应用后端推送的修订事件（包含说话人改绑）。
   *
   * 支持两种格式：
   * - 单条：{ index, sentence }
   * - 批量：{ sentences: [...] }
   */
  function applyRevisedSubtitle(data) {
    if (!data) return;

    const revisions = Array.isArray(data.sentences)
      ? data.sentences
      : [data.sentence ? { ...data.sentence, index: data.index ?? data.sentence.index } : data];

    if (revisions.length === 0) return;
    pauseHistory();

    let revisedCount = 0;
    revisions.forEach((revision) => {
      const sentenceIndex = revision.index ?? revision.sentenceIndex;
      if (sentenceIndex === undefined || sentenceIndex === null) {
        return;
      }

      const existingIndex = subtitles.value.findIndex((item) => item.sentenceIndex === sentenceIndex);
      const isDraft = revision.is_draft ?? (existingIndex >= 0 ? subtitles.value[existingIndex].isDraft : false);
      const isFinalized = revision.is_finalized ?? !isDraft;
      const speakerFields = normalizeSpeakerFields(revision, { stripSpeaker: Boolean(isDraft) });

      const hasTiming =
        revision.start !== undefined
        || revision.end !== undefined
        || revision.words !== undefined;
      let normalizedTiming = {};
      if (hasTiming) {
        const normalized = applyOffsetToSentenceData({
          start: revision.start ?? 0,
          end: revision.end ?? (revision.start ?? 0),
          words: revision.words || [],
        });
        normalizedTiming = {
          ...(revision.start !== undefined ? { start: normalized.start } : {}),
          ...(revision.end !== undefined ? { end: normalized.end } : {}),
          ...(revision.words !== undefined ? { words: normalized.words } : {}),
        };
      }

      if (existingIndex >= 0) {
        const current = subtitles.value[existingIndex];
        subtitles.value[existingIndex] = {
          ...current,
          ...(revision.text !== undefined ? { text: revision.text } : {}),
          ...normalizedTiming,
          ...(revision.warning_type !== undefined ? { warning_type: revision.warning_type } : {}),
          ...(revision.source !== undefined ? { source: revision.source } : {}),
          isDraft,
          isFinalized,
          sentenceIndex,
          ...speakerFields,
        };
      } else {
        const normalized = applyOffsetToSentenceData({
          start: revision.start ?? 0,
          end: revision.end ?? 0,
          words: revision.words || [],
        });
        const newSubtitle = {
          id: buildUniqueSubtitleId(`revised-${sentenceIndex}`),
          sentenceIndex,
          start: normalized.start,
          end: normalized.end,
          text: revision.text || "",
          isDirty: false,
          isModified: revision.is_modified ?? false,
          originalText: revision.original_text ?? null,
          chunk_id: null,
          isDraft,
          isFinalized,
          words: normalized.words || [],
          confidence: revision.confidence ?? null,
          display_confidence: revision.display_confidence,
          confidence_source: revision.confidence_source,
          warning_type: revision.warning_type || "none",
          source: revision.source || "revised",
          ...speakerFields,
        };
        const insertIndex = findInsertIndex(normalized.start);
        subtitles.value.splice(insertIndex, 0, newSubtitle);
      }
      revisedCount += 1;
    });

    if (revisedCount > 0) {
      subtitles.value.sort((left, right) => {
        const byStart = (left.start ?? 0) - (right.start ?? 0);
        if (byStart !== 0) return byStart;
        return (left.end ?? 0) - (right.end ?? 0);
      });
      updateDualStreamProgress();
    }
    resumeHistory();
  }

  /**
   * 应用后端推送的说话人 profile 更新。
   */
  function applySpeakerProfiles(data) {
    if (!data || !Array.isArray(data.profiles) || data.profiles.length === 0) {
      return;
    }

    pauseHistory();
    const nextProfiles = new Map(speakerProfiles.value);
    data.profiles.forEach((profile) => {
      if (!profile?.speaker_id) return;
      nextProfiles.set(profile.speaker_id, profile);
    });
    speakerProfiles.value = nextProfiles;

    subtitles.value = subtitles.value.map((subtitle) => {
      if (subtitle.isDraft || !subtitle.speaker_id) {
        return subtitle;
      }
      const profile = speakerProfiles.value.get(subtitle.speaker_id);
      if (!profile) {
        return subtitle;
      }
      return {
        ...subtitle,
        speaker_label: profile.display_name || subtitle.speaker_label || subtitle.speaker_id,
        speaker_color_key: profile.color_key || subtitle.speaker_color_key,
      };
    });
    resumeHistory();
  }

  function buildUniqueSubtitleId(baseId) {
    const existingIds = new Set(subtitles.value.map((s) => s.id));
    if (!existingIds.has(baseId)) {
      return baseId;
    }
    let counter = 1;
    let candidate = `${baseId}-u${counter}`;
    while (existingIds.has(candidate)) {
      counter += 1;
      candidate = `${baseId}-u${counter}`;
    }
    return candidate;
  }

  /**
   * 查找按时间顺序的插入位置
   */
  function findInsertIndex(startTime) {
    let left = 0;
    let right = subtitles.value.length;

    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (subtitles.value[mid].start < startTime) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    return left;
  }

  /**
   * 更新双流进度统计
   */
  function updateDualStreamProgress() {
    const chunkIds = Array.from(chunkSubtitleMap.value.keys());
    const totalChunks = chunkIds.length;

    let processedChunks = 0;
    let finalizedChunks = 0;

    chunkIds.forEach((chunkId) => {
      const chunkSubtitles = subtitles.value.filter(
        (s) => s.chunk_id === chunkId
      );
      if (chunkSubtitles.length === 0) return;
      processedChunks++;

      const hasDraft = chunkSubtitles.some((s) => s.isDraft);
      const hasFinal = chunkSubtitles.some((s) => !s.isDraft);

      // 仅在 chunk 内全部为定稿时计入慢流完成
      if (!hasDraft && hasFinal) {
        finalizedChunks++;
      }
    });

    const fastStream =
      totalChunks > 0
        ? Math.round((processedChunks / totalChunks) * 100)
        : 0;
    const slowStream =
      totalChunks > 0
        ? Math.round((finalizedChunks / totalChunks) * 100)
        : 0;

    dualStreamProgress.value = {
      fastStream,
      slowStream,
      totalChunks,
      draftChunks: Math.max(processedChunks - finalizedChunks, 0),
      finalizedChunks,
    };
  }

  /**
   * V3.1.0: 从 SSE 后端推送更新双流进度
   * 这是更准确的进度来源，因为后端知道真实的 Chunk 处理进度
   *
   * @param {Object} progress - 从 SSE progress.overall 事件获取的进度数据
   *   - fastStream: SenseVoice 进度 (0-100)
   *   - slowStream: Whisper 进度 (0-100)
   *   - totalChunks: 总 Chunk 数
   */
  function updateDualStreamProgressFromSSE(progress) {
    if (!progress) return;

    dualStreamProgress.value = {
      ...dualStreamProgress.value,
      fastStream: progress.fastStream ?? dualStreamProgress.value.fastStream,
      slowStream: progress.slowStream ?? dualStreamProgress.value.slowStream,
      totalChunks: progress.totalChunks ?? dualStreamProgress.value.totalChunks,
    };

    console.log('[ProjectStore] 双流进度从 SSE 更新:', dualStreamProgress.value);
  }

  /**
   * 获取草稿字幕数量
   */
  const draftSubtitleCount = computed(
    () => subtitles.value.filter((s) => s.isDraft).length
  );

  /**
   * 获取定稿字幕数量
   */
  const finalizedSubtitleCount = computed(
    () => subtitles.value.filter((s) => !s.isDraft && s.chunk_id).length
  );

  /**
   * 导出SRT字符串
   * V3.1.1+dev.20260106.04: 导出前自动修复时间戳重叠
   */
  function generateSRT() {
    // 修复时间戳重叠（使用1ms间隔）
    const repairedSubtitles = repairSubtitleOverlaps(subtitles.value, 1);

    let srtContent = "";
    repairedSubtitles.forEach((sub, index) => {
      srtContent += `${index + 1}\n`;
      srtContent += `${formatTimestamp(sub.start)} --> ${formatTimestamp(
        sub.end
      )}\n`;
      srtContent += `${sub.text}\n\n`;
    });
    return srtContent;
  }

  /**
   * 同步播放器时间
   */
  function seekTo(time) {
    player.value.currentTime = time;
  }

  function setIsPlaying(isPlaying) {
    player.value.isPlaying = Boolean(isPlaying);
  }

  function setPlayerSeeking(isSeeking) {
    player.value.isSeeking = Boolean(isSeeking);
  }

  /**
   * 保存项目（持久化到后端 + 本地强制保存）
   */
  async function saveProject() {
    // TODO: 调用后端API保存编辑后的字幕
    const srtContent = generateSRT();
    // await api.saveSubtitle(meta.value.jobId, srtContent)

    const cacheKey = primaryId.value;
    if (!cacheKey) {
      return;
    }

    // 强制立即保存到本地存储
    await smartSaver.forceSave({
      jobId: cacheKey,
      subtitles: subtitles.value,
      meta: meta.value,
    });

    meta.value.lastSaved = Date.now();
    meta.value.isDirty = false;
    subtitles.value.forEach((s) => (s.isDirty = false));
    console.log("[ProjectStore] 项目已保存");
  }

  /**
   * 重置项目状态
   */
  function resetProject() {
    subtitles.value = [];
    meta.value = {
      jobId: null,
      projectId: null,
      mode: "normal",
      taskMode: "transcribe",
      flavor: "full",
      videoPath: null,
      audioPath: null,
      peaksPath: null,
      duration: 0,
      filename: "",
      title: "",
      videoFormat: null,
      hasProxyVideo: false,
      lastSaved: Date.now(),
      isDirty: false,
      // 渐进式加载相关（状态由 useProxyVideo composable 管理）
      currentResolution: null,
      // Phase 0.5: 能力协商快照留位（本期不参与行为判断）
      capabilitySnapshot: null,
      // V3.2.0+dev.20260130.09: 字幕全局时间偏移（秒）
      subtitleOffset: subtitleOffset.value,
    };
    player.value = {
      currentTime: 0,
      isPlaying: false,
      playbackRate: 1.0,
      volume: 1.0,
      isSeeking: false,
    };
    clearHistory();
    // Phase 5: 清除双模态架构状态
    chunkSubtitleMap.value.clear();
    dualStreamProgress.value = {
      fastStream: 0,
      slowStream: 0,
      totalChunks: 0,
      draftChunks: 0,
      finalizedChunks: 0,
    };
    deletedSentenceIndices.value.clear();
    speakerProfiles.value = new Map();
    console.log("[ProjectStore] 项目已重置");
  }

  // ========== 9. 辅助函数 ==========

  /**
   * 解析SRT字符串
   */
  function parseSRT(srtContent) {
    const blocks = srtContent.trim().split(/\n\n+/);
    return blocks
      .map((block) => {
        const lines = block.split("\n");
        const timeMatch = lines[1]?.match(
          /(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})/
        );
        if (!timeMatch) return null;

        return {
          start: parseTimestamp(timeMatch[1]),
          end: parseTimestamp(timeMatch[2]),
          text: lines.slice(2).join("\n"),
        };
      })
      .filter(Boolean);
  }

  /**
   * 解析时间戳字符串
   */
  function parseTimestamp(ts) {
    // "00:01:23,456" => 83.456 秒
    const [h, m, s] = ts.replace(",", ".").split(":");
    return parseInt(h) * 3600 + parseInt(m) * 60 + parseFloat(s);
  }

  /**
   * 格式化时间戳
   */
  function formatTimestamp(sec) {
    // 83.456 => "00:01:23,456"
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.round((sec % 1) * 1000);
    return `${h.toString().padStart(2, "0")}:${m
      .toString()
      .padStart(2, "0")}:${s.toString().padStart(2, "0")},${ms
      .toString()
      .padStart(3, "0")}`;
  }

  return {
    // 状态
    meta,
    subtitles,
    player,
    view,
    subtitleOffset,

    // Phase 5: 双模态架构状态
    chunkSubtitleMap,
    dualStreamProgress,
    speakerProfiles,

    // 计算属性
    primaryId,
    totalSubtitles,
    currentSubtitle,
    isDirty,
    validationErrors,

    // Phase 5: 双模态架构计算属性
    draftSubtitleCount,
    finalizedSubtitleCount,

    // 历史记录
    canUndo,
    canRedo,
    undo,
    redo,
    clearHistory,
    pauseHistory, // 暂停历史记录（用于 SSE 推送等系统操作）
    resumeHistory, // 恢复历史记录

    // 操作方法
    patchMeta,
    setIdentity,
    setMediaPaths,
    setProjectTitle,
    setProjectDuration,
    setCurrentResolution,
    setZoomLevel,
    setSelectedSubtitleId,
    setPlayerVolume,
    setPlaybackRate,
    setIsPlaying,
    setPlayerSeeking,
    insertSubtitleAt,
    removeSubtitleAt,
    importSRT,
    importSegments,
    loadFromProjectData,
    restoreProject,
    updateSubtitle,
    addSubtitle,
    removeSubtitle,
    markSentenceDeleted,
    isSentenceDeleted,
    splitSubtitle,  // 字幕切分
    mergeSubtitles, // 字幕合并
    generateSRT,
    seekTo,
    saveProject,
    resetProject,

    // Phase 5: 双模态架构方法
    appendOrUpdateDraft,
    replaceChunk,
    finalizeDraftSubtitlesOnCancel,
    restoreChunk, // V3.1.0: 断点续传字幕恢复
    applyRevisedSubtitle,
    applySpeakerProfiles,
    updateDualStreamProgress,
    updateDualStreamProgressFromSSE,  // V3.1.0: 从 SSE 更新双流进度

    // 辅助方法
    setSubtitleOffset,
    toBaseTime,
    toDisplayTime,
    applyOffsetToSegments,
    applyOffsetToSentenceData,
    formatTimestamp,
    parseTimestamp,
  };
});
