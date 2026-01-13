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
  });

  // ========== 2. 字幕数据（Single Source of Truth） ==========
  const subtitles = ref([]);

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
  const totalSubtitles = computed(() => subtitles.value.length);

  const currentSubtitle = computed(() => {
    return subtitles.value.find(
      (s) =>
        player.value.currentTime >= s.start && player.value.currentTime < s.end
    );
  });

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
      if (!meta.value.jobId) return;

      // 更新内存缓存
      memoryCache.set(meta.value.jobId, {
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
        jobId: meta.value.jobId,
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
      start: item.start,
      end: item.end,
      text: item.text,
      isDirty: false,
      // Phase 5: 双模态架构新增字段
      chunk_id: null, // 物理切片ID
      isDraft: false, // 已导入的SRT都是定稿
      words: [], // 字级置信度数据
      // V3.1.2+dev.20260112.01: SRT 无置信度数据，置空避免显示误导性的 100%
      confidence: null,
      display_confidence: null,
      confidence_source: 'srt_fallback',  // 标记来源用于 UI 判断
      warning_type: "none", // 警告类型: none/low_confidence/high_perplexity/both
      source: "imported", // 来源: sensevoice/whisper/imported
    }));

    meta.value = {
      ...meta.value,
      ...metadata,
      lastSaved: Date.now(),
      isDirty: false,
    };

    // 清除历史记录，避免撤销到空状态
    clearHistory();
    // 清除 Chunk 映射
    chunkSubtitleMap.value.clear();
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
      start: seg.start,
      end: seg.end,
      text: seg.text,
      isDirty: false,
      chunk_id: null,
      isDraft: false,
      words: [],
      confidence: seg.confidence || 1.0,
      display_confidence: seg.display_confidence,  // V3.1.2: 映射后准确率
      confidence_source: seg.confidence_source,    // V3.1.2: 置信度来源
      warning_type: "none",
      source: seg.source || "imported",
    }));

    meta.value = {
      ...meta.value,
      ...metadata,
      lastSaved: Date.now(),
      isDirty: false,
    };

    // 清除历史记录，避免撤销到空状态
    clearHistory();
    // 清除 Chunk 映射
    chunkSubtitleMap.value.clear();
  }

  /**
   * 从缓存/存储恢复项目
   */
  async function restoreProject(jobId) {
    try {
      // 优先从内存缓存获取
      if (memoryCache.has(jobId)) {
        const cached = memoryCache.get(jobId);
        subtitles.value = cached.subtitles;
        meta.value = cached.meta;
        // 恢复后清除历史记录，防止撤回到转录期间的状态
        clearHistory();
        console.log("[ProjectStore] 项目已从内存缓存恢复");
        return true;
      }

      // 使用智能保存系统恢复（支持 IndexedDB + localStorage 备份）
      const saved = await smartSaver.restoreFromBackup(jobId);
      if (saved) {
        subtitles.value = saved.subtitles;
        meta.value = saved.meta;
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
  function updateSubtitle(id, payload) {
    const index = subtitles.value.findIndex((s) => s.id === id);
    if (index === -1) return;

    const current = subtitles.value[index];

    // 如果用户修改了文本，清空模型置信度，避免误导性徽章
    const isTextEdited = payload.text !== undefined && payload.text !== current.text;
    const sanitizedPayload = isTextEdited
      ? {
          ...payload,
          confidence: null,
          display_confidence: null,
          confidence_source: 'manual',
        }
      : payload;

    subtitles.value[index] = {
      ...current,
      ...sanitizedPayload,
      isDirty: true,
    };
    meta.value.isDirty = true;
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
  function removeSubtitle(id) {
    const index = subtitles.value.findIndex((s) => s.id === id);
    if (index !== -1) {
      subtitles.value.splice(index, 1);
      meta.value.isDirty = true;
    }
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
      source: 'split',
    };

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
      confidence = 0.8,
      display_confidence,  // V3.1.2: 映射后准确率
      confidence_source,   // V3.1.2: 置信度来源
      words = [],
      warning_type = "none",
    } = sentenceData;

    // 生成唯一ID
    const subtitleId = `draft-${chunk_id}-${sentenceIndex}`;

    // 查找是否已存在
    const existingIndex = subtitles.value.findIndex((s) => s.id === subtitleId);

    // V3.1.2+dev.20260111.01: 包含 display_confidence 和 confidence_source
    const subtitleData = {
      id: subtitleId,
      start,
      end,
      text,
      isDirty: false,
      chunk_id,
      isDraft: true, // 标记为草稿
      words,
      confidence,
      display_confidence,  // V3.1.2: 映射后准确率
      confidence_source,   // V3.1.2: 置信度来源
      warning_type,
      source: "sensevoice",
      sentenceIndex, // 保留原始句子索引
    };

    if (existingIndex >= 0) {
      // 更新现有草稿
      subtitles.value[existingIndex] = subtitleData;
      console.log(`[ProjectStore] 更新草稿字幕: ${subtitleId}`);
    } else {
      // 按时间顺序插入
      const insertIndex = findInsertIndex(start);
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

    // 1. 删除该 Chunk 的所有旧字幕
    const oldSubtitleIds = chunkSubtitleMap.value.get(chunk_id) || [];
    subtitles.value = subtitles.value.filter(
      (s) => !oldSubtitleIds.includes(s.id)
    );

    console.log(
      `[ProjectStore] 删除 Chunk ${chunk_id} 的 ${oldSubtitleIds.length} 个旧字幕`
    );

    // 2. 添加新的定稿字幕
    const newSubtitleIds = [];
    sentences.forEach((sentence, idx) => {
      const subtitleId = `final-${chunk_id}-${idx}`;
      // V3.1.2+dev.20260111.01: 包含 display_confidence 和 confidence_source
      const subtitleData = {
        id: subtitleId,
        start: sentence.start,
        end: sentence.end,
        text: sentence.text,
        isDirty: false,
        chunk_id,
        isDraft: false, // 定稿
        words: sentence.words || [],
        confidence: sentence.confidence ?? 1.0,
        display_confidence: sentence.display_confidence,  // V3.1.2: 映射后准确率
        confidence_source: sentence.confidence_source,    // V3.1.2: 置信度来源
        warning_type: sentence.warning_type || "none",
        source: sentence.source || "whisper",
      };

      // 按时间顺序插入
      const insertIndex = findInsertIndex(sentence.start);
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
      // 使用 restored 前缀标识恢复的字幕
      const subtitleId = `restored-${chunk_id}-${sentence.index ?? idx}`;
      // V3.1.2+dev.20260111.01: 包含 display_confidence 和 confidence_source
      const subtitleData = {
        id: subtitleId,
        start: sentence.start,
        end: sentence.end,
        text: sentence.text,
        isDirty: false,
        chunk_id,
        isDraft: sentence.is_draft ?? false,
        isRestored: true, // 标记为恢复的字幕
        words: sentence.words || [],
        confidence: sentence.confidence ?? 1.0,
        display_confidence: sentence.display_confidence,  // V3.1.2: 映射后准确率
        confidence_source: sentence.confidence_source,    // V3.1.2: 置信度来源
        warning_type: sentence.warning_type || "none",
        source: sentence.source || "restored",
        sentenceIndex: sentence.index,
      };

      // 按时间顺序插入
      const insertIndex = findInsertIndex(sentence.start);
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

  /**
   * 保存项目（持久化到后端 + 本地强制保存）
   */
  async function saveProject() {
    // TODO: 调用后端API保存编辑后的字幕
    const srtContent = generateSRT();
    // await api.saveSubtitle(meta.value.jobId, srtContent)

    // 强制立即保存到本地存储
    await smartSaver.forceSave({
      jobId: meta.value.jobId,
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

    // Phase 5: 双模态架构状态
    chunkSubtitleMap,
    dualStreamProgress,

    // 计算属性
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
    importSRT,
    importSegments,
    restoreProject,
    updateSubtitle,
    addSubtitle,
    removeSubtitle,
    splitSubtitle,  // 字幕切分
    generateSRT,
    seekTo,
    saveProject,
    resetProject,

    // Phase 5: 双模态架构方法
    appendOrUpdateDraft,
    replaceChunk,
    restoreChunk, // V3.1.0: 断点续传字幕恢复
    updateDualStreamProgress,
    updateDualStreamProgressFromSSE,  // V3.1.0: 从 SSE 更新双流进度

    // 辅助方法
    formatTimestamp,
    parseTimestamp,
  };
});
