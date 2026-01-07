/**
 * SmartSaver - 智能保存系统
 *
 * 提供多策略自动保存功能：
 * - 防抖保存：日常编辑时减少 I/O
 * - 空闲保存：利用浏览器空闲时间
 * - 紧急保存：页面关闭时同步备份
 * - [预留] Worker 保存：大数据量时的异步处理
 *
 * 设计原则：
 * 1. 数据安全优先：多重备份机制
 * 2. 性能友好：避免阻塞主线程
 * 3. 可扩展：预留 Web Worker 接口
 */

import localforage from "localforage";

// 保存策略枚举
export const SaveStrategy = {
  DEBOUNCE: "debounce", // 防抖：小数据量，日常编辑
  IDLE: "idle", // 空闲：中等数据量
  WORKER: "worker", // Worker：大数据量（预留）
  IMMEDIATE: "immediate", // 立即：关键操作后
};

// 配置常量
const CONFIG = {
  // 数据量阈值
  SMALL_COUNT: 500, // < 500 条视为小数据量
  LARGE_COUNT: 2000, // > 2000 条视为大数据量
  SMALL_SIZE: 200 * 1024, // < 200KB 视为小数据
  LARGE_SIZE: 1024 * 1024, // > 1MB 视为大数据

  // 时间配置
  DEBOUNCE_DELAY: 2000, // 防抖延迟 2 秒
  IDLE_TIMEOUT: 5000, // 空闲回调最长等待 5 秒
  MIN_SAVE_INTERVAL: 1000, // 最小保存间隔 1 秒

  // 存储键前缀
  PROJECT_PREFIX: "project-",
  BACKUP_PREFIX: "project-backup-",

  // [V3.1.0] localStorage 配额管理
  MAX_BACKUP_COUNT: 5, // 最多保留5个备份
  BACKUP_WARNING_THRESHOLD: 0.8, // 配额使用超过80%时警告
  MAX_CONSECUTIVE_FAILURES: 3, // 最大连续失败次数后禁用备份
};

class SmartSaver {
  constructor() {
    // 状态
    this.lastSaveTime = 0;
    this.pendingIdleCallback = null;
    this.debounceTimer = null;
    this.isSaving = false;

    // [V3.1.0] 连续失败计数和禁用标记
    this.consecutiveFailures = 0;
    this.backupDisabled = false;

    // Worker 相关（预留）
    this.worker = null;
    this.useWorker = false; // 是否启用 Worker 模式

    // 回调
    this.onSaveSuccess = null;
    this.onSaveError = null;

    // 初始化页面关闭保护
    this._initBeforeUnloadHandler();
  }

  /**
   * 初始化页面关闭/刷新时的紧急保存
   */
  _initBeforeUnloadHandler() {
    // 存储待保存的数据引用
    this._pendingData = null;

    window.addEventListener("beforeunload", (event) => {
      if (this._pendingData) {
        this._emergencySave(this._pendingData);
      }
    });

    // 页面可见性变化时也尝试保存（移动端切换应用）
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden" && this._pendingData) {
        this._emergencySave(this._pendingData);
      }
    });
  }

  /**
   * 紧急同步保存到 localStorage（页面关闭时使用）
   * [V3.1.0] 增强：配额检查、LRU淘汰、失败计数
   */
  _emergencySave(data) {
    // 如果备份已被禁用（连续失败过多），跳过
    if (this.backupDisabled) {
      console.warn("[SmartSaver] 备份已禁用，跳过紧急备份");
      return;
    }

    try {
      const key = `${CONFIG.BACKUP_PREFIX}${data.jobId}`;
      const plain = this._toPlainObject(data.subtitles, data.meta);
      const jsonStr = JSON.stringify(plain);

      // [V3.1.0] 检查 localStorage 可用空间
      const storageInfo = this._getStorageInfo();
      if (storageInfo.usageRatio > CONFIG.BACKUP_WARNING_THRESHOLD) {
        console.warn(`[SmartSaver] localStorage 使用率 ${(storageInfo.usageRatio * 100).toFixed(1)}%，执行 LRU 清理`);
        this._cleanupOldBackups();
      }

      // 尝试保存
      localStorage.setItem(key, jsonStr);
      console.log("[SmartSaver] 紧急备份已保存到 localStorage");

      // 重置失败计数
      this.consecutiveFailures = 0;
    } catch (error) {
      this.consecutiveFailures++;
      console.error(`[SmartSaver] 紧急备份失败 (${this.consecutiveFailures}/${CONFIG.MAX_CONSECUTIVE_FAILURES}):`, error);

      // [V3.1.0] 如果是配额错误，尝试清理后重试一次
      if (error.name === "QuotaExceededError") {
        this._cleanupOldBackups();

        // 重试一次
        try {
          const key = `${CONFIG.BACKUP_PREFIX}${data.jobId}`;
          const plain = this._toPlainObject(data.subtitles, data.meta);
          localStorage.setItem(key, JSON.stringify(plain));
          console.log("[SmartSaver] 清理后紧急备份成功");
          this.consecutiveFailures = 0;
          return;
        } catch (retryError) {
          console.error("[SmartSaver] 清理后备份仍失败:", retryError);
        }
      }

      // 连续失败过多，禁用备份功能避免无限循环
      if (this.consecutiveFailures >= CONFIG.MAX_CONSECUTIVE_FAILURES) {
        console.error("[SmartSaver] 连续失败次数过多，禁用 localStorage 备份");
        this.backupDisabled = true;
      }
    }
  }

  /**
   * [V3.1.0] 获取 localStorage 使用情况
   */
  _getStorageInfo() {
    let totalSize = 0;
    let backupCount = 0;

    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      const value = localStorage.getItem(key);
      totalSize += (key.length + value.length) * 2; // UTF-16 编码

      if (key.startsWith(CONFIG.BACKUP_PREFIX)) {
        backupCount++;
      }
    }

    // 估算配额（大多数浏览器为 5-10MB）
    const estimatedQuota = 5 * 1024 * 1024;

    return {
      totalSize,
      backupCount,
      usageRatio: totalSize / estimatedQuota,
    };
  }

  /**
   * [V3.1.0] 清理旧备份（LRU 策略）
   */
  _cleanupOldBackups() {
    const backups = [];

    // 收集所有备份信息
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith(CONFIG.BACKUP_PREFIX)) {
        try {
          const value = localStorage.getItem(key);
          const data = JSON.parse(value);
          backups.push({
            key,
            savedAt: data.savedAt || 0,
            size: (key.length + value.length) * 2,
          });
        } catch (e) {
          // 解析失败的备份直接删除
          localStorage.removeItem(key);
        }
      }
    }

    // 按时间排序（最旧的在前）
    backups.sort((a, b) => a.savedAt - b.savedAt);

    // 删除超出限制的旧备份
    while (backups.length > CONFIG.MAX_BACKUP_COUNT) {
      const oldest = backups.shift();
      console.log(`[SmartSaver] 清理旧备份: ${oldest.key}`);
      localStorage.removeItem(oldest.key);
    }

    // 如果仍然空间不足，继续删除最旧的
    let storageInfo = this._getStorageInfo();
    while (storageInfo.usageRatio > CONFIG.BACKUP_WARNING_THRESHOLD && backups.length > 0) {
      const oldest = backups.shift();
      console.log(`[SmartSaver] 空间不足，继续清理: ${oldest.key}`);
      localStorage.removeItem(oldest.key);
      storageInfo = this._getStorageInfo();
    }
  }

  /**
   * 评估应使用的保存策略
   */
  evaluateStrategy(subtitles, changeType = "normal") {
    const count = subtitles.length;

    // 批量操作强制使用特定策略
    if (changeType === "batch" || changeType === "replace-all") {
      return count > CONFIG.LARGE_COUNT
        ? SaveStrategy.WORKER
        : SaveStrategy.IDLE;
    }

    // 根据数据量选择策略
    if (count < CONFIG.SMALL_COUNT) {
      return SaveStrategy.DEBOUNCE;
    } else if (count < CONFIG.LARGE_COUNT) {
      return SaveStrategy.IDLE;
    } else {
      // 大数据量：如果 Worker 可用则使用，否则降级到 IDLE
      return this.useWorker ? SaveStrategy.WORKER : SaveStrategy.IDLE;
    }
  }

  /**
   * 智能保存入口
   * @param {Object} data - { jobId, subtitles, meta }
   * @param {string} changeType - 'normal' | 'batch' | 'replace-all' | 'critical'
   */
  save(data, changeType = "normal") {
    if (!data.jobId) return;

    // 更新待保存数据（用于紧急备份）
    this._pendingData = data;

    // 更新内存缓存
    this._updateMemoryCache(data);

    // 关键操作立即保存
    if (changeType === "critical") {
      this._immediateSave(data);
      return;
    }

    // 评估并执行保存策略
    const strategy = this.evaluateStrategy(data.subtitles, changeType);

    switch (strategy) {
      case SaveStrategy.DEBOUNCE:
        this._debounceSave(data);
        break;
      case SaveStrategy.IDLE:
        this._idleSave(data);
        break;
      case SaveStrategy.WORKER:
        this._workerSave(data);
        break;
      default:
        this._debounceSave(data);
    }
  }

  /**
   * 立即保存（同步感知，实际异步执行）
   */
  async _immediateSave(data) {
    // 取消待执行的保存
    this._cancelPending();

    await this._doSave(data);
  }

  /**
   * 防抖保存
   */
  _debounceSave(data) {
    // 清除之前的定时器
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.debounceTimer = setTimeout(() => {
      this._doSave(data);
    }, CONFIG.DEBOUNCE_DELAY);
  }

  /**
   * 空闲时保存
   */
  _idleSave(data) {
    // 取消之前的空闲回调
    if (this.pendingIdleCallback) {
      cancelIdleCallback(this.pendingIdleCallback);
    }

    this.pendingIdleCallback = requestIdleCallback(
      () => {
        this._doSave(data);
      },
      { timeout: CONFIG.IDLE_TIMEOUT }
    );
  }

  /**
   * Worker 保存（预留实现）
   * TODO: 实现 Web Worker 异步保存
   */
  _workerSave(data) {
    // 当前降级到空闲保存
    // 未来实现：
    // 1. 懒加载 Worker
    // 2. 计算增量 Diff
    // 3. 使用 Transferable 传输数据
    // 4. Worker 内直接写 IndexedDB

    console.log("[SmartSaver] Worker 模式暂未实现，降级到 IDLE 模式");
    this._idleSave(data);
  }

  /**
   * 实际执行保存操作
   */
  async _doSave(data) {
    // 防止过于频繁的保存
    const now = Date.now();
    if (now - this.lastSaveTime < CONFIG.MIN_SAVE_INTERVAL) {
      // 延迟执行
      this._debounceSave(data);
      return;
    }

    if (this.isSaving) {
      // 已有保存进行中，延迟执行
      this._debounceSave(data);
      return;
    }

    this.isSaving = true;

    try {
      const key = `${CONFIG.PROJECT_PREFIX}${data.jobId}`;
      const plain = this._toPlainObject(data.subtitles, data.meta);

      await localforage.setItem(key, plain);

      this.lastSaveTime = Date.now();
      this._pendingData = null; // 清除待保存标记

      // 清除 localStorage 备份（已成功保存到 IndexedDB）
      this._clearBackup(data.jobId);

      if (this.onSaveSuccess) {
        this.onSaveSuccess(data.jobId);
      }
    } catch (error) {
      console.error("[SmartSaver] 保存失败:", error);

      // 保存失败时，尝试备份到 localStorage
      this._emergencySave(data);

      if (this.onSaveError) {
        this.onSaveError(error, data.jobId);
      }
    } finally {
      this.isSaving = false;
    }
  }

  /**
   * 取消所有待执行的保存
   */
  _cancelPending() {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
      this.debounceTimer = null;
    }
    if (this.pendingIdleCallback) {
      cancelIdleCallback(this.pendingIdleCallback);
      this.pendingIdleCallback = null;
    }
  }

  /**
   * 更新内存缓存
   */
  _updateMemoryCache(data) {
    // 内存缓存由 projectStore 管理，这里可以扩展
  }

  /**
   * 转换为纯对象（去除 Vue 响应式代理）
   */
  _toPlainObject(subtitles, meta) {
    return JSON.parse(
      JSON.stringify({
        subtitles,
        meta,
        savedAt: Date.now(),
      })
    );
  }

  /**
   * 清除 localStorage 备份
   */
  _clearBackup(jobId) {
    try {
      localStorage.removeItem(`${CONFIG.BACKUP_PREFIX}${jobId}`);
    } catch (e) {
      // 忽略清除失败
    }
  }

  /**
   * 从备份恢复数据
   * @returns {Object|null} 恢复的数据或 null
   */
  async restoreFromBackup(jobId) {
    // 优先从 IndexedDB 恢复
    try {
      const key = `${CONFIG.PROJECT_PREFIX}${jobId}`;
      const saved = await localforage.getItem(key);
      if (saved) {
        return saved;
      }
    } catch (error) {
      console.warn("[SmartSaver] IndexedDB 恢复失败:", error);
    }

    // 尝试从 localStorage 备份恢复
    try {
      const backupKey = `${CONFIG.BACKUP_PREFIX}${jobId}`;
      const backup = localStorage.getItem(backupKey);
      if (backup) {
        console.log("[SmartSaver] 从 localStorage 备份恢复");
        return JSON.parse(backup);
      }
    } catch (error) {
      console.warn("[SmartSaver] localStorage 恢复失败:", error);
    }

    return null;
  }

  /**
   * 强制立即保存（供外部调用）
   */
  async forceSave(data) {
    await this._immediateSave(data);
  }

  /**
   * 清理资源
   */
  destroy() {
    this._cancelPending();
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
  }
}

// 导出单例
export const smartSaver = new SmartSaver();
export default smartSaver;
