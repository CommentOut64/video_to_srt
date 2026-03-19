const path = require("path");
const {
  app,
  BrowserWindow,
  ipcMain,
  nativeTheme,
  shell,
} = require("electron");
const fs = require("fs");
const http = require("http");
const https = require("https");
const {
  resolveShellLogDir,
  buildTimestampedLogPath,
  ensureLogDir,
  safeAppendLog,
} = require("./shell_logging");
const {
  resolveShellRuntimeConfig,
} = require("./shell_runtime_config");

const BACKEND_BASE_URL =
  process.env.ANCHORFLUX_BACKEND_URL || "http://127.0.0.1:8000";
const READY_ENDPOINT = `${BACKEND_BASE_URL.replace(/\/+$/, "")}/api/system/ready`;
const SHUTDOWN_ENDPOINT = `${BACKEND_BASE_URL.replace(/\/+$/, "")}/api/system/shutdown`;
const READY_TIMEOUT_MS = Number(process.env.ANCHORFLUX_READY_TIMEOUT_MS || 60000);
const READY_INTERVAL_MS = 1000;
const SHUTDOWN_REQUEST_TIMEOUT_MS = Number(
  process.env.ANCHORFLUX_SHUTDOWN_REQUEST_TIMEOUT_MS || 4000
);
const SHELL_FORCE_EXIT_MS = Number(
  process.env.ANCHORFLUX_SHELL_FORCE_EXIT_MS || 15000
);
const WINDOW_BG_COLOR = "#0b1220";
const SHELL_RUNTIME_CONFIG = resolveShellRuntimeConfig({ env: process.env });
const DEBUG_FLAGS = SHELL_RUNTIME_CONFIG.debugFlags;
const PERFORMANCE_FLAGS = SHELL_RUNTIME_CONFIG.performanceFlags;
const SHELL_LOG_DIR = resolveShellLogDir({
  env: process.env,
  execPath: process.execPath,
  cwd: process.cwd(),
});
const SHELL_MAIN_LOG_PATH =
  process.env.ANCHORFLUX_SHELL_MAIN_LOG ||
  buildTimestampedLogPath(SHELL_LOG_DIR, "electron-main");

let mainWindow = null;
let isBackendShutdownTriggered = false;
let isAppExitInProgress = false;
let appOrigin = null;

try {
  ensureLogDir(SHELL_LOG_DIR);
} catch (_) {
  // 日志目录创建失败时不阻断 UI 启动，避免排障工具反向造成不可用。
}

function writeShellLog(level, message, error = null) {
  try {
    safeAppendLog(SHELL_MAIN_LOG_PATH, level, message, error);
  } catch (_) {
    // 日志写入失败时保持静默，避免递归报错。
  }
}

function buildDebugFlags() {
  return {
    nativeContextMenuEnabled: Boolean(DEBUG_FLAGS.nativeContextMenuEnabled),
    devToolsEnabled: Boolean(DEBUG_FLAGS.devToolsEnabled),
    rendererProfilingEnabled: Boolean(DEBUG_FLAGS.rendererProfilingEnabled),
    openDevToolsOnLaunch: Boolean(DEBUG_FLAGS.openDevToolsOnLaunch),
  };
}

async function collectRuntimeDiagnostics() {
  const diagnostics = {
    timestamp: Date.now(),
    debugFlags: buildDebugFlags(),
    performanceFlags: {
      antiThrottlingEnabled: Boolean(PERFORMANCE_FLAGS.antiThrottlingEnabled),
      metricsLoggingEnabled: Boolean(PERFORMANCE_FLAGS.metricsLoggingEnabled),
      diagnosticsIntervalMs: PERFORMANCE_FLAGS.diagnosticsIntervalMs,
    },
    appMetrics:
      typeof app.getAppMetrics === "function" ? app.getAppMetrics() : [],
    mainWindow: null,
  };

  if (!mainWindow || mainWindow.isDestroyed()) {
    return diagnostics;
  }

  const webContents = mainWindow.webContents;
  diagnostics.mainWindow = {
    id: webContents.id,
    url: webContents.getURL(),
    isCrashed:
      typeof webContents.isCrashed === "function" ? webContents.isCrashed() : false,
    isDevToolsOpened:
      typeof webContents.isDevToolsOpened === "function"
        ? webContents.isDevToolsOpened()
        : false,
  };

  return diagnostics;
}

try {
  appOrigin = new URL(BACKEND_BASE_URL).origin;
} catch (_) {
  appOrigin = null;
}

function setupDiagnostics() {
  process.on("uncaughtException", (error) => {
    writeShellLog("ERROR", "主进程 uncaughtException", error);
  });
  process.on("unhandledRejection", (reason) => {
    writeShellLog("ERROR", "主进程 unhandledRejection", reason);
  });

  app.on("render-process-gone", (_, contents, details) => {
    writeShellLog(
      "ERROR",
      `render-process-gone id=${contents?.id ?? "unknown"} reason=${details?.reason ?? "unknown"} exitCode=${details?.exitCode ?? "unknown"}`
    );
  });

  app.on("child-process-gone", (_, details) => {
    writeShellLog(
      "ERROR",
      `child-process-gone type=${details?.type ?? "unknown"} reason=${details?.reason ?? "unknown"} exitCode=${details?.exitCode ?? "unknown"}`
    );
  });

  app.on("web-contents-created", (_, contents) => {
    contents.on(
      "did-fail-load",
      (_, errorCode, errorDescription, validatedURL, isMainFrame) => {
        if (!isMainFrame) {
          return;
        }
        writeShellLog(
          "ERROR",
          `did-fail-load id=${contents.id} code=${errorCode} desc=${errorDescription} url=${validatedURL}`
        );
      }
    );

    contents.on("console-message", (_, level, message, line, sourceId) => {
      if (level < 2) {
        return;
      }
      writeShellLog(
        "WARN",
        `console-message id=${contents.id} level=${level} source=${sourceId}:${line} msg=${message}`
      );
    });

    contents.on("unresponsive", () => {
      writeShellLog("WARN", `webContents unresponsive id=${contents.id}`);
    });
    contents.on("responsive", () => {
      writeShellLog("INFO", `webContents responsive id=${contents.id}`);
    });
  });
}

function isInternalUrl(rawUrl) {
  if (!rawUrl || !appOrigin) {
    return false;
  }
  try {
    return new URL(rawUrl).origin === appOrigin;
  } catch (_) {
    return false;
  }
}

function openExternalUrl(rawUrl) {
  if (!rawUrl || isInternalUrl(rawUrl)) {
    return;
  }
  shell.openExternal(rawUrl).catch((error) => {
    writeShellLog("WARN", `打开外部链接失败: ${rawUrl}`, error);
  });
}

function requestBackendShutdown() {
  if (isBackendShutdownTriggered) {
    return Promise.resolve(true);
  }
  isBackendShutdownTriggered = true;
  return new Promise((resolve) => {
    let isResolved = false;
    const done = (success) => {
      if (isResolved) return;
      isResolved = true;
      resolve(Boolean(success));
    };

    try {
      const target = new URL(SHUTDOWN_ENDPOINT);
      const payload = JSON.stringify({
        cleanup_temp: false,
        force: false,
      });
      const client = target.protocol === "https:" ? https : http;
      const req = client.request(
        {
          protocol: target.protocol,
          hostname: target.hostname,
          port: target.port || (target.protocol === "https:" ? 443 : 80),
          path: `${target.pathname}${target.search}`,
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Content-Length": Buffer.byteLength(payload),
          },
        },
        (response) => {
          response.resume();
          done(true);
        }
      );
      req.on("error", () => done(false));
      req.setTimeout(SHUTDOWN_REQUEST_TIMEOUT_MS, () => {
        req.destroy(new Error("shutdown request timeout"));
        done(false);
      });
      req.write(payload);
      req.end();
    } catch (_) {
      // 关闭链路不抛异常，避免阻断壳进程退出。
      done(false);
    }
  });
}

function exitAppWithGuard() {
  if (isAppExitInProgress) {
    return;
  }
  isAppExitInProgress = true;

  const forceExitTimer = setTimeout(() => {
    app.exit(0);
  }, SHELL_FORCE_EXIT_MS);

  requestBackendShutdown()
    .catch(() => false)
    .finally(() => {
      clearTimeout(forceExitTimer);
      app.exit(0);
    });
}

function resolveWindowIconPath() {
  const candidates = [
    path.join(process.resourcesPath || "", "icon.ico"),
    path.join(__dirname, "build", "icon.ico"),
    path.join(__dirname, "icon.ico"),
  ];
  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return null;
}

function focusMainWindow() {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return false;
  }
  if (mainWindow.isMinimized()) {
    mainWindow.restore();
  }
  if (!mainWindow.isVisible()) {
    mainWindow.show();
  }
  mainWindow.focus();
  return true;
}

function pushStatus(status, message) {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  mainWindow.webContents.send("shell:status", {
    status,
    message,
    timestamp: Date.now(),
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitBackendReady() {
  const deadline = Date.now() + READY_TIMEOUT_MS;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(READY_ENDPOINT, {
        method: "GET",
        cache: "no-store",
      });
      if (response.ok) {
        const payload = await response.json();
        if (payload && payload.success === true && payload.ready === true) {
          pushStatus("ready", "后端已就绪，正在打开主界面...");
          return true;
        }
      }
    } catch (_) {
      // 启动期连接失败属于预期，继续轮询
    }
    pushStatus("waiting", "后端启动中，请稍候...");
    await sleep(READY_INTERVAL_MS);
  }
  return false;
}

function createMainWindow() {
  const iconPath = resolveWindowIconPath();
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 960,
    minWidth: 1280,
    minHeight: 720,
    show: false,
    backgroundColor: WINDOW_BG_COLOR,
    ...(iconPath ? { icon: iconPath } : {}),
    ...(process.platform === "win32" ? { titleBarStyle: "default" } : {}),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      devTools: DEBUG_FLAGS.devToolsEnabled,
    },
  });

  mainWindow.removeMenu();
  mainWindow.loadFile(path.join(__dirname, "loading.html"));
  writeShellLog("INFO", "已加载 loading.html");

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (isInternalUrl(url)) {
      return { action: "allow" };
    }
    openExternalUrl(url);
    return { action: "deny" };
  });

  mainWindow.webContents.on("will-navigate", (event, url) => {
    if (isInternalUrl(url)) {
      return;
    }
    event.preventDefault();
    openExternalUrl(url);
  });

  mainWindow.on("closed", () => {
    writeShellLog("INFO", "主窗口已关闭");
    mainWindow = null;
  });

  mainWindow.webContents.on("before-input-event", (event, input) => {
    if (!DEBUG_FLAGS.devToolsEnabled) {
      return;
    }

    const normalizedKey = String(input?.key || "").toLowerCase();
    const shouldToggleDevTools =
      normalizedKey === "f12"
      || ((input?.control || input?.meta) && input?.shift && normalizedKey === "i");

    if (!shouldToggleDevTools) {
      return;
    }

    event.preventDefault();
    if (mainWindow.webContents.isDevToolsOpened()) {
      mainWindow.webContents.closeDevTools();
      return;
    }
    mainWindow.webContents.openDevTools({ mode: "detach", activate: true });
  });
}

async function startShell() {
  if (focusMainWindow()) {
    writeShellLog("INFO", "检测到现有窗口，执行聚焦");
    return;
  }
  createMainWindow();
  const isReady = await waitBackendReady();
  if (!isReady) {
    writeShellLog("ERROR", "等待后端就绪超时");
    pushStatus(
      "error",
      "后端启动超时，请关闭后重试；若问题持续，请检查日志。"
    );
    mainWindow.show();
    return;
  }

  try {
    writeShellLog("INFO", `准备加载主界面 URL: ${BACKEND_BASE_URL}`);
    await mainWindow.loadURL(BACKEND_BASE_URL);
    writeShellLog("INFO", `主界面加载完成 URL: ${BACKEND_BASE_URL}`);
    if (
      DEBUG_FLAGS.devToolsEnabled
      && DEBUG_FLAGS.openDevToolsOnLaunch
      && !mainWindow.webContents.isDevToolsOpened()
    ) {
      mainWindow.webContents.openDevTools({
        mode: "detach",
        activate: false,
      });
    }
    focusMainWindow();
  } catch (error) {
    writeShellLog("ERROR", "加载主界面失败", error);
    pushStatus(
      "error",
      `加载主界面失败：${error instanceof Error ? error.message : String(error)}`
    );
    mainWindow.show();
  }
}

const gotSingleInstanceLock = app.requestSingleInstanceLock();
setupDiagnostics();
ipcMain.handle("shell:get-debug-flags", async () => buildDebugFlags());
ipcMain.handle("shell:get-runtime-diagnostics", async () => collectRuntimeDiagnostics());
ipcMain.handle("shell:open-devtools", async () => {
  if (!mainWindow || mainWindow.isDestroyed() || !DEBUG_FLAGS.devToolsEnabled) {
    return { success: false, reason: "devtools-disabled" };
  }

  mainWindow.webContents.openDevTools({ mode: "detach", activate: true });
  return { success: true };
});
writeShellLog("INFO", `Shell 启动: backend=${BACKEND_BASE_URL}`);

if (!gotSingleInstanceLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (!focusMainWindow() && app.isReady()) {
      startShell();
    }
  });
  app.whenReady().then(() => {
    nativeTheme.themeSource = "dark";
    writeShellLog("INFO", `调试标志: ${JSON.stringify(buildDebugFlags())}`);
    startShell();
  });
}

app.on("window-all-closed", () => {
  if (!gotSingleInstanceLock) {
    app.exit(0);
    return;
  }
  exitAppWithGuard();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    startShell();
  } else {
    focusMainWindow();
  }
});
