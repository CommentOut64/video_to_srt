const path = require("path");
const { app, BrowserWindow } = require("electron");
const fs = require("fs");

const BACKEND_BASE_URL =
  process.env.ANCHORFLUX_BACKEND_URL || "http://127.0.0.1:8000";
const READY_ENDPOINT = `${BACKEND_BASE_URL.replace(/\/+$/, "")}/api/system/ready`;
const SHUTDOWN_ENDPOINT = `${BACKEND_BASE_URL.replace(/\/+$/, "")}/api/system/shutdown`;
const READY_TIMEOUT_MS = Number(process.env.ANCHORFLUX_READY_TIMEOUT_MS || 60000);
const READY_INTERVAL_MS = 1000;
const SHUTDOWN_REQUEST_TIMEOUT_MS = Number(
  process.env.ANCHORFLUX_SHUTDOWN_REQUEST_TIMEOUT_MS || 4000
);
const SHUTDOWN_RETRY_INTERVAL_MS = Number(
  process.env.ANCHORFLUX_SHUTDOWN_RETRY_INTERVAL_MS || 600
);
const SHUTDOWN_GRACE_PERIOD_MS = Number(
  process.env.ANCHORFLUX_SHUTDOWN_GRACE_PERIOD_MS || 12000
);
const WINDOW_BG_COLOR = "#0b1220";
const TITLEBAR_BG_COLOR = "#111827";
const TITLEBAR_SYMBOL_COLOR = "#e5e7eb";

let mainWindow = null;
let isBackendShutdownTriggered = false;

async function requestBackendShutdown(reason = "window-all-closed") {
  if (isBackendShutdownTriggered) {
    return;
  }
  isBackendShutdownTriggered = true;
  const deadline = Date.now() + SHUTDOWN_GRACE_PERIOD_MS;
  while (Date.now() < deadline) {
    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      SHUTDOWN_REQUEST_TIMEOUT_MS
    );
    try {
      const response = await fetch(SHUTDOWN_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        cache: "no-store",
        signal: controller.signal,
        body: JSON.stringify({
          cleanup_temp: false,
          force: false,
        }),
      });
      if (response && response.ok) {
        return;
      }
    } catch (_) {
      // 后端可能仍在启动中，稍后重试。
    } finally {
      clearTimeout(timeoutId);
    }
    await sleep(SHUTDOWN_RETRY_INTERVAL_MS);
  }
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
    ...(process.platform === "win32"
      ? {
          titleBarStyle: "hidden",
          titleBarOverlay: {
            color: TITLEBAR_BG_COLOR,
            symbolColor: TITLEBAR_SYMBOL_COLOR,
            height: 34,
          },
        }
      : {}),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  mainWindow.removeMenu();
  mainWindow.loadFile(path.join(__dirname, "loading.html"));

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

async function startShell() {
  if (focusMainWindow()) {
    return;
  }
  createMainWindow();
  const isReady = await waitBackendReady();
  if (!isReady) {
    pushStatus(
      "error",
      "后端启动超时，请关闭后重试；若问题持续，请检查日志。"
    );
    mainWindow.show();
    return;
  }

  try {
    await mainWindow.loadURL(BACKEND_BASE_URL);
    focusMainWindow();
  } catch (error) {
    pushStatus(
      "error",
      `加载主界面失败：${error instanceof Error ? error.message : String(error)}`
    );
    mainWindow.show();
  }
}

const gotSingleInstanceLock = app.requestSingleInstanceLock();

if (!gotSingleInstanceLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (!focusMainWindow() && app.isReady()) {
      startShell();
    }
  });
  app.whenReady().then(startShell);
}

app.on("window-all-closed", () => {
  if (!gotSingleInstanceLock) {
    app.exit(0);
    return;
  }
  requestBackendShutdown().finally(() => {
    app.exit(0);
  });
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    startShell();
  } else {
    focusMainWindow();
  }
});
