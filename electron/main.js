const path = require("path");
const { app, BrowserWindow, nativeTheme, shell } = require("electron");
const fs = require("fs");
const http = require("http");
const https = require("https");

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

let mainWindow = null;
let isBackendShutdownTriggered = false;
let isAppExitInProgress = false;
let appOrigin = null;

try {
  appOrigin = new URL(BACKEND_BASE_URL).origin;
} catch (_) {
  appOrigin = null;
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
  shell.openExternal(rawUrl).catch(() => {});
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
    },
  });

  mainWindow.removeMenu();
  mainWindow.loadFile(path.join(__dirname, "loading.html"));

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
  app.whenReady().then(() => {
    nativeTheme.themeSource = "dark";
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
