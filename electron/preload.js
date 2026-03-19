const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("anchorfluxShell", {
  debugFlags: null,
  getDebugFlags() {
    return this.debugFlags || {
      nativeContextMenuEnabled: false,
      devToolsEnabled: false,
      rendererProfilingEnabled: false,
      openDevToolsOnLaunch: false,
    };
  },
  openDevTools() {
    return ipcRenderer.invoke("shell:open-devtools");
  },
  getRuntimeDiagnostics() {
    return ipcRenderer.invoke("shell:get-runtime-diagnostics");
  },
  onStatusChange(callback) {
    if (typeof callback !== "function") {
      return () => {};
    }
    const listener = (_, payload) => callback(payload);
    ipcRenderer.on("shell:status", listener);
    return () => {
      ipcRenderer.removeListener("shell:status", listener);
    };
  },
});

ipcRenderer.invoke("shell:get-debug-flags")
  .then((payload) => {
    if (window.anchorfluxShell) {
      window.anchorfluxShell.debugFlags = payload || null;
    }
  })
  .catch(() => {
    if (window.anchorfluxShell) {
      window.anchorfluxShell.debugFlags = null;
    }
  });
