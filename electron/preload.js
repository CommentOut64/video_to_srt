const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("anchorfluxShell", {
  mediaCapabilities: null,
  debugFlags: null,
  getMediaCapabilities() {
    return this.mediaCapabilities || {
      activeGpuPreference: "unknown",
      h264DirectPlay: true,
      hevcDirectPlay: false,
      fallbackOrder: ["discrete", "integrated", "software_proxy"],
      gpuFeatureStatus: {},
    };
  },
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

ipcRenderer.invoke("shell:get-media-capabilities")
  .then((payload) => {
    if (window.anchorfluxShell) {
      window.anchorfluxShell.mediaCapabilities = payload || null;
    }
  })
  .catch(() => {
    if (window.anchorfluxShell) {
      window.anchorfluxShell.mediaCapabilities = null;
    }
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
