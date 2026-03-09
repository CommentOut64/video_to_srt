const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("anchorfluxShell", {
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
  onRuntimeInfo(callback) {
    if (typeof callback !== "function") {
      return () => {};
    }
    const listener = (_, payload) => callback(payload);
    ipcRenderer.on("shell:runtime-info", listener);
    return () => {
      ipcRenderer.removeListener("shell:runtime-info", listener);
    };
  },
  async getRuntimeInfo() {
    return ipcRenderer.invoke("shell:get-runtime-info");
  },
});
