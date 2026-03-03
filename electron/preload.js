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
});
