const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("workWithWindows", {
  /** Simple ping — proves the preload bridge is working */
  ping: () => "pong",

  /** Platform info */
  platform: process.platform,

  /** Filesystem operations via IPC to Node.js fs */
  fileSystem: (params) => ipcRenderer.invoke("fs:op", params),

  /** Detect available Windows drives and WSL mounts */
  detectDrives: () => ipcRenderer.invoke("fs:detect-drives"),

  /** Screen control — delegates to win-automation.ts via single IPC channel */
  screenControl: (params) => ipcRenderer.invoke("screen-control", params),

  /** Run an arbitrary PowerShell command */
  executePowerShell: (params) => ipcRenderer.invoke("powershell:exec", params)
});
