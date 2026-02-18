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

  /** Screen control — maps action names to IPC channels */
  screenControl: (params) => {
    // Normalize common LLM action name variants
    const actionAliases = {
      press_key: "key_press",
      keypress: "key_press",
      move: "mouse_move",
      move_mouse: "mouse_move",
      focus: "focus_window",
      activate: "focus_window",
      activate_window: "focus_window",
      resize: "resize_window",
      move_window: "resize_window",
      minimize: "minimize_window",
      maximize: "maximize_window",
      restore: "restore_window",
      list: "list_windows",
      double_click: "click"
    };
    const originalAction = params.action;
    params = {
      ...params,
      action: actionAliases[params.action] || params.action
    };
    if (originalAction === "double_click") params.doubleClick = true;
    const channelMap = {
      screenshot: "screen:screenshot",
      click: "screen:mouse-click",
      mouse_move: "screen:mouse-move",
      type: "screen:keyboard-type",
      key_press: "screen:key-press",
      scroll: "screen:scroll",
      list_windows: "window:list-windows",
      focus_window: "window:focus-window",
      resize_window: "window:resize-window",
      minimize_window: "window:set-state",
      maximize_window: "window:set-state",
      restore_window: "window:set-state",
      window_screenshot: "window:screenshot"
    };
    const stateMap = {
      minimize_window: "minimize",
      maximize_window: "maximize",
      restore_window: "restore"
    };
    const channel = channelMap[params.action];
    if (!channel) {
      return Promise.resolve({
        success: false,
        error: `Unknown action: ${params.action}`
      });
    }
    const { action, ...rest } = params;
    const state = stateMap[action];
    if (state) rest.state = state;
    return ipcRenderer.invoke(channel, rest);
  }
});
