const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("workWithWindows", {
  /** Simple ping — proves the preload bridge is working */
  ping: () => "pong",

  /** Platform info */
  platform: process.platform,

  /** Screen control — maps action names to IPC channels */
  screenControl: (params) => {
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
      minimize_window: "window:minimize-window",
      maximize_window: "window:maximize-window",
      restore_window: "window:restore-window",
      window_screenshot: "window:screenshot"
    };
    const channel = channelMap[params.action];
    if (!channel) {
      return Promise.resolve({
        success: false,
        error: `Unknown action: ${params.action}`
      });
    }
    const { action, ...rest } = params;
    return ipcRenderer.invoke(channel, rest);
  }
});
