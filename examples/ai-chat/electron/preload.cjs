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
