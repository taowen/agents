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
      scroll: "screen:scroll"
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
