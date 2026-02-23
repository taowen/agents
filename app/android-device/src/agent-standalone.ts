/**
 * agent-standalone.ts
 *
 * JS executor that runs in a standalone Hermes runtime (inside the
 * AccessibilityService process). The server-side agent loop sends
 * exec_js requests; this file provides `executeCodeForServer()` which
 * wraps host functions with safety limits and status updates.
 *
 * Prompt & tool definitions are still generated in prompt.ts and
 * exposed via `__DEVICE_PROMPT__` for the server handshake.
 */

import "./app-prompt"; // side-effect: sets globalThis.__DEVICE_PROMPT__

// Declare globals provided by C++ host functions (for TypeScript only)
declare global {
  var get_screen: () => string;
  var take_screenshot: () => string;
  var click: (
    target: string | { desc?: string; x?: number; y?: number }
  ) => boolean;
  var long_click: (
    target: string | { desc?: string; x?: number; y?: number }
  ) => boolean;
  var scroll: (direction: string) => boolean;
  var scroll_element: (text: string, direction: string) => string;
  var type_text: (text: string) => boolean;
  var press_home: () => boolean;
  var press_back: () => boolean;
  var press_recents: () => boolean;
  var show_notifications: () => boolean;
  var launch_app: (name: string) => string;
  var list_apps: () => string;
  var speak: (text: string) => boolean;
  var sleep: (ms: number) => void;
  var log: (msg: string) => void;
  var update_status: (text: string) => void;
  var ask_user: (question: string) => string;
  var hide_overlay: () => void;
  var executeCodeForServer: (code: string) => {
    result: string;
    screenshots?: string[];
    executionLog?: Array<{ fn: string; args: string; result: string }>;
  };
  var __DEVICE_PROMPT__: { systemPrompt: string; tools: unknown[] };
}

const MAX_GET_SCREEN_PER_EXEC = 5;
const EXEC_TIMEOUT_MS = 30_000;

/**
 * Save and wrap all host functions with update_status + safety limits.
 * Returns a cleanup function that restores originals.
 */
function wrapHostFunctions(opts: {
  deadline: number;
  maxGetScreen: number;
  onGetScreen?: (tree: string) => void;
  onScreenshot: (b64: string) => void;
  onAction?: (entry: { fn: string; args: string; result: string }) => void;
}): { getScreenCount: () => number; restore: () => void } {
  let getScreenCount = 0;

  const origGetScreen = get_screen;
  const origTakeScreenshot = take_screenshot;
  const origClick = click;
  const origLongClick = long_click;
  const origTypeText = type_text;
  const origLaunchApp = launch_app;
  const origScroll = scroll;
  const origScrollElement = scroll_element;
  const origPressHome = press_home;
  const origPressBack = press_back;
  const origPressRecents = press_recents;
  const origShowNotifications = show_notifications;
  const origSpeak = speak;

  globalThis.get_screen = function (): string {
    if (Date.now() > opts.deadline) throw new Error("Script execution timeout");
    getScreenCount++;
    if (getScreenCount > opts.maxGetScreen) {
      throw new Error(
        "get_screen() called " +
          getScreenCount +
          " times. Max is " +
          opts.maxGetScreen +
          ". Return and plan next actions in a new execute_js call."
      );
    }
    const tree = origGetScreen();
    opts.onAction?.({
      fn: "get_screen",
      args: "",
      result: tree.length + " chars"
    });
    if (opts.onGetScreen) opts.onGetScreen(tree);
    return tree;
  };

  globalThis.take_screenshot = function (): string {
    if (Date.now() > opts.deadline) throw new Error("Script execution timeout");
    const b64 = origTakeScreenshot();
    if (b64.startsWith("ERROR:")) {
      opts.onAction?.({ fn: "take_screenshot", args: "", result: b64 });
      return b64;
    }
    opts.onScreenshot(b64);
    opts.onAction?.({ fn: "take_screenshot", args: "", result: "captured" });
    return "screenshot captured - image will be sent to you";
  };

  globalThis.click = function (
    target: string | { desc?: string; x?: number; y?: number }
  ): boolean {
    const desc =
      typeof target === "string"
        ? target
        : target.desc || "(" + target.x + "," + target.y + ")";
    const r = origClick(target);
    opts.onAction?.({ fn: "click", args: desc, result: String(r) });
    return r;
  };

  globalThis.long_click = function (
    target: string | { desc?: string; x?: number; y?: number }
  ): boolean {
    const desc =
      typeof target === "string"
        ? target
        : target.desc || "(" + target.x + "," + target.y + ")";
    const r = origLongClick(target);
    opts.onAction?.({ fn: "long_click", args: desc, result: String(r) });
    return r;
  };

  globalThis.type_text = function (text: string): boolean {
    const r = origTypeText(text);
    opts.onAction?.({ fn: "type_text", args: text, result: String(r) });
    return r;
  };

  globalThis.launch_app = function (name: string): string {
    const r = origLaunchApp(name);
    opts.onAction?.({ fn: "launch_app", args: name, result: String(r) });
    return r;
  };

  globalThis.scroll = function (direction: string): boolean {
    const r = origScroll(direction);
    opts.onAction?.({ fn: "scroll", args: direction, result: String(r) });
    return r;
  };

  globalThis.scroll_element = function (
    text: string,
    direction: string
  ): string {
    const r = origScrollElement(text, direction);
    opts.onAction?.({
      fn: "scroll_element",
      args: text + " " + direction,
      result: String(r)
    });
    return r;
  };

  globalThis.press_home = function (): boolean {
    const r = origPressHome();
    opts.onAction?.({ fn: "press_home", args: "", result: String(r) });
    return r;
  };

  globalThis.press_back = function (): boolean {
    const r = origPressBack();
    opts.onAction?.({ fn: "press_back", args: "", result: String(r) });
    return r;
  };

  globalThis.press_recents = function (): boolean {
    const r = origPressRecents();
    opts.onAction?.({ fn: "press_recents", args: "", result: String(r) });
    return r;
  };

  globalThis.show_notifications = function (): boolean {
    const r = origShowNotifications();
    opts.onAction?.({ fn: "show_notifications", args: "", result: String(r) });
    return r;
  };

  globalThis.speak = function (text: string): boolean {
    if (Date.now() > opts.deadline) throw new Error("Script execution timeout");
    const r = origSpeak(text);
    opts.onAction?.({ fn: "speak", args: text, result: String(r) });
    return r;
  };

  return {
    getScreenCount: () => getScreenCount,
    restore() {
      globalThis.get_screen = origGetScreen;
      globalThis.take_screenshot = origTakeScreenshot;
      globalThis.click = origClick;
      globalThis.long_click = origLongClick;
      globalThis.type_text = origTypeText;
      globalThis.launch_app = origLaunchApp;
      globalThis.scroll = origScroll;
      globalThis.scroll_element = origScrollElement;
      globalThis.press_home = origPressHome;
      globalThis.press_back = origPressBack;
      globalThis.press_recents = origPressRecents;
      globalThis.show_notifications = origShowNotifications;
      globalThis.speak = origSpeak;
    }
  };
}

/**
 * executeCodeForServer — called from DeviceConnection.java for cloud exec_js.
 * Same host function wrapping (with update_status) but returns { result, screenshots[] }.
 */
function executeCodeForServer(code: string): {
  result: string;
  screenshots: string[];
  executionLog: Array<{ fn: string; args: string; result: string }>;
} {
  const capturedScreenshots: string[] = [];
  const logEntries: Array<{ fn: string; args: string; result: string }> = [];
  let lastGetScreenResult: string | null = null;

  const wrapped = wrapHostFunctions({
    deadline: Date.now() + EXEC_TIMEOUT_MS,
    maxGetScreen: MAX_GET_SCREEN_PER_EXEC,
    onGetScreen(tree) {
      lastGetScreenResult = tree;
    },
    onScreenshot(b64) {
      capturedScreenshots.push(b64);
    },
    onAction(entry) {
      logEntries.push(entry);
      update_status(entry.fn + " " + entry.args);
    }
  });

  try {
    let result = (0, eval)(code);
    if (result === undefined && lastGetScreenResult !== null) {
      result = lastGetScreenResult;
    }
    return {
      result: result === undefined ? "undefined" : String(result),
      screenshots: capturedScreenshots,
      executionLog: logEntries
    };
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg.includes("'return' not in a function")) {
      return {
        result:
          "[JS Error] Top-level 'return' is not allowed. The code runs in global scope — use the last expression as the result instead of 'return'.",
        screenshots: capturedScreenshots,
        executionLog: logEntries
      };
    }
    return {
      result: "[JS Error] " + msg,
      screenshots: capturedScreenshots,
      executionLog: logEntries
    };
  } finally {
    wrapped.restore();
  }
}

globalThis.executeCodeForServer = executeCodeForServer;
