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

import "./prompt"; // side-effect: sets globalThis.__DEVICE_PROMPT__

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
  var sleep: (ms: number) => void;
  var log: (msg: string) => void;
  var update_status: (text: string) => void;
  var ask_user: (question: string) => string;
  var hide_overlay: () => void;
  var executeCodeForServer: (code: string) => {
    result: string;
    screenshots?: string[];
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
    update_status("read screen");
    const tree = origGetScreen();
    if (opts.onGetScreen) opts.onGetScreen(tree);
    return tree;
  };

  globalThis.take_screenshot = function (): string {
    if (Date.now() > opts.deadline) throw new Error("Script execution timeout");
    update_status("take screenshot");
    const b64 = origTakeScreenshot();
    if (b64.startsWith("ERROR:")) {
      return b64;
    }
    opts.onScreenshot(b64);
    return "screenshot captured - image will be sent to you";
  };

  globalThis.click = function (
    target: string | { desc?: string; x?: number; y?: number }
  ): boolean {
    const desc =
      typeof target === "string"
        ? target
        : target.desc || "(" + target.x + "," + target.y + ")";
    update_status("click " + desc);
    return origClick(target);
  };

  globalThis.long_click = function (
    target: string | { desc?: string; x?: number; y?: number }
  ): boolean {
    const desc =
      typeof target === "string"
        ? target
        : target.desc || "(" + target.x + "," + target.y + ")";
    update_status("long click " + desc);
    return origLongClick(target);
  };

  globalThis.type_text = function (text: string): boolean {
    update_status("type " + text.substring(0, 15));
    return origTypeText(text);
  };

  globalThis.launch_app = function (name: string): string {
    update_status("launch " + name);
    return origLaunchApp(name);
  };

  globalThis.scroll = function (direction: string): boolean {
    update_status("scroll " + direction);
    return origScroll(direction);
  };

  globalThis.scroll_element = function (
    text: string,
    direction: string
  ): string {
    update_status("scroll " + text + " " + direction);
    return origScrollElement(text, direction);
  };

  globalThis.press_home = function (): boolean {
    update_status("press home");
    return origPressHome();
  };

  globalThis.press_back = function (): boolean {
    update_status("press back");
    return origPressBack();
  };

  globalThis.press_recents = function (): boolean {
    update_status("press recents");
    return origPressRecents();
  };

  globalThis.show_notifications = function (): boolean {
    update_status("show notifications");
    return origShowNotifications();
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
} {
  const capturedScreenshots: string[] = [];
  let lastGetScreenResult: string | null = null;

  const wrapped = wrapHostFunctions({
    deadline: Date.now() + EXEC_TIMEOUT_MS,
    maxGetScreen: MAX_GET_SCREEN_PER_EXEC,
    onGetScreen(tree) {
      lastGetScreenResult = tree;
    },
    onScreenshot(b64) {
      capturedScreenshots.push(b64);
    }
  });

  try {
    let result = (0, eval)(code);
    if (result === undefined && lastGetScreenResult !== null) {
      result = lastGetScreenResult;
    }
    return {
      result: result === undefined ? "undefined" : String(result),
      screenshots: capturedScreenshots
    };
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    return {
      result: "[JS Error] " + msg,
      screenshots: capturedScreenshots
    };
  } finally {
    wrapped.restore();
  }
}

globalThis.executeCodeForServer = executeCodeForServer;
