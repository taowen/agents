/**
 * browser-standalone.ts
 *
 * JS executor that runs in a standalone Hermes runtime for the "browser"
 * agent type. The server-side agent loop sends exec_js requests; this file
 * provides `executeCodeForServer()` which wraps browser host functions with
 * safety limits and status updates.
 *
 * Prompt & tool definitions are generated in browser-prompt.ts and
 * exposed via `__DEVICE_PROMPT__` for the server handshake.
 */

import "./browser-prompt"; // side-effect: sets globalThis.__DEVICE_PROMPT__

// Declare globals provided by C++ host functions (for TypeScript only)
declare global {
  var get_page: () => string;
  var click_element: (id: number) => boolean;
  var type_text: (id: number, text: string) => boolean;
  var goto_url: (url: string) => boolean;
  var scroll_page: (direction: string) => boolean;
  var go_back: () => boolean;
  var take_screenshot: () => string;
  var switch_ua: (mode: string) => string;
  var set_viewport: (width: number, height: number) => string;
  var sleep: (ms: number) => void;
  var log: (msg: string) => void;
  var ask_user: (question: string) => string;
  var update_status: (text: string) => void;
  var hide_overlay: () => void;
  var executeCodeForServer: (code: string) => {
    result: string;
    screenshots?: string[];
  };
  var __DEVICE_PROMPT__: { systemPrompt: string; tools: unknown[] };
}

const MAX_GET_PAGE_PER_EXEC = 5;
const EXEC_TIMEOUT_MS = 30_000;

/**
 * executeCodeForServer — called from DeviceConnection.java for cloud exec_js.
 * Wraps browser host functions with update_status + safety limits.
 */
function executeCodeForServer(code: string): {
  result: string;
  screenshots: string[];
} {
  const capturedScreenshots: string[] = [];
  let lastGetPageResult: string | null = null;
  let getPageCount = 0;
  const deadline = Date.now() + EXEC_TIMEOUT_MS;

  // Save originals
  const origGetPage = get_page;
  const origClickElement = click_element;
  const origTypeText = type_text;
  const origGotoUrl = goto_url;
  const origScrollPage = scroll_page;
  const origGoBack = go_back;
  const origTakeScreenshot = take_screenshot;
  const origSwitchUa = switch_ua;
  const origSetViewport = set_viewport;

  // Wrap with status updates and limits
  globalThis.get_page = function (): string {
    if (Date.now() > deadline) throw new Error("Script execution timeout");
    getPageCount++;
    if (getPageCount > MAX_GET_PAGE_PER_EXEC) {
      throw new Error(
        "get_page() called " +
          getPageCount +
          " times. Max is " +
          MAX_GET_PAGE_PER_EXEC +
          ". Return and plan next actions in a new execute_js call."
      );
    }
    update_status("read page");
    const tree = origGetPage();
    lastGetPageResult = tree;
    return tree;
  };

  globalThis.click_element = function (id: number): boolean {
    if (Date.now() > deadline) throw new Error("Script execution timeout");
    update_status("click [" + id + "]");
    return origClickElement(id);
  };

  globalThis.type_text = function (id: number, text: string): boolean {
    if (Date.now() > deadline) throw new Error("Script execution timeout");
    update_status("type [" + id + "] " + text.substring(0, 15));
    return origTypeText(id, text);
  };

  globalThis.goto_url = function (url: string): boolean {
    if (Date.now() > deadline) throw new Error("Script execution timeout");
    update_status("goto " + url.substring(0, 30));
    return origGotoUrl(url);
  };

  globalThis.scroll_page = function (direction: string): boolean {
    if (Date.now() > deadline) throw new Error("Script execution timeout");
    update_status("scroll " + direction);
    return origScrollPage(direction);
  };

  globalThis.go_back = function (): boolean {
    if (Date.now() > deadline) throw new Error("Script execution timeout");
    update_status("go back");
    return origGoBack();
  };

  globalThis.take_screenshot = function (): string {
    if (Date.now() > deadline) throw new Error("Script execution timeout");
    update_status("take screenshot");
    const b64 = origTakeScreenshot();
    if (b64.startsWith("ERROR:")) {
      return b64;
    }
    capturedScreenshots.push(b64);
    return "screenshot captured - image will be sent to you";
  };

  globalThis.switch_ua = function (mode: string): string {
    if (Date.now() > deadline) throw new Error("Script execution timeout");
    update_status("switch UA " + mode);
    return origSwitchUa(mode);
  };

  globalThis.set_viewport = function (width: number, height: number): string {
    if (Date.now() > deadline) throw new Error("Script execution timeout");
    update_status("set viewport " + width + "x" + height);
    return origSetViewport(width, height);
  };

  try {
    let result = (0, eval)(code);
    if (result === undefined && lastGetPageResult !== null) {
      result = lastGetPageResult;
    }
    return {
      result: result === undefined ? "undefined" : String(result),
      screenshots: capturedScreenshots
    };
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg.includes("'return' not in a function")) {
      return {
        result:
          "[JS Error] Top-level 'return' is not allowed. The code runs in global scope — use the last expression as the result instead of 'return'.",
        screenshots: capturedScreenshots
      };
    }
    return {
      result: "[JS Error] " + msg,
      screenshots: capturedScreenshots
    };
  } finally {
    // Restore originals
    globalThis.get_page = origGetPage;
    globalThis.click_element = origClickElement;
    globalThis.type_text = origTypeText;
    globalThis.goto_url = origGotoUrl;
    globalThis.scroll_page = origScrollPage;
    globalThis.go_back = origGoBack;
    globalThis.take_screenshot = origTakeScreenshot;
    globalThis.switch_ua = origSwitchUa;
    globalThis.set_viewport = origSetViewport;
  }
}

globalThis.executeCodeForServer = executeCodeForServer;
