import AccessibilityBridge from "./NativeAccessibilityBridge";

const MAX_GET_SCREEN_PER_EXEC = 5;
const TIMEOUT_MS = 30_000;

let _actionLog: string[] = [];
let _getScreenCount = 0;
let _deadline = 0;

function appendLog(entry: string) {
  _actionLog.push(entry);
}

function checkTimeout() {
  if (Date.now() > _deadline) {
    throw new Error("Script execution timeout (" + TIMEOUT_MS / 1000 + "s)");
  }
}

// --- Register globals ---

(globalThis as any).get_screen = function get_screen(): string {
  checkTimeout();
  _getScreenCount++;
  if (_getScreenCount > MAX_GET_SCREEN_PER_EXEC) {
    throw new Error(
      "get_screen() called " +
        _getScreenCount +
        " times in one execute_js. Max is " +
        MAX_GET_SCREEN_PER_EXEC +
        ". Return result and plan next actions in a new execute_js call."
    );
  }
  const tree = AccessibilityBridge.getScreen();
  appendLog("[get_screen] (" + tree.length + " chars)");
  return tree;
};

(globalThis as any).click = function click(
  target: string | { desc?: string; x?: number; y?: number }
): boolean {
  checkTimeout();
  if (typeof target === "string") {
    const result = AccessibilityBridge.clickByText(target);
    appendLog('[click] text="' + target + '" -> ' + result);
    return result;
  }
  if (target && typeof target === "object") {
    if (target.desc !== undefined) {
      const result = AccessibilityBridge.clickByDesc(target.desc);
      appendLog('[click] desc="' + target.desc + '" -> ' + result);
      return result;
    }
    if (target.x !== undefined && target.y !== undefined) {
      const result = AccessibilityBridge.clickByCoords(target.x, target.y);
      appendLog(
        "[click] coords=(" + target.x + "," + target.y + ") -> " + result
      );
      return result;
    }
  }
  appendLog("[click] Error: invalid argument");
  return false;
};

(globalThis as any).long_click = function long_click(
  target: string | { desc?: string; x?: number; y?: number }
): boolean {
  checkTimeout();
  if (typeof target === "string") {
    const result = AccessibilityBridge.longClickByText(target);
    appendLog('[long_click] text="' + target + '" -> ' + result);
    return result;
  }
  if (target && typeof target === "object") {
    if (target.desc !== undefined) {
      const result = AccessibilityBridge.longClickByDesc(target.desc);
      appendLog('[long_click] desc="' + target.desc + '" -> ' + result);
      return result;
    }
    if (target.x !== undefined && target.y !== undefined) {
      const result = AccessibilityBridge.longClickByCoords(target.x, target.y);
      appendLog(
        "[long_click] coords=(" + target.x + "," + target.y + ") -> " + result
      );
      return result;
    }
  }
  appendLog("[long_click] Error: invalid argument");
  return false;
};

(globalThis as any).scroll = function scroll(direction: string): boolean {
  checkTimeout();
  const result = AccessibilityBridge.scrollScreen(direction);
  appendLog("[scroll] " + direction + " -> " + result);
  return result;
};

(globalThis as any).scroll_element = function scroll_element(
  text: string,
  direction: string
): string {
  checkTimeout();
  const result = AccessibilityBridge.scrollElement(text, direction);
  appendLog('[scroll_element] "' + text + '" ' + direction + " -> " + result);
  return result;
};

(globalThis as any).type_text = function type_text(text: string): boolean {
  checkTimeout();
  const result = AccessibilityBridge.typeText(text);
  appendLog('[type_text] "' + text + '" -> ' + result);
  return result;
};

(globalThis as any).press_home = function press_home(): boolean {
  checkTimeout();
  const result = AccessibilityBridge.pressHome();
  appendLog("[press_home] -> " + result);
  return result;
};

(globalThis as any).press_back = function press_back(): boolean {
  checkTimeout();
  const result = AccessibilityBridge.pressBack();
  appendLog("[press_back] -> " + result);
  return result;
};

(globalThis as any).press_recents = function press_recents(): boolean {
  checkTimeout();
  const result = AccessibilityBridge.pressRecents();
  appendLog("[press_recents] -> " + result);
  return result;
};

(globalThis as any).show_notifications =
  function show_notifications(): boolean {
    checkTimeout();
    const result = AccessibilityBridge.showNotifications();
    appendLog("[show_notifications] -> " + result);
    return result;
  };

(globalThis as any).sleep = function sleep(ms: number): void {
  checkTimeout();
  AccessibilityBridge.sleepMs(ms);
  appendLog("[sleep] " + ms + "ms");
};

(globalThis as any).log = function log(msg: string): void {
  appendLog("[log] " + msg);
};

(globalThis as any).launch_app = function launch_app(name: string): string {
  checkTimeout();
  const result = AccessibilityBridge.launchApp(name);
  appendLog('[launch_app] "' + name + '" -> ' + result);
  return result;
};

(globalThis as any).list_apps = function list_apps(): string {
  checkTimeout();
  const result = AccessibilityBridge.listApps();
  appendLog("[list_apps] returned " + result.split("\n").length + " apps");
  return result;
};

// --- executeCode entry point ---

export function executeCode(code: string): string {
  _actionLog = [];
  _getScreenCount = 0;
  _deadline = Date.now() + TIMEOUT_MS;

  try {
    // indirect eval to execute in global scope
    const result = (0, eval)(code);
    const resultStr = result === undefined ? "undefined" : String(result);
    if (_actionLog.length > 0) {
      _actionLog.push("[Script returned] " + resultStr);
      return _actionLog.join("\n");
    }
    return resultStr;
  } catch (e: any) {
    const error = "[JS Error] " + (e.message || String(e));
    if (_actionLog.length > 0) {
      _actionLog.push(error);
      return _actionLog.join("\n");
    }
    return error;
  }
}
