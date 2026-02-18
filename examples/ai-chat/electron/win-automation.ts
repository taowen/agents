/**
 * Windows desktop automation via PowerShell scripts.
 * Zero Electron dependencies — uses only Node.js built-ins.
 */
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";

const execFileAsync = promisify(execFile);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCRIPTS_DIR = path.join(__dirname, "scripts");

// ---- Types ----

export interface ScreenControlParams {
  action: string;
  x?: number;
  y?: number;
  text?: string;
  key?: string;
  modifiers?: string[];
  button?: string;
  doubleClick?: boolean;
  direction?: string;
  amount?: number;
  handle?: number;
  title?: string;
  width?: number;
  height?: number;
}

export interface ScreenControlResult {
  success: boolean;
  error?: string;
  width?: number;
  height?: number;
  base64?: string;
  action?: string;
  windows?: Array<Record<string, unknown>>;
  message?: string;
  [key: string]: unknown;
}

// ---- Virtual key code mapping ----

const VK_MAP: Record<string, number> = {
  enter: 0x0d,
  return: 0x0d,
  tab: 0x09,
  escape: 0x1b,
  esc: 0x1b,
  backspace: 0x08,
  delete: 0x2e,
  insert: 0x2d,
  home: 0x24,
  end: 0x23,
  pageup: 0x21,
  pagedown: 0x22,
  up: 0x26,
  down: 0x28,
  left: 0x25,
  right: 0x27,
  space: 0x20,
  f1: 0x70,
  f2: 0x71,
  f3: 0x72,
  f4: 0x73,
  f5: 0x74,
  f6: 0x75,
  f7: 0x76,
  f8: 0x77,
  f9: 0x78,
  f10: 0x79,
  f11: 0x7a,
  f12: 0x7b,
  printscreen: 0x2c,
  capslock: 0x14,
  numlock: 0x90,
  scrolllock: 0x91,
  pause: 0x13,
  ctrl: 0xa2,
  control: 0xa2,
  lctrl: 0xa2,
  rctrl: 0xa3,
  shift: 0xa0,
  lshift: 0xa0,
  rshift: 0xa1,
  alt: 0xa4,
  lalt: 0xa4,
  ralt: 0xa5,
  win: 0x5b,
  lwin: 0x5b,
  rwin: 0x5c
};

function getVkCode(key: string): number | null {
  const lower = key.toLowerCase();
  if (VK_MAP[lower] !== undefined) return VK_MAP[lower];
  if (key.length === 1) {
    const code = key.toUpperCase().charCodeAt(0);
    if (code >= 0x41 && code <= 0x5a) return code;
    if (code >= 0x30 && code <= 0x39) return code;
  }
  return null;
}

// ---- PowerShell runner ----

function script(name: string): string {
  return path.join(SCRIPTS_DIR, name);
}

async function runPowerShell(
  scriptPath: string,
  opts: {
    env?: Record<string, string>;
    timeout?: number;
    maxBuffer?: number;
  } = {}
): Promise<{ stdout: string; stderr: string }> {
  const { env = {}, timeout = 5000, maxBuffer = 1024 * 1024 } = opts;
  const { stdout, stderr } = await execFileAsync(
    "powershell.exe",
    [
      "-NoProfile",
      "-NonInteractive",
      "-ExecutionPolicy",
      "Bypass",
      "-File",
      scriptPath
    ],
    { timeout, maxBuffer, windowsHide: true, env: { ...process.env, ...env } }
  );
  return { stdout: stdout.trim(), stderr: stderr.trim() };
}

// ---- Coordinate offset state ----
// After window_screenshot, we remember the window's screen position so
// click/move/scroll coordinates (which are window-relative from the LLM's
// perspective) get translated to desktop-absolute coordinates automatically.
// A full-desktop screenshot clears the offset (coordinates become absolute).
let lastWindowOffset: { left: number; top: number } | null = null;

/** Convert window-relative coords to desktop-absolute when offset is known */
function toDesktopCoords(x: number, y: number): { x: number; y: number } {
  if (lastWindowOffset) {
    return { x: x + lastWindowOffset.left, y: y + lastWindowOffset.top };
  }
  return { x, y };
}

// ---- Screen handlers ----

async function handleScreenshot(): Promise<ScreenControlResult> {
  try {
    // Full desktop screenshot — coordinates are absolute, clear any window offset
    lastWindowOffset = null;
    const { stdout } = await runPowerShell(script("screen-screenshot.ps1"), {
      timeout: 15000,
      maxBuffer: 20 * 1024 * 1024
    });
    const lines = stdout.split(/\r?\n/);
    const [width, height] = lines[0].split("x").map(Number);
    const base64 = lines.slice(1).join("");
    return { success: true, width, height, base64 };
  } catch (err: any) {
    return { success: false, error: err.message };
  }
}

async function handleMouseClick(
  params: ScreenControlParams
): Promise<ScreenControlResult> {
  try {
    const { x, y, button = "left", doubleClick = false } = params;
    const abs = toDesktopCoords(x!, y!);
    const flags: Record<string, { down: number; up: number }> = {
      left: { down: 0x0002, up: 0x0004 },
      right: { down: 0x0008, up: 0x0010 },
      middle: { down: 0x0020, up: 0x0040 }
    };
    const f = flags[button!] || flags.left;
    await runPowerShell(script("screen-click.ps1"), {
      env: {
        X: String(Math.round(abs.x)),
        Y: String(Math.round(abs.y)),
        DOWN_FLAG: String(f.down),
        UP_FLAG: String(f.up),
        CLICK_COUNT: String(doubleClick ? 2 : 1)
      }
    });
    return { success: true, action: "click", x, y, button, doubleClick };
  } catch (err: any) {
    return { success: false, error: err.message };
  }
}

async function handleMouseMove(
  params: ScreenControlParams
): Promise<ScreenControlResult> {
  try {
    const { x, y } = params;
    const abs = toDesktopCoords(x!, y!);
    await runPowerShell(script("screen-move.ps1"), {
      env: { X: String(Math.round(abs.x)), Y: String(Math.round(abs.y)) }
    });
    return { success: true, action: "mouse_move", x, y };
  } catch (err: any) {
    return { success: false, error: err.message };
  }
}

async function handleKeyboardType(
  params: ScreenControlParams
): Promise<ScreenControlResult> {
  try {
    const { text } = params;
    await runPowerShell(script("screen-type.ps1"), { env: { TEXT: text! } });
    return { success: true, action: "type", textLength: text!.length };
  } catch (err: any) {
    return { success: false, error: err.message };
  }
}

async function handleKeyPress(
  params: ScreenControlParams
): Promise<ScreenControlResult> {
  try {
    const { key, modifiers = [] } = params;
    const vk = getVkCode(key!);
    if (vk === null) return { success: false, error: `Unknown key: ${key}` };

    let pressScript = "";
    for (const mod of modifiers) {
      const modVk = getVkCode(mod);
      if (modVk === null)
        return { success: false, error: `Unknown modifier: ${mod}` };
      pressScript += `[WinInput]::keybd_event(${modVk}, 0, 0, [IntPtr]::Zero)\n`;
    }
    pressScript += `[WinInput]::keybd_event(${vk}, 0, 0, [IntPtr]::Zero)\n`;
    pressScript += `[WinInput]::keybd_event(${vk}, 0, [WinInput]::KEYEVENTF_KEYUP, [IntPtr]::Zero)\n`;
    for (const mod of [...modifiers].reverse()) {
      const modVk = getVkCode(mod);
      pressScript += `[WinInput]::keybd_event(${modVk}, 0, [WinInput]::KEYEVENTF_KEYUP, [IntPtr]::Zero)\n`;
    }

    await runPowerShell(script("screen-keypress.ps1"), {
      env: { PRESS_SCRIPT: pressScript }
    });
    return { success: true, action: "key_press", key, modifiers };
  } catch (err: any) {
    return { success: false, error: err.message };
  }
}

async function handleScroll(
  params: ScreenControlParams
): Promise<ScreenControlResult> {
  try {
    const { x, y, direction = "down", amount = 3 } = params;
    const delta = direction === "up" ? 120 * amount : -120 * amount;
    const env: Record<string, string> = { X: "", Y: "", DELTA: String(delta) };
    if (x !== undefined && y !== undefined) {
      const abs = toDesktopCoords(x, y);
      env.X = String(Math.round(abs.x));
      env.Y = String(Math.round(abs.y));
    }
    await runPowerShell(script("screen-scroll.ps1"), { env });
    return { success: true, action: "scroll", direction, amount };
  } catch (err: any) {
    return { success: false, error: err.message };
  }
}

// ---- Window handlers ----

async function handleListWindows(): Promise<ScreenControlResult> {
  try {
    const { stdout } = await runPowerShell(script("window-list.ps1"), {
      timeout: 10000
    });
    let windows: Array<Record<string, unknown>> = [];
    if (stdout) {
      const parsed = JSON.parse(stdout);
      windows = Array.isArray(parsed) ? parsed : [parsed];
    }
    return { success: true, windows };
  } catch (err: any) {
    return { success: false, error: err.message };
  }
}

async function handleFocusWindow(
  params: ScreenControlParams
): Promise<ScreenControlResult> {
  try {
    const { handle, title } = params;
    if (!handle && !title)
      return { success: false, error: "Provide handle or title" };
    const env: Record<string, string> = {};
    if (handle) env.HWND = String(handle);
    if (title) env.TITLE = title;
    const { stdout } = await runPowerShell(script("window-focus.ps1"), { env });
    return { success: true, message: stdout };
  } catch (err: any) {
    return { success: false, error: err.stderr?.trim() || err.message };
  }
}

async function handleResizeWindow(
  params: ScreenControlParams
): Promise<ScreenControlResult> {
  try {
    const { handle, x, y, width, height } = params;
    if (!handle) return { success: false, error: "handle is required" };
    const env: Record<string, string> = { HWND: String(handle) };
    if (x !== undefined) env.X = String(x);
    if (y !== undefined) env.Y = String(y);
    if (width !== undefined) env.W = String(width);
    if (height !== undefined) env.H = String(height);
    const { stdout } = await runPowerShell(script("window-resize.ps1"), {
      env
    });
    return { success: true, message: stdout };
  } catch (err: any) {
    return { success: false, error: err.message };
  }
}

const SW_MAP: Record<string, number> = { minimize: 6, maximize: 3, restore: 9 };

async function handleSetState(
  params: ScreenControlParams & { state: string }
): Promise<ScreenControlResult> {
  try {
    const { handle, state } = params;
    if (!handle) return { success: false, error: "handle is required" };
    const swCmd = SW_MAP[state];
    if (swCmd === undefined)
      return { success: false, error: `Unknown state: ${state}` };
    await runPowerShell(script("window-set-state.ps1"), {
      env: { HWND: String(handle), SW_CMD: String(swCmd) }
    });
    return { success: true, message: state };
  } catch (err: any) {
    return { success: false, error: err.message };
  }
}

async function handleWindowScreenshot(
  params: ScreenControlParams
): Promise<ScreenControlResult> {
  try {
    const { handle } = params;
    if (!handle) return { success: false, error: "handle is required" };
    const { stdout } = await runPowerShell(script("window-screenshot.ps1"), {
      timeout: 15000,
      maxBuffer: 20 * 1024 * 1024,
      env: { HWND: String(handle) }
    });
    const lines = stdout.split(/\r?\n/);
    // First line format: "left,top,WxH"
    const headerMatch = lines[0].match(/^(-?\d+),(-?\d+),(\d+)x(\d+)$/);
    if (!headerMatch)
      return { success: false, error: `Unexpected header: ${lines[0]}` };
    const [, leftStr, topStr, wStr, hStr] = headerMatch;
    const width = Number(wStr);
    const height = Number(hStr);
    // Remember window position so subsequent click/move/scroll coords get offset
    lastWindowOffset = { left: Number(leftStr), top: Number(topStr) };
    const base64 = lines.slice(1).join("");
    return { success: true, width, height, base64 };
  } catch (err: any) {
    return { success: false, error: err.stderr?.trim() || err.message };
  }
}

// ---- Foreground window debug helper ----

export async function getForegroundWindow(): Promise<{
  handle: number;
  title: string;
}> {
  try {
    const { stdout } = await runPowerShell(script("debug-foreground.ps1"));
    const [handleStr, ...titleParts] = stdout.split("|");
    return {
      handle: parseInt(handleStr, 10) || 0,
      title: titleParts.join("|")
    };
  } catch {
    return { handle: 0, title: "(unknown)" };
  }
}

// ---- Unified dispatch (same interface as preload.cjs screenControl) ----

const stateMap: Record<string, string> = {
  minimize_window: "minimize",
  maximize_window: "maximize",
  restore_window: "restore"
};

// Alias map for common LLM misspellings of action names
const actionAliases: Record<string, string> = {
  press_key: "key_press",
  keypress: "key_press",
  move: "mouse_move",
  move_mouse: "mouse_move",
  focus: "focus_window",
  resize: "resize_window",
  minimize: "minimize_window",
  maximize: "maximize_window",
  restore: "restore_window",
  list: "list_windows"
};

export interface ScreenControlOptions {
  debugDir?: string;
  debugStep?: number;
}

export async function screenControl(
  params: ScreenControlParams,
  options?: ScreenControlOptions
): Promise<ScreenControlResult> {
  const action = actionAliases[params.action] ?? params.action;
  const { debugDir, debugStep } = options ?? {};

  if (debugDir) {
    const before = await getForegroundWindow();
    const prefix = debugStep !== undefined ? `step${debugStep}` : "?";
    fs.appendFileSync(
      path.join(debugDir, "focus-log.txt"),
      `[${prefix}] BEFORE ${action}: foreground = ${before.handle} "${before.title}"\n`
    );
  }

  let result: ScreenControlResult;
  switch (action) {
    case "screenshot":
      result = await handleScreenshot();
      break;
    case "click":
      result = await handleMouseClick(params);
      break;
    case "mouse_move":
      result = await handleMouseMove(params);
      break;
    case "type":
      result = await handleKeyboardType(params);
      break;
    case "key_press":
      result = await handleKeyPress(params);
      break;
    case "scroll":
      result = await handleScroll(params);
      break;
    case "list_windows":
      result = await handleListWindows();
      break;
    case "focus_window":
      result = await handleFocusWindow(params);
      break;
    case "resize_window":
      result = await handleResizeWindow(params);
      break;
    case "minimize_window":
    case "maximize_window":
    case "restore_window":
      result = await handleSetState({ ...params, state: stateMap[action] });
      break;
    case "window_screenshot":
      result = await handleWindowScreenshot(params);
      break;
    default:
      result = { success: false, error: `Unknown action: ${action}` };
  }

  if (debugDir) {
    const after = await getForegroundWindow();
    const prefix = debugStep !== undefined ? `step${debugStep}` : "?";
    fs.appendFileSync(
      path.join(debugDir, "focus-log.txt"),
      `[${prefix}] AFTER  ${action}: foreground = ${after.handle} "${after.title}"\n`
    );

    if (result.base64) {
      const filename = `${prefix}-${action}.png`;
      fs.writeFileSync(
        path.join(debugDir, filename),
        Buffer.from(result.base64, "base64")
      );
    }
  }

  return result;
}
