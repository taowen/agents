/**
 * Windows desktop automation via PowerShell scripts.
 * Zero Electron dependencies — uses only Node.js built-ins.
 */
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import path from "node:path";
import fs from "node:fs";
import os from "node:os";
import { fileURLToPath } from "node:url";

import type {
  ScreenControlParams,
  ScreenControlResult
} from "../src/shared/screen-control-types.ts";
import {
  normalizeAction,
  isDoubleClickAlias
} from "../src/shared/action-aliases.ts";

export type { ScreenControlParams, ScreenControlResult };

const execFileAsync = promisify(execFile);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCRIPTS_DIR = path.join(__dirname, "scripts");

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

// ---- Run arbitrary PowerShell command ----

export async function runPowerShellCommand(
  command: string,
  extraEnv?: Record<string, string>
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  try {
    const { stdout, stderr } = await execFileAsync(
      "powershell.exe",
      [
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        command
      ],
      {
        timeout: 30000,
        maxBuffer: 5 * 1024 * 1024,
        windowsHide: true,
        env: extraEnv ? { ...process.env, ...extraEnv } : process.env
      }
    );
    return { stdout: stdout.trimEnd(), stderr: stderr.trimEnd(), exitCode: 0 };
  } catch (err: any) {
    // execFile rejects on non-zero exit or timeout
    return {
      stdout: (err.stdout ?? "").trimEnd(),
      stderr: (err.stderr ?? err.message ?? "").trimEnd(),
      exitCode:
        err.code === "ERR_CHILD_PROCESS_STDIO_MAXBUFFER" ? 1 : (err.status ?? 1)
    };
  }
}

// ---- PowerShell executor factory ----

export function createPowerShellExecutor(options?: {
  cloudUrl?: string;
  getSessionCookie?: () => Promise<string | undefined>;
}): (
  command: string
) => Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const cloudDriveScript = path.join(SCRIPTS_DIR, "cloud-drive.ps1");
  const cloudUrl = options?.cloudUrl;
  const getSessionCookie = options?.getSessionCookie;

  return async (command: string) => {
    const fullCommand = cloudUrl
      ? `. '${cloudDriveScript}'; ${command}`
      : command;
    const extraEnv: Record<string, string> = {};
    if (cloudUrl) extraEnv.CLOUD_BASE = cloudUrl;
    if (getSessionCookie) {
      const cookie = await getSessionCookie();
      if (cookie) extraEnv.CLOUD_COOKIE = cookie;
    }
    return runPowerShellCommand(
      fullCommand,
      Object.keys(extraEnv).length ? extraEnv : undefined
    );
  };
}

// ---- Cloud drive cleanup ----

export function cleanupCloudDrive(): void {
  const dllPath = path.join(os.tmpdir(), "CloudFSProvider-v2.dll");
  try {
    fs.unlinkSync(dllPath);
  } catch {}
}

// ---- Coordinate offset state ----
// After window_screenshot, we remember the window's screen position so
// click/move/scroll coordinates (which are window-relative from the LLM's
// perspective) get translated to desktop-absolute coordinates automatically.
let lastWindowOffset: { left: number; top: number } | null = null;

/** Convert window-relative coords to desktop-absolute when offset is known */
function toDesktopCoords(x: number, y: number): { x: number; y: number } {
  if (lastWindowOffset) {
    return { x: x + lastWindowOffset.left, y: y + lastWindowOffset.top };
  }
  return { x, y };
}

// ---- Screen handlers ----

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
    return {
      success: true,
      action: "click",
      x,
      y,
      button,
      doubleClick,
      desktopX: abs.x,
      desktopY: abs.y
    };
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
    return {
      success: true,
      action: "mouse_move",
      x,
      y,
      desktopX: abs.x,
      desktopY: abs.y
    };
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
    const result: ScreenControlResult = {
      success: true,
      action: "scroll",
      direction,
      amount
    };
    if (x !== undefined && y !== undefined) {
      const abs = toDesktopCoords(x, y);
      env.X = String(Math.round(abs.x));
      env.Y = String(Math.round(abs.y));
      result.desktopX = abs.x;
      result.desktopY = abs.y;
    }
    await runPowerShell(script("screen-scroll.ps1"), { env });
    return result;
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

async function handleWindowAccessibility(
  handle: number
): Promise<{
  tree: string;
  left: number;
  top: number;
  w: number;
  h: number;
} | null> {
  try {
    const { stdout, stderr } = await runPowerShell(
      script("window-accessibility.ps1"),
      {
        timeout: 10000,
        maxBuffer: 1024 * 1024,
        env: { HWND: String(handle) }
      }
    );
    if (!stdout) return null;
    const lines = stdout.split(/\r?\n/);
    const headerMatch = lines[0]?.match(/^(-?\d+),(-?\d+),(\d+)x(\d+)$/);
    if (!headerMatch) return null;
    const [, leftStr, topStr, wStr, hStr] = headerMatch;
    const tree = lines.slice(1).join("\n");
    return {
      tree,
      left: Number(leftStr),
      top: Number(topStr),
      w: Number(wStr),
      h: Number(hStr)
    };
  } catch {
    return null;
  }
}

function isTreeUseful(tree: string): { useful: boolean; reason: string } {
  const lines = tree.split("\n");
  // Count element lines (lines containing "[SomeType]")
  const elementLines = lines.filter((l) => /\[[\w]+\]/.test(l.trim()));
  if (elementLines.length < 3)
    return { useful: false, reason: `only ${elementLines.length} elements` };

  // Check interactive elements specifically — if many exist but none has Name,
  // the tree is useless for interaction (e.g. Calculator with 28 anonymous Buttons)
  const interactiveTypes =
    /\[(Button|MenuItem|CheckBox|RadioButton|TabItem|ListItem|TreeItem|ComboBox)\]/;
  const interactiveLines = elementLines.filter((l) => interactiveTypes.test(l));
  if (interactiveLines.length >= 3) {
    const namedInteractive = interactiveLines.filter((l) =>
      /Name="[^"]+"/.test(l)
    );
    if (namedInteractive.length === 0) {
      return {
        useful: false,
        reason: `${interactiveLines.length} interactive elements but none has Name`
      };
    }
  }

  // Check if at least one element has a non-empty Name
  const hasName = elementLines.some((l) => /Name="[^"]+"/.test(l));
  if (!hasName)
    return {
      useful: false,
      reason: `${elementLines.length} elements but none has Name`
    };
  return {
    useful: true,
    reason: `${elementLines.length} elements (${interactiveLines.length} interactive) with names`
  };
}

async function handleWindowScreenshot(
  params: ScreenControlParams
): Promise<ScreenControlResult> {
  try {
    const { handle } = params;
    if (!handle) return { success: false, error: "handle is required" };

    const mode = params.mode ?? "auto";
    let a11yFallbackReason: string | undefined;

    // Try accessibility tree for auto and accessibility modes
    if (mode === "auto" || mode === "accessibility") {
      const a11y = await handleWindowAccessibility(handle);
      if (a11y) {
        lastWindowOffset = { left: a11y.left, top: a11y.top };
        if (mode === "accessibility") {
          return {
            success: true,
            width: a11y.w,
            height: a11y.h,
            accessibilityTree: a11y.tree,
            windowLeft: a11y.left,
            windowTop: a11y.top,
            a11yDiagnostics: `accessibility: tree returned (${a11y.tree.split("\n").filter((l) => /\[[\w]+\]/.test(l.trim())).length} elements)`
          };
        }
        // auto mode: use tree only if useful
        const treeCheck = isTreeUseful(a11y.tree);
        if (treeCheck.useful) {
          return {
            success: true,
            width: a11y.w,
            height: a11y.h,
            accessibilityTree: a11y.tree,
            windowLeft: a11y.left,
            windowTop: a11y.top,
            a11yDiagnostics: `auto: tree accepted (${treeCheck.reason})`
          };
        }
        // Fall through to pixel mode — remember why tree was rejected
        a11yFallbackReason = `auto: tree rejected (${treeCheck.reason}), fell back to pixel`;
      } else if (mode === "accessibility") {
        // Accessibility explicitly requested but failed
        return {
          success: true,
          width: 0,
          height: 0,
          accessibilityTree: "",
          windowLeft: 0,
          windowTop: 0,
          a11yDiagnostics: "accessibility: tree fetch failed"
        };
      }
      // Fall through to pixel mode
    }

    // Pixel mode (or auto fallback)
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
    const windowLeft = Number(leftStr);
    const windowTop = Number(topStr);
    lastWindowOffset = { left: windowLeft, top: windowTop };
    const base64 = lines.slice(1).join("");
    return {
      success: true,
      width,
      height,
      base64,
      windowLeft,
      windowTop,
      a11yDiagnostics: a11yFallbackReason
    };
  } catch (err: any) {
    return { success: false, error: err.stderr?.trim() || err.message };
  }
}

// ---- Annotate handler ----

async function handleAnnotate(
  params: ScreenControlParams
): Promise<ScreenControlResult> {
  try {
    const { x, y, base64 } = params;
    if (!base64)
      return { success: false, error: "No screenshot available to annotate" };
    if (x === undefined || y === undefined)
      return { success: false, error: "x and y are required for annotate" };

    const tmpFile = path.join(os.tmpdir(), `annotate-${Date.now()}.png`);
    fs.writeFileSync(tmpFile, Buffer.from(base64, "base64"));

    try {
      const env: Record<string, string> = {
        IMAGE_PATH: tmpFile,
        X: String(Math.round(x)),
        Y: String(Math.round(y))
      };
      if (params.normX !== undefined) env.NORM_X = String(params.normX);
      if (params.normY !== undefined) env.NORM_Y = String(params.normY);
      const { stdout } = await runPowerShell(script("annotate-image.ps1"), {
        timeout: 15000,
        maxBuffer: 20 * 1024 * 1024,
        env
      });
      const lines = stdout.split(/\r?\n/);
      const [width, height] = lines[0].split("x").map(Number);
      const annotatedBase64 = lines.slice(1).join("");
      return { success: true, width, height, base64: annotatedBase64, x, y };
    } finally {
      try {
        fs.unlinkSync(tmpFile);
      } catch {}
    }
  } catch (err: any) {
    return { success: false, error: err.message };
  }
}

// ---- Unified dispatch (same interface as preload.cjs screenControl) ----

const stateMap: Record<string, string> = {
  minimize_window: "minimize",
  maximize_window: "maximize",
  restore_window: "restore"
};

export async function screenControl(
  params: ScreenControlParams
): Promise<ScreenControlResult> {
  const action = normalizeAction(params.action);

  if (isDoubleClickAlias(params.action)) {
    params = { ...params, doubleClick: true };
  }

  switch (action) {
    case "click":
      return handleMouseClick(params);
    case "annotate":
      return handleAnnotate(params);
    case "mouse_move":
      return handleMouseMove(params);
    case "type":
      return handleKeyboardType(params);
    case "key_press":
      return handleKeyPress(params);
    case "scroll":
      return handleScroll(params);
    case "list_windows":
      return handleListWindows();
    case "focus_window":
      return handleFocusWindow(params);
    case "resize_window":
      return handleResizeWindow(params);
    case "minimize_window":
    case "maximize_window":
    case "restore_window":
      return handleSetState({ ...params, state: stateMap[action] });
    case "window_screenshot":
      return handleWindowScreenshot(params);
    default:
      return { success: false, error: `Unknown action: ${action}` };
  }
}
