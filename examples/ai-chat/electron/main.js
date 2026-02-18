import { app, BrowserWindow, ipcMain } from "electron";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFile, execFileSync } from "node:child_process";
import { promisify } from "node:util";
import fs from "node:fs";
import fsPromises from "node:fs/promises";

const execFileAsync = promisify(execFile);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCRIPTS_DIR = path.join(__dirname, "scripts");

const PROD_URL = "https://ai.connect-screen.com";
const APP_URL = process.env.AGENT_URL || PROD_URL;

// Virtual key code mapping for special keys
const VK_MAP = {
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
  // Modifier keys
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

function getVkCode(key) {
  const lower = key.toLowerCase();
  if (VK_MAP[lower] !== undefined) return VK_MAP[lower];
  // Single character — use its ASCII/VK code (uppercase)
  if (key.length === 1) {
    const code = key.toUpperCase().charCodeAt(0);
    // Letters A-Z: VK codes match ASCII uppercase
    if (code >= 0x41 && code <= 0x5a) return code;
    // Digits 0-9: VK codes match ASCII
    if (code >= 0x30 && code <= 0x39) return code;
  }
  return null;
}

function script(name) {
  return path.join(SCRIPTS_DIR, name);
}

async function runPowerShell(
  scriptPath,
  { env = {}, timeout = 5000, maxBuffer = 1024 * 1024 } = {}
) {
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

// --- IPC Handlers ---

function setupScreenHandlers() {
  ipcMain.handle("screen:screenshot", async () => {
    try {
      const { stdout } = await runPowerShell(script("screen-screenshot.ps1"), {
        timeout: 15000,
        maxBuffer: 20 * 1024 * 1024
      });
      const lines = stdout.split(/\r?\n/);
      const [width, height] = lines[0].split("x").map(Number);
      const base64 = lines.slice(1).join("");
      return { success: true, width, height, base64 };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  ipcMain.handle("screen:mouse-click", async (_event, params) => {
    try {
      const { x, y, button = "left", doubleClick = false } = params;
      const flags = {
        left: { down: 0x0002, up: 0x0004 },
        right: { down: 0x0008, up: 0x0010 },
        middle: { down: 0x0020, up: 0x0040 }
      };
      const f = flags[button] || flags.left;
      await runPowerShell(script("screen-click.ps1"), {
        env: {
          X: String(Math.round(x)),
          Y: String(Math.round(y)),
          DOWN_FLAG: String(f.down),
          UP_FLAG: String(f.up),
          CLICK_COUNT: String(doubleClick ? 2 : 1)
        }
      });
      return { success: true, action: "click", x, y, button, doubleClick };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  ipcMain.handle("screen:mouse-move", async (_event, params) => {
    try {
      const { x, y } = params;
      await runPowerShell(script("screen-move.ps1"), {
        env: { X: String(Math.round(x)), Y: String(Math.round(y)) }
      });
      return { success: true, action: "mouse_move", x, y };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  ipcMain.handle("screen:keyboard-type", async (_event, params) => {
    try {
      const { text } = params;
      await runPowerShell(script("screen-type.ps1"), {
        env: { TEXT: text }
      });
      return { success: true, action: "type", textLength: text.length };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  ipcMain.handle("screen:key-press", async (_event, params) => {
    try {
      const { key, modifiers = [] } = params;
      const vk = getVkCode(key);
      if (vk === null) {
        return { success: false, error: `Unknown key: ${key}` };
      }

      let pressScript = "";
      // Press modifier keys down
      for (const mod of modifiers) {
        const modVk = getVkCode(mod);
        if (modVk === null) {
          return { success: false, error: `Unknown modifier: ${mod}` };
        }
        pressScript += `[WinInput]::keybd_event(${modVk}, 0, 0, [IntPtr]::Zero)\n`;
      }
      // Press and release the main key
      pressScript += `[WinInput]::keybd_event(${vk}, 0, 0, [IntPtr]::Zero)\n`;
      pressScript += `[WinInput]::keybd_event(${vk}, 0, [WinInput]::KEYEVENTF_KEYUP, [IntPtr]::Zero)\n`;
      // Release modifier keys (reverse order)
      for (const mod of [...modifiers].reverse()) {
        const modVk = getVkCode(mod);
        pressScript += `[WinInput]::keybd_event(${modVk}, 0, [WinInput]::KEYEVENTF_KEYUP, [IntPtr]::Zero)\n`;
      }

      await runPowerShell(script("screen-keypress.ps1"), {
        env: { PRESS_SCRIPT: pressScript }
      });
      return { success: true, action: "key_press", key, modifiers };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  ipcMain.handle("screen:scroll", async (_event, params) => {
    try {
      const { x, y, direction = "down", amount = 3 } = params;
      const delta = direction === "up" ? 120 * amount : -120 * amount;
      const env = { X: "", Y: "", DELTA: String(delta) };
      if (x !== undefined && y !== undefined) {
        env.X = String(Math.round(x));
        env.Y = String(Math.round(y));
      }
      await runPowerShell(script("screen-scroll.ps1"), { env });
      return { success: true, action: "scroll", direction, amount };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });
}

function setupWindowHandlers() {
  ipcMain.handle("window:list-windows", async () => {
    try {
      const { stdout } = await runPowerShell(script("window-list.ps1"), {
        timeout: 10000
      });
      let windows = [];
      if (stdout) {
        const parsed = JSON.parse(stdout);
        windows = Array.isArray(parsed) ? parsed : [parsed];
      }
      return { success: true, windows };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  ipcMain.handle("window:focus-window", async (_event, params) => {
    try {
      const { handle, title } = params;
      if (!handle && !title)
        return { success: false, error: "Provide handle or title" };
      const env = {};
      if (handle) env.HWND = String(handle);
      if (title) env.TITLE = title;
      const { stdout } = await runPowerShell(script("window-focus.ps1"), {
        env
      });
      return { success: true, message: stdout };
    } catch (err) {
      return { success: false, error: err.stderr?.trim() || err.message };
    }
  });

  ipcMain.handle("window:resize-window", async (_event, params) => {
    try {
      const { handle, x, y, width, height } = params;
      if (!handle) return { success: false, error: "handle is required" };
      const env = { HWND: String(handle) };
      if (x !== undefined) env.X = String(x);
      if (y !== undefined) env.Y = String(y);
      if (width !== undefined) env.W = String(width);
      if (height !== undefined) env.H = String(height);
      const { stdout } = await runPowerShell(script("window-resize.ps1"), {
        env
      });
      return { success: true, message: stdout };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  // Unified minimize/maximize/restore handler
  const SW_MAP = { minimize: 6, maximize: 3, restore: 9 };
  ipcMain.handle("window:set-state", async (_event, params) => {
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
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  ipcMain.handle("window:screenshot", async (_event, params) => {
    try {
      const { handle } = params;
      if (!handle) return { success: false, error: "handle is required" };
      const { stdout } = await runPowerShell(script("window-screenshot.ps1"), {
        timeout: 15000,
        maxBuffer: 20 * 1024 * 1024,
        env: { HWND: String(handle) }
      });
      const lines = stdout.split(/\r?\n/);
      const [width, height] = lines[0].split("x").map(Number);
      const base64 = lines.slice(1).join("");
      return { success: true, width, height, base64 };
    } catch (err) {
      return { success: false, error: err.stderr?.trim() || err.message };
    }
  });
}

function setupFileSystemHandlers() {
  // Detect available Windows drives and WSL distros
  ipcMain.handle("fs:detect-drives", async () => {
    const drives = [];

    // Scan A:-Z: for existing drives
    for (let code = 65; code <= 90; code++) {
      const letter = String.fromCharCode(code);
      const root = `${letter}:\\`;
      try {
        fs.statSync(root);
        drives.push({
          mountPoint: `/mnt/${letter.toLowerCase()}`,
          root
        });
      } catch {
        // Drive doesn't exist
      }
    }

    // Detect WSL distros
    try {
      const output = execFileSync("wsl", ["--list", "--quiet"], {
        encoding: "utf16le",
        windowsHide: true,
        timeout: 5000
      });
      const distros = output
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter(Boolean);
      if (distros.length > 0) {
        drives.push({
          mountPoint: "/mnt/wsl",
          root: `\\\\wsl.localhost\\${distros[0]}\\`
        });
      }
    } catch {
      // WSL not available
    }

    return drives;
  });

  // Unified fs operations IPC channel
  ipcMain.handle("fs:op", async (_event, params) => {
    const { op } = params;
    try {
      switch (op) {
        case "readFile": {
          const content = await fsPromises.readFile(params.path, "utf8");
          return { ok: true, result: content };
        }
        case "readFileBuffer": {
          const buf = await fsPromises.readFile(params.path);
          return { ok: true, result: new Uint8Array(buf) };
        }
        case "writeFile": {
          await fsPromises.writeFile(params.path, params.content);
          return { ok: true };
        }
        case "appendFile": {
          await fsPromises.appendFile(params.path, params.content);
          return { ok: true };
        }
        case "stat":
        case "lstat": {
          const s = await (op === "lstat"
            ? fsPromises.lstat(params.path)
            : fsPromises.stat(params.path));
          return {
            ok: true,
            result: {
              isFile: s.isFile(),
              isDirectory: s.isDirectory(),
              isSymbolicLink: s.isSymbolicLink(),
              mode: s.mode,
              size: s.size,
              mtime: s.mtime.toISOString()
            }
          };
        }
        case "exists": {
          try {
            await fsPromises.stat(params.path);
            return { ok: true, result: true };
          } catch {
            return { ok: true, result: false };
          }
        }
        case "mkdir": {
          await fsPromises.mkdir(params.path, {
            recursive: !!params.recursive
          });
          return { ok: true };
        }
        case "readdir": {
          const entries = await fsPromises.readdir(params.path);
          return { ok: true, result: entries };
        }
        case "readdirWithFileTypes": {
          const dirents = await fsPromises.readdir(params.path, {
            withFileTypes: true
          });
          return {
            ok: true,
            result: dirents.map((d) => ({
              name: d.name,
              isFile: d.isFile(),
              isDirectory: d.isDirectory(),
              isSymbolicLink: d.isSymbolicLink()
            }))
          };
        }
        case "rm": {
          await fsPromises.rm(params.path, {
            recursive: !!params.recursive,
            force: !!params.force
          });
          return { ok: true };
        }
        case "cp": {
          await fsPromises.cp(params.src, params.dest, {
            recursive: !!params.recursive
          });
          return { ok: true };
        }
        case "mv": {
          await fsPromises.rename(params.src, params.dest);
          return { ok: true };
        }
        case "chmod": {
          await fsPromises.chmod(params.path, params.mode);
          return { ok: true };
        }
        case "symlink": {
          await fsPromises.symlink(params.target, params.linkPath);
          return { ok: true };
        }
        case "link": {
          await fsPromises.link(params.existingPath, params.newPath);
          return { ok: true };
        }
        case "readlink": {
          const target = await fsPromises.readlink(params.path);
          return { ok: true, result: target };
        }
        case "realpath": {
          const resolved = await fsPromises.realpath(params.path);
          return { ok: true, result: resolved };
        }
        case "utimes": {
          await fsPromises.utimes(
            params.path,
            new Date(params.atime),
            new Date(params.mtime)
          );
          return { ok: true };
        }
        default:
          return { ok: false, error: `Unknown fs op: ${op}`, code: "ENOSYS" };
      }
    } catch (err) {
      return { ok: false, error: err.message, code: err.code };
    }
  });
}

// Log file for renderer console output (readable from WSL)
const LOG_FILE = path.join(__dirname, "renderer.log");
fs.writeFileSync(
  LOG_FILE,
  `--- Electron started ${new Date().toISOString()} ---\n`
);

function appendLog(level, msg) {
  const line = `[${new Date().toISOString()}] [${level}] ${msg}\n`;
  fs.appendFileSync(LOG_FILE, line);
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1000,
    height: 700,
    title: "Windows Agent",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  // Capture renderer console.log and write to file
  win.webContents.on("console-message", (event) => {
    const levels = ["verbose", "info", "warning", "error"];
    appendLog(levels[event.level] || "unknown", event.message);
  });

  win.loadURL(APP_URL);
}

app.whenReady().then(() => {
  setupScreenHandlers();
  setupWindowHandlers();
  setupFileSystemHandlers();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  app.quit();
});
