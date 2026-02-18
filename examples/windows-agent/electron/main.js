import { app, BrowserWindow, ipcMain } from "electron";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import fs from "node:fs";

const execFileAsync = promisify(execFile);

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PROD_URL = "https://ai.connect-screen.com/windows-agent";
const APP_URL = process.env.AGENT_URL || PROD_URL;

// Shared Win32 P/Invoke C# code for input simulation
const WIN_INPUT_CS = `
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public class WinInput {
    [DllImport("user32.dll")]
    public static extern bool SetCursorPos(int x, int y);

    [DllImport("user32.dll")]
    public static extern void mouse_event(uint dwFlags, int dx, int dy, int dwData, IntPtr dwExtraInfo);

    [DllImport("user32.dll")]
    public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, IntPtr dwExtraInfo);

    public const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
    public const uint MOUSEEVENTF_LEFTUP = 0x0004;
    public const uint MOUSEEVENTF_RIGHTDOWN = 0x0008;
    public const uint MOUSEEVENTF_RIGHTUP = 0x0010;
    public const uint MOUSEEVENTF_MIDDLEDOWN = 0x0020;
    public const uint MOUSEEVENTF_MIDDLEUP = 0x0040;
    public const uint MOUSEEVENTF_WHEEL = 0x0800;

    public const uint KEYEVENTF_KEYUP = 0x0002;
}
"@ -ErrorAction SilentlyContinue
`;

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

async function runPowerShell(
  script,
  { timeout = 5000, maxBuffer = 1024 * 1024 } = {}
) {
  const { stdout, stderr } = await execFileAsync(
    "powershell.exe",
    ["-NoProfile", "-NonInteractive", "-Command", script],
    { timeout, maxBuffer, windowsHide: true }
  );
  return { stdout: stdout.trim(), stderr: stderr.trim() };
}

// --- IPC Handlers ---

function setupScreenHandlers() {
  // Screenshot: captures the primary screen and returns dimensions + base64 PNG
  ipcMain.handle("screen:screenshot", async () => {
    try {
      const script = `
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms

$screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap($screen.Width, $screen.Height)
$gfx = [System.Drawing.Graphics]::FromImage($bmp)
$gfx.CopyFromScreen($screen.Location, [System.Drawing.Point]::Empty, $screen.Size)
$gfx.Dispose()

# Resize to max 1280px wide to reduce base64 size
$maxW = 1280
$outBmp = $bmp
if ($bmp.Width -gt $maxW) {
    $ratio = $maxW / $bmp.Width
    $newH = [int]($bmp.Height * $ratio)
    $outBmp = New-Object System.Drawing.Bitmap($maxW, $newH)
    $g = [System.Drawing.Graphics]::FromImage($outBmp)
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $g.DrawImage($bmp, 0, 0, $maxW, $newH)
    $g.Dispose()
    $bmp.Dispose()
}

$ms = New-Object System.IO.MemoryStream
$outBmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
$outBmp.Dispose()

$bytes = $ms.ToArray()
$ms.Dispose()

Write-Output "$($screen.Width)x$($screen.Height)"
Write-Output ([Convert]::ToBase64String($bytes))
`;
      const { stdout } = await runPowerShell(script, {
        timeout: 15000,
        maxBuffer: 20 * 1024 * 1024
      });
      const lines = stdout.split(/\r?\n/);
      const dimensions = lines[0]; // e.g. "1920x1080"
      const base64 = lines.slice(1).join("");
      const [width, height] = dimensions.split("x").map(Number);
      return { success: true, width, height, base64 };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  // Mouse click at (x, y)
  ipcMain.handle("screen:mouse-click", async (_event, params) => {
    try {
      const { x, y, button = "left", doubleClick = false } = params;
      let downFlag, upFlag;
      switch (button) {
        case "right":
          downFlag = "MOUSEEVENTF_RIGHTDOWN";
          upFlag = "MOUSEEVENTF_RIGHTUP";
          break;
        case "middle":
          downFlag = "MOUSEEVENTF_MIDDLEDOWN";
          upFlag = "MOUSEEVENTF_MIDDLEUP";
          break;
        default:
          downFlag = "MOUSEEVENTF_LEFTDOWN";
          upFlag = "MOUSEEVENTF_LEFTUP";
      }
      const clickCount = doubleClick ? 2 : 1;
      let clickScript = "";
      for (let i = 0; i < clickCount; i++) {
        clickScript += `
[WinInput]::mouse_event([WinInput]::${downFlag}, 0, 0, 0, [IntPtr]::Zero)
[WinInput]::mouse_event([WinInput]::${upFlag}, 0, 0, 0, [IntPtr]::Zero)
`;
        if (i < clickCount - 1) {
          clickScript += "Start-Sleep -Milliseconds 50\n";
        }
      }
      const script = `${WIN_INPUT_CS}
[WinInput]::SetCursorPos(${Math.round(x)}, ${Math.round(y)})
Start-Sleep -Milliseconds 10
${clickScript}
Write-Output "clicked ${button} at ${x},${y}"
`;
      await runPowerShell(script);
      return { success: true, action: "click", x, y, button, doubleClick };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  // Mouse move to (x, y)
  ipcMain.handle("screen:mouse-move", async (_event, params) => {
    try {
      const { x, y } = params;
      const script = `${WIN_INPUT_CS}
[WinInput]::SetCursorPos(${Math.round(x)}, ${Math.round(y)})
Write-Output "moved to ${x},${y}"
`;
      await runPowerShell(script);
      return { success: true, action: "mouse_move", x, y };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  // Type text via clipboard (supports Unicode)
  ipcMain.handle("screen:keyboard-type", async (_event, params) => {
    try {
      const { text } = params;
      // Escape single quotes for PowerShell by doubling them
      const escaped = text.replace(/'/g, "''");
      const script = `${WIN_INPUT_CS}
Add-Type -AssemblyName System.Windows.Forms

# Save current clipboard
$saved = $null
try { $saved = [System.Windows.Forms.Clipboard]::GetText() } catch {}

# Set text to clipboard and paste
[System.Windows.Forms.Clipboard]::SetText('${escaped}')
Start-Sleep -Milliseconds 50

# Ctrl+V
[WinInput]::keybd_event(0xA2, 0, 0, [IntPtr]::Zero)
[WinInput]::keybd_event(0x56, 0, 0, [IntPtr]::Zero)
[WinInput]::keybd_event(0x56, 0, [WinInput]::KEYEVENTF_KEYUP, [IntPtr]::Zero)
[WinInput]::keybd_event(0xA2, 0, [WinInput]::KEYEVENTF_KEYUP, [IntPtr]::Zero)

Start-Sleep -Milliseconds 50

# Restore clipboard
if ($saved -ne $null) {
    [System.Windows.Forms.Clipboard]::SetText($saved)
} else {
    [System.Windows.Forms.Clipboard]::Clear()
}

Write-Output "typed text"
`;
      await runPowerShell(script);
      return { success: true, action: "type", textLength: text.length };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  // Key press (single key or key combo with modifiers)
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

      const script = `${WIN_INPUT_CS}
${pressScript}
Write-Output "pressed ${key}"
`;
      await runPowerShell(script);
      return { success: true, action: "key_press", key, modifiers };
    } catch (err) {
      return { success: false, error: err.message };
    }
  });

  // Scroll (up/down/left/right)
  ipcMain.handle("screen:scroll", async (_event, params) => {
    try {
      const { x, y, direction = "down", amount = 3 } = params;
      // WHEEL_DELTA = 120 per notch
      const delta = direction === "up" ? 120 * amount : -120 * amount;
      let moveCmd = "";
      if (x !== undefined && y !== undefined) {
        moveCmd = `[WinInput]::SetCursorPos(${Math.round(x)}, ${Math.round(y)})\nStart-Sleep -Milliseconds 10\n`;
      }
      const script = `${WIN_INPUT_CS}
${moveCmd}[WinInput]::mouse_event([WinInput]::MOUSEEVENTF_WHEEL, 0, 0, ${delta}, [IntPtr]::Zero)
Write-Output "scrolled ${direction} by ${amount}"
`;
      await runPowerShell(script);
      return { success: true, action: "scroll", direction, amount };
    } catch (err) {
      return { success: false, error: err.message };
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
  win.webContents.on("console-message", (_event, level, message) => {
    const levels = ["verbose", "info", "warning", "error"];
    appendLog(levels[level] || "unknown", message);
  });

  win.loadURL(APP_URL);
}

app.whenReady().then(() => {
  setupScreenHandlers();
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
