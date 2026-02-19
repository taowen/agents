import { app, BrowserWindow, ipcMain } from "electron";
import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs";
import { NodeFsAdapter } from "./node-fs-adapter.ts";
import { detectDrives } from "./detect-drives.ts";
import { screenControl, runPowerShellCommand } from "./win-automation.ts";
import type { FsStat } from "just-bash";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PROD_URL = "https://ai.connect-screen.com";
const APP_URL = process.env.AGENT_URL || PROD_URL;

// --- IPC Handlers ---

function setupScreenControl(): void {
  ipcMain.handle("screen-control", (_event, params) => screenControl(params));
}

function setupPowerShellHandler(): void {
  ipcMain.handle("powershell:exec", (_event, params: { command: string }) =>
    runPowerShellCommand(params.command)
  );
}

/**
 * Serialize FsStat for IPC transport (Date → ISO string).
 */
function serializeStat(s: FsStat) {
  return {
    isFile: s.isFile,
    isDirectory: s.isDirectory,
    isSymbolicLink: s.isSymbolicLink,
    mode: s.mode,
    size: s.size,
    mtime: s.mtime.toISOString()
  };
}

function setupFileSystemHandlers(): void {
  ipcMain.handle("fs:detect-drives", async () => {
    return detectDrives();
  });

  // Pass-through adapter: renderer-side WindowsFsAdapter already converts
  // paths to Windows absolute paths, so we use them as-is.
  const fsAdapter = new NodeFsAdapter("");

  ipcMain.handle("fs:op", async (_event, params) => {
    const { op } = params;
    try {
      switch (op) {
        case "readFile":
          return { ok: true, result: await fsAdapter.readFile(params.path) };
        case "readFileBuffer":
          return {
            ok: true,
            result: await fsAdapter.readFileBuffer(params.path)
          };
        case "writeFile":
          await fsAdapter.writeFile(params.path, params.content);
          return { ok: true };
        case "appendFile":
          await fsAdapter.appendFile(params.path, params.content);
          return { ok: true };
        case "stat":
          return {
            ok: true,
            result: serializeStat(await fsAdapter.stat(params.path))
          };
        case "lstat":
          return {
            ok: true,
            result: serializeStat(await fsAdapter.lstat(params.path))
          };
        case "exists":
          return { ok: true, result: await fsAdapter.exists(params.path) };
        case "mkdir":
          await fsAdapter.mkdir(params.path, {
            recursive: !!params.recursive
          });
          return { ok: true };
        case "readdir":
          return { ok: true, result: await fsAdapter.readdir(params.path) };
        case "readdirWithFileTypes":
          return {
            ok: true,
            result: await fsAdapter.readdirWithFileTypes!(params.path)
          };
        case "rm":
          await fsAdapter.rm(params.path, {
            recursive: !!params.recursive,
            force: !!params.force
          });
          return { ok: true };
        case "cp":
          await fsAdapter.cp(params.src, params.dest, {
            recursive: !!params.recursive
          });
          return { ok: true };
        case "mv":
          await fsAdapter.mv(params.src, params.dest);
          return { ok: true };
        case "chmod":
          await fsAdapter.chmod(params.path, params.mode);
          return { ok: true };
        case "symlink":
          await fsAdapter.symlink(params.target, params.linkPath);
          return { ok: true };
        case "link":
          await fsAdapter.link(params.existingPath, params.newPath);
          return { ok: true };
        case "readlink":
          return { ok: true, result: await fsAdapter.readlink(params.path) };
        case "realpath":
          return { ok: true, result: await fsAdapter.realpath(params.path) };
        case "utimes":
          await fsAdapter.utimes(
            params.path,
            new Date(params.atime),
            new Date(params.mtime)
          );
          return { ok: true };
        default:
          return { ok: false, error: `Unknown fs op: ${op}`, code: "ENOSYS" };
      }
    } catch (err: any) {
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

function appendLog(level: string, msg: string): void {
  const line = `[${new Date().toISOString()}] [${level}] ${msg}\n`;
  fs.appendFileSync(LOG_FILE, line);
}

function createWindow(): void {
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
    appendLog(
      levels[(event as any).level] || "unknown",
      (event as any).message
    );
  });

  win.loadURL(APP_URL);
}

app.whenReady().then(() => {
  setupScreenControl();
  setupPowerShellHandler();
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
