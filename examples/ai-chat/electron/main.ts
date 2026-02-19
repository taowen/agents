import { app, BrowserWindow, dialog, ipcMain, session } from "electron";
import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs";
import os from "node:os";
import http from "node:http";
import { NodeFsAdapter } from "./node-fs-adapter.ts";
import { detectDrives } from "./detect-drives.ts";
import {
  screenControl,
  createPowerShellExecutor,
  cleanupCloudDrive
} from "./win-automation.ts";
import type { FsStat } from "just-bash";

// --- Crash log: as early as possible ---
const CRASH_LOG = path.join(os.tmpdir(), "windows-agent-crash.log");

process.on("uncaughtException", (err) => {
  const msg = `[${new Date().toISOString()}] ${err.stack || err.message}\n`;
  try {
    fs.appendFileSync(CRASH_LOG, msg);
  } catch {}
  dialog.showErrorBox("Windows Agent Crash", err.stack || err.message);
  app.exit(1);
});

process.on("unhandledRejection", (reason) => {
  const msg = `[${new Date().toISOString()}] Unhandled rejection: ${reason}\n`;
  try {
    fs.appendFileSync(CRASH_LOG, msg);
  } catch {}
});

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const PROD_URL = "https://ai.connect-screen.com";
const APP_URL = process.env.AGENT_URL || PROD_URL;

async function getSessionCookie(): Promise<string | undefined> {
  const cookies = await session.defaultSession.cookies.get({
    name: "session",
    url: APP_URL
  });
  return cookies[0]?.value;
}

const psExecutor = createPowerShellExecutor({
  cloudUrl: APP_URL,
  getSessionCookie
});

// --- IPC Handlers ---

function setupScreenControl(): void {
  ipcMain.handle("screen-control", (_event, params) => screenControl(params));
}

function setupPowerShellHandler(): void {
  ipcMain.handle("powershell:exec", (_event, params: { command: string }) =>
    psExecutor(params.command)
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

// Log file for renderer console output (readable from WSL in dev, userData in production)
const LOG_DIR = app.isPackaged ? app.getPath("userData") : __dirname;
const LOG_FILE = path.join(LOG_DIR, "renderer.log");
try {
  fs.mkdirSync(LOG_DIR, { recursive: true });
  fs.writeFileSync(
    LOG_FILE,
    `--- Electron started ${new Date().toISOString()} ---\n`
  );
} catch {}

function appendLog(level: string, msg: string): void {
  try {
    const line = `[${new Date().toISOString()}] [${level}] ${msg}\n`;
    fs.appendFileSync(LOG_FILE, line);
  } catch {}
}

// --- Debug HTTP Server ---

function setupDebugHttpServer(): void {
  // Map of pending debug task requests: taskId → response callback
  const pendingTasks = new Map<
    string,
    { res: http.ServerResponse; timer: ReturnType<typeof setTimeout> }
  >();
  let taskIdCounter = 0;

  // Listen for results from the renderer
  ipcMain.on(
    "debug:task-result",
    (_event, data: { taskId: string; response?: string; error?: string }) => {
      const pending = pendingTasks.get(data.taskId);
      if (!pending) return;
      clearTimeout(pending.timer);
      pendingTasks.delete(data.taskId);

      const body = JSON.stringify(
        data.error ? { error: data.error } : { response: data.response }
      );
      pending.res.writeHead(200, { "Content-Type": "application/json" });
      pending.res.end(body);
    }
  );

  const server = http.createServer(async (req, res) => {
    // CORS headers for convenience
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    if (req.method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }

    const url = new URL(req.url || "/", `http://${req.headers.host}`);

    // --- GET /status ---
    if (req.method === "GET" && url.pathname === "/status") {
      const hasCookie = !!(await getSessionCookie());
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ appUrl: APP_URL, hasCookie }));
      return;
    }

    // --- POST /exec ---
    if (req.method === "POST" && url.pathname === "/exec") {
      const body = await readBody(req);
      try {
        const { command } = JSON.parse(body);
        if (!command) {
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "missing 'command' field" }));
          return;
        }
        const result = await psExecutor(command);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
      } catch (err: any) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err.message }));
      }
      return;
    }

    // --- POST /task ---
    if (req.method === "POST" && url.pathname === "/task") {
      const body = await readBody(req);
      let prompt: string;
      try {
        prompt = JSON.parse(body).prompt;
      } catch {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "invalid JSON or missing 'prompt'" }));
        return;
      }

      const win = BrowserWindow.getAllWindows()[0];
      if (!win) {
        res.writeHead(503, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "no renderer window" }));
        return;
      }

      const taskId = String(++taskIdCounter);
      // Timeout after 5 minutes
      const timer = setTimeout(
        () => {
          pendingTasks.delete(taskId);
          res.writeHead(504, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "task timed out" }));
        },
        5 * 60 * 1000
      );

      pendingTasks.set(taskId, { res, timer });
      win.webContents.send("debug:task", { taskId, prompt });
      return;
    }

    // --- 404 ---
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "not found" }));
  });

  server.on("error", (err: NodeJS.ErrnoException) => {
    appendLog("warning", `Debug HTTP server failed to start: ${err.message}`);
  });

  server.listen(9960, "127.0.0.1", () => {
    appendLog("info", "Debug HTTP server listening on http://127.0.0.1:9960");
  });
}

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks).toString()));
    req.on("error", reject);
  });
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
  // Bypass CORS for LLM API requests — the renderer calls the LLM provider
  // directly (e.g. ark.cn-beijing.volces.com) which doesn't set CORS headers.
  const appOrigin = new URL(APP_URL).origin;
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    const url = details.url;
    // Only patch responses from external origins (not the app itself)
    if (!url.startsWith(appOrigin)) {
      const headers = details.responseHeaders ?? {};
      headers["access-control-allow-origin"] = ["*"];
      headers["access-control-allow-headers"] = ["*"];
      headers["access-control-allow-methods"] = [
        "GET, POST, PUT, DELETE, OPTIONS"
      ];
      callback({ responseHeaders: headers });
      return;
    }
    callback({});
  });

  setupScreenControl();
  setupPowerShellHandler();
  setupFileSystemHandlers();
  setupDebugHttpServer();
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

app.on("will-quit", () => {
  cleanupCloudDrive();
});
