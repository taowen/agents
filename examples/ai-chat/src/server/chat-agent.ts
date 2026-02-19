import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { AIChatAgent } from "@cloudflare/ai-chat";
import type { Connection } from "agents";
import type { Schedule } from "agents";
import * as Sentry from "@sentry/cloudflare";
import { instrumentDurableObjectWithSentry } from "@sentry/cloudflare";
import {
  streamText,
  generateText,
  convertToModelMessages,
  pruneMessages,
  tool,
  stepCountIs,
  type ToolSet
} from "ai";
import { z } from "zod";
import {
  parseFstab,
  DEFAULT_FSTAB,
  createMountCommands,
  createGitCommands,
  mountEntry,
  D1FsAdapter,
  R2FsAdapter,
  GoogleDriveFsAdapter
} from "vfs";
import type { FsTypeRegistry, FstabEntry } from "vfs";
import { Bash, InMemoryFs, MountableFs } from "just-bash";
import type { LlmFileConfig } from "../client/api";
import {
  createBrowserState,
  createBrowserTool,
  type BrowserState
} from "./browser-tool";
import { createSessionsCommand } from "./session-commands";
import { buildSystemPrompt } from "./system-prompt";
import { MIME_TYPES, getExtension } from "../shared/file-utils";
import { normalizePath } from "vfs";

/**
 * AI Chat Agent with sandboxed bash tool via just-bash.
 * Filesystem is backed by D1, scoped to the authenticated user.
 */
class ChatAgentBase extends AIChatAgent {
  maxPersistedMessages = 200;

  private bash!: Bash;
  private mountableFs!: MountableFs;
  private mounted = false;
  private mountPromise: Promise<void> | null = null;
  private browserState: BrowserState;
  private userId: string | null = null;
  private sessionUuid: string | null = null;
  private cachedLlmConfig: {
    data: LlmFileConfig | null;
    fetchedAt: number;
  } | null = null;
  private static SETTINGS_TTL = 60_000; // 60 seconds

  private cachedBridgeDevices: {
    data: { deviceName: string }[];
    fetchedAt: number;
  } | null = null;
  private static BRIDGE_DEVICES_TTL = 30_000; // 30 seconds
  private mcpServersLoaded = false;

  // System prompt cache — computed once per session, invalidated on clear history
  private cachedSystemPrompt: string | null = null;
  private cachedDynamicContext: string | null = null;
  private cachedLlmModel: Awaited<
    ReturnType<ChatAgentBase["getLlmModel"]>
  > | null = null;
  private cachedTools: ToolSet | null = null;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.browserState = createBrowserState();
  }

  private async getCachedLlmConfig(): Promise<LlmFileConfig | null> {
    const now = Date.now();
    if (
      this.cachedLlmConfig &&
      now - this.cachedLlmConfig.fetchedAt < ChatAgentBase.SETTINGS_TTL
    ) {
      return this.cachedLlmConfig.data;
    }
    try {
      const buf = await this.mountableFs.readFileBuffer("/etc/llm.json");
      const data = JSON.parse(new TextDecoder().decode(buf)) as LlmFileConfig;
      this.cachedLlmConfig = { data, fetchedAt: now };
      return data;
    } catch {
      this.cachedLlmConfig = { data: null, fetchedAt: now };
      return null;
    }
  }

  /**
   * Build the filesystem type registry for fstab-driven mounting.
   */
  private buildFsTypeRegistry(): FsTypeRegistry {
    return {
      d1: (entry: FstabEntry) =>
        new D1FsAdapter(this.env.DB, this.userId!, entry.mountPoint),
      r2: (entry: FstabEntry) =>
        this.env.R2
          ? new R2FsAdapter(this.env.R2, this.userId!, entry.mountPoint)
          : null,
      // Legacy compat: treat agentfs as d1
      agentfs: (entry: FstabEntry) =>
        new D1FsAdapter(this.env.DB, this.userId!, entry.mountPoint),
      gdrive: (entry: FstabEntry) =>
        this.env.GOOGLE_CLIENT_ID && this.env.GOOGLE_CLIENT_SECRET
          ? new GoogleDriveFsAdapter(
              this.env.DB,
              this.userId!,
              entry.mountPoint,
              this.env.GOOGLE_CLIENT_ID,
              this.env.GOOGLE_CLIENT_SECRET,
              entry.options.root_folder_id || undefined
            )
          : null
    };
  }

  /**
   * Initialize bash + filesystem for the given userId.
   * Phase 1 (bootstrap): only mount /etc → D1 so fstab can be read.
   * Phase 2 (doFstabMount): reads fstab and mounts remaining entries.
   */
  private initBash(userId: string): void {
    if (this.bash && this.userId === userId) return;
    this.userId = userId;

    const db = this.env.DB;
    const inMemoryFs = new InMemoryFs();
    inMemoryFs.mkdirSync("/mnt");
    const fs = new MountableFs({ base: inMemoryFs });
    this.mountableFs = fs;

    // Phase 1: bootstrap — mount only /etc so fstab can be read
    const etcFs = new D1FsAdapter(db, userId, "/etc");
    fs.mount("/etc", etcFs, "d1");

    const mountOptions = {
      fsTypeRegistry: this.buildFsTypeRegistry(),
      protectedMounts: ["/etc", "/home/user", "/data"],
      r2Bucket: this.env.R2,
      userId
    };

    this.bash = new Bash({
      fs,
      customCommands: [
        ...createMountCommands(fs, undefined, mountOptions),
        ...createGitCommands(fs, mountOptions),
        createSessionsCommand(db, userId, this.env.ChatAgent)
      ],
      cwd: "/home/user",
      network: { dangerouslyAllowFullInternetAccess: true },
      executionLimits: {
        maxCommandCount: 1000,
        maxLoopIterations: 1000,
        maxCallDepth: 50,
        maxStringLength: 1_048_576
      }
    });

    // Reset mount state for new user context
    this.mounted = false;
    this.mountPromise = null;
  }

  /**
   * Ensure fstab-declared mounts are applied. Called once before first bash exec.
   */
  private async ensureMounted(): Promise<void> {
    if (this.mounted) return;
    if (!this.mountPromise) {
      this.mountPromise = this.doFstabMount().then(
        () => {
          this.mounted = true;
        },
        (err) => {
          console.error("ensureMounted: fstab mount failed:", err);
          this.mountPromise = null;
          throw err;
        }
      );
    }
    await this.mountPromise;
  }

  /**
   * Read /etc/fstab via D1FsAdapter, parse entries, mount remaining entries.
   * Phase 2 of the two-phase boot: /etc is already mounted from Phase 1.
   */
  private async doFstabMount(): Promise<void> {
    return Sentry.startSpan({ name: "doFstabMount", op: "mount" }, async () => {
      const db = this.env.DB;
      const userId = this.userId!;

      // Batch: check /etc, read fstab, check /home/user — 1 round trip
      const [etcRow, fstabRow, homeRow] = await Sentry.startSpan(
        { name: "fstab.batchRead", op: "db.query" },
        () =>
          db.batch([
            db
              .prepare("SELECT 1 FROM files WHERE user_id=? AND path=?")
              .bind(userId, "/etc"),
            db
              .prepare(
                "SELECT CAST(content AS TEXT) as content FROM files WHERE user_id=? AND path=?"
              )
              .bind(userId, "/etc/fstab"),
            db
              .prepare("SELECT 1 FROM files WHERE user_id=? AND path=?")
              .bind(userId, "/home/user")
          ])
      );

      // Collect follow-up writes needed
      const writeStmts: D1PreparedStatement[] = [];

      if (!etcRow.results.length) {
        writeStmts.push(
          db
            .prepare(
              `INSERT OR IGNORE INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
             VALUES (?, ?, ?, ?, NULL, 1, 16877, 0, unixepoch('now'))`
            )
            .bind(userId, "/etc", "/", "etc")
        );
      }

      if (!homeRow.results.length) {
        writeStmts.push(
          db
            .prepare(
              `INSERT OR IGNORE INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
             VALUES (?, ?, ?, ?, NULL, 1, 16877, 0, unixepoch('now'))`
            )
            .bind(userId, "/home", "/", "home")
        );
        writeStmts.push(
          db
            .prepare(
              `INSERT OR IGNORE INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
             VALUES (?, ?, ?, ?, NULL, 1, 16877, 0, unixepoch('now'))`
            )
            .bind(userId, "/home/user", "/home", "user")
        );
      }

      let fstabContent: string;
      const fstabResult = fstabRow.results[0] as
        | { content: string | null }
        | undefined;
      if (fstabResult?.content) {
        fstabContent = fstabResult.content;
      } else {
        fstabContent = DEFAULT_FSTAB;
        const encoded = new TextEncoder().encode(fstabContent);
        writeStmts.push(
          db
            .prepare(
              `INSERT INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
             VALUES (?, ?, ?, ?, ?, 0, 33188, ?, unixepoch('now'))
             ON CONFLICT(user_id, path) DO UPDATE SET content=excluded.content, size=excluded.size, mtime=unixepoch('now')`
            )
            .bind(
              userId,
              "/etc/fstab",
              "/etc",
              "fstab",
              encoded,
              encoded.length
            )
        );
      }

      if (writeStmts.length > 0) {
        await Sentry.startSpan(
          { name: "fstab.batchWrite", op: "db.query" },
          () => db.batch(writeStmts)
        );
      }

      // Migration: auto-upgrade old agentfs-only fstab to new d1/r2 format
      let entries = parseFstab(fstabContent);
      const hasD1OrR2 = entries.some((e) => e.type === "d1" || e.type === "r2");
      if (!hasD1OrR2) {
        // Preserve git mount lines, replace everything else with new default
        const gitLines = fstabContent.split("\n").filter((line) => {
          const t = line.trim();
          if (!t || t.startsWith("#")) return false;
          return t.split(/\s+/)[2] === "git";
        });
        fstabContent = DEFAULT_FSTAB;
        if (gitLines.length > 0) {
          fstabContent += gitLines.join("\n") + "\n";
        }
        // Write upgraded fstab back to D1
        try {
          const encoded = new TextEncoder().encode(fstabContent);
          await db
            .prepare(
              `INSERT INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
               VALUES (?, ?, ?, ?, ?, 0, 33188, ?, unixepoch('now'))
               ON CONFLICT(user_id, path) DO UPDATE SET content=excluded.content, size=excluded.size, mtime=unixepoch('now')`
            )
            .bind(
              userId,
              "/etc/fstab",
              "/etc",
              "fstab",
              encoded,
              encoded.length
            )
            .run();
        } catch (e) {
          console.error("fstab migration write failed:", e);
        }
        entries = parseFstab(fstabContent);
      }

      // Mount all entries using mountEntry from vfs (skips /etc which is already mounted)
      const mountOptions = {
        fsTypeRegistry: this.buildFsTypeRegistry(),
        protectedMounts: ["/etc", "/home/user", "/data"],
        r2Bucket: this.env.R2,
        userId
      };
      for (const entry of entries) {
        try {
          await Sentry.startSpan(
            { name: `mount ${entry.mountPoint} (${entry.type})`, op: "mount" },
            () => mountEntry(entry, null, this.mountableFs, mountOptions)
          );
        } catch (e) {
          console.error(`fstab: mount failed for ${entry.mountPoint}:`, e);
        }
      }
    });
  }

  /**
   * Read MCP server config from /etc/mcp-servers.json.
   */
  private async readMcpConfig(): Promise<
    { name: string; url: string; headers?: Record<string, string> }[]
  > {
    try {
      const buf = await this.mountableFs.readFileBuffer(
        "/etc/mcp-servers.json"
      );
      const text = new TextDecoder().decode(buf);
      return JSON.parse(text);
    } catch {
      return [];
    }
  }

  /**
   * Write MCP server config to /etc/mcp-servers.json.
   */
  private async writeMcpConfig(
    entries: { name: string; url: string; headers?: Record<string, string> }[]
  ): Promise<void> {
    const text = JSON.stringify(entries, null, 2);
    await this.mountableFs.writeFile("/etc/mcp-servers.json", text);
  }

  /**
   * Auto-connect MCP servers from /etc/mcp-servers.json on session start.
   */
  private async ensureMcpServers(): Promise<void> {
    if (this.mcpServersLoaded) return;
    this.mcpServersLoaded = true;

    const config = await this.readMcpConfig();
    if (config.length === 0) return;

    const existing = this.getMcpServers();
    const connectedNames = new Set(
      Object.values(existing.servers).map((s) => s.name)
    );

    const callbackHost =
      (await this.ctx.storage.get<string>("callbackOrigin")) ?? "";
    const callbackPath = `mcp-callback/${this.ctx.id.toString()}`;

    for (const entry of config) {
      if (connectedNames.has(entry.name)) continue;
      try {
        await this.addMcpServer(entry.name, entry.url, {
          callbackHost,
          callbackPath,
          transport: entry.headers ? { headers: entry.headers } : undefined
        });
      } catch (e) {
        console.error(
          `ensureMcpServers: failed to connect "${entry.name}":`,
          e
        );
      }
    }
  }

  /**
   * Build the LLM model instance based on user settings.
   */
  private async getLlmModel() {
    const config = this.userId ? await this.getCachedLlmConfig() : null;

    if (!config) {
      // Builtin: no /etc/llm.json means use built-in provider
      const apiKey = this.env.ARK_API_KEY;
      const baseURL = "https://ark.cn-beijing.volces.com/api/v3";
      const model = "doubao-seed-2-0-pro-260215";
      return createOpenAICompatible({
        name: "llm",
        baseURL,
        apiKey,
        includeUsage: true
      })(model);
    }

    const { provider, api_key: apiKey, base_url: baseURL, model } = config;

    return provider === "openai-compatible"
      ? createOpenAICompatible({
          name: "llm",
          baseURL,
          apiKey,
          includeUsage: true
        })(model)
      : createGoogleGenerativeAI({ baseURL, apiKey })(model);
  }

  /**
   * Callback invoked by the DO alarm for scheduled/recurring tasks.
   */
  /** Read the user's timezone from DO storage, defaulting to UTC. */
  private async getTimezone(): Promise<string> {
    const stored = await this.ctx.storage.get<string>("timezone");
    return stored || "UTC";
  }

  async executeScheduledTask(
    payload: { description: string; prompt: string; timezone?: string },
    schedule: Schedule<{
      description: string;
      prompt: string;
      timezone?: string;
    }>
  ) {
    // Restore state (memory may be empty after alarm wake)
    if (!this.userId) {
      const uid = await this.getUserId();
      this.initBash(uid);
    }

    const model = await this.getLlmModel();

    const now = new Date();
    const tz = payload.timezone || (await this.getTimezone());

    const result = await generateText({
      model,
      system:
        "You are a scheduled task executor. Execute the task and report the result.\n" +
        `Current UTC time: ${now.toISOString()}\n` +
        `User timezone: ${tz}`,
      prompt: payload.prompt,
      tools: {
        bash: this.createBashTool(),
        browser: createBrowserTool(this.browserState, this.env.MYBROWSER)
      },
      stopWhen: stepCountIs(10)
    });

    // Inject result into chat history
    const userMsg = {
      id: crypto.randomUUID(),
      role: "user" as const,
      parts: [
        {
          type: "text" as const,
          text: `[Scheduled Task] ${new Date().toISOString()} - ${payload.description}`
        }
      ]
    };
    const assistantMsg = {
      id: crypto.randomUUID(),
      role: "assistant" as const,
      parts: [{ type: "text" as const, text: result.text }]
    };
    await this.persistMessages([...this.messages, userMsg, assistantMsg]);
  }

  /**
   * Extract userId from request header, with fallback to DO-local storage.
   */
  private async getUserId(request?: Request): Promise<string> {
    if (request) {
      const uid = request.headers.get("x-user-id");
      if (uid) {
        // Persist for hibernation recovery
        await this.ctx.storage.put("userId", uid);
        return uid;
      }
    }
    // Recover from storage (after hibernation)
    const stored = await this.ctx.storage.get<string>("userId");
    if (stored) return stored;
    throw new Error("No userId available");
  }

  /**
   * Extract sessionUuid from request header, with fallback to DO-local storage.
   */
  private async getSessionUuid(request?: Request): Promise<string | null> {
    if (request) {
      const sid = request.headers.get("x-session-id");
      if (sid) {
        await this.ctx.storage.put("sessionUuid", sid);
        this.sessionUuid = sid;
        return sid;
      }
    }
    if (this.sessionUuid) return this.sessionUuid;
    const stored = await this.ctx.storage.get<string>("sessionUuid");
    if (stored) {
      this.sessionUuid = stored;
      return stored;
    }
    return null;
  }

  onError(connectionOrError: Connection | unknown, error?: unknown): void {
    if (error !== undefined) {
      console.error("ChatAgent error:", error);
      if (error instanceof Error) Sentry.captureException(error);
      super.onError(connectionOrError as Connection, error);
    } else {
      console.error("ChatAgent error:", connectionOrError);
      if (connectionOrError instanceof Error)
        Sentry.captureException(connectionOrError);
    }
  }

  async onRequest(request: Request): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname.endsWith("/get-schedules")) {
      const schedules = this.getSchedules();
      return Response.json(schedules);
    }
    if (url.pathname.startsWith("/api/files")) {
      return this.handleFileRequest(request);
    }
    return super.onRequest(request);
  }

  /**
   * Handle file manager API requests using the DO's MountableFs.
   * This ensures the file manager sees the exact same filesystem as the bash agent,
   * including git mounts under /mnt.
   */
  private async handleFileRequest(request: Request): Promise<Response> {
    // Ensure bash + mounts are initialized
    const uid = await this.getUserId(request);
    this.initBash(uid);
    await this.ensureMounted();

    const url = new URL(request.url);
    const fs = this.mountableFs;

    try {
      // GET /api/files/stat?path=<path>
      if (url.pathname === "/api/files/stat" && request.method === "GET") {
        const rawPath = url.searchParams.get("path") || "/";
        const path = normalizePath(rawPath);
        const st = await fs.stat(path);
        return Response.json({
          isFile: st.isFile,
          isDirectory: st.isDirectory,
          isSymbolicLink: st.isSymbolicLink,
          mode: st.mode,
          size: st.size,
          mtime: st.mtime?.toISOString() ?? null
        });
      }

      // GET /api/files/list?path=<dir>
      if (url.pathname === "/api/files/list" && request.method === "GET") {
        const rawPath = url.searchParams.get("path") || "/";
        const path = normalizePath(rawPath);

        const names = await fs.readdir(path);
        const entries = await Promise.all(
          names.map(async (name: string) => {
            try {
              const childPath = path === "/" ? `/${name}` : `${path}/${name}`;
              const st = await fs.stat(childPath);
              return {
                name,
                isDirectory: st.isDirectory,
                size: st.size,
                mtime: st.mtime?.toISOString() ?? null
              };
            } catch {
              return { name, isDirectory: false, size: 0, mtime: null };
            }
          })
        );
        return Response.json({ entries });
      }

      // GET /api/files/content?path=<file>
      if (url.pathname === "/api/files/content" && request.method === "GET") {
        const rawPath = url.searchParams.get("path") || "";
        const path = normalizePath(rawPath);
        const buffer = await fs.readFileBuffer(path);
        const ext = getExtension(rawPath);
        const contentType = MIME_TYPES[ext] || "application/octet-stream";
        return new Response(buffer as BodyInit, {
          headers: { "Content-Type": contentType }
        });
      }

      // PUT /api/files/content?path=<file>
      if (url.pathname === "/api/files/content" && request.method === "PUT") {
        const rawPath = url.searchParams.get("path") || "";
        const path = normalizePath(rawPath);
        const buffer = new Uint8Array(await request.arrayBuffer());
        await fs.writeFile(path, buffer);
        return Response.json({ ok: true });
      }

      // POST /api/files/mkdir
      if (url.pathname === "/api/files/mkdir" && request.method === "POST") {
        const body = (await request.json()) as { path?: string };
        const rawPath = body.path || "";
        const path = normalizePath(rawPath);
        await fs.mkdir(path, { recursive: true });
        return Response.json({ ok: true });
      }

      // DELETE /api/files?path=<path>&recursive=0|1
      if (url.pathname === "/api/files" && request.method === "DELETE") {
        const rawPath = url.searchParams.get("path") || "";
        const path = normalizePath(rawPath);
        const recursive = url.searchParams.get("recursive") === "1";
        await fs.rm(path, { recursive });
        return Response.json({ ok: true });
      }

      return Response.json({ error: "Not found" }, { status: 404 });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("ENOENT")) {
        return Response.json({ error: msg }, { status: 404 });
      }
      if (msg.includes("EISDIR")) {
        return Response.json({ error: msg }, { status: 400 });
      }
      if (msg.includes("EBUSY")) {
        return Response.json({ error: msg }, { status: 400 });
      }
      return Response.json({ error: msg }, { status: 500 });
    }
  }

  async onConnect(
    connection: import("agents").Connection,
    ctx: import("agents").ConnectionContext
  ) {
    // Capture userId + sessionUuid from the initial WebSocket upgrade request
    const uid = await this.getUserId(ctx.request);
    await this.getSessionUuid(ctx.request);
    this.initBash(uid);
    // Persist origin for MCP OAuth callback URL construction
    const origin = new URL(ctx.request.url).origin;
    await this.ctx.storage.put("callbackOrigin", origin);
    return super.onConnect(connection, ctx);
  }

  /** Short session directory name derived from the DO ID. */
  private get sessionDir(): string {
    return this.ctx.id.toString().slice(0, 12);
  }

  /**
   * Write chat history to /home/user/.chat/{sessionDir}/ as per-message files.
   * All writes use INSERT OR IGNORE — only new messages actually hit disk.
   */
  private async writeChatHistory() {
    const userId = this.userId;
    if (!userId || this.messages.length === 0) return;
    const db = this.env.DB;
    const enc = new TextEncoder();
    const sd = this.sessionDir;
    const base = `/home/user/.chat/${sd}`;
    const toolsDir = `${base}/tools`;

    const mkdirSql = `INSERT OR IGNORE INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
       VALUES (?, ?, ?, ?, NULL, 1, 16877, 0, unixepoch('now'))`;
    const fileSql = `INSERT OR IGNORE INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
       VALUES (?, ?, ?, ?, ?, 0, 33188, ?, unixepoch('now'))`;

    const stmts: D1PreparedStatement[] = [
      db
        .prepare(mkdirSql)
        .bind(userId, "/home/user/.chat", "/home/user", ".chat"),
      db.prepare(mkdirSql).bind(userId, base, "/home/user/.chat", sd),
      db.prepare(mkdirSql).bind(userId, toolsDir, base, "tools"),
      // Ensure .memory/ directory exists
      db
        .prepare(mkdirSql)
        .bind(userId, "/home/user/.memory", "/home/user", ".memory")
    ];

    // Write .meta.md (INSERT OR IGNORE — only on first write)
    const firstUserMsg = this.messages.find((m) => m.role === "user");
    let title = "Untitled";
    if (firstUserMsg) {
      for (const part of firstUserMsg.parts) {
        if (part.type === "text" && part.text) {
          title = part.text.slice(0, 100).replace(/\n/g, " ");
          break;
        }
      }
    }
    const date = new Date().toISOString().slice(0, 10);
    const sessionUuid = this.sessionUuid || "unknown";
    const metaContent = `title: ${title}\ndate: ${date}\nsession: ${sessionUuid}\n`;
    const metaBuf = enc.encode(metaContent);
    stmts.push(
      db
        .prepare(fileSql)
        .bind(
          userId,
          `${base}/.meta.md`,
          base,
          ".meta.md",
          metaBuf,
          metaBuf.length
        )
    );

    for (let i = 0; i < this.messages.length; i++) {
      const msg = this.messages[i];
      const num = String(i + 1).padStart(4, "0");
      let text = "";

      for (const part of msg.parts) {
        if (part.type === "text") {
          text += part.text + "\n";
        } else if ("toolCallId" in part && part.toolCallId) {
          const p = part as {
            type: string;
            toolCallId: string;
            toolName?: string;
            input?: Record<string, unknown>;
            output?: unknown;
          };
          const toolName = p.toolName || p.type.replace("tool-", "");
          const shortId = p.toolCallId.slice(0, 8);
          const inputStr =
            toolName === "bash" && p.input?.command
              ? `\`${p.input.command}\``
              : JSON.stringify(p.input || {}).slice(0, 100);
          text += `[${toolName}(${inputStr}) → tools/${shortId}.txt]\n`;

          if (p.output != null) {
            let output: string;
            if (
              toolName === "bash" &&
              typeof p.output === "object" &&
              p.output !== null
            ) {
              const r = p.output as {
                stdout?: string;
                stderr?: string;
                exitCode?: number;
              };
              output = `$ exit ${r.exitCode ?? "?"}\n`;
              if (r.stdout) output += r.stdout;
              if (r.stderr) output += `\n--- stderr ---\n${r.stderr}`;
            } else {
              output =
                typeof p.output === "string"
                  ? p.output
                  : JSON.stringify(p.output, null, 2);
            }
            const buf = enc.encode(output);
            stmts.push(
              db
                .prepare(fileSql)
                .bind(
                  userId,
                  `${toolsDir}/${shortId}.txt`,
                  toolsDir,
                  `${shortId}.txt`,
                  buf,
                  buf.length
                )
            );
          }
        }
      }

      const fileName = `${num}-${msg.role}.md`;
      const buf = enc.encode(text);
      stmts.push(
        db
          .prepare(fileSql)
          .bind(userId, `${base}/${fileName}`, base, fileName, buf, buf.length)
      );
    }

    for (let i = 0; i < stmts.length; i += 100) {
      await db.batch(stmts.slice(i, i + 100));
    }
  }

  private createBashTool() {
    return tool({
      description:
        "Execute a bash command in a sandboxed virtual filesystem. " +
        "Supports ls, grep, awk, sed, find, cat, echo, mkdir, cp, mv, sort, uniq, wc, head, tail, curl, and more. " +
        "Use curl to fetch content from URLs. Files persist across commands within the session.",
      inputSchema: z.object({
        command: z.string().describe("The bash command to execute")
      }),
      execute: async ({ command }) => {
        return Sentry.startSpan(
          { name: `bash: ${command.slice(0, 80)}`, op: "tool.bash" },
          async () => {
            await Sentry.startSpan({ name: "ensureMounted", op: "mount" }, () =>
              this.ensureMounted()
            );
            return await Sentry.startSpan(
              { name: "bash.exec", op: "exec" },
              () => this.bash.exec(command)
            );
          }
        );
      }
    });
  }

  private async readMemoryBlock(): Promise<string> {
    const memPaths: [string, string][] = [
      ["/home/user/.memory/profile.md", "User Profile"],
      ["/home/user/.memory/preferences.md", "User Preferences"],
      ["/home/user/.memory/entities.md", "Known Entities"]
    ];
    try {
      const memResults = await this.env.DB.batch(
        memPaths.map(([p]) =>
          this.env.DB.prepare(
            "SELECT CAST(content AS TEXT) as content FROM files WHERE user_id=? AND path=?"
          ).bind(this.userId!, p)
        )
      );
      const sections: string[] = [];
      for (let i = 0; i < memPaths.length; i++) {
        const row = memResults[i].results[0] as
          | { content: string | null }
          | undefined;
        if (row?.content) {
          sections.push(`## ${memPaths[i][1]}\n${row.content}`);
        }
      }
      if (sections.length > 0) {
        return "\n\n# Memory\n" + sections.join("\n\n");
      }
    } catch (e) {
      console.error("memory read:", e);
    }
    return "";
  }

  /**
   * Fetch the list of connected remote desktop devices from BridgeManager DO.
   * Cached for 30s to avoid hammering the DO on every message.
   */
  private async getAvailableBridgeDevices(): Promise<{ deviceName: string }[]> {
    const now = Date.now();
    if (
      this.cachedBridgeDevices &&
      now - this.cachedBridgeDevices.fetchedAt <
        ChatAgentBase.BRIDGE_DEVICES_TTL
    ) {
      return this.cachedBridgeDevices.data;
    }
    try {
      const id = this.env.BridgeManager.idFromName(this.userId!);
      const stub = this.env.BridgeManager.get(id);
      const resp = await stub.fetch(
        new Request("http://bridge/devices", {
          method: "GET",
          headers: { "x-partykit-room": this.userId! }
        })
      );
      const data = (await resp.json()) as { deviceName: string }[];
      this.cachedBridgeDevices = { data, fetchedAt: now };
      return data;
    } catch (e) {
      console.error("getAvailableBridgeDevices:", e);
      return [];
    }
  }

  /**
   * Send a message to a remote desktop agent via BridgeManager and wait for the response.
   */
  private async sendToRemoteDesktop(
    deviceName: string,
    content: string
  ): Promise<string> {
    const id = this.env.BridgeManager.idFromName(this.userId!);
    const stub = this.env.BridgeManager.get(id);
    const resp = await stub.fetch(
      new Request("http://bridge/message", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-partykit-room": this.userId!
        },
        body: JSON.stringify({ deviceName, content })
      })
    );
    const data = (await resp.json()) as { response?: string; error?: string };
    if (data.error) {
      return `[Error] ${data.error}`;
    }
    return data.response || "[No response from remote desktop]";
  }

  private createTools(bridgeDevices?: { deviceName: string }[]): ToolSet {
    const tools: ToolSet = {
      bash: this.createBashTool(),
      browser: createBrowserTool(this.browserState, this.env.MYBROWSER),
      schedule_task: tool({
        description:
          "Schedule a one-time task. Provide EITHER delaySeconds (e.g. 60 for 1 minute) " +
          "OR scheduledAt (ISO 8601 datetime). Prefer delaySeconds for relative times like 'in 5 minutes'.",
        inputSchema: z.object({
          description: z.string().describe("Brief description of the task"),
          prompt: z
            .string()
            .describe(
              "The detailed prompt/instruction for the AI to execute when the task fires"
            ),
          delaySeconds: z
            .number()
            .optional()
            .describe(
              "Delay in seconds from now. e.g. 60 = 1 minute, 3600 = 1 hour. Use this for relative times."
            ),
          scheduledAt: z
            .string()
            .optional()
            .describe(
              "ISO 8601 datetime string for absolute time, e.g. '2025-06-01T09:00:00Z'"
            )
        }),
        execute: async ({ description, prompt, delaySeconds, scheduledAt }) => {
          let when: Date | number;
          if (delaySeconds != null) {
            if (delaySeconds <= 0)
              return { error: "delaySeconds must be positive" };
            when = delaySeconds;
          } else if (scheduledAt) {
            when = new Date(scheduledAt);
            if (when.getTime() <= Date.now())
              return { error: "Scheduled time must be in the future" };
          } else {
            return { error: "Provide either delaySeconds or scheduledAt" };
          }
          const tz = await this.getTimezone();
          const s = await this.schedule(when, "executeScheduledTask" as any, {
            description,
            prompt,
            timezone: tz
          });
          return {
            success: true,
            id: s.id,
            scheduledAt: new Date(s.time * 1000).toISOString(),
            description
          };
        }
      }),
      schedule_recurring: tool({
        description:
          "Schedule a recurring task using a cron expression. " +
          "Examples: '0 9 * * *' = daily at 9am UTC, '0 */2 * * *' = every 2 hours, '0 9 * * 1-5' = weekdays at 9am UTC.",
        inputSchema: z.object({
          description: z
            .string()
            .describe("Brief description of the recurring task"),
          prompt: z
            .string()
            .describe(
              "The detailed prompt/instruction for the AI to execute each time"
            ),
          cron: z
            .string()
            .describe(
              "Cron expression (5 fields: minute hour day-of-month month day-of-week)"
            )
        }),
        execute: async ({ description, prompt, cron }) => {
          const tz = await this.getTimezone();
          const s = await this.schedule(cron, "executeScheduledTask" as any, {
            description,
            prompt,
            timezone: tz
          });
          return {
            success: true,
            id: s.id,
            cron,
            description,
            nextRun: new Date(s.time * 1000).toISOString()
          };
        }
      }),
      manage_tasks: tool({
        description:
          "List all scheduled tasks, or cancel a specific task by ID.",
        inputSchema: z.object({
          action: z.enum(["list", "cancel"]).describe("Action to perform"),
          taskId: z
            .string()
            .optional()
            .describe("Task ID to cancel (required for cancel action)")
        }),
        execute: async ({ action, taskId }) => {
          if (action === "list") {
            const schedules = this.getSchedules();
            return schedules.map((s) => {
              let description = "";
              try {
                const p =
                  typeof s.payload === "string"
                    ? JSON.parse(s.payload)
                    : s.payload;
                description = p.description || "";
              } catch {}
              return {
                id: s.id,
                type: s.type,
                description,
                nextRun: new Date(s.time * 1000).toISOString(),
                ...(s.type === "cron" ? { cron: (s as any).cron } : {})
              };
            });
          }
          if (action === "cancel" && taskId) {
            const ok = await this.cancelSchedule(taskId);
            return ok
              ? { success: true, cancelled: taskId }
              : { error: "Task not found" };
          }
          return { error: "Invalid action or missing taskId" };
        }
      })
    };

    // Add remote_desktop tool when bridge devices are available
    if (bridgeDevices && bridgeDevices.length > 0) {
      tools.remote_desktop = tool({
        description:
          "Send a message to a connected remote desktop agent. " +
          "The remote agent can see the screen, control mouse/keyboard, and execute commands. " +
          "It maintains conversation context across calls — you can give follow-up instructions. " +
          "Describe what you want done in natural language. Returns the agent's text response.",
        inputSchema: z.object({
          message: z
            .string()
            .describe("What to do on the remote desktop, in natural language"),
          device: z
            .string()
            .optional()
            .describe("Device name (omit if only one device)")
        }),
        execute: async ({ message, device }) => {
          const targetDevice =
            device ||
            (bridgeDevices.length === 1
              ? bridgeDevices[0].deviceName
              : "default");
          return this.sendToRemoteDesktop(targetDevice, message);
        }
      });
    }

    return tools;
  }

  async onChatMessage(
    _onFinish?: unknown,
    options?: { body?: Record<string, unknown> }
  ) {
    return Sentry.startSpan({ name: "onChatMessage", op: "chat" }, async () => {
      // Ensure we have userId + sessionUuid (may need recovery after hibernation)
      if (!this.userId) {
        await Sentry.startSpan(
          { name: "getUserId", op: "db.query" },
          async () => {
            const uid = await this.getUserId();
            this.initBash(uid);
          }
        );
      }
      if (!this.sessionUuid) {
        await this.getSessionUuid();
      }

      // Persist client-reported timezone for scheduled tasks and hibernation recovery
      const clientTz = options?.body?.timezone;
      if (typeof clientTz === "string" && clientTz) {
        await this.ctx.storage.put("timezone", clientTz);
      }
      const timezone =
        (typeof clientTz === "string" && clientTz) ||
        (await this.getTimezone());

      // Configure bash TZ env var and /etc/timezone so commands discover time natively
      this.bash.setEnv("TZ", timezone);
      await this.mountableFs.writeFile("/etc/timezone", timezone + "\n");

      // Fire-and-forget: sync chat history to /home/user/.chat/
      this.writeChatHistory().catch((e) =>
        console.error("writeChatHistory:", e)
      );

      // Auto-connect MCP servers from /etc/mcp-servers.json
      try {
        await this.ensureMcpServers();
      } catch (e) {
        console.error("ensureMcpServers failed:", e);
      }

      // Compute system prompt, dynamic context, LLM model, and tools once per session.
      // Invalidate when this is the first message or after clear history (messages.length <= 1).
      const shouldRecompute =
        !this.cachedSystemPrompt || this.messages.length <= 1;

      if (shouldRecompute) {
        // Ensure jsonSchema is initialized for getAITools() — needed after DO hibernation
        try {
          await this.mcp.ensureJsonSchema();
        } catch (e) {
          console.error("mcp.ensureJsonSchema failed:", e);
        }

        const [llmModel, memoryBlock, bridgeDevices] = await Promise.all([
          this.getLlmModel(),
          this.readMemoryBlock(),
          this.getAvailableBridgeDevices()
        ]);

        this.cachedLlmModel = llmModel;
        this.cachedSystemPrompt = buildSystemPrompt();

        // Build dynamic context
        const dynamicParts = [
          `Chat history directory: /home/user/.chat/${this.sessionDir}/`,
          memoryBlock
        ];
        if (bridgeDevices.length > 0) {
          const names = bridgeDevices.map((d) => d.deviceName).join(", ");
          dynamicParts.push(
            `\nConnected remote desktop devices: ${names}. ` +
              "Use the remote_desktop tool to send instructions to these devices. " +
              "The remote agent can see the screen, control mouse/keyboard, and maintains conversation context."
          );
        }
        // Inject connected MCP server info
        const mcpState = this.getMcpServers();
        const mcpEntries = Object.entries(mcpState.servers);
        if (mcpEntries.length > 0) {
          const mcpToolsList = mcpState.tools || [];
          const mcpLines = mcpEntries.map(([id, s]) => {
            const toolNames =
              mcpToolsList
                .filter((t) => t.serverId === id)
                .map((t) => t.name)
                .join(", ") || "none";
            return `- ${s.name} (${s.state}, id: ${id}): tools=[${toolNames}]`;
          });
          dynamicParts.push(`\nConnected MCP servers:\n${mcpLines.join("\n")}`);
        }
        this.cachedDynamicContext = dynamicParts.filter(Boolean).join("\n");

        // Cache tools (including MCP tools)
        let mcpTools: ToolSet = {};
        try {
          mcpTools = this.mcp.getAITools();
        } catch (e) {
          console.error("mcp.getAITools() failed:", e);
        }
        this.cachedTools = {
          ...this.createTools(bridgeDevices),
          ...mcpTools
        } as ToolSet;
      }

      const messages = await Sentry.startSpan(
        { name: "convertMessages", op: "serialize" },
        async () =>
          pruneMessages({
            messages: await convertToModelMessages(this.messages),
            toolCalls: "before-last-2-messages",
            reasoning: "before-last-message"
          })
      );

      messages.unshift({ role: "system", content: this.cachedDynamicContext! });

      const result = streamText({
        model: this.cachedLlmModel!,
        system: this.cachedSystemPrompt!,
        messages,
        tools: this.cachedTools!,
        stopWhen: stepCountIs(10)
      });

      return result.toUIMessageStreamResponse({
        messageMetadata: ({ part }) => {
          if (part.type === "finish") {
            return {
              usage: {
                inputTokens: part.totalUsage.inputTokens,
                outputTokens: part.totalUsage.outputTokens,
                cacheReadTokens:
                  part.totalUsage.inputTokenDetails?.cacheReadTokens,
                cacheWriteTokens:
                  part.totalUsage.inputTokenDetails?.cacheWriteTokens
              }
            };
          }
          return undefined;
        }
      });
    });
  }
}

export const ChatAgent = instrumentDurableObjectWithSentry(
  (env: Env) => ({
    dsn: env.SENTRY_DSN,
    tracesSampleRate: 1.0
  }),
  ChatAgentBase
);
