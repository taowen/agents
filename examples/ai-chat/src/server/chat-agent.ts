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
  stepCountIs
} from "ai";
import { z } from "zod";
import {
  parseFstab,
  DEFAULT_FSTAB,
  GitFs,
  parseGitCredentials,
  findCredential,
  syncDirtyGitMounts,
  D1FsAdapter,
  R2FsAdapter
} from "vfs";
import { Bash, InMemoryFs, MountableFs } from "just-bash";
import { getSettings, type UserSettings } from "./db";
import {
  createBrowserState,
  createBrowserTool,
  type BrowserState
} from "./browser-tool";
import { createD1MountCommands } from "./mount-commands";
import { createSessionsCommand } from "./session-commands";
import { buildSystemPrompt } from "./system-prompt";

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
  private cachedSettings: {
    data: UserSettings | null;
    fetchedAt: number;
  } | null = null;
  private static SETTINGS_TTL = 60_000; // 60 seconds

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.browserState = createBrowserState();
  }

  private async getCachedSettings(): Promise<UserSettings | null> {
    const now = Date.now();
    if (
      this.cachedSettings &&
      now - this.cachedSettings.fetchedAt < ChatAgentBase.SETTINGS_TTL
    ) {
      return this.cachedSettings.data;
    }
    return Sentry.startSpan(
      { name: "getCachedSettings", op: "db.query" },
      async () => {
        const data = await getSettings(this.env.DB, this.userId!);
        this.cachedSettings = { data, fetchedAt: now };
        return data;
      }
    );
  }

  /**
   * Initialize bash + filesystem for the given userId.
   * Must be called before first bash exec.
   */
  private initBash(userId: string): void {
    if (this.bash && this.userId === userId) return;
    this.userId = userId;

    const db = this.env.DB;
    const inMemoryFs = new InMemoryFs();
    inMemoryFs.mkdirSync("/mnt");
    const fs = new MountableFs({ base: inMemoryFs });
    this.mountableFs = fs;

    // Mount /etc and /home/user via D1FsAdapter
    const etcFs = new D1FsAdapter(db, userId, "/etc");
    fs.mount("/etc", etcFs);

    const homeFs = new D1FsAdapter(db, userId, "/home/user");
    fs.mount("/home/user", homeFs);

    // Mount /data via R2 (if R2 binding is available)
    if (this.env.R2) {
      const dataFs = new R2FsAdapter(this.env.R2, userId, "/data");
      fs.mount("/data", dataFs);
    }

    this.bash = new Bash({
      fs,
      customCommands: [
        ...createD1MountCommands(fs),
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
   * Read /etc/fstab via D1FsAdapter, parse entries, mount git entries.
   */
  private async doFstabMount(): Promise<void> {
    return Sentry.startSpan({ name: "doFstabMount", op: "mount" }, async () => {
      const db = this.env.DB;
      const userId = this.userId!;

      // Batch: check /etc, read fstab, check /home/user, read git-credentials — 1 round trip
      const [etcRow, fstabRow, homeRow, credRow] = await Sentry.startSpan(
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
              .bind(userId, "/home/user"),
            db
              .prepare(
                "SELECT CAST(content AS TEXT) as content FROM files WHERE user_id=? AND path=?"
              )
              .bind(userId, "/etc/git-credentials")
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

      let gitCredentials: string | null = null;
      const credResult = credRow.results[0] as
        | { content: string | null }
        | undefined;
      if (credResult?.content) {
        gitCredentials = credResult.content;
      }

      const entries = parseFstab(fstabContent);
      for (const entry of entries) {
        if (entry.type === "agentfs") continue;

        if (entry.type === "git") {
          const ref = entry.options.ref || "main";
          const depth = entry.options.depth
            ? parseInt(entry.options.depth, 10)
            : 1;

          let username = entry.options.username;
          let password = entry.options.password;

          if (!username && gitCredentials) {
            try {
              const creds = parseGitCredentials(gitCredentials);
              const match = findCredential(creds, entry.device);
              if (match) {
                username = match.username;
                password = match.password;
              }
            } catch (e) {
              Sentry.captureException(e, { level: "warning" });
            }
          }

          const onAuth = username
            ? () => ({ username: username!, password })
            : undefined;

          const gitFs = new GitFs({
            url: entry.device,
            ref,
            depth: isNaN(depth) || depth < 1 ? 1 : depth,
            onAuth
          });

          try {
            this.mountableFs.mount(entry.mountPoint, gitFs);
          } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            if (msg.includes("already mounted")) continue;
            console.error(`fstab: failed to mount ${entry.mountPoint}: ${msg}`);
            continue;
          }

          try {
            await Sentry.startSpan(
              { name: `git.clone ${entry.mountPoint}`, op: "git" },
              () => gitFs.init()
            );
          } catch (e) {
            const msg = e instanceof Error ? e.message : String(e);
            console.error(
              `fstab: clone failed for ${entry.device} at ${entry.mountPoint}: ${msg}`
            );
            try {
              this.mountableFs.unmount(entry.mountPoint);
            } catch (unmountErr) {
              Sentry.captureException(unmountErr, { level: "warning" });
            }
          }
        }
      }
    });
  }

  /**
   * Build the LLM model instance based on user settings.
   */
  private async getLlmModel() {
    let provider: string = "builtin";

    if (this.userId) {
      const settings = await this.getCachedSettings();
      if (settings?.llm_provider) provider = settings.llm_provider;
    }

    if (provider === "builtin") {
      const apiKey = this.env.ARK_API_KEY;
      const baseURL = "https://ark.cn-beijing.volces.com/api/v3";
      const model = "doubao-seed-2-0-pro-260215";
      return createOpenAICompatible({ name: "llm", baseURL, apiKey })(model);
    }

    let apiKey = this.env.GOOGLE_AI_API_KEY;
    let baseURL = "https://generativelanguage.googleapis.com/v1beta";
    let model = "gemini-2.0-flash";

    if (this.userId) {
      const settings = await this.getCachedSettings();
      if (settings?.llm_api_key) apiKey = settings.llm_api_key;
      if (settings?.llm_base_url) baseURL = settings.llm_base_url;
      if (settings?.llm_model) model = settings.llm_model;
    }

    return provider === "openai-compatible"
      ? createOpenAICompatible({ name: "llm", baseURL, apiKey })(model)
      : createGoogleGenerativeAI({ baseURL, apiKey })(model);
  }

  /**
   * Callback invoked by the DO alarm for scheduled/recurring tasks.
   */
  async executeScheduledTask(
    payload: { description: string; prompt: string },
    schedule: Schedule<{ description: string; prompt: string }>
  ) {
    // Restore state (memory may be empty after alarm wake)
    if (!this.userId) {
      const uid = await this.getUserId();
      this.initBash(uid);
    }

    const model = await this.getLlmModel();

    const result = await generateText({
      model,
      system:
        "You are a scheduled task executor. The following is a scheduled task you need to execute. Complete it and report the result.",
      prompt: payload.prompt,
      tools: {
        bash: this.createBashTool(),
        browser: createBrowserTool(this.browserState, this.env.MYBROWSER)
      },
      maxSteps: 10
    });

    // Inject result into chat history
    const userMsg = {
      id: crypto.randomUUID(),
      role: "user" as const,
      parts: [
        {
          type: "text" as const,
          text: `[Scheduled Task] ${payload.description}`
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

  onError(connection: Connection, error: unknown): void {
    console.error("ChatAgent error:", error);
    if (error instanceof Error) {
      Sentry.captureException(error);
    }
    super.onError(connection, error);
  }

  async onRequest(request: Request): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname.endsWith("/get-schedules")) {
      const schedules = this.getSchedules();
      return Response.json(schedules);
    }
    return super.onRequest(request);
  }

  async onConnect(
    connection: import("agents").Connection,
    ctx: import("agents").ConnectionContext
  ) {
    // Capture userId + sessionUuid from the initial WebSocket upgrade request
    const uid = await this.getUserId(ctx.request);
    await this.getSessionUuid(ctx.request);
    this.initBash(uid);
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
            const result = await Sentry.startSpan(
              { name: "bash.exec", op: "exec" },
              () => this.bash.exec(command)
            );
            try {
              await Sentry.startSpan({ name: "gitSync", op: "git" }, () =>
                syncDirtyGitMounts(this.mountableFs, command)
              );
            } catch (e) {
              result.stderr += `\ngit sync error: ${e instanceof Error ? e.message : e}`;
            }
            return {
              stdout: result.stdout,
              stderr: result.stderr,
              exitCode: result.exitCode
            };
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

  private createTools() {
    return {
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
          const s = await this.schedule(when, "executeScheduledTask" as any, {
            description,
            prompt
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
          const s = await this.schedule(cron, "executeScheduledTask" as any, {
            description,
            prompt
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
  }

  async onChatMessage() {
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

      // Fire-and-forget: sync chat history to /home/user/.chat/
      this.writeChatHistory().catch((e) =>
        console.error("writeChatHistory:", e)
      );

      const llmModel = await this.getLlmModel();
      const memoryBlock = await this.readMemoryBlock();

      const messages = await Sentry.startSpan(
        { name: "convertMessages", op: "serialize" },
        async () =>
          pruneMessages({
            messages: await convertToModelMessages(this.messages),
            toolCalls: "before-last-2-messages",
            reasoning: "before-last-message"
          })
      );

      const result = streamText({
        model: llmModel,
        system: buildSystemPrompt({
          sessionDir: this.sessionDir,
          memoryBlock
        }),
        messages,
        tools: this.createTools(),
        stopWhen: stepCountIs(10)
      });

      return result.toUIMessageStreamResponse();
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
