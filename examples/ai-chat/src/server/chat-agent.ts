import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { AIChatAgent } from "@cloudflare/ai-chat";
import type { Connection } from "agents";
import * as Sentry from "@sentry/cloudflare";
import { instrumentDurableObjectWithSentry } from "@sentry/cloudflare";
import {
  streamText,
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
  syncDirtyGitMounts
} from "vfs";
import { Bash, InMemoryFs, MountableFs } from "just-bash";
import { D1FsAdapter } from "./d1-fs-adapter";
import { getSettings, type UserSettings } from "./db";
import {
  createBrowserState,
  createBrowserTool,
  type BrowserState
} from "./browser-tool";
import { createD1MountCommands } from "./mount-commands";

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

    this.bash = new Bash({
      fs,
      customCommands: createD1MountCommands(fs),
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

  onError(connection: Connection, error: unknown): void {
    console.error("ChatAgent error:", error);
    if (error instanceof Error) {
      Sentry.captureException(error);
    }
    super.onError(connection, error);
  }

  async onRequest(request: Request): Promise<Response> {
    return super.onRequest(request);
  }

  async onConnect(
    connection: import("agents").Connection,
    ctx: import("agents").ConnectionContext
  ) {
    // Capture userId from the initial WebSocket upgrade request
    const uid = await this.getUserId(ctx.request);
    this.initBash(uid);
    return super.onConnect(connection, ctx);
  }

  async onChatMessage() {
    return Sentry.startSpan({ name: "onChatMessage", op: "chat" }, async () => {
      // Ensure we have userId (may need recovery after hibernation)
      if (!this.userId) {
        await Sentry.startSpan(
          { name: "getUserId", op: "db.query" },
          async () => {
            const uid = await this.getUserId();
            this.initBash(uid);
          }
        );
      }

      // Read LLM settings from D1
      let apiKey = this.env.GOOGLE_AI_API_KEY;
      let provider: string = "google";
      let baseURL = "https://generativelanguage.googleapis.com/v1beta";
      let model = "gemini-2.0-flash";

      if (this.userId) {
        const settings = await this.getCachedSettings();
        if (settings?.llm_api_key) apiKey = settings.llm_api_key;
        if (settings?.llm_provider) provider = settings.llm_provider;
        if (settings?.llm_base_url) baseURL = settings.llm_base_url;
        if (settings?.llm_model) model = settings.llm_model;
      }

      const llmModel =
        provider === "openai-compatible"
          ? createOpenAICompatible({ name: "llm", baseURL, apiKey })(model)
          : createGoogleGenerativeAI({ baseURL, apiKey })(model);

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
        system:
          "You are a helpful assistant with a sandboxed virtual bash environment (not a real Linux shell). " +
          "Available commands: ls, cat, grep, awk, sed, find, echo, mkdir, cp, mv, rm, sort, uniq, wc, head, tail, " +
          "curl, diff, jq, base64, tree, du, df, stat, file, tr, cut, paste, date, uname, id, uptime, hostname, whoami, " +
          "mount (no args shows mounts), and more. Use `help` to list all commands. " +
          "NOT available: git, apt, npm, pip, python, node, tar, gzip, ssh, wget, docker, sudo, " +
          "and any package managers or compilers. " +
          "There are no /proc, /sys, or /dev filesystems. " +
          "Use curl to fetch content from URLs. " +
          "Use `mount` (no args) to see current mounts, `df` to see filesystem info. " +
          "Files in /home/user and /etc persist across sessions (stored in durable storage). " +
          "Files outside these directories only persist within the current session. " +
          "/etc/fstab controls what gets mounted on startup. " +
          "To add a persistent git mount, append to /etc/fstab: " +
          'echo "https://github.com/user/repo  /mnt/repo  git  ref=main,depth=1  0  0" >> /etc/fstab ' +
          "(it will be mounted on the next session). " +
          "You can also mount git repos dynamically for the current session: " +
          "mkdir -p /mnt/<repo-name> && mount -t git <url> /mnt/<repo-name>. " +
          "IMPORTANT: Always mount under /mnt/<name>, never directly to /mnt itself. " +
          "Do NOT mount inside /home/user as it would conflict with persistent storage. " +
          "Options via -o: ref (branch/tag, default main), depth (clone depth, default 1), username, password. " +
          "For private repos: mount -t git -o username=user,password=token <url> /mnt/<repo-name>. " +
          "If a GitHub account is connected (via Settings), " +
          "private GitHub repos are automatically authenticated when mounting. " +
          "Git repos mounted via mount -t git are read-write. " +
          "Any file changes are automatically committed and pushed after each command. " +
          "If a GitHub account is connected, authentication is automatic. " +
          "Unmount with: umount /mnt/<repo-name>. " +
          "You also have a browser tool for browsing real web pages. " +
          "Use the browser tool when you need to interact with SPAs, JavaScript-rendered content, or pages that curl can't handle well. " +
          "The browser tool supports actions: goto (navigate to URL), click (click an element by CSS selector), " +
          "type (type text into an input by CSS selector), screenshot (capture current page), " +
          "scroll (scroll up or down), extract (extract text from page or specific element), " +
          "set_cookies (inject cookies for authentication - user provides cookie JSON), close (close browser). " +
          "Each browser action returns a screenshot so you can see the page. " +
          "For sites requiring login: the user can export cookies from their own browser and provide them. " +
          "Use set_cookies with the cookies JSON, then goto the target URL to access as the authenticated user. " +
          "Prefer curl for simple requests; use the browser for complex web pages that need JavaScript rendering.",
        messages,
        tools: {
          bash: tool({
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
                  await Sentry.startSpan(
                    { name: "ensureMounted", op: "mount" },
                    () => this.ensureMounted()
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
          }),
          browser: createBrowserTool(this.browserState, this.env.MYBROWSER)
        },
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
