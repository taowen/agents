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
  stepCountIs,
  type ToolSet
} from "ai";
import type { Bash } from "just-bash";
import type { MountableFs } from "just-bash";

import { buildSystemPrompt } from "./system-prompt";

// Extracted modules
import { initBash, doFstabMount } from "vfs";
import type { FsBindings } from "vfs";
import { createSessionsCommand } from "./session-commands";
import { ensureMcpServers } from "./mcp-config";
import {
  getCachedLlmConfig,
  getLlmModel,
  type LlmConfigCache
} from "./llm-config";
import { handleFileRequest } from "./file-api";
import { writeChatHistory, readMemoryBlock } from "./chat-history";
import { createBashTool, createTools, createDeviceTools } from "./tools";

interface DeviceInfo {
  type: "device";
  deviceName: string;
  deviceId: string;
  connectedAt: number;
}

interface PendingTask {
  resolve: (value: { result: string; success: boolean }) => void;
  reject: (reason: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

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
  private userId: string | null = null;
  private sessionUuid: string | null = null;
  private cachedLlmConfig: LlmConfigCache = null;
  private mcpServersLoaded = false;
  private pendingTasks = new Map<string, PendingTask>();

  private isDeviceSession(): boolean {
    return !!this.sessionUuid?.startsWith("device-");
  }

  private getDeviceName(): string {
    return this.sessionUuid!.slice("device-".length);
  }

  /** Get all device WebSockets connected to this DO. */
  private getDeviceWebSockets(): WebSocket[] {
    return this.ctx.getWebSockets("device");
  }

  /** Get the first connected device WebSocket, or null. */
  private getDeviceWs(): WebSocket | null {
    const sockets = this.getDeviceWebSockets();
    return sockets.length > 0 ? sockets[0] : null;
  }

  // System prompt cache — computed once per session, invalidated on clear history
  private cachedSystemPrompt: string | null = null;
  private cachedDynamicContext: string | null = null;
  private cachedLlmModel: ReturnType<typeof getLlmModel> | null = null;
  private cachedTools: ToolSet | null = null;

  // Quota check cache — avoid per-message D1 queries
  private quotaCheckCache: { exceeded: boolean; checkedAt: number } | null =
    null;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    const parentAlarm = this.alarm.bind(this);
    (this as any).alarm = async () => {
      const deviceSockets = this.getDeviceWebSockets();
      for (const ws of deviceSockets) {
        try {
          ws.send(JSON.stringify({ type: "ping" }));
        } catch {
          // Dead socket — runtime will clean it up
        }
      }

      if (deviceSockets.length > 0) {
        await this.ctx.storage.setAlarm(Date.now() + 30_000);
      }
      await parentAlarm();
    };
  }

  private get fsBindings(): FsBindings {
    return {
      db: this.env.DB,
      r2: this.env.R2,
      googleClientId: this.env.GOOGLE_CLIENT_ID,
      googleClientSecret: this.env.GOOGLE_CLIENT_SECRET
    };
  }

  /**
   * Initialize bash + filesystem for the given userId.
   */
  private doInitBash(userId: string): void {
    if (this.bash && this.userId === userId) return;
    this.userId = userId;
    const { bash, mountableFs } = initBash({
      bindings: this.fsBindings,
      userId,
      customCommands: [
        createSessionsCommand(this.env.DB, userId, this.env.ChatAgent)
      ]
    });
    this.bash = bash;
    this.mountableFs = mountableFs;
    this.mounted = false;
    this.mountPromise = null;
  }

  /**
   * Ensure fstab-declared mounts are applied. Called once before first bash exec.
   */
  private async ensureMounted(): Promise<void> {
    if (this.mounted) return;
    if (!this.mountPromise) {
      this.mountPromise = doFstabMount(
        this.mountableFs,
        this.fsBindings,
        this.userId!
      ).then(
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
   * Extract userId from request header, with fallback to DO-local storage.
   */
  private async getUserId(request?: Request): Promise<string> {
    if (request) {
      const uid = request.headers.get("x-user-id");
      if (uid) {
        await this.ctx.storage.put("userId", uid);
        return uid;
      }
    }
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

  /** Read the user's timezone from DO storage, defaulting to UTC. */
  private async getTimezone(): Promise<string> {
    const stored = await this.ctx.storage.get<string>("timezone");
    return stored || "UTC";
  }

  /** Short session directory name derived from the DO ID. */
  private get sessionDir(): string {
    return this.ctx.id.toString().slice(0, 12);
  }

  /** Tag all Sentry events with session/user context for correlation. */
  private applySentryTags(): void {
    if (this.userId) {
      Sentry.setUser({ id: this.userId });
      Sentry.setTag("user_id", this.userId);
    }
    if (this.sessionUuid) {
      Sentry.setTag("session_uuid", this.sessionUuid);
    }
    Sentry.setTag("do_id", this.ctx.id.toString());
  }

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);

    // Device liveness check — used by tools and API to check if a device is online
    if (url.pathname.endsWith("/status")) {
      const deviceSockets = this.getDeviceWebSockets();
      return Response.json({ online: deviceSockets.length > 0 });
    }

    // Device WebSocket — custom protocol, bypasses agents framework
    if (url.pathname.endsWith("/device-connect")) {
      return this.handleDeviceConnect(request);
    }

    // Task dispatch endpoint — used by device_agent tool
    if (url.pathname.endsWith("/dispatch")) {
      return this.handleDispatch(request);
    }

    const response = await super.fetch(request);
    if (response.status >= 500) {
      const body = await response.clone().text();
      Sentry.captureMessage(
        `DO ${request.method} ${url.pathname} → ${response.status}: ${body.slice(0, 200)}`,
        "error"
      );
    }
    return response;
  }

  private async handleDeviceConnect(request: Request): Promise<Response> {
    const upgradeHeader = request.headers.get("Upgrade");
    if (!upgradeHeader || upgradeHeader.toLowerCase() !== "websocket") {
      return new Response("Expected WebSocket", { status: 426 });
    }

    // Persist userId from the auth middleware
    const uid = request.headers.get("x-user-id");
    if (uid) {
      await this.ctx.storage.put("userId", uid);
      this.userId = uid;
    }
    const sid = request.headers.get("x-session-id");
    if (sid) {
      await this.ctx.storage.put("sessionUuid", sid);
      this.sessionUuid = sid;
    }

    const pair = new WebSocketPair();
    // Tag as "device" so we can distinguish from web client connections
    this.ctx.acceptWebSocket(pair[1], ["device"]);

    // Start heartbeat alarm if not already running
    const alarm = await this.ctx.storage.getAlarm();
    if (!alarm) {
      await this.ctx.storage.setAlarm(Date.now() + 30_000);
    }

    return new Response(null, { status: 101, webSocket: pair[0] });
  }

  private async handleDispatch(request: Request): Promise<Response> {
    // Ensure userId is available
    const uid = request.headers.get("x-user-id");
    if (uid) {
      await this.ctx.storage.put("userId", uid);
      this.userId = uid;
    }

    const body = (await request.json()) as {
      task: string;
      timeoutMs?: number;
    };
    const { task, timeoutMs = 5 * 60 * 1000 } = body;

    const deviceWs = this.getDeviceWs();
    if (!deviceWs) {
      return Response.json({ error: "No device connected" }, { status: 404 });
    }

    const taskId = crypto.randomUUID();

    try {
      deviceWs.send(
        JSON.stringify({ type: "task", taskId, description: task })
      );
    } catch (e) {
      return Response.json(
        { error: "Failed to send to device" },
        { status: 502 }
      );
    }

    const resultPromise = new Promise<{ result: string; success: boolean }>(
      (resolve, reject) => {
        const timer = setTimeout(() => {
          this.pendingTasks.delete(taskId);
          reject(new Error("Task timed out"));
        }, timeoutMs);
        this.pendingTasks.set(taskId, { resolve, reject, timer });
      }
    );

    try {
      const { result, success } = await resultPromise;
      return Response.json({ taskId, result, success });
    } catch (e) {
      return Response.json(
        { error: e instanceof Error ? e.message : "Unknown error" },
        { status: 504 }
      );
    }
  }

  onError(connectionOrError: Connection | unknown, error?: unknown): void {
    this.applySentryTags();
    const err = error !== undefined ? error : connectionOrError;
    console.error("ChatAgent error:", err);
    Sentry.captureException(err);
    if (error !== undefined) {
      super.onError(connectionOrError as Connection, error);
    }
  }

  async onRequest(request: Request): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname.endsWith("/get-schedules")) {
      const schedules = this.getSchedules();
      return Response.json(schedules);
    }
    if (url.pathname.endsWith("/get-usage")) {
      // Diagnostic: total message count and sample metadata
      const totalRows = this.ctx.storage.sql
        .exec(`SELECT COUNT(*) as cnt FROM cf_ai_chat_agent_messages`)
        .toArray();
      const totalCount = (totalRows[0] as { cnt: number })?.cnt ?? 0;

      const withUsage = this.ctx.storage.sql
        .exec(
          `SELECT COUNT(*) as cnt FROM cf_ai_chat_agent_messages WHERE json_extract(message, '$.metadata.usage') IS NOT NULL`
        )
        .toArray();
      const usageCount = (withUsage[0] as { cnt: number })?.cnt ?? 0;

      // Sample one message to see its JSON structure
      const sample = this.ctx.storage.sql
        .exec(
          `SELECT json_extract(message, '$.metadata') as meta FROM cf_ai_chat_agent_messages LIMIT 1`
        )
        .toArray();
      const sampleMeta =
        sample.length > 0 ? (sample[0] as { meta: string })?.meta : "NO_ROWS";

      console.log(
        `[get-usage] total_msgs=${totalCount} with_usage=${usageCount} sample_meta=${sampleMeta}`
      );

      const since = url.searchParams.get("since");
      const baseQuery = `SELECT
          strftime('%Y-%m-%dT%H', created_at) as hour,
          COALESCE(json_extract(message, '$.metadata.apiKeyType'), 'unknown') as api_key_type,
          COUNT(*) as request_count,
          SUM(json_extract(message, '$.metadata.usage.inputTokens')) as input_tokens,
          SUM(json_extract(message, '$.metadata.usage.cacheReadTokens')) as cache_read_tokens,
          SUM(json_extract(message, '$.metadata.usage.outputTokens')) as output_tokens
        FROM cf_ai_chat_agent_messages
        WHERE json_extract(message, '$.metadata.usage') IS NOT NULL
        GROUP BY hour, api_key_type`;
      const rows = since
        ? this.ctx.storage.sql.exec(
            baseQuery + ` HAVING hour >= ? ORDER BY hour`,
            since
          )
        : this.ctx.storage.sql.exec(baseQuery + ` ORDER BY hour`);
      return Response.json(rows.toArray());
    }
    if (url.pathname.startsWith("/api/files")) {
      const uid = await this.getUserId(request);
      this.doInitBash(uid);
      await this.ensureMounted();
      this.applySentryTags();
      return handleFileRequest(request, this.mountableFs);
    }
    return super.onRequest(request);
  }

  async onConnect(
    connection: import("agents").Connection,
    ctx: import("agents").ConnectionContext
  ) {
    const uid = await this.getUserId(ctx.request);
    await this.getSessionUuid(ctx.request);
    this.doInitBash(uid);
    this.applySentryTags();
    const origin = new URL(ctx.request.url).origin;
    await this.ctx.storage.put("callbackOrigin", origin);

    // Send current messages to newly connected client.
    // This fixes a race where the client's SWR cache returns stale messages
    // (from before the latest user message was persisted), especially when
    // the user navigates away during streaming and then returns.
    connection.send(
      JSON.stringify({
        type: "cf_agent_chat_messages",
        messages: this.messages
      })
    );

    return super.onConnect(connection, ctx);
  }

  /**
   * Callback invoked by the DO alarm for scheduled/recurring tasks.
   */
  async executeScheduledTask(
    payload: { description: string; prompt: string; timezone?: string },
    schedule: Schedule<{
      description: string;
      prompt: string;
      timezone?: string;
    }>
  ) {
    return Sentry.startSpan(
      { name: "executeScheduledTask", op: "schedule" },
      async () => {
        try {
          if (!this.userId) {
            const uid = await this.getUserId();
            this.doInitBash(uid);
          }
          if (!this.sessionUuid) {
            await this.getSessionUuid();
          }
          this.applySentryTags();

          const { data: llmConfig, cache } = await getCachedLlmConfig(
            this.mountableFs,
            this.cachedLlmConfig
          );
          this.cachedLlmConfig = cache;
          const model = getLlmModel(this.env, llmConfig);

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
              bash: createBashTool(this.bash, () => this.ensureMounted())
            },
            stopWhen: stepCountIs(10)
          });

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
        } catch (e) {
          console.error("executeScheduledTask failed:", e);
          Sentry.captureException(e);
          const errorMsg = {
            id: crypto.randomUUID(),
            role: "assistant" as const,
            parts: [
              {
                type: "text" as const,
                text: `[Scheduled Task Failed] ${new Date().toISOString()} - ${payload.description}\nError: ${e instanceof Error ? e.message : String(e)}`
              }
            ]
          };
          try {
            await this.persistMessages([...this.messages, errorMsg]);
          } catch (persistErr) {
            console.error("Failed to persist error message:", persistErr);
          }
          throw e;
        }
      }
    );
  }

  /**
   * Pure relay for device sessions — no LLM, sends user text directly to the device WebSocket.
   */
  private async handleDeviceRelay(): Promise<Response> {
    const lastMsg = this.messages[this.messages.length - 1];
    const userText = lastMsg.parts
      .filter((p: any) => p.type === "text")
      .map((p: any) => (p as { text: string }).text)
      .join("\n");

    const deviceWs = this.getDeviceWs();
    if (!deviceWs) {
      return new Response("Device not connected", {
        headers: { "Content-Type": "text/plain" }
      });
    }

    const taskId = crypto.randomUUID();

    try {
      deviceWs.send(
        JSON.stringify({ type: "task", taskId, description: userText })
      );
    } catch (e) {
      return new Response("Failed to send to device", {
        headers: { "Content-Type": "text/plain" }
      });
    }

    const resultPromise = new Promise<{ result: string; success: boolean }>(
      (resolve, reject) => {
        const timer = setTimeout(
          () => {
            this.pendingTasks.delete(taskId);
            reject(new Error("Task timed out"));
          },
          5 * 60 * 1000
        );
        this.pendingTasks.set(taskId, { resolve, reject, timer });
      }
    );

    try {
      const { result, success } = await resultPromise;
      return new Response(result || (success ? "Done." : "Task failed."), {
        headers: { "Content-Type": "text/plain" }
      });
    } catch (e) {
      return new Response(e instanceof Error ? e.message : "Task failed", {
        headers: { "Content-Type": "text/plain" }
      });
    }
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
            this.doInitBash(uid);
          }
        );
      }
      if (!this.sessionUuid) {
        await this.getSessionUuid();
      }
      this.applySentryTags();

      // Device sessions are pure relays — no LLM, no bash, no tools
      if (this.isDeviceSession()) {
        return this.handleDeviceRelay();
      }

      // Ensure /etc and fstab mounts are ready before any filesystem access
      await this.ensureMounted();

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
      Sentry.startSpan({ name: "writeChatHistory", op: "db.query" }, () =>
        writeChatHistory(
          this.messages,
          this.env.DB,
          this.userId!,
          this.sessionUuid,
          this.sessionDir
        )
      ).catch((e) => {
        console.error("writeChatHistory:", e);
        Sentry.captureException(e);
      });

      // Auto-connect MCP servers from /etc/mcp-servers.json
      if (!this.mcpServersLoaded) {
        this.mcpServersLoaded = true;
        try {
          const callbackHost =
            (await this.ctx.storage.get<string>("callbackOrigin")) ?? "";
          const callbackPath = `mcp-callback/${this.ctx.id.toString()}`;
          await Sentry.startSpan({ name: "ensureMcpServers", op: "mcp" }, () =>
            ensureMcpServers(
              this.mountableFs,
              () => this.getMcpServers(),
              (name, url, opts) => this.addMcpServer(name, url, opts),
              callbackHost,
              callbackPath
            )
          );
        } catch (e) {
          console.error("ensureMcpServers failed:", e);
          Sentry.captureException(e);
        }
      }

      // Determine if this session uses the builtin key (no /etc/llm.json)
      const { data: llmConfigForKeyCheck, cache: keyCheckCache } =
        await getCachedLlmConfig(this.mountableFs, this.cachedLlmConfig);
      this.cachedLlmConfig = keyCheckCache;
      const isBuiltinKey = llmConfigForKeyCheck === null;

      // Quota gate: if builtin key and user is over quota, reject
      if (isBuiltinKey) {
        const now = Date.now();
        const QUOTA_CACHE_TTL = 30_000; // 30 seconds
        if (
          !this.quotaCheckCache ||
          now - this.quotaCheckCache.checkedAt > QUOTA_CACHE_TTL
        ) {
          const row = await this.env.DB.prepare(
            `SELECT builtin_quota_exceeded_at FROM users WHERE id = ?`
          )
            .bind(this.userId!)
            .first<{ builtin_quota_exceeded_at: string | null }>();
          this.quotaCheckCache = {
            exceeded: !!row?.builtin_quota_exceeded_at,
            checkedAt: now
          };
        }
        if (this.quotaCheckCache.exceeded) {
          throw new Error(
            "You have exceeded the builtin API key usage quota. " +
              "Please configure your own API key in Settings to continue using the service."
          );
        }
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
          Sentry.captureException(e);
        }

        const { data: llmConfig, cache: newLlmCache } =
          await getCachedLlmConfig(this.mountableFs, this.cachedLlmConfig);
        this.cachedLlmConfig = newLlmCache;

        const [llmModel, memoryBlock] = await Promise.all([
          Sentry.startSpan({ name: "getLlmModel", op: "config" }, () =>
            getLlmModel(this.env, llmConfig)
          ),
          Sentry.startSpan({ name: "readMemoryBlock", op: "db.query" }, () =>
            readMemoryBlock(this.env.DB, this.userId!)
          )
        ]);

        this.cachedLlmModel = llmModel;
        this.cachedSystemPrompt = buildSystemPrompt();

        // Build dynamic context
        const dynamicParts = [
          `Chat history directory: /home/user/.chat/${this.sessionDir}/`,
          memoryBlock
        ];
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
          Sentry.captureException(e);
        }
        this.cachedTools = {
          ...createTools({
            bashTool: createBashTool(this.bash, () => this.ensureMounted()),
            schedule: (when, method, payload) =>
              this.schedule(when, method, payload),
            getSchedules: () => this.getSchedules(),
            cancelSchedule: (id) => this.cancelSchedule(id),
            getTimezone: () => this.getTimezone()
          }),
          ...createDeviceTools(this.env, this.userId!),
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

      const apiKeyType = isBuiltinKey ? "builtin" : "custom";
      const sessionId = this.sessionUuid;
      const userId = this.userId!;
      const env = this.env;
      const doStorage = this.ctx.storage;

      const result = streamText({
        model: this.cachedLlmModel!,
        system: this.cachedSystemPrompt!,
        messages,
        tools: this.cachedTools!,
        stopWhen: stepCountIs(10),
        onFinish: async () => {
          // Proactive D1 write: aggregate current hour's usage from DO SQLite
          try {
            const hour = new Date().toISOString().slice(0, 13);
            const hourRows = doStorage.sql
              .exec(
                `SELECT COALESCE(json_extract(message, '$.metadata.apiKeyType'), 'unknown') as api_key_type,
                 COUNT(*) as request_count,
                 SUM(json_extract(message, '$.metadata.usage.inputTokens')) as input_tokens,
                 SUM(json_extract(message, '$.metadata.usage.cacheReadTokens')) as cache_read_tokens,
                 SUM(json_extract(message, '$.metadata.usage.outputTokens')) as output_tokens
                 FROM cf_ai_chat_agent_messages
                 WHERE json_extract(message, '$.metadata.usage') IS NOT NULL
                   AND strftime('%Y-%m-%dT%H', created_at) = ?
                 GROUP BY api_key_type`,
                hour
              )
              .toArray() as {
              api_key_type: string;
              request_count: number;
              input_tokens: number;
              cache_read_tokens: number;
              output_tokens: number;
            }[];
            for (const row of hourRows) {
              env.DB.prepare(
                `INSERT OR REPLACE INTO usage_archive (user_id, session_id, hour, api_key_type, request_count, input_tokens, cache_read_tokens, cache_write_tokens, output_tokens)
                 VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)`
              )
                .bind(
                  userId,
                  sessionId,
                  hour,
                  row.api_key_type,
                  row.request_count,
                  row.input_tokens || 0,
                  row.cache_read_tokens || 0,
                  row.output_tokens || 0
                )
                .run()
                .catch((e: unknown) =>
                  console.error("usage_archive D1 write failed:", e)
                );
            }
          } catch (e) {
            console.error("onFinish usage_archive write failed:", e);
          }
        }
      });

      return result.toUIMessageStreamResponse({
        messageMetadata: ({ part }) => {
          if (part.type === "finish") {
            return {
              usage: {
                inputTokens: part.totalUsage.inputTokens,
                outputTokens: part.totalUsage.outputTokens,
                cacheReadTokens:
                  part.totalUsage.inputTokenDetails?.cacheReadTokens
              },
              apiKeyType
            };
          }
          return undefined;
        }
      });
    });
  }

  // ---- Device WebSocket handling (Hibernatable API) ----

  async webSocketMessage(ws: WebSocket, message: string | ArrayBuffer) {
    const tags = this.ctx.getTags(ws);
    if (tags.includes("device")) {
      return this.handleDeviceWsMessage(ws, message);
    }
    // Agents framework handles web client messages
    return super.webSocketMessage(ws, message);
  }

  async webSocketClose(
    ws: WebSocket,
    code: number,
    reason: string,
    wasClean: boolean
  ) {
    const tags = this.ctx.getTags(ws);
    if (tags.includes("device")) {
      // Reject all pending tasks
      for (const [taskId, pending] of this.pendingTasks) {
        clearTimeout(pending.timer);
        this.pendingTasks.delete(taskId);
        pending.reject(new Error("Device disconnected"));
      }

      return;
    }
    return super.webSocketClose(ws, code, reason, wasClean);
  }

  async webSocketError(ws: WebSocket, error: unknown) {
    const tags = this.ctx.getTags(ws);
    if (tags.includes("device")) {
      // webSocketClose will fire after this — cleanup happens there
      return;
    }
    return super.webSocketError(ws, error);
  }

  private async handleDeviceWsMessage(
    ws: WebSocket,
    message: string | ArrayBuffer
  ): Promise<void> {
    if (typeof message !== "string") return;

    try {
      const data = JSON.parse(message);

      if (data.type === "ready") {
        const info: DeviceInfo = {
          type: "device",
          deviceName: data.deviceName || "Unknown Device",
          deviceId: data.deviceId || "unknown",
          connectedAt: Date.now()
        };
        ws.serializeAttachment(info);

        // Mark device online in D1
        try {
          const userId = await this.getUserId();
          const sessionId = await this.getSessionUuid();
          if (userId && sessionId) {
            // Ensure session row exists
            await this.env.DB.prepare(
              `INSERT INTO sessions (id, user_id, title) VALUES (?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET updated_at = datetime('now')`
            )
              .bind(sessionId, userId, `Device: ${info.deviceName}`)
              .run();
          }
        } catch (e) {
          console.error("Failed to update device_online on ready:", e);
        }
        return;
      }

      if (data.type === "llm_request") {
        await this.handleLlmRequest(ws, data);
        return;
      }

      if (data.type === "result") {
        const pending = this.pendingTasks.get(data.taskId);
        if (pending) {
          clearTimeout(pending.timer);
          this.pendingTasks.delete(data.taskId);
          pending.resolve({
            result: data.result || "",
            success: data.success !== false
          });
        }
        return;
      }

      // pong — no action needed
    } catch {
      // ignore malformed messages
    }
  }

  /**
   * Handle an LLM request from the device: use ChatAgent's own config,
   * call upstream, persist messages, respond.
   */
  private async handleLlmRequest(
    ws: WebSocket,
    data: { requestId: string; body: Record<string, unknown> }
  ): Promise<void> {
    const { requestId, body } = data;

    try {
      // Ensure userId and bash are initialized
      if (!this.userId) {
        const uid = await this.getUserId();
        this.doInitBash(uid);
      }
      await this.ensureMounted();

      // Use ChatAgent's own LLM config (with caching)
      const { data: llmConfig, cache } = await getCachedLlmConfig(
        this.mountableFs,
        this.cachedLlmConfig
      );
      this.cachedLlmConfig = cache;

      let upstreamBaseURL = this.env.BUILTIN_LLM_BASE_URL;
      let upstreamApiKey = this.env.BUILTIN_LLM_API_KEY;
      let upstreamModel = this.env.BUILTIN_LLM_MODEL;
      let apiKeyType = "builtin";

      if (llmConfig) {
        upstreamBaseURL = llmConfig.base_url;
        upstreamApiKey = llmConfig.api_key;
        upstreamModel = llmConfig.model || upstreamModel;
        apiKeyType = "custom";
      }

      // Call upstream LLM
      const upstreamRes = await fetch(`${upstreamBaseURL}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${upstreamApiKey}`
        },
        body: JSON.stringify({
          ...body,
          model: (body.model as string) || upstreamModel
        })
      });
      const responseBody = (await upstreamRes.json()) as Record<
        string,
        unknown
      >;

      // NOTE: We do NOT persist messages here. When the web client sends a
      // message to a device session, the AIChatAgent framework already persists
      // both the user message (before onChatMessage) and the assistant message
      // (after handleDeviceRelay returns its Response). Persisting here too
      // would cause duplicate messages.

      // Write usage to usage_archive for quota tracking
      const usage = (responseBody as { usage?: Record<string, number> }).usage;
      if (usage) {
        const proxyHour = new Date().toISOString().slice(0, 13);
        this.env.DB.prepare(
          `INSERT INTO usage_archive (user_id, session_id, hour, api_key_type, request_count, input_tokens, cache_read_tokens, cache_write_tokens, output_tokens)
           VALUES (?, ?, ?, ?, 1, ?, ?, 0, ?)
           ON CONFLICT(user_id, session_id, hour, api_key_type) DO UPDATE SET
             request_count = request_count + 1,
             input_tokens = input_tokens + excluded.input_tokens,
             cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens,
             output_tokens = output_tokens + excluded.output_tokens`
        )
          .bind(
            this.userId!,
            this.sessionUuid,
            proxyHour,
            apiKeyType,
            usage.prompt_tokens || 0,
            (usage as any).prompt_tokens_details?.cached_tokens || 0,
            usage.completion_tokens || 0
          )
          .run()
          .catch((e: unknown) =>
            console.error("LLM request usage_archive write failed:", e)
          );
      }

      // Send LLM response back to device
      ws.send(
        JSON.stringify({
          type: "llm_response",
          requestId,
          body: JSON.stringify(responseBody)
        })
      );
    } catch (e) {
      console.error("ChatAgent: LLM request failed:", e);
      ws.send(
        JSON.stringify({
          type: "llm_response",
          requestId,
          body: JSON.stringify({
            error: {
              message:
                e instanceof Error ? e.message : "LLM request failed on server"
            }
          })
        })
      );
    }
  }
}

export const ChatAgent = instrumentDurableObjectWithSentry(
  (env: Env) => ({
    dsn: env.SENTRY_DSN,
    tracesSampleRate: 1.0
  }),
  ChatAgentBase
);
