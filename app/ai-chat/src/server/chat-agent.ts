import { AIChatAgent } from "@cloudflare/ai-chat";
import type { Connection } from "agents";
import type { Schedule } from "agents";
import * as Sentry from "@sentry/cloudflare";
import { instrumentDurableObjectWithSentry } from "@sentry/cloudflare";
import {
  streamText,
  convertToModelMessages,
  pruneMessages,
  stepCountIs,
  type ToolSet,
  type UIMessage
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
import {
  createBashTool,
  createTools,
  createDeviceExecTool,
  createDeviceTools
} from "./tools";
import { DeviceHub, isDeviceSession } from "./device-hub";
import {
  queryUsageData,
  logUsageDiagnostics,
  checkQuota,
  archiveSessionUsage,
  type QuotaCache
} from "./usage-tracker";
import { runScheduledTask } from "./scheduled-tasks";

interface DeviceTool {
  function?: { name?: string; description?: string };
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
  private deviceHub: DeviceHub;

  // System prompt cache — computed once per session, invalidated on clear history
  private cachedSystemPrompt: string | null = null;
  private cachedDynamicContext: string | null = null;
  private cachedLlmModel: ReturnType<typeof getLlmModel> | null = null;
  private cachedTools: ToolSet | null = null;

  // Quota check cache — avoid per-message D1 queries
  private quotaCheckCache: QuotaCache | null = null;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.deviceHub = new DeviceHub(ctx);
    const parentAlarm = this.alarm.bind(this);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- framework workaround: alarm() not exposed on AIChatAgent type
    (this as any).alarm = async () => {
      await this.deviceHub.sendHeartbeats();
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
      const deviceSockets = this.deviceHub.getWebSockets();
      return Response.json({ online: deviceSockets.length > 0 });
    }

    // Device WebSocket — custom protocol, bypasses agents framework
    if (url.pathname.endsWith("/device-connect")) {
      return this.deviceHub.handleConnect(request);
    }

    // Dispatch a task to this device session (called by send_to_device tool)
    if (url.pathname.endsWith("/dispatch-task") && request.method === "POST") {
      const { text } = await request.json<{ text: string }>();
      const result = await this.handleDeviceInitiatedTask(text);
      return Response.json({ result });
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
      logUsageDiagnostics(this.ctx.storage.sql);
      const since = url.searchParams.get("since");
      return Response.json(queryUsageData(this.ctx.storage.sql, since));
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
    _schedule: Schedule<{
      description: string;
      prompt: string;
      timezone?: string;
    }>
  ) {
    return Sentry.startSpan(
      { name: "executeScheduledTask", op: "schedule" },
      async () => {
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
        const timezone = await this.getTimezone();

        await runScheduledTask({
          messages: this.messages,
          persistMessages: (msgs) => this.persistMessages(msgs),
          bash: this.bash,
          ensureMounted: () => this.ensureMounted(),
          model,
          timezone,
          payload
        });
      }
    );
  }

  /**
   * Resolve LLM config and enforce quota for builtin key users.
   * Shared by onChatMessage and handleDeviceChatMessage.
   */
  private async resolveQuotaAndModel(): Promise<{
    llmModel: ReturnType<typeof getLlmModel>;
    apiKeyType: "builtin" | "custom";
    isBuiltinKey: boolean;
  }> {
    const { data: llmConfig, cache } = await getCachedLlmConfig(
      this.mountableFs,
      this.cachedLlmConfig
    );
    this.cachedLlmConfig = cache;
    const isBuiltinKey = llmConfig === null;

    if (isBuiltinKey) {
      this.quotaCheckCache = await checkQuota(
        this.env.DB,
        this.userId!,
        this.quotaCheckCache
      );
      if (this.quotaCheckCache.exceeded) {
        throw new Error(
          "You have exceeded the builtin API key usage quota. " +
            "Please configure your own API key in Settings to continue using the service."
        );
      }
    }

    const llmModel = getLlmModel(this.env, llmConfig);
    return {
      llmModel,
      apiKeyType: isBuiltinKey ? "builtin" : "custom",
      isBuiltinKey
    };
  }

  /**
   * Create the onFinish callback for archiving session usage.
   */
  private createOnFinish(): () => Promise<void> {
    const { DB } = this.env;
    const sql = this.ctx.storage.sql;
    const userId = this.userId!;
    const sessionId = this.sessionUuid;
    return async () => {
      try {
        await archiveSessionUsage(DB, sql, userId, sessionId);
      } catch (e) {
        console.error("onFinish usage_archive write failed:", e);
      }
    };
  }

  /**
   * Wrap a streamText result into a UI message stream response with usage metadata.
   */
  private toStreamResponse(
    result: ReturnType<typeof streamText>,
    apiKeyType: string
  ): Response {
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

      // Device sessions use streamText with device-reported prompt + execute_js tool
      if (isDeviceSession(this.sessionUuid)) {
        return this.handleDeviceChatMessage();
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

      // Quota gate + LLM config resolution
      const { apiKeyType, isBuiltinKey } = await this.resolveQuotaAndModel();

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
              this.schedule(when, method as keyof typeof this, payload),
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

      const result = streamText({
        model: this.cachedLlmModel!,
        system: this.cachedSystemPrompt!,
        messages,
        tools: this.cachedTools!,
        stopWhen: stepCountIs(10),
        onFinish: this.createOnFinish()
      });

      return this.toStreamResponse(result, apiKeyType);
    });
  }

  // ---- Device session streamText handler ----

  private async handleDeviceChatMessage(): Promise<Response> {
    // userId and bash are guaranteed by onChatMessage caller
    await this.ensureMounted();

    const { llmModel, apiKeyType } = await this.resolveQuotaAndModel();

    // Read device-reported system prompt and tool description from storage
    const deviceSystemPrompt =
      (await this.ctx.storage.get<string>("deviceSystemPrompt")) ||
      "You are a mobile automation assistant.";
    const deviceTools = await this.ctx.storage.get<DeviceTool[]>("deviceTools");

    // Build tool description from device-reported tools (for execute_js description)
    let toolDesc: string | undefined;
    if (deviceTools && deviceTools.length > 0) {
      const fn = deviceTools[0]?.function;
      if (fn?.description) {
        toolDesc = fn.description;
      }
    }

    const tools = createDeviceExecTool(this.deviceHub, toolDesc);

    const messages = await pruneMessages({
      messages: await convertToModelMessages(this.messages),
      toolCalls: "before-last-2-messages",
      reasoning: "before-last-message"
    });

    const result = streamText({
      model: llmModel,
      system: deviceSystemPrompt,
      messages,
      tools,
      stopWhen: stepCountIs(30),
      onFinish: this.createOnFinish()
    });

    return this.toStreamResponse(result, apiKeyType);
  }

  // ---- Device-initiated task (from device WebSocket) ----

  private async handleDeviceInitiatedTask(text: string): Promise<string> {
    return Sentry.startSpan(
      { name: "handleDeviceInitiatedTask", op: "device_task" },
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

          const userMsg: UIMessage = {
            id: crypto.randomUUID(),
            role: "user",
            parts: [{ type: "text", text }]
          };

          // Reuse the same path as web-initiated messages:
          // saveMessages → onChatMessage → handleDeviceChatMessage → streamText → _reply
          await this.saveMessages([...this.messages, userMsg]);

          // Extract final text from the last assistant message for task_done
          const lastMsg = this.messages[this.messages.length - 1];
          let resultText = "";
          if (lastMsg?.role === "assistant") {
            for (const part of lastMsg.parts) {
              if (part.type === "text") resultText += part.text;
            }
          }
          this.deviceHub.sendTaskDone(resultText || "done");
          return resultText || "done";
        } catch (e) {
          console.error("handleDeviceInitiatedTask failed:", e);
          Sentry.captureException(e);
          const errMsg = e instanceof Error ? e.message : String(e);
          this.deviceHub.sendTaskDone("Error: " + errMsg);
          return "Error: " + errMsg;
        }
      }
    );
  }

  // ---- Device WebSocket handling (Hibernatable API) ----

  async webSocketMessage(ws: WebSocket, message: string | ArrayBuffer) {
    const tags = this.ctx.getTags(ws);
    if (tags.includes("device")) {
      return this.deviceHub.handleMessage(ws, message, {
        getUserId: () => this.getUserId(),
        getSessionUuid: () => this.getSessionUuid(),
        onUserTask: (text) => {
          this.handleDeviceInitiatedTask(text);
        },
        env: this.env
      });
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
      this.deviceHub.handleClose();
      return;
    }
    return super.webSocketClose(ws, code, reason, wasClean);
  }

  async webSocketError(ws: WebSocket, error: unknown) {
    const tags = this.ctx.getTags(ws);
    if (tags.includes("device")) {
      this.deviceHub.handleError();
      return;
    }
    return super.webSocketError(ws, error);
  }
}

export const ChatAgent = instrumentDurableObjectWithSentry(
  (env: Env) => ({
    dsn: env.SENTRY_DSN,
    tracesSampleRate: 1.0
  }),
  ChatAgentBase
);
