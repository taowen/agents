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

import { buildSystemPrompt } from "./system-prompt";
import {
  DebugRingBuffer,
  buildRequestSnapshot,
  instrumentLlmCall,
  getSentryTraceContext
} from "./llm-debug-buffer";

// Extracted modules
import { getCachedLlmConfig, getLlmModel } from "./llm-config";
import { handleFileRequest } from "./file-api";
import { writeChatHistory, readMemoryBlock } from "./chat-history";
import {
  createBashTool,
  createTools,
  createDeviceExecTool,
  createDeviceTools,
  createSearchTool
} from "./tools";
import { DeviceHub, isDeviceSession } from "./device-hub";
import { ensureMcpServers } from "./mcp-config";
import { queryUsageData, logUsageDiagnostics } from "./usage-tracker";
import { SessionContext } from "./session-context";

interface DeviceTool {
  function?: { name?: string; description?: string };
}

/**
 * AI Chat Agent with sandboxed bash tool via just-bash.
 * Filesystem is backed by D1, scoped to the authenticated user.
 */
class ChatAgentBase extends AIChatAgent {
  maxPersistedMessages = 200;

  private session: SessionContext;
  private deviceHub: DeviceHub;
  private debugBuffer = new DebugRingBuffer(20, this.ctx.storage.sql);

  // Deferred promise for internal tasks (device-initiated and scheduled): resolved by onFinish
  private internalTaskDeferred: {
    resolve: (text: string) => void;
  } | null = null;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.session = new SessionContext(ctx, env);
    this.deviceHub = new DeviceHub(ctx, this.debugBuffer);
    const parentAlarm = this.alarm.bind(this);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- framework workaround: alarm() not exposed on AIChatAgent type
    (this as any).alarm = async () => {
      await this.deviceHub.sendHeartbeats();
      await parentAlarm();
    };
  }

  /** Tag all Sentry events with session/user context for correlation. */
  private applySentryTags(): void {
    if (this.session.userId) {
      Sentry.setUser({ id: this.session.userId });
      Sentry.setTag("user_id", this.session.userId);
    }
    if (this.session.sessionUuid) {
      Sentry.setTag("session_uuid", this.session.sessionUuid);
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
      const startTime = Date.now();
      const { text } = await request.json<{ text: string }>();
      const { traceId, spanId } = getSentryTraceContext();
      try {
        const result = await this.handleDeviceInitiatedTask(text);
        this.debugBuffer.push({
          type: "do_call",
          timestamp: new Date().toISOString(),
          traceId,
          spanId,
          direction: "inbound",
          endpoint: "/dispatch-task",
          request: { text },
          response: { result },
          durationMs: Date.now() - startTime
        });
        return Response.json({ result });
      } catch (e) {
        this.debugBuffer.push({
          type: "do_call",
          timestamp: new Date().toISOString(),
          traceId,
          spanId,
          direction: "inbound",
          endpoint: "/dispatch-task",
          request: { text },
          response: null,
          durationMs: Date.now() - startTime,
          error: String(e)
        });
        throw e;
      }
    }

    // Collect debug context for bug reports (called by API worker)
    if (url.pathname.endsWith("/collect-debug-context")) {
      return Response.json({
        debugEntries: this.debugBuffer.getAll(),
        messages: this.messages,
        bufferSize: this.debugBuffer.size,
        messageCount: this.messages.length,
        doId: this.ctx.id.toString()
      });
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
    if (
      url.pathname.endsWith("/cancel-all-schedules") &&
      request.method === "POST"
    ) {
      const schedules = this.getSchedules();
      for (const s of schedules) {
        await this.cancelSchedule(s.id);
      }
      return Response.json({ cancelled: schedules.length });
    }
    if (url.pathname.endsWith("/destroy") && request.method === "POST") {
      const schedules = this.getSchedules();
      for (const s of schedules) {
        await this.cancelSchedule(s.id);
      }
      await this.ctx.storage.deleteAll();
      return Response.json({ ok: true });
    }
    if (url.pathname.startsWith("/api/files")) {
      const uid = await this.session.getUserId(request);
      this.session.doInitBash(uid);
      await this.session.ensureMounted();
      this.applySentryTags();
      return handleFileRequest(request, this.session.mountableFs);
    }
    return super.onRequest(request);
  }

  async onConnect(
    connection: import("agents").Connection,
    ctx: import("agents").ConnectionContext
  ) {
    const uid = await this.session.getUserId(ctx.request);
    await this.session.getSessionUuid(ctx.request);
    this.session.doInitBash(uid);
    this.applySentryTags();
    const origin = new URL(ctx.request.url).origin;
    await this.ctx.storage.put("callbackOrigin", origin);

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
        await this.session.ensureReady();
        this.applySentryTags();

        this.debugBuffer.push({
          type: "schedule",
          timestamp: new Date().toISOString(),
          action: "execute",
          taskId: schedule.id ?? "",
          description: payload.description
        });

        // Check if session still exists in D1 — it may have been deleted by the user
        if (this.session.userId && this.session.sessionUuid) {
          const row = await this.env.DB.prepare(
            "SELECT 1 FROM sessions WHERE id = ? AND user_id = ?"
          )
            .bind(this.session.sessionUuid, this.session.userId)
            .first();
          if (!row) {
            // Session was deleted — cancel all schedules and destroy this DO
            console.log(
              `Session ${this.session.sessionUuid} deleted, cleaning up DO`
            );
            this.debugBuffer.push({
              type: "schedule",
              timestamp: new Date().toISOString(),
              action: "session_deleted",
              taskId: schedule.id ?? "",
              description: payload.description
            });
            const schedules = this.getSchedules();
            for (const s of schedules) {
              await this.cancelSchedule(s.id);
            }
            this.ctx.storage.deleteAll();
            return;
          }
        }

        const now = new Date();
        const tz = payload.timezone || (await this.session.getTimezone());
        const userMsg: UIMessage = {
          id: crypto.randomUUID(),
          role: "user",
          parts: [
            {
              type: "text",
              text: `[Scheduled Task] ${now.toISOString()} (${tz}) - ${payload.description}\n\n${payload.prompt}`
            }
          ]
        };

        const resultPromise = new Promise<string>((resolve) => {
          this.internalTaskDeferred = { resolve };
        });

        try {
          await this.saveMessages([...this.messages, userMsg]);
          await resultPromise;
        } catch (e) {
          // Clean up deferred on error
          this.internalTaskDeferred = null;
          console.error("executeScheduledTask failed:", e);
          Sentry.captureException(e);

          const errorMsg: UIMessage = {
            id: crypto.randomUUID(),
            role: "assistant",
            parts: [
              {
                type: "text",
                text: `[Scheduled Task Failed] ${new Date().toISOString()} - ${payload.description}\nError: ${e instanceof Error ? e.message : String(e)}`
              }
            ]
          };
          try {
            await this.persistMessages([...this.messages, errorMsg]);
          } catch (persistErr) {
            console.error("Failed to persist error message:", persistErr);
          }
        }
      }
    );
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
    options?: { abortSignal?: AbortSignal; body?: Record<string, unknown> }
  ) {
    return Sentry.startSpan({ name: "onChatMessage", op: "chat" }, async () => {
      // Ensure we have userId + sessionUuid (may need recovery after hibernation)
      if (!this.session.userId) {
        await Sentry.startSpan({ name: "ensureReady", op: "db.query" }, () =>
          this.session.ensureReady()
        );
      } else if (!this.session.sessionUuid) {
        await this.session.getSessionUuid();
      }
      this.applySentryTags();

      // Device sessions use streamText with device-reported prompt + execute_js tool
      if (isDeviceSession(this.session.sessionUuid)) {
        return this.handleDeviceChatMessage(options?.abortSignal);
      }

      // Ensure /etc and fstab mounts are ready before any filesystem access
      await this.session.ensureMounted();

      // Persist client-reported timezone for scheduled tasks and hibernation recovery
      const clientTz = options?.body?.timezone;
      if (typeof clientTz === "string" && clientTz) {
        await this.ctx.storage.put("timezone", clientTz);
      }
      const timezone =
        (typeof clientTz === "string" && clientTz) ||
        (await this.session.getTimezone());

      // Configure bash TZ env var and /etc/timezone so commands discover time natively
      this.session.bash.setEnv("TZ", timezone);
      await this.session.mountableFs.writeFile(
        "/etc/timezone",
        timezone + "\n"
      );

      // Fire-and-forget: sync chat history to /home/user/.chat/
      Sentry.startSpan({ name: "writeChatHistory", op: "db.query" }, () =>
        writeChatHistory(
          this.messages,
          this.env.DB,
          this.session.userId!,
          this.session.sessionUuid,
          this.session.sessionDir
        )
      ).catch((e) => {
        console.error("writeChatHistory:", e);
        Sentry.captureException(e);
      });

      // Auto-connect MCP servers from /etc/mcp-servers.json
      if (!this.session.mcpServersLoaded) {
        this.session.mcpServersLoaded = true;
        try {
          const callbackHost =
            (await this.ctx.storage.get<string>("callbackOrigin")) ?? "";
          const callbackPath = `mcp-callback/${this.ctx.id.toString()}`;
          await Sentry.startSpan({ name: "ensureMcpServers", op: "mcp" }, () =>
            ensureMcpServers(
              this.session.mountableFs,
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
      const { apiKeyType } = await this.session.resolveQuotaAndModel();

      // Compute system prompt, dynamic context, LLM model, and tools once per session.
      // Invalidate when this is the first message or after clear history (messages.length <= 1).
      const shouldRecompute =
        !this.session.cachedSystemPrompt || this.messages.length <= 1;

      if (shouldRecompute) {
        // Ensure jsonSchema is initialized for getAITools() — needed after DO hibernation
        try {
          await this.mcp.ensureJsonSchema();
        } catch (e) {
          console.error("mcp.ensureJsonSchema failed:", e);
          Sentry.captureException(e);
        }

        const { data: llmConfig, cache: newLlmCache } =
          await getCachedLlmConfig(
            this.session.mountableFs,
            this.session.cachedLlmConfig
          );
        this.session.cachedLlmConfig = newLlmCache;

        const [llmModel, memoryBlock] = await Promise.all([
          Sentry.startSpan({ name: "getLlmModel", op: "config" }, () =>
            getLlmModel(this.env, llmConfig)
          ),
          Sentry.startSpan({ name: "readMemoryBlock", op: "db.query" }, () =>
            readMemoryBlock(this.env.DB, this.session.userId!)
          )
        ]);

        this.session.cachedLlmModel = llmModel;
        this.session.cachedSystemPrompt = buildSystemPrompt();

        // Build dynamic context
        const dynamicParts = [
          `Chat history directory: /home/user/.chat/${this.session.sessionDir}/`,
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
        this.session.cachedDynamicContext = dynamicParts
          .filter(Boolean)
          .join("\n");

        // Cache tools (including MCP tools)
        let mcpTools: ToolSet = {};
        try {
          mcpTools = this.mcp.getAITools();
        } catch (e) {
          console.error("mcp.getAITools() failed:", e);
          Sentry.captureException(e);
        }
        this.session.cachedTools = {
          ...createTools({
            bashTool: createBashTool(this.session.bash, () =>
              this.session.ensureMounted()
            ),
            schedule: (when, method, payload) =>
              this.schedule(when, method as keyof typeof this, payload),
            getSchedules: () => this.getSchedules(),
            cancelSchedule: (id) => this.cancelSchedule(id),
            getTimezone: () => this.session.getTimezone(),
            debugBuffer: this.debugBuffer
          }),
          ...createDeviceTools(
            this.env,
            this.session.userId!,
            this.debugBuffer
          ),
          search: createSearchTool(this.env),
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

      messages.unshift({
        role: "system",
        content: this.session.cachedDynamicContext!
      });

      const { onResponse, hookAbort } = instrumentLlmCall(
        this.debugBuffer,
        buildRequestSnapshot(
          this.session.cachedSystemPrompt!,
          this.session.cachedDynamicContext!,
          messages,
          this.session.cachedTools!,
          this.session.cachedModelId!
        )
      );
      const archiveUsage = this.session.createUsageArchiver();

      const result = streamText({
        model: this.session.cachedLlmModel!,
        system: this.session.cachedSystemPrompt!,
        messages,
        tools: this.session.cachedTools!,
        stopWhen: stepCountIs(10),
        onFinish: async (event) => {
          onResponse(event);
          await archiveUsage();
          if (this.internalTaskDeferred) {
            this.internalTaskDeferred.resolve(event.text || "done");
            this.internalTaskDeferred = null;
          }
        },
        abortSignal: options?.abortSignal
      });
      hookAbort(options?.abortSignal);

      return this.toStreamResponse(result, apiKeyType);
    });
  }

  // ---- Device session streamText handler ----

  private async handleDeviceChatMessage(
    abortSignal?: AbortSignal
  ): Promise<Response> {
    // userId and bash are guaranteed by onChatMessage caller
    await this.session.ensureMounted();

    const { llmModel, apiKeyType, modelId } =
      await this.session.resolveQuotaAndModel();

    // Read device-reported system prompt and tool description from storage
    const deviceSystemPrompt =
      (await this.ctx.storage.get<string>("deviceSystemPrompt")) ||
      "You are a mobile automation assistant.";
    const deviceTools = await this.ctx.storage.get<DeviceTool[]>("deviceTools");

    // Build tool name and description from device-reported tools
    let toolDesc: string | undefined;
    let toolName: string | undefined;
    if (deviceTools && deviceTools.length > 0) {
      const fn = deviceTools[0]?.function;
      toolName = fn?.name;
      toolDesc = fn?.description;
    }

    const tools = createDeviceExecTool(this.deviceHub, toolDesc, toolName);

    const messages = await pruneMessages({
      messages: await convertToModelMessages(this.messages),
      toolCalls: "before-last-2-messages",
      reasoning: "before-last-message"
    });

    const { onResponse, hookAbort } = instrumentLlmCall(
      this.debugBuffer,
      buildRequestSnapshot(deviceSystemPrompt, "", messages, tools, modelId)
    );
    const archiveUsage = this.session.createUsageArchiver();

    const result = streamText({
      model: llmModel,
      system: deviceSystemPrompt,
      messages,
      tools,
      stopWhen: stepCountIs(30),
      onFinish: async (event) => {
        onResponse(event);
        await archiveUsage();
        // Resolve the deferred so handleDeviceInitiatedTask can return the result
        if (this.internalTaskDeferred) {
          this.internalTaskDeferred.resolve(event.text || "done");
          this.internalTaskDeferred = null;
        }
        this.deviceHub.sendTaskDone("done");
      },
      abortSignal
    });
    hookAbort(abortSignal);

    return this.toStreamResponse(result, apiKeyType);
  }

  // ---- Device-initiated task (from device WebSocket) ----

  private async handleDeviceInitiatedTask(text: string): Promise<string> {
    return Sentry.startSpan(
      { name: "handleDeviceInitiatedTask", op: "device_task" },
      async () => {
        try {
          await this.session.ensureReady();
          this.applySentryTags();

          const userMsg: UIMessage = {
            id: crypto.randomUUID(),
            role: "user",
            parts: [{ type: "text", text }]
          };

          // Set up a deferred promise that onFinish in handleDeviceChatMessage will resolve
          const resultPromise = new Promise<string>((resolve) => {
            this.internalTaskDeferred = { resolve };
          });

          // Reuse the same path as web-initiated messages:
          // saveMessages → onChatMessage → handleDeviceChatMessage → streamText → _reply
          await this.saveMessages([...this.messages, userMsg]);

          // Wait for the stream to fully complete (resolved by onFinish)
          const resultText = await resultPromise;
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
        getUserId: () => this.session.getUserId(),
        getSessionUuid: () => this.session.getSessionUuid(),
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
