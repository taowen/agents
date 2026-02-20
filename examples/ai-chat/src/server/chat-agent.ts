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
import {
  createBrowserState,
  createBrowserTool,
  type BrowserState
} from "./browser-tool";
import { buildSystemPrompt } from "./system-prompt";

// Extracted modules
import { initBash, doFstabMount } from "./fs-init";
import { ensureMcpServers } from "./mcp-config";
import {
  getCachedLlmConfig,
  getLlmModel,
  type LlmConfigCache
} from "./llm-config";
import { handleFileRequest } from "./file-api";
import { writeChatHistory, readMemoryBlock } from "./chat-history";
import { createBashTool, createTools } from "./tools";

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
  private cachedLlmConfig: LlmConfigCache = null;
  private mcpServersLoaded = false;

  // System prompt cache — computed once per session, invalidated on clear history
  private cachedSystemPrompt: string | null = null;
  private cachedDynamicContext: string | null = null;
  private cachedLlmModel: ReturnType<typeof getLlmModel> | null = null;
  private cachedTools: ToolSet | null = null;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.browserState = createBrowserState();
  }

  /**
   * Initialize bash + filesystem for the given userId.
   */
  private doInitBash(userId: string): void {
    if (this.bash && this.userId === userId) return;
    this.userId = userId;
    const { bash, mountableFs } = initBash(this.env, userId);
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
        this.env,
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
      const uid = await this.getUserId(request);
      this.doInitBash(uid);
      await this.ensureMounted();
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
        try {
          if (!this.userId) {
            const uid = await this.getUserId();
            this.doInitBash(uid);
          }

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
              bash: createBashTool(this.bash, () => this.ensureMounted()),
              browser: createBrowserTool(this.browserState, this.env.MYBROWSER)
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
            browserState: this.browserState,
            mybrowser: this.env.MYBROWSER,
            schedule: (when, method, payload) =>
              this.schedule(when, method, payload),
            getSchedules: () => this.getSchedules(),
            cancelSchedule: (id) => this.cancelSchedule(id),
            getTimezone: () => this.getTimezone()
          }),
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
