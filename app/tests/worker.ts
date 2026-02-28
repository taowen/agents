import {
  AIChatAgent,
  type OnChatMessageOptions
} from "../ai-chat/src/lib/ai-chat";
import type {
  UIMessage as ChatMessage,
  StreamTextOnFinishCallback,
  ToolSet
} from "ai";
import { Hono } from "hono";
import { routeAgentRequest } from "../ai-chat/src/lib/agents";
import {
  DebugRingBuffer,
  type DebugEntry,
  type LlmInteractionEntry
} from "../ai-chat/src/server/llm-debug-buffer";
import { apiRoutes } from "../ai-chat/src/server/api";
import { handleAuthRoutes } from "../ai-chat/src/server/auth";

// Test Env is an intentional subset of production Env — it omits
// Google/Sentry/GitHub bindings that aren't needed in integration tests,
// but includes everything the LLM proxy and quota paths require.
export type Env = {
  TestChatAgent: DurableObjectNamespace<TestChatAgent>;
  ChatAgent: DurableObjectNamespace<TestChatAgent>;
  DB: D1Database;
  R2: R2Bucket;
  OTP_KV: KVNamespace;
  AUTH_SECRET: string;
  BUILTIN_LLM_MODEL: string;
  BUILTIN_LLM_BASE_URL: string;
  BUILTIN_LLM_API_KEY: string;
  BUILTIN_LLM_PROVIDER: string;
  ADMIN_SECRET: string;
};

export class TestChatAgent extends AIChatAgent<Env> {
  observability = undefined;
  private debugBuffer = new DebugRingBuffer(20, this.ctx.storage.sql);

  // Deferred promise for internal tasks (scheduled / device-initiated):
  // resolved by onChatMessage, mirrors production internalTaskDeferred.
  private internalTaskDeferred: { resolve: (text: string) => void } | null =
    null;

  async onChatMessage(
    _onFinish: StreamTextOnFinishCallback<ToolSet>,
    _options?: OnChatMessageOptions
  ) {
    // Resolve deferred if set — mirrors production streamText.onFinish callback
    if (this.internalTaskDeferred) {
      this.internalTaskDeferred.resolve("Hello from chat agent!");
      this.internalTaskDeferred = null;
    }
    return new Response("Hello from chat agent!", {
      headers: { "Content-Type": "text/plain" }
    });
  }

  /**
   * Mirrors production handleDeviceInitiatedTask: creates a user message from
   * device text, sets up deferred, calls saveMessages → onChatMessage pipeline,
   * returns the assistant's response text.
   */
  async handleDeviceInitiatedTask(text: string): Promise<string> {
    try {
      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        parts: [{ type: "text", text }]
      };

      const resultPromise = new Promise<string>((resolve) => {
        this.internalTaskDeferred = { resolve };
      });

      await this.saveMessages([...this.messages, userMsg]);
      const resultText = await resultPromise;
      return resultText || "done";
    } catch (e) {
      this.internalTaskDeferred = null;
      return "Error: " + (e instanceof Error ? e.message : String(e));
    }
  }

  /**
   * Mirrors production executeScheduledTask: D1 session guard, message
   * construction, saveMessages → onChatMessage → deferred pattern, error
   * handling.  Returns a result object so tests can assert outcomes.
   */
  async executeScheduledTask(payload: {
    description: string;
    prompt: string;
    timezone?: string;
  }): Promise<{ success: boolean; error?: string }> {
    // ---- D1 session guard (same logic as production) ----
    const userId = await this.ctx.storage.get<string>("userId");
    const sessionUuid = await this.ctx.storage.get<string>("sessionUuid");

    if (userId && sessionUuid) {
      const row = await this.env.DB.prepare(
        "SELECT 1 FROM sessions WHERE id = ? AND user_id = ?"
      )
        .bind(sessionUuid, userId)
        .first();
      if (!row) {
        const schedules = this.getSchedules();
        for (const s of schedules) {
          await this.cancelSchedule(s.id);
        }
        await this.ctx.storage.deleteAll();
        return { success: false, error: "session_deleted" };
      }
    }

    // ---- Build user message ----
    const tz = payload.timezone || "UTC";
    const now = new Date();
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      parts: [
        {
          type: "text",
          text: `[Scheduled Task] ${now.toISOString()} (${tz}) - ${payload.description}\n\n${payload.prompt}`
        }
      ]
    };

    // ---- Deferred + saveMessages ----
    const resultPromise = new Promise<string>((resolve) => {
      this.internalTaskDeferred = { resolve };
    });

    try {
      await this.saveMessages([...this.messages, userMsg]);
      await resultPromise;
      return { success: true };
    } catch (e) {
      this.internalTaskDeferred = null;

      const errorMsg: ChatMessage = {
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
      } catch {}
      return { success: false, error: String(e) };
    }
  }

  getPersistedMessages(): ChatMessage[] {
    return (
      this.sql`select * from cf_ai_chat_agent_messages order by created_at` ||
      []
    ).map((row) => JSON.parse(row.message as string));
  }

  getMessageCount(): number {
    const rows =
      this.sql`select count(*) as cnt from cf_ai_chat_agent_messages` || [];
    return Number((rows[0] as { cnt: number })?.cnt ?? 0);
  }

  async setStorageValue(key: string, value: string): Promise<void> {
    await this.ctx.storage.put(key, value);
  }

  async getStorageValue(key: string): Promise<string | undefined> {
    return this.ctx.storage.get<string>(key);
  }

  // ---- Debug buffer methods for testing ----

  debugBufferPush(entry: DebugEntry): number {
    return this.debugBuffer.push(entry);
  }

  debugBufferGetAll(): DebugEntry[] {
    return this.debugBuffer.getAll();
  }

  debugBufferSize(): number {
    return this.debugBuffer.size;
  }

  debugBufferUpdateResponse(
    rowId: number,
    response: LlmInteractionEntry["response"]
  ): void {
    this.debugBuffer.updateResponse(rowId, response);
  }

  debugBufferReset(maxSize?: number): void {
    this.ctx.storage.sql.exec("DROP TABLE IF EXISTS debug_entries");
    this.debugBuffer = new DebugRingBuffer(maxSize ?? 20, this.ctx.storage.sql);
  }
}

// ---- Hono app for HTTP API testing ----

type AppEnv = { Bindings: Env; Variables: { userId: string } };

const app = new Hono<AppEnv>();

// Test helper endpoints (direct D1/R2 access)
app.post("/test-api/d1-exec", async (c) => {
  const { sql, params } = await c.req.json<{
    sql: string;
    params?: unknown[];
  }>();
  const stmt = c.env.DB.prepare(sql);
  const result = params ? await stmt.bind(...params).all() : await stmt.all();
  return c.json(result);
});

app.post("/test-api/r2-get", async (c) => {
  const { key } = await c.req.json<{ key: string }>();
  const obj = await c.env.R2.get(key);
  if (!obj) return c.json({ found: false });
  const text = await obj.text();
  return c.json({ found: true, body: text });
});

// Public auth routes (no auth required) — mirrors production index.ts
app.all("/auth/*", (c) => handleAuthRoutes(c.req.raw, c.env));

// Test auth middleware: reads x-test-user-id header
// In tests, set this header to simulate an authenticated user.
app.use("/api/*", async (c, next) => {
  const userId = c.req.header("x-test-user-id");
  if (!userId) {
    return c.json({ error: "Unauthorized" }, 401);
  }
  c.set("userId", userId);
  return next();
});

// API routes (same as production, mounted under /api)
app.route("/api", apiRoutes);

// Agent catch-all (WebSocket + routeAgentRequest)
app.all("*", async (c) => {
  return (
    (await routeAgentRequest(c.req.raw, c.env)) ||
    new Response("Not found", { status: 404 })
  );
});

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext) {
    return app.fetch(request, env, ctx);
  }
};
