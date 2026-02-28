/**
 * Bug reporting integration tests.
 *
 * Tests both:
 * 1. Debug ring buffer via DO methods (data layer)
 * 2. POST /api/sessions/:id/report-bug end-to-end HTTP flow → R2
 */
import { env } from "cloudflare:test";
import { describe, it, expect, beforeAll } from "vitest";
import { getServerByName } from "partyserver";
import type {
  DebugEntry,
  LlmInteractionEntry
} from "../ai-chat/src/server/llm-debug-buffer";
import {
  connectChatWS,
  applyD1Schema,
  createTestUser,
  apiRequest
} from "./test-utils";

function makeLlmEntry(
  overrides?: Partial<LlmInteractionEntry>
): LlmInteractionEntry {
  return {
    type: "llm",
    timestamp: new Date().toISOString(),
    traceId: "trace-1",
    spanId: "span-1",
    request: {
      systemPrompt: "You are a test assistant",
      dynamicContext: "",
      messages: [],
      toolNames: [],
      modelId: "test-model"
    },
    response: null,
    ...overrides
  };
}

describe("Bug reporting", () => {
  let db: D1Database;

  beforeAll(async () => {
    db = (env as unknown as { DB: D1Database }).DB;
    await applyD1Schema(db);
  });

  // ---- Debug ring buffer tests (via DO methods) ----

  const room = "debug-buffer-test-room";

  async function ensureDOInitialized() {
    const { ws } = await connectChatWS(`/agents/test-chat-agent/${room}`);
    ws.close(1000);
  }

  it("debug ring buffer stores and retrieves entries in order", async () => {
    await ensureDOInitialized();
    const stub = await getServerByName(env.TestChatAgent, room);
    await stub.debugBufferReset(20);

    await stub.debugBufferPush(makeLlmEntry({ traceId: "trace-1" }));
    await stub.debugBufferPush(makeLlmEntry({ traceId: "trace-2" }));
    await stub.debugBufferPush(makeLlmEntry({ traceId: "trace-3" }));

    const all = (await stub.debugBufferGetAll()) as DebugEntry[];
    expect(all).toHaveLength(3);
    expect(all[0].traceId).toBe("trace-1");
    expect(all[1].traceId).toBe("trace-2");
    expect(all[2].traceId).toBe("trace-3");
  });

  it("ring buffer evicts oldest beyond maxSize", async () => {
    await ensureDOInitialized();
    const stub = await getServerByName(env.TestChatAgent, room);
    await stub.debugBufferReset(3);

    for (let i = 0; i < 5; i++) {
      await stub.debugBufferPush(makeLlmEntry({ traceId: `trace-${i}` }));
    }

    const all = (await stub.debugBufferGetAll()) as DebugEntry[];
    expect(all).toHaveLength(3);
    expect(all[0].traceId).toBe("trace-2");
    expect(all[1].traceId).toBe("trace-3");
    expect(all[2].traceId).toBe("trace-4");
  });

  it("ring buffer updateResponse modifies entry in place", async () => {
    await ensureDOInitialized();
    const stub = await getServerByName(env.TestChatAgent, room);
    await stub.debugBufferReset(20);

    const rowId = (await stub.debugBufferPush(makeLlmEntry())) as number;

    const response: LlmInteractionEntry["response"] = {
      text: "Updated response",
      finishReason: "stop",
      usage: { inputTokens: 10, outputTokens: 20 },
      stepCount: 1,
      steps: []
    };

    await stub.debugBufferUpdateResponse(rowId, response);

    const all = (await stub.debugBufferGetAll()) as LlmInteractionEntry[];
    expect(all).toHaveLength(1);
    expect(all[0].response).toBeDefined();
    expect(all[0].response!.text).toBe("Updated response");
    expect(all[0].response!.finishReason).toBe("stop");
  });

  // ---- HTTP API end-to-end test ----

  it("POST /api/sessions/:id/report-bug writes payload to R2", async () => {
    const userId = await createTestUser(db);

    // Create a session first
    const sessionRes = await apiRequest("POST", "/api/sessions", userId);
    const session = await sessionRes.json<{ id: string }>();

    // Submit bug report via HTTP API
    const bugRes = await apiRequest(
      "POST",
      `/api/sessions/${session.id}/report-bug`,
      userId,
      { description: "Something is broken" }
    );
    expect(bugRes.status).toBe(200);

    const { reportId } = await bugRes.json<{ reportId: string }>();
    expect(reportId).toMatch(/^BUG-[A-Z0-9]+-[A-F0-9]{4}$/);

    // Verify the payload was written to R2
    const r2 = (env as unknown as { R2: R2Bucket }).R2;
    const r2Key = `bug-reports/${reportId}.json`;
    const obj = await r2.get(r2Key);
    expect(obj).not.toBeNull();

    const payload = JSON.parse(await obj!.text());
    expect(payload.reportId).toBe(reportId);
    expect(payload.description).toBe("Something is broken");
    expect(payload.userId).toBe(userId);
    expect(payload.sessionId).toBe(session.id);
    expect(payload.capturedAt).toBeDefined();
  });

  it("POST /api/sessions/:id/report-bug unauthenticated returns 401", async () => {
    const res = await apiRequest(
      "POST",
      "/api/sessions/some-id/report-bug",
      "",
      { description: "test" }
    );
    expect(res.status).toBe(401);
  });
});
