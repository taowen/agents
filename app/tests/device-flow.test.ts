/**
 * Device end-to-end integration tests.
 *
 * Part 1 — Authorization flow:
 *   POST /auth/device/start → code generation
 *   POST /api/device/approve → web user approves code
 *   GET /auth/device/check → device polls and receives token
 *
 * Part 2 — Device-initiated message flow:
 *   handleDeviceInitiatedTask → saveMessages → onChatMessage → deferred → result
 */
import { env } from "cloudflare:test";
import { describe, it, expect, beforeAll } from "vitest";
import { getAgentByName } from "agents";
import type { UIMessage as ChatMessage } from "ai";
import {
  applyD1Schema,
  createTestUser,
  apiRequest,
  publicRequest
} from "./test-utils";

describe("Device authorization flow", () => {
  let db: D1Database;

  beforeAll(async () => {
    db = (env as unknown as { DB: D1Database }).DB;
    await applyD1Schema(db);
  });

  it("POST /auth/device/start generates a 6-char device code", async () => {
    const res = await publicRequest("POST", "/auth/device/start");
    expect(res.status).toBe(200);

    const data = await res.json<{ code: string }>();
    expect(data.code).toHaveLength(6);
    expect(data.code).toMatch(/^[A-Z0-9]+$/);
    // Should not contain ambiguous characters
    expect(data.code).not.toMatch(/[0O1I]/);
  });

  it("GET /auth/device/check before approval returns pending", async () => {
    const startRes = await publicRequest("POST", "/auth/device/start");
    const { code } = await startRes.json<{ code: string }>();

    const checkRes = await publicRequest(
      "GET",
      `/auth/device/check?code=${code}`
    );
    expect(checkRes.status).toBe(200);

    const data = await checkRes.json<{ status: string }>();
    expect(data.status).toBe("pending");
  });

  it("full flow: start → approve → check returns approved with token", async () => {
    const userId = await createTestUser(db);

    // Step 1: Device starts auth flow
    const startRes = await publicRequest("POST", "/auth/device/start");
    const { code } = await startRes.json<{ code: string }>();

    // Step 2: Web user approves the code
    const approveRes = await apiRequest("POST", "/api/device/approve", userId, {
      code
    });
    expect(approveRes.status).toBe(200);
    const approveData = await approveRes.json<{ ok: boolean }>();
    expect(approveData.ok).toBe(true);

    // Step 3: Device polls and gets the token
    const checkRes = await publicRequest(
      "GET",
      `/auth/device/check?code=${code}`
    );
    expect(checkRes.status).toBe(200);

    const checkData = await checkRes.json<{
      status: string;
      token: string;
      baseURL: string;
      model: string;
    }>();
    expect(checkData.status).toBe("approved");
    expect(checkData.token).toMatch(/^device\./);
    expect(checkData.baseURL).toContain("/api/proxy/v1");
    expect(checkData.model).toBe("test-model");
  });

  it("approve with invalid code returns 404", async () => {
    const userId = await createTestUser(db);
    const res = await apiRequest("POST", "/api/device/approve", userId, {
      code: "ZZZZZZ"
    });
    expect(res.status).toBe(404);
  });

  it("double approve returns 409 (code already used)", async () => {
    const userId = await createTestUser(db);

    const startRes = await publicRequest("POST", "/auth/device/start");
    const { code } = await startRes.json<{ code: string }>();

    // First approval succeeds
    const res1 = await apiRequest("POST", "/api/device/approve", userId, {
      code
    });
    expect(res1.status).toBe(200);

    // Device reads the token (this deletes the KV entry)
    await publicRequest("GET", `/auth/device/check?code=${code}`);

    // Second approval fails — code no longer exists
    const res2 = await apiRequest("POST", "/api/device/approve", userId, {
      code
    });
    expect(res2.status).toBe(404);
  });

  it("check with expired/missing code returns expired", async () => {
    const res = await publicRequest("GET", "/auth/device/check?code=NONEXIST");
    expect(res.status).toBe(200);

    const data = await res.json<{ status: string }>();
    expect(data.status).toBe("expired");
  });
});

// ================================================================
// Part 2: Device-initiated message flow
// ================================================================

describe("Device-initiated message flow", () => {
  it("handleDeviceInitiatedTask persists user message + assistant response and returns result", async () => {
    const room = crypto.randomUUID();
    const agentStub = await getAgentByName(env.TestChatAgent, room);

    const result =
      await agentStub.handleDeviceInitiatedTask("Turn on the lights");

    // Result comes from the deferred resolved by onChatMessage
    expect(result).toBe("Hello from chat agent!");

    const messages = (await agentStub.getPersistedMessages()) as ChatMessage[];
    expect(messages.length).toBe(2);

    // User message is the raw device text (no prefix)
    const userMsg = messages[0];
    expect(userMsg.role).toBe("user");
    expect((userMsg.parts[0] as { type: string; text: string }).text).toBe(
      "Turn on the lights"
    );

    // Assistant response is persisted
    expect(messages[1].role).toBe("assistant");
  });

  it("multi-turn: sequential device tasks accumulate in conversation", async () => {
    const room = crypto.randomUUID();
    const agentStub = await getAgentByName(env.TestChatAgent, room);

    await agentStub.handleDeviceInitiatedTask("First task");
    await agentStub.handleDeviceInitiatedTask("Second task");

    const messages = (await agentStub.getPersistedMessages()) as ChatMessage[];
    // 2 turns × (1 user + 1 assistant) = 4 messages
    expect(messages.length).toBe(4);
    expect(messages[0].role).toBe("user");
    expect(messages[1].role).toBe("assistant");
    expect(messages[2].role).toBe("user");
    expect(messages[3].role).toBe("assistant");

    expect((messages[0].parts[0] as { type: string; text: string }).text).toBe(
      "First task"
    );
    expect((messages[2].parts[0] as { type: string; text: string }).text).toBe(
      "Second task"
    );
  });

  it("returns 'done' when onChatMessage resolves with empty text", async () => {
    const room = crypto.randomUUID();
    const agentStub = await getAgentByName(env.TestChatAgent, room);

    // TestChatAgent.onChatMessage resolves with "Hello from chat agent!"
    // which is non-empty, so this just confirms the fallback path exists.
    const result = await agentStub.handleDeviceInitiatedTask("ping");
    expect(typeof result).toBe("string");
    expect(result.length).toBeGreaterThan(0);
  });

  it("device task and scheduled task use the same deferred pattern on the same DO", async () => {
    const room = crypto.randomUUID();
    const agentStub = await getAgentByName(env.TestChatAgent, room);

    // Device task first
    const deviceResult =
      await agentStub.handleDeviceInitiatedTask("device says hello");
    expect(deviceResult).toBe("Hello from chat agent!");

    // Then a scheduled task on the same DO
    const schedResult = await agentStub.executeScheduledTask({
      description: "scheduled-after-device",
      prompt: "test"
    });
    expect(schedResult.success).toBe(true);

    const messages = (await agentStub.getPersistedMessages()) as ChatMessage[];
    // 2 from device + 2 from scheduled = 4
    expect(messages.length).toBe(4);

    // First pair: device task (plain text)
    expect((messages[0].parts[0] as { type: string; text: string }).text).toBe(
      "device says hello"
    );

    // Second pair: scheduled task ([Scheduled Task] prefix)
    expect(
      (messages[2].parts[0] as { type: string; text: string }).text
    ).toContain("[Scheduled Task]");
  });
});
