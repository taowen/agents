/**
 * Scheduled tasks — DO-level business flow tests.
 *
 * Tests the executeScheduledTask deferred pattern:
 *   saveMessages → onChatMessage → deferred.resolve → caller unblocks
 *
 * Also covers D1 session guard and error handling.
 */
import { env } from "cloudflare:test";
import { describe, it, expect, beforeAll } from "vitest";
import { getAgentByName } from "agents";
import type { UIMessage as ChatMessage } from "ai";
import { applyD1Schema, createTestUser } from "./test-utils";

describe("Scheduled tasks — DO business flow", () => {
  let db: D1Database;

  beforeAll(async () => {
    db = (env as unknown as { DB: D1Database }).DB;
    await applyD1Schema(db);
  });

  it("executeScheduledTask persists [Scheduled Task] user message + assistant response", async () => {
    const room = crypto.randomUUID();
    const agentStub = await getAgentByName(env.TestChatAgent, room);

    const result = await agentStub.executeScheduledTask({
      description: "daily-report",
      prompt: "Generate the daily report",
      timezone: "Asia/Shanghai"
    });

    expect(result.success).toBe(true);

    const messages = (await agentStub.getPersistedMessages()) as ChatMessage[];
    expect(messages.length).toBe(2);

    // User message has the correct format
    const userMsg = messages[0];
    expect(userMsg.role).toBe("user");
    const userText = (userMsg.parts[0] as { type: string; text: string }).text;
    expect(userText).toContain("[Scheduled Task]");
    expect(userText).toContain("(Asia/Shanghai)");
    expect(userText).toContain("daily-report");
    expect(userText).toContain("Generate the daily report");

    // Assistant message is present (from onChatMessage)
    const assistantMsg = messages[1];
    expect(assistantMsg.role).toBe("assistant");
  });

  it("user message format: [Scheduled Task] ISO (TZ) - description\\n\\nprompt", async () => {
    const room = crypto.randomUUID();
    const agentStub = await getAgentByName(env.TestChatAgent, room);

    await agentStub.executeScheduledTask({
      description: "test-format",
      prompt: "do the thing",
      timezone: "US/Eastern"
    });

    const messages = (await agentStub.getPersistedMessages()) as ChatMessage[];
    const userText = (messages[0].parts[0] as { type: string; text: string })
      .text;

    // Verify the exact format: [Scheduled Task] ISO (TZ) - desc\n\nprompt
    const match = userText.match(
      /^\[Scheduled Task\] (\d{4}-\d{2}-\d{2}T[\d:.]+Z) \((.+?)\) - (.+?)\n\n(.+)$/s
    );
    expect(match).not.toBeNull();
    expect(match![2]).toBe("US/Eastern");
    expect(match![3]).toBe("test-format");
    expect(match![4]).toBe("do the thing");
    // Verify ISO timestamp is valid
    expect(new Date(match![1]).getTime()).toBeGreaterThan(0);
  });

  it("defaults timezone to UTC when payload.timezone is absent", async () => {
    const room = crypto.randomUUID();
    const agentStub = await getAgentByName(env.TestChatAgent, room);

    await agentStub.executeScheduledTask({
      description: "no-tz",
      prompt: "test"
    });

    const messages = (await agentStub.getPersistedMessages()) as ChatMessage[];
    const userText = (messages[0].parts[0] as { type: string; text: string })
      .text;
    expect(userText).toContain("(UTC)");
  });

  it("deferred resolves: executeScheduledTask returns after onChatMessage completes", async () => {
    const room = crypto.randomUUID();
    const agentStub = await getAgentByName(env.TestChatAgent, room);

    // The deferred should resolve without hanging — this test verifies
    // that the saveMessages → onChatMessage → deferred.resolve pipeline works.
    const result = await agentStub.executeScheduledTask({
      description: "deferred-test",
      prompt: "verify deferred resolution"
    });

    expect(result.success).toBe(true);
    // After resolution, messages should be persisted
    const count = await agentStub.getMessageCount();
    expect(count).toBe(2); // 1 user + 1 assistant
  });

  it("normal chat unaffected: no deferred means onChatMessage is a no-op for deferred", async () => {
    const room = crypto.randomUUID();
    const agentStub = await getAgentByName(env.TestChatAgent, room);

    // First: normal chat (no deferred set)
    const userMsg: ChatMessage = {
      id: "user-normal",
      role: "user",
      parts: [{ type: "text", text: "Hello, normal chat" }]
    };
    await agentStub.saveMessages([userMsg]);

    const messages = (await agentStub.getPersistedMessages()) as ChatMessage[];
    expect(messages.length).toBe(2); // 1 user + 1 assistant from onChatMessage

    // Then: scheduled task still works after normal chat
    const result = await agentStub.executeScheduledTask({
      description: "after-chat",
      prompt: "test"
    });
    expect(result.success).toBe(true);

    const allMessages =
      (await agentStub.getPersistedMessages()) as ChatMessage[];
    expect(allMessages.length).toBe(4); // 2 from chat + 2 from scheduled task
  });

  it("D1 session guard: missing session triggers cleanup and returns early", async () => {
    const room = crypto.randomUUID();
    const agentStub = await getAgentByName(env.TestChatAgent, room);

    // Set userId and sessionUuid in DO storage (simulates production state)
    const userId = await createTestUser(db);
    await agentStub.setStorageValue("userId", userId);
    await agentStub.setStorageValue("sessionUuid", "nonexistent-session-id");

    const result = await agentStub.executeScheduledTask({
      description: "orphaned-task",
      prompt: "should not run"
    });

    expect(result.success).toBe(false);
    expect(result.error).toBe("session_deleted");
  });

  it("D1 session guard: existing session allows task to proceed", async () => {
    const room = crypto.randomUUID();
    const agentStub = await getAgentByName(env.TestChatAgent, room);

    // Create a real user + session in D1
    const userId = await createTestUser(db);
    const sessionId = crypto.randomUUID();
    await db
      .prepare("INSERT INTO sessions (id, user_id, title) VALUES (?, ?, ?)")
      .bind(sessionId, userId, "Test Session")
      .run();

    // Set the same userId/sessionUuid in DO storage
    await agentStub.setStorageValue("userId", userId);
    await agentStub.setStorageValue("sessionUuid", sessionId);

    const result = await agentStub.executeScheduledTask({
      description: "valid-session-task",
      prompt: "this should run"
    });

    expect(result.success).toBe(true);

    const messages = (await agentStub.getPersistedMessages()) as ChatMessage[];
    expect(messages.length).toBe(2);

    const userText = (messages[0].parts[0] as { type: string; text: string })
      .text;
    expect(userText).toContain("valid-session-task");
  });

  it("D1 session guard: skipped when userId/sessionUuid not set", async () => {
    const room = crypto.randomUUID();
    const agentStub = await getAgentByName(env.TestChatAgent, room);

    // Don't set userId/sessionUuid — guard should be skipped
    const result = await agentStub.executeScheduledTask({
      description: "no-session-info",
      prompt: "test"
    });

    expect(result.success).toBe(true);

    const messages = (await agentStub.getPersistedMessages()) as ChatMessage[];
    expect(messages.length).toBe(2);
  });
});
