/**
 * Multi-turn chat conversation flow tests.
 *
 * Covers single-turn messages, multi-turn conversations, message ID handling,
 * tool-call deduplication, and new-connection message broadcast.
 *
 * Absorbs scenarios from duplicate-message.test.ts (BUG-MLYJD5QH-8922).
 */
import { createExecutionContext, env } from "cloudflare:test";
import { describe, it, expect } from "vitest";
import { getServerByName } from "partyserver";
import type { UIMessage as ChatMessage } from "ai";
import {
  connectChatWS,
  sendChatRequest,
  extractStartMessageId
} from "./test-utils";
import worker from "./worker";

describe("Multi-turn chat", () => {
  it("single turn: send message, get streaming response with 1 user + 1 assistant persisted", async () => {
    const room = crypto.randomUUID();
    const { ws } = await connectChatWS(`/agents/test-chat-agent/${room}`);

    const userMsg: ChatMessage = {
      id: "user-1",
      role: "user",
      parts: [{ type: "text", text: "Hello" }]
    };

    await sendChatRequest(ws, "req-1", [userMsg]);

    const agentStub = await getServerByName(env.TestChatAgent, room);
    const messages = (await agentStub.getPersistedMessages()) as ChatMessage[];
    expect(messages.length).toBe(2);
    expect(messages[0].role).toBe("user");
    expect(messages[1].role).toBe("assistant");

    ws.close(1000);
  });

  it("start event includes server messageId matching persisted assistant ID", async () => {
    const room = crypto.randomUUID();
    const { ws } = await connectChatWS(`/agents/test-chat-agent/${room}`);

    const userMsg: ChatMessage = {
      id: "user-1",
      role: "user",
      parts: [{ type: "text", text: "Hello" }]
    };

    const received = await sendChatRequest(ws, "req-1", [userMsg]);

    const serverMessageId = extractStartMessageId(received);
    expect(serverMessageId).toBeDefined();
    expect(serverMessageId).toMatch(/^assistant_/);

    const agentStub = await getServerByName(env.TestChatAgent, room);
    const messages = (await agentStub.getPersistedMessages()) as ChatMessage[];
    const serverAssistant = messages.find((m) => m.role === "assistant")!;
    expect(serverAssistant.id).toBe(serverMessageId);

    ws.close(1000);
  });

  it("multi-turn: client uses server messageId, exactly 2 assistants after 2 turns", async () => {
    const room = crypto.randomUUID();
    const { ws } = await connectChatWS(`/agents/test-chat-agent/${room}`);

    // Turn 1
    const userMsg1: ChatMessage = {
      id: "user-1",
      role: "user",
      parts: [{ type: "text", text: "Hello" }]
    };

    const received = await sendChatRequest(ws, "req-1", [userMsg1]);
    const serverMessageId = extractStartMessageId(received);
    expect(serverMessageId).toBeDefined();

    const agentStub = await getServerByName(env.TestChatAgent, room);
    const messagesAfterTurn1 =
      (await agentStub.getPersistedMessages()) as ChatMessage[];
    expect(messagesAfterTurn1.length).toBe(2);

    const serverAssistant = messagesAfterTurn1.find(
      (m) => m.role === "assistant"
    )!;

    // Turn 2: client uses server's messageId
    const clientAssistantMsg: ChatMessage = {
      id: serverMessageId!,
      role: "assistant",
      parts: serverAssistant.parts
    };

    const userMsg2: ChatMessage = {
      id: "user-2",
      role: "user",
      parts: [{ type: "text", text: "How are you?" }]
    };

    await sendChatRequest(ws, "req-2", [
      userMsg1,
      clientAssistantMsg,
      userMsg2
    ]);

    const messagesAfterTurn2 =
      (await agentStub.getPersistedMessages()) as ChatMessage[];
    const assistantMessages = messagesAfterTurn2.filter(
      (m) => m.role === "assistant"
    );
    expect(assistantMessages.length).toBe(2);

    ws.close(1000);
  });

  it("regression fixed: client ignores messageId, dedup reconciles (2 assistants)", async () => {
    const room = crypto.randomUUID();
    const { ws } = await connectChatWS(`/agents/test-chat-agent/${room}`);

    // Turn 1
    const userMsg1: ChatMessage = {
      id: "user-1",
      role: "user",
      parts: [{ type: "text", text: "Hello" }]
    };

    await sendChatRequest(ws, "req-1", [userMsg1]);

    const agentStub = await getServerByName(env.TestChatAgent, room);
    const messagesAfterTurn1 =
      (await agentStub.getPersistedMessages()) as ChatMessage[];
    const serverAssistant = messagesAfterTurn1.find(
      (m) => m.role === "assistant"
    )!;

    // Turn 2: client deliberately uses a DIFFERENT ID
    const clientAssistantMsg: ChatMessage = {
      id: "rogue-client-id",
      role: "assistant",
      parts: serverAssistant.parts
    };

    const userMsg2: ChatMessage = {
      id: "user-2",
      role: "user",
      parts: [{ type: "text", text: "How are you?" }]
    };

    await sendChatRequest(ws, "req-2", [
      userMsg1,
      clientAssistantMsg,
      userMsg2
    ]);

    const messagesAfterTurn2 =
      (await agentStub.getPersistedMessages()) as ChatMessage[];
    const assistantMessages = messagesAfterTurn2.filter(
      (m) => m.role === "assistant"
    );

    // _reconcileAssistantIdsWithServerState matches the rogue client ID to the
    // server's assistant by content, replacing the ID before persistence.
    // The upsert then updates the existing row instead of creating a duplicate.
    // Result: only 2 assistant messages (turn 1 + turn 2), not 3.
    expect(assistantMessages.length).toBe(2);

    ws.close(1000);
  });

  it("tool-call responses deduplicated by toolCallId", async () => {
    const room = crypto.randomUUID();

    const ctx = createExecutionContext();
    const req = new Request(
      `http://example.com/agents/test-chat-agent/${room}`,
      { headers: { Upgrade: "websocket" } }
    );
    const res = await worker.fetch(req, env, ctx);
    expect(res.status).toBe(101);
    const ws = res.webSocket as WebSocket;
    ws.accept();
    await ctx.waitUntil(Promise.resolve());

    const agentStub = await getServerByName(env.TestChatAgent, room);

    const toolCallId = "call_test_123";
    await agentStub.persistMessages([
      {
        id: "user-1",
        role: "user",
        parts: [{ type: "text", text: "What time is it?" }]
      },
      {
        id: "server-assistant-id",
        role: "assistant",
        parts: [
          {
            type: "tool-getTime",
            toolCallId,
            state: "output-available",
            input: { tz: "UTC" },
            output: "12:00"
          }
        ] as ChatMessage["parts"]
      }
    ]);

    // Client sends messages with a DIFFERENT assistant ID but SAME toolCallId
    await agentStub.persistMessages([
      {
        id: "user-1",
        role: "user",
        parts: [{ type: "text", text: "What time is it?" }]
      },
      {
        id: "client-different-id",
        role: "assistant",
        parts: [
          {
            type: "tool-getTime",
            toolCallId,
            state: "output-available",
            input: { tz: "UTC" },
            output: "12:00"
          }
        ] as ChatMessage["parts"]
      }
    ]);

    const messages = (await agentStub.getPersistedMessages()) as ChatMessage[];
    const assistantMessages = messages.filter((m) => m.role === "assistant");

    // Tool-call messages are deduplicated by toolCallId — only 1 assistant message
    expect(assistantMessages.length).toBe(1);

    ws.close(1000);
  });

  it("new connection receives empty cf_agent_chat_messages (history loaded via /get-messages)", async () => {
    const room = crypto.randomUUID();
    const { ws: ws1 } = await connectChatWS(`/agents/test-chat-agent/${room}`);

    // Send a message on the first connection
    const userMsg: ChatMessage = {
      id: "user-1",
      role: "user",
      parts: [{ type: "text", text: "Hello" }]
    };
    await sendChatRequest(ws1, "req-1", [userMsg]);

    // Open a second connection manually: register listener BEFORE accept()
    // to avoid missing the cf_agent_chat_messages sent during onConnect.
    const ctx2 = createExecutionContext();
    const req2 = new Request(
      `http://example.com/agents/test-chat-agent/${room}`,
      { headers: { Upgrade: "websocket" } }
    );
    const res2 = await worker.fetch(req2, env, ctx2);
    expect(res2.status).toBe(101);
    const ws2 = res2.webSocket as WebSocket;

    const messagePromise = new Promise<unknown>((resolve) => {
      const timeout = setTimeout(() => resolve(null), 2000);
      ws2.addEventListener("message", (e: MessageEvent) => {
        const data = JSON.parse(e.data as string);
        if (data.type === "cf_agent_chat_messages") {
          clearTimeout(timeout);
          resolve(data);
        }
      });
    });

    ws2.accept();
    const messagesReceived = await messagePromise;

    // onConnect now sends empty messages intentionally — initial message
    // loading is handled by the HTTP /get-messages endpoint to avoid
    // redundant bandwidth on WebSocket connect.
    expect(messagesReceived).not.toBeNull();
    const chatMessagesEvent = messagesReceived as {
      type: string;
      messages: ChatMessage[];
    };
    expect(chatMessagesEvent.type).toBe("cf_agent_chat_messages");
    expect(chatMessagesEvent.messages.length).toBe(0);

    ws1.close(1000);
    ws2.close(1000);
  });
});
