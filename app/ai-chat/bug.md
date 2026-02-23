 ---                                                                                
  BUG-MLYJD5QH-8922 Diagnosis: Duplicate Assistant Messages       
                                                                                     
  User Report                                                     

  "怀疑是bug啊。我：你好！有什么我可以帮你的吗？（回复了两次）"
  (I suspect this is a bug. I said: Hello! Is there anything I can help you with?
  (replied twice))

  Observed Behavior

  The DO message history contains 10 messages where there should be 8. Text-only
  assistant responses are duplicated with different IDs and slightly different
  metadata:
  #: 0
  Role: user
  ID: 4ZKhd7BuWDqP7nHL
  finishReason: -
  Text: 你好
  ────────────────────────────────────────
  #: 1
  Role: assistant
  ID: assistant_1771812315347_aufp0j...
  finishReason: missing
  Text: 你好！有什么我可以帮你的吗？
  ────────────────────────────────────────
  #: 2
  Role: assistant
  ID: sWkbVruU3Zts6JKl
  finishReason: stop
  Text: 你好！有什么我可以帮你的吗？
  ────────────────────────────────────────
  #: ...
  Role:
  ID:
  finishReason:
  Text:
  ────────────────────────────────────────
  #: 6
  Role: assistant
  ID: assistant_1771812351648_mfg5tr...
  finishReason: missing
  Text: 是的，你上一条消息确实...
  ────────────────────────────────────────
  #: 7
  Role: assistant
  ID: fAxMmnNzURqaeM8c
  finishReason: stop
  Text: 是的，你上一条消息确实...
  Responses with tool calls (messages 4, 9) are NOT duplicated.

  Root Cause

  The bug is a server-client message ID mismatch in packages/ai-chat/src/index.ts
  caused by a missing messageId in the SSE start event.

  Chain of causation:

  1. toUIMessageStreamResponse() is called without originalMessages
  (chat-agent.ts:454). The AI SDK only includes messageId in the SSE start event when
   originalMessages is provided.
  2. Server and client generate independent IDs. The server creates
  assistant_${Date.now()}_${random} at index.ts:1864. The client's useChat (from
  @ai-sdk/react) creates its own nanoid. Since the start event has no messageId,
  neither side updates to match the other (index.ts:1665 — the if (data.messageId !=
  null) guard is never true).
  3. Next user message triggers duplication. When the user sends the next message,
  the client includes the assistant message with the client's ID in its messages
  array. The server calls persistMessages() (index.ts:412) which INSERT-upserts by
  ID. Since the client's ID differs from the server's, a new row is created rather
  than updating the existing one. DB now has two assistant messages with identical
  text but different IDs.
  4. Tool-call responses are immune because _resolveMessageForToolMerge()
  (index.ts:1043) deduplicates by matching toolCallId across messages, remapping the
  client's ID to the server's existing row. Text-only messages have no toolCallId, so
   no deduplication occurs.

  Secondary issue: The finishReason field is missing from the server-persisted
  message metadata. At index.ts:1675-1681, the server captures data.messageMetadata
  (which only has usage + apiKeyType from the user callback). The finishReason
  conversion into messageMetadata happens AFTER at index.ts:1706-1724, but only for
  the broadcast copy (eventToSend), not for message.metadata.

  Fix Location

  packages/ai-chat/src/index.ts, _streamSSEReply method, around line 1705:

  When broadcasting a start event that lacks messageId, inject the server's
  message.id so the client adopts it:

  let eventToSend: unknown = data;
  // Ensure start event includes messageId for server-client ID sync
  if (data.type === "start" && data.messageId == null) {
    eventToSend = { ...(data as object), messageId: message.id };
  }
  // existing finishReason conversion...
  if (data.type === "finish" && "finishReason" in data) {
    // ...
  }

  This ensures the client receives the server's assistant_* ID in the start event and
   uses it, preventing the ID mismatch that causes duplicates on the next message
  send.