import type { UIMessage } from "ai";
import { archiveLlmRequestUsage } from "./usage-tracker";

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

export function isDeviceSession(sessionUuid: string | null): boolean {
  return !!sessionUuid?.startsWith("device-");
}

export function getDeviceName(sessionUuid: string): string {
  return sessionUuid.slice("device-".length);
}

export interface LlmProxyConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
  apiKeyType: string;
}

export class DeviceHub {
  private pendingTasks = new Map<string, PendingTask>();
  private streamWriter: WritableStreamDefaultWriter<Uint8Array> | null = null;
  private streamEncoder = new TextEncoder();
  private emittedToolCallIds = new Set<string>();
  private initialToolMessageCount = 0;

  constructor(private ctx: DurableObjectState) {}

  // --- WebSocket management ---

  getWebSockets(): WebSocket[] {
    return this.ctx.getWebSockets("device");
  }

  getWs(): WebSocket | null {
    const sockets = this.getWebSockets();
    return sockets.length > 0 ? sockets[0] : null;
  }

  // --- Heartbeat (called from alarm) ---

  async sendHeartbeats(): Promise<boolean> {
    const deviceSockets = this.getWebSockets();
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
    return deviceSockets.length > 0;
  }

  // --- HTTP endpoints ---

  async handleConnect(request: Request): Promise<Response> {
    const upgradeHeader = request.headers.get("Upgrade");
    if (!upgradeHeader || upgradeHeader.toLowerCase() !== "websocket") {
      return new Response("Expected WebSocket", { status: 426 });
    }

    // Persist userId from the auth middleware
    const uid = request.headers.get("x-user-id");
    if (uid) {
      await this.ctx.storage.put("userId", uid);
    }
    const sid = request.headers.get("x-session-id");
    if (sid) {
      await this.ctx.storage.put("sessionUuid", sid);
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

  async handleDispatch(
    request: Request,
    getUserId: () => Promise<string>
  ): Promise<Response> {
    // Ensure userId is available
    const uid = request.headers.get("x-user-id");
    if (uid) {
      await this.ctx.storage.put("userId", uid);
    }

    const body = (await request.json()) as {
      task: string;
      timeoutMs?: number;
    };
    const { task, timeoutMs = 5 * 60 * 1000 } = body;

    const deviceWs = this.getWs();
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

  // --- SSE stream helpers ---

  private writeStreamEvent(data: Record<string, unknown>): void {
    if (!this.streamWriter) return;
    const line = `data: ${JSON.stringify(data)}\n\n`;
    this.streamWriter.write(this.streamEncoder.encode(line));
  }

  private closeStream(errorMessage?: string): void {
    if (!this.streamWriter) return;
    try {
      if (errorMessage) {
        const id = `error-${Date.now()}`;
        this.writeStreamEvent({ type: "text-start", id });
        this.writeStreamEvent({ type: "text-delta", id, delta: errorMessage });
        this.writeStreamEvent({ type: "text-end", id });
        this.writeStreamEvent({ type: "finish", messageMetadata: {} });
      }
      this.streamWriter.close();
    } catch {
      /* stream may already be closed */
    }
    this.streamWriter = null;
    this.emittedToolCallIds.clear();
    this.initialToolMessageCount = 0;
  }

  // --- Core relay ---

  relay(messages: UIMessage[]): Response {
    const lastMsg = messages[messages.length - 1];
    const userText = lastMsg.parts
      .filter((p: any) => p.type === "text")
      .map((p: any) => (p as { text: string }).text)
      .join("\n");

    const deviceWs = this.getWs();
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

    // Create SSE stream
    const { readable, writable } = new TransformStream<Uint8Array>();
    this.streamWriter = writable.getWriter();
    this.emittedToolCallIds.clear();
    this.initialToolMessageCount = 0;

    // Write stream start
    this.writeStreamEvent({ type: "start" });

    // Set up result/timeout handling — stream is filled asynchronously
    const resultPromise = new Promise<{ result: string; success: boolean }>(
      (resolve, reject) => {
        const timer = setTimeout(
          () => {
            this.pendingTasks.delete(taskId);
            this.closeStream("Task timed out");
            reject(new Error("Task timed out"));
          },
          5 * 60 * 1000
        );
        this.pendingTasks.set(taskId, { resolve, reject, timer });
      }
    );

    // When device sends result, write final text and close
    resultPromise.then(
      ({ result, success }) => {
        const text = result || (success ? "Done." : "Task failed.");
        const id = `result-${Date.now()}`;
        this.writeStreamEvent({ type: "text-start", id });
        this.writeStreamEvent({ type: "text-delta", id, delta: text });
        this.writeStreamEvent({ type: "text-end", id });
        this.writeStreamEvent({ type: "finish", messageMetadata: {} });
        this.closeStream();
      },
      (_err) => {
        // Timeout and disconnect already handled by closeStream
      }
    );

    return new Response(readable, {
      headers: { "Content-Type": "text/event-stream" }
    });
  }

  // --- Device WebSocket message handling ---

  async handleMessage(
    ws: WebSocket,
    message: string | ArrayBuffer,
    callbacks: {
      getUserId: () => Promise<string>;
      getSessionUuid: () => Promise<string | null>;
      getLlmProxyConfig: () => Promise<LlmProxyConfig>;
      ensureMounted: () => Promise<void>;
      env: Env;
    }
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
          const userId = await callbacks.getUserId();
          const sessionId = await callbacks.getSessionUuid();
          if (userId && sessionId) {
            await callbacks.env.DB.prepare(
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
        const config = await callbacks.getLlmProxyConfig();
        await callbacks.ensureMounted();
        const userId = await callbacks.getUserId();
        const sessionId = await callbacks.getSessionUuid();
        await this.handleLlmRequest(
          ws,
          data,
          config,
          callbacks.env.DB,
          userId,
          sessionId
        );
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

  // --- Device WebSocket close/error ---

  handleClose(): void {
    // Close device SSE stream before rejecting tasks
    this.closeStream("Device disconnected");

    // Reject all pending tasks
    for (const [taskId, pending] of this.pendingTasks) {
      clearTimeout(pending.timer);
      this.pendingTasks.delete(taskId);
      pending.reject(new Error("Device disconnected"));
    }
  }

  handleError(): void {
    // webSocketClose will fire after this — cleanup happens there
  }

  // --- Internal: LLM proxy + usage ---

  private async handleLlmRequest(
    ws: WebSocket,
    data: { requestId: string; body: Record<string, unknown> },
    config: LlmProxyConfig,
    db: D1Database,
    userId: string,
    sessionId: string | null
  ): Promise<void> {
    const { requestId, body } = data;

    try {
      // Call upstream LLM
      const upstreamRes = await fetch(`${config.baseUrl}/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${config.apiKey}`
        },
        body: JSON.stringify({
          ...body,
          model: (body.model as string) || config.model
        })
      });
      const responseBody = (await upstreamRes.json()) as Record<
        string,
        unknown
      >;

      // Emit tool results (from request body) to SSE stream for Web UI visibility
      if (this.streamWriter) {
        const messages = body.messages as
          | Array<Record<string, unknown>>
          | undefined;
        if (messages) {
          // On first call, record how many tool messages exist so we only emit new ones
          if (this.initialToolMessageCount === 0) {
            this.initialToolMessageCount = messages.filter(
              (m) => m.role === "tool"
            ).length;
          }
          // Emit only newly added tool results (skip initial ones)
          let toolMsgSeen = 0;
          for (const msg of messages) {
            if (msg.role === "tool") {
              toolMsgSeen++;
              if (toolMsgSeen <= this.initialToolMessageCount) continue;
              const toolCallId = msg.tool_call_id as string;
              if (
                toolCallId &&
                !this.emittedToolCallIds.has(`result-${toolCallId}`)
              ) {
                this.emittedToolCallIds.add(`result-${toolCallId}`);
                this.writeStreamEvent({
                  type: "tool-output-available",
                  toolCallId,
                  output: msg.content
                });
              }
            }
          }
          // Update initial count so next round's new results are detected
          this.initialToolMessageCount = messages.filter(
            (m) => m.role === "tool"
          ).length;
        }

        // Emit tool calls (from LLM response) to SSE stream
        const choices = (responseBody as any).choices as
          | Array<Record<string, any>>
          | undefined;
        const toolCalls = choices?.[0]?.message?.tool_calls as
          | Array<{
              id: string;
              function: { name: string; arguments: string };
            }>
          | undefined;
        if (toolCalls) {
          for (const tc of toolCalls) {
            if (!this.emittedToolCallIds.has(tc.id)) {
              this.emittedToolCallIds.add(tc.id);
              let parsedArgs: unknown;
              try {
                parsedArgs = JSON.parse(tc.function.arguments);
              } catch {
                parsedArgs = tc.function.arguments;
              }
              this.writeStreamEvent({
                type: "tool-input-start",
                toolCallId: tc.id,
                toolName: tc.function.name
              });
              this.writeStreamEvent({
                type: "tool-input-available",
                toolCallId: tc.id,
                toolName: tc.function.name,
                input: parsedArgs
              });
            }
          }
        }
      }

      // Write usage to usage_archive for quota tracking
      const usage = (responseBody as { usage?: Record<string, number> }).usage;
      if (usage) {
        archiveLlmRequestUsage(db, userId, sessionId, config.apiKeyType, usage);
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
