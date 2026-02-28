import type { DebugRingBuffer } from "./llm-debug-buffer";

interface DeviceInfo {
  type: "device";
  deviceName: string;
  deviceId: string;
  connectedAt: number;
}

export interface ExecLogEntry {
  fn: string;
  args: string;
  result: string;
}

export interface ExecResult {
  result: string;
  screenshots: string[];
  executionLog?: ExecLogEntry[];
}

interface PendingExec {
  resolve: (value: ExecResult) => void;
  reject: (reason: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

export function isDeviceSession(sessionUuid: string | null): boolean {
  return !!sessionUuid?.startsWith("device-");
}

export function getDeviceName(sessionUuid: string): string {
  return sessionUuid.slice("device-".length);
}

/** Build the isolated DO name used for ChatAgent stubs. */
export function chatAgentIsolatedName(
  userId: string,
  sessionId: string
): string {
  return encodeURIComponent(`${userId}:${sessionId}`);
}

/** Get a ChatAgent DO stub for the given user + session. */
export function getChatAgentStub(
  ns: DurableObjectNamespace,
  userId: string,
  sessionId: string
): DurableObjectStub {
  return ns.get(ns.idFromName(chatAgentIsolatedName(userId, sessionId)));
}

export interface DeviceRecord {
  name: string;
  sessionId: string;
  title: string;
  online: boolean;
  stub: DurableObjectStub;
}

/** Query D1 for device sessions and check DO liveness for each. */
export async function findDevices(
  env: Env,
  userId: string
): Promise<DeviceRecord[]> {
  const rows = await env.DB.prepare(
    "SELECT id, title FROM sessions WHERE user_id = ? AND id LIKE 'device-%'"
  )
    .bind(userId)
    .all<{ id: string; title: string }>();

  if (rows.results.length === 0) return [];

  const checks = await Promise.allSettled(
    rows.results.map(async (s) => {
      const stub = getChatAgentStub(env.ChatAgent, userId, s.id);
      const res = await stub.fetch(new Request("http://agent/status"));
      const body = (await res.json()) as { online: boolean };
      return {
        name: s.id.replace("device-", ""),
        sessionId: s.id,
        title: s.title || "",
        online: body.online,
        stub
      };
    })
  );

  return checks
    .filter(
      (r): r is PromiseFulfilledResult<DeviceRecord> => r.status === "fulfilled"
    )
    .map((r) => r.value);
}

export class DeviceHub {
  private pendingExecs = new Map<string, PendingExec>();

  constructor(
    private ctx: DurableObjectState,
    private debugBuffer?: DebugRingBuffer
  ) {}

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

    this.debugBuffer?.push({
      type: "device_connection",
      timestamp: new Date().toISOString(),
      event: "connect"
    });

    return new Response(null, { status: 101, webSocket: pair[0] });
  }

  // --- Execute code on device ---

  /**
   * Send code to the device for execution and wait for the result.
   * Uses the device-reported exec type (defaults to "exec_js" for Android).
   */
  async execOnDevice(code: string, timeoutMs = 60_000): Promise<ExecResult> {
    const deviceWs = this.getWs();
    if (!deviceWs) {
      throw new Error("Device not connected");
    }

    const execType =
      (await this.ctx.storage.get<string>("execType")) || "exec_js";
    const execId = crypto.randomUUID();

    try {
      deviceWs.send(JSON.stringify({ type: execType, execId, code }));
    } catch {
      throw new Error(`Failed to send ${execType} to device`);
    }

    return new Promise<ExecResult>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pendingExecs.delete(execId);
        reject(new Error(`${execType} timed out`));
      }, timeoutMs);
      this.pendingExecs.set(execId, { resolve, reject, timer });
    });
  }

  // --- Device WebSocket message handling ---

  /**
   * Send a task_done message to the device after server-side agent completes.
   */
  sendTaskDone(result: string): void {
    const deviceWs = this.getWs();
    if (!deviceWs) return;
    try {
      deviceWs.send(JSON.stringify({ type: "task_done", result }));
    } catch {
      // Dead socket — ignore
    }
  }

  async handleMessage(
    ws: WebSocket,
    message: string | ArrayBuffer,
    callbacks: {
      getUserId: () => Promise<string>;
      getSessionUuid: () => Promise<string | null>;
      onUserTask?: (text: string) => void;
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

        this.debugBuffer?.push({
          type: "device_connection",
          timestamp: new Date().toISOString(),
          event: "ready",
          deviceName: info.deviceName,
          deviceId: info.deviceId
        });

        // Store device system prompt, tools, and exec type if provided
        if (data.systemPrompt) {
          await this.ctx.storage.put("deviceSystemPrompt", data.systemPrompt);
        }
        if (data.tools) {
          await this.ctx.storage.put("deviceTools", data.tools);
        }
        if (data.execType) {
          await this.ctx.storage.put("execType", data.execType);
        }

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

      if (data.type === "user_task") {
        if (data.text && callbacks.onUserTask) {
          callbacks.onUserTask(data.text);
        }
        return;
      }

      if (data.type === "exec_result") {
        const pending = this.pendingExecs.get(data.execId);
        if (pending) {
          clearTimeout(pending.timer);
          this.pendingExecs.delete(data.execId);
          pending.resolve({
            result: data.result || "",
            screenshots: data.screenshots || [],
            executionLog: data.executionLog
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
    this.debugBuffer?.push({
      type: "device_connection",
      timestamp: new Date().toISOString(),
      event: "disconnect"
    });
    // Reject all pending execs
    for (const [execId, pending] of this.pendingExecs) {
      clearTimeout(pending.timer);
      this.pendingExecs.delete(execId);
      pending.reject(new Error("Device disconnected"));
    }
  }

  handleError(): void {
    this.debugBuffer?.push({
      type: "device_connection",
      timestamp: new Date().toISOString(),
      event: "error"
    });
    // webSocketClose will fire after this — cleanup happens there
  }
}
