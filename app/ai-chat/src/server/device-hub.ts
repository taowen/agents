interface DeviceInfo {
  type: "device";
  deviceName: string;
  deviceId: string;
  connectedAt: number;
}

interface PendingExec {
  resolve: (value: { result: string; screenshots: string[] }) => void;
  reject: (reason: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

export function isDeviceSession(sessionUuid: string | null): boolean {
  return !!sessionUuid?.startsWith("device-");
}

export function getDeviceName(sessionUuid: string): string {
  return sessionUuid.slice("device-".length);
}

export class DeviceHub {
  private pendingExecs = new Map<string, PendingExec>();

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

  // --- Execute JS on device ---

  /**
   * Send JavaScript code to the device for execution and wait for the result.
   * Uses exec_js/exec_result message pair over the device WebSocket.
   */
  async execOnDevice(
    code: string,
    timeoutMs = 60_000
  ): Promise<{ result: string; screenshots: string[] }> {
    const deviceWs = this.getWs();
    if (!deviceWs) {
      throw new Error("Device not connected");
    }

    const execId = crypto.randomUUID();

    try {
      deviceWs.send(JSON.stringify({ type: "exec_js", execId, code }));
    } catch {
      throw new Error("Failed to send exec_js to device");
    }

    return new Promise<{ result: string; screenshots: string[] }>(
      (resolve, reject) => {
        const timer = setTimeout(() => {
          this.pendingExecs.delete(execId);
          reject(new Error("exec_js timed out"));
        }, timeoutMs);
        this.pendingExecs.set(execId, { resolve, reject, timer });
      }
    );
  }

  // --- Device WebSocket message handling ---

  async handleMessage(
    ws: WebSocket,
    message: string | ArrayBuffer,
    callbacks: {
      getUserId: () => Promise<string>;
      getSessionUuid: () => Promise<string | null>;
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

        // Store device system prompt and tools if provided
        if (data.systemPrompt) {
          await this.ctx.storage.put("deviceSystemPrompt", data.systemPrompt);
        }
        if (data.tools) {
          await this.ctx.storage.put("deviceTools", data.tools);
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

      if (data.type === "exec_result") {
        const pending = this.pendingExecs.get(data.execId);
        if (pending) {
          clearTimeout(pending.timer);
          this.pendingExecs.delete(data.execId);
          pending.resolve({
            result: data.result || "",
            screenshots: data.screenshots || []
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
    // Reject all pending execs
    for (const [execId, pending] of this.pendingExecs) {
      clearTimeout(pending.timer);
      this.pendingExecs.delete(execId);
      pending.reject(new Error("Device disconnected"));
    }
  }

  handleError(): void {
    // webSocketClose will fire after this — cleanup happens there
  }
}
