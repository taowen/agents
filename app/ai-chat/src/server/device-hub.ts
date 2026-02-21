/**
 * DeviceHub Durable Object — per-user hub for device connections and task dispatch.
 *
 * Uses Hibernatable WebSocket API: device info is stored as WebSocket attachments
 * so it survives DO hibernation. `this.ctx.getWebSockets()` enumerates live connections.
 *
 * Routes:
 *   /connect   — WebSocket upgrade (device connects here)
 *   /devices   — GET list of online devices
 *   /dispatch  — POST { task, deviceId? } to dispatch a task and wait for result
 *
 * Device protocol (over WebSocket):
 *   Device→Hub: { type: "ready", deviceName, deviceId }
 *   Device→Hub: { type: "result", taskId, result, success }
 *   Device→Hub: { type: "pong" }
 *   Hub→Device: { type: "task", taskId, description }
 *   Hub→Device: { type: "ping" }
 */

import { DurableObject } from "cloudflare:workers";

interface DeviceInfo {
  deviceId: string;
  deviceName: string;
  connectedAt: number;
}

interface PendingTask {
  resolve: (value: { result: string; success: boolean }) => void;
  reject: (reason: Error) => void;
  timer: ReturnType<typeof setTimeout>;
  deviceId: string;
}

export class DeviceHub extends DurableObject<Env> {
  private pendingTasks = new Map<string, PendingTask>();

  /** Get all live WebSockets that have sent a "ready" message (have attachment). */
  private getReadyDevices(): Array<{ ws: WebSocket; info: DeviceInfo }> {
    const results: Array<{ ws: WebSocket; info: DeviceInfo }> = [];
    for (const ws of this.ctx.getWebSockets()) {
      const info = ws.deserializeAttachment() as DeviceInfo | null;
      if (info?.deviceId) {
        results.push({ ws, info });
      }
    }
    return results;
  }

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname.endsWith("/connect")) {
      return this.handleConnect(request);
    }
    if (url.pathname.endsWith("/devices")) {
      return this.handleListDevices();
    }
    if (url.pathname.endsWith("/dispatch")) {
      return this.handleDispatch(request);
    }

    return new Response("Not found", { status: 404 });
  }

  private handleConnect(request: Request): Response {
    const upgradeHeader = request.headers.get("Upgrade");
    if (!upgradeHeader || upgradeHeader.toLowerCase() !== "websocket") {
      return new Response("Expected WebSocket", { status: 426 });
    }

    const pair = new WebSocketPair();
    this.ctx.acceptWebSocket(pair[1]);

    // Start heartbeat alarm if not already running
    this.ctx.storage.getAlarm().then((alarm) => {
      if (!alarm) {
        this.ctx.storage.setAlarm(Date.now() + 30_000);
      }
    });

    return new Response(null, { status: 101, webSocket: pair[0] });
  }

  private handleListDevices(): Response {
    const list = this.getReadyDevices().map((d) => d.info);
    return Response.json(list);
  }

  private async handleDispatch(request: Request): Promise<Response> {
    const body = (await request.json()) as {
      task: string;
      deviceId?: string;
      timeoutMs?: number;
    };
    const { task, deviceId, timeoutMs = 5 * 60 * 1000 } = body;

    // Find target device
    const devices = this.getReadyDevices();
    let target: { ws: WebSocket; info: DeviceInfo } | undefined;
    if (deviceId) {
      target = devices.find((d) => d.info.deviceId === deviceId);
    } else {
      target = devices[0];
    }

    if (!target) {
      return Response.json(
        { error: "No device available", devices: devices.map((d) => d.info) },
        { status: 404 }
      );
    }

    const taskId = crypto.randomUUID();

    // Send task to device
    try {
      target.ws.send(
        JSON.stringify({ type: "task", taskId, description: task })
      );
    } catch (e) {
      return Response.json(
        { error: "Failed to send to device" },
        { status: 502 }
      );
    }

    // Wait for result with timeout
    const resultPromise = new Promise<{ result: string; success: boolean }>(
      (resolve, reject) => {
        const timer = setTimeout(() => {
          this.pendingTasks.delete(taskId);
          reject(new Error("Task timed out"));
        }, timeoutMs);

        this.pendingTasks.set(taskId, {
          resolve,
          reject,
          timer,
          deviceId: target.info.deviceId
        });
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

  async webSocketMessage(
    ws: WebSocket,
    message: string | ArrayBuffer
  ): Promise<void> {
    if (typeof message !== "string") return;

    try {
      const data = JSON.parse(message);

      if (data.type === "ready") {
        // Persist device info as WebSocket attachment (survives hibernation)
        const info: DeviceInfo = {
          deviceId: data.deviceId || "unknown",
          deviceName: data.deviceName || "Unknown Device",
          connectedAt: Date.now()
        };
        ws.serializeAttachment(info);
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

      // pong — no action needed, just confirms device is alive
    } catch {
      // ignore malformed messages
    }
  }

  async webSocketClose(ws: WebSocket): Promise<void> {
    const info = ws.deserializeAttachment() as DeviceInfo | null;
    if (!info?.deviceId) return;

    // Reject all pending tasks dispatched to this device
    for (const [taskId, pending] of this.pendingTasks) {
      if (pending.deviceId === info.deviceId) {
        clearTimeout(pending.timer);
        this.pendingTasks.delete(taskId);
        pending.reject(new Error("Device disconnected"));
      }
    }
  }

  async webSocketError(ws: WebSocket): Promise<void> {
    // webSocketClose will fire after this — cleanup happens there
  }

  async alarm(): Promise<void> {
    // Heartbeat: ping all connected devices
    const sockets = this.ctx.getWebSockets();
    for (const ws of sockets) {
      try {
        ws.send(JSON.stringify({ type: "ping" }));
      } catch {
        // Dead socket — runtime will clean it up
      }
    }

    // Reschedule if any sockets are still connected
    if (sockets.length > 0) {
      this.ctx.storage.setAlarm(Date.now() + 30_000);
    }
  }
}
