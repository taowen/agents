import { DurableObject } from "cloudflare:workers";
import * as Sentry from "@sentry/cloudflare";

const REQUEST_TIMEOUT_MS = 30_000;
const HEARTBEAT_INTERVAL_MS = 30_000;

interface PendingRequest {
  resolve: (value: Response) => void;
  timer: ReturnType<typeof setTimeout>;
}

/**
 * TunnelRelay Durable Object — one instance per tunnel name.
 *
 * - /tunnel/connect  — accepts the tunneld CLI WebSocket (tagged "tunnel")
 * - All other paths   — proxied to the tunneld client over WebSocket
 */
class TunnelRelayBase extends DurableObject<Env> {
  private pendingRequests = new Map<string, PendingRequest>();

  // ── WebSocket helpers ──

  private getTunnelWs(): WebSocket | null {
    const sockets = this.ctx.getWebSockets("tunnel");
    return sockets.length > 0 ? sockets[0] : null;
  }

  // ── HTTP entry point (called by the Worker fetch) ──

  override async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);

    if (url.pathname === "/tunnel/connect") {
      return this.handleConnect(request);
    }

    // Everything else: proxy through the tunnel
    return this.proxyRequest(request);
  }

  // ── Tunnel client WebSocket connect ──

  private async handleConnect(request: Request): Promise<Response> {
    const upgradeHeader = request.headers.get("Upgrade");
    if (!upgradeHeader || upgradeHeader.toLowerCase() !== "websocket") {
      return new Response("Expected WebSocket", { status: 426 });
    }

    // Close any existing tunnel connection (one connection per tunnel)
    const existing = this.getTunnelWs();
    if (existing) {
      try {
        existing.close(1000, "replaced");
      } catch {
        // already closed
      }
    }

    const pair = new WebSocketPair();
    this.ctx.acceptWebSocket(pair[1], ["tunnel"]);

    // Start heartbeat alarm
    const alarm = await this.ctx.storage.getAlarm();
    if (!alarm) {
      await this.ctx.storage.setAlarm(Date.now() + HEARTBEAT_INTERVAL_MS);
    }

    return new Response(null, { status: 101, webSocket: pair[0] });
  }

  // ── Proxy an HTTP request over the tunnel WebSocket ──

  private async proxyRequest(request: Request): Promise<Response> {
    const ws = this.getTunnelWs();
    if (!ws) {
      return new Response("Tunnel offline", { status: 502 });
    }

    const id = crypto.randomUUID();
    const url = new URL(request.url);

    // Read body as base64
    let bodyB64: string | null = null;
    if (request.body) {
      const buf = await request.arrayBuffer();
      if (buf.byteLength > 0) {
        const bytes = new Uint8Array(buf);
        let binary = "";
        for (let i = 0; i < bytes.length; i += 8192) {
          binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
        }
        bodyB64 = btoa(binary);
      }
    }

    // Collect headers
    const headers: Record<string, string> = {};
    for (const [k, v] of request.headers.entries()) {
      // Skip hop-by-hop and CF internal headers
      if (
        k.startsWith("cf-") ||
        k === "host" ||
        k === "connection" ||
        k === "upgrade"
      ) {
        continue;
      }
      headers[k] = v;
    }

    try {
      ws.send(
        JSON.stringify({
          type: "request",
          id,
          method: request.method,
          url: url.pathname + url.search,
          headers,
          body: bodyB64
        })
      );
    } catch {
      return new Response("Tunnel connection broken", { status: 502 });
    }

    return new Promise<Response>((resolve) => {
      const timer = setTimeout(() => {
        this.pendingRequests.delete(id);
        resolve(new Response("Tunnel request timed out", { status: 504 }));
      }, REQUEST_TIMEOUT_MS);

      this.pendingRequests.set(id, { resolve, timer });
    });
  }

  // ── WebSocket message handling ──

  override async webSocketMessage(
    ws: WebSocket,
    message: string | ArrayBuffer
  ): Promise<void> {
    if (typeof message !== "string") return;

    try {
      const data = JSON.parse(message);

      if (data.type === "register") {
        // Acknowledge registration
        ws.send(JSON.stringify({ type: "registered", name: data.name }));
        return;
      }

      if (data.type === "response") {
        const pending = this.pendingRequests.get(data.id);
        if (!pending) return;

        clearTimeout(pending.timer);
        this.pendingRequests.delete(data.id);

        // Decode base64 body
        let body: ArrayBuffer | null = null;
        if (data.body) {
          const bin = atob(data.body);
          body = Uint8Array.from(bin, (c) => c.charCodeAt(0)).buffer;
        }

        const respHeaders = new Headers(data.headers || {});
        pending.resolve(
          new Response(body, { status: data.status, headers: respHeaders })
        );
        return;
      }

      if (data.type === "pong") {
        // Heartbeat acknowledged — nothing to do
        return;
      }
    } catch {
      // Ignore malformed messages
    }
  }

  override async webSocketClose(
    ws: WebSocket,
    code: number,
    reason: string,
    wasClean: boolean
  ): Promise<void> {
    // Also handles the post-error case — webSocketError fires before this.
    // Reject all pending requests
    for (const [id, pending] of this.pendingRequests) {
      clearTimeout(pending.timer);
      this.pendingRequests.delete(id);
      pending.resolve(new Response("Tunnel disconnected", { status: 502 }));
    }
  }

  // ── Heartbeat alarm ──

  override async alarm(): Promise<void> {
    const tunnelSockets = this.ctx.getWebSockets("tunnel");
    for (const ws of tunnelSockets) {
      try {
        ws.send(JSON.stringify({ type: "ping" }));
      } catch {
        // Dead socket
      }
    }

    if (tunnelSockets.length > 0) {
      await this.ctx.storage.setAlarm(Date.now() + HEARTBEAT_INTERVAL_MS);
    }
  }
}

// ── Sentry-instrumented export ──

export const TunnelRelay = Sentry.instrumentDurableObjectWithSentry(
  (env: Env) => ({
    dsn: env.SENTRY_DSN,
    tracesSampleRate: 1.0
  }),
  TunnelRelayBase
);
