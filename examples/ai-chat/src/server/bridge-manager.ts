import { Agent } from "agents";
import type { Connection, WSMessage } from "agents";

/**
 * Protocol message types for the bridge between ChatAgent and remote desktop agents.
 * Only text is exchanged — images are processed locally by the remote agent.
 */

interface BridgeRegisterMessage {
  type: "cf_agent_bridge_register";
  deviceName: string;
}

interface BridgeResponseMessage {
  type: "cf_agent_bridge_response";
  messageId: string;
  content: string;
}

interface BridgeSubscribeMessage {
  type: "cf_agent_bridge_subscribe";
}

interface BridgeLogMessage {
  type: "cf_agent_bridge_log";
  message: string;
}

interface BridgeMessageToClient {
  type: "cf_agent_bridge_message";
  messageId: string;
  content: string;
}

type BridgeClientMessage =
  | BridgeRegisterMessage
  | BridgeResponseMessage
  | BridgeSubscribeMessage
  | BridgeLogMessage;

interface PendingMessage {
  resolve: (response: string) => void;
  timer: ReturnType<typeof setTimeout>;
}

const MESSAGE_TIMEOUT_MS = 120_000; // 120s for complex desktop tasks

/**
 * BridgeManager DO — per-user hub for remote desktop agents.
 *
 * Device registrations are persisted to SQL so they survive DO hibernation.
 * Active WebSocket connections are maintained by the runtime through hibernation.
 *
 * - Electron clients connect via WebSocket (register device, receive messages, send responses)
 * - Browser clients subscribe as viewers (receive device list updates + activity logs)
 * - ChatAgent calls via HTTP (GET /devices, POST /message)
 */
export class BridgeManager extends Agent<Env> {
  private pendingMessages = new Map<string, PendingMessage>(); // messageId → pending

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    // Create tables if they don't exist
    this.sql`
      CREATE TABLE IF NOT EXISTS bridge_devices (
        connection_id TEXT PRIMARY KEY,
        device_name TEXT NOT NULL
      )
    `;
    this.sql`
      CREATE TABLE IF NOT EXISTS bridge_viewers (
        connection_id TEXT PRIMARY KEY
      )
    `;
  }

  /**
   * Get all connected devices by cross-referencing SQL registrations
   * with currently active WebSocket connections.
   */
  private getActiveDevices(): { deviceName: string; connection: Connection }[] {
    // Get all active connection IDs
    const activeConnIds = new Set<string>();
    for (const conn of this.getConnections()) {
      activeConnIds.add(conn.id);
    }

    // Query registered devices from SQL
    const rows = this.sql<{ connection_id: string; device_name: string }>`
      SELECT connection_id, device_name FROM bridge_devices
    `;

    const devices: { deviceName: string; connection: Connection }[] = [];
    const staleIds: string[] = [];

    for (const row of rows) {
      if (activeConnIds.has(row.connection_id)) {
        const conn = this.getConnection(row.connection_id);
        if (conn) {
          devices.push({ deviceName: row.device_name, connection: conn });
        }
      } else {
        staleIds.push(row.connection_id);
      }
    }

    // Clean up stale registrations (disconnected during hibernation)
    for (const id of staleIds) {
      this.sql`DELETE FROM bridge_devices WHERE connection_id = ${id}`;
    }

    return devices;
  }

  /**
   * Get all active viewer connections by cross-referencing SQL with live connections.
   */
  private getActiveViewers(): Connection[] {
    const activeConnIds = new Set<string>();
    for (const conn of this.getConnections()) {
      activeConnIds.add(conn.id);
    }

    const rows = this.sql<{ connection_id: string }>`
      SELECT connection_id FROM bridge_viewers
    `;

    const viewers: Connection[] = [];
    const staleIds: string[] = [];

    for (const row of rows) {
      if (activeConnIds.has(row.connection_id)) {
        const conn = this.getConnection(row.connection_id);
        if (conn) {
          viewers.push(conn);
        }
      } else {
        staleIds.push(row.connection_id);
      }
    }

    for (const id of staleIds) {
      this.sql`DELETE FROM bridge_viewers WHERE connection_id = ${id}`;
    }

    return viewers;
  }

  /**
   * Broadcast a message to all active viewers.
   */
  private broadcastToViewers(msg: Record<string, unknown>): void {
    const payload = JSON.stringify(msg);
    for (const viewer of this.getActiveViewers()) {
      viewer.send(payload);
    }
  }

  /**
   * Build and broadcast the current device list to all viewers.
   */
  private broadcastDeviceList(): void {
    const devices = this.getActiveDevices();
    this.broadcastToViewers({
      type: "cf_agent_bridge_devices",
      devices: devices.map((d) => ({
        deviceName: d.deviceName,
        status: "connected"
      }))
    });
  }

  onMessage(connection: Connection, message: WSMessage): void {
    if (typeof message !== "string") return;

    let parsed: BridgeClientMessage;
    try {
      parsed = JSON.parse(message);
    } catch {
      return;
    }

    switch (parsed.type) {
      case "cf_agent_bridge_register": {
        // Persist to SQL so it survives hibernation
        this.sql`
          INSERT OR REPLACE INTO bridge_devices (connection_id, device_name)
          VALUES (${connection.id}, ${parsed.deviceName})
        `;
        connection.send(
          JSON.stringify({
            type: "cf_agent_bridge_registered",
            deviceName: parsed.deviceName
          })
        );
        // Notify all viewers about updated device list
        this.broadcastDeviceList();
        break;
      }
      case "cf_agent_bridge_subscribe": {
        // Register as viewer
        this.sql`
          INSERT OR REPLACE INTO bridge_viewers (connection_id)
          VALUES (${connection.id})
        `;
        // Immediately send current device list
        const devices = this.getActiveDevices();
        connection.send(
          JSON.stringify({
            type: "cf_agent_bridge_devices",
            devices: devices.map((d) => ({
              deviceName: d.deviceName,
              status: "connected"
            }))
          })
        );
        break;
      }
      case "cf_agent_bridge_log": {
        // Look up the device name for this connection
        const rows = this.sql<{ device_name: string }>`
          SELECT device_name FROM bridge_devices WHERE connection_id = ${connection.id}
        `;
        const deviceName = rows[0]?.device_name ?? "unknown";
        // Relay to all viewers
        this.broadcastToViewers({
          type: "cf_agent_bridge_device_log",
          deviceName,
          time: new Date().toISOString(),
          message: parsed.message
        });
        break;
      }
      case "cf_agent_bridge_response": {
        const pending = this.pendingMessages.get(parsed.messageId);
        if (pending) {
          clearTimeout(pending.timer);
          this.pendingMessages.delete(parsed.messageId);
          pending.resolve(parsed.content);
        }
        // Broadcast log to viewers
        this.broadcastToViewers({
          type: "cf_agent_bridge_device_log",
          deviceName: "system",
          time: new Date().toISOString(),
          message: `Task response received (${parsed.content.length} chars)`
        });
        break;
      }
    }
  }

  onClose(connection: Connection): void {
    // Look up device name before deleting
    const rows = this.sql<{ device_name: string }>`
      SELECT device_name FROM bridge_devices WHERE connection_id = ${connection.id}
    `;
    const deviceName = rows[0]?.device_name;

    // Remove from SQL (both device and viewer tables)
    this.sql`DELETE FROM bridge_devices WHERE connection_id = ${connection.id}`;
    this.sql`DELETE FROM bridge_viewers WHERE connection_id = ${connection.id}`;

    // Resolve any pending messages for this device with an error
    if (deviceName) {
      for (const [messageId, pending] of this.pendingMessages) {
        clearTimeout(pending.timer);
        this.pendingMessages.delete(messageId);
        pending.resolve(
          `[Error] Device "${deviceName}" disconnected before responding.`
        );
      }
      // Notify viewers about updated device list
      this.broadcastDeviceList();
    }
  }

  async onRequest(request: Request): Promise<Response> {
    const url = new URL(request.url);

    // GET /devices — list connected devices
    if (url.pathname.endsWith("/devices") && request.method === "GET") {
      const devices = this.getActiveDevices();
      return Response.json(devices.map((d) => ({ deviceName: d.deviceName })));
    }

    // POST /message — send message to a device and wait for response
    if (url.pathname.endsWith("/message") && request.method === "POST") {
      const body = (await request.json()) as {
        deviceName: string;
        content: string;
      };

      // Find the device
      const devices = this.getActiveDevices();
      const device = devices.find((d) => d.deviceName === body.deviceName);
      if (!device) {
        return Response.json(
          { error: `Device "${body.deviceName}" is not connected.` },
          { status: 404 }
        );
      }

      const messageId = crypto.randomUUID();

      // Send message to the device
      const outgoing: BridgeMessageToClient = {
        type: "cf_agent_bridge_message",
        messageId,
        content: body.content
      };
      device.connection.send(JSON.stringify(outgoing));

      // Broadcast log to viewers
      this.broadcastToViewers({
        type: "cf_agent_bridge_device_log",
        deviceName: body.deviceName,
        time: new Date().toISOString(),
        message: `Received task: ${body.content.slice(0, 100)}`
      });

      // Wait for response (long-poll)
      const response = await new Promise<string>((resolve) => {
        const timer = setTimeout(() => {
          this.pendingMessages.delete(messageId);
          resolve(
            `[Error] Device "${body.deviceName}" did not respond within ${MESSAGE_TIMEOUT_MS / 1000}s.`
          );
        }, MESSAGE_TIMEOUT_MS);

        this.pendingMessages.set(messageId, { resolve, timer });
      });

      return Response.json({ response });
    }

    // Fall through to default Agent handling (WebSocket upgrade etc.)
    return super.onRequest(request);
  }
}
