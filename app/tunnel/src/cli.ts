#!/usr/bin/env node

import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { createInterface } from "node:readline";
import WebSocket from "ws";

// ── Config ──

const DEFAULT_SERVER = "https://ai.connect-screen.com";
const TOKEN_DIR = join(homedir(), ".tunneld");
const TOKEN_FILE = join(TOKEN_DIR, "token");
const RECONNECT_DELAY_MS = 3_000;
const DEVICE_POLL_INTERVAL_MS = 3_000;

// ── CLI args ──

function usage(): never {
  console.error("Usage: tunneld <name> <port> [--server <url>]");
  console.error("");
  console.error(
    "  name     Tunnel subdomain (e.g. 'my-app' → my-app.cscreen.cc)"
  );
  console.error("  port     Local HTTP port to forward to");
  console.error("");
  console.error("Options:");
  console.error(
    "  --server <url>   Server base URL (default: https://ai.connect-screen.com)"
  );
  process.exit(1);
}

const args = process.argv.slice(2);
let tunnelName = "";
let localPort = 0;
let serverBase = DEFAULT_SERVER;

for (let i = 0; i < args.length; i++) {
  if (args[i] === "--server" && args[i + 1]) {
    serverBase = args[++i];
  } else if (!tunnelName) {
    tunnelName = args[i];
  } else if (!localPort) {
    localPort = parseInt(args[i], 10);
  }
}

if (!tunnelName || !localPort || isNaN(localPort)) {
  usage();
}

// Keep in sync with host regex in app/ai-chat/src/server/index.ts
const TUNNEL_NAME_RE = /^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$/;
if (!TUNNEL_NAME_RE.test(tunnelName)) {
  console.error(
    `Invalid tunnel name: "${tunnelName}". Must match DNS subdomain rules (lowercase alphanumeric + hyphens).`
  );
  process.exit(1);
}

// ── Token management ──

function loadToken(): string | null {
  try {
    return readFileSync(TOKEN_FILE, "utf-8").trim();
  } catch {
    return null;
  }
}

function saveToken(token: string): void {
  mkdirSync(TOKEN_DIR, { recursive: true });
  writeFileSync(TOKEN_FILE, token, { mode: 0o600 });
}

// ── Device auth flow ──

async function authenticate(): Promise<string> {
  const cached = loadToken();
  if (cached) {
    // Verify token is still valid
    const res = await fetch(`${serverBase}/api/usage`, {
      headers: { Authorization: `Bearer ${cached}` }
    });
    if (res.ok) {
      return cached;
    }
    console.log("Cached token expired, re-authenticating...");
  }

  // Start device auth flow
  const startRes = await fetch(`${serverBase}/auth/device/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({})
  });

  if (!startRes.ok) {
    throw new Error(
      `Failed to start device auth: ${startRes.status} ${await startRes.text()}`
    );
  }

  const { code } = (await startRes.json()) as { code: string };

  console.log("");
  console.log("  ┌─────────────────────────────────────────┐");
  console.log(`  │  DEVICE CODE: ${code}                       │`);
  console.log("  │                                         │");
  console.log(`  │  Approve at: ${serverBase}/device        `);
  console.log("  └─────────────────────────────────────────┘");
  console.log("");
  console.log("Waiting for approval...");

  // Poll for approval
  while (true) {
    await new Promise((r) => setTimeout(r, DEVICE_POLL_INTERVAL_MS));

    const checkRes = await fetch(
      `${serverBase}/auth/device/check?code=${encodeURIComponent(code)}`
    );

    if (!checkRes.ok) {
      const body = await checkRes.text();
      if (checkRes.status === 410) {
        throw new Error("Device code expired. Please try again.");
      }
      continue;
    }

    const data = (await checkRes.json()) as { status: string; token?: string };

    if (data.status === "approved" && data.token) {
      saveToken(data.token);
      console.log("Authenticated successfully!");
      return data.token;
    }

    if (data.status === "expired") {
      throw new Error("Device code expired. Please try again.");
    }

    // Still pending — keep polling
  }
}

// ── Request handling ──

async function handleRequest(
  msg: {
    id: string;
    method: string;
    url: string;
    headers: Record<string, string>;
    body: string | null;
  },
  ws: WebSocket
): Promise<void> {
  try {
    // Decode base64 body
    let body: Buffer | undefined;
    if (msg.body) {
      body = Buffer.from(msg.body, "base64");
    }

    const localUrl = `http://localhost:${localPort}${msg.url}`;

    const res = await fetch(localUrl, {
      method: msg.method,
      headers: msg.headers,
      body: body as BodyInit | undefined,
      duplex: body ? "half" : undefined
    } as RequestInit);

    // Read response body as base64
    const resBuf = await res.arrayBuffer();
    const resB64 =
      resBuf.byteLength > 0 ? Buffer.from(resBuf).toString("base64") : null;

    // Collect response headers
    const resHeaders: Record<string, string> = {};
    for (const [k, v] of res.headers.entries()) {
      resHeaders[k] = v;
    }

    ws.send(
      JSON.stringify({
        type: "response",
        id: msg.id,
        status: res.status,
        headers: resHeaders,
        body: resB64
      })
    );

    console.log(`${msg.method} ${msg.url} → ${res.status}`);
  } catch (err: any) {
    // Send 502 back to the relay
    ws.send(
      JSON.stringify({
        type: "response",
        id: msg.id,
        status: 502,
        headers: { "content-type": "text/plain" },
        body: Buffer.from(`Local server error: ${err.message}`).toString(
          "base64"
        )
      })
    );

    console.error(`${msg.method} ${msg.url} → 502 (${err.message})`);
  }
}

// ── WebSocket connection ──

function connect(token: string): void {
  // Build WebSocket URL
  const serverUrl = new URL(serverBase);
  const wsProtocol = serverUrl.protocol === "https:" ? "wss:" : "ws:";
  const tunnelHost = `${tunnelName}.cscreen.cc`;
  const wsUrl = `${wsProtocol}//${tunnelHost}/tunnel/connect?token=${encodeURIComponent(token)}`;

  console.log(`Connecting to ${tunnelHost}...`);

  const ws = new WebSocket(wsUrl);

  ws.on("open", () => {
    // Register tunnel name
    ws.send(JSON.stringify({ type: "register", name: tunnelName }));
  });

  ws.on("message", (data: Buffer) => {
    try {
      const msg = JSON.parse(data.toString());

      if (msg.type === "registered") {
        console.log(`Tunnel active: https://${tunnelName}.cscreen.cc`);
        console.log(`Forwarding to http://localhost:${localPort}`);
        console.log("");
        return;
      }

      if (msg.type === "request") {
        handleRequest(msg, ws);
        return;
      }

      if (msg.type === "ping") {
        ws.send(JSON.stringify({ type: "pong" }));
        return;
      }
    } catch {
      // Ignore malformed messages
    }
  });

  ws.on("close", (code: number, reason: Buffer) => {
    console.log(
      `Disconnected (code=${code}). Reconnecting in ${RECONNECT_DELAY_MS / 1000}s...`
    );
    setTimeout(() => connect(token), RECONNECT_DELAY_MS);
  });

  ws.on("error", (err: Error) => {
    console.error(`WebSocket error: ${err.message}`);
    // close event will fire after this — reconnect happens there
  });

  // Graceful shutdown
  const shutdown = () => {
    console.log("\nShutting down...");
    ws.close(1000, "client shutdown");
    process.exit(0);
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

// ── Main ──

async function main(): Promise<void> {
  console.log(`tunneld v0.1.0`);
  console.log(`Tunnel: ${tunnelName} → localhost:${localPort}`);
  console.log("");

  const token = await authenticate();
  connect(token);
}

main().catch((err) => {
  console.error(`Fatal: ${err.message}`);
  process.exit(1);
});
