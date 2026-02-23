import { createExecutionContext, env } from "cloudflare:test";
import { expect } from "vitest";
import worker from "./worker";
import type { Env } from "./worker";

// Re-export MessageType from the ai-chat package
export { MessageType } from "@cloudflare/ai-chat/types";

// D1 Schema SQL (from app/ai-chat/migrations/0001_init.sql)
const SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  email TEXT NOT NULL,
  name TEXT,
  picture TEXT,
  builtin_quota_exceeded_at TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id),
  title TEXT DEFAULT 'New Chat',
  device_online INTEGER DEFAULT 0,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at);

CREATE TABLE IF NOT EXISTS files (
  user_id TEXT NOT NULL,
  path TEXT NOT NULL,
  parent_path TEXT NOT NULL,
  name TEXT NOT NULL,
  content BLOB,
  is_directory INTEGER DEFAULT 0,
  mode INTEGER DEFAULT 33188,
  size INTEGER DEFAULT 0,
  mtime REAL DEFAULT (unixepoch('now')),
  PRIMARY KEY (user_id, path)
);
CREATE INDEX IF NOT EXISTS idx_files_parent ON files(user_id, parent_path);

CREATE TABLE IF NOT EXISTS usage_archive (
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  hour TEXT NOT NULL,
  api_key_type TEXT NOT NULL DEFAULT 'unknown',
  request_count INTEGER DEFAULT 0,
  input_tokens INTEGER DEFAULT 0,
  cache_read_tokens INTEGER DEFAULT 0,
  cache_write_tokens INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0,
  PRIMARY KEY (user_id, session_id, hour, api_key_type)
);
CREATE INDEX IF NOT EXISTS idx_usage_archive_user_hour ON usage_archive(user_id, hour);
`;

/**
 * Apply the D1 schema from the ai-chat migrations to the test database.
 */
export async function applyD1Schema(db: D1Database): Promise<void> {
  const statements = SCHEMA_SQL.split(";")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  // Use prepare().run() instead of exec() to avoid miniflare D1 metadata issues
  for (const stmt of statements) {
    await db.prepare(stmt).run();
  }
}

/**
 * Create a test user in D1 and return the userId.
 */
export async function createTestUser(
  db: D1Database,
  overrides?: { id?: string; email?: string; name?: string }
): Promise<string> {
  const userId =
    overrides?.id ?? `test-user-${crypto.randomUUID().slice(0, 8)}`;
  const email = overrides?.email ?? `${userId}@test.example`;
  const name = overrides?.name ?? "Test User";
  await db
    .prepare("INSERT INTO users (id, email, name) VALUES (?, ?, ?)")
    .bind(userId, email, name)
    .run();
  return userId;
}

/**
 * Extract the server's messageId from the broadcast start event.
 */
export function extractStartMessageId(received: unknown[]): string | undefined {
  for (const msg of received) {
    const m = msg as { type: string; body?: string };
    if (m.type !== "cf_agent_use_chat_response" || !m.body) continue;
    try {
      const parsed = JSON.parse(m.body);
      if (parsed.type === "start" && parsed.messageId) {
        return parsed.messageId;
      }
    } catch {}
  }
  return undefined;
}

/**
 * Make an authenticated HTTP request through the test worker.
 * Sets x-test-user-id header to simulate authentication.
 */
export async function apiRequest(
  method: string,
  path: string,
  userId: string,
  body?: unknown
): Promise<Response> {
  const ctx = createExecutionContext();
  const init: RequestInit = {
    method,
    headers: {
      "x-test-user-id": userId,
      "Content-Type": "application/json"
    }
  };
  if (body !== undefined) {
    init.body = JSON.stringify(body);
  }
  const req = new Request(`http://example.com${path}`, init);
  return worker.fetch(req, env as unknown as Env, ctx);
}

/**
 * Make an unauthenticated HTTP request through the test worker.
 */
export async function publicRequest(
  method: string,
  path: string,
  body?: unknown
): Promise<Response> {
  const ctx = createExecutionContext();
  const init: RequestInit = {
    method,
    headers: { "Content-Type": "application/json" }
  };
  if (body !== undefined) {
    init.body = JSON.stringify(body);
  }
  const req = new Request(`http://example.com${path}`, init);
  return worker.fetch(req, env as unknown as Env, ctx);
}

/**
 * Connect to the chat agent via WebSocket and return the socket + execution context.
 */
export async function connectChatWS(
  path: string
): Promise<{ ws: WebSocket; ctx: ExecutionContext }> {
  const ctx = createExecutionContext();
  const req = new Request(`http://example.com${path}`, {
    headers: { Upgrade: "websocket" }
  });
  const res = await worker.fetch(req, env, ctx);
  expect(res.status).toBe(101);
  const ws = res.webSocket as WebSocket;
  expect(ws).toBeDefined();
  ws.accept();
  return { ws, ctx };
}

/**
 * Send a CF_AGENT_USE_CHAT_REQUEST and wait for the streaming response to complete.
 * Returns all WebSocket messages received during the stream.
 */
export async function sendChatRequest(
  ws: WebSocket,
  requestId: string,
  messages: unknown[],
  timeoutMs = 3000
): Promise<unknown[]> {
  const received: unknown[] = [];

  const donePromise = new Promise<boolean>((resolve) => {
    const timeout = setTimeout(() => resolve(false), timeoutMs);
    const listener = (e: MessageEvent) => {
      const data = JSON.parse(e.data as string);
      received.push(data);
      if (data.type === "cf_agent_use_chat_response" && data.done) {
        clearTimeout(timeout);
        ws.removeEventListener("message", listener);
        resolve(true);
      }
    };
    ws.addEventListener("message", listener);
  });

  ws.send(
    JSON.stringify({
      type: "cf_agent_use_chat_request",
      id: requestId,
      init: {
        method: "POST",
        body: JSON.stringify({ messages })
      }
    })
  );

  const done = await donePromise;
  expect(done).toBe(true);
  return received;
}
