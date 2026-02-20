/**
 * API routes for session management and user settings.
 * All routes require authentication (userId set by auth middleware).
 */

import { Hono } from "hono";
import * as Sentry from "@sentry/cloudflare";
import {
  getUser,
  listSessions,
  createSession,
  updateSessionTitle,
  deleteSession,
  getMemoryFiles,
  putMemoryFiles
} from "./db";
import { handleFileRoutes } from "./api-files";

type ApiEnv = { Bindings: Env; Variables: { userId: string } };

const api = new Hono<ApiEnv>();

// GET /user
api.get("/user", async (c) => {
  const userId = c.get("userId");
  const user = await getUser(c.env.DB, userId);
  if (!user) {
    return c.json({ error: "User not found" }, 404);
  }
  return c.json({
    id: user.id,
    email: user.email,
    name: user.name,
    picture: user.picture
  });
});

// GET /sessions
api.get("/sessions", async (c) => {
  const userId = c.get("userId");
  const sessions = await listSessions(c.env.DB, userId);
  return c.json(sessions);
});

// POST /sessions
api.post("/sessions", async (c) => {
  const userId = c.get("userId");
  let title: string | undefined;
  try {
    const body = await c.req.json<{ title?: string }>();
    title = body.title;
  } catch {
    // empty body is fine
  }
  const session = await createSession(c.env.DB, userId, title);
  return c.json(session, 201);
});

// PATCH /sessions/:id
api.patch("/sessions/:id", async (c) => {
  const userId = c.get("userId");
  const sessionId = c.req.param("id");
  const body = await c.req.json<{ title?: string }>();
  if (!body.title) {
    return c.json({ error: "title is required" }, 400);
  }
  const updated = await updateSessionTitle(
    c.env.DB,
    sessionId,
    userId,
    body.title
  );
  if (!updated) {
    return c.json({ error: "Session not found" }, 404);
  }
  return c.json({ ok: true });
});

// DELETE /sessions/:id
api.delete("/sessions/:id", async (c) => {
  const userId = c.get("userId");
  const sessionId = c.req.param("id");
  const deleted = await deleteSession(c.env.DB, sessionId, userId);
  if (!deleted) {
    return c.json({ error: "Session not found" }, 404);
  }
  return c.json({ ok: true });
});

// GET /sessions/:id/schedules — proxy to DO
api.get("/sessions/:id/schedules", async (c) => {
  const userId = c.get("userId");
  const sessionId = c.req.param("id");
  const id = c.env.ChatAgent.idFromName(`${userId}:${sessionId}`);
  const stub = c.env.ChatAgent.get(id);
  return stub.fetch(
    new Request("http://agent/get-schedules", {
      headers: {
        "x-user-id": userId,
        "x-session-id": sessionId,
        "x-partykit-room": `${userId}:${sessionId}`
      }
    })
  );
});

// POST /sessions/:id/report-bug — handled directly in the worker
api.post("/sessions/:id/report-bug", async (c) => {
  const userId = c.get("userId");
  const sessionId = c.req.param("id");

  const { description } = await c.req.json<{ description: string }>();
  const reportId = `BUG-${Date.now().toString(36).toUpperCase()}-${Array.from(
    crypto.getRandomValues(new Uint8Array(2))
  )
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
    .toUpperCase()}`;

  // Compute sessionDir from the DO ID (first 12 hex chars)
  const doId = c.env.ChatAgent.idFromName(`${userId}:${sessionId}`);
  const sessionDir = doId.toString().slice(0, 12);

  // Query D1 for recent chat messages
  const chatPrefix = `/home/user/.chat/${sessionDir}/`;
  const rows = await c.env.DB.prepare(
    `SELECT path, CAST(content AS TEXT) as content FROM files
     WHERE user_id = ? AND parent_path = ? AND is_directory = 0
     ORDER BY mtime DESC LIMIT 10`
  )
    .bind(userId, chatPrefix.slice(0, -1))
    .all<{ path: string; content: string }>();

  const recentMessages = rows.results.map((r) => {
    const text = r.content || "";
    return { path: r.path, text: text.slice(0, 500) };
  });

  Sentry.withScope((scope) => {
    scope.setUser({ id: userId });
    scope.setTag("report_id", reportId);
    scope.setTag("user_id", userId);
    scope.setTag("session_uuid", sessionId);
    scope.setContext("bug_report", {
      description,
      reportId,
      sessionUuid: sessionId,
      userId
    });
    scope.setContext("recent_messages", { messages: recentMessages });
    Sentry.captureMessage(`[Bug Report ${reportId}] ${description}`, "warning");
  });

  return c.json({ reportId });
});

// GET /memory
api.get("/memory", async (c) => {
  const userId = c.get("userId");
  const data = await getMemoryFiles(c.env.DB, userId);
  return c.json(data);
});

// PUT /memory
api.put("/memory", async (c) => {
  const userId = c.get("userId");
  const body = await c.req.json<{
    profile?: string;
    preferences?: string;
    entities?: string;
  }>();
  await putMemoryFiles(c.env.DB, userId, body);
  return c.json({ ok: true });
});

// File Manager routes — delegate to existing handler
api.all("/files/*", async (c) => {
  const userId = c.get("userId");
  const fileResponse = await handleFileRoutes(c.req.raw, c.env, userId);
  if (fileResponse) return fileResponse;
  return c.json({ error: "Not found" }, 404);
});

export { api as apiRoutes };
