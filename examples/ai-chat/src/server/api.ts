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

type UsageRow = {
  hour: string;
  request_count: number;
  input_tokens: number | null;
  cache_read_tokens: number | null;
  cache_write_tokens: number | null;
  output_tokens: number | null;
};

async function cacheSessionUsage(
  db: D1Database,
  userId: string,
  sessionId: string,
  rows: UsageRow[]
): Promise<void> {
  if (rows.length === 0) return;
  const stmts = rows.map((r) =>
    db
      .prepare(
        `INSERT OR REPLACE INTO usage_archive (user_id, session_id, hour, request_count, input_tokens, cache_read_tokens, cache_write_tokens, output_tokens)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .bind(
        userId,
        sessionId,
        r.hour,
        r.request_count,
        r.input_tokens || 0,
        r.cache_read_tokens || 0,
        r.cache_write_tokens || 0,
        r.output_tokens || 0
      )
  );
  for (let i = 0; i < stmts.length; i += 100) {
    await db.batch(stmts.slice(i, i + 100));
  }
}

async function fetchDoUsage(
  env: Env,
  userId: string,
  sessionId: string,
  since?: string
): Promise<UsageRow[]> {
  const doId = env.ChatAgent.idFromName(`${userId}:${sessionId}`);
  const stub = env.ChatAgent.get(doId);
  const url = since
    ? `http://agent/get-usage?since=${encodeURIComponent(since)}`
    : "http://agent/get-usage";
  const res = await stub.fetch(
    new Request(url, {
      headers: { "x-partykit-room": `${userId}:${sessionId}` }
    })
  );
  return res.ok ? ((await res.json()) as UsageRow[]) : [];
}

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

  // Archive usage before deletion (full fetch — last chance)
  try {
    const rows = await fetchDoUsage(c.env, userId, sessionId);
    await cacheSessionUsage(c.env.DB, userId, sessionId, rows);
  } catch (e) {
    console.error("usage archive failed:", e);
    // Don't block deletion if archiving fails
  }

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

// GET /usage — hourly token usage (incremental cache + D1 archive)
api.get("/usage", async (c) => {
  try {
    const userId = c.get("userId");
    const start =
      c.req.query("start") || new Date().toISOString().slice(0, 11) + "00";
    const end = c.req.query("end") || new Date().toISOString().slice(0, 13);

    const sessions = await listSessions(c.env.DB, userId);
    const activeIds = sessions.map((s) => s.id);

    // 1. Query max cached hour per active session
    const maxHours = new Map<string, string>();
    if (activeIds.length > 0) {
      const placeholders = activeIds.map(() => "?").join(",");
      const cached = await c.env.DB.prepare(
        `SELECT session_id, MAX(hour) as max_hour FROM usage_archive
       WHERE user_id = ? AND session_id IN (${placeholders})
       GROUP BY session_id`
      )
        .bind(userId, ...activeIds)
        .all<{ session_id: string; max_hour: string }>();
      for (const r of cached.results) {
        maxHours.set(r.session_id, r.max_hour);
      }
    }

    // 2. D1 archive — exclude active sessions (their data comes fresh from DO)
    let archivedQuery = `SELECT hour, SUM(request_count) as request_count,
     SUM(input_tokens) as input_tokens, SUM(cache_read_tokens) as cache_read_tokens,
     SUM(cache_write_tokens) as cache_write_tokens, SUM(output_tokens) as output_tokens
     FROM usage_archive WHERE user_id = ? AND hour >= ? AND hour <= ?`;
    const archivedBinds: unknown[] = [userId, start, end];
    if (activeIds.length > 0) {
      const ph = activeIds.map(() => "?").join(",");
      archivedQuery += ` AND session_id NOT IN (${ph})`;
      archivedBinds.push(...activeIds);
    }
    archivedQuery += ` GROUP BY hour`;
    const archived = await c.env.DB.prepare(archivedQuery)
      .bind(...archivedBinds)
      .all<{
        hour: string;
        request_count: number;
        input_tokens: number;
        cache_read_tokens: number;
        cache_write_tokens: number;
        output_tokens: number;
      }>();

    // 3. Incremental fetch from active DOs (only >= max cached hour)
    const activeResults = await Promise.allSettled(
      sessions.map(async (s) => {
        const since = maxHours.get(s.id); // undefined → full fetch
        const rows = await fetchDoUsage(c.env, userId, s.id, since);
        return { sessionId: s.id, rows };
      })
    );

    // 4. Fire-and-forget: cache incremental data to D1
    const cachePromises: Promise<void>[] = [];
    for (const r of activeResults) {
      if (r.status !== "fulfilled") continue;
      cachePromises.push(
        cacheSessionUsage(c.env.DB, userId, r.value.sessionId, r.value.rows)
      );
    }
    c.executionCtx.waitUntil(Promise.allSettled(cachePromises));

    // 5. Merge: D1 archive (non-active) + fresh DO data → aggregate by hour
    const map = new Map<
      string,
      {
        hour: string;
        request_count: number;
        input_tokens: number;
        cache_read_tokens: number;
        cache_write_tokens: number;
        output_tokens: number;
      }
    >();

    for (const row of archived.results) {
      map.set(row.hour, { ...row });
    }

    for (const result of activeResults) {
      if (result.status !== "fulfilled") continue;
      for (const row of result.value.rows) {
        if (row.hour < start || row.hour > end) continue;
        const existing = map.get(row.hour);
        if (existing) {
          existing.request_count += row.request_count || 0;
          existing.input_tokens += row.input_tokens || 0;
          existing.cache_read_tokens += row.cache_read_tokens || 0;
          existing.cache_write_tokens += row.cache_write_tokens || 0;
          existing.output_tokens += row.output_tokens || 0;
        } else {
          map.set(row.hour, {
            hour: row.hour,
            request_count: row.request_count || 0,
            input_tokens: row.input_tokens || 0,
            cache_read_tokens: row.cache_read_tokens || 0,
            cache_write_tokens: row.cache_write_tokens || 0,
            output_tokens: row.output_tokens || 0
          });
        }
      }
    }

    const merged = [...map.values()].sort((a, b) =>
      a.hour.localeCompare(b.hour)
    );
    return c.json(merged);
  } catch (e) {
    console.error("GET /usage failed:", e);
    Sentry.captureException(e);
    return c.json({ error: String(e) }, 500);
  }
});

// File Manager routes — delegate to existing handler
api.all("/files/*", async (c) => {
  const userId = c.get("userId");
  const fileResponse = await handleFileRoutes(c.req.raw, c.env, userId);
  if (fileResponse) return fileResponse;
  return c.json({ error: "Not found" }, 404);
});

export { api as apiRoutes };
