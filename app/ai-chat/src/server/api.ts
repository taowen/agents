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
import { createDeviceToken } from "./auth";

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
  const isolatedName = encodeURIComponent(`${userId}:${sessionId}`);
  const doId = env.ChatAgent.idFromName(isolatedName);
  const stub = env.ChatAgent.get(doId);
  const url = since
    ? `http://agent/get-usage?since=${encodeURIComponent(since)}`
    : "http://agent/get-usage";
  const res = await stub.fetch(
    new Request(url, {
      headers: { "x-partykit-room": isolatedName }
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
  const isolatedName = encodeURIComponent(`${userId}:${sessionId}`);
  const id = c.env.ChatAgent.idFromName(isolatedName);
  const stub = c.env.ChatAgent.get(id);
  return stub.fetch(
    new Request("http://agent/get-schedules", {
      headers: {
        "x-user-id": userId,
        "x-session-id": sessionId,
        "x-partykit-room": isolatedName
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
  const doId = c.env.ChatAgent.idFromName(
    encodeURIComponent(`${userId}:${sessionId}`)
  );
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

    console.log(
      `[usage] userId=${userId} sessions=${activeIds.length} start=${start} end=${end}`
    );
    Sentry.addBreadcrumb({
      category: "usage",
      message: `sessions=${activeIds.length} start=${start} end=${end}`,
      level: "info"
    });

    // 1-2. D1 archive queries (wrapped so failures don't block DO fetches)
    const maxHours = new Map<string, string>();
    let archivedResults: {
      hour: string;
      request_count: number;
      input_tokens: number;
      cache_read_tokens: number;
      cache_write_tokens: number;
      output_tokens: number;
    }[] = [];

    try {
      // 1. Query max cached hour per active session
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
      archivedResults = archived.results;
      console.log(
        `[usage] D1 archive: ${archivedResults.length} rows, maxHours=${maxHours.size} sessions cached`
      );
      Sentry.addBreadcrumb({
        category: "usage",
        message: `D1 archive: ${archivedResults.length} rows, maxHours=${maxHours.size}`,
        level: "info"
      });
    } catch (e) {
      console.error(
        "D1 usage_archive query failed (continuing with DO data):",
        e
      );
      Sentry.captureException(e);
    }

    // 3. Incremental fetch from active DOs (only >= max cached hour)
    const activeResults = await Promise.allSettled(
      sessions.map(async (s) => {
        const since = maxHours.get(s.id); // undefined → full fetch
        const rows = await fetchDoUsage(c.env, userId, s.id, since);
        return { sessionId: s.id, rows };
      })
    );

    // 3b. Log each DO result
    for (const [i, r] of activeResults.entries()) {
      if (r.status === "fulfilled") {
        console.log(
          `[usage] DO ${sessions[i].id}: ${r.value.rows.length} rows`
        );
      } else {
        console.error(`[usage] DO ${sessions[i].id} FAILED:`, r.reason);
      }
    }
    Sentry.addBreadcrumb({
      category: "usage",
      message: `DO fetches: ${activeResults.filter((r) => r.status === "fulfilled").length}/${activeResults.length} succeeded`,
      level: "info"
    });

    // 4. Fire-and-forget: cache incremental data to D1
    const cachePromises: Promise<void>[] = [];
    for (const r of activeResults) {
      if (r.status === "rejected") {
        console.error("fetchDoUsage failed:", r.reason);
        continue;
      }
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

    for (const row of archivedResults) {
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

    // 6. Add device_messages usage
    const deviceUsage = await c.env.DB.prepare(
      `SELECT
        strftime('%Y-%m-%dT%H', created_at) as hour,
        COUNT(*) as request_count,
        SUM(json_extract(message, '$.metadata.usage.inputTokens')) as input_tokens,
        SUM(json_extract(message, '$.metadata.usage.cacheReadTokens')) as cache_read_tokens,
        SUM(json_extract(message, '$.metadata.usage.cacheWriteTokens')) as cache_write_tokens,
        SUM(json_extract(message, '$.metadata.usage.outputTokens')) as output_tokens
      FROM device_messages
      WHERE user_id = ? AND strftime('%Y-%m-%dT%H', created_at) >= ? AND strftime('%Y-%m-%dT%H', created_at) <= ?
        AND json_extract(message, '$.metadata.usage') IS NOT NULL
      GROUP BY hour`
    )
      .bind(userId, start, end)
      .all<UsageRow>();

    console.log(`[usage] device_messages: ${deviceUsage.results.length} rows`);
    Sentry.addBreadcrumb({
      category: "usage",
      message: `device_messages: ${deviceUsage.results.length} rows`,
      level: "info"
    });

    for (const row of deviceUsage.results) {
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

    const merged = [...map.values()].sort((a, b) =>
      a.hour.localeCompare(b.hour)
    );

    console.log(
      `[usage] merged: ${merged.length} hours, total_input=${merged.reduce((s, r) => s + r.input_tokens, 0)}`
    );
    Sentry.addBreadcrumb({
      category: "usage",
      message: `merged: ${merged.length} hours`,
      level: "info"
    });

    return c.json(merged);
  } catch (e) {
    console.error("GET /usage failed:", e);
    Sentry.captureException(e);
    return c.json({ error: String(e) }, 500);
  }
});

// POST /device/approve — web user approves a device code
api.post("/device/approve", async (c) => {
  const userId = c.get("userId");
  const { code } = await c.req.json<{ code: string }>();
  const upperCode = code?.toUpperCase();
  if (!upperCode) {
    return c.json({ error: "Missing code" }, 400);
  }

  const raw = await c.env.OTP_KV.get(`device-login:${upperCode}`);
  if (!raw) {
    return c.json({ error: "Code expired or invalid" }, 404);
  }
  const data = JSON.parse(raw) as { status: string };
  if (data.status !== "pending") {
    return c.json({ error: "Code already used" }, 409);
  }

  const token = await createDeviceToken(userId, c.env.AUTH_SECRET);
  const origin = new URL(c.req.url).origin;

  await c.env.OTP_KV.put(
    `device-login:${upperCode}`,
    JSON.stringify({
      status: "approved",
      token,
      baseURL: `${origin}/api/proxy/v1`,
      model: c.env.BUILTIN_LLM_MODEL
    }),
    { expirationTtl: 60 } // short TTL — device will read it soon
  );

  return c.json({ ok: true });
});

// POST /proxy/v1/chat/completions — OpenAI-compatible LLM proxy
api.post("/proxy/v1/chat/completions", async (c) => {
  const userId = c.get("userId");
  const body = await c.req.json();

  // Forward to upstream LLM
  const upstreamRes = await fetch(
    `${c.env.BUILTIN_LLM_BASE_URL}/chat/completions`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${c.env.BUILTIN_LLM_API_KEY}`
      },
      body: JSON.stringify({
        ...body,
        model: body.model || c.env.BUILTIN_LLM_MODEL
      })
    }
  );

  if (!upstreamRes.ok) {
    const errText = await upstreamRes.text();
    return new Response(errText, {
      status: upstreamRes.status,
      headers: { "Content-Type": "application/json" }
    });
  }

  const responseBody = (await upstreamRes.json()) as {
    choices?: { message?: { content?: string; tool_calls?: unknown } }[];
    usage?: {
      prompt_tokens?: number;
      completion_tokens?: number;
      prompt_tokens_details?: { cached_tokens?: number };
    };
  };

  // Fire-and-forget: store assistant message with usage in D1
  const choice = responseBody.choices?.[0]?.message;
  if (choice) {
    const assistantMsg = {
      role: "assistant",
      content: choice.content,
      tool_calls: choice.tool_calls,
      metadata: {
        usage: {
          inputTokens: responseBody.usage?.prompt_tokens || 0,
          outputTokens: responseBody.usage?.completion_tokens || 0,
          cacheReadTokens:
            responseBody.usage?.prompt_tokens_details?.cached_tokens || 0,
          cacheWriteTokens: 0
        }
      }
    };
    c.executionCtx.waitUntil(
      c.env.DB.prepare(
        "INSERT INTO device_messages (id, user_id, message) VALUES (?, ?, ?)"
      )
        .bind(crypto.randomUUID(), userId, JSON.stringify(assistantMsg))
        .run()
    );
  }

  return Response.json(responseBody);
});

// GET /device/ws — WebSocket upgrade, forwarded to DeviceHub DO
api.get("/device/ws", async (c) => {
  const userId = c.get("userId");
  const id = c.env.DeviceHub.idFromName(userId);
  const stub = c.env.DeviceHub.get(id);
  const url = new URL(c.req.url);
  url.pathname = "/connect";
  return stub.fetch(
    new Request(url.toString(), {
      method: c.req.method,
      headers: c.req.raw.headers,
      body: c.req.raw.body
    })
  );
});

// GET /devices — list online devices for the current user
api.get("/devices", async (c) => {
  const userId = c.get("userId");
  const id = c.env.DeviceHub.idFromName(userId);
  const stub = c.env.DeviceHub.get(id);
  const res = await stub.fetch(new Request("http://hub/devices"));
  return new Response(res.body, { status: res.status, headers: res.headers });
});

// File Manager routes — delegate to existing handler
api.all("/files/*", async (c) => {
  const userId = c.get("userId");
  const fileResponse = await handleFileRoutes(c.req.raw, c.env, userId);
  if (fileResponse) return fileResponse;
  return c.json({ error: "Not found" }, 404);
});

export { api as apiRoutes };
