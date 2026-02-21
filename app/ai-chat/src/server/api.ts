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
import { QUOTA_LIMITS } from "./quota-config";

type ApiEnv = { Bindings: Env; Variables: { userId: string } };

type UsageRow = {
  hour: string;
  api_key_type: string;
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
        `INSERT OR REPLACE INTO usage_archive (user_id, session_id, hour, api_key_type, request_count, input_tokens, cache_read_tokens, cache_write_tokens, output_tokens)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .bind(
        userId,
        sessionId,
        r.hour,
        r.api_key_type || "unknown",
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
    return `${r.path}: ${text.slice(0, 200)}`;
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
      api_key_type: string;
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
      let archivedQuery = `SELECT hour, api_key_type, SUM(request_count) as request_count,
       SUM(input_tokens) as input_tokens, SUM(cache_read_tokens) as cache_read_tokens,
       SUM(cache_write_tokens) as cache_write_tokens, SUM(output_tokens) as output_tokens
       FROM usage_archive WHERE user_id = ? AND hour >= ? AND hour <= ?`;
      const archivedBinds: unknown[] = [userId, start, end];
      if (activeIds.length > 0) {
        const ph = activeIds.map(() => "?").join(",");
        archivedQuery += ` AND session_id NOT IN (${ph})`;
        archivedBinds.push(...activeIds);
      }
      archivedQuery += ` GROUP BY hour, api_key_type`;
      const archived = await c.env.DB.prepare(archivedQuery)
        .bind(...archivedBinds)
        .all<{
          hour: string;
          api_key_type: string;
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

    // 5. Merge: D1 archive (non-active) + fresh DO data → aggregate by hour|api_key_type
    type MergedRow = {
      hour: string;
      api_key_type: string;
      request_count: number;
      input_tokens: number;
      cache_read_tokens: number;
      cache_write_tokens: number;
      output_tokens: number;
    };
    const map = new Map<string, MergedRow>();

    const mergeKey = (hour: string, apiKeyType: string) =>
      `${hour}|${apiKeyType}`;

    const addToMap = (
      hour: string,
      apiKeyType: string,
      rc: number,
      it: number,
      crt: number,
      cwt: number,
      ot: number
    ) => {
      const key = mergeKey(hour, apiKeyType);
      const existing = map.get(key);
      if (existing) {
        existing.request_count += rc;
        existing.input_tokens += it;
        existing.cache_read_tokens += crt;
        existing.cache_write_tokens += cwt;
        existing.output_tokens += ot;
      } else {
        map.set(key, {
          hour,
          api_key_type: apiKeyType,
          request_count: rc,
          input_tokens: it,
          cache_read_tokens: crt,
          cache_write_tokens: cwt,
          output_tokens: ot
        });
      }
    };

    for (const row of archivedResults) {
      addToMap(
        row.hour,
        row.api_key_type || "unknown",
        row.request_count || 0,
        row.input_tokens || 0,
        row.cache_read_tokens || 0,
        row.cache_write_tokens || 0,
        row.output_tokens || 0
      );
    }

    for (const result of activeResults) {
      if (result.status !== "fulfilled") continue;
      for (const row of result.value.rows) {
        if (row.hour < start || row.hour > end) continue;
        addToMap(
          row.hour,
          (row as UsageRow).api_key_type || "unknown",
          row.request_count || 0,
          row.input_tokens || 0,
          row.cache_read_tokens || 0,
          row.cache_write_tokens || 0,
          row.output_tokens || 0
        );
      }
    }

    const merged = [...map.values()].sort(
      (a, b) =>
        a.hour.localeCompare(b.hour) ||
        a.api_key_type.localeCompare(b.api_key_type)
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

// GET /quota — current user's builtin usage vs limits
api.get("/quota", async (c) => {
  const userId = c.get("userId");
  const now = new Date();
  const currentHour = now.toISOString().slice(0, 13);
  const todayStart = now.toISOString().slice(0, 10) + "T00";

  const row = await c.env.DB.prepare(
    `SELECT
      SUM(CASE WHEN hour = ? THEN request_count ELSE 0 END) as hourly_reqs,
      SUM(CASE WHEN hour = ? THEN input_tokens + output_tokens ELSE 0 END) as hourly_tokens,
      SUM(request_count) as daily_reqs,
      SUM(input_tokens + output_tokens) as daily_tokens
    FROM usage_archive
    WHERE user_id = ? AND api_key_type = 'builtin' AND hour >= ?`
  )
    .bind(currentHour, currentHour, userId, todayStart)
    .first<{
      hourly_reqs: number;
      hourly_tokens: number;
      daily_reqs: number;
      daily_tokens: number;
    }>();

  const userRow = await c.env.DB.prepare(
    `SELECT builtin_quota_exceeded_at FROM users WHERE id = ?`
  )
    .bind(userId)
    .first<{ builtin_quota_exceeded_at: string | null }>();

  return c.json({
    exceeded: !!userRow?.builtin_quota_exceeded_at,
    exceededAt: userRow?.builtin_quota_exceeded_at || null,
    hourly: {
      requests: row?.hourly_reqs || 0,
      tokens: row?.hourly_tokens || 0,
      requestLimit: QUOTA_LIMITS.HOURLY_REQUEST_LIMIT,
      tokenLimit: QUOTA_LIMITS.HOURLY_TOKEN_LIMIT
    },
    daily: {
      requests: row?.daily_reqs || 0,
      tokens: row?.daily_tokens || 0,
      requestLimit: QUOTA_LIMITS.DAILY_REQUEST_LIMIT,
      tokenLimit: QUOTA_LIMITS.DAILY_TOKEN_LIMIT
    }
  });
});

// POST /admin/reenable-user — clear builtin_quota_exceeded_at (protected by ADMIN_SECRET)
api.post("/admin/reenable-user", async (c) => {
  const secret = c.req.header("x-admin-secret");
  if (!secret || secret !== c.env.ADMIN_SECRET) {
    return c.json({ error: "Unauthorized" }, 401);
  }
  const { user_id } = await c.req.json<{ user_id: string }>();
  if (!user_id) {
    return c.json({ error: "user_id is required" }, 400);
  }
  await c.env.DB.prepare(
    `UPDATE users SET builtin_quota_exceeded_at = NULL WHERE id = ?`
  )
    .bind(user_id)
    .run();
  return c.json({ ok: true });
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

  // Resolve LLM config: prefer user's custom config over builtin
  const llmRow = await c.env.DB.prepare(
    "SELECT content FROM files WHERE user_id = ? AND path = ?"
  )
    .bind(userId, "/etc/llm.json")
    .first<{ content: ArrayBuffer | null }>();

  let upstreamBaseURL = c.env.BUILTIN_LLM_BASE_URL;
  let upstreamApiKey = c.env.BUILTIN_LLM_API_KEY;
  let upstreamModel = c.env.BUILTIN_LLM_MODEL;
  let apiKeyType = "builtin";

  if (llmRow?.content) {
    try {
      const cfg = JSON.parse(new TextDecoder().decode(llmRow.content));
      if (cfg.base_url && cfg.api_key) {
        upstreamBaseURL = cfg.base_url;
        upstreamApiKey = cfg.api_key;
        upstreamModel = cfg.model || upstreamModel;
        apiKeyType = "custom";
      }
    } catch {}
  }

  // Quota check only for builtin key
  if (apiKeyType === "builtin") {
    const quotaRow = await c.env.DB.prepare(
      `SELECT builtin_quota_exceeded_at FROM users WHERE id = ?`
    )
      .bind(userId)
      .first<{ builtin_quota_exceeded_at: string | null }>();
    if (quotaRow?.builtin_quota_exceeded_at) {
      return c.json(
        {
          error: {
            message:
              "Builtin API key usage quota exceeded. Please configure your own API key in Settings.",
            type: "quota_exceeded"
          }
        },
        429
      );
    }
  }

  const body = await c.req.json();

  // Forward to resolved upstream LLM
  const upstreamRes = await fetch(`${upstreamBaseURL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${upstreamApiKey}`
    },
    body: JSON.stringify({
      ...body,
      model: body.model || upstreamModel
    })
  });

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

  // Fire-and-forget: write to usage_archive for quota enforcement
  const choice = responseBody.choices?.[0]?.message;
  if (choice) {
    const inputTokens = responseBody.usage?.prompt_tokens || 0;
    const outputTokens = responseBody.usage?.completion_tokens || 0;
    const cacheReadTokens =
      responseBody.usage?.prompt_tokens_details?.cached_tokens || 0;

    const proxyHour = new Date().toISOString().slice(0, 13);
    c.executionCtx.waitUntil(
      c.env.DB.prepare(
        `INSERT INTO usage_archive (user_id, session_id, hour, api_key_type, request_count, input_tokens, cache_read_tokens, cache_write_tokens, output_tokens)
         VALUES (?, '__proxy__', ?, ?, 1, ?, ?, 0, ?)
         ON CONFLICT(user_id, session_id, hour, api_key_type) DO UPDATE SET
           request_count = request_count + 1,
           input_tokens = input_tokens + excluded.input_tokens,
           cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens,
           output_tokens = output_tokens + excluded.output_tokens`
      )
        .bind(
          userId,
          proxyHour,
          apiKeyType,
          inputTokens,
          cacheReadTokens,
          outputTokens
        )
        .run()
        .catch((e: unknown) =>
          console.error("proxy usage_archive write failed:", e)
        )
    );
  }

  return Response.json(responseBody);
});

// GET /devices — list online devices for the current user (checks DO liveness)
api.get("/devices", async (c) => {
  const userId = c.get("userId");
  const rows = await c.env.DB.prepare(
    "SELECT id, title FROM sessions WHERE user_id = ? AND id LIKE 'device-%'"
  )
    .bind(userId)
    .all<{ id: string; title: string }>();

  // Check each device DO for actual WebSocket liveness
  const results = await Promise.allSettled(
    rows.results.map(async (s) => {
      const stub = c.env.ChatAgent.get(
        c.env.ChatAgent.idFromName(encodeURIComponent(`${userId}:${s.id}`))
      );
      const res = await stub.fetch(new Request("http://agent/status"));
      const body = (await res.json()) as { online: boolean };
      return { ...s, online: body.online };
    })
  );

  const online = results
    .map((r, i) =>
      r.status === "fulfilled" ? r.value : { ...rows.results[i], online: false }
    )
    .filter((r) => r.online);

  return c.json(
    online.map((r) => ({
      deviceName: r.id.replace("device-", ""),
      sessionId: r.id,
      title: r.title
    }))
  );
});

// File Manager routes — delegate to existing handler
api.all("/files/*", async (c) => {
  const userId = c.get("userId");
  const fileResponse = await handleFileRoutes(c.req.raw, c.env, userId);
  if (fileResponse) return fileResponse;
  return c.json({ error: "Not found" }, 404);
});

export { api as apiRoutes };
