/**
 * API routes for session management and user settings.
 * All routes require authentication (userId set by auth middleware).
 */

import { Hono } from "hono";
import * as Sentry from "@sentry/cloudflare";
import { submitBugReport } from "./bug-report";
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
import {
  getChatAgentStub,
  chatAgentIsolatedName,
  findDevices
} from "./device-hub";
import {
  aggregateUsage,
  fetchDoUsage,
  cacheSessionUsage
} from "./usage-aggregator";
import {
  resolveLlmConfig,
  callUpstreamLlm,
  archiveProxyUsage
} from "./llm-proxy";
import { checkQuota } from "./usage-tracker";

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

  // Archive usage before deletion (full fetch — last chance)
  try {
    const rows = await fetchDoUsage(c.env, userId, sessionId);
    await cacheSessionUsage(c.env.DB, userId, sessionId, rows);
  } catch (e) {
    console.error("usage archive failed:", e);
    // Don't block deletion if archiving fails
  }

  // Cancel all scheduled tasks and destroy the DO
  try {
    const stub = getChatAgentStub(c.env.ChatAgent, userId, sessionId);
    await stub.fetch(new Request("http://agent/destroy", { method: "POST" }));
  } catch (e) {
    console.error("destroy DO failed:", e);
    // Don't block deletion if destruction fails
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
  const stub = getChatAgentStub(c.env.ChatAgent, userId, sessionId);
  return stub.fetch(
    new Request("http://agent/get-schedules", {
      headers: {
        "x-user-id": userId,
        "x-session-id": sessionId,
        "x-partykit-room": chatAgentIsolatedName(userId, sessionId)
      }
    })
  );
});

// POST /sessions/:id/report-bug
api.post("/sessions/:id/report-bug", async (c) => {
  const userId = c.get("userId");
  const sessionId = c.req.param("id");
  const { description } = await c.req.json<{ description: string }>();

  const result = await submitBugReport({
    db: c.env.DB,
    r2: c.env.R2,
    chatAgentNs: c.env.ChatAgent,
    userId,
    sessionId,
    description
  });

  return c.json(result);
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

    const merged = await aggregateUsage(
      c.env,
      c.env.DB,
      userId,
      start,
      end,
      (p) => c.executionCtx.waitUntil(p)
    );
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
  const config = await resolveLlmConfig(c.env.DB, userId, c.env);

  // Quota check only for builtin key
  if (config.apiKeyType === "builtin") {
    const { exceeded } = await checkQuota(c.env.DB, userId, null);
    if (exceeded) {
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
  const result = await callUpstreamLlm(config, body);

  if (!result.ok) {
    return result.response;
  }

  // Fire-and-forget: write to usage_archive for quota enforcement
  const usagePromise = archiveProxyUsage(
    c.env.DB,
    userId,
    config.apiKeyType,
    result.body
  );
  if (usagePromise) {
    c.executionCtx.waitUntil(usagePromise);
  }

  return Response.json(result.body);
});

// GET /devices — list online devices for the current user (checks DO liveness)
api.get("/devices", async (c) => {
  const userId = c.get("userId");
  const devices = await findDevices(c.env, userId);

  return c.json(
    devices
      .filter((d) => d.online)
      .map((d) => ({
        deviceName: d.name,
        sessionId: d.sessionId,
        title: d.title
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
