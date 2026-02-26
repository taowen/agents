import * as Sentry from "@sentry/cloudflare";
import { Hono } from "hono";
import { routeAgentRequest } from "agents";
import { handleAuthRoutes, requireAuth, handleIncomingEmail } from "./auth";
import { apiRoutes } from "./api";
import { handleGitHubOAuth } from "./github-oauth";
import { QUOTA_LIMITS } from "./quota-config";

export { ChatAgent } from "./chat-agent";
export { TunnelRelay } from "./tunnel-relay";

type AppEnv = { Bindings: Env; Variables: { userId: string } };

const app = new Hono<AppEnv>();

// 0. Tunnel host detection — *.cscreen.cc
app.all("*", async (c, next) => {
  const host = c.req.header("host") || "";
  const tunnelMatch = host.match(
    /^([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)\.cscreen\.cc$/
  );
  if (!tunnelMatch) return next();

  const tunnelName = tunnelMatch[1];

  const url = new URL(c.req.url);

  // /tunnel/connect requires auth (tunneld CLI must authenticate)
  if (url.pathname === "/tunnel/connect") {
    const result = await requireAuth(c.req.raw, c.env);
    if (result instanceof Response) return result;
  }

  // Route to TunnelRelay DO by tunnel name
  const id = c.env.TunnelRelay.idFromName(tunnelName);
  const stub = c.env.TunnelRelay.get(id);
  return stub.fetch(c.req.raw);
});

// 1. Public auth routes (Google OAuth + Email login) — no auth required
app.all("/auth/*", (c) => handleAuthRoutes(c.req.raw, c.env));

// 1.5 Public download page & R2 public file serving — no auth required
app.get("/download", (c) => {
  const apkUrl =
    "https://ai.connect-screen.com/api/public/connect-screen-28d7fda.apk";
  const winZipUrl =
    "https://ai.connect-screen.com/api/public/connect-screen-win-572d4ce.zip";
  const apkQrUrl = `https://api.qrserver.com/v1/create-qr-code/?data=${encodeURIComponent(apkUrl)}&size=200x200`;
  const winQrUrl = `https://api.qrserver.com/v1/create-qr-code/?data=${encodeURIComponent(winZipUrl)}&size=200x200`;
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Connect Screen – Download</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh;background:#f5f5f5;color:#333}
.container{display:flex;gap:32px;flex-wrap:wrap;justify-content:center;padding:32px}
.card{background:#fff;border-radius:16px;padding:40px 36px;text-align:center;box-shadow:0 2px 16px rgba(0,0,0,.08);max-width:360px;width:100%}
h1{font-size:22px;margin-bottom:6px}
p.sub{color:#666;margin-bottom:24px;font-size:14px}
img.qr{border-radius:8px;margin-bottom:20px}
a.btn{display:inline-block;color:#fff;text-decoration:none;padding:12px 28px;border-radius:8px;font-size:15px;font-weight:500;transition:background .15s}
a.btn.android{background:#2563eb}
a.btn.android:hover{background:#1d4ed8}
a.btn.windows{background:#0078d4}
a.btn.windows:hover{background:#006abc}
p.hint{margin-top:20px;font-size:12px;color:#999}
a.link{display:block;margin-top:12px;font-size:12px;color:#2563eb;word-break:break-all}
.icon{font-size:36px;margin-bottom:12px}
</style>
</head>
<body>
<div class="container">
<div class="card">
<div class="icon">&#x1f4f1;</div>
<h1>Android</h1>
<p class="sub">Scan the QR code or tap below to download the APK.</p>
<img class="qr" src="${apkQrUrl}" width="200" height="200" alt="Android QR Code">
<br>
<a class="btn android" href="${apkUrl}">Download APK</a>
<a class="link" href="${apkUrl}">${apkUrl}</a>
<p class="hint">Requires Android 8.0+. Allow installs from unknown sources.</p>
</div>
<div class="card">
<div class="icon">&#x1f5a5;</div>
<h1>Windows</h1>
<p class="sub">Scan the QR code or tap below to download the PowerShell agent.</p>
<img class="qr" src="${winQrUrl}" width="200" height="200" alt="Windows QR Code">
<br>
<a class="btn windows" href="${winZipUrl}">Download ZIP</a>
<a class="link" href="${winZipUrl}">${winZipUrl}</a>
<p class="hint">Requires Windows 10/11 with PowerShell 5.1+. Extract and run connect.ps1.</p>
</div>
</div>
</body>
</html>`;
  return c.html(html);
});

app.get("/api/public/:key", async (c) => {
  const key = c.req.param("key");
  const object = await c.env.R2_PUBLIC.get(key);
  if (!object) return c.notFound();
  const headers = new Headers();
  object.writeHttpMetadata(headers);
  headers.set("etag", object.httpEtag);
  if (key.endsWith(".apk")) {
    headers.set("content-type", "application/vnd.android.package-archive");
    headers.set("content-disposition", `attachment; filename="${key}"`);
  } else if (key.endsWith(".zip")) {
    headers.set("content-type", "application/zip");
    headers.set("content-disposition", `attachment; filename="${key}"`);
  }
  return new Response(object.body, { headers });
});

// Serve static assets for non-tunnel, non-API paths (before auth — allows unauthenticated SPA access)
app.use("*", async (c, next) => {
  const host = c.req.header("host") || "";
  // Tunnel domain — always continue to worker handlers
  if (host.endsWith(".cscreen.cc")) return next();

  const path = new URL(c.req.url).pathname;
  // Known worker paths — continue to auth + handlers
  if (
    path.startsWith("/agents/") ||
    path.startsWith("/api/") ||
    path.startsWith("/oauth/") ||
    path.startsWith("/mcp-callback/")
  ) {
    return next();
  }
  // Everything else on main domain — serve from static assets
  return c.env.ASSETS.fetch(c.req.raw);
});

// 2. Auth middleware — gates everything below
app.use("*", async (c, next) => {
  const result = await requireAuth(c.req.raw, c.env);
  if (result instanceof Response) return result;
  c.set("userId", result);
  return next();
});

// 3. API sub-app (Hono strips /api prefix automatically)
app.route("/api", apiRoutes);

// 4. GitHub OAuth (D1-based, per-user)
app.all("/oauth/github/*", (c) =>
  handleGitHubOAuth(c.req.raw, c.env, c.get("userId"))
);
app.all("/oauth/github", (c) =>
  handleGitHubOAuth(c.req.raw, c.env, c.get("userId"))
);

// 5. MCP OAuth callback — route directly to the DO by hex ID
app.all("/mcp-callback/:doId", (c) => {
  const id = c.env.ChatAgent.idFromString(c.req.param("doId"));
  const stub = c.env.ChatAgent.get(id);
  return stub.fetch(c.req.raw);
});

// 6. Agent catch-all (userId injection, session isolation, routeAgentRequest)
app.all("*", async (c) => {
  const userId = c.get("userId");
  const url = new URL(c.req.url);
  const headers = new Headers(c.req.raw.headers);
  headers.set("x-user-id", userId);

  const agentMatch = url.pathname.match(/\/agents\/([^/]+)\/([^/?]+)(\/.*)?/);
  if (agentMatch) {
    const agentName = agentMatch[1];
    const restOfPath = agentMatch[3] || "";
    const sessionId = decodeURIComponent(agentMatch[2]);
    headers.set("x-session-id", sessionId);
    const isolatedName = encodeURIComponent(`${userId}:${sessionId}`);
    url.pathname = `/agents/${agentName}/${isolatedName}${restOfPath}`;
  }
  const agentReq = new Request(url.toString(), {
    method: c.req.method,
    headers,
    body: c.req.raw.body
  });
  return (
    (await routeAgentRequest(agentReq, c.env)) ||
    new Response("Not found", { status: 404 })
  );
});

const sentryHandler = Sentry.withSentry(
  (env: Env) => ({
    dsn: env.SENTRY_DSN,
    tracesSampleRate: 1.0
  }),
  {
    fetch: (req, env, ctx) => app.fetch(req, env, ctx)
  } satisfies ExportedHandler<Env>
);

export default {
  ...sentryHandler,
  async email(message: ForwardableEmailMessage, env: Env) {
    await handleIncomingEmail(message, env);
  },
  async scheduled(_event: ScheduledEvent, env: Env, ctx: ExecutionContext) {
    const now = new Date();
    const currentHour = now.toISOString().slice(0, 13);
    const todayStart = now.toISOString().slice(0, 10) + "T00";

    try {
      // Find users exceeding any builtin-key quota limit
      const overQuota = await env.DB.prepare(
        `SELECT user_id,
          SUM(CASE WHEN hour = ? THEN request_count ELSE 0 END) as hourly_reqs,
          SUM(CASE WHEN hour = ? THEN input_tokens + output_tokens ELSE 0 END) as hourly_tokens,
          SUM(request_count) as daily_reqs,
          SUM(input_tokens + output_tokens) as daily_tokens
        FROM usage_archive
        WHERE api_key_type = 'builtin' AND hour >= ?
        GROUP BY user_id
        HAVING hourly_reqs > ? OR hourly_tokens > ? OR daily_reqs > ? OR daily_tokens > ?`
      )
        .bind(
          currentHour,
          currentHour,
          todayStart,
          QUOTA_LIMITS.HOURLY_REQUEST_LIMIT,
          QUOTA_LIMITS.HOURLY_TOKEN_LIMIT,
          QUOTA_LIMITS.DAILY_REQUEST_LIMIT,
          QUOTA_LIMITS.DAILY_TOKEN_LIMIT
        )
        .all<{
          user_id: string;
          hourly_reqs: number;
          hourly_tokens: number;
          daily_reqs: number;
          daily_tokens: number;
        }>();

      if (overQuota.results.length > 0) {
        const userIds = overQuota.results.map((r) => r.user_id);
        console.log(
          `[cron] disabling ${userIds.length} over-quota users:`,
          userIds
        );
        // Batch update: disable users who don't already have the flag set
        const stmts = userIds.map((uid) =>
          env.DB.prepare(
            `UPDATE users SET builtin_quota_exceeded_at = datetime('now') WHERE id = ? AND builtin_quota_exceeded_at IS NULL`
          ).bind(uid)
        );
        for (let i = 0; i < stmts.length; i += 100) {
          await env.DB.batch(stmts.slice(i, i + 100));
        }
      }
    } catch (e) {
      console.error("[cron] quota enforcement failed:", e);
    }
  }
};
