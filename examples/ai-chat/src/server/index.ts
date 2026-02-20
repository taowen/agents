import * as Sentry from "@sentry/cloudflare";
import { Hono } from "hono";
import { routeAgentRequest } from "agents";
import { handleAuthRoutes, requireAuth, handleIncomingEmail } from "./auth";
import { apiRoutes } from "./api";
import { handleGitHubOAuth } from "./github-oauth";

export { ChatAgent } from "./chat-agent";
export { DeviceHub } from "./device-hub";

type AppEnv = { Bindings: Env; Variables: { userId: string } };

const app = new Hono<AppEnv>();

// 1. Public auth routes (Google OAuth + Email login) — no auth required
app.all("/auth/*", (c) => handleAuthRoutes(c.req.raw, c.env));

// 1.5 Public download page & R2 public file serving — no auth required
app.get("/download", (c) => {
  const apkUrl = "https://ai.connect-screen.com/api/public/connect-screen.apk";
  const qrUrl = `https://api.qrserver.com/v1/create-qr-code/?data=${encodeURIComponent(apkUrl)}&size=200x200`;
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Connect Screen – Download</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;display:flex;align-items:center;justify-content:center;min-height:100vh;background:#f5f5f5;color:#333}
.card{background:#fff;border-radius:16px;padding:48px 40px;text-align:center;box-shadow:0 2px 16px rgba(0,0,0,.08);max-width:400px;width:90%}
h1{font-size:24px;margin-bottom:8px}
p.sub{color:#666;margin-bottom:32px;font-size:14px}
img{border-radius:8px;margin-bottom:24px}
a.btn{display:inline-block;background:#2563eb;color:#fff;text-decoration:none;padding:12px 32px;border-radius:8px;font-size:16px;font-weight:500;transition:background .15s}
a.btn:hover{background:#1d4ed8}
p.hint{margin-top:24px;font-size:12px;color:#999}
</style>
</head>
<body>
<div class="card">
<h1>Connect Screen</h1>
<p class="sub">Scan the QR code or tap the button below to download the Android app.</p>
<img src="${qrUrl}" width="200" height="200" alt="QR Code">
<br>
<a class="btn" href="${apkUrl}">Download APK</a>
<p class="hint">Requires Android 8.0+. You may need to allow installs from unknown sources.</p>
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
  }
  return new Response(object.body, { headers });
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
  }
};
