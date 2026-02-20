import * as Sentry from "@sentry/cloudflare";
import { Hono } from "hono";
import { routeAgentRequest } from "agents";
import { handleAuthRoutes, requireAuth, handleIncomingEmail } from "./auth";
import { apiRoutes } from "./api";
import { handleGitHubOAuth } from "./github-oauth";

export { ChatAgent } from "./chat-agent";

type AppEnv = { Bindings: Env; Variables: { userId: string } };

const app = new Hono<AppEnv>();

// 1. Public auth routes (Google OAuth + Email login) — no auth required
app.all("/auth/*", (c) => handleAuthRoutes(c.req.raw, c.env));

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
