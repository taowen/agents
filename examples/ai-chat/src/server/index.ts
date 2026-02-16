import * as Sentry from "@sentry/cloudflare";
import { routeAgentRequest } from "agents";
import { handleAuthRoutes, requireAuth } from "./auth";
import { handleApiRoutes } from "./api";
import { handleGitHubOAuth } from "./github-oauth";

export { ChatAgent } from "./chat-agent";

export default Sentry.withSentry(
  (env: Env) => ({
    dsn: env.SENTRY_DSN,
    tracesSampleRate: 1.0
  }),
  {
    async fetch(request: Request, env: Env) {
      // 1. Public auth routes (Google OAuth)
      const authResponse = await handleAuthRoutes(request, env);
      if (authResponse) return authResponse;

      // 2. Require authentication for everything else
      const authResult = await requireAuth(request, env);
      if (authResult instanceof Response) return authResult;
      const userId = authResult;

      // 3. API routes (session/settings CRUD)
      const apiResponse = await handleApiRoutes(request, env, userId);
      if (apiResponse) return apiResponse;

      // 4. GitHub OAuth (D1-based, per-user)
      const ghResponse = await handleGitHubOAuth(request, env, userId);
      if (ghResponse) return ghResponse;

      // 5. Route to ChatAgent DO with userId header injected
      const headers = new Headers(request.headers);
      headers.set("x-user-id", userId);
      const agentReq = new Request(request.url, {
        method: request.method,
        headers,
        body: request.body
      });
      return (
        (await routeAgentRequest(agentReq, env)) ||
        new Response("Not found", { status: 404 })
      );
    }
  } satisfies ExportedHandler<Env>
);
