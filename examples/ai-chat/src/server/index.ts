import * as Sentry from "@sentry/cloudflare";
import { routeAgentRequest } from "agents";
import { handleAuthRoutes, requireAuth, handleIncomingEmail } from "./auth";
import { handleApiRoutes } from "./api";
import { handleLlmRoutes } from "./llm-proxy";
import { handleGitHubOAuth } from "./github-oauth";

export { ChatAgent } from "./chat-agent";

const sentryHandler = Sentry.withSentry(
  (env: Env) => ({
    dsn: env.SENTRY_DSN,
    tracesSampleRate: 1.0
  }),
  {
    async fetch(request: Request, env: Env) {
      // 1. Public auth routes (Google OAuth + Email login)
      const authResponse = await handleAuthRoutes(request, env);
      if (authResponse) return authResponse;

      // 2. Require authentication for everything else
      const authResult = await requireAuth(request, env);
      if (authResult instanceof Response) return authResult;
      const userId = authResult;

      // 3. API routes (session/settings CRUD)
      const apiResponse = await handleApiRoutes(request, env, userId);
      if (apiResponse) return apiResponse;

      // 3b. LLM config route (returns full LLM config for client-side calls)
      const llmResponse = await handleLlmRoutes(request, env, userId);
      if (llmResponse) return llmResponse;

      // 4. GitHub OAuth (D1-based, per-user)
      const ghResponse = await handleGitHubOAuth(request, env, userId);
      if (ghResponse) return ghResponse;

      // 5. Route to ChatAgent DO with userId + sessionId headers injected
      const headers = new Headers(request.headers);
      headers.set("x-user-id", userId);
      const url = new URL(request.url);
      const nameMatch = url.pathname.match(/\/agents\/[^/]+\/([^/?]+)/);
      if (nameMatch) {
        headers.set("x-session-id", decodeURIComponent(nameMatch[1]));
      }
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

export default {
  ...sentryHandler,
  async email(message: ForwardableEmailMessage, env: Env) {
    await handleIncomingEmail(message, env);
  }
};
