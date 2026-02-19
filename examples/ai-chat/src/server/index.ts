import * as Sentry from "@sentry/cloudflare";
import { routeAgentRequest } from "agents";
import { handleAuthRoutes, requireAuth, handleIncomingEmail } from "./auth";
import { handleApiRoutes } from "./api";
import { handleLlmRoutes } from "./llm-proxy";
import { handleGitHubOAuth } from "./github-oauth";

export { ChatAgent } from "./chat-agent";
export { BridgeManager } from "./bridge-manager";

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

      // 5. MCP OAuth callback — route directly to the DO by hex ID
      //    The request URL is forwarded as-is so origin+pathname match the stored callback_url.
      const url = new URL(request.url);
      if (url.pathname.startsWith("/mcp-callback/")) {
        const doId = url.pathname.split("/")[2];
        if (doId) {
          const id = env.ChatAgent.idFromString(doId);
          const stub = env.ChatAgent.get(id);
          return stub.fetch(request);
        }
      }

      // 6. Route to Agent DOs with userId header injected
      const headers = new Headers(request.headers);
      headers.set("x-user-id", userId);

      const agentMatch = url.pathname.match(
        /\/agents\/([^/]+)\/([^/?]+)(\/.*)?/
      );
      if (agentMatch) {
        const agentName = agentMatch[1];
        const restOfPath = agentMatch[3] || "";
        if (agentName === "bridge-manager") {
          // BridgeManager: DO name = userId only (shared across all sessions)
          url.pathname = `/agents/${agentName}/${encodeURIComponent(userId)}${restOfPath}`;
        } else {
          // ChatAgent: DO name = userId:sessionId for per-session isolation
          const sessionId = decodeURIComponent(agentMatch[2]);
          headers.set("x-session-id", sessionId);
          const isolatedName = encodeURIComponent(`${userId}:${sessionId}`);
          url.pathname = `/agents/${agentName}/${isolatedName}${restOfPath}`;
        }
      }
      const agentReq = new Request(url.toString(), {
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
