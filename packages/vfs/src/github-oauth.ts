/**
 * GitHub OAuth flow for git mounts.
 *
 * Worker-side: createGitHubOAuthRoutes() — handles /oauth/github/* routes
 * DO-side: handleGitHubOAuthDORequest() — handles /github-oauth-config and /store-github-token
 */

import type { AgentFS } from "agentfs-sdk/cloudflare";
import { upsertCredential } from "./git-credentials";

// ---- Deps interface ----

export interface GitHubOAuthDeps {
  fetchDO(agentId: string, path: string, init?: RequestInit): Promise<Response>;
  githubClientId?: string;
  githubClientSecret?: string;
}

// ---- OAuth CSRF helpers ----

interface OAuthConfig {
  clientId: string;
  clientSecret: string;
}

export async function generateOAuthState(
  agentId: string,
  secret: string
): Promise<string> {
  const payload = JSON.stringify({ agentId, ts: Date.now() });
  const payloadB64 = btoa(payload);
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const sig = await crypto.subtle.sign(
    "HMAC",
    key,
    new TextEncoder().encode(payloadB64)
  );
  const sigHex = Array.from(new Uint8Array(sig))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  return `${payloadB64}.${sigHex}`;
}

export async function verifyOAuthState(
  state: string,
  secret: string
): Promise<{ agentId: string; ts: number } | null> {
  const dotIdx = state.indexOf(".");
  if (dotIdx === -1) return null;
  const payloadB64 = state.slice(0, dotIdx);
  const sigHex = state.slice(dotIdx + 1);

  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["verify"]
  );
  const sigBytes = new Uint8Array(sigHex.length / 2);
  for (let i = 0; i < sigHex.length; i += 2) {
    sigBytes[i / 2] = parseInt(sigHex.slice(i, i + 2), 16);
  }
  const valid = await crypto.subtle.verify(
    "HMAC",
    key,
    sigBytes,
    new TextEncoder().encode(payloadB64)
  );
  if (!valid) return null;

  try {
    const payload = JSON.parse(atob(payloadB64));
    // Reject states older than 10 minutes
    if (Date.now() - payload.ts > 10 * 60 * 1000) return null;
    return payload;
  } catch (err) {
    console.error("Failed to parse OAuth state payload:", err);
    return null;
  }
}

// ---- Worker-side: OAuth route handler factory ----

async function getGitHubOAuthConfig(
  deps: GitHubOAuthDeps,
  agentId: string
): Promise<OAuthConfig | null> {
  // Try DO storage first
  try {
    const res = await deps.fetchDO(agentId, "/github-oauth-config");
    if (res.ok) {
      const config = (await res.json()) as Partial<OAuthConfig>;
      if (config.clientId && config.clientSecret) {
        return config as OAuthConfig;
      }
    }
  } catch (err) {
    console.error("Failed to fetch OAuth config from DO storage:", err);
  }

  // Fall back to env vars passed via deps
  if (deps.githubClientId && deps.githubClientSecret) {
    return {
      clientId: deps.githubClientId,
      clientSecret: deps.githubClientSecret
    };
  }

  return null;
}

export function createGitHubOAuthRoutes(
  deps: GitHubOAuthDeps
): (request: Request) => Promise<Response | null> {
  return async (request: Request): Promise<Response | null> => {
    const url = new URL(request.url);

    // Client-facing config read (never exposes secret)
    if (url.pathname === "/oauth/github/config" && request.method === "GET") {
      const agentId = url.searchParams.get("agent_id") || "default";
      const config = await getGitHubOAuthConfig(deps, agentId);
      return Response.json({
        clientId: config?.clientId || "",
        configured: !!(config?.clientId && config?.clientSecret)
      });
    }

    // Client-facing config write
    if (url.pathname === "/oauth/github/config" && request.method === "POST") {
      const agentId = url.searchParams.get("agent_id") || "default";
      const body = await request.json();
      const res = await deps.fetchDO(agentId, "/github-oauth-config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      return new Response(await res.text(), {
        status: res.status,
        headers: { "Content-Type": "application/json" }
      });
    }

    // OAuth initiation
    if (url.pathname === "/oauth/github" && request.method === "GET") {
      const agentId = url.searchParams.get("agent_id") || "default";
      const config = await getGitHubOAuthConfig(deps, agentId);
      if (!config) {
        return new Response("GitHub OAuth not configured", { status: 400 });
      }
      console.log(`OAuth: initiating GitHub flow for agent=${agentId}`);
      const state = await generateOAuthState(agentId, config.clientSecret);
      const redirectUri = `${url.origin}/oauth/github/callback`;
      const ghUrl = new URL("https://github.com/login/oauth/authorize");
      ghUrl.searchParams.set("client_id", config.clientId);
      ghUrl.searchParams.set("redirect_uri", redirectUri);
      ghUrl.searchParams.set("scope", "repo");
      ghUrl.searchParams.set("state", state);
      return Response.redirect(ghUrl.toString(), 302);
    }

    // OAuth callback
    if (url.pathname === "/oauth/github/callback" && request.method === "GET") {
      const code = url.searchParams.get("code");
      const state = url.searchParams.get("state");
      if (!code || !state) {
        return new Response("Missing code or state", { status: 400 });
      }

      // Extract agentId from state payload (unverified) to look up config
      let agentId: string;
      try {
        const dotIdx = state.indexOf(".");
        if (dotIdx === -1) throw new Error("invalid");
        agentId = JSON.parse(atob(state.slice(0, dotIdx))).agentId;
      } catch (err) {
        console.warn("OAuth callback: failed to parse state:", err);
        return new Response("Invalid state", { status: 400 });
      }

      const config = await getGitHubOAuthConfig(deps, agentId);
      if (!config) {
        return new Response("GitHub OAuth not configured", { status: 400 });
      }

      // Now verify HMAC with the stored secret
      const payload = await verifyOAuthState(state, config.clientSecret);
      if (!payload) {
        console.warn(
          `OAuth callback: invalid/expired state for agent=${agentId}`
        );
        return new Response("Invalid or expired state", { status: 403 });
      }

      // Exchange code for token
      const tokenRes = await fetch(
        "https://github.com/login/oauth/access_token",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json"
          },
          body: JSON.stringify({
            client_id: config.clientId,
            client_secret: config.clientSecret,
            code
          })
        }
      );
      const tokenData = (await tokenRes.json()) as {
        access_token?: string;
        error?: string;
      };
      if (!tokenData.access_token) {
        console.error(
          `OAuth callback: token exchange failed for agent=${payload.agentId}:`,
          tokenData.error || "unknown"
        );
        return new Response(
          `GitHub token exchange failed: ${tokenData.error || "unknown"}`,
          { status: 502 }
        );
      }

      // Store token in the DO via internal fetch
      await deps.fetchDO(payload.agentId, "/store-github-token", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: tokenData.access_token })
      });

      console.log(
        `OAuth: successfully completed flow for agent=${payload.agentId}`
      );
      return Response.redirect(url.origin + "/", 302);
    }

    return null;
  };
}

// ---- DO-side: OAuth request handler ----

async function readAgentFsJson(
  agentFs: AgentFS,
  path: string
): Promise<Record<string, string>> {
  try {
    const data = await agentFs.readFile(path);
    const text =
      typeof data === "string" ? data : new TextDecoder().decode(data);
    return JSON.parse(text);
  } catch (err) {
    // ENOENT is expected for first-time reads; log anything else
    if (err instanceof Error && !err.message.includes("ENOENT")) {
      console.error(`readAgentFsJson(${path}): unexpected error:`, err);
    }
    return {};
  }
}

export async function handleGitHubOAuthDORequest(
  request: Request,
  agentFs: AgentFS
): Promise<Response | null> {
  const url = new URL(request.url);

  // Read GitHub OAuth config (internal, used by worker)
  if (url.pathname === "/github-oauth-config" && request.method === "GET") {
    const config = await readAgentFsJson(agentFs, "/etc/github-oauth.json");
    return Response.json(config);
  }

  // Save GitHub OAuth config
  if (url.pathname === "/github-oauth-config" && request.method === "POST") {
    const body = (await request.json()) as {
      clientId?: string;
      clientSecret?: string;
    };
    const existing = await readAgentFsJson(agentFs, "/etc/github-oauth.json");
    if (body.clientId) existing.clientId = body.clientId;
    if (body.clientSecret) existing.clientSecret = body.clientSecret;
    await agentFs.writeFile("/etc/github-oauth.json", JSON.stringify(existing));
    return Response.json({ ok: true });
  }

  // Store GitHub OAuth token as git credential
  if (url.pathname === "/store-github-token" && request.method === "POST") {
    const { token } = (await request.json()) as { token: string };

    // Read existing /etc/git-credentials or start fresh
    let existing = "";
    try {
      const data = await agentFs.readFile("/etc/git-credentials");
      existing =
        typeof data === "string" ? data : new TextDecoder().decode(data);
    } catch (err) {
      console.warn(
        "store-github-token: git-credentials not found, creating new:",
        err
      );
    }

    const updated = upsertCredential(existing, {
      protocol: "https",
      host: "github.com",
      username: "oauth2",
      password: token
    });

    await agentFs.writeFile("/etc/git-credentials", updated);
    return new Response("OK", { status: 200 });
  }

  return null;
}
