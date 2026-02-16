import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { routeAgentRequest } from "agents";
import { AIChatAgent } from "@cloudflare/ai-chat";
import {
  streamText,
  convertToModelMessages,
  pruneMessages,
  tool,
  stepCountIs
} from "ai";
import { z } from "zod";
import { AgentFS, type CloudflareStorage } from "agentfs-sdk/cloudflare";
import {
  AgentFsAdapter,
  mountFstabEntries,
  createMountCommands,
  upsertCredential,
  syncDirtyGitMounts
} from "vfs";
import { Bash, InMemoryFs, MountableFs } from "just-bash";

// ---- OAuth CSRF helpers ----

async function generateOAuthState(
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

async function verifyOAuthState(
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
  } catch {
    return null;
  }
}

// ---- OAuth config helpers ----

interface OAuthConfig {
  clientId: string;
  clientSecret: string;
}

/** Fetch a DO stub with the required partyserver room header. */
function fetchDO(
  env: Env,
  agentId: string,
  path: string,
  init?: RequestInit
): Promise<Response> {
  const doId = env.ChatAgent.idFromName(agentId);
  const stub = env.ChatAgent.get(doId);
  const headers = new Headers(init?.headers);
  headers.set("x-partykit-room", agentId);
  return stub.fetch(new Request(`https://do${path}`, { ...init, headers }));
}

/** Read GitHub OAuth config from DO storage, falling back to env vars. */
async function getGitHubOAuthConfig(
  env: Env,
  agentId: string
): Promise<OAuthConfig | null> {
  // Try DO storage first
  try {
    const res = await fetchDO(env, agentId, "/github-oauth-config");
    if (res.ok) {
      const config = (await res.json()) as Partial<OAuthConfig>;
      if (config.clientId && config.clientSecret) {
        return config as OAuthConfig;
      }
    }
  } catch {
    /* fall through */
  }

  // Fall back to env vars
  if (env.GITHUB_CLIENT_ID && env.GITHUB_CLIENT_SECRET) {
    return {
      clientId: env.GITHUB_CLIENT_ID,
      clientSecret: env.GITHUB_CLIENT_SECRET
    };
  }

  return null;
}

// ---- OAuth route handler ----

async function handleOAuthRoutes(
  request: Request,
  env: Env
): Promise<Response | null> {
  const url = new URL(request.url);

  // Client-facing config read (never exposes secret)
  if (url.pathname === "/oauth/github/config" && request.method === "GET") {
    const agentId = url.searchParams.get("agent_id") || "default";
    const config = await getGitHubOAuthConfig(env, agentId);
    return Response.json({
      clientId: config?.clientId || "",
      configured: !!(config?.clientId && config?.clientSecret)
    });
  }

  // Client-facing config write
  if (url.pathname === "/oauth/github/config" && request.method === "POST") {
    const agentId = url.searchParams.get("agent_id") || "default";
    const body = await request.json();
    const res = await fetchDO(env, agentId, "/github-oauth-config", {
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
    const config = await getGitHubOAuthConfig(env, agentId);
    if (!config) {
      return new Response("GitHub OAuth not configured", { status: 400 });
    }
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
    } catch {
      return new Response("Invalid state", { status: 400 });
    }

    const config = await getGitHubOAuthConfig(env, agentId);
    if (!config) {
      return new Response("GitHub OAuth not configured", { status: 400 });
    }

    // Now verify HMAC with the stored secret
    const payload = await verifyOAuthState(state, config.clientSecret);
    if (!payload) {
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
      return new Response(
        `GitHub token exchange failed: ${tokenData.error || "unknown"}`,
        { status: 502 }
      );
    }

    // Store token in the DO via internal fetch
    await fetchDO(env, payload.agentId, "/store-github-token", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: tokenData.access_token })
    });

    return Response.redirect(url.origin + "/", 302);
  }

  return null;
}

/**
 * AI Chat Agent with sandboxed bash tool via just-bash.
 */
export class ChatAgent extends AIChatAgent {
  // Keep the last 200 messages in SQLite storage
  maxPersistedMessages = 200;

  private bash: Bash;
  private agentFs: AgentFS;
  private mountableFs: MountableFs;
  private mounted = false;
  private mountPromise: Promise<void> | null = null;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    const agentFs = AgentFS.create(ctx.storage as unknown as CloudflareStorage);
    this.agentFs = agentFs;

    const inMemoryFs = new InMemoryFs();
    inMemoryFs.mkdirSync("/mnt");
    const fs = new MountableFs({ base: inMemoryFs });
    this.mountableFs = fs;

    // /etc is always mounted (hardcoded, persistent via AgentFS)
    const etcFs = new AgentFsAdapter(agentFs, "/etc");
    fs.mount("/etc", etcFs);

    this.bash = new Bash({
      fs,
      customCommands: createMountCommands(fs, agentFs),
      cwd: "/home/user",
      network: { dangerouslyAllowFullInternetAccess: true },
      executionLimits: {
        maxCommandCount: 1000,
        maxLoopIterations: 1000,
        maxCallDepth: 50,
        maxStringLength: 1_048_576
      }
    });
  }

  /**
   * Ensure fstab-declared mounts are applied. Called once before first bash exec.
   * Concurrent-safe: multiple calls share a single promise.
   */
  private async ensureMounted(): Promise<void> {
    if (this.mounted) return;
    if (!this.mountPromise) {
      this.mountPromise = mountFstabEntries(
        this.agentFs,
        this.mountableFs
      ).then(
        () => {
          this.mounted = true;
        },
        (err) => {
          this.mountPromise = null;
          throw err;
        }
      );
    }
    await this.mountPromise;
  }

  private async readAgentFsJson(path: string): Promise<Record<string, string>> {
    try {
      const data = await this.agentFs.readFile(path);
      const text =
        typeof data === "string" ? data : new TextDecoder().decode(data);
      return JSON.parse(text);
    } catch {
      return {};
    }
  }

  async onRequest(request: Request): Promise<Response> {
    const url = new URL(request.url);

    // Read GitHub OAuth config (internal, used by worker)
    if (url.pathname === "/github-oauth-config" && request.method === "GET") {
      const config = await this.readAgentFsJson("/etc/github-oauth.json");
      return Response.json(config);
    }

    // Save GitHub OAuth config
    if (url.pathname === "/github-oauth-config" && request.method === "POST") {
      const body = (await request.json()) as {
        clientId?: string;
        clientSecret?: string;
      };
      const existing = await this.readAgentFsJson("/etc/github-oauth.json");
      if (body.clientId) existing.clientId = body.clientId;
      if (body.clientSecret) existing.clientSecret = body.clientSecret;
      await this.agentFs.writeFile(
        "/etc/github-oauth.json",
        JSON.stringify(existing)
      );
      return Response.json({ ok: true });
    }

    // Store GitHub OAuth token as git credential
    if (url.pathname === "/store-github-token" && request.method === "POST") {
      const { token } = (await request.json()) as { token: string };

      // Read existing /etc/git-credentials or start fresh
      let existing = "";
      try {
        const data = await this.agentFs.readFile("/etc/git-credentials");
        existing =
          typeof data === "string" ? data : new TextDecoder().decode(data);
      } catch {
        // file doesn't exist yet
      }

      const updated = upsertCredential(existing, {
        protocol: "https",
        host: "github.com",
        username: "oauth2",
        password: token
      });

      await this.agentFs.writeFile("/etc/git-credentials", updated);
      return new Response("OK", { status: 200 });
    }

    return super.onRequest(request);
  }

  async onChatMessage() {
    await this.ensureMounted();

    const google = createGoogleGenerativeAI({
      baseURL: "https://api.whatai.cc/v1beta",
      apiKey: this.env.GOOGLE_AI_API_KEY
    });

    const result = streamText({
      model: google("gemini-3-flash-preview"),
      system:
        "You are a helpful assistant with a sandboxed virtual bash environment (not a real Linux shell). " +
        "Available commands: ls, cat, grep, awk, sed, find, echo, mkdir, cp, mv, rm, sort, uniq, wc, head, tail, " +
        "curl, diff, jq, base64, tree, du, df, stat, file, tr, cut, paste, date, uname, id, uptime, hostname, whoami, " +
        "mount (no args shows mounts), and more. Use `help` to list all commands. " +
        "NOT available: git, apt, npm, pip, python, node, tar, gzip, ssh, wget, docker, sudo, " +
        "and any package managers or compilers. " +
        "There are no /proc, /sys, or /dev filesystems. " +
        "Use curl to fetch content from URLs. " +
        "Use `mount` (no args) to see current mounts, `df` to see filesystem info. " +
        "Files in /home/user and /etc persist across sessions (stored in durable storage). " +
        "Files outside these directories only persist within the current session. " +
        "/etc/fstab controls what gets mounted on startup. " +
        "To add a persistent git mount, append to /etc/fstab: " +
        'echo "https://github.com/user/repo  /mnt/repo  git  ref=main,depth=1  0  0" >> /etc/fstab ' +
        "(it will be mounted on the next session). " +
        "You can also mount git repos dynamically for the current session: " +
        "mkdir -p /mnt/<repo-name> && mount -t git <url> /mnt/<repo-name>. " +
        "IMPORTANT: Always mount under /mnt/<name>, never directly to /mnt itself. " +
        "Do NOT mount inside /home/user as it would conflict with persistent storage. " +
        "Options via -o: ref (branch/tag, default main), depth (clone depth, default 1), username, password. " +
        "For private repos: mount -t git -o username=user,password=token <url> /mnt/<repo-name>. " +
        "If a GitHub account is connected (via the GitHub button in the header), " +
        "private GitHub repos are automatically authenticated when mounting. " +
        "Git repos mounted via mount -t git are read-write. " +
        "Any file changes are automatically committed and pushed after each command. " +
        "If a GitHub account is connected, authentication is automatic. " +
        "Unmount with: umount /mnt/<repo-name>.",
      // Prune old tool calls and reasoning to save tokens on long conversations
      messages: pruneMessages({
        messages: await convertToModelMessages(this.messages),
        toolCalls: "before-last-2-messages",
        reasoning: "before-last-message"
      }),
      tools: {
        bash: tool({
          description:
            "Execute a bash command in a sandboxed virtual filesystem. " +
            "Supports ls, grep, awk, sed, find, cat, echo, mkdir, cp, mv, sort, uniq, wc, head, tail, curl, and more. " +
            "Use curl to fetch content from URLs. Files persist across commands within the session.",
          inputSchema: z.object({
            command: z.string().describe("The bash command to execute")
          }),
          execute: async ({ command }) => {
            const result = await this.bash.exec(command);
            // Auto-commit & push any git mount changes
            try {
              await syncDirtyGitMounts(this.mountableFs, command);
            } catch (e) {
              result.stderr += `\ngit sync error: ${e instanceof Error ? e.message : e}`;
            }
            return {
              stdout: result.stdout,
              stderr: result.stderr,
              exitCode: result.exitCode
            };
          }
        })
      },
      stopWhen: stepCountIs(5)
    });

    return result.toUIMessageStreamResponse();
  }
}

export default {
  async fetch(request: Request, env: Env) {
    // Handle OAuth routes before agent routing
    const oauthResponse = await handleOAuthRoutes(request, env);
    if (oauthResponse) return oauthResponse;

    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
