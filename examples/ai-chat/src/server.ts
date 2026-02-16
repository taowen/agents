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
import puppeteer from "@cloudflare/puppeteer";
import type { Browser, Page } from "@cloudflare/puppeteer";

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
  private browser: Browser | null = null;
  private page: Page | null = null;
  private browserCloseTimeout: ReturnType<typeof setTimeout> | null = null;

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

  private async getOrLaunchPage(): Promise<Page> {
    // Reset idle timer
    if (this.browserCloseTimeout) {
      clearTimeout(this.browserCloseTimeout);
      this.browserCloseTimeout = null;
    }
    this.browserCloseTimeout = setTimeout(
      () => this.closeBrowser(),
      5 * 60 * 1000
    );

    // Reuse existing if still valid
    if (this.browser?.isConnected() && this.page && !this.page.isClosed()) {
      return this.page;
    }

    // Clean up stale references
    await this.closeBrowser();

    this.browser = await puppeteer.launch(this.env.MYBROWSER, {
      keep_alive: 600000
    });
    this.page = await this.browser.newPage();
    await this.page.setUserAgent(
      "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    );
    await this.page.setViewport({ width: 1280, height: 720 });
    return this.page;
  }

  private async closeBrowser(): Promise<void> {
    if (this.browserCloseTimeout) {
      clearTimeout(this.browserCloseTimeout);
      this.browserCloseTimeout = null;
    }
    try {
      if (this.page && !this.page.isClosed()) await this.page.close();
    } catch {}
    try {
      if (this.browser?.isConnected()) await this.browser.close();
    } catch {}
    this.page = null;
    this.browser = null;
  }

  private async capturePageState(page: Page): Promise<{
    screenshot: string;
    url: string;
    title: string;
    text: string;
  }> {
    const screenshot = (await page.screenshot({
      encoding: "base64"
    })) as string;
    const title = await page.title();
    const url = page.url();
    const text = await page.evaluate(() => {
      const body = document.body;
      if (!body) return "";
      return body.innerText.slice(0, 8000);
    });
    return { screenshot, url, title, text };
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
        "Unmount with: umount /mnt/<repo-name>. " +
        "You also have a browser tool for browsing real web pages. " +
        "Use the browser tool when you need to interact with SPAs, JavaScript-rendered content, or pages that curl can't handle well. " +
        "The browser tool supports actions: goto (navigate to URL), click (click an element by CSS selector), " +
        "type (type text into an input by CSS selector), screenshot (capture current page), " +
        "scroll (scroll up or down), extract (extract text from page or specific element), " +
        "set_cookies (inject cookies for authentication - user provides cookie JSON), close (close browser). " +
        "Each browser action returns a screenshot so you can see the page. " +
        "For sites requiring login: the user can export cookies from their own browser and provide them. " +
        "Use set_cookies with the cookies JSON, then goto the target URL to access as the authenticated user. " +
        "Prefer curl for simple requests; use the browser for complex web pages that need JavaScript rendering.",
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
        }),
        browser: tool({
          description:
            "Browse web pages using a real browser. Supports navigation, clicking, typing, screenshots, scrolling, and text extraction. " +
            "Use this for JavaScript-heavy pages, SPAs, or when you need to interact with a page. " +
            "Each action returns a screenshot of the current page state.",
          inputSchema: z.object({
            action: z
              .enum([
                "goto",
                "click",
                "type",
                "screenshot",
                "scroll",
                "extract",
                "set_cookies",
                "close"
              ])
              .describe("The browser action to perform"),
            url: z
              .string()
              .optional()
              .describe("URL to navigate to (for goto action)"),
            selector: z
              .string()
              .optional()
              .describe(
                "CSS selector for the target element (for click, type, extract actions)"
              ),
            text: z
              .string()
              .optional()
              .describe("Text to type (for type action)"),
            direction: z
              .enum(["up", "down"])
              .optional()
              .describe("Scroll direction (for scroll action)"),
            cookies: z
              .string()
              .optional()
              .describe(
                "JSON array of cookie objects for set_cookies action. Each cookie needs: name, value, domain. Optional: path, expires, httpOnly, secure, sameSite."
              )
          }),
          execute: async ({
            action,
            url,
            selector,
            text,
            direction,
            cookies
          }) => {
            try {
              if (action === "close") {
                await this.closeBrowser();
                return {
                  action,
                  success: true,
                  url: "",
                  title: "",
                  text: "Browser closed",
                  screenshot: ""
                };
              }

              const page = await this.getOrLaunchPage();

              switch (action) {
                case "goto": {
                  if (!url)
                    return {
                      action,
                      success: false,
                      error: "url is required for goto action",
                      url: "",
                      title: "",
                      text: "",
                      screenshot: ""
                    };
                  await page.goto(url, {
                    waitUntil: "networkidle0",
                    timeout: 30000
                  });
                  const state = await this.capturePageState(page);
                  return { action, success: true, ...state };
                }
                case "click": {
                  if (!selector)
                    return {
                      action,
                      success: false,
                      error: "selector is required for click action",
                      url: "",
                      title: "",
                      text: "",
                      screenshot: ""
                    };
                  await page.waitForSelector(selector, { timeout: 5000 });
                  await page.click(selector);
                  try {
                    await page.waitForNetworkIdle({ timeout: 5000 });
                  } catch {}
                  const state = await this.capturePageState(page);
                  return { action, success: true, ...state };
                }
                case "type": {
                  if (!selector)
                    return {
                      action,
                      success: false,
                      error: "selector is required for type action",
                      url: "",
                      title: "",
                      text: "",
                      screenshot: ""
                    };
                  if (!text)
                    return {
                      action,
                      success: false,
                      error: "text is required for type action",
                      url: "",
                      title: "",
                      text: "",
                      screenshot: ""
                    };
                  await page.waitForSelector(selector, { timeout: 5000 });
                  await page.click(selector, { clickCount: 3 });
                  await page.type(selector, text);
                  const state = await this.capturePageState(page);
                  return { action, success: true, ...state };
                }
                case "screenshot": {
                  const state = await this.capturePageState(page);
                  return { action, success: true, ...state };
                }
                case "scroll": {
                  const scrollDir = direction === "up" ? -500 : 500;
                  await page.evaluate((d) => window.scrollBy(0, d), scrollDir);
                  await new Promise((r) => setTimeout(r, 500));
                  const state = await this.capturePageState(page);
                  return { action, success: true, ...state };
                }
                case "extract": {
                  let extracted: string;
                  if (selector) {
                    extracted = await page.$eval(selector, (el) =>
                      (el as HTMLElement).innerText.slice(0, 16000)
                    );
                  } else {
                    extracted = await page.evaluate(() =>
                      document.body.innerText.slice(0, 16000)
                    );
                  }
                  const state = await this.capturePageState(page);
                  return { action, success: true, ...state, text: extracted };
                }
                case "set_cookies": {
                  if (!cookies)
                    return {
                      action,
                      success: false,
                      error: "cookies is required for set_cookies action",
                      url: "",
                      title: "",
                      text: "",
                      screenshot: ""
                    };
                  const parsed = JSON.parse(cookies);
                  const cookieArray = Array.isArray(parsed) ? parsed : [parsed];
                  await page.setCookie(...cookieArray);
                  const state = await this.capturePageState(page);
                  return {
                    action,
                    success: true,
                    ...state,
                    text: `Set ${cookieArray.length} cookie(s)`
                  };
                }
                default:
                  return {
                    action,
                    success: false,
                    error: `Unknown action: ${action}`,
                    url: "",
                    title: "",
                    text: "",
                    screenshot: ""
                  };
              }
            } catch (err) {
              const errorMsg = err instanceof Error ? err.message : String(err);
              // Try to capture current state even on error
              let screenshot = "";
              let pageUrl = "";
              let pageTitle = "";
              try {
                if (this.page && !this.page.isClosed()) {
                  const state = await this.capturePageState(this.page);
                  screenshot = state.screenshot;
                  pageUrl = state.url;
                  pageTitle = state.title;
                }
              } catch {}
              return {
                action,
                success: false,
                error: errorMsg,
                url: pageUrl,
                title: pageTitle,
                text: "",
                screenshot
              };
            }
          },
          toModelOutput: (output) => {
            const result = output as {
              action: string;
              success: boolean;
              url: string;
              title: string;
              text: string;
              screenshot: string;
              error?: string;
            };
            const parts: Array<{ type: string; [key: string]: unknown }> = [];
            // Text summary
            const lines: string[] = [];
            lines.push(`Action: ${result.action} | Success: ${result.success}`);
            if (result.error) lines.push(`Error: ${result.error}`);
            if (result.url) lines.push(`URL: ${result.url}`);
            if (result.title) lines.push(`Title: ${result.title}`);
            if (result.text) lines.push(`Text:\n${result.text}`);
            parts.push({ type: "text", text: lines.join("\n") });
            // Screenshot as image
            if (result.screenshot) {
              parts.push({
                type: "image-data",
                data: result.screenshot,
                mediaType: "image/png"
              });
            }
            return { type: "content", value: parts };
          }
        })
      },
      stopWhen: stepCountIs(10)
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
