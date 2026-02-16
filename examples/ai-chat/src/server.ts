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
  syncDirtyGitMounts,
  createGitHubOAuthRoutes,
  handleGitHubOAuthDORequest
} from "vfs";
import { Bash, InMemoryFs, MountableFs } from "just-bash";

import {
  createBrowserState,
  createBrowserTool,
  type BrowserState
} from "./browser-tool";

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
  private browserState: BrowserState;

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

    this.browserState = createBrowserState();
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
          console.error("ensureMounted: fstab mount failed:", err);
          this.mountPromise = null;
          throw err;
        }
      );
    }
    await this.mountPromise;
  }

  async onRequest(request: Request): Promise<Response> {
    const oauthRes = await handleGitHubOAuthDORequest(request, this.agentFs);
    if (oauthRes) return oauthRes;
    return super.onRequest(request);
  }

  async onChatMessage() {
    console.log("onChatMessage: processing new message");
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
        browser: createBrowserTool(this.browserState, this.env.MYBROWSER)
      },
      stopWhen: stepCountIs(10)
    });

    return result.toUIMessageStreamResponse();
  }
}

export default {
  async fetch(request: Request, env: Env) {
    // Handle OAuth routes before agent routing
    const handleOAuth = createGitHubOAuthRoutes({
      fetchDO(agentId, path, init) {
        const doId = env.ChatAgent.idFromName(agentId);
        const stub = env.ChatAgent.get(doId);
        const headers = new Headers(init?.headers);
        headers.set("x-partykit-room", agentId);
        return stub.fetch(
          new Request(`https://do${path}`, { ...init, headers })
        );
      },
      githubClientId: env.GITHUB_CLIENT_ID,
      githubClientSecret: env.GITHUB_CLIENT_SECRET
    });
    const oauthResponse = await handleOAuth(request);
    if (oauthResponse) return oauthResponse;

    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
