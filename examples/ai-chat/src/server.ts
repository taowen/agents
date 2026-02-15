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
import { AgentFS } from "agentfs-sdk/cloudflare";
import { AgentFsAdapter } from "./agentfs-adapter";
import { Bash, InMemoryFs, MountableFs } from "just-bash";

/**
 * AI Chat Agent with sandboxed bash tool via just-bash.
 */
export class ChatAgent extends AIChatAgent {
  // Keep the last 200 messages in SQLite storage
  maxPersistedMessages = 200;

  private bash: Bash;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    const agentFs = AgentFS.create(ctx.storage);
    const homeFs = new AgentFsAdapter(agentFs);
    const fs = new MountableFs({ base: new InMemoryFs() });
    fs.mount("/home/user", homeFs);
    this.bash = new Bash({
      fs,
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

  async onChatMessage() {
    const google = createGoogleGenerativeAI({
      baseURL: "https://api.whatai.cc/v1beta",
      apiKey: this.env.GOOGLE_AI_API_KEY
    });

    const result = streamText({
      model: google("gemini-3-flash-preview"),
      system:
        "You are a helpful assistant. You can execute bash commands in a sandboxed virtual filesystem. " +
        "The bash environment supports common commands like ls, grep, awk, sed, find, cat, echo, curl, etc. " +
        "Use curl to fetch content from URLs. Files in /home/user persist across sessions (stored in durable storage). " +
        "Files outside /home/user only persist within the current session.",
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
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
