import { Agent } from "pi";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import {
  createBrowserState,
  createBrowserTool,
  closeBrowser
} from "./browser-tool.ts";
import type { AgentEvent } from "pi";
import * as Sentry from "@sentry/cloudflare";

const SYSTEM_PROMPT = `You are a browser automation agent. You have a "browser" tool that controls a headless Chrome browser.

AVAILABLE ACTIONS:
- goto: Navigate to a URL. Requires "url" parameter.
- click: Click an element. Requires "selector" (CSS selector).
- type: Type text into an element. Requires "selector" and "text".
- screenshot: Take a screenshot of the current page.
- scroll: Scroll the page. Optional "direction" ("up" or "down", default "down").
- extract: Extract text from page or element. Optional "selector" for specific element.
- set_cookies: Set cookies. Requires "cookies" (JSON string).
- close: Close the browser.

WORKFLOW:
1. Start by navigating to a URL with goto.
2. After each action you receive a screenshot and page text — use these to decide the next step.
3. Use CSS selectors to interact with elements. Inspect the page text to find appropriate selectors.
4. When the task is complete, summarize your findings and close the browser.
5. If a page has dynamic content, wait for it to load by taking a screenshot after navigation.

TIPS:
- Use simple, robust CSS selectors (tag names, classes, IDs).
- If a click doesn't work, try a different selector or scroll to make the element visible.
- Extract text when you need to read content from the page.
- Always close the browser when you're done.`;

const MAX_TURNS = 30;

interface Env {
  MYBROWSER: Fetcher;
  LLM_BASE_URL: string;
  LLM_API_KEY: string;
  LLM_MODEL: string;
  SENTRY_DSN: string;
}

export default Sentry.withSentry(
  (env: Env) => ({
    dsn: env.SENTRY_DSN,
    tracesSampleRate: 1.0
  }),
  {
    async fetch(
      request: Request,
      env: Env,
      ctx: ExecutionContext
    ): Promise<Response> {
      const url = new URL(request.url);

      if (url.pathname === "/api/agent" && request.method === "POST") {
        return handleAgentRequest(request, env, ctx);
      }

      if (url.pathname === "/api/report" && request.method === "POST") {
        return handleReportRequest(request);
      }

      return new Response("Not found", { status: 404 });
    }
  } satisfies ExportedHandler<Env>
);

async function handleAgentRequest(
  request: Request,
  env: Env,
  ctx: ExecutionContext
): Promise<Response> {
  const body = (await request.json()) as { task: string };
  const task = body.task;
  if (!task) {
    return new Response(JSON.stringify({ error: "task is required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" }
    });
  }

  const provider = createOpenAICompatible({
    name: "llm",
    baseURL: env.LLM_BASE_URL,
    apiKey: env.LLM_API_KEY
  });
  const model = provider.chatModel(env.LLM_MODEL);

  const browserState = createBrowserState();
  const browserTool = createBrowserTool(browserState, env.MYBROWSER);

  const agent = new Agent({
    initialState: {
      systemPrompt: SYSTEM_PROMPT,
      model,
      tools: [browserTool]
    }
  });

  let turnCount = 0;

  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();
  const encoder = new TextEncoder();

  function sendEvent(event: string, data: unknown) {
    const json = JSON.stringify(data);
    writer.write(encoder.encode(`event: ${event}\ndata: ${json}\n\n`));
  }

  const unsubscribe = agent.subscribe((event: AgentEvent) => {
    switch (event.type) {
      case "message_start":
      case "message_update":
      case "message_end": {
        const msg = event.message;
        if (
          msg.role === "assistant" &&
          "content" in msg &&
          Array.isArray(msg.content)
        ) {
          const textParts = msg.content
            .filter((c: any) => c.type === "text")
            .map((c: any) => c.text)
            .join("");
          if (textParts) {
            sendEvent(event.type, {
              role: "assistant",
              text: textParts
            });
            if (event.type === "message_end") {
              Sentry.addBreadcrumb({
                category: "agent",
                message: textParts.slice(0, 200),
                level: "info"
              });
            }
          }
        }
        break;
      }
      case "tool_execution_start": {
        const args = event.args as Record<string, unknown> | undefined;
        sendEvent("tool_execution_start", {
          toolCallId: event.toolCallId,
          toolName: event.toolName,
          args
        });
        Sentry.addBreadcrumb({
          category: "tool",
          message: `${event.toolName}.${args?.action ?? "unknown"}`,
          data: {
            url: args?.url,
            selector: args?.selector
          },
          level: "info"
        });
        break;
      }
      case "tool_execution_end":
        sendEvent("tool_execution_end", {
          toolCallId: event.toolCallId,
          toolName: event.toolName,
          result: event.result,
          isError: event.isError
        });
        Sentry.addBreadcrumb({
          category: "tool",
          message: `${event.toolName} ${event.isError ? "failed" : "succeeded"}`,
          level: event.isError ? "error" : "info"
        });
        break;
      case "turn_end":
        turnCount++;
        Sentry.addBreadcrumb({
          category: "agent",
          message: `Turn ${turnCount} ended`,
          level: "info"
        });
        if (turnCount >= MAX_TURNS) {
          agent.abort();
          sendEvent("error", { message: `Max turns (${MAX_TURNS}) reached` });
        }
        break;
      case "agent_end":
        sendEvent("agent_end", {});
        Sentry.addBreadcrumb({
          category: "agent",
          message: "Agent finished",
          level: "info"
        });
        break;
    }
  });

  Sentry.setTag("llm_model", env.LLM_MODEL);
  Sentry.setTag("task", task.slice(0, 100));

  // Run the agent loop, then clean up
  const runAgent = async () => {
    try {
      await Sentry.startSpan(
        { name: "agent.prompt", op: "ai.agent" },
        (span) => {
          const traceHeader = Sentry.spanToTraceHeader(span);
          sendEvent("trace_context", { sentryTrace: traceHeader });
          return agent.prompt(task);
        }
      );
    } catch (err) {
      Sentry.captureException(err);
      const msg = err instanceof Error ? err.message : String(err);
      sendEvent("error", { message: msg });
    } finally {
      unsubscribe();
      await closeBrowser(browserState);
      writer.close();
    }
  };

  // Use waitUntil so the request doesn't get killed if the client disconnects
  ctx.waitUntil(runAgent());

  return new Response(readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive"
    }
  });
}

async function handleReportRequest(request: Request): Promise<Response> {
  const body = (await request.json()) as {
    description?: string;
    task?: string;
    events?: unknown[];
    sentryTrace?: string;
  };

  const description = body.description || "No description provided";
  const err = new Error(`Bug Report: ${description}`);
  err.name = "UserBugReport";

  const captureOptions: Sentry.ExclusiveEventHintOrCaptureContext = {
    tags: {
      source: "user_bug_report",
      ...(body.sentryTrace
        ? { agent_trace_id: body.sentryTrace.split("-")[0] }
        : {})
    },
    extra: {
      userDescription: description,
      task: body.task,
      recentEvents: JSON.stringify(body.events)
    }
  };

  let eventId: string;
  if (body.sentryTrace) {
    eventId = Sentry.continueTrace(
      { sentryTrace: body.sentryTrace, baggage: undefined },
      () => Sentry.captureException(err, captureOptions)
    );
  } else {
    eventId = Sentry.captureException(err, captureOptions);
  }

  return new Response(JSON.stringify({ reportId: eventId }), {
    headers: { "Content-Type": "application/json" }
  });
}
