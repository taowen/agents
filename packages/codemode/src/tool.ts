import { tool, type Tool } from "ai";
import { z } from "zod";
import type { ToolSet } from "ai";
import { generateTypes, type ToolDescriptors } from "./types";
import { createCodeExecutor } from "./executor";

const DEFAULT_DESCRIPTION = `Execute code to achieve a goal.

Available:
{{types}}

Write an async arrow function that returns the result.`;

export interface CreateCodeToolOptions {
  tools: ToolDescriptors | ToolSet;
  loader: WorkerLoader;
  /**
   * Optional filter for outbound requests from sandboxed code.
   * Tool calls (codemode://) are handled automatically.
   * Return a Response to allow/modify, or null to block.
   */
  onFetch?: (request: Request) => Promise<Response | null> | Response | null;
  /**
   * Custom tool description. Use {{types}} as a placeholder for the generated type definitions.
   */
  description?: string;
}

const codeSchema = z.object({
  code: z.string().describe("JavaScript async arrow function to execute")
});

type CodeInput = z.infer<typeof codeSchema>;
type CodeOutput = { code: string; result: unknown };

/**
 * Create a codemode tool that allows LLMs to write and execute code
 * with access to your tools in a sandboxed environment.
 *
 * Returns an AI SDK compatible tool.
 */
export function experimental_createCodeTool(
  options: CreateCodeToolOptions
): Tool<CodeInput, CodeOutput> {
  const types = generateTypes(options.tools);
  const tools = options.tools;

  // Create globalOutbound that intercepts tool calls
  const globalOutbound = {
    fetch: async (
      input: RequestInfo | URL,
      init?: RequestInit
    ): Promise<Response> => {
      const url = new URL(
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.href
            : input.url
      );

      // Handle tool calls via codemode:// protocol
      if (url.protocol === "codemode:") {
        const toolName = url.pathname.replace(/^\//, "");
        const tool = tools[toolName];

        if (!tool) {
          return Response.json(
            { error: `Tool "${toolName}" not found` },
            { status: 404 }
          );
        }

        try {
          const args = init?.body ? JSON.parse(init.body as string) : {};
          const execute =
            "execute" in tool
              ? (tool.execute as (args: unknown) => Promise<unknown>)
              : undefined;

          if (!execute) {
            return Response.json(
              { error: `Tool "${toolName}" has no execute function` },
              { status: 400 }
            );
          }

          const result = await execute(args);
          return Response.json({ result });
        } catch (err) {
          return Response.json(
            { error: err instanceof Error ? err.message : String(err) },
            { status: 500 }
          );
        }
      }

      // For other requests, use onFetch filter or block
      if (options.onFetch) {
        const request = new Request(input, init);
        const response = await options.onFetch(request);
        return response ?? new Response("Blocked", { status: 403 });
      }

      // Default: block all non-tool requests
      return new Response("Blocked", { status: 403 });
    }
  };

  const executor = createCodeExecutor({
    loader: options.loader,
    globalOutbound: globalOutbound as Fetcher
  });

  const description = (options.description ?? DEFAULT_DESCRIPTION).replace(
    "{{types}}",
    types
  );

  return tool({
    description,
    inputSchema: codeSchema,
    execute: async ({ code }) => {
      const result = await executor(code);
      return { code, result };
    }
  });
}
