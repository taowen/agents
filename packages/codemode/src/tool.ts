import { tool, type Tool } from "ai";
import { z } from "zod";
import type { ToolSet } from "ai";
import { generateTypes, type ToolDescriptors } from "./types";
import { createCodeExecutor } from "./executor";
import type { CodeModeProxy } from "./proxy";

const DEFAULT_DESCRIPTION = `Execute code to achieve a goal.

Available:
{{types}}

Write an async arrow function that returns the result.`;

export interface CreateCodeToolOptions {
  tools: ToolDescriptors | ToolSet;
  loader: WorkerLoader;
  /**
   * Service binding to CodeModeProxy.
   * Create with: ctx.exports.CodeModeProxy({ props: { binding, name, callback } })
   */
  proxy: Fetcher<CodeModeProxy>;
  /**
   * Optional outbound fetch handler to filter requests from sandboxed code.
   * Set to null to block all outbound requests (default).
   * Provide a Fetcher to allow filtered outbound requests.
   */
  globalOutbound?: Fetcher | null;
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

  const executor = createCodeExecutor({
    loader: options.loader,
    proxy: options.proxy,
    globalOutbound: options.globalOutbound
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
