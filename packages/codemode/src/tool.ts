import { z } from "zod";
import type { ToolSet } from "ai";
import { generateTypes, type ToolDescriptors } from "./types";
import { createCodeExecutor, type CodeExecutorOptions } from "./executor";

const DEFAULT_DESCRIPTION = `Execute code to achieve a goal.

Available:
{{types}}

Write an async arrow function that returns the result.`;

export interface CreateCodeToolOptions extends CodeExecutorOptions {
  tools: ToolDescriptors | ToolSet;
  /**
   * Custom tool description. Use {{types}} as a placeholder for the generated type definitions.
   * @default "Execute code to achieve a goal.\n\nAvailable:\n{{types}}\n\nWrite an async arrow function that returns the result."
   */
  description?: string;
}

export const codeToolInputSchema = z.object({
  code: z.string().describe("JavaScript async arrow function to execute")
});

export interface CodeToolResult {
  code: string;
  result: unknown;
}

export interface CodeTool {
  description: string;
  inputSchema: typeof codeToolInputSchema;
  execute: (input: { code: string }) => Promise<CodeToolResult>;
}

/**
 * Create a codemode tool that allows LLMs to write and execute code
 * with access to your tools in a sandboxed environment.
 *
 * Returns a framework-agnostic tool object that you can adapt to your LLM framework.
 */
export function createCodeTool(options: CreateCodeToolOptions): CodeTool {
  const types = generateTypes(options.tools);
  const execute = createCodeExecutor({
    loader: options.loader,
    proxy: options.proxy
  });

  const description = (options.description ?? DEFAULT_DESCRIPTION).replace(
    "{{types}}",
    types
  );

  return {
    description,
    inputSchema: codeToolInputSchema,
    execute: async ({ code }) => {
      const result = await execute(code);
      return { code, result };
    }
  };
}
