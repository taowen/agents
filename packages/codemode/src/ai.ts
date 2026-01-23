import { generateObject, tool, type ToolSet } from "ai";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";
import { generateTypes } from "./types";
import { createCodeExecutor } from "./executor";
import type { CodeModeProxy } from "./proxy";

export { CodeModeProxy } from "./proxy";

/**
 * @deprecated Use `experimental_createCodeTool` from `@cloudflare/codemode` instead.
 * This function makes an internal LLM call which is unnecessary - your LLM can write the code directly.
 */
export async function experimental_codemode(options: {
  tools: ToolSet;
  prompt: string;
  globalOutbound: Fetcher;
  loader: WorkerLoader;
  proxy: Fetcher<CodeModeProxy>;
}): Promise<{
  prompt: string;
  tools: ToolSet;
}> {
  const generatedTypes = generateTypes(options.tools);
  const prompt = `${options.prompt}
  You are a helpful assistant. You have access to the "codemode" tool that can do different things:

  ${getToolDescriptions(options.tools)}

  If the user asks to do anything that be achieveable by the codemode tool, then simply pass over control to it by giving it a simple function description. Don't be too verbose.

  `;

  const executor = createCodeExecutor({
    loader: options.loader,
    proxy: options.proxy,
    globalOutbound: options.globalOutbound
  });

  const codemodeTool = tool({
    description: "codemode: a tool that can generate code to achieve a goal",
    inputSchema: z.object({
      functionDescription: z.string()
    }),
    outputSchema: z.object({
      code: z.string(),
      result: z.any()
    }),
    execute: async ({ functionDescription }) => {
      const response = await generateObject({
        model: openai("gpt-4.1"),
        schema: z.object({
          code: z.string()
        }),
        prompt: `You are a code generating machine.

      In addition to regular javascript, you can also use the following functions:

      ${generatedTypes}

      Respond only with the code, nothing else. Output javascript code.

      Generate an async function that achieves the goal. This async function doesn't accept any arguments.

      Here is user input: ${functionDescription}`
      });

      const result = await executor(response.object.code);
      return { code: response.object.code, result };
    }
  });

  return { prompt, tools: { codemode: codemodeTool } };
}

function getToolDescriptions(tools: ToolSet) {
  return Object.entries(tools)
    .map(([_toolName, tool]) => {
      return `\n- ${tool.description?.trim()}`;
    })
    .join("");
}
