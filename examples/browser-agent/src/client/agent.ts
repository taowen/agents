import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import type { ModelMessage } from "ai";
import { streamText, stepCountIs, tool } from "ai";
import { Bash, InMemoryFs } from "just-bash";
import { z } from "zod";

export interface AgentCallbacks {
  onTextDelta: (delta: string) => void;
  onToolCall: (command: string) => void;
  onToolResult: (result: {
    stdout: string;
    stderr: string;
    exitCode: number;
  }) => void;
}

const fs = new InMemoryFs();
const bash = new Bash({ fs, cwd: "/home" });

const bashTool = tool({
  description: "Execute a bash command in the browser-side virtual filesystem",
  inputSchema: z.object({
    command: z.string().describe("The bash command to execute")
  }),
  execute: async ({ command }) => {
    const result = await bash.exec(command);
    return {
      stdout: result.stdout,
      stderr: result.stderr,
      exitCode: result.exitCode
    };
  }
});

const history: ModelMessage[] = [];

let modelId: string | undefined;

async function getModel() {
  if (!modelId) {
    const resp = await fetch("/api/config");
    const config = (await resp.json()) as { model: string };
    modelId = config.model;
  }
  const provider = createOpenAICompatible({
    name: "proxy",
    baseURL: window.location.origin + "/api/v1"
  });
  return provider(modelId);
}

export async function chat(userMessage: string, callbacks: AgentCallbacks) {
  history.push({ role: "user", content: userMessage });

  const model = await getModel();

  const result = streamText({
    model,
    system:
      "You are a helpful assistant with access to a bash shell running in the user's browser. " +
      "The filesystem is an in-memory virtual filesystem. You can create files, run commands, etc. " +
      "The working directory is /home.",
    messages: history,
    tools: { bash: bashTool },
    stopWhen: stepCountIs(10)
  });

  for await (const event of result.fullStream) {
    switch (event.type) {
      case "text-delta":
        callbacks.onTextDelta(event.text);
        break;
      case "tool-call":
        if (event.toolName === "bash") {
          callbacks.onToolCall((event.input as { command: string }).command);
        }
        break;
      case "tool-result":
        if (event.toolName === "bash") {
          callbacks.onToolResult(
            event.output as { stdout: string; stderr: string; exitCode: number }
          );
        }
        break;
    }
  }

  // Append all assistant/tool messages from all steps to history
  const response = await result.response;
  history.push(...response.messages);
}

export function resetAgent() {
  history.length = 0;
}
