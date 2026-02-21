import { generateText, stepCountIs } from "ai";
import type { Bash } from "just-bash";
import type { UIMessage } from "ai";
import { createBashTool } from "./tools";

/**
 * Execute a scheduled/recurring task: run LLM with bash tool, persist results.
 */
export async function runScheduledTask(params: {
  messages: UIMessage[];
  persistMessages: (msgs: UIMessage[]) => Promise<void>;
  bash: Bash;
  ensureMounted: () => Promise<void>;
  model: Parameters<typeof generateText>[0]["model"];
  timezone: string;
  payload: { description: string; prompt: string; timezone?: string };
}): Promise<void> {
  const {
    messages,
    persistMessages,
    bash,
    ensureMounted,
    model,
    timezone,
    payload
  } = params;
  const now = new Date();
  const tz = payload.timezone || timezone;

  try {
    const result = await generateText({
      model,
      system:
        "You are a scheduled task executor. Execute the task and report the result.\n" +
        `Current UTC time: ${now.toISOString()}\n` +
        `User timezone: ${tz}`,
      prompt: payload.prompt,
      tools: {
        bash: createBashTool(bash, ensureMounted)
      },
      stopWhen: stepCountIs(10)
    });

    const userMsg = {
      id: crypto.randomUUID(),
      role: "user" as const,
      parts: [
        {
          type: "text" as const,
          text: `[Scheduled Task] ${new Date().toISOString()} - ${payload.description}`
        }
      ]
    };
    const assistantMsg = {
      id: crypto.randomUUID(),
      role: "assistant" as const,
      parts: [{ type: "text" as const, text: result.text }]
    };
    await persistMessages([...messages, userMsg, assistantMsg]);
  } catch (e) {
    console.error("runScheduledTask failed:", e);
    const errorMsg = {
      id: crypto.randomUUID(),
      role: "assistant" as const,
      parts: [
        {
          type: "text" as const,
          text: `[Scheduled Task Failed] ${new Date().toISOString()} - ${payload.description}\nError: ${e instanceof Error ? e.message : String(e)}`
        }
      ]
    };
    try {
      await persistMessages([...messages, errorMsg]);
    } catch (persistErr) {
      console.error("Failed to persist error message:", persistErr);
    }
    throw e;
  }
}
