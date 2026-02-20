import type { ChatMessage, ToolCall, ToolDefinition } from "./types";

const MAX_RETRIES = 2;
const RETRY_DELAY_MS = 2000;

export interface LlmResponse {
  content: string | null;
  toolCalls: ToolCall[] | null;
}

export class LlmClient {
  private apiUrl: string;
  private apiKey: string;
  private model: string;

  constructor(apiUrl: string, apiKey: string, model: string) {
    this.apiUrl = apiUrl;
    this.apiKey = apiKey;
    this.model = model;
  }

  async chat(
    messages: ChatMessage[],
    tools: ToolDefinition[],
    signal?: AbortSignal
  ): Promise<LlmResponse> {
    const body: Record<string, unknown> = {
      model: this.model,
      messages
    };
    if (tools.length > 0) {
      body.tools = tools;
    }

    const bodyStr = JSON.stringify(body);
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      if (attempt > 0) {
        await new Promise((r) => setTimeout(r, RETRY_DELAY_MS));
      }

      try {
        const response = await fetch(this.apiUrl, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${this.apiKey}`,
            "Content-Type": "application/json"
          },
          body: bodyStr,
          signal
        });

        if (!response.ok) {
          const errorBody = await response.text();
          lastError = new Error(
            `LLM API error ${response.status}: ${errorBody}`
          );
          if (
            (response.status === 429 || response.status >= 500) &&
            attempt < MAX_RETRIES
          ) {
            continue;
          }
          throw lastError;
        }

        const data = await response.json();
        const choice = data.choices[0];
        const message = choice.message;

        return {
          content: message.content ?? null,
          toolCalls:
            message.tool_calls && message.tool_calls.length > 0
              ? message.tool_calls
              : null
        };
      } catch (e: any) {
        if (e.name === "AbortError") throw e;
        lastError = e;
        if (attempt < MAX_RETRIES) continue;
        throw e;
      }
    }

    throw lastError ?? new Error("LLM request failed after retries");
  }
}
