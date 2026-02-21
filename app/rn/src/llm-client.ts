/**
 * LLM API client with retry logic for the Hermes agent runtime.
 * Extracted from agent-standalone.ts.
 */

import type { ChatMessage, ToolCall, LlmConfig } from "./types";
import type { ToolDefinition } from "./types";

// Declare host functions (provided by C++ runtime)
declare function http_post(
  url: string,
  headersJson: string,
  body: string
): string;
declare function llm_chat(body: string): string;
declare function sleep(ms: number): void;

const MAX_RETRIES = 2;
const RETRY_DELAY_MS = 2000;

export interface LlmResponse {
  content: string | null;
  toolCalls: ToolCall[] | null;
}

export function callLLM(
  messages: ChatMessage[],
  tools: ToolDefinition[] | unknown[],
  config: LlmConfig
): LlmResponse {
  const payload: Record<string, unknown> = {
    model: config.model,
    messages
  };
  if (tools && tools.length > 0) {
    payload.tools = tools;
  }
  const body = JSON.stringify(payload);

  // Use llm_chat (WebSocket via DeviceConnection) when available,
  // falling back to http_post for local/direct LLM calls
  const useLlmChat =
    typeof llm_chat === "function" &&
    (!config.baseURL || config.baseURL.includes("connect-screen.com"));

  let lastError = "";

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    if (attempt > 0) {
      sleep(RETRY_DELAY_MS);
    }

    let responseStr: string;
    if (useLlmChat) {
      responseStr = llm_chat(body);
    } else {
      let apiUrl = config.baseURL;
      if (apiUrl.endsWith("/")) apiUrl = apiUrl.slice(0, -1);
      apiUrl += "/chat/completions";
      const headers = JSON.stringify({
        Authorization: "Bearer " + config.apiKey,
        "Content-Type": "application/json"
      });
      responseStr = http_post(apiUrl, headers, body);
    }

    let data: any;
    try {
      data = JSON.parse(responseStr);
    } catch {
      lastError =
        "Failed to parse LLM response: " + responseStr.substring(0, 200);
      if (attempt < MAX_RETRIES) continue;
      throw new Error(lastError);
    }

    if (data.error) {
      lastError = "LLM API error: " + JSON.stringify(data.error);
      const status = data.error.status || data.error.code || 0;
      if ((status === 429 || status >= 500) && attempt < MAX_RETRIES) continue;
      throw new Error(lastError);
    }

    if (!data.choices || !data.choices[0]) {
      lastError =
        "LLM response missing choices: " + responseStr.substring(0, 200);
      if (attempt < MAX_RETRIES) continue;
      throw new Error(lastError);
    }

    const message = data.choices[0].message;
    return {
      content: message.content || null,
      toolCalls:
        message.tool_calls && message.tool_calls.length > 0
          ? message.tool_calls
          : null
    };
  }

  throw new Error(lastError || "LLM request failed after retries");
}
