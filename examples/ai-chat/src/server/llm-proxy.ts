/**
 * LLM config endpoint — returns the full LLM configuration (including API key)
 * for the authenticated user, so the client can call the LLM provider directly.
 */

import { getSettings } from "./db";

export interface LlmConfig {
  provider: "openai-compatible" | "google";
  baseURL: string;
  apiKey: string;
  model: string;
}

export async function handleLlmRoutes(
  request: Request,
  env: Env,
  userId: string
): Promise<Response | null> {
  const url = new URL(request.url);

  // GET /api/llm/config — return full LLM config for client-side direct calls
  if (url.pathname === "/api/llm/config" && request.method === "GET") {
    const settings = await getSettings(env.DB, userId);
    const llmProvider = settings?.llm_provider ?? "builtin";

    let config: LlmConfig;

    if (llmProvider === "builtin") {
      config = {
        provider: "openai-compatible",
        baseURL: "https://ark.cn-beijing.volces.com/api/v3",
        apiKey: env.ARK_API_KEY,
        model: "doubao-seed-2-0-pro-260215"
      };
    } else if (llmProvider === "openai-compatible") {
      config = {
        provider: "openai-compatible",
        baseURL: settings?.llm_base_url || "",
        apiKey: settings?.llm_api_key || "",
        model: settings?.llm_model || ""
      };
    } else {
      // google
      config = {
        provider: "google",
        baseURL:
          settings?.llm_base_url ||
          "https://generativelanguage.googleapis.com/v1beta",
        apiKey: settings?.llm_api_key || env.GOOGLE_AI_API_KEY,
        model: settings?.llm_model || "gemini-2.0-flash"
      };
    }

    return Response.json(config);
  }

  return null;
}
