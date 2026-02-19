import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import type { MountableFs } from "just-bash";
import type { LlmFileConfig } from "../client/api";

export type LlmConfigCache = {
  data: LlmFileConfig | null;
  fetchedAt: number;
} | null;

const SETTINGS_TTL = 60_000; // 60 seconds

export async function getCachedLlmConfig(
  mountableFs: MountableFs,
  cache: LlmConfigCache
): Promise<{ data: LlmFileConfig | null; cache: LlmConfigCache }> {
  const now = Date.now();
  if (cache && now - cache.fetchedAt < SETTINGS_TTL) {
    return { data: cache.data, cache };
  }
  try {
    const buf = await mountableFs.readFileBuffer("/etc/llm.json");
    const data = JSON.parse(new TextDecoder().decode(buf)) as LlmFileConfig;
    const newCache = { data, fetchedAt: now };
    return { data, cache: newCache };
  } catch {
    const newCache = { data: null, fetchedAt: now };
    return { data: null, cache: newCache };
  }
}

/**
 * Build the LLM model instance based on user settings.
 */
export function getLlmModel(env: Env, config: LlmFileConfig | null) {
  if (!config) {
    // Builtin: no /etc/llm.json means use built-in provider
    const apiKey = env.BUILTIN_LLM_API_KEY;
    const baseURL = env.BUILTIN_LLM_BASE_URL;
    const model = env.BUILTIN_LLM_MODEL;
    const provider = env.BUILTIN_LLM_PROVIDER;
    return provider === "openai-compatible"
      ? createOpenAICompatible({
          name: "llm",
          baseURL,
          apiKey,
          includeUsage: true
        })(model)
      : createGoogleGenerativeAI({ baseURL, apiKey })(model);
  }

  const { provider, api_key: apiKey, base_url: baseURL, model } = config;

  return provider === "openai-compatible"
    ? createOpenAICompatible({
        name: "llm",
        baseURL,
        apiKey,
        includeUsage: true
      })(model)
    : createGoogleGenerativeAI({ baseURL, apiKey })(model);
}
