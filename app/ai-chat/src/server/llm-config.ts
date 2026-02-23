import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import type { MountableFs } from "just-bash";
import type { LlmFileConfig } from "../client/api";

export type LlmConfigCache = {
  data: LlmFileConfig | null;
  fetchedAt: number;
} | null;

/**
 * Resolved LLM configuration — shared between DO-side (streamText) and
 * Worker-side (proxy endpoint). Produced by `resolveFromRaw()`.
 */
export interface ResolvedLlmConfig {
  baseURL: string;
  apiKey: string;
  model: string;
  provider: string;
  apiKeyType: "builtin" | "custom";
}

/**
 * Resolve raw /etc/llm.json content (or null for builtin) into a
 * canonical config object. This is the single source of truth for
 * "builtin vs custom" determination and field extraction.
 */
export function resolveFromRaw(
  raw: LlmFileConfig | null,
  env: {
    BUILTIN_LLM_BASE_URL: string;
    BUILTIN_LLM_API_KEY: string;
    BUILTIN_LLM_MODEL: string;
    BUILTIN_LLM_PROVIDER?: string;
  }
): ResolvedLlmConfig {
  if (raw && raw.base_url && raw.api_key) {
    return {
      baseURL: raw.base_url,
      apiKey: raw.api_key,
      model: raw.model || env.BUILTIN_LLM_MODEL,
      provider: raw.provider || "openai-compatible",
      apiKeyType: "custom"
    };
  }

  return {
    baseURL: env.BUILTIN_LLM_BASE_URL,
    apiKey: env.BUILTIN_LLM_API_KEY,
    model: env.BUILTIN_LLM_MODEL,
    provider: env.BUILTIN_LLM_PROVIDER || "openai-compatible",
    apiKeyType: "builtin"
  };
}

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
 * Build the LLM model instance from a ResolvedLlmConfig.
 */
function buildModel(cfg: ResolvedLlmConfig) {
  return cfg.provider === "openai-compatible"
    ? createOpenAICompatible({
        name: "llm",
        baseURL: cfg.baseURL,
        apiKey: cfg.apiKey,
        includeUsage: true
      })(cfg.model)
    : createGoogleGenerativeAI({
        baseURL: cfg.baseURL,
        apiKey: cfg.apiKey
      })(cfg.model);
}

/**
 * Build the LLM model instance based on user settings.
 */
export function getLlmModel(env: Env, config: LlmFileConfig | null) {
  return buildModel(resolveFromRaw(config, env));
}
