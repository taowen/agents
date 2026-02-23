/**
 * LLM proxy: config resolution, quota check, upstream call, usage archival.
 * Extracted from api.ts to keep route handlers thin.
 */

import { resolveFromRaw, type ResolvedLlmConfig } from "./llm-config";
import type { LlmFileConfig } from "../client/api";
import { incrementProxyUsage } from "./usage-archive";

export type { ResolvedLlmConfig as LlmProxyResult };

/**
 * Resolve LLM config for a user: read /etc/llm.json from D1, then
 * delegate to the shared resolveFromRaw() for builtin-vs-custom logic.
 */
export async function resolveLlmConfig(
  db: D1Database,
  userId: string,
  env: {
    BUILTIN_LLM_BASE_URL: string;
    BUILTIN_LLM_API_KEY: string;
    BUILTIN_LLM_MODEL: string;
    BUILTIN_LLM_PROVIDER?: string;
  }
): Promise<ResolvedLlmConfig> {
  let raw: LlmFileConfig | null = null;

  const llmRow = await db
    .prepare("SELECT content FROM files WHERE user_id = ? AND path = ?")
    .bind(userId, "/etc/llm.json")
    .first<{ content: ArrayBuffer | null }>();

  if (llmRow?.content) {
    try {
      raw = JSON.parse(
        new TextDecoder().decode(llmRow.content)
      ) as LlmFileConfig;
    } catch {}
  }

  return resolveFromRaw(raw, env);
}

type UpstreamResponseBody = {
  choices?: { message?: { content?: string; tool_calls?: unknown } }[];
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    prompt_tokens_details?: { cached_tokens?: number };
  };
};

/**
 * Forward a chat completion request to the upstream LLM.
 * Returns either the parsed JSON response or an error Response.
 */
export async function callUpstreamLlm(
  config: ResolvedLlmConfig,
  body: Record<string, unknown>
): Promise<
  { ok: true; body: UpstreamResponseBody } | { ok: false; response: Response }
> {
  const upstreamRes = await fetch(`${config.baseURL}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.apiKey}`
    },
    body: JSON.stringify({
      ...body,
      model: body.model || config.model
    })
  });

  if (!upstreamRes.ok) {
    const errText = await upstreamRes.text();
    return {
      ok: false,
      response: new Response(errText, {
        status: upstreamRes.status,
        headers: { "Content-Type": "application/json" }
      })
    };
  }

  const responseBody = (await upstreamRes.json()) as UpstreamResponseBody;
  return { ok: true, body: responseBody };
}

/**
 * Write proxy usage to D1 usage_archive (fire-and-forget).
 */
export function archiveProxyUsage(
  db: D1Database,
  userId: string,
  apiKeyType: string,
  responseBody: UpstreamResponseBody
): Promise<void> | undefined {
  const choice = responseBody.choices?.[0]?.message;
  if (!choice) return;

  const inputTokens = responseBody.usage?.prompt_tokens || 0;
  const outputTokens = responseBody.usage?.completion_tokens || 0;
  const cacheReadTokens =
    responseBody.usage?.prompt_tokens_details?.cached_tokens || 0;

  const proxyHour = new Date().toISOString().slice(0, 13);
  return incrementProxyUsage(
    db,
    userId,
    proxyHour,
    apiKeyType,
    inputTokens,
    cacheReadTokens,
    outputTokens
  )
    .run()
    .then(() => {})
    .catch((e: unknown) =>
      console.error("proxy usage_archive write failed:", e)
    );
}
