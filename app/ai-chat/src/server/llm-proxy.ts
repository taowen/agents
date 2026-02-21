/**
 * LLM proxy: config resolution, quota check, upstream call, usage archival.
 * Extracted from api.ts to keep route handlers thin.
 */

export interface LlmProxyResult {
  baseURL: string;
  apiKey: string;
  model: string;
  apiKeyType: "builtin" | "custom";
}

/**
 * Resolve LLM config for a user: prefer custom /etc/llm.json over builtin env vars.
 */
export async function resolveLlmConfig(
  db: D1Database,
  userId: string,
  env: {
    BUILTIN_LLM_BASE_URL: string;
    BUILTIN_LLM_API_KEY: string;
    BUILTIN_LLM_MODEL: string;
  }
): Promise<LlmProxyResult> {
  const llmRow = await db
    .prepare("SELECT content FROM files WHERE user_id = ? AND path = ?")
    .bind(userId, "/etc/llm.json")
    .first<{ content: ArrayBuffer | null }>();

  if (llmRow?.content) {
    try {
      const cfg = JSON.parse(new TextDecoder().decode(llmRow.content));
      if (cfg.base_url && cfg.api_key) {
        return {
          baseURL: cfg.base_url,
          apiKey: cfg.api_key,
          model: cfg.model || env.BUILTIN_LLM_MODEL,
          apiKeyType: "custom"
        };
      }
    } catch {}
  }

  return {
    baseURL: env.BUILTIN_LLM_BASE_URL,
    apiKey: env.BUILTIN_LLM_API_KEY,
    model: env.BUILTIN_LLM_MODEL,
    apiKeyType: "builtin"
  };
}

/**
 * Check if a user's builtin quota is exceeded. Returns error response or null.
 */
export async function checkProxyQuota(
  db: D1Database,
  userId: string
): Promise<{ exceeded: boolean }> {
  const quotaRow = await db
    .prepare(`SELECT builtin_quota_exceeded_at FROM users WHERE id = ?`)
    .bind(userId)
    .first<{ builtin_quota_exceeded_at: string | null }>();
  return { exceeded: !!quotaRow?.builtin_quota_exceeded_at };
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
  config: LlmProxyResult,
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
  return db
    .prepare(
      `INSERT INTO usage_archive (user_id, session_id, hour, api_key_type, request_count, input_tokens, cache_read_tokens, cache_write_tokens, output_tokens)
       VALUES (?, '__proxy__', ?, ?, 1, ?, ?, 0, ?)
       ON CONFLICT(user_id, session_id, hour, api_key_type) DO UPDATE SET
         request_count = request_count + 1,
         input_tokens = input_tokens + excluded.input_tokens,
         cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens,
         output_tokens = output_tokens + excluded.output_tokens`
    )
    .bind(
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
