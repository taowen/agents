/**
 * LLM config endpoint — returns the full LLM configuration (including API key)
 * for the authenticated user, so the client can call the LLM provider directly.
 */

export interface LlmConfig {
  provider: "openai-compatible" | "google";
  baseURL: string;
  apiKey: string;
  model: string;
}

async function readLlmConfig(
  db: D1Database,
  userId: string
): Promise<{
  provider: string;
  api_key: string;
  base_url: string;
  model: string;
} | null> {
  const row = await db
    .prepare(
      "SELECT CAST(content AS TEXT) as content FROM files WHERE user_id=? AND path=?"
    )
    .bind(userId, "/etc/llm.json")
    .first<{ content: string | null }>();
  if (!row?.content) return null;
  try {
    return JSON.parse(row.content);
  } catch {
    return null;
  }
}

export async function handleLlmRoutes(
  request: Request,
  env: Env,
  userId: string
): Promise<Response | null> {
  const url = new URL(request.url);

  // GET /api/llm/config — return full LLM config for client-side direct calls
  if (url.pathname === "/api/llm/config" && request.method === "GET") {
    const fileConfig = await readLlmConfig(env.DB, userId);

    let config: LlmConfig;

    if (!fileConfig) {
      // Builtin
      config = {
        provider: (env.BUILTIN_LLM_PROVIDER || "google") as
          | "openai-compatible"
          | "google",
        baseURL: env.BUILTIN_LLM_BASE_URL,
        apiKey: env.BUILTIN_LLM_API_KEY,
        model: env.BUILTIN_LLM_MODEL
      };
    } else if (fileConfig.provider === "openai-compatible") {
      config = {
        provider: "openai-compatible",
        baseURL: fileConfig.base_url || "",
        apiKey: fileConfig.api_key || "",
        model: fileConfig.model || ""
      };
    } else {
      // google
      config = {
        provider: "google",
        baseURL: fileConfig.base_url || "",
        apiKey: fileConfig.api_key || "",
        model: fileConfig.model || ""
      };
    }

    return Response.json(config);
  }

  return null;
}
