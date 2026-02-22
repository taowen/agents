export interface QuotaCache {
  exceeded: boolean;
  checkedAt: number;
}

/**
 * Query per-hour usage data from DO SQLite (for /get-usage endpoint).
 */
export function queryUsageData(
  doSql: SqlStorage,
  since?: string | null
): unknown[] {
  const baseQuery = `SELECT
      strftime('%Y-%m-%dT%H', created_at) as hour,
      COALESCE(json_extract(message, '$.metadata.apiKeyType'), 'unknown') as api_key_type,
      COUNT(*) as request_count,
      SUM(json_extract(message, '$.metadata.usage.inputTokens')) as input_tokens,
      SUM(json_extract(message, '$.metadata.usage.cacheReadTokens')) as cache_read_tokens,
      SUM(json_extract(message, '$.metadata.usage.outputTokens')) as output_tokens
    FROM cf_ai_chat_agent_messages
    WHERE json_extract(message, '$.metadata.usage') IS NOT NULL
    GROUP BY hour, api_key_type`;
  const rows = since
    ? doSql.exec(baseQuery + ` HAVING hour >= ? ORDER BY hour`, since)
    : doSql.exec(baseQuery + ` ORDER BY hour`);
  return rows.toArray();
}

/**
 * Log diagnostic info about stored messages (for /get-usage debugging).
 */
export function logUsageDiagnostics(doSql: SqlStorage): void {
  const totalRows = doSql
    .exec(`SELECT COUNT(*) as cnt FROM cf_ai_chat_agent_messages`)
    .toArray();
  const totalCount = (totalRows[0] as { cnt: number })?.cnt ?? 0;

  const withUsage = doSql
    .exec(
      `SELECT COUNT(*) as cnt FROM cf_ai_chat_agent_messages WHERE json_extract(message, '$.metadata.usage') IS NOT NULL`
    )
    .toArray();
  const usageCount = (withUsage[0] as { cnt: number })?.cnt ?? 0;

  const sample = doSql
    .exec(
      `SELECT json_extract(message, '$.metadata') as meta FROM cf_ai_chat_agent_messages LIMIT 1`
    )
    .toArray();
  const sampleMeta =
    sample.length > 0 ? (sample[0] as { meta: string })?.meta : "NO_ROWS";

  console.log(
    `[get-usage] total_msgs=${totalCount} with_usage=${usageCount} sample_meta=${sampleMeta}`
  );
}

const QUOTA_CACHE_TTL = 30_000; // 30 seconds

/**
 * Check if a user has exceeded the builtin API key quota.
 * Returns an updated QuotaCache (with TTL-based caching).
 */
export async function checkQuota(
  db: D1Database,
  userId: string,
  cache: QuotaCache | null
): Promise<QuotaCache> {
  const now = Date.now();
  if (cache && now - cache.checkedAt < QUOTA_CACHE_TTL) {
    return cache;
  }
  const row = await db
    .prepare(`SELECT builtin_quota_exceeded_at FROM users WHERE id = ?`)
    .bind(userId)
    .first<{ builtin_quota_exceeded_at: string | null }>();
  return {
    exceeded: !!row?.builtin_quota_exceeded_at,
    checkedAt: now
  };
}

/**
 * Archive current hour's usage from DO SQLite → D1 usage_archive table.
 * Called in onFinish after each LLM stream completes.
 */
export async function archiveSessionUsage(
  db: D1Database,
  doSql: SqlStorage,
  userId: string,
  sessionId: string | null
): Promise<void> {
  const hour = new Date().toISOString().slice(0, 13);
  const hourRows = doSql
    .exec(
      `SELECT COALESCE(json_extract(message, '$.metadata.apiKeyType'), 'unknown') as api_key_type,
       COUNT(*) as request_count,
       SUM(json_extract(message, '$.metadata.usage.inputTokens')) as input_tokens,
       SUM(json_extract(message, '$.metadata.usage.cacheReadTokens')) as cache_read_tokens,
       SUM(json_extract(message, '$.metadata.usage.outputTokens')) as output_tokens
       FROM cf_ai_chat_agent_messages
       WHERE json_extract(message, '$.metadata.usage') IS NOT NULL
         AND strftime('%Y-%m-%dT%H', created_at) = ?
       GROUP BY api_key_type`,
      hour
    )
    .toArray() as {
    api_key_type: string;
    request_count: number;
    input_tokens: number;
    cache_read_tokens: number;
    output_tokens: number;
  }[];
  for (const row of hourRows) {
    db.prepare(
      `INSERT OR REPLACE INTO usage_archive (user_id, session_id, hour, api_key_type, request_count, input_tokens, cache_read_tokens, cache_write_tokens, output_tokens)
       VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)`
    )
      .bind(
        userId,
        sessionId,
        hour,
        row.api_key_type,
        row.request_count,
        row.input_tokens || 0,
        row.cache_read_tokens || 0,
        row.output_tokens || 0
      )
      .run()
      .catch((e: unknown) =>
        console.error("usage_archive D1 write failed:", e)
      );
  }
}
