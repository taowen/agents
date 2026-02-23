/**
 * Centralized usage_archive D1 writes.
 *
 * Two distinct write semantics exist:
 * - Snapshot (INSERT OR REPLACE): overwrites the entire row for a session+hour.
 *   Used by DO-side archival (usage-tracker) and aggregator caching.
 * - Incremental (INSERT … ON CONFLICT DO UPDATE += excluded): atomically
 *   accumulates counters. Used by the stateless LLM proxy endpoint.
 */

/**
 * Upsert a full snapshot row — overwrites any previous data for this
 * (user, session, hour, apiKeyType) tuple.
 */
export function upsertSessionSnapshot(
  db: D1Database,
  userId: string,
  sessionId: string | null,
  hour: string,
  apiKeyType: string,
  requestCount: number,
  inputTokens: number,
  cacheReadTokens: number,
  cacheWriteTokens: number,
  outputTokens: number
): D1PreparedStatement {
  return db
    .prepare(
      `INSERT OR REPLACE INTO usage_archive (user_id, session_id, hour, api_key_type, request_count, input_tokens, cache_read_tokens, cache_write_tokens, output_tokens)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
    )
    .bind(
      userId,
      sessionId,
      hour,
      apiKeyType,
      requestCount,
      inputTokens,
      cacheReadTokens,
      cacheWriteTokens,
      outputTokens
    );
}

/**
 * Increment proxy usage counters — adds to existing row or inserts a new one.
 * session_id is always '__proxy__' for proxy-originated usage.
 */
export function incrementProxyUsage(
  db: D1Database,
  userId: string,
  hour: string,
  apiKeyType: string,
  inputTokens: number,
  cacheReadTokens: number,
  outputTokens: number
): D1PreparedStatement {
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
    .bind(userId, hour, apiKeyType, inputTokens, cacheReadTokens, outputTokens);
}
