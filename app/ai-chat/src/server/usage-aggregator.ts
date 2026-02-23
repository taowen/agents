/**
 * Usage data aggregation: D1 archive queries + DO polling + merge.
 * Extracted from api.ts to keep route handlers thin.
 */

import * as Sentry from "@sentry/cloudflare";
import { listSessions } from "./db";
import { upsertSessionSnapshot } from "./usage-archive";

export type UsageRow = {
  hour: string;
  api_key_type: string;
  request_count: number;
  input_tokens: number | null;
  cache_read_tokens: number | null;
  cache_write_tokens: number | null;
  output_tokens: number | null;
};

type MergedRow = {
  hour: string;
  api_key_type: string;
  request_count: number;
  input_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  output_tokens: number;
};

export async function cacheSessionUsage(
  db: D1Database,
  userId: string,
  sessionId: string,
  rows: UsageRow[]
): Promise<void> {
  if (rows.length === 0) return;
  const stmts = rows.map((r) =>
    upsertSessionSnapshot(
      db,
      userId,
      sessionId,
      r.hour,
      r.api_key_type || "unknown",
      r.request_count,
      r.input_tokens || 0,
      r.cache_read_tokens || 0,
      r.cache_write_tokens || 0,
      r.output_tokens || 0
    )
  );
  for (let i = 0; i < stmts.length; i += 100) {
    await db.batch(stmts.slice(i, i + 100));
  }
}

export async function fetchDoUsage(
  env: Env,
  userId: string,
  sessionId: string,
  since?: string
): Promise<UsageRow[]> {
  const isolatedName = encodeURIComponent(`${userId}:${sessionId}`);
  const doId = env.ChatAgent.idFromName(isolatedName);
  const stub = env.ChatAgent.get(doId);
  const url = since
    ? `http://agent/get-usage?since=${encodeURIComponent(since)}`
    : "http://agent/get-usage";
  const res = await stub.fetch(
    new Request(url, {
      headers: { "x-partykit-room": isolatedName }
    })
  );
  return res.ok ? ((await res.json()) as UsageRow[]) : [];
}

/**
 * Aggregate usage data across D1 archive (for deleted sessions)
 * and live DO instances (for active sessions).
 *
 * Returns merged rows sorted by hour + api_key_type.
 * Also fires cache-update writes via waitUntil.
 */
export async function aggregateUsage(
  env: Env,
  db: D1Database,
  userId: string,
  start: string,
  end: string,
  waitUntil: (p: Promise<unknown>) => void
): Promise<MergedRow[]> {
  const sessions = await listSessions(db, userId);
  const activeIds = sessions.map((s) => s.id);

  console.log(
    `[usage] userId=${userId} sessions=${activeIds.length} start=${start} end=${end}`
  );
  Sentry.addBreadcrumb({
    category: "usage",
    message: `sessions=${activeIds.length} start=${start} end=${end}`,
    level: "info"
  });

  // 1-2. D1 archive queries
  const maxHours = new Map<string, string>();
  let archivedResults: MergedRow[] = [];

  try {
    // 1. Query max cached hour per active session
    if (activeIds.length > 0) {
      const placeholders = activeIds.map(() => "?").join(",");
      const cached = await db
        .prepare(
          `SELECT session_id, MAX(hour) as max_hour FROM usage_archive
         WHERE user_id = ? AND session_id IN (${placeholders})
         GROUP BY session_id`
        )
        .bind(userId, ...activeIds)
        .all<{ session_id: string; max_hour: string }>();
      for (const r of cached.results) {
        maxHours.set(r.session_id, r.max_hour);
      }
    }

    // 2. D1 archive — exclude active sessions (their data comes fresh from DO)
    let archivedQuery = `SELECT hour, api_key_type, SUM(request_count) as request_count,
       SUM(input_tokens) as input_tokens, SUM(cache_read_tokens) as cache_read_tokens,
       SUM(cache_write_tokens) as cache_write_tokens, SUM(output_tokens) as output_tokens
       FROM usage_archive WHERE user_id = ? AND hour >= ? AND hour <= ?`;
    const archivedBinds: unknown[] = [userId, start, end];
    if (activeIds.length > 0) {
      const ph = activeIds.map(() => "?").join(",");
      archivedQuery += ` AND session_id NOT IN (${ph})`;
      archivedBinds.push(...activeIds);
    }
    archivedQuery += ` GROUP BY hour, api_key_type`;
    const archived = await db
      .prepare(archivedQuery)
      .bind(...archivedBinds)
      .all<MergedRow>();
    archivedResults = archived.results;
    console.log(
      `[usage] D1 archive: ${archivedResults.length} rows, maxHours=${maxHours.size} sessions cached`
    );
    Sentry.addBreadcrumb({
      category: "usage",
      message: `D1 archive: ${archivedResults.length} rows, maxHours=${maxHours.size}`,
      level: "info"
    });
  } catch (e) {
    console.error(
      "D1 usage_archive query failed (continuing with DO data):",
      e
    );
    Sentry.captureException(e);
  }

  // 3. Incremental fetch from active DOs (only >= max cached hour)
  const activeResults = await Promise.allSettled(
    sessions.map(async (s) => {
      const since = maxHours.get(s.id);
      const rows = await fetchDoUsage(env, userId, s.id, since);
      return { sessionId: s.id, rows };
    })
  );

  // 3b. Log each DO result
  for (const [i, r] of activeResults.entries()) {
    if (r.status === "fulfilled") {
      console.log(`[usage] DO ${sessions[i].id}: ${r.value.rows.length} rows`);
    } else {
      console.error(`[usage] DO ${sessions[i].id} FAILED:`, r.reason);
    }
  }
  Sentry.addBreadcrumb({
    category: "usage",
    message: `DO fetches: ${activeResults.filter((r) => r.status === "fulfilled").length}/${activeResults.length} succeeded`,
    level: "info"
  });

  // 4. Fire-and-forget: cache incremental data to D1
  const cachePromises: Promise<void>[] = [];
  for (const r of activeResults) {
    if (r.status === "rejected") {
      console.error("fetchDoUsage failed:", r.reason);
      continue;
    }
    cachePromises.push(
      cacheSessionUsage(db, userId, r.value.sessionId, r.value.rows)
    );
  }
  waitUntil(Promise.allSettled(cachePromises));

  // 5. Merge: D1 archive (non-active) + fresh DO data
  const map = new Map<string, MergedRow>();

  const addToMap = (
    hour: string,
    apiKeyType: string,
    rc: number,
    it: number,
    crt: number,
    cwt: number,
    ot: number
  ) => {
    const key = `${hour}|${apiKeyType}`;
    const existing = map.get(key);
    if (existing) {
      existing.request_count += rc;
      existing.input_tokens += it;
      existing.cache_read_tokens += crt;
      existing.cache_write_tokens += cwt;
      existing.output_tokens += ot;
    } else {
      map.set(key, {
        hour,
        api_key_type: apiKeyType,
        request_count: rc,
        input_tokens: it,
        cache_read_tokens: crt,
        cache_write_tokens: cwt,
        output_tokens: ot
      });
    }
  };

  for (const row of archivedResults) {
    addToMap(
      row.hour,
      row.api_key_type || "unknown",
      row.request_count || 0,
      row.input_tokens || 0,
      row.cache_read_tokens || 0,
      row.cache_write_tokens || 0,
      row.output_tokens || 0
    );
  }

  for (const result of activeResults) {
    if (result.status !== "fulfilled") continue;
    for (const row of result.value.rows) {
      if (row.hour < start || row.hour > end) continue;
      addToMap(
        row.hour,
        row.api_key_type || "unknown",
        row.request_count || 0,
        row.input_tokens || 0,
        row.cache_read_tokens || 0,
        row.cache_write_tokens || 0,
        row.output_tokens || 0
      );
    }
  }

  const merged = [...map.values()].sort(
    (a, b) =>
      a.hour.localeCompare(b.hour) ||
      a.api_key_type.localeCompare(b.api_key_type)
  );

  console.log(
    `[usage] merged: ${merged.length} hours, total_input=${merged.reduce((s, r) => s + r.input_tokens, 0)}`
  );
  Sentry.addBreadcrumb({
    category: "usage",
    message: `merged: ${merged.length} hours`,
    level: "info"
  });

  return merged;
}
