/**
 * Quota and usage HTTP API integration tests.
 *
 * Tests the full HTTP path: worker.fetch → Hono auth middleware → apiRoutes → D1.
 * Covers GET /api/quota and GET /api/usage endpoints.
 */
import { env } from "cloudflare:test";
import { describe, it, expect, beforeAll } from "vitest";
import { applyD1Schema, createTestUser, apiRequest } from "./test-utils";
import { QUOTA_LIMITS } from "../ai-chat/src/server/quota-config";

describe("Quota and usage API", () => {
  let db: D1Database;

  beforeAll(async () => {
    db = (env as unknown as { DB: D1Database }).DB;
    await applyD1Schema(db);
  });

  it("GET /api/quota returns zero usage for new user", async () => {
    const userId = await createTestUser(db);
    const res = await apiRequest("GET", "/api/quota", userId);
    expect(res.status).toBe(200);

    const data = await res.json<{
      exceeded: boolean;
      exceededAt: string | null;
      hourly: {
        requests: number;
        tokens: number;
        requestLimit: number;
        tokenLimit: number;
      };
      daily: {
        requests: number;
        tokens: number;
        requestLimit: number;
        tokenLimit: number;
      };
    }>();

    expect(data.exceeded).toBe(false);
    expect(data.exceededAt).toBeNull();
    expect(data.hourly.requests).toBe(0);
    expect(data.hourly.tokens).toBe(0);
    expect(data.hourly.requestLimit).toBe(QUOTA_LIMITS.HOURLY_REQUEST_LIMIT);
    expect(data.hourly.tokenLimit).toBe(QUOTA_LIMITS.HOURLY_TOKEN_LIMIT);
    expect(data.daily.requests).toBe(0);
    expect(data.daily.tokens).toBe(0);
    expect(data.daily.requestLimit).toBe(QUOTA_LIMITS.DAILY_REQUEST_LIMIT);
    expect(data.daily.tokenLimit).toBe(QUOTA_LIMITS.DAILY_TOKEN_LIMIT);
  });

  it("GET /api/quota reflects exceeded state from D1", async () => {
    const userId = await createTestUser(db);

    // Manually set quota exceeded flag in D1
    await db
      .prepare(
        "UPDATE users SET builtin_quota_exceeded_at = datetime('now') WHERE id = ?"
      )
      .bind(userId)
      .run();

    const res = await apiRequest("GET", "/api/quota", userId);
    expect(res.status).toBe(200);

    const data = await res.json<{
      exceeded: boolean;
      exceededAt: string | null;
    }>();
    expect(data.exceeded).toBe(true);
    expect(data.exceededAt).not.toBeNull();
  });

  it("GET /api/quota reflects usage from usage_archive", async () => {
    const userId = await createTestUser(db);
    const now = new Date();
    const currentHour = now.toISOString().slice(0, 13);

    // Insert usage data directly into D1
    await db
      .prepare(
        `INSERT INTO usage_archive (user_id, session_id, hour, api_key_type, request_count, input_tokens, output_tokens)
         VALUES (?, ?, ?, 'builtin', 5, 1000, 500)`
      )
      .bind(userId, "test-session", currentHour)
      .run();

    const res = await apiRequest("GET", "/api/quota", userId);
    expect(res.status).toBe(200);

    const data = await res.json<{
      hourly: { requests: number; tokens: number };
      daily: { requests: number; tokens: number };
    }>();
    expect(data.hourly.requests).toBe(5);
    expect(data.hourly.tokens).toBe(1500); // input + output
    expect(data.daily.requests).toBe(5);
    expect(data.daily.tokens).toBe(1500);
  });

  it("GET /api/usage returns empty array for new user", async () => {
    const userId = await createTestUser(db);
    const res = await apiRequest("GET", "/api/usage", userId);
    expect(res.status).toBe(200);

    const data = await res.json<unknown[]>();
    expect(Array.isArray(data)).toBe(true);
  });

  it("GET /api/usage returns archived usage data", async () => {
    const userId = await createTestUser(db);
    const now = new Date();
    const currentHour = now.toISOString().slice(0, 13);

    // Insert usage archive for a deleted session (no active session = pure D1 data)
    await db
      .prepare(
        `INSERT INTO usage_archive (user_id, session_id, hour, api_key_type, request_count, input_tokens, output_tokens)
         VALUES (?, ?, ?, 'builtin', 10, 2000, 1000)`
      )
      .bind(userId, "deleted-session", currentHour)
      .run();

    const res = await apiRequest(
      "GET",
      `/api/usage?start=${currentHour}&end=${currentHour}`,
      userId
    );
    expect(res.status).toBe(200);

    const data = await res.json<
      Array<{
        hour: string;
        api_key_type: string;
        request_count: number;
        input_tokens: number;
        output_tokens: number;
      }>
    >();
    expect(data.length).toBeGreaterThanOrEqual(1);
    const row = data.find(
      (r) => r.hour === currentHour && r.api_key_type === "builtin"
    );
    expect(row).toBeDefined();
    expect(row!.request_count).toBeGreaterThanOrEqual(10);
  });

  it("unauthenticated request returns 401", async () => {
    const res = await apiRequest("GET", "/api/quota", "");
    expect(res.status).toBe(401);
  });
});
