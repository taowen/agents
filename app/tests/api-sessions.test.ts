/**
 * Session management HTTP API integration tests.
 *
 * Tests the full HTTP path: worker.fetch → Hono auth middleware → apiRoutes → D1.
 */
import { env } from "cloudflare:test";
import { describe, it, expect, beforeAll } from "vitest";
import { applyD1Schema, createTestUser, apiRequest } from "./test-utils";

describe("Session management API", () => {
  let db: D1Database;

  beforeAll(async () => {
    db = (env as unknown as { DB: D1Database }).DB;
    await applyD1Schema(db);
  });

  it("unauthenticated request returns 401", async () => {
    const res = await apiRequest("GET", "/api/sessions", "");
    // Empty string userId → auth middleware rejects
    expect(res.status).toBe(401);
  });

  it("POST /api/sessions creates session with default title", async () => {
    const userId = await createTestUser(db);
    const res = await apiRequest("POST", "/api/sessions", userId);
    expect(res.status).toBe(201);

    const session = await res.json<{
      id: string;
      title: string;
      user_id: string;
    }>();
    expect(session.id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
    );
    expect(session.title).toBe("New Chat");
    expect(session.user_id).toBe(userId);
  });

  it("POST /api/sessions with custom title", async () => {
    const userId = await createTestUser(db);
    const res = await apiRequest("POST", "/api/sessions", userId, {
      title: "My Custom Chat"
    });
    expect(res.status).toBe(201);

    const session = await res.json<{ title: string }>();
    expect(session.title).toBe("My Custom Chat");
  });

  it("GET /api/sessions lists sessions ordered by updated_at DESC", async () => {
    const userId = await createTestUser(db);
    await apiRequest("POST", "/api/sessions", userId, { title: "First" });
    await apiRequest("POST", "/api/sessions", userId, { title: "Second" });

    const res = await apiRequest("GET", "/api/sessions", userId);
    expect(res.status).toBe(200);

    const sessions = await res.json<
      Array<{ title: string; updated_at: string }>
    >();
    expect(sessions.length).toBeGreaterThanOrEqual(2);

    // Verify DESC ordering
    for (let i = 1; i < sessions.length; i++) {
      const prev = new Date(sessions[i - 1].updated_at).getTime();
      const curr = new Date(sessions[i].updated_at).getTime();
      expect(prev).toBeGreaterThanOrEqual(curr);
    }
  });

  it("PATCH /api/sessions/:id renames session", async () => {
    const userId = await createTestUser(db);
    const createRes = await apiRequest("POST", "/api/sessions", userId, {
      title: "Original"
    });
    const { id } = await createRes.json<{ id: string }>();

    const patchRes = await apiRequest("PATCH", `/api/sessions/${id}`, userId, {
      title: "Renamed"
    });
    expect(patchRes.status).toBe(200);

    const listRes = await apiRequest("GET", "/api/sessions", userId);
    const sessions = await listRes.json<Array<{ id: string; title: string }>>();
    const found = sessions.find((s) => s.id === id);
    expect(found?.title).toBe("Renamed");
  });

  it("PATCH /api/sessions/:id without title returns 400", async () => {
    const userId = await createTestUser(db);
    const createRes = await apiRequest("POST", "/api/sessions", userId);
    const { id } = await createRes.json<{ id: string }>();

    const patchRes = await apiRequest(
      "PATCH",
      `/api/sessions/${id}`,
      userId,
      {}
    );
    expect(patchRes.status).toBe(400);
  });

  it("DELETE /api/sessions/:id removes session", async () => {
    const userId = await createTestUser(db);
    const createRes = await apiRequest("POST", "/api/sessions", userId, {
      title: "To Delete"
    });
    const { id } = await createRes.json<{ id: string }>();

    const deleteRes = await apiRequest("DELETE", `/api/sessions/${id}`, userId);
    expect(deleteRes.status).toBe(200);

    const listRes = await apiRequest("GET", "/api/sessions", userId);
    const sessions = await listRes.json<Array<{ id: string }>>();
    expect(sessions.find((s) => s.id === id)).toBeUndefined();
  });

  it("DELETE /api/sessions/:id for non-existent session returns 404", async () => {
    const userId = await createTestUser(db);
    const res = await apiRequest(
      "DELETE",
      "/api/sessions/nonexistent-id",
      userId
    );
    expect(res.status).toBe(404);
  });

  it("session isolation: user A cannot see user B sessions", async () => {
    const userA = await createTestUser(db);
    const userB = await createTestUser(db);

    await apiRequest("POST", "/api/sessions", userA, { title: "A's session" });
    await apiRequest("POST", "/api/sessions", userB, { title: "B's session" });

    const resA = await apiRequest("GET", "/api/sessions", userA);
    const sessionsA = await resA.json<
      Array<{ title: string; user_id: string }>
    >();

    const resB = await apiRequest("GET", "/api/sessions", userB);
    const sessionsB = await resB.json<
      Array<{ title: string; user_id: string }>
    >();

    expect(sessionsA.every((s) => s.user_id === userA)).toBe(true);
    expect(sessionsB.every((s) => s.user_id === userB)).toBe(true);
    expect(sessionsA.some((s) => s.title === "B's session")).toBe(false);
    expect(sessionsB.some((s) => s.title === "A's session")).toBe(false);
  });

  it("session isolation: user A cannot delete user B session", async () => {
    const userA = await createTestUser(db);
    const userB = await createTestUser(db);

    const createRes = await apiRequest("POST", "/api/sessions", userB, {
      title: "B's private"
    });
    const { id } = await createRes.json<{ id: string }>();

    // User A tries to delete user B's session
    const deleteRes = await apiRequest("DELETE", `/api/sessions/${id}`, userA);
    expect(deleteRes.status).toBe(404);

    // Session still exists for user B
    const listRes = await apiRequest("GET", "/api/sessions", userB);
    const sessions = await listRes.json<Array<{ id: string }>>();
    expect(sessions.find((s) => s.id === id)).toBeDefined();
  });
});
