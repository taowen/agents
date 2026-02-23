/**
 * Memory management HTTP API integration tests.
 *
 * Tests the full HTTP path: worker.fetch → Hono auth middleware → apiRoutes → D1.
 */
import { env } from "cloudflare:test";
import { describe, it, expect, beforeAll } from "vitest";
import { applyD1Schema, createTestUser, apiRequest } from "./test-utils";

describe("Memory management API", () => {
  let db: D1Database;

  beforeAll(async () => {
    db = (env as unknown as { DB: D1Database }).DB;
    await applyD1Schema(db);
  });

  it("GET /api/memory with no prior data returns empty strings", async () => {
    const userId = await createTestUser(db);
    const res = await apiRequest("GET", "/api/memory", userId);
    expect(res.status).toBe(200);

    const data = await res.json<{
      profile: string;
      preferences: string;
      entities: string;
    }>();
    expect(data.profile).toBe("");
    expect(data.preferences).toBe("");
    expect(data.entities).toBe("");
  });

  it("PUT /api/memory saves profile and GET retrieves it", async () => {
    const userId = await createTestUser(db);

    const putRes = await apiRequest("PUT", "/api/memory", userId, {
      profile: "Name: Test User\nRole: Developer"
    });
    expect(putRes.status).toBe(200);

    const getRes = await apiRequest("GET", "/api/memory", userId);
    const data = await getRes.json<{ profile: string }>();
    expect(data.profile).toBe("Name: Test User\nRole: Developer");
  });

  it("PUT /api/memory saves all three memory files", async () => {
    const userId = await createTestUser(db);

    await apiRequest("PUT", "/api/memory", userId, {
      profile: "Profile data",
      preferences: "Prefers dark mode",
      entities: "Project: TestApp"
    });

    const res = await apiRequest("GET", "/api/memory", userId);
    const data = await res.json<{
      profile: string;
      preferences: string;
      entities: string;
    }>();
    expect(data.profile).toBe("Profile data");
    expect(data.preferences).toBe("Prefers dark mode");
    expect(data.entities).toBe("Project: TestApp");
  });

  it("PUT /api/memory update overwrites previous content", async () => {
    const userId = await createTestUser(db);

    await apiRequest("PUT", "/api/memory", userId, {
      profile: "Old profile"
    });
    await apiRequest("PUT", "/api/memory", userId, {
      profile: "New profile"
    });

    const res = await apiRequest("GET", "/api/memory", userId);
    const data = await res.json<{ profile: string }>();
    expect(data.profile).toBe("New profile");
  });

  it("memory isolation: user A cannot see user B memory", async () => {
    const userA = await createTestUser(db);
    const userB = await createTestUser(db);

    await apiRequest("PUT", "/api/memory", userA, {
      profile: "Alice's profile"
    });
    await apiRequest("PUT", "/api/memory", userB, {
      profile: "Bob's profile"
    });

    const resA = await apiRequest("GET", "/api/memory", userA);
    const dataA = await resA.json<{ profile: string }>();
    expect(dataA.profile).toBe("Alice's profile");

    const resB = await apiRequest("GET", "/api/memory", userB);
    const dataB = await resB.json<{ profile: string }>();
    expect(dataB.profile).toBe("Bob's profile");
  });

  it("unauthenticated request returns 401", async () => {
    const res = await apiRequest("GET", "/api/memory", "");
    expect(res.status).toBe(401);
  });
});
