/**
 * Device authorization end-to-end integration tests.
 *
 * Tests the full device auth flow:
 * POST /auth/device/start → code generation
 * POST /api/device/approve → web user approves code
 * GET /auth/device/check → device polls and receives token
 */
import { env } from "cloudflare:test";
import { describe, it, expect, beforeAll } from "vitest";
import {
  applyD1Schema,
  createTestUser,
  apiRequest,
  publicRequest
} from "./test-utils";

describe("Device authorization flow", () => {
  let db: D1Database;

  beforeAll(async () => {
    db = (env as unknown as { DB: D1Database }).DB;
    await applyD1Schema(db);
  });

  it("POST /auth/device/start generates a 6-char device code", async () => {
    const res = await publicRequest("POST", "/auth/device/start");
    expect(res.status).toBe(200);

    const data = await res.json<{ code: string }>();
    expect(data.code).toHaveLength(6);
    expect(data.code).toMatch(/^[A-Z0-9]+$/);
    // Should not contain ambiguous characters
    expect(data.code).not.toMatch(/[0O1I]/);
  });

  it("GET /auth/device/check before approval returns pending", async () => {
    const startRes = await publicRequest("POST", "/auth/device/start");
    const { code } = await startRes.json<{ code: string }>();

    const checkRes = await publicRequest(
      "GET",
      `/auth/device/check?code=${code}`
    );
    expect(checkRes.status).toBe(200);

    const data = await checkRes.json<{ status: string }>();
    expect(data.status).toBe("pending");
  });

  it("full flow: start → approve → check returns approved with token", async () => {
    const userId = await createTestUser(db);

    // Step 1: Device starts auth flow
    const startRes = await publicRequest("POST", "/auth/device/start");
    const { code } = await startRes.json<{ code: string }>();

    // Step 2: Web user approves the code
    const approveRes = await apiRequest("POST", "/api/device/approve", userId, {
      code
    });
    expect(approveRes.status).toBe(200);
    const approveData = await approveRes.json<{ ok: boolean }>();
    expect(approveData.ok).toBe(true);

    // Step 3: Device polls and gets the token
    const checkRes = await publicRequest(
      "GET",
      `/auth/device/check?code=${code}`
    );
    expect(checkRes.status).toBe(200);

    const checkData = await checkRes.json<{
      status: string;
      token: string;
      baseURL: string;
      model: string;
    }>();
    expect(checkData.status).toBe("approved");
    expect(checkData.token).toMatch(/^device\./);
    expect(checkData.baseURL).toContain("/api/proxy/v1");
    expect(checkData.model).toBe("test-model");
  });

  it("approve with invalid code returns 404", async () => {
    const userId = await createTestUser(db);
    const res = await apiRequest("POST", "/api/device/approve", userId, {
      code: "ZZZZZZ"
    });
    expect(res.status).toBe(404);
  });

  it("double approve returns 409 (code already used)", async () => {
    const userId = await createTestUser(db);

    const startRes = await publicRequest("POST", "/auth/device/start");
    const { code } = await startRes.json<{ code: string }>();

    // First approval succeeds
    const res1 = await apiRequest("POST", "/api/device/approve", userId, {
      code
    });
    expect(res1.status).toBe(200);

    // Device reads the token (this deletes the KV entry)
    await publicRequest("GET", `/auth/device/check?code=${code}`);

    // Second approval fails — code no longer exists
    const res2 = await apiRequest("POST", "/api/device/approve", userId, {
      code
    });
    expect(res2.status).toBe(404);
  });

  it("check with expired/missing code returns expired", async () => {
    const res = await publicRequest("GET", "/auth/device/check?code=NONEXIST");
    expect(res.status).toBe(200);

    const data = await res.json<{ status: string }>();
    expect(data.status).toBe("expired");
  });
});
