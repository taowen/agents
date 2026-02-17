/**
 * Google Drive OAuth helpers.
 *
 * Drive authorization is handled at login time (auth.ts requests drive scope
 * together with openid/email/profile). This module only provides:
 *   GET  /oauth/gdrive/status    — check if credentials exist
 *   POST /oauth/gdrive/disconnect — revoke token + delete credentials
 */

/// <reference types="@cloudflare/workers-types" />

const GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke";

interface GDriveOAuthEnv {
  DB: D1Database;
  GOOGLE_CLIENT_ID?: string;
  GOOGLE_CLIENT_SECRET?: string;
}

export async function handleGDriveOAuth(
  request: Request,
  env: GDriveOAuthEnv,
  userId: string
): Promise<Response | null> {
  const url = new URL(request.url);

  // GET /oauth/gdrive/status — check if credentials exist
  if (url.pathname === "/oauth/gdrive/status" && request.method === "GET") {
    const row = await env.DB.prepare(
      "SELECT 1 FROM files WHERE user_id = ? AND path = ?"
    )
      .bind(userId, "/etc/gdrive-credentials.json")
      .first();

    return Response.json({
      connected: !!row,
      configured: !!(env.GOOGLE_CLIENT_ID && env.GOOGLE_CLIENT_SECRET)
    });
  }

  // POST /oauth/gdrive/disconnect — revoke token + delete credentials
  if (
    url.pathname === "/oauth/gdrive/disconnect" &&
    request.method === "POST"
  ) {
    // Try to read and revoke the existing token
    try {
      const row = await env.DB.prepare(
        "SELECT CAST(content AS TEXT) as content FROM files WHERE user_id = ? AND path = ?"
      )
        .bind(userId, "/etc/gdrive-credentials.json")
        .first<{ content: string | null }>();

      if (row?.content) {
        const creds = JSON.parse(row.content) as { access_token?: string };
        if (creds.access_token) {
          // Best-effort revoke
          await fetch(
            `${GOOGLE_REVOKE_URL}?token=${encodeURIComponent(creds.access_token)}`,
            { method: "POST" }
          ).catch(() => {});
        }
      }
    } catch {
      // Non-fatal — continue to delete
    }

    // Delete credentials file from D1
    await env.DB.prepare("DELETE FROM files WHERE user_id = ? AND path = ?")
      .bind(userId, "/etc/gdrive-credentials.json")
      .run();

    return Response.json({ ok: true });
  }

  return null;
}
