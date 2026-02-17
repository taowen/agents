/**
 * Google OAuth + session cookie authentication.
 *
 * Session cookie format: base64({userId,exp}).hmac_hex
 * HMAC-SHA256 signed with AUTH_SECRET env var.
 */

import { findOrCreateUser, getUser, getUserByEmail } from "./db";

const COOKIE_NAME = "session";
const COOKIE_MAX_AGE = 30 * 24 * 60 * 60; // 30 days in seconds
const EMAIL_DOMAIN = "connect-screen.com";
const EMAIL_TOKEN_TTL = 5 * 60; // 5 minutes in seconds

// ---- HMAC helpers ----

async function hmacSign(payload: string, secret: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const sig = await crypto.subtle.sign(
    "HMAC",
    key,
    new TextEncoder().encode(payload)
  );
  return Array.from(new Uint8Array(sig))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function hmacVerify(
  payload: string,
  sigHex: string,
  secret: string
): Promise<boolean> {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["verify"]
  );
  const sigBytes = new Uint8Array(sigHex.length / 2);
  for (let i = 0; i < sigHex.length; i += 2) {
    sigBytes[i / 2] = parseInt(sigHex.slice(i, i + 2), 16);
  }
  return crypto.subtle.verify(
    "HMAC",
    key,
    sigBytes,
    new TextEncoder().encode(payload)
  );
}

// ---- Cookie helpers ----

async function createSessionCookie(
  userId: string,
  secret: string
): Promise<string> {
  const exp = Date.now() + COOKIE_MAX_AGE * 1000;
  const payloadB64 = btoa(JSON.stringify({ userId, exp }));
  const sig = await hmacSign(payloadB64, secret);
  const value = `${payloadB64}.${sig}`;
  return `${COOKIE_NAME}=${value}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${COOKIE_MAX_AGE}; Secure`;
}

async function validateSessionCookie(
  request: Request,
  secret: string
): Promise<string | null> {
  const cookieHeader = request.headers.get("Cookie");
  if (!cookieHeader) return null;

  const match = cookieHeader.match(
    new RegExp(`(?:^|;\\s*)${COOKIE_NAME}=([^;]+)`)
  );
  if (!match) return null;

  const value = match[1];
  const dotIdx = value.indexOf(".");
  if (dotIdx === -1) return null;

  const payloadB64 = value.slice(0, dotIdx);
  const sigHex = value.slice(dotIdx + 1);

  const valid = await hmacVerify(payloadB64, sigHex, secret);
  if (!valid) return null;

  try {
    const payload = JSON.parse(atob(payloadB64)) as {
      userId: string;
      exp: number;
    };
    if (Date.now() > payload.exp) return null;
    return payload.userId;
  } catch {
    return null;
  }
}

// ---- OAuth state (CSRF protection) ----

async function generateOAuthState(secret: string): Promise<string> {
  const payload = JSON.stringify({
    ts: Date.now(),
    nonce: crypto.randomUUID()
  });
  const payloadB64 = btoa(payload);
  const sig = await hmacSign(payloadB64, secret);
  return `${payloadB64}.${sig}`;
}

async function verifyOAuthState(
  state: string,
  secret: string
): Promise<boolean> {
  const dotIdx = state.indexOf(".");
  if (dotIdx === -1) return false;
  const payloadB64 = state.slice(0, dotIdx);
  const sigHex = state.slice(dotIdx + 1);
  const valid = await hmacVerify(payloadB64, sigHex, secret);
  if (!valid) return false;
  try {
    const payload = JSON.parse(atob(payloadB64)) as { ts: number };
    // Reject states older than 10 minutes
    if (Date.now() - payload.ts > 10 * 60 * 1000) return false;
    return true;
  } catch {
    return false;
  }
}

// ---- Public API ----

/**
 * Handle auth routes. Returns Response if matched, null otherwise.
 */
export async function handleAuthRoutes(
  request: Request,
  env: Env
): Promise<Response | null> {
  const url = new URL(request.url);

  // GET /auth/google — redirect to Google OAuth
  if (url.pathname === "/auth/google" && request.method === "GET") {
    const state = await generateOAuthState(env.AUTH_SECRET);
    const redirectUri = `${url.origin}/auth/google/callback`;
    const googleUrl = new URL("https://accounts.google.com/o/oauth2/v2/auth");
    googleUrl.searchParams.set("client_id", env.GOOGLE_CLIENT_ID);
    googleUrl.searchParams.set("redirect_uri", redirectUri);
    googleUrl.searchParams.set("response_type", "code");
    googleUrl.searchParams.set(
      "scope",
      "openid email profile https://www.googleapis.com/auth/drive"
    );
    googleUrl.searchParams.set("access_type", "offline");
    googleUrl.searchParams.set("state", state);
    return Response.redirect(googleUrl.toString(), 302);
  }

  // GET /auth/google/callback — exchange code, set cookie
  if (url.pathname === "/auth/google/callback" && request.method === "GET") {
    const code = url.searchParams.get("code");
    const state = url.searchParams.get("state");
    if (!code || !state) {
      return new Response("Missing code or state", { status: 400 });
    }

    const stateValid = await verifyOAuthState(state, env.AUTH_SECRET);
    if (!stateValid) {
      return new Response("Invalid or expired state", { status: 403 });
    }

    // Exchange code for tokens
    const tokenRes = await fetch("https://oauth2.googleapis.com/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        code,
        client_id: env.GOOGLE_CLIENT_ID,
        client_secret: env.GOOGLE_CLIENT_SECRET,
        redirect_uri: `${url.origin}/auth/google/callback`,
        grant_type: "authorization_code"
      })
    });

    const tokenData = (await tokenRes.json()) as {
      access_token?: string;
      refresh_token?: string;
      expires_in?: number;
      error?: string;
    };
    if (!tokenData.access_token) {
      return new Response(
        `Google token exchange failed: ${tokenData.error || "unknown"}`,
        { status: 502 }
      );
    }

    // Fetch user profile
    const profileRes = await fetch(
      "https://www.googleapis.com/oauth2/v2/userinfo",
      {
        headers: { Authorization: `Bearer ${tokenData.access_token}` }
      }
    );
    const profile = (await profileRes.json()) as {
      id: string;
      email: string;
      name?: string;
      picture?: string;
    };

    if (!profile.id || !profile.email) {
      return new Response("Failed to fetch user profile", { status: 502 });
    }

    // Upsert user in D1
    await findOrCreateUser(env.DB, {
      id: profile.id,
      email: profile.email,
      name: profile.name,
      picture: profile.picture
    });

    // Store Google Drive credentials (access_token + refresh_token)
    await storeGDriveCredentials(env.DB, profile.id, tokenData);

    // Set session cookie and redirect
    const cookie = await createSessionCookie(profile.id, env.AUTH_SECRET);
    return new Response(null, {
      status: 302,
      headers: {
        Location: "/",
        "Set-Cookie": cookie
      }
    });
  }

  // GET /auth/logout — clear cookie
  if (url.pathname === "/auth/logout" && request.method === "GET") {
    return new Response(null, {
      status: 302,
      headers: {
        Location: "/",
        "Set-Cookie": `${COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0; Secure`
      }
    });
  }

  // GET /auth/status — check authentication
  if (url.pathname === "/auth/status" && request.method === "GET") {
    const userId = await validateSessionCookie(request, env.AUTH_SECRET);
    if (!userId) {
      return Response.json({ authenticated: false });
    }
    const user = await getUser(env.DB, userId);
    if (!user) {
      return Response.json({ authenticated: false });
    }
    return Response.json({
      authenticated: true,
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        picture: user.picture
      }
    });
  }

  // POST /auth/email/start — generate token, store in KV
  if (url.pathname === "/auth/email/start" && request.method === "POST") {
    const token = crypto.randomUUID().replace(/-/g, "").slice(0, 16);
    await env.OTP_KV.put(
      `email-login:${token}`,
      JSON.stringify({ status: "pending" }),
      { expirationTtl: EMAIL_TOKEN_TTL }
    );
    return Response.json({
      token,
      address: `login-${token}@${EMAIL_DOMAIN}`
    });
  }

  // GET /auth/email/check?token= — poll KV status
  if (url.pathname === "/auth/email/check" && request.method === "GET") {
    const token = url.searchParams.get("token");
    if (!token) {
      return new Response("Missing token", { status: 400 });
    }
    const raw = await env.OTP_KV.get(`email-login:${token}`);
    if (!raw) {
      return Response.json({ status: "expired" });
    }
    const data = JSON.parse(raw) as { status: string; email?: string };
    return Response.json(data);
  }

  // POST /auth/email/confirm?token= — confirm login, set session cookie
  if (url.pathname === "/auth/email/confirm" && request.method === "POST") {
    const token = url.searchParams.get("token");
    if (!token) {
      return new Response("Missing token", { status: 400 });
    }
    const raw = await env.OTP_KV.get(`email-login:${token}`);
    if (!raw) {
      return new Response("Token expired", { status: 410 });
    }
    const data = JSON.parse(raw) as { status: string; email?: string };
    if (data.status !== "received" || !data.email) {
      return new Response("Email not yet received", { status: 400 });
    }

    // Reuse existing user if one already exists with this email (e.g. from Google OAuth)
    const existing = await getUserByEmail(env.DB, data.email);
    const userId = existing ? existing.id : `email:${data.email}`;
    if (!existing) {
      await findOrCreateUser(env.DB, {
        id: userId,
        email: data.email,
        name: data.email.split("@")[0]
      });
    }

    // Clean up KV
    await env.OTP_KV.delete(`email-login:${token}`);

    const cookie = await createSessionCookie(userId, env.AUTH_SECRET);
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Set-Cookie": cookie
      }
    });
  }

  return null;
}

/**
 * Handle incoming email from Cloudflare Email Routing.
 * Extracts token from recipient address and sender email, updates KV.
 */
export async function handleIncomingEmail(
  message: ForwardableEmailMessage,
  env: Env
): Promise<void> {
  const to = message.to;
  // Extract token from login-{token}@connect-screen.com
  const match = to.match(/^login-([a-f0-9]+)@/i);
  if (!match) return;

  const token = match[1];
  const from = message.from;

  const raw = await env.OTP_KV.get(`email-login:${token}`);
  if (!raw) return; // expired or invalid

  const data = JSON.parse(raw) as { status: string };
  if (data.status !== "pending") return; // already processed

  await env.OTP_KV.put(
    `email-login:${token}`,
    JSON.stringify({ status: "received", email: from }),
    { expirationTtl: EMAIL_TOKEN_TTL }
  );
}

/**
 * Store Google Drive credentials in D1 as /etc/gdrive-credentials.json.
 * If refresh_token is present (first login or re-consent), store full credentials.
 * If absent (subsequent logins), update only access_token + expires_at.
 */
async function storeGDriveCredentials(
  db: D1Database,
  userId: string,
  tokenData: {
    access_token?: string;
    refresh_token?: string;
    expires_in?: number;
  }
): Promise<void> {
  if (!tokenData.access_token) return;

  const expiresAt = Date.now() + (tokenData.expires_in || 3600) * 1000;

  let credentials: {
    access_token: string;
    refresh_token: string;
    expires_at: number;
  };

  if (tokenData.refresh_token) {
    // First authorization or re-consent: store everything
    credentials = {
      access_token: tokenData.access_token,
      refresh_token: tokenData.refresh_token,
      expires_at: expiresAt
    };
  } else {
    // Subsequent login: keep existing refresh_token, update access_token
    try {
      const row = await db
        .prepare(
          "SELECT CAST(content AS TEXT) as content FROM files WHERE user_id = ? AND path = ?"
        )
        .bind(userId, "/etc/gdrive-credentials.json")
        .first<{ content: string | null }>();
      if (row?.content) {
        const existing = JSON.parse(row.content) as { refresh_token?: string };
        credentials = {
          access_token: tokenData.access_token,
          refresh_token: existing.refresh_token || "",
          expires_at: expiresAt
        };
      } else {
        // No existing credentials and no refresh_token — store what we have
        credentials = {
          access_token: tokenData.access_token,
          refresh_token: "",
          expires_at: expiresAt
        };
      }
    } catch {
      credentials = {
        access_token: tokenData.access_token,
        refresh_token: "",
        expires_at: expiresAt
      };
    }
  }

  const content = JSON.stringify(credentials);
  const encoded = new TextEncoder().encode(content);

  await db.batch([
    db
      .prepare(
        `INSERT OR IGNORE INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
         VALUES (?, ?, ?, ?, NULL, 1, 16877, 0, unixepoch('now'))`
      )
      .bind(userId, "/etc", "/", "etc"),
    db
      .prepare(
        `INSERT INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
         VALUES (?, ?, ?, ?, ?, 0, 33188, ?, unixepoch('now'))
         ON CONFLICT(user_id, path) DO UPDATE SET
           content = excluded.content, size = excluded.size, mtime = unixepoch('now')`
      )
      .bind(
        userId,
        "/etc/gdrive-credentials.json",
        "/etc",
        "gdrive-credentials.json",
        encoded,
        encoded.length
      )
  ]);
}

/**
 * Validate session cookie and return userId, or 401 Response.
 */
export async function requireAuth(
  request: Request,
  env: Env
): Promise<string | Response> {
  const userId = await validateSessionCookie(request, env.AUTH_SECRET);
  if (!userId) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401,
      headers: { "Content-Type": "application/json" }
    });
  }
  return userId;
}
