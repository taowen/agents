/**
 * Auth route handlers: Google OAuth, Email login, Device authorization.
 * Also includes handleIncomingEmail and storeGDriveCredentials.
 */

import {
  findOrCreateUser,
  getUser,
  getUserByEmail,
  mkdirStatement,
  writeFileStatement
} from "./db";
import {
  createSessionCookie,
  validateSessionCookie,
  generateOAuthState,
  verifyOAuthState,
  generateDeviceCode,
  isTokenRevoked,
  COOKIE_NAME,
  REVOKE_KV_TTL
} from "./auth-tokens";

const EMAIL_DOMAIN = "connect-screen.com";
const EMAIL_TOKEN_TTL = 5 * 60; // 5 minutes in seconds
const DEVICE_CODE_TTL = 10 * 60; // 10 minutes in seconds

/**
 * Handle auth routes. Returns Response if matched, null otherwise.
 */
export async function handleAuthRoutes(
  request: Request,
  env: Env
): Promise<Response> {
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

  // GET /auth/logout — revoke all tokens, clear cookie
  if (url.pathname === "/auth/logout" && request.method === "GET") {
    const payload = await validateSessionCookie(request, env.AUTH_SECRET);
    if (payload) {
      await env.OTP_KV.put(
        `auth-revoke:${payload.userId}`,
        String(Date.now()),
        { expirationTtl: REVOKE_KV_TTL }
      );
    }
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
    const payload = await validateSessionCookie(request, env.AUTH_SECRET);
    if (!payload) {
      return Response.json({ authenticated: false });
    }
    if (await isTokenRevoked(env.OTP_KV, payload.userId, payload.iat)) {
      return Response.json({ authenticated: false });
    }
    const user = await getUser(env.DB, payload.userId);
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

  // POST /auth/device/start — generate device code for RN app login
  if (url.pathname === "/auth/device/start" && request.method === "POST") {
    const code = generateDeviceCode();
    await env.OTP_KV.put(
      `device-login:${code}`,
      JSON.stringify({ status: "pending" }),
      { expirationTtl: DEVICE_CODE_TTL }
    );
    return Response.json({ code });
  }

  // GET /auth/device/check?code= — RN app polls for approval
  if (url.pathname === "/auth/device/check" && request.method === "GET") {
    const code = url.searchParams.get("code")?.toUpperCase();
    if (!code) {
      return new Response("Missing code", { status: 400 });
    }
    const raw = await env.OTP_KV.get(`device-login:${code}`);
    if (!raw) {
      return Response.json({ status: "expired" });
    }
    const data = JSON.parse(raw) as {
      status: string;
      token?: string;
      baseURL?: string;
      model?: string;
    };
    if (data.status === "approved") {
      await env.OTP_KV.delete(`device-login:${code}`);
    }
    return Response.json(data);
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
    const data = JSON.parse(raw) as {
      status: string;
      email?: string;
      reason?: string;
    };
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

    const existing = await getUserByEmail(env.DB, data.email);
    const userId = existing ? existing.id : `email:${data.email}`;
    if (!existing) {
      await findOrCreateUser(env.DB, {
        id: userId,
        email: data.email,
        name: data.email.split("@")[0]
      });
    }

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

  return new Response("Not found", { status: 404 });
}

// ---- Email authentication helpers ----

/** Extract bare email from a MIME From header like "Name <addr>" or "addr" */
function extractEmail(headerValue: string): string {
  const angleMatch = headerValue.match(/<([^>]+)>/);
  return (angleMatch ? angleMatch[1] : headerValue).trim().toLowerCase();
}

/** Extract d= domain from a DKIM-Signature header */
function extractDkimDomain(dkimHeader: string): string | null {
  const match = dkimHeader.match(/\bd=([^;\s]+)/i);
  return match ? match[1].toLowerCase() : null;
}

/**
 * Handle incoming email from Cloudflare Email Routing.
 */
export async function handleIncomingEmail(
  message: ForwardableEmailMessage,
  env: Env
): Promise<void> {
  const to = message.to;
  const match = to.match(/^login-([a-f0-9]+)@/i);
  if (!match) return;

  const token = match[1];

  const dkimHeader = message.headers.get("DKIM-Signature");
  if (!dkimHeader) {
    await rejectToKV(env, token, "No DKIM signature found");
    message.setReject("Missing DKIM signature");
    return;
  }
  const dkimDomain = extractDkimDomain(dkimHeader);
  if (!dkimDomain) {
    await rejectToKV(env, token, "Invalid DKIM signature");
    message.setReject("Invalid DKIM signature");
    return;
  }

  const mimeFromRaw = message.headers.get("From");
  if (!mimeFromRaw) {
    await rejectToKV(env, token, "Missing From header");
    message.setReject("Missing From header");
    return;
  }
  const mimeFrom = extractEmail(mimeFromRaw);
  const fromDomain = mimeFrom.split("@")[1];

  if (dkimDomain !== fromDomain) {
    await rejectToKV(
      env,
      token,
      `DKIM domain (${dkimDomain}) does not match From domain (${fromDomain})`
    );
    message.setReject("DKIM domain does not match sender");
    return;
  }

  const raw = await env.OTP_KV.get(`email-login:${token}`);
  if (!raw) return;

  const data = JSON.parse(raw) as { status: string };
  if (data.status !== "pending") return;

  await env.OTP_KV.put(
    `email-login:${token}`,
    JSON.stringify({ status: "received", email: mimeFrom }),
    { expirationTtl: EMAIL_TOKEN_TTL }
  );
}

async function rejectToKV(
  env: Env,
  token: string,
  reason: string
): Promise<void> {
  const raw = await env.OTP_KV.get(`email-login:${token}`);
  if (!raw) return;
  await env.OTP_KV.put(
    `email-login:${token}`,
    JSON.stringify({ status: "rejected", reason }),
    { expirationTtl: EMAIL_TOKEN_TTL }
  );
}

/**
 * Store Google Drive credentials in D1 as /etc/gdrive-credentials.json.
 */
export async function storeGDriveCredentials(
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
    credentials = {
      access_token: tokenData.access_token,
      refresh_token: tokenData.refresh_token,
      expires_at: expiresAt
    };
  } else {
    try {
      const row = await db
        .prepare(
          "SELECT CAST(content AS TEXT) as content FROM files WHERE user_id = ? AND path = ?"
        )
        .bind(userId, "/etc/gdrive-credentials.json")
        .first<{ content: string | null }>();
      if (row?.content) {
        const existing = JSON.parse(row.content) as {
          refresh_token?: string;
        };
        credentials = {
          access_token: tokenData.access_token,
          refresh_token: existing.refresh_token || "",
          expires_at: expiresAt
        };
      } else {
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
    mkdirStatement(db, userId, "/etc", "/", "etc"),
    writeFileStatement(
      db,
      userId,
      "/etc/gdrive-credentials.json",
      "/etc",
      "gdrive-credentials.json",
      encoded
    )
  ]);
}
