/**
 * Google OAuth + session cookie authentication.
 *
 * Session cookie format: base64({userId,exp}).hmac_hex
 * HMAC-SHA256 signed with AUTH_SECRET env var.
 */

import { findOrCreateUser, getUser } from "./db";

const COOKIE_NAME = "session";
const COOKIE_MAX_AGE = 30 * 24 * 60 * 60; // 30 days in seconds

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
    googleUrl.searchParams.set("scope", "openid email profile");
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

  return null;
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
