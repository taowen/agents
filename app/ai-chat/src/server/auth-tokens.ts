/**
 * Auth token primitives: HMAC helpers, session cookies, device tokens,
 * OAuth state CSRF, and token revocation check.
 * Pure crypto functions — no external service dependencies.
 */

const COOKIE_NAME = "session";
const COOKIE_MAX_AGE = 30 * 24 * 60 * 60; // 30 days in seconds
const DEVICE_TOKEN_MAX_AGE = 90 * 24 * 60 * 60 * 1000; // 90 days in ms
const DEVICE_CODE_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"; // no 0/O/1/I

export const REVOKE_KV_TTL = Math.ceil(DEVICE_TOKEN_MAX_AGE / 1000); // 90 days

export type TokenPayload = { userId: string; iat: number };

// Re-export COOKIE_NAME for logout route
export { COOKIE_NAME };

// ---- HMAC helpers ----

export async function hmacSign(
  payload: string,
  secret: string
): Promise<string> {
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

export async function hmacVerify(
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

export async function createSessionCookie(
  userId: string,
  secret: string
): Promise<string> {
  const iat = Date.now();
  const exp = iat + COOKIE_MAX_AGE * 1000;
  const payloadB64 = btoa(JSON.stringify({ userId, iat, exp }));
  const sig = await hmacSign(payloadB64, secret);
  const value = `${payloadB64}.${sig}`;
  return `${COOKIE_NAME}=${value}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${COOKIE_MAX_AGE}; Secure`;
}

export async function validateSessionCookie(
  request: Request,
  secret: string
): Promise<TokenPayload | null> {
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
      iat: number;
      exp: number;
    };
    if (Date.now() > payload.exp) return null;
    if (!payload.iat) return null;
    return { userId: payload.userId, iat: payload.iat };
  } catch {
    return null;
  }
}

// ---- OAuth state (CSRF protection) ----

export async function generateOAuthState(secret: string): Promise<string> {
  const payload = JSON.stringify({
    ts: Date.now(),
    nonce: crypto.randomUUID()
  });
  const payloadB64 = btoa(payload);
  const sig = await hmacSign(payloadB64, secret);
  return `${payloadB64}.${sig}`;
}

export async function verifyOAuthState(
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
    if (Date.now() - payload.ts > 10 * 60 * 1000) return false;
    return true;
  } catch {
    return false;
  }
}

// ---- Device token helpers ----

export async function createDeviceToken(
  userId: string,
  secret: string
): Promise<string> {
  const payloadB64 = btoa(
    JSON.stringify({
      userId,
      iat: Date.now(),
      exp: Date.now() + DEVICE_TOKEN_MAX_AGE
    })
  );
  const sig = await hmacSign(payloadB64, secret);
  return `device.${payloadB64}.${sig}`;
}

export async function validateDeviceToken(
  token: string,
  secret: string
): Promise<TokenPayload | null> {
  if (!token.startsWith("device.")) return null;
  const rest = token.slice(7); // strip "device."
  const dotIdx = rest.indexOf(".");
  if (dotIdx === -1) return null;
  const payloadB64 = rest.slice(0, dotIdx);
  const sigHex = rest.slice(dotIdx + 1);
  const valid = await hmacVerify(payloadB64, sigHex, secret);
  if (!valid) return null;
  try {
    const payload = JSON.parse(atob(payloadB64)) as {
      userId: string;
      iat: number;
      exp: number;
    };
    if (!payload.exp || Date.now() > payload.exp) return null;
    if (!payload.iat) return null;
    return { userId: payload.userId, iat: payload.iat };
  } catch {
    return null;
  }
}

export function generateDeviceCode(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(6));
  return Array.from(bytes)
    .map((b) => DEVICE_CODE_CHARS[b % DEVICE_CODE_CHARS.length])
    .join("");
}

// ---- Token revocation ----

export async function isTokenRevoked(
  kv: KVNamespace,
  userId: string,
  iat: number
): Promise<boolean> {
  const raw = await kv.get(`auth-revoke:${userId}`);
  if (!raw) return false;
  return iat < Number(raw);
}
