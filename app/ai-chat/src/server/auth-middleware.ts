/**
 * Auth middleware: validates session cookie or Bearer device token.
 */

import {
  validateSessionCookie,
  validateDeviceToken,
  isTokenRevoked
} from "./auth-tokens";

/**
 * Validate session cookie or Bearer device token and return userId, or 401 Response.
 */
export async function requireAuth(
  request: Request,
  env: Env
): Promise<string | Response> {
  // Try session cookie first
  const cookiePayload = await validateSessionCookie(request, env.AUTH_SECRET);
  if (
    cookiePayload &&
    !(await isTokenRevoked(env.OTP_KV, cookiePayload.userId, cookiePayload.iat))
  ) {
    return cookiePayload.userId;
  }

  // Try Bearer token
  const authHeader = request.headers.get("Authorization");
  if (authHeader?.startsWith("Bearer ")) {
    const token = authHeader.slice(7);
    const tokenPayload = await validateDeviceToken(token, env.AUTH_SECRET);
    if (
      tokenPayload &&
      !(await isTokenRevoked(env.OTP_KV, tokenPayload.userId, tokenPayload.iat))
    ) {
      return tokenPayload.userId;
    }
  }

  // Try ?token= query param (needed for WebSocket connections from devices)
  const url = new URL(request.url);
  const queryToken = url.searchParams.get("token");
  if (queryToken) {
    const tokenPayload = await validateDeviceToken(queryToken, env.AUTH_SECRET);
    if (
      tokenPayload &&
      !(await isTokenRevoked(env.OTP_KV, tokenPayload.userId, tokenPayload.iat))
    ) {
      return tokenPayload.userId;
    }
  }

  return new Response(JSON.stringify({ error: "Unauthorized" }), {
    status: 401,
    headers: { "Content-Type": "application/json" }
  });
}
