/**
 * Auth barrel — re-exports from sub-modules so existing import paths stay valid.
 *
 * Implementation split into:
 *   auth-tokens.ts     — HMAC, cookies, device tokens, OAuth state, revocation
 *   auth-routes.ts     — handleAuthRoutes, handleIncomingEmail, storeGDriveCredentials
 *   auth-middleware.ts  — requireAuth
 */

export { handleAuthRoutes, handleIncomingEmail } from "./auth-routes";
export { requireAuth } from "./auth-middleware";
export { createDeviceToken, validateDeviceToken } from "./auth-tokens";
