/**
 * API routes for session management and user settings.
 * All routes require authentication (userId already validated).
 */

import {
  getUser,
  listSessions,
  createSession,
  updateSessionTitle,
  deleteSession,
  getSettings,
  upsertSettings
} from "./db";
import type { UserSettings } from "./db";

export async function handleApiRoutes(
  request: Request,
  env: Env,
  userId: string
): Promise<Response | null> {
  const url = new URL(request.url);

  // GET /api/user
  if (url.pathname === "/api/user" && request.method === "GET") {
    const user = await getUser(env.DB, userId);
    if (!user) {
      return Response.json({ error: "User not found" }, { status: 404 });
    }
    return Response.json({
      id: user.id,
      email: user.email,
      name: user.name,
      picture: user.picture
    });
  }

  // GET /api/sessions
  if (url.pathname === "/api/sessions" && request.method === "GET") {
    const sessions = await listSessions(env.DB, userId);
    return Response.json(sessions);
  }

  // POST /api/sessions
  if (url.pathname === "/api/sessions" && request.method === "POST") {
    let title: string | undefined;
    try {
      const body = (await request.json()) as { title?: string };
      title = body.title;
    } catch {
      // empty body is fine
    }
    const session = await createSession(env.DB, userId, title);
    return Response.json(session, { status: 201 });
  }

  // PATCH /api/sessions/:id
  const patchMatch = url.pathname.match(/^\/api\/sessions\/([^/]+)$/);
  if (patchMatch && request.method === "PATCH") {
    const sessionId = patchMatch[1];
    const body = (await request.json()) as { title?: string };
    if (!body.title) {
      return Response.json({ error: "title is required" }, { status: 400 });
    }
    const updated = await updateSessionTitle(
      env.DB,
      sessionId,
      userId,
      body.title
    );
    if (!updated) {
      return Response.json({ error: "Session not found" }, { status: 404 });
    }
    return Response.json({ ok: true });
  }

  // DELETE /api/sessions/:id
  const deleteMatch = url.pathname.match(/^\/api\/sessions\/([^/]+)$/);
  if (deleteMatch && request.method === "DELETE") {
    const sessionId = deleteMatch[1];
    const deleted = await deleteSession(env.DB, sessionId, userId);
    if (!deleted) {
      return Response.json({ error: "Session not found" }, { status: 404 });
    }
    return Response.json({ ok: true });
  }

  // GET /api/settings
  if (url.pathname === "/api/settings" && request.method === "GET") {
    const settings = await getSettings(env.DB, userId);
    if (!settings) {
      return Response.json({
        llm_provider: "builtin"
      });
    }
    // Mask secrets
    return Response.json({
      github_client_id: settings.github_client_id,
      github_configured: !!(
        settings.github_client_id && settings.github_client_secret
      ),
      llm_api_key_set: !!settings.llm_api_key,
      llm_provider: settings.llm_provider ?? "builtin",
      llm_base_url: settings.llm_base_url,
      llm_model: settings.llm_model
    });
  }

  // PUT /api/settings
  if (url.pathname === "/api/settings" && request.method === "PUT") {
    const body = (await request.json()) as Partial<
      Omit<UserSettings, "user_id">
    >;
    await upsertSettings(env.DB, userId, body);
    return Response.json({ ok: true });
  }

  return null;
}
