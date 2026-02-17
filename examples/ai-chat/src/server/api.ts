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

  // GET /api/sessions/:id/schedules
  const schedulesMatch = url.pathname.match(
    /^\/api\/sessions\/([^/]+)\/schedules$/
  );
  if (schedulesMatch && request.method === "GET") {
    const sessionId = schedulesMatch[1];
    const id = env.ChatAgent.idFromName(sessionId);
    const stub = env.ChatAgent.get(id);
    return stub.fetch(new Request("http://agent/get-schedules"));
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

  // GET /api/memory
  if (url.pathname === "/api/memory" && request.method === "GET") {
    const paths = [
      "/home/user/.memory/profile.md",
      "/home/user/.memory/preferences.md",
      "/home/user/.memory/entities.md"
    ];
    const results = await env.DB.batch(
      paths.map((p) =>
        env.DB.prepare(
          "SELECT CAST(content AS TEXT) as content FROM files WHERE user_id=? AND path=?"
        ).bind(userId, p)
      )
    );
    const keys = ["profile", "preferences", "entities"];
    const data: Record<string, string> = {};
    for (let i = 0; i < keys.length; i++) {
      const row = results[i].results[0] as
        | { content: string | null }
        | undefined;
      data[keys[i]] = row?.content ?? "";
    }
    return Response.json(data);
  }

  // PUT /api/memory
  if (url.pathname === "/api/memory" && request.method === "PUT") {
    const body = (await request.json()) as {
      profile?: string;
      preferences?: string;
      entities?: string;
    };
    const enc = new TextEncoder();
    const mkdirSql = `INSERT OR IGNORE INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
       VALUES (?, ?, ?, ?, NULL, 1, 16877, 0, unixepoch('now'))`;
    const fileSql = `INSERT INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
       VALUES (?, ?, ?, ?, ?, 0, 33188, ?, unixepoch('now'))
       ON CONFLICT(user_id, path) DO UPDATE SET content=excluded.content, size=excluded.size, mtime=unixepoch('now')`;

    const stmts: D1PreparedStatement[] = [
      env.DB.prepare(mkdirSql).bind(
        userId,
        "/home/user/.memory",
        "/home/user",
        ".memory"
      )
    ];

    const fileMap: Record<string, string | undefined> = {
      "profile.md": body.profile,
      "preferences.md": body.preferences,
      "entities.md": body.entities
    };
    for (const [name, content] of Object.entries(fileMap)) {
      if (content === undefined) continue;
      const buf = enc.encode(content);
      stmts.push(
        env.DB.prepare(fileSql).bind(
          userId,
          `/home/user/.memory/${name}`,
          "/home/user/.memory",
          name,
          buf,
          buf.length
        )
      );
    }
    await env.DB.batch(stmts);
    return Response.json({ ok: true });
  }

  return null;
}
