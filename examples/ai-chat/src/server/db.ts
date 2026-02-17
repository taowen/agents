/**
 * D1 database helpers — typed queries for users, sessions, settings.
 */

export interface User {
  id: string;
  email: string;
  name: string | null;
  picture: string | null;
  created_at: string;
  updated_at: string;
}

export interface Session {
  id: string;
  user_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export type LlmProvider = "builtin" | "google" | "openai-compatible";

export interface UserSettings {
  user_id: string;
  github_client_id: string | null;
  github_client_secret: string | null;
  llm_api_key: string | null;
  llm_provider: LlmProvider | null;
  llm_base_url: string | null;
  llm_model: string | null;
}

export async function findOrCreateUser(
  db: D1Database,
  user: { id: string; email: string; name?: string; picture?: string }
): Promise<User> {
  await db
    .prepare(
      `INSERT INTO users (id, email, name, picture)
       VALUES (?, ?, ?, ?)
       ON CONFLICT(id) DO UPDATE SET
         email = excluded.email,
         name = excluded.name,
         picture = excluded.picture,
         updated_at = datetime('now')`
    )
    .bind(user.id, user.email, user.name ?? null, user.picture ?? null)
    .run();

  const row = await db
    .prepare("SELECT * FROM users WHERE id = ?")
    .bind(user.id)
    .first<User>();
  return row!;
}

export async function getUser(
  db: D1Database,
  userId: string
): Promise<User | null> {
  return db
    .prepare("SELECT * FROM users WHERE id = ?")
    .bind(userId)
    .first<User>();
}

export async function getUserByEmail(
  db: D1Database,
  email: string
): Promise<User | null> {
  return db
    .prepare("SELECT * FROM users WHERE email = ?")
    .bind(email)
    .first<User>();
}

export async function listSessions(
  db: D1Database,
  userId: string
): Promise<Session[]> {
  const result = await db
    .prepare(
      "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC"
    )
    .bind(userId)
    .all<Session>();
  return result.results;
}

export async function createSession(
  db: D1Database,
  userId: string,
  title?: string
): Promise<Session> {
  const id = crypto.randomUUID();
  await db
    .prepare("INSERT INTO sessions (id, user_id, title) VALUES (?, ?, ?)")
    .bind(id, userId, title ?? "New Chat")
    .run();

  const row = await db
    .prepare("SELECT * FROM sessions WHERE id = ?")
    .bind(id)
    .first<Session>();
  return row!;
}

export async function updateSessionTitle(
  db: D1Database,
  sessionId: string,
  userId: string,
  title: string
): Promise<boolean> {
  const result = await db
    .prepare(
      "UPDATE sessions SET title = ?, updated_at = datetime('now') WHERE id = ? AND user_id = ?"
    )
    .bind(title, sessionId, userId)
    .run();
  return (result.meta.changes ?? 0) > 0;
}

export async function deleteSession(
  db: D1Database,
  sessionId: string,
  userId: string
): Promise<boolean> {
  const result = await db
    .prepare("DELETE FROM sessions WHERE id = ? AND user_id = ?")
    .bind(sessionId, userId)
    .run();
  return (result.meta.changes ?? 0) > 0;
}

export async function getSettings(
  db: D1Database,
  userId: string
): Promise<UserSettings | null> {
  return db
    .prepare("SELECT * FROM user_settings WHERE user_id = ?")
    .bind(userId)
    .first<UserSettings>();
}

export async function upsertSettings(
  db: D1Database,
  userId: string,
  partial: Partial<Omit<UserSettings, "user_id">>
): Promise<void> {
  const existing = await getSettings(db, userId);

  if (!existing) {
    await db
      .prepare(
        `INSERT INTO user_settings (user_id, github_client_id, github_client_secret, llm_api_key, llm_provider, llm_base_url, llm_model)
         VALUES (?, ?, ?, ?, ?, ?, ?)`
      )
      .bind(
        userId,
        partial.github_client_id ?? null,
        partial.github_client_secret ?? null,
        partial.llm_api_key ?? null,
        partial.llm_provider ?? "builtin",
        partial.llm_base_url ?? null,
        partial.llm_model ?? null
      )
      .run();
  } else {
    await db
      .prepare(
        `UPDATE user_settings SET
           github_client_id = ?,
           github_client_secret = ?,
           llm_api_key = ?,
           llm_provider = ?,
           llm_base_url = ?,
           llm_model = ?
         WHERE user_id = ?`
      )
      .bind(
        partial.github_client_id !== undefined
          ? partial.github_client_id
          : existing.github_client_id,
        partial.github_client_secret !== undefined
          ? partial.github_client_secret
          : existing.github_client_secret,
        partial.llm_api_key !== undefined
          ? partial.llm_api_key
          : existing.llm_api_key,
        partial.llm_provider !== undefined
          ? partial.llm_provider
          : existing.llm_provider,
        partial.llm_base_url !== undefined
          ? partial.llm_base_url
          : existing.llm_base_url,
        partial.llm_model !== undefined
          ? partial.llm_model
          : existing.llm_model,
        userId
      )
      .run();
  }
}
