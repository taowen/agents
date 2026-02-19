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
