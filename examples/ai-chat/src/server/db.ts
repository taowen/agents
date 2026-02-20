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

export interface MemoryFiles {
  profile: string;
  preferences: string;
  entities: string;
}

export async function getMemoryFiles(
  db: D1Database,
  userId: string
): Promise<MemoryFiles> {
  const paths = [
    "/home/user/.memory/profile.md",
    "/home/user/.memory/preferences.md",
    "/home/user/.memory/entities.md"
  ];
  const results = await db.batch(
    paths.map((p) =>
      db
        .prepare(
          "SELECT CAST(content AS TEXT) as content FROM files WHERE user_id=? AND path=?"
        )
        .bind(userId, p)
    )
  );
  const keys = ["profile", "preferences", "entities"] as const;
  const data: Record<string, string> = {};
  for (let i = 0; i < keys.length; i++) {
    const row = results[i].results[0] as { content: string | null } | undefined;
    data[keys[i]] = row?.content ?? "";
  }
  return data as MemoryFiles;
}

export async function putMemoryFiles(
  db: D1Database,
  userId: string,
  body: { profile?: string; preferences?: string; entities?: string }
): Promise<void> {
  const enc = new TextEncoder();
  const mkdirSql = `INSERT OR IGNORE INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
     VALUES (?, ?, ?, ?, NULL, 1, 16877, 0, unixepoch('now'))`;
  const fileSql = `INSERT INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
     VALUES (?, ?, ?, ?, ?, 0, 33188, ?, unixepoch('now'))
     ON CONFLICT(user_id, path) DO UPDATE SET content=excluded.content, size=excluded.size, mtime=unixepoch('now')`;

  const stmts: D1PreparedStatement[] = [
    db
      .prepare(mkdirSql)
      .bind(userId, "/home/user/.memory", "/home/user", ".memory")
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
      db
        .prepare(fileSql)
        .bind(
          userId,
          `/home/user/.memory/${name}`,
          "/home/user/.memory",
          name,
          buf,
          buf.length
        )
    );
  }
  await db.batch(stmts);
}
