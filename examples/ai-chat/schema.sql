CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,               -- Google OAuth sub
  email TEXT NOT NULL,
  name TEXT,
  picture TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,               -- UUID
  user_id TEXT NOT NULL REFERENCES users(id),
  title TEXT DEFAULT 'New Chat',
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, updated_at);

CREATE TABLE IF NOT EXISTS user_settings (
  user_id TEXT PRIMARY KEY REFERENCES users(id),
  github_client_id TEXT,
  github_client_secret TEXT,
  llm_api_key TEXT,
  llm_provider TEXT DEFAULT 'google',
  llm_base_url TEXT DEFAULT 'https://generativelanguage.googleapis.com/v1beta',
  llm_model TEXT DEFAULT 'gemini-2.0-flash'
);

CREATE TABLE IF NOT EXISTS files (
  user_id TEXT NOT NULL,
  path TEXT NOT NULL,
  parent_path TEXT NOT NULL,         -- for efficient readdir
  name TEXT NOT NULL,                -- basename
  content BLOB,
  is_directory INTEGER DEFAULT 0,
  mode INTEGER DEFAULT 33188,        -- 0o100644
  size INTEGER DEFAULT 0,
  mtime REAL DEFAULT (unixepoch('now')),
  PRIMARY KEY (user_id, path)
);
CREATE INDEX IF NOT EXISTS idx_files_parent ON files(user_id, parent_path);
