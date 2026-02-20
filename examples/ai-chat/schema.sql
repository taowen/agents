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

DROP TABLE IF EXISTS user_settings;

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

CREATE TABLE IF NOT EXISTS usage_archive (
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  hour TEXT NOT NULL,
  request_count INTEGER DEFAULT 0,
  input_tokens INTEGER DEFAULT 0,
  cache_read_tokens INTEGER DEFAULT 0,
  cache_write_tokens INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0,
  PRIMARY KEY (user_id, session_id, hour)
);
CREATE INDEX IF NOT EXISTS idx_usage_archive_user_hour ON usage_archive(user_id, hour);
