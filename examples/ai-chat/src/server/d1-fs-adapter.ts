/**
 * D1FsAdapter — IFileSystem backed by D1 `files` table, scoped to a user.
 *
 * Mirrors AgentFsAdapter method-by-method so just-bash behaviour is identical.
 * Error messages must match the ENOENT/EEXIST/etc. strings that just-bash
 * pattern-matches on.
 */

type BufferEncoding =
  | "utf8"
  | "utf-8"
  | "ascii"
  | "binary"
  | "base64"
  | "hex"
  | "latin1";

interface ReadFileOptions {
  encoding?: BufferEncoding | null;
}

interface WriteFileOptions {
  encoding?: BufferEncoding;
}

interface FsStat {
  isFile: boolean;
  isDirectory: boolean;
  isSymbolicLink: boolean;
  mode: number;
  size: number;
  mtime: Date;
}

interface MkdirOptions {
  recursive?: boolean;
}

interface RmOptions {
  recursive?: boolean;
  force?: boolean;
}

interface CpOptions {
  recursive?: boolean;
}

// ---- Helpers ----

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

function toBuffer(content: string | Uint8Array, encoding?: string): Uint8Array {
  if (content instanceof Uint8Array) return content;
  switch (encoding) {
    case "base64":
      return Uint8Array.from(atob(content), (c) => c.charCodeAt(0));
    case "hex": {
      const bytes = new Uint8Array(content.length / 2);
      for (let i = 0; i < content.length; i += 2) {
        bytes[i / 2] = parseInt(content.slice(i, i + 2), 16);
      }
      return bytes;
    }
    case "binary":
    case "latin1":
      return Uint8Array.from(content, (c) => c.charCodeAt(0));
    default:
      return textEncoder.encode(content);
  }
}

function fromBuffer(buffer: Uint8Array, encoding?: string): string {
  switch (encoding) {
    case "base64":
      return btoa(String.fromCharCode(...buffer));
    case "hex":
      return Array.from(buffer)
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
    case "binary":
    case "latin1":
      return String.fromCharCode(...buffer);
    default:
      return textDecoder.decode(buffer);
  }
}

function getEncoding(
  options?: ReadFileOptions | WriteFileOptions | BufferEncoding | null
): string | undefined {
  if (options === null || options === undefined) return undefined;
  if (typeof options === "string") return options;
  return (options as { encoding?: string }).encoding ?? undefined;
}

function parentPath(p: string): string {
  if (p === "/") return "/";
  const last = p.lastIndexOf("/");
  return last === 0 ? "/" : p.slice(0, last);
}

function baseName(p: string): string {
  const last = p.lastIndexOf("/");
  return last === -1 ? p : p.slice(last + 1);
}

// ---- D1FsAdapter ----

export class D1FsAdapter {
  private db: D1Database;
  private userId: string;
  private rootPrefix: string;

  constructor(db: D1Database, userId: string, rootPrefix: string = "") {
    this.db = db;
    this.userId = userId;
    this.rootPrefix =
      rootPrefix && rootPrefix !== "/"
        ? (rootPrefix.startsWith("/") ? rootPrefix : `/${rootPrefix}`).replace(
            /\/$/,
            ""
          )
        : "";
  }

  private normalizePath(path: string): string {
    if (!path || path === "/") {
      return this.rootPrefix || "/";
    }
    let normalized =
      path.endsWith("/") && path !== "/" ? path.slice(0, -1) : path;
    if (!normalized.startsWith("/")) normalized = `/${normalized}`;
    const parts = normalized.split("/").filter((p) => p && p !== ".");
    const resolved: string[] = [];
    for (const part of parts) {
      if (part === "..") resolved.pop();
      else resolved.push(part);
    }
    const relativePath = `/${resolved.join("/")}` || "/";
    if (this.rootPrefix) {
      return relativePath === "/"
        ? this.rootPrefix
        : `${this.rootPrefix}${relativePath}`;
    }
    return relativePath;
  }

  // ---- Core operations ----

  async readFile(
    path: string,
    options?: ReadFileOptions | BufferEncoding
  ): Promise<string> {
    const buffer = await this.readFileBuffer(path);
    const encoding = getEncoding(options);
    return fromBuffer(buffer, encoding);
  }

  async readFileBuffer(path: string): Promise<Uint8Array> {
    const normalized = this.normalizePath(path);
    const row = await this.db
      .prepare(
        "SELECT content, is_directory FROM files WHERE user_id = ? AND path = ?"
      )
      .bind(this.userId, normalized)
      .first<{ content: ArrayBuffer | null; is_directory: number }>();

    if (!row) {
      throw new Error(`ENOENT: no such file or directory, open '${path}'`);
    }
    if (row.is_directory) {
      throw new Error(
        `EISDIR: illegal operation on a directory, read '${path}'`
      );
    }
    if (row.content === null) return new Uint8Array(0);
    return new Uint8Array(row.content);
  }

  async writeFile(
    path: string,
    content: string | Uint8Array,
    options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    const normalized = this.normalizePath(path);
    const encoding = getEncoding(options);
    const buffer = toBuffer(content, encoding);

    // Auto-ensure parent directories exist
    await this.ensureParentDirs(normalized);

    const pp = parentPath(normalized);
    const bn = baseName(normalized);

    await this.db
      .prepare(
        `INSERT INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
         VALUES (?, ?, ?, ?, ?, 0, 33188, ?, unixepoch('now'))
         ON CONFLICT(user_id, path) DO UPDATE SET
           content = excluded.content,
           size = excluded.size,
           mtime = unixepoch('now'),
           is_directory = 0`
      )
      .bind(this.userId, normalized, pp, bn, buffer, buffer.length)
      .run();
  }

  async appendFile(
    path: string,
    content: string | Uint8Array,
    options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    const encoding = getEncoding(options);
    const newBuffer = toBuffer(content, encoding);
    let existingBuffer: Uint8Array;
    try {
      existingBuffer = await this.readFileBuffer(path);
    } catch {
      existingBuffer = new Uint8Array(0);
    }
    const combined = new Uint8Array(existingBuffer.length + newBuffer.length);
    combined.set(existingBuffer);
    combined.set(newBuffer, existingBuffer.length);
    await this.writeFile(path, combined);
  }

  async exists(path: string): Promise<boolean> {
    const normalized = this.normalizePath(path);
    const row = await this.db
      .prepare("SELECT 1 FROM files WHERE user_id = ? AND path = ?")
      .bind(this.userId, normalized)
      .first();
    return !!row;
  }

  async stat(path: string): Promise<FsStat> {
    const normalized = this.normalizePath(path);
    const row = await this.db
      .prepare(
        "SELECT is_directory, mode, size, mtime FROM files WHERE user_id = ? AND path = ?"
      )
      .bind(this.userId, normalized)
      .first<{
        is_directory: number;
        mode: number;
        size: number;
        mtime: number;
      }>();

    if (!row) {
      throw new Error(`ENOENT: no such file or directory, stat '${path}'`);
    }
    return {
      isFile: !row.is_directory,
      isDirectory: !!row.is_directory,
      isSymbolicLink: false,
      mode: row.mode,
      size: row.size,
      mtime: new Date(row.mtime * 1000)
    };
  }

  async lstat(path: string): Promise<FsStat> {
    return this.stat(path);
  }

  async mkdir(path: string, options?: MkdirOptions): Promise<void> {
    const normalized = this.normalizePath(path);

    if (options?.recursive) {
      await this.ensureParentDirs(normalized);
    }

    // Check parent exists (unless recursive already handled it)
    if (!options?.recursive) {
      const pp = parentPath(normalized);
      if (pp !== normalized) {
        const parentExists = await this.exists(pp);
        if (!parentExists) {
          throw new Error(`ENOENT: no such file or directory, mkdir '${path}'`);
        }
      }
    }

    // Check if already exists
    const existing = await this.db
      .prepare("SELECT is_directory FROM files WHERE user_id = ? AND path = ?")
      .bind(this.userId, normalized)
      .first<{ is_directory: number }>();

    if (existing) {
      if (!options?.recursive) {
        throw new Error(`EEXIST: file already exists, mkdir '${path}'`);
      }
      return;
    }

    const pp = parentPath(normalized);
    const bn = baseName(normalized);

    await this.db
      .prepare(
        `INSERT INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
         VALUES (?, ?, ?, ?, NULL, 1, 16877, 0, unixepoch('now'))`
      )
      .bind(this.userId, normalized, pp, bn)
      .run();
  }

  async readdir(path: string): Promise<string[]> {
    const normalized = this.normalizePath(path);

    // Verify directory exists
    const dir = await this.db
      .prepare("SELECT is_directory FROM files WHERE user_id = ? AND path = ?")
      .bind(this.userId, normalized)
      .first<{ is_directory: number }>();

    if (!dir) {
      throw new Error(`ENOENT: no such file or directory, scandir '${path}'`);
    }

    const result = await this.db
      .prepare(
        "SELECT name FROM files WHERE user_id = ? AND parent_path = ? ORDER BY name"
      )
      .bind(this.userId, normalized)
      .all<{ name: string }>();

    return result.results.map((r) => r.name);
  }

  async rm(path: string, options?: RmOptions): Promise<void> {
    const normalized = this.normalizePath(path);

    const row = await this.db
      .prepare("SELECT is_directory FROM files WHERE user_id = ? AND path = ?")
      .bind(this.userId, normalized)
      .first<{ is_directory: number }>();

    if (!row) {
      if (!options?.force) {
        throw new Error(`ENOENT: no such file or directory, rm '${path}'`);
      }
      return;
    }

    if (row.is_directory) {
      if (!options?.recursive) {
        // Check if directory is empty
        const children = await this.db
          .prepare(
            "SELECT 1 FROM files WHERE user_id = ? AND parent_path = ? LIMIT 1"
          )
          .bind(this.userId, normalized)
          .first();
        if (children) {
          throw new Error(`ENOTEMPTY: directory not empty, rm '${path}'`);
        }
      }

      if (options?.recursive) {
        // Delete all descendants and the directory itself
        await this.db
          .prepare(
            "DELETE FROM files WHERE user_id = ? AND (path = ? OR path LIKE ?)"
          )
          .bind(this.userId, normalized, `${normalized}/%`)
          .run();
        return;
      }
    }

    await this.db
      .prepare("DELETE FROM files WHERE user_id = ? AND path = ?")
      .bind(this.userId, normalized)
      .run();
  }

  async cp(src: string, dest: string, options?: CpOptions): Promise<void> {
    const srcStat = await this.stat(src);

    if (srcStat.isFile) {
      const buffer = await this.readFileBuffer(src);
      await this.writeFile(dest, buffer);
    } else if (srcStat.isDirectory) {
      if (!options?.recursive) {
        throw new Error(`EISDIR: is a directory, cp '${src}'`);
      }
      await this.mkdir(dest, { recursive: true });
      const children = await this.readdir(src);
      for (const child of children) {
        const srcChild = src === "/" ? `/${child}` : `${src}/${child}`;
        const destChild = dest === "/" ? `/${child}` : `${dest}/${child}`;
        await this.cp(srcChild, destChild, options);
      }
    }
  }

  async mv(src: string, dest: string): Promise<void> {
    const srcNorm = this.normalizePath(src);
    const destNorm = this.normalizePath(dest);

    const srcRow = await this.db
      .prepare("SELECT 1 FROM files WHERE user_id = ? AND path = ?")
      .bind(this.userId, srcNorm)
      .first();

    if (!srcRow) {
      throw new Error(`ENOENT: no such file or directory, mv '${src}'`);
    }

    await this.ensureParentDirs(destNorm);

    const destPp = parentPath(destNorm);
    const destBn = baseName(destNorm);

    // Update the entry itself
    await this.db
      .prepare(
        `UPDATE files SET path = ?, parent_path = ?, name = ?, mtime = unixepoch('now')
         WHERE user_id = ? AND path = ?`
      )
      .bind(destNorm, destPp, destBn, this.userId, srcNorm)
      .run();

    // Update descendants (for directories)
    const prefix = `${srcNorm}/`;
    const destPrefix = `${destNorm}/`;
    const descendants = await this.db
      .prepare("SELECT path FROM files WHERE user_id = ? AND path LIKE ?")
      .bind(this.userId, `${prefix}%`)
      .all<{ path: string }>();

    for (const row of descendants.results) {
      const newPath = destPrefix + row.path.slice(prefix.length);
      const newPp = parentPath(newPath);
      const newBn = baseName(newPath);
      await this.db
        .prepare(
          "UPDATE files SET path = ?, parent_path = ?, name = ? WHERE user_id = ? AND path = ?"
        )
        .bind(newPath, newPp, newBn, this.userId, row.path)
        .run();
    }
  }

  resolvePath(base: string, path: string): string {
    if (path.startsWith("/")) return this.normalizePath(path);
    const combined = base === "/" ? `/${path}` : `${base}/${path}`;
    return this.normalizePath(combined);
  }

  getAllPaths(): string[] {
    return [];
  }

  async chmod(path: string, _mode: number): Promise<void> {
    const normalized = this.normalizePath(path);
    const pathExists = await this.exists(normalized);
    if (!pathExists) {
      throw new Error(`ENOENT: no such file or directory, chmod '${path}'`);
    }
  }

  async symlink(_target: string, _linkPath: string): Promise<void> {
    throw new Error("ENOSYS: function not implemented, symlink");
  }

  async link(existingPath: string, newPath: string): Promise<void> {
    const srcStat = await this.stat(existingPath);
    if (!srcStat.isFile) {
      throw new Error(`EPERM: operation not permitted, link '${existingPath}'`);
    }
    const destExists = await this.exists(newPath);
    if (destExists) {
      throw new Error(`EEXIST: file already exists, link '${newPath}'`);
    }
    const buffer = await this.readFileBuffer(existingPath);
    await this.writeFile(newPath, buffer);
  }

  async readlink(_path: string): Promise<string> {
    throw new Error("ENOSYS: function not implemented, readlink");
  }

  async realpath(path: string): Promise<string> {
    await this.stat(path);
    return this.resolvePath("/", path);
  }

  async utimes(path: string, _atime: Date, _mtime: Date): Promise<void> {
    await this.stat(path);
  }

  // ---- Private helpers ----

  private async ensureParentDirs(normalizedPath: string): Promise<void> {
    const parts = normalizedPath.split("/").filter(Boolean);
    if (parts.length <= 1) return;

    const stmts: D1PreparedStatement[] = [];
    let current = "";
    for (let i = 0; i < parts.length - 1; i++) {
      current += `/${parts[i]}`;
      const pp = parentPath(current);
      const bn = baseName(current);
      stmts.push(
        this.db
          .prepare(
            `INSERT OR IGNORE INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
             VALUES (?, ?, ?, ?, NULL, 1, 16877, 0, unixepoch('now'))`
          )
          .bind(this.userId, current, pp, bn)
      );
    }
    if (stmts.length > 0) await this.db.batch(stmts);
  }
}
