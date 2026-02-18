/**
 * Minimal in-memory filesystem adapter for isomorphic-git.
 *
 * Shared between GitFs (runtime) and mock-git-server (tests).
 */

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

interface MemStat {
  type: "file" | "dir" | "symlink";
  mode: number;
  size: number;
  mtimeMs: number;
  mtime: Date;
  ctimeMs: number;
  ctime: Date;
  dev: number;
  ino: number;
  uid: number;
  gid: number;
  isFile(): boolean;
  isDirectory(): boolean;
  isSymbolicLink(): boolean;
}

export type IsomorphicGitMemFs = ReturnType<typeof createIsomorphicGitMemFs>;

export function createIsomorphicGitMemFs() {
  const files = new Map<string, Uint8Array>();
  const dirs = new Set<string>(["/", "."]);

  function ensureParent(filepath: string) {
    const parts = filepath.split("/");
    for (let i = 1; i < parts.length; i++) {
      const dir = parts.slice(0, i).join("/") || "/";
      dirs.add(dir);
    }
  }

  function makeStat(type: "file" | "dir" | "symlink", size: number): MemStat {
    const now = Date.now();
    return {
      type,
      mode: type === "dir" ? 0o40755 : 0o100644,
      size,
      mtimeMs: now,
      mtime: new Date(now),
      ctimeMs: now,
      ctime: new Date(now),
      dev: 0,
      ino: 0,
      uid: 0,
      gid: 0,
      isFile() {
        return type === "file";
      },
      isDirectory() {
        return type === "dir";
      },
      isSymbolicLink() {
        return type === "symlink";
      }
    };
  }

  return {
    async readFile(filepath: string, opts?: any): Promise<Uint8Array | string> {
      const data = files.get(filepath);
      if (data === undefined) {
        const err = new Error(
          `ENOENT: no such file or directory, open '${filepath}'`
        );
        (err as any).code = "ENOENT";
        throw err;
      }
      const encoding = typeof opts === "string" ? opts : opts?.encoding;
      if (encoding) return textDecoder.decode(data);
      return data;
    },
    async writeFile(
      filepath: string,
      data: Uint8Array | string,
      _opts?: any
    ): Promise<void> {
      ensureParent(filepath);
      const buf = typeof data === "string" ? textEncoder.encode(data) : data;
      files.set(filepath, new Uint8Array(buf));
    },
    async mkdir(filepath: string, _opts?: any): Promise<void> {
      dirs.add(filepath);
    },
    async rmdir(filepath: string): Promise<void> {
      dirs.delete(filepath);
    },
    async unlink(filepath: string): Promise<void> {
      files.delete(filepath);
    },
    async stat(filepath: string): Promise<MemStat> {
      if (dirs.has(filepath)) return makeStat("dir", 0);
      if (files.has(filepath))
        return makeStat("file", files.get(filepath)!.length);
      const err = new Error(
        `ENOENT: no such file or directory, stat '${filepath}'`
      );
      (err as any).code = "ENOENT";
      throw err;
    },
    async lstat(filepath: string): Promise<MemStat> {
      if (dirs.has(filepath)) return makeStat("dir", 0);
      if (files.has(filepath))
        return makeStat("file", files.get(filepath)!.length);
      const err = new Error(
        `ENOENT: no such file or directory, stat '${filepath}'`
      );
      (err as any).code = "ENOENT";
      throw err;
    },
    async readdir(filepath: string): Promise<string[]> {
      const prefix = filepath === "/" || filepath === "." ? "" : filepath + "/";
      const entries = new Set<string>();
      for (const f of files.keys()) {
        if (prefix && !f.startsWith(prefix)) continue;
        if (!prefix && f.includes("/")) {
          entries.add(f.split("/")[0]);
          continue;
        }
        const rest = prefix ? f.slice(prefix.length) : f;
        if (rest && !rest.includes("/")) entries.add(rest);
        else if (rest) entries.add(rest.split("/")[0]);
      }
      for (const d of dirs) {
        if (d === filepath || d === "/" || d === ".") continue;
        if (prefix && !d.startsWith(prefix)) continue;
        const rest = prefix ? d.slice(prefix.length) : d;
        if (rest && !rest.includes("/")) entries.add(rest);
        else if (rest) entries.add(rest.split("/")[0]);
      }
      return [...entries];
    },
    async readlink(filepath: string): Promise<string> {
      const data = files.get(filepath);
      if (data === undefined) {
        const err = new Error(
          `ENOENT: no such file or directory, readlink '${filepath}'`
        );
        (err as any).code = "ENOENT";
        throw err;
      }
      return textDecoder.decode(data);
    },
    async symlink(target: string, filepath: string): Promise<void> {
      ensureParent(filepath);
      files.set(filepath, textEncoder.encode(target));
    },
    /** Snapshot all files and dirs for serialization. */
    _snapshot(): { files: Record<string, string>; dirs: string[] } {
      const out: Record<string, string> = {};
      for (const [k, v] of files) {
        out[k] = Buffer.from(v).toString("base64");
      }
      return { files: out, dirs: [...dirs] };
    },
    /** Restore from a snapshot. */
    _restore(snap: { files: Record<string, string>; dirs: string[] }): void {
      files.clear();
      dirs.clear();
      for (const d of snap.dirs) dirs.add(d);
      for (const [k, b64] of Object.entries(snap.files)) {
        files.set(k, new Uint8Array(Buffer.from(b64, "base64")));
      }
    },
    /** Expose internal files map (for mock-git-server pack generation). */
    _files: files,
    /** Expose internal dirs set (for mock-git-server). */
    _dirs: dirs
  };
}
