/**
 * GitFs — read-only IFileSystem backed by a remote Git repository.
 *
 * Uses isomorphic-git to shallow-clone (depth=1, noCheckout) into an in-memory
 * fs adapter, then serves readdir/readFile/stat directly from the packfile.
 */

import git from "isomorphic-git";
import http from "isomorphic-git/http/web";

// ---- Types matching just-bash IFileSystem ----

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

function normalizePath(path: string): string {
  if (!path || path === "/") return "/";
  let normalized =
    path.endsWith("/") && path !== "/" ? path.slice(0, -1) : path;
  if (!normalized.startsWith("/")) normalized = `/${normalized}`;
  const parts = normalized.split("/").filter((p) => p && p !== ".");
  const resolved: string[] = [];
  for (const part of parts) {
    if (part === "..") resolved.pop();
    else resolved.push(part);
  }
  return `/${resolved.join("/")}` || "/";
}

/** Strip leading '/' — isomorphic-git filepaths must not start with '/' */
function toGitPath(normalizedPath: string): string {
  if (normalizedPath === "/") return "";
  return normalizedPath.startsWith("/")
    ? normalizedPath.slice(1)
    : normalizedPath;
}

function roError(syscall: string, path: string): Error {
  const err = new Error(`EROFS: read-only file system, ${syscall} '${path}'`);
  (err as any).code = "EROFS";
  return err;
}

function enoent(syscall: string, path: string): Error {
  const err = new Error(
    `ENOENT: no such file or directory, ${syscall} '${path}'`
  );
  (err as any).code = "ENOENT";
  return err;
}

function enotdir(syscall: string, path: string): Error {
  const err = new Error(`ENOTDIR: not a directory, ${syscall} '${path}'`);
  (err as any).code = "ENOTDIR";
  return err;
}

// ---- Minimal in-memory fs for isomorphic-git ----

interface MemStat {
  type: "file" | "dir" | "symlink";
  mode: number;
  size: number;
  mtimeMs: number;
  isFile(): boolean;
  isDirectory(): boolean;
  isSymbolicLink(): boolean;
}

function createIsomorphicGitMemFs() {
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
    return {
      type,
      mode: type === "dir" ? 0o40755 : 0o100644,
      size,
      mtimeMs: Date.now(),
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
      // When encoding is requested (e.g. 'utf8'), return string like Node.js fs
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
      return this.stat(filepath);
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
    }
  };
}

// ---- Tree entry from isomorphic-git ----

interface GitTreeEntry {
  mode: string;
  path: string;
  oid: string;
  type: "blob" | "tree" | "commit";
}

// ---- GitFs options ----

export interface GitFsOptions {
  url: string;
  ref?: string;
  corsProxy?: string;
  depth?: number;
  onAuth?: () => { username: string; password?: string };
}

// ---- GitFs ----

export class GitFs {
  private url: string;
  private ref: string;
  private corsProxy?: string;
  private depth: number;
  private onAuth?: () => { username: string; password?: string };

  private memFs = createIsomorphicGitMemFs();
  private cache: object = {};
  private commitOid: string | null = null;
  private commitMtime: Date = new Date(0);
  private initPromise: Promise<void> | null = null;
  private treeCache = new Map<string, GitTreeEntry[]>();

  private static readonly DIR = "/repo";
  private static readonly GITDIR = "/repo/.git";

  constructor(opts: GitFsOptions) {
    this.url = opts.url;
    this.ref = opts.ref ?? "main";
    this.corsProxy = opts.corsProxy;
    this.depth = opts.depth ?? 1;
    this.onAuth = opts.onAuth;
  }

  // ---- Init ----

  /**
   * Eagerly initialize by cloning the repository.
   * Called by mount command so errors surface at mount time.
   */
  async init(): Promise<void> {
    return this.ensureInitialized();
  }

  private async ensureInitialized(): Promise<void> {
    if (this.commitOid) return;
    if (!this.initPromise) {
      this.initPromise = this.doInit().catch((err) => {
        // Clear the cached promise so next access can retry
        this.initPromise = null;
        throw err;
      });
    }
    await this.initPromise;
  }

  private async doInit(): Promise<void> {
    await git.clone({
      fs: this.memFs,
      http,
      dir: GitFs.DIR,
      gitdir: GitFs.GITDIR,
      url: this.url,
      ref: this.ref,
      corsProxy: this.corsProxy,
      singleBranch: true,
      noCheckout: true,
      noTags: true,
      depth: this.depth,
      onAuth: this.onAuth,
      cache: this.cache
    });

    this.commitOid = await git.resolveRef({
      fs: this.memFs,
      gitdir: GitFs.GITDIR,
      ref: "HEAD"
    });

    // Read commit timestamp for mtime
    const [commit] = await git.log({
      fs: this.memFs,
      gitdir: GitFs.GITDIR,
      ref: "HEAD",
      depth: 1,
      cache: this.cache
    });
    if (commit) {
      this.commitMtime = new Date(commit.commit.committer.timestamp * 1000);
    }
  }

  // ---- Internal tree/blob reading ----

  private async readTreeEntries(dirPath: string): Promise<GitTreeEntry[]> {
    const gitPath = toGitPath(dirPath);
    const cacheKey = gitPath;
    const cached = this.treeCache.get(cacheKey);
    if (cached) return cached;

    await this.ensureInitialized();

    const result = await git.readTree({
      fs: this.memFs,
      gitdir: GitFs.GITDIR,
      oid: this.commitOid!,
      filepath: gitPath || undefined,
      cache: this.cache
    });
    const entries = result.tree as GitTreeEntry[];
    this.treeCache.set(cacheKey, entries);
    return entries;
  }

  /**
   * Find the tree entry for a given path by looking in its parent tree.
   * Returns null if not found.
   */
  private async findEntry(filePath: string): Promise<GitTreeEntry | null> {
    const normalized = normalizePath(filePath);
    if (normalized === "/") {
      // Root is always a directory
      return { mode: "040000", path: "", oid: "", type: "tree" };
    }
    const lastSlash = normalized.lastIndexOf("/");
    const parentDir = lastSlash === 0 ? "/" : normalized.slice(0, lastSlash);
    const name = normalized.slice(lastSlash + 1);

    try {
      const entries = await this.readTreeEntries(parentDir);
      return entries.find((e) => e.path === name) ?? null;
    } catch {
      return null;
    }
  }

  // ---- IFileSystem: read operations ----

  async readFile(
    path: string,
    options?: ReadFileOptions | BufferEncoding
  ): Promise<string> {
    const buffer = await this.readFileBuffer(path);
    const encoding = getEncoding(options);
    return fromBuffer(buffer, encoding);
  }

  async readFileBuffer(path: string): Promise<Uint8Array> {
    const normalized = normalizePath(path);
    const gitPath = toGitPath(normalized);

    await this.ensureInitialized();

    try {
      const { blob } = await git.readBlob({
        fs: this.memFs,
        gitdir: GitFs.GITDIR,
        oid: this.commitOid!,
        filepath: gitPath,
        cache: this.cache
      });
      return blob;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("NotFoundError") || msg.includes("Could not find")) {
        throw enoent("open", path);
      }
      throw e;
    }
  }

  async exists(path: string): Promise<boolean> {
    const entry = await this.findEntry(path);
    return entry !== null;
  }

  async stat(path: string): Promise<FsStat> {
    const normalized = normalizePath(path);
    const entry = await this.findEntry(normalized);
    if (!entry) throw enoent("stat", path);

    const isDir = entry.type === "tree";
    const isSymlink = entry.mode === "120000";
    const mode = parseInt(entry.mode, 8);
    return {
      isFile: entry.type === "blob" && !isSymlink,
      isDirectory: isDir,
      isSymbolicLink: isSymlink,
      mode,
      size: 0, // git tree doesn't store file size
      mtime: this.commitMtime
    };
  }

  async lstat(path: string): Promise<FsStat> {
    return this.stat(path);
  }

  async readdir(path: string): Promise<string[]> {
    const normalized = normalizePath(path);

    // Verify it's a directory
    const entry = await this.findEntry(normalized);
    if (!entry) throw enoent("scandir", path);
    if (entry.type !== "tree") throw enotdir("scandir", path);

    const entries = await this.readTreeEntries(normalized);
    return entries.map((e) => e.path).sort();
  }

  async readlink(path: string): Promise<string> {
    const normalized = normalizePath(path);
    const entry = await this.findEntry(normalized);
    if (!entry) throw enoent("readlink", path);
    if (entry.mode !== "120000") {
      const err = new Error(`EINVAL: invalid argument, readlink '${path}'`);
      (err as any).code = "EINVAL";
      throw err;
    }
    // In git, symlinks are stored as blobs containing the target path
    const blob = await this.readFileBuffer(path);
    return textDecoder.decode(blob);
  }

  async realpath(path: string): Promise<string> {
    await this.stat(path); // throws ENOENT if not found
    return normalizePath(path);
  }

  resolvePath(base: string, path: string): string {
    if (path.startsWith("/")) return normalizePath(path);
    const combined = base === "/" ? `/${path}` : `${base}/${path}`;
    return normalizePath(combined);
  }

  getAllPaths(): string[] {
    return [];
  }

  // ---- IFileSystem: write operations (all read-only) ----

  async writeFile(
    path: string,
    _content: string | Uint8Array,
    _options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    throw roError("write", path);
  }

  async appendFile(
    path: string,
    _content: string | Uint8Array,
    _options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    throw roError("write", path);
  }

  async mkdir(path: string, _options?: MkdirOptions): Promise<void> {
    throw roError("mkdir", path);
  }

  async rm(path: string, _options?: RmOptions): Promise<void> {
    throw roError("rm", path);
  }

  async cp(src: string, _dest: string, _options?: CpOptions): Promise<void> {
    throw roError("cp", src);
  }

  async mv(src: string, _dest: string): Promise<void> {
    throw roError("rename", src);
  }

  async chmod(path: string, _mode: number): Promise<void> {
    throw roError("chmod", path);
  }

  async symlink(_target: string, linkPath: string): Promise<void> {
    throw roError("symlink", linkPath);
  }

  async link(_existingPath: string, newPath: string): Promise<void> {
    throw roError("link", newPath);
  }

  async utimes(path: string, _atime: Date, _mtime: Date): Promise<void> {
    throw roError("utimes", path);
  }
}
