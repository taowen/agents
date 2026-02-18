/**
 * GitFs — IFileSystem backed by a remote Git repository with overlay writes.
 *
 * Uses isomorphic-git to shallow-clone (depth=1, noCheckout) into an in-memory
 * fs adapter, then serves readdir/readFile/stat directly from the packfile.
 *
 * Writes are buffered in an overlay. Call commitAndPush() to materialize the
 * overlay into the git working directory, commit, and push to the remote.
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

// ---- Public types ----

export interface GitStatus {
  modified: string[];
  added: string[];
  deleted: string[];
}

// ---- Helpers ----

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

function toBuffer(content: string | Uint8Array, encoding?: string): Uint8Array {
  if (content instanceof Uint8Array) return content;
  return textEncoder.encode(content);
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

function eexist(syscall: string, path: string): Error {
  const err = new Error(`EEXIST: file already exists, ${syscall} '${path}'`);
  (err as any).code = "EEXIST";
  return err;
}

// ---- Minimal in-memory fs for isomorphic-git ----

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
  http?: any; // optional: inject mock HTTP transport for testing
}

// ---- GitFs ----

export class GitFs {
  private url: string;
  private ref: string | undefined;
  private corsProxy?: string;
  private depth: number;
  private onAuth?: () => { username: string; password?: string };
  private httpTransport: any;

  private memFs = createIsomorphicGitMemFs();
  private cache: object = {};
  private commitOid: string | null = null;
  private remoteOid: string | null = null;
  private commitMtime: Date = new Date(0);
  private initPromise: Promise<void> | null = null;
  private treeCache = new Map<string, GitTreeEntry[]>();

  // ---- Overlay state ----
  private overlay = new Map<string, Uint8Array>(); // new/modified files
  private overlayDirs = new Set<string>(); // new directories
  private deleted = new Set<string>(); // deleted paths

  private static readonly DIR = "/repo";
  private static readonly GITDIR = "/repo/.git";

  constructor(opts: GitFsOptions) {
    this.url = opts.url;
    this.ref = opts.ref;
    this.corsProxy = opts.corsProxy;
    this.depth = opts.depth ?? 1;
    this.onAuth = opts.onAuth;
    this.httpTransport = opts.http ?? http;
  }

  // ---- Public: overlay status and URL ----

  /** Returns true if there are uncommitted overlay changes. */
  isDirty(): boolean {
    return this.overlay.size > 0 || this.deleted.size > 0;
  }

  /** Expose the remote URL (for credential lookup). */
  getUrl(): string {
    return this.url;
  }

  /** Expose the resolved ref (available after init). */
  getRef(): string | undefined {
    return this.ref;
  }

  /** Returns true if there are local commits not yet pushed. */
  hasUnpushedCommits(): boolean {
    return this.commitOid !== this.remoteOid;
  }

  /** Return overlay status: added, modified, and deleted files. */
  async getStatus(): Promise<GitStatus> {
    await this.ensureInitialized();
    const modified: string[] = [];
    const added: string[] = [];
    const deleted: string[] = [];

    for (const path of this.overlay.keys()) {
      const gitPath = toGitPath(path);
      try {
        await git.readBlob({
          fs: this.memFs,
          gitdir: GitFs.GITDIR,
          oid: this.commitOid!,
          filepath: gitPath,
          cache: this.cache
        });
        modified.push(path);
      } catch {
        added.push(path);
      }
    }

    for (const path of this.deleted) {
      deleted.push(path);
    }

    return { modified, added, deleted };
  }

  /**
   * Commit overlay changes without pushing.
   * Returns the new commit OID.
   */
  async commit(
    message: string,
    author: { name: string; email: string }
  ): Promise<string> {
    await this.ensureInitialized();

    if (!this.isDirty()) {
      throw new Error("nothing to commit, working tree clean");
    }

    const DIR = GitFs.DIR;

    // 1. Checkout working directory from current commit
    await git.checkout({
      fs: this.memFs,
      dir: DIR,
      ref: this.ref,
      cache: this.cache
    });

    // 2. Materialize overlay files into memFs working directory
    for (const [path, content] of this.overlay) {
      const fsPath = `${DIR}${path}`;
      await this.memFs.writeFile(fsPath, content);
    }

    // 3. Stage added/modified files
    for (const path of this.overlay.keys()) {
      const gitPath = toGitPath(path);
      await git.add({
        fs: this.memFs,
        dir: DIR,
        filepath: gitPath,
        cache: this.cache
      });
    }

    // 4. Stage deletions
    for (const path of this.deleted) {
      const gitPath = toGitPath(path);
      try {
        await git.remove({
          fs: this.memFs,
          dir: DIR,
          filepath: gitPath,
          cache: this.cache
        });
      } catch {
        // File may not exist in git tree — skip
      }
    }

    // 5. Commit
    this.commitOid = await git.commit({
      fs: this.memFs,
      dir: DIR,
      message,
      author,
      cache: this.cache
    });

    // 6. Clear overlay and caches
    this.treeCache.clear();
    this.overlay.clear();
    this.overlayDirs.clear();
    this.deleted.clear();

    return this.commitOid;
  }

  /**
   * Push local commits to the remote.
   */
  async push(
    onAuth?: () => { username: string; password?: string }
  ): Promise<void> {
    await this.ensureInitialized();

    await git.push({
      fs: this.memFs,
      http: this.httpTransport,
      dir: GitFs.DIR,
      onAuth: onAuth ?? this.onAuth,
      cache: this.cache
    });

    this.remoteOid = this.commitOid;
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
    // Auto-detect default branch if not specified
    if (!this.ref) {
      const info = await git.getRemoteInfo({
        http: this.httpTransport,
        url: this.url,
        corsProxy: this.corsProxy,
        onAuth: this.onAuth
      });
      this.ref = info.HEAD ?? "main";
    }

    await git.clone({
      fs: this.memFs,
      http: this.httpTransport,
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

    this.remoteOid = this.commitOid;
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

  // ---- Overlay helpers ----

  /** Ensure parent directories exist in the overlay. */
  private ensureOverlayParents(filePath: string): void {
    const parts = filePath.split("/").filter(Boolean);
    for (let i = 1; i < parts.length; i++) {
      const dir = "/" + parts.slice(0, i).join("/");
      this.overlayDirs.add(dir);
      this.deleted.delete(dir);
    }
  }

  /** Check if a path (or any ancestor) is in the deleted set. */
  private isDeleted(normalizedPath: string): boolean {
    if (this.deleted.has(normalizedPath)) return true;
    // Check if any ancestor is deleted
    const parts = normalizedPath.split("/").filter(Boolean);
    for (let i = 1; i < parts.length; i++) {
      const ancestor = "/" + parts.slice(0, i).join("/");
      if (this.deleted.has(ancestor)) return true;
    }
    return false;
  }

  /**
   * Collect overlay entries that are direct children of a directory.
   */
  private getOverlayChildren(dirPath: string): Set<string> {
    const prefix = dirPath === "/" ? "/" : dirPath + "/";
    const children = new Set<string>();

    for (const path of this.overlay.keys()) {
      if (path.startsWith(prefix)) {
        const rest = path.slice(prefix.length);
        const name = rest.split("/")[0];
        if (name) children.add(name);
      }
    }
    for (const path of this.overlayDirs) {
      if (path.startsWith(prefix)) {
        const rest = path.slice(prefix.length);
        const name = rest.split("/")[0];
        if (name) children.add(name);
      }
    }

    return children;
  }

  // ---- IFileSystem: read operations (overlay-aware) ----

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

    // Check overlay first
    const overlayData = this.overlay.get(normalized);
    if (overlayData !== undefined) return overlayData;

    // Check if deleted
    if (this.isDeleted(normalized)) throw enoent("open", path);

    // Fall through to git tree
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
    const normalized = normalizePath(path);

    // Check overlay
    if (this.overlay.has(normalized)) return true;
    if (this.overlayDirs.has(normalized)) return true;

    // Check if deleted
    if (this.isDeleted(normalized)) return false;

    // Check if overlay children exist under this path (making it a virtual dir)
    const overlayChildren = this.getOverlayChildren(normalized);
    if (overlayChildren.size > 0) return true;

    // Fall through to git tree
    const entry = await this.findEntry(path);
    return entry !== null;
  }

  async stat(path: string): Promise<FsStat> {
    const normalized = normalizePath(path);

    // Check overlay files
    const overlayData = this.overlay.get(normalized);
    if (overlayData !== undefined) {
      return {
        isFile: true,
        isDirectory: false,
        isSymbolicLink: false,
        mode: 0o100644,
        size: overlayData.length,
        mtime: new Date()
      };
    }

    // Check overlay directories
    if (this.overlayDirs.has(normalized) || normalized === "/") {
      // Also check if there are overlay children
      return {
        isFile: false,
        isDirectory: true,
        isSymbolicLink: false,
        mode: 0o40755,
        size: 0,
        mtime: new Date()
      };
    }

    // Check overlay children (path is a virtual directory containing overlay files)
    const overlayChildren = this.getOverlayChildren(normalized);

    // Check if deleted
    if (this.isDeleted(normalized)) {
      if (overlayChildren.size > 0) {
        // Directory was deleted but overlay added new files under it
        return {
          isFile: false,
          isDirectory: true,
          isSymbolicLink: false,
          mode: 0o40755,
          size: 0,
          mtime: new Date()
        };
      }
      throw enoent("stat", path);
    }

    if (overlayChildren.size > 0) {
      // This path has overlay children — it's at least a directory
      // Try git tree first to get proper stats
      try {
        const entry = await this.findEntry(normalized);
        if (entry) {
          const isDir = entry.type === "tree";
          const isSymlink = entry.mode === "120000";
          const mode = parseInt(entry.mode, 8);
          return {
            isFile: entry.type === "blob" && !isSymlink,
            isDirectory: isDir,
            isSymbolicLink: isSymlink,
            mode,
            size: 0,
            mtime: this.commitMtime
          };
        }
      } catch {
        // fall through
      }
      return {
        isFile: false,
        isDirectory: true,
        isSymbolicLink: false,
        mode: 0o40755,
        size: 0,
        mtime: new Date()
      };
    }

    // Fall through to git tree
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

    // Collect overlay children for this directory
    const overlayChildren = this.getOverlayChildren(normalized);

    // Check if deleted — but overlay may have re-created it
    const pathDeleted = this.isDeleted(normalized);

    let gitEntryNames: string[] = [];
    if (!pathDeleted) {
      // Try git tree
      try {
        const entry = await this.findEntry(normalized);
        if (entry && entry.type === "tree") {
          const entries = await this.readTreeEntries(normalized);
          gitEntryNames = entries.map((e) => e.path);
        } else if (!entry && overlayChildren.size === 0) {
          throw enoent("scandir", path);
        } else if (
          entry &&
          entry.type !== "tree" &&
          overlayChildren.size === 0
        ) {
          throw enotdir("scandir", path);
        }
      } catch (e) {
        // If it's an ENOENT/ENOTDIR we generated, and there are overlay children, ignore
        const code = (e as any)?.code;
        if (
          overlayChildren.size === 0 &&
          !this.overlayDirs.has(normalized) &&
          normalized !== "/"
        ) {
          throw e;
        }
        if (
          code !== "ENOENT" &&
          code !== "ENOTDIR" &&
          !(e as Error).message?.includes("ENOENT")
        ) {
          throw e;
        }
      }
    } else if (
      overlayChildren.size === 0 &&
      !this.overlayDirs.has(normalized)
    ) {
      throw enoent("scandir", path);
    }

    // Merge git entries + overlay children, minus deleted
    const result = new Set<string>();
    for (const name of gitEntryNames) {
      const childPath =
        normalized === "/" ? `/${name}` : `${normalized}/${name}`;
      if (!this.isDeleted(childPath)) {
        result.add(name);
      }
    }
    for (const name of overlayChildren) {
      result.add(name);
    }

    return Array.from(result).sort();
  }

  async readlink(path: string): Promise<string> {
    const normalized = normalizePath(path);

    // Overlay files are not symlinks
    if (this.overlay.has(normalized)) {
      const err = new Error(`EINVAL: invalid argument, readlink '${path}'`);
      (err as any).code = "EINVAL";
      throw err;
    }

    if (this.isDeleted(normalized)) throw enoent("readlink", path);

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

  // ---- IFileSystem: write operations (overlay-backed) ----

  async writeFile(
    path: string,
    content: string | Uint8Array,
    options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    const normalized = normalizePath(path);
    const buf = toBuffer(content, getEncoding(options));
    this.overlay.set(normalized, buf);
    this.ensureOverlayParents(normalized);
    this.deleted.delete(normalized);
  }

  async appendFile(
    path: string,
    content: string | Uint8Array,
    options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    const normalized = normalizePath(path);
    const appendBuf = toBuffer(content, getEncoding(options));

    // Read existing content (overlay or git tree)
    let existing: Uint8Array;
    try {
      existing = await this.readFileBuffer(path);
    } catch {
      existing = new Uint8Array(0);
    }

    // Concatenate
    const merged = new Uint8Array(existing.length + appendBuf.length);
    merged.set(existing, 0);
    merged.set(appendBuf, existing.length);

    this.overlay.set(normalized, merged);
    this.ensureOverlayParents(normalized);
    this.deleted.delete(normalized);
  }

  async mkdir(path: string, options?: MkdirOptions): Promise<void> {
    const normalized = normalizePath(path);

    if (options?.recursive) {
      // mkdir -p: create all parent dirs, silently succeed if exists
      const parts = normalized.split("/").filter(Boolean);
      for (let i = 1; i <= parts.length; i++) {
        const dir = "/" + parts.slice(0, i).join("/");
        this.overlayDirs.add(dir);
        this.deleted.delete(dir);
      }
      return;
    }

    // Check if already exists
    const exists = await this.exists(normalized);
    if (exists) throw eexist("mkdir", path);

    this.overlayDirs.add(normalized);
    this.ensureOverlayParents(normalized);
    this.deleted.delete(normalized);
  }

  async rm(path: string, options?: RmOptions): Promise<void> {
    const normalized = normalizePath(path);

    const exists = await this.exists(normalized);
    if (!exists) {
      if (options?.force) return;
      throw enoent("rm", path);
    }

    // Remove from overlay
    this.overlay.delete(normalized);
    this.overlayDirs.delete(normalized);

    // If recursive, remove all children from overlay
    if (options?.recursive) {
      const prefix = normalized + "/";
      for (const key of [...this.overlay.keys()]) {
        if (key.startsWith(prefix)) this.overlay.delete(key);
      }
      for (const key of [...this.overlayDirs]) {
        if (key.startsWith(prefix)) this.overlayDirs.delete(key);
      }
      // Also remove children from deleted set since parent covers them
      for (const key of [...this.deleted]) {
        if (key.startsWith(prefix)) this.deleted.delete(key);
      }
    }

    // Mark as deleted (so git tree entries are hidden)
    this.deleted.add(normalized);
  }

  async cp(src: string, dest: string, options?: CpOptions): Promise<void> {
    const srcStat = await this.stat(src);
    if (srcStat.isDirectory) {
      if (!options?.recursive) {
        const err = new Error(`cp: ${src} is a directory (not copied)`);
        throw err;
      }
      await this.mkdir(dest, { recursive: true });
      const children = await this.readdir(src);
      for (const child of children) {
        const srcChild = src === "/" ? `/${child}` : `${src}/${child}`;
        const destChild = dest === "/" ? `/${child}` : `${dest}/${child}`;
        await this.cp(srcChild, destChild, options);
      }
    } else {
      const content = await this.readFileBuffer(src);
      await this.writeFile(dest, content);
    }
  }

  async mv(src: string, dest: string): Promise<void> {
    await this.cp(src, dest, { recursive: true });
    await this.rm(src, { recursive: true });
  }

  async chmod(_path: string, _mode: number): Promise<void> {
    // No-op for overlay (git doesn't track fine-grained permissions)
  }

  async symlink(_target: string, _linkPath: string): Promise<void> {
    // Symlink creation in overlay not supported — store as a regular file
    const normalized = normalizePath(_linkPath);
    const buf = textEncoder.encode(_target);
    this.overlay.set(normalized, buf);
    this.ensureOverlayParents(normalized);
    this.deleted.delete(normalized);
  }

  async link(existingPath: string, newPath: string): Promise<void> {
    // Hard link: just copy the content
    const content = await this.readFileBuffer(existingPath);
    await this.writeFile(newPath, content);
  }

  async utimes(_path: string, _atime: Date, _mtime: Date): Promise<void> {
    // No-op — overlay doesn't track timestamps
  }

  // ---- Commit and push overlay changes ----

  /**
   * Materialize overlay into the git working directory, commit, and push.
   * After a successful push, clears the overlay and updates commitOid.
   */
  async commitAndPush(
    message: string,
    author: { name: string; email: string },
    onAuth?: () => { username: string; password?: string }
  ): Promise<void> {
    await this.commit(message, author);
    await this.push(onAuth);
  }
}
