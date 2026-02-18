/**
 * GitFs — IFileSystem backed by a remote Git repository with overlay writes.
 *
 * Uses isomorphic-git to shallow-clone (depth=1, noCheckout) into an in-memory
 * fs adapter, then serves readdir/readFile/stat directly from the packfile.
 *
 * Overlay writes are persisted to R2 via R2FsAdapter, surviving DO hibernation.
 * Git pack data and metadata are stored as /.git/pack.json and /.git/meta.json.
 */

import git from "isomorphic-git";
import http from "isomorphic-git/http/web";
import type {
  BufferEncoding,
  ReadFileOptions,
  WriteFileOptions,
  FsStat,
  MkdirOptions,
  RmOptions,
  CpOptions
} from "just-bash";
import { toBuffer, fromBuffer, getEncoding } from "just-bash";
import { R2FsAdapter } from "./r2-fs-adapter";
import { createIsomorphicGitMemFs } from "./isomorphic-git-memfs";

// ---- Public types ----

export interface GitStatus {
  modified: string[];
  added: string[];
  deleted: string[];
}

// ---- Helpers ----

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

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

/** Check if a normalized path is inside /.git/ (internal state, not user files). */
function isGitInternal(normalized: string): boolean {
  return normalized === "/.git" || normalized.startsWith("/.git/");
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
  r2Bucket: R2Bucket;
  userId: string;
  mountPoint: string; // e.g. "/mnt/repo"
}

interface GitMetadata {
  commitOid: string;
  remoteOid: string | null;
  ref: string;
  url: string;
  commitMtime: number; // epoch ms
  deleted: string[];
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

  // ---- Overlay persisted in R2 ----
  private r2Fs: R2FsAdapter;
  private deleted = new Set<string>(); // deleted paths (persisted in meta.json)

  private static readonly DIR = "/repo";
  private static readonly GITDIR = "/repo/.git";

  constructor(opts: GitFsOptions) {
    this.url = opts.url;
    this.ref = opts.ref;
    this.corsProxy = opts.corsProxy;
    this.depth = opts.depth ?? 1;
    this.onAuth = opts.onAuth;
    this.httpTransport = opts.http ?? http;
    this.r2Fs = new R2FsAdapter(opts.r2Bucket, opts.userId, opts.mountPoint);
  }

  // ---- Public: overlay status and URL ----

  /** Returns true if there are uncommitted overlay changes. */
  async isDirty(): Promise<boolean> {
    if (this.deleted.size > 0) return true;
    try {
      const entries = await this.r2Fs.readdir("/");
      return entries.some((e) => e !== ".git");
    } catch {
      return false;
    }
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

    const overlayFiles = await this.getOverlayFiles();

    for (const path of overlayFiles) {
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

  /** Get all overlay file paths (excluding .git/ internals). */
  private async getOverlayFiles(): Promise<string[]> {
    const files: string[] = [];
    const walk = async (dir: string) => {
      let entries: string[];
      try {
        entries = await this.r2Fs.readdir(dir);
      } catch {
        return;
      }
      for (const name of entries) {
        if (name === ".git") continue;
        const childPath = dir === "/" ? `/${name}` : `${dir}/${name}`;
        try {
          const st = await this.r2Fs.stat(childPath);
          if (st.isDirectory) {
            await walk(childPath);
          } else {
            files.push(childPath);
          }
        } catch {
          // skip
        }
      }
    };
    await walk("/");
    return files;
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

    const dirty = await this.isDirty();
    if (!dirty) {
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
    const overlayFiles = await this.getOverlayFiles();
    for (const path of overlayFiles) {
      const content = await this.readFileBuffer(path);
      const fsPath = `${DIR}${path}`;
      await this.memFs.writeFile(fsPath, content);
    }

    // 3. Stage added/modified files
    for (const path of overlayFiles) {
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
    await this.clearR2Overlay();
    this.deleted.clear();

    // 7. Persist pack + metadata to R2
    await this.savePackToR2();
    await this.saveMetadata();

    return this.commitOid;
  }

  /** Remove all non-.git/ overlay entries from R2. */
  private async clearR2Overlay(): Promise<void> {
    try {
      const entries = await this.r2Fs.readdir("/");
      for (const name of entries) {
        if (name === ".git") continue;
        await this.r2Fs.rm(`/${name}`, { recursive: true, force: true });
      }
    } catch {
      // no entries
    }
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
    await this.saveMetadata();
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

  // ---- R2 pack/metadata serialization ----

  private async savePackToR2(): Promise<void> {
    const snap = this.memFs._snapshot();
    await this.r2Fs.writeFile("/.git/pack.json", JSON.stringify(snap));
  }

  private async loadPackFromR2(): Promise<boolean> {
    try {
      const data = await this.r2Fs.readFile("/.git/pack.json", {
        encoding: "utf8"
      });
      const snap = JSON.parse(data as string);
      this.memFs._restore(snap);
      return true;
    } catch {
      return false;
    }
  }

  private async saveMetadata(): Promise<void> {
    const meta: GitMetadata = {
      commitOid: this.commitOid!,
      remoteOid: this.remoteOid,
      ref: this.ref!,
      url: this.url,
      commitMtime: this.commitMtime.getTime(),
      deleted: [...this.deleted]
    };
    await this.r2Fs.writeFile("/.git/meta.json", JSON.stringify(meta));
  }

  private async loadMetadata(): Promise<GitMetadata | null> {
    try {
      const data = await this.r2Fs.readFile("/.git/meta.json", {
        encoding: "utf8"
      });
      return JSON.parse(data as string) as GitMetadata;
    } catch {
      return null;
    }
  }

  private async doInit(): Promise<void> {
    // Try restoring from R2 first (survives hibernation)
    const meta = await this.loadMetadata();
    if (meta) {
      const packLoaded = await this.loadPackFromR2();
      if (packLoaded) {
        this.commitOid = meta.commitOid;
        this.remoteOid = meta.remoteOid;
        this.ref = meta.ref;
        this.commitMtime = new Date(meta.commitMtime);
        this.deleted = new Set(meta.deleted);
        return;
      }
    }

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

    // Persist to R2 after initial clone
    await this.savePackToR2();
    await this.saveMetadata();
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
   * Excludes .git/ internal entries.
   */
  private async getOverlayChildren(dirPath: string): Promise<Set<string>> {
    const children = new Set<string>();
    try {
      const entries = await this.r2Fs.readdir(dirPath);
      for (const name of entries) {
        if (name !== ".git") children.add(name);
      }
    } catch {
      // ENOENT — no overlay dir
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

    // Check R2 overlay first (skip .git/ internals)
    if (!isGitInternal(normalized)) {
      try {
        return await this.r2Fs.readFileBuffer(normalized);
      } catch {
        // not in overlay — fall through
      }
    }

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

    // .git/ internals are hidden from the user
    if (isGitInternal(normalized)) return false;

    // Check R2 overlay
    try {
      if (await this.r2Fs.exists(normalized)) return true;
    } catch {
      // fall through
    }

    // Check if deleted
    if (this.isDeleted(normalized)) return false;

    // Check if overlay children exist under this path (making it a virtual dir)
    const overlayChildren = await this.getOverlayChildren(normalized);
    if (overlayChildren.size > 0) return true;

    // Fall through to git tree
    const entry = await this.findEntry(path);
    return entry !== null;
  }

  async stat(path: string): Promise<FsStat> {
    const normalized = normalizePath(path);

    // Check R2 overlay first (skip .git/ internals)
    if (!isGitInternal(normalized)) {
      try {
        return await this.r2Fs.stat(normalized);
      } catch {
        // not in R2 overlay — fall through
      }
    }

    if (normalized === "/") {
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
    const overlayChildren = await this.getOverlayChildren(normalized);

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
    const overlayChildren = await this.getOverlayChildren(normalized);

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
        if (overlayChildren.size === 0 && normalized !== "/") {
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
    } else if (overlayChildren.size === 0) {
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

  // ---- IFileSystem: write operations (R2-backed) ----

  async writeFile(
    path: string,
    content: string | Uint8Array,
    options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    const normalized = normalizePath(path);
    const buf = toBuffer(content, getEncoding(options));
    await this.r2Fs.writeFile(normalized, buf);
    if (this.deleted.delete(normalized)) {
      await this.saveMetadata();
    }
  }

  async appendFile(
    path: string,
    content: string | Uint8Array,
    options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    const normalized = normalizePath(path);
    const appendBuf = toBuffer(content, getEncoding(options));

    // Read existing content (R2 overlay or git tree)
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

    await this.r2Fs.writeFile(normalized, merged);
    if (this.deleted.delete(normalized)) {
      await this.saveMetadata();
    }
  }

  async mkdir(path: string, options?: MkdirOptions): Promise<void> {
    const normalized = normalizePath(path);

    if (options?.recursive) {
      await this.r2Fs.mkdir(normalized, { recursive: true });
      const parts = normalized.split("/").filter(Boolean);
      let changed = false;
      for (let i = 1; i <= parts.length; i++) {
        const dir = "/" + parts.slice(0, i).join("/");
        if (this.deleted.delete(dir)) changed = true;
      }
      if (changed) await this.saveMetadata();
      return;
    }

    const pathExists = await this.exists(normalized);
    if (pathExists) throw eexist("mkdir", path);
    // Use recursive: true for R2 to avoid double-normalization issue in R2FsAdapter
    await this.r2Fs.mkdir(normalized, { recursive: true });
    if (this.deleted.delete(normalized)) {
      await this.saveMetadata();
    }
  }

  async rm(path: string, options?: RmOptions): Promise<void> {
    const normalized = normalizePath(path);

    const pathExists = await this.exists(normalized);
    if (!pathExists) {
      if (options?.force) return;
      throw enoent("rm", path);
    }

    // Remove from R2 overlay
    try {
      await this.r2Fs.rm(normalized, options);
    } catch {
      // may not exist in R2 (only in git tree)
    }

    // Clean child deleted entries when recursive
    if (options?.recursive) {
      const prefix = normalized + "/";
      for (const key of [...this.deleted]) {
        if (key.startsWith(prefix)) this.deleted.delete(key);
      }
    }

    // Mark as deleted (so git tree entries are hidden)
    this.deleted.add(normalized);
    await this.saveMetadata();
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
    await this.r2Fs.writeFile(normalized, buf);
    if (this.deleted.delete(normalized)) {
      await this.saveMetadata();
    }
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
