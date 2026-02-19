/**
 * GitRepo — Git repository core: protocol operations, state management, R2 persistence.
 *
 * Uses isomorphic-git to shallow-clone (depth=1, noCheckout) into an in-memory
 * fs adapter. Pack data and metadata are stored in R2 via R2FsAdapter.
 *
 * Overlay writes are tracked by marking deleted paths and persisting state
 * in /.git/meta.json. The overlay files themselves live in R2FsAdapter.
 */

import git from "isomorphic-git";
import http from "isomorphic-git/http/web";
import { R2FsAdapter } from "./r2-fs-adapter";
import { createIsomorphicGitMemFs } from "./isomorphic-git-memfs";

// ---- Public types ----

export interface GitStatus {
  modified: string[];
  added: string[];
  deleted: string[];
}

export interface GitTreeEntry {
  mode: string;
  path: string;
  oid: string;
  type: "blob" | "tree" | "commit";
}

export interface LogEntry {
  oid: string;
  message: string;
  author: { name: string; email: string; timestamp: number };
  committer: { name: string; email: string; timestamp: number };
}

// ---- Helpers ----

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

// ---- GitRepo options ----

export interface GitRepoOptions {
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
  depth?: number;
}

// ---- GitRepo ----

export class GitRepo {
  private url: string;
  private ref: string | undefined;
  private corsProxy?: string;
  private depth: number | undefined;
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
  readonly r2Overlay: R2FsAdapter;
  private deleted = new Set<string>(); // deleted paths (persisted in meta.json)

  private static readonly DIR = "/repo";
  private static readonly GITDIR = "/repo/.git";

  constructor(opts: GitRepoOptions) {
    this.url = opts.url;
    this.ref = opts.ref;
    this.corsProxy = opts.corsProxy;
    this.depth = opts.depth;
    this.onAuth = opts.onAuth;
    this.httpTransport = opts.http ?? http;
    this.r2Overlay = new R2FsAdapter(
      opts.r2Bucket,
      opts.userId,
      opts.mountPoint
    );
  }

  // ---- Init ----

  async init(): Promise<void> {
    return this.ensureInitialized();
  }

  private async ensureInitialized(): Promise<void> {
    if (this.commitOid) return;
    if (!this.initPromise) {
      this.initPromise = this.doInit().catch((err) => {
        this.initPromise = null;
        throw err;
      });
    }
    await this.initPromise;
  }

  private async doInit(): Promise<void> {
    // Try restoring from R2 first (survives hibernation)
    const meta = await this.loadMetadata();
    if (meta && meta.depth === this.depth) {
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
      dir: GitRepo.DIR,
      gitdir: GitRepo.GITDIR,
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
      gitdir: GitRepo.GITDIR,
      ref: "HEAD"
    });

    // Read commit timestamp for mtime
    const [commit] = await git.log({
      fs: this.memFs,
      gitdir: GitRepo.GITDIR,
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

  // ---- R2 pack/metadata serialization ----

  private async savePackToR2(): Promise<void> {
    const snap = this.memFs._snapshot();
    await this.r2Overlay.writeFile("/.git/pack.json", JSON.stringify(snap));
  }

  private async loadPackFromR2(): Promise<boolean> {
    try {
      const data = await this.r2Overlay.readFile("/.git/pack.json", {
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
      deleted: [...this.deleted],
      depth: this.depth
    };
    await this.r2Overlay.writeFile("/.git/meta.json", JSON.stringify(meta));
  }

  private async loadMetadata(): Promise<GitMetadata | null> {
    try {
      const data = await this.r2Overlay.readFile("/.git/meta.json", {
        encoding: "utf8"
      });
      return JSON.parse(data as string) as GitMetadata;
    } catch {
      return null;
    }
  }

  // ---- Public: state queries ----

  getUrl(): string {
    return this.url;
  }

  getRef(): string | undefined {
    return this.ref;
  }

  getCommitOid(): string | null {
    return this.commitOid;
  }

  hasUnpushedCommits(): boolean {
    return this.commitOid !== this.remoteOid;
  }

  getCommitMtime(): Date {
    return this.commitMtime;
  }

  async isDirty(): Promise<boolean> {
    if (this.deleted.size > 0) return true;
    try {
      const entries = await this.r2Overlay.readdir("/");
      return entries.some((e) => e !== ".git");
    } catch {
      return false;
    }
  }

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
          gitdir: GitRepo.GITDIR,
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

  async getLog(maxCount?: number): Promise<LogEntry[]> {
    await this.ensureInitialized();
    const commits = await git.log({
      fs: this.memFs,
      gitdir: GitRepo.GITDIR,
      ref: "HEAD",
      depth: maxCount,
      cache: this.cache
    });
    return commits.map((c) => ({
      oid: c.oid,
      message: c.commit.message,
      author: {
        name: c.commit.author.name,
        email: c.commit.author.email,
        timestamp: c.commit.author.timestamp
      },
      committer: {
        name: c.commit.committer.name,
        email: c.commit.committer.email,
        timestamp: c.commit.committer.timestamp
      }
    }));
  }

  // ---- Overlay state ----

  isDeleted(normalizedPath: string): boolean {
    if (this.deleted.has(normalizedPath)) return true;
    const parts = normalizedPath.split("/").filter(Boolean);
    for (let i = 1; i < parts.length; i++) {
      const ancestor = "/" + parts.slice(0, i).join("/");
      if (this.deleted.has(ancestor)) return true;
    }
    return false;
  }

  markDeleted(path: string): void {
    this.deleted.add(path);
  }

  /** Remove exact path from deleted set. Returns true if it was present. */
  unmarkDeleted(path: string): boolean {
    return this.deleted.delete(path);
  }

  /** Remove child entries from deleted set (when parent is being deleted recursively). */
  cleanChildDeletedEntries(path: string): void {
    const prefix = path + "/";
    for (const key of [...this.deleted]) {
      if (key.startsWith(prefix)) this.deleted.delete(key);
    }
  }

  async persistDeletedState(): Promise<void> {
    await this.saveMetadata();
  }

  async getOverlayChildren(dirPath: string): Promise<Set<string>> {
    const children = new Set<string>();
    try {
      const entries = await this.r2Overlay.readdir(dirPath);
      for (const name of entries) {
        if (name !== ".git") children.add(name);
      }
    } catch {
      // ENOENT — no overlay dir
    }
    return children;
  }

  // ---- Git tree reading ----

  async readBlob(filepath: string): Promise<Uint8Array> {
    const gitPath = toGitPath(normalizePath(filepath));
    await this.ensureInitialized();

    const { blob } = await git.readBlob({
      fs: this.memFs,
      gitdir: GitRepo.GITDIR,
      oid: this.commitOid!,
      filepath: gitPath,
      cache: this.cache
    });
    return blob;
  }

  async findEntry(filepath: string): Promise<GitTreeEntry | null> {
    const normalized = normalizePath(filepath);
    if (normalized === "/") {
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

  async readTreeEntries(dirPath: string): Promise<GitTreeEntry[]> {
    const gitPath = toGitPath(dirPath);
    const cacheKey = gitPath;
    const cached = this.treeCache.get(cacheKey);
    if (cached) return cached;

    await this.ensureInitialized();

    const result = await git.readTree({
      fs: this.memFs,
      gitdir: GitRepo.GITDIR,
      oid: this.commitOid!,
      filepath: gitPath || undefined,
      cache: this.cache
    });
    const entries = result.tree as GitTreeEntry[];
    this.treeCache.set(cacheKey, entries);
    return entries;
  }

  // ---- Git operations ----

  async commit(
    message: string,
    author: { name: string; email: string }
  ): Promise<string> {
    await this.ensureInitialized();

    const dirty = await this.isDirty();
    if (!dirty) {
      throw new Error("nothing to commit, working tree clean");
    }

    const DIR = GitRepo.DIR;

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
      const content = await this.readOverlayFileBuffer(path);
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

  async push(
    onAuth?: () => { username: string; password?: string }
  ): Promise<void> {
    await this.ensureInitialized();

    await git.push({
      fs: this.memFs,
      http: this.httpTransport,
      dir: GitRepo.DIR,
      onAuth: onAuth ?? this.onAuth,
      cache: this.cache
    });

    this.remoteOid = this.commitOid;
    await this.saveMetadata();
  }

  // ---- Internal helpers ----

  private async getOverlayFiles(): Promise<string[]> {
    const files: string[] = [];
    const walk = async (dir: string) => {
      let entries: string[];
      try {
        entries = await this.r2Overlay.readdir(dir);
      } catch {
        return;
      }
      for (const name of entries) {
        if (name === ".git") continue;
        const childPath = dir === "/" ? `/${name}` : `${dir}/${name}`;
        try {
          const st = await this.r2Overlay.stat(childPath);
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

  private async readOverlayFileBuffer(path: string): Promise<Uint8Array> {
    return this.r2Overlay.readFileBuffer(path);
  }

  private async clearR2Overlay(): Promise<void> {
    try {
      const entries = await this.r2Overlay.readdir("/");
      for (const name of entries) {
        if (name === ".git") continue;
        await this.r2Overlay.rm(`/${name}`, { recursive: true, force: true });
      }
    } catch {
      // no entries
    }
  }
}
