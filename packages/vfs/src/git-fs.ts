/**
 * GitFs — IFileSystem adapter backed by a remote Git repository with overlay writes.
 *
 * Composes a GitRepo instance for git protocol/state and implements IFileSystem
 * by merging overlay writes (R2) with the git tree, respecting deleted paths.
 */

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
import { GitRepo } from "./git-repo";
import type { GitStatus, LogEntry } from "./git-repo";

// Re-export types so existing consumers keep working
export type { GitStatus } from "./git-repo";

// ---- Helpers ----

const textEncoder = new TextEncoder();

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

// ---- GitFs options (same shape as before for backwards compat) ----

export interface GitFsOptions {
  url: string;
  ref?: string;
  corsProxy?: string;
  depth?: number;
  onAuth?: () => { username: string; password?: string };
  http?: any;
  r2Bucket: R2Bucket;
  userId: string;
  mountPoint: string;
}

// ---- GitFs ----

export class GitFs {
  private repo: GitRepo;

  constructor(opts: GitFsOptions) {
    this.repo = new GitRepo(opts);
  }

  // ---- Proxy: git operations ----

  async init(): Promise<void> {
    return this.repo.init();
  }

  async commit(
    message: string,
    author: { name: string; email: string }
  ): Promise<string> {
    return this.repo.commit(message, author);
  }

  async push(
    onAuth?: () => { username: string; password?: string }
  ): Promise<void> {
    return this.repo.push(onAuth);
  }

  async pull(
    onAuth?: () => { username: string; password?: string }
  ): Promise<{ updated: boolean; fromOid: string; toOid: string }> {
    return this.repo.pull(onAuth);
  }

  async commitAndPush(
    message: string,
    author: { name: string; email: string },
    onAuth?: () => { username: string; password?: string }
  ): Promise<void> {
    await this.repo.commit(message, author);
    await this.repo.push(onAuth);
  }

  // ---- Proxy: state queries ----

  getUrl(): string {
    return this.repo.getUrl();
  }

  getRef(): string | undefined {
    return this.repo.getRef();
  }

  getCommitOid(): string | null {
    return this.repo.getCommitOid();
  }

  hasUnpushedCommits(): boolean {
    return this.repo.hasUnpushedCommits();
  }

  async isDirty(): Promise<boolean> {
    return this.repo.isDirty();
  }

  async getStatus(): Promise<GitStatus> {
    return this.repo.getStatus();
  }

  async getLog(maxCount?: number): Promise<LogEntry[]> {
    return this.repo.getLog(maxCount);
  }

  async readBlobUtf8(path: string): Promise<string> {
    return this.repo.readBlobUtf8(path);
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
        return await this.repo.r2Overlay.readFileBuffer(normalized);
      } catch {
        // not in overlay — fall through
      }
    }

    // Check if deleted
    if (this.repo.isDeleted(normalized)) throw enoent("open", path);

    // Fall through to git tree
    try {
      return await this.repo.readBlob(normalized);
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

    if (isGitInternal(normalized)) return false;

    // Check R2 overlay
    try {
      if (await this.repo.r2Overlay.exists(normalized)) return true;
    } catch {
      // fall through
    }

    // Check if deleted
    if (this.repo.isDeleted(normalized)) return false;

    // Check if overlay children exist under this path (making it a virtual dir)
    const overlayChildren = await this.repo.getOverlayChildren(normalized);
    if (overlayChildren.size > 0) return true;

    // Fall through to git tree
    const entry = await this.repo.findEntry(path);
    return entry !== null;
  }

  async stat(path: string): Promise<FsStat> {
    const normalized = normalizePath(path);

    // Check R2 overlay first (skip .git/ internals)
    if (!isGitInternal(normalized)) {
      try {
        return await this.repo.r2Overlay.stat(normalized);
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
    const overlayChildren = await this.repo.getOverlayChildren(normalized);

    // Check if deleted
    if (this.repo.isDeleted(normalized)) {
      if (overlayChildren.size > 0) {
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
      try {
        const entry = await this.repo.findEntry(normalized);
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
            mtime: this.repo.getCommitMtime()
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
    const entry = await this.repo.findEntry(normalized);
    if (!entry) throw enoent("stat", path);

    const isDir = entry.type === "tree";
    const isSymlink = entry.mode === "120000";
    const mode = parseInt(entry.mode, 8);
    return {
      isFile: entry.type === "blob" && !isSymlink,
      isDirectory: isDir,
      isSymbolicLink: isSymlink,
      mode,
      size: 0,
      mtime: this.repo.getCommitMtime()
    };
  }

  async lstat(path: string): Promise<FsStat> {
    return this.stat(path);
  }

  async readdir(path: string): Promise<string[]> {
    const normalized = normalizePath(path);

    const overlayChildren = await this.repo.getOverlayChildren(normalized);

    const pathDeleted = this.repo.isDeleted(normalized);

    let gitEntryNames: string[] = [];
    if (!pathDeleted) {
      try {
        const entry = await this.repo.findEntry(normalized);
        if (entry && entry.type === "tree") {
          const entries = await this.repo.readTreeEntries(normalized);
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
      if (!this.repo.isDeleted(childPath)) {
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

    if (this.repo.isDeleted(normalized)) throw enoent("readlink", path);

    const entry = await this.repo.findEntry(normalized);
    if (!entry) throw enoent("readlink", path);
    if (entry.mode !== "120000") {
      const err = new Error(`EINVAL: invalid argument, readlink '${path}'`);
      (err as any).code = "EINVAL";
      throw err;
    }
    const blob = await this.readFileBuffer(path);
    return new TextDecoder().decode(blob);
  }

  async realpath(path: string): Promise<string> {
    await this.stat(path);
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
    await this.repo.r2Overlay.writeFile(normalized, buf);
    if (this.repo.unmarkDeleted(normalized)) {
      await this.repo.persistDeletedState();
    }
  }

  async appendFile(
    path: string,
    content: string | Uint8Array,
    options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    const normalized = normalizePath(path);
    const appendBuf = toBuffer(content, getEncoding(options));

    let existing: Uint8Array;
    try {
      existing = await this.readFileBuffer(path);
    } catch {
      existing = new Uint8Array(0);
    }

    const merged = new Uint8Array(existing.length + appendBuf.length);
    merged.set(existing, 0);
    merged.set(appendBuf, existing.length);

    await this.repo.r2Overlay.writeFile(normalized, merged);
    if (this.repo.unmarkDeleted(normalized)) {
      await this.repo.persistDeletedState();
    }
  }

  async mkdir(path: string, options?: MkdirOptions): Promise<void> {
    const normalized = normalizePath(path);

    if (options?.recursive) {
      await this.repo.r2Overlay.mkdir(normalized, { recursive: true });
      const parts = normalized.split("/").filter(Boolean);
      let changed = false;
      for (let i = 1; i <= parts.length; i++) {
        const dir = "/" + parts.slice(0, i).join("/");
        if (this.repo.unmarkDeleted(dir)) changed = true;
      }
      if (changed) await this.repo.persistDeletedState();
      return;
    }

    const pathExists = await this.exists(normalized);
    if (pathExists) throw eexist("mkdir", path);
    await this.repo.r2Overlay.mkdir(normalized, { recursive: true });
    if (this.repo.unmarkDeleted(normalized)) {
      await this.repo.persistDeletedState();
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
      await this.repo.r2Overlay.rm(normalized, options);
    } catch {
      // may not exist in R2 (only in git tree)
    }

    // Clean child deleted entries when recursive
    if (options?.recursive) {
      this.repo.cleanChildDeletedEntries(normalized);
    }

    // Mark as deleted (so git tree entries are hidden)
    this.repo.markDeleted(normalized);
    await this.repo.persistDeletedState();
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
    // No-op for overlay
  }

  async symlink(_target: string, _linkPath: string): Promise<void> {
    const normalized = normalizePath(_linkPath);
    const buf = textEncoder.encode(_target);
    await this.repo.r2Overlay.writeFile(normalized, buf);
    if (this.repo.unmarkDeleted(normalized)) {
      await this.repo.persistDeletedState();
    }
  }

  async link(existingPath: string, newPath: string): Promise<void> {
    const content = await this.readFileBuffer(existingPath);
    await this.writeFile(newPath, content);
  }

  async utimes(_path: string, _atime: Date, _mtime: Date): Promise<void> {
    // No-op
  }
}
