/**
 * AgentFsAdapter — just-bash IFileSystem backed by AgentFS (Cloudflare DO SQLite).
 *
 * This is a local copy of AgentFsWrapper from agentfs-sdk/just-bash with:
 * - The problematic `import { AgentFS } from "../../index_node.js"` removed
 *   (that import pulls in Node.js native bindings which break Cloudflare Workers)
 * - Added realpath() and utimes() methods required by just-bash IFileSystem
 */

import type { AgentFS } from "agentfs-sdk/cloudflare";

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

// ---- Helpers (from agentfs-sdk/just-bash/AgentFs.js) ----

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

// ---- AgentFsAdapter ----

export class AgentFsAdapter {
  private agentFs: AgentFS;

  constructor(agentFs: AgentFS) {
    this.agentFs = agentFs;
  }

  private normalizePath(path: string): string {
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

  private dirname(path: string): string {
    const normalized = this.normalizePath(path);
    if (normalized === "/") return "/";
    const lastSlash = normalized.lastIndexOf("/");
    return lastSlash === 0 ? "/" : normalized.slice(0, lastSlash);
  }

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
    try {
      const data = await this.agentFs.readFile(normalized);
      if (typeof data === "string") return textEncoder.encode(data);
      return new Uint8Array(data);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("not found") || msg.includes("ENOENT")) {
        throw new Error(`ENOENT: no such file or directory, open '${path}'`);
      }
      throw e;
    }
  }

  async writeFile(
    path: string,
    content: string | Uint8Array,
    options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    const normalized = this.normalizePath(path);
    const encoding = getEncoding(options);
    const buffer = toBuffer(content, encoding);
    await this.agentFs.writeFile(normalized, Buffer.from(buffer));
  }

  async appendFile(
    path: string,
    content: string | Uint8Array,
    options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    const normalized = this.normalizePath(path);
    const encoding = getEncoding(options);
    const newBuffer = toBuffer(content, encoding);
    let existingBuffer: Uint8Array;
    try {
      existingBuffer = await this.readFileBuffer(normalized);
    } catch {
      existingBuffer = new Uint8Array(0);
    }
    const combined = new Uint8Array(existingBuffer.length + newBuffer.length);
    combined.set(existingBuffer);
    combined.set(newBuffer, existingBuffer.length);
    await this.writeFile(normalized, combined);
  }

  async exists(path: string): Promise<boolean> {
    const normalized = this.normalizePath(path);
    try {
      await this.agentFs.access(normalized);
      return true;
    } catch {
      return false;
    }
  }

  async stat(path: string): Promise<FsStat> {
    const normalized = this.normalizePath(path);
    try {
      const stats = await this.agentFs.stat(normalized);
      return {
        isFile: stats.isFile(),
        isDirectory: stats.isDirectory(),
        isSymbolicLink: stats.isSymbolicLink(),
        mode: stats.mode,
        size: stats.size,
        mtime: new Date((stats.mtime as number) * 1000)
      };
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("not found") || msg.includes("ENOENT")) {
        throw new Error(`ENOENT: no such file or directory, stat '${path}'`);
      }
      throw e;
    }
  }

  async lstat(path: string): Promise<FsStat> {
    return this.stat(path);
  }

  async mkdir(path: string, options?: MkdirOptions): Promise<void> {
    const normalized = this.normalizePath(path);
    if (options?.recursive) {
      const parent = this.dirname(normalized);
      if (parent !== "/" && parent !== normalized) {
        const parentExists = await this.exists(parent);
        if (!parentExists) await this.mkdir(parent, { recursive: true });
      }
    }
    try {
      await this.agentFs.mkdir(normalized);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("EEXIST") || msg.includes("already exists")) {
        if (!options?.recursive) {
          throw new Error(`EEXIST: file already exists, mkdir '${path}'`);
        }
        return;
      }
      if (msg.includes("ENOENT") || msg.includes("not found")) {
        throw new Error(`ENOENT: no such file or directory, mkdir '${path}'`);
      }
      throw e;
    }
  }

  async readdir(path: string): Promise<string[]> {
    const normalized = this.normalizePath(path);
    try {
      const entries = await this.agentFs.readdir(normalized);
      return entries.sort();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("not found") || msg.includes("ENOENT")) {
        throw new Error(`ENOENT: no such file or directory, scandir '${path}'`);
      }
      throw e;
    }
  }

  async rm(path: string, options?: RmOptions): Promise<void> {
    const normalized = this.normalizePath(path);
    try {
      await this.agentFs.rm(normalized, {
        force: options?.force,
        recursive: options?.recursive
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("ENOENT") || msg.includes("not found")) {
        if (!options?.force) {
          throw new Error(`ENOENT: no such file or directory, rm '${path}'`);
        }
        return;
      }
      if (msg.includes("ENOTEMPTY") || msg.includes("not empty")) {
        throw new Error(`ENOTEMPTY: directory not empty, rm '${path}'`);
      }
      if (msg.includes("EISDIR")) {
        try {
          await this.agentFs.rmdir(normalized);
          return;
        } catch (rmdirErr) {
          const rmdirMsg =
            rmdirErr instanceof Error ? rmdirErr.message : String(rmdirErr);
          if (rmdirMsg.includes("ENOTEMPTY") || rmdirMsg.includes("not empty"))
            throw new Error(`ENOTEMPTY: directory not empty, rm '${path}'`);
          throw rmdirErr;
        }
      }
      throw e;
    }
  }

  async cp(src: string, dest: string, options?: CpOptions): Promise<void> {
    const srcNorm = this.normalizePath(src);
    const destNorm = this.normalizePath(dest);
    const srcStat = await this.stat(srcNorm);
    if (srcStat.isFile) {
      await this.agentFs.copyFile(srcNorm, destNorm);
    } else if (srcStat.isDirectory) {
      if (!options?.recursive) {
        throw new Error(`EISDIR: is a directory, cp '${src}'`);
      }
      await this.mkdir(destNorm, { recursive: true });
      const children = await this.readdir(srcNorm);
      for (const child of children) {
        const srcChild = srcNorm === "/" ? `/${child}` : `${srcNorm}/${child}`;
        const destChild =
          destNorm === "/" ? `/${child}` : `${destNorm}/${child}`;
        await this.cp(srcChild, destChild, options);
      }
    }
  }

  async mv(src: string, dest: string): Promise<void> {
    const srcNorm = this.normalizePath(src);
    const destNorm = this.normalizePath(dest);
    try {
      await this.agentFs.rename(srcNorm, destNorm);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("ENOENT") || msg.includes("not found")) {
        throw new Error(`ENOENT: no such file or directory, mv '${src}'`);
      }
      throw e;
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

  async symlink(target: string, linkPath: string): Promise<void> {
    const normalized = this.normalizePath(linkPath);
    const pathExists = await this.exists(normalized);
    if (pathExists) {
      throw new Error(`EEXIST: file already exists, symlink '${linkPath}'`);
    }
    await this.agentFs.symlink(target, normalized);
  }

  async link(existingPath: string, newPath: string): Promise<void> {
    const existingNorm = this.normalizePath(existingPath);
    const newNorm = this.normalizePath(newPath);
    const existingStat = await this.stat(existingNorm);
    if (!existingStat.isFile) {
      throw new Error(`EPERM: operation not permitted, link '${existingPath}'`);
    }
    const newExists = await this.exists(newNorm);
    if (newExists) {
      throw new Error(`EEXIST: file already exists, link '${newPath}'`);
    }
    await this.agentFs.copyFile(existingNorm, newNorm);
  }

  async readlink(path: string): Promise<string> {
    const normalized = this.normalizePath(path);
    return await this.agentFs.readlink(normalized);
  }

  async realpath(path: string): Promise<string> {
    await this.stat(path);
    return this.resolvePath("/", path);
  }

  async utimes(path: string, _atime: Date, _mtime: Date): Promise<void> {
    await this.stat(path);
  }
}
