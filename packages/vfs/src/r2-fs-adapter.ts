/**
 * R2FsAdapter — IFileSystem backed by R2 object storage, scoped to a user.
 *
 * Key mapping:
 *   file:      {userId}{normalizedPath}       e.g. "abc123/data/report.csv"
 *   directory: {userId}{normalizedPath}/      e.g. "abc123/data/docs/"
 *
 * Error messages match ENOENT/EEXIST/etc. strings that just-bash pattern-matches on.
 */

/// <reference types="@cloudflare/workers-types" />

import type {
  IFileSystem,
  BufferEncoding,
  FsStat,
  MkdirOptions,
  RmOptions,
  CpOptions,
  ReadFileOptions,
  WriteFileOptions
} from "just-bash";
import { toBuffer, fromBuffer, getEncoding } from "just-bash";
import {
  parentPath,
  baseName,
  normalizePath as normalizePathBase
} from "./fs-helpers";

export class R2FsAdapter implements IFileSystem {
  private bucket: R2Bucket;
  private userId: string;
  private rootPrefix: string;

  constructor(bucket: R2Bucket, userId: string, rootPrefix: string = "") {
    this.bucket = bucket;
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
    const relativePath = normalizePathBase(path);
    if (this.rootPrefix) {
      return relativePath === "/"
        ? this.rootPrefix
        : `${this.rootPrefix}${relativePath}`;
    }
    return relativePath;
  }

  /** R2 key for a file: {userId}{normalizedPath} */
  private fileKey(normalizedPath: string): string {
    return `${this.userId}${normalizedPath}`;
  }

  /** R2 key for a directory marker: {userId}{normalizedPath}/ */
  private dirKey(normalizedPath: string): string {
    return `${this.userId}${normalizedPath}/`;
  }

  /** R2 prefix for listing children: {userId}{normalizedPath}/ */
  private listPrefix(normalizedPath: string): string {
    return `${this.userId}${normalizedPath}/`;
  }

  private isRootPath(normalizedPath: string): boolean {
    return normalizedPath === (this.rootPrefix || "/");
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

    // Check if it's a directory
    const dirObj = await this.bucket.head(this.dirKey(normalized));
    if (dirObj) {
      throw new Error(
        `EISDIR: illegal operation on a directory, read '${path}'`
      );
    }

    const obj = await this.bucket.get(this.fileKey(normalized));
    if (!obj) {
      throw new Error(`ENOENT: no such file or directory, open '${path}'`);
    }
    return new Uint8Array(await obj.arrayBuffer());
  }

  async writeFile(
    path: string,
    content: string | Uint8Array,
    options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    const normalized = this.normalizePath(path);
    const encoding = getEncoding(options);
    const buffer = toBuffer(content, encoding);

    await this.ensureParentDirs(normalized);

    await this.bucket.put(this.fileKey(normalized), buffer, {
      customMetadata: {
        mode: "33188",
        mtime: String(Math.floor(Date.now() / 1000))
      }
    });
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

    if (this.isRootPath(normalized)) return true;

    // Check file key
    const fileObj = await this.bucket.head(this.fileKey(normalized));
    if (fileObj) return true;

    // Check directory marker
    const dirObj = await this.bucket.head(this.dirKey(normalized));
    if (dirObj) return true;

    // Check implicit directory (any objects with this prefix)
    const listed = await this.bucket.list({
      prefix: this.listPrefix(normalized),
      limit: 1
    });
    return listed.objects.length > 0;
  }

  async stat(path: string): Promise<FsStat> {
    const normalized = this.normalizePath(path);

    // Root of mount always exists as directory
    if (this.isRootPath(normalized)) {
      return {
        isFile: false,
        isDirectory: true,
        isSymbolicLink: false,
        mode: 16877,
        size: 0,
        mtime: new Date()
      };
    }

    // Check as file first
    const fileObj = await this.bucket.head(this.fileKey(normalized));
    if (fileObj) {
      const mode = fileObj.customMetadata?.mode
        ? parseInt(fileObj.customMetadata.mode, 10)
        : 33188;
      const mtime = fileObj.customMetadata?.mtime
        ? new Date(parseInt(fileObj.customMetadata.mtime, 10) * 1000)
        : fileObj.uploaded;
      return {
        isFile: true,
        isDirectory: false,
        isSymbolicLink: false,
        mode,
        size: fileObj.size,
        mtime
      };
    }

    // Check as explicit directory marker
    const dirObj = await this.bucket.head(this.dirKey(normalized));
    if (dirObj) {
      const mode = dirObj.customMetadata?.mode
        ? parseInt(dirObj.customMetadata.mode, 10)
        : 16877;
      const mtime = dirObj.customMetadata?.mtime
        ? new Date(parseInt(dirObj.customMetadata.mtime, 10) * 1000)
        : dirObj.uploaded;
      return {
        isFile: false,
        isDirectory: true,
        isSymbolicLink: false,
        mode,
        size: 0,
        mtime
      };
    }

    // Check as implicit directory (has children)
    const listed = await this.bucket.list({
      prefix: this.listPrefix(normalized),
      limit: 1
    });
    if (listed.objects.length > 0) {
      return {
        isFile: false,
        isDirectory: true,
        isSymbolicLink: false,
        mode: 16877,
        size: 0,
        mtime: new Date()
      };
    }

    throw new Error(`ENOENT: no such file or directory, stat '${path}'`);
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
    const fileObj = await this.bucket.head(this.fileKey(normalized));
    if (fileObj) {
      if (!options?.recursive) {
        throw new Error(`EEXIST: file already exists, mkdir '${path}'`);
      }
      return;
    }
    const dirObj = await this.bucket.head(this.dirKey(normalized));
    if (dirObj) {
      if (!options?.recursive) {
        throw new Error(`EEXIST: file already exists, mkdir '${path}'`);
      }
      return;
    }

    await this.bucket.put(this.dirKey(normalized), new Uint8Array(0), {
      customMetadata: {
        type: "directory",
        mode: "16877",
        mtime: String(Math.floor(Date.now() / 1000))
      }
    });
  }

  async readdir(path: string): Promise<string[]> {
    const normalized = this.normalizePath(path);

    // Verify directory exists (root always exists)
    if (!this.isRootPath(normalized)) {
      const dirObj = await this.bucket.head(this.dirKey(normalized));
      if (!dirObj) {
        // Check implicit directory
        const listed = await this.bucket.list({
          prefix: this.listPrefix(normalized),
          limit: 1
        });
        if (listed.objects.length === 0) {
          throw new Error(
            `ENOENT: no such file or directory, scandir '${path}'`
          );
        }
      }
    }

    const prefix = this.listPrefix(normalized);
    const names = new Set<string>();

    let cursor: string | undefined;
    do {
      const listed = await this.bucket.list({
        prefix,
        delimiter: "/",
        cursor
      });

      // Files: extract basename from key
      for (const obj of listed.objects) {
        const rel = obj.key.slice(prefix.length);
        // Skip directory markers (they end with /)
        if (!rel || rel.endsWith("/")) continue;
        // Only direct children (no further slashes)
        if (!rel.includes("/")) {
          names.add(rel);
        }
      }

      // Subdirectories from delimited prefixes
      for (const dp of listed.delimitedPrefixes) {
        const rel = dp.slice(prefix.length);
        // Remove trailing slash
        const name = rel.endsWith("/") ? rel.slice(0, -1) : rel;
        if (name) names.add(name);
      }

      cursor = listed.truncated ? listed.cursor : undefined;
    } while (cursor);

    return Array.from(names).sort();
  }

  async rm(path: string, options?: RmOptions): Promise<void> {
    const normalized = this.normalizePath(path);

    // Check file
    const fileObj = await this.bucket.head(this.fileKey(normalized));
    if (fileObj) {
      await this.bucket.delete(this.fileKey(normalized));
      return;
    }

    // Check directory
    const dirObj = await this.bucket.head(this.dirKey(normalized));
    const isImplicitDir =
      !dirObj &&
      (
        await this.bucket.list({
          prefix: this.listPrefix(normalized),
          limit: 1
        })
      ).objects.length > 0;

    if (!dirObj && !isImplicitDir) {
      if (!options?.force) {
        throw new Error(`ENOENT: no such file or directory, rm '${path}'`);
      }
      return;
    }

    // It's a directory
    if (!options?.recursive) {
      // Check if directory is empty
      const children = await this.bucket.list({
        prefix: this.listPrefix(normalized),
        delimiter: "/",
        limit: 2 // dir marker + 1 child
      });
      const hasRealChildren =
        children.objects.some((o) => o.key !== this.dirKey(normalized)) ||
        children.delimitedPrefixes.length > 0;
      if (hasRealChildren) {
        throw new Error(`ENOTEMPTY: directory not empty, rm '${path}'`);
      }
    }

    // Delete all objects under prefix (recursive)
    const prefix = this.listPrefix(normalized);
    let cursor: string | undefined;
    do {
      const listed = await this.bucket.list({ prefix, cursor });
      if (listed.objects.length > 0) {
        const keys = listed.objects.map((o) => o.key);
        // R2 delete supports up to 1000 keys at a time
        for (let i = 0; i < keys.length; i += 1000) {
          await this.bucket.delete(keys.slice(i, i + 1000));
        }
      }
      cursor = listed.truncated ? listed.cursor : undefined;
    } while (cursor);

    // Also delete the directory marker itself
    if (dirObj) {
      await this.bucket.delete(this.dirKey(normalized));
    }
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
    // R2 has no rename — copy then delete
    const srcStat = await this.stat(src);
    if (srcStat.isDirectory) {
      await this.cp(src, dest, { recursive: true });
      await this.rm(src, { recursive: true });
    } else {
      const buffer = await this.readFileBuffer(src);
      await this.writeFile(dest, buffer);
      await this.rm(src);
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
    const exists = await this.exists(normalized);
    if (!exists) {
      throw new Error(`ENOENT: no such file or directory, chmod '${path}'`);
    }
    // R2 would require re-uploading to change customMetadata — skip for now
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
    // R2 would require re-uploading to change customMetadata — skip for now
  }

  // ---- Private helpers ----

  private async ensureParentDirs(normalizedPath: string): Promise<void> {
    const parts = normalizedPath.split("/").filter(Boolean);
    if (parts.length <= 1) return;

    // Build list of ancestor paths that need directory markers
    let current = "";
    for (let i = 0; i < parts.length - 1; i++) {
      current += `/${parts[i]}`;
      if (this.isRootPath(current)) continue;

      const dirObj = await this.bucket.head(this.dirKey(current));
      if (!dirObj) {
        await this.bucket.put(this.dirKey(current), new Uint8Array(0), {
          customMetadata: {
            type: "directory",
            mode: "16877",
            mtime: String(Math.floor(Date.now() / 1000))
          }
        });
      }
    }
  }
}
