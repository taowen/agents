import fsPromises from "node:fs/promises";
import type {
  BufferEncoding,
  CpOptions,
  DirentEntry,
  FileContent,
  FsStat,
  IFileSystem,
  MkdirOptions,
  ReadFileOptions,
  RmOptions,
  WriteFileOptions
} from "just-bash";

/**
 * IFileSystem implementation backed by node:fs/promises.
 *
 * Two modes:
 * - rootPath is a Windows root (e.g. "C:\\"): unix paths are translated to Windows absolute paths
 * - rootPath is "" (empty string): pass-through mode, paths are used as-is
 */
export class NodeFsAdapter implements IFileSystem {
  private rootPath: string;

  constructor(rootPath: string) {
    this.rootPath = rootPath;
  }

  /**
   * Convert a Unix-style path to a native path.
   * When rootPath is set, translates "/" → rootPath, "/foo/bar" → rootPath + "foo\bar".
   * When rootPath is empty, returns the path unchanged (pass-through).
   */
  private toNativePath(path: string): string {
    if (!this.rootPath) return path;
    if (path === "/") return this.rootPath;
    const suffix = path.slice(1).replace(/\//g, "\\");
    return this.rootPath + suffix;
  }

  async readFile(
    path: string,
    _options?: ReadFileOptions | BufferEncoding
  ): Promise<string> {
    return fsPromises.readFile(this.toNativePath(path), "utf8");
  }

  async readFileBuffer(path: string): Promise<Uint8Array> {
    const buf = await fsPromises.readFile(this.toNativePath(path));
    return new Uint8Array(buf);
  }

  async writeFile(
    path: string,
    content: FileContent,
    _options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    await fsPromises.writeFile(this.toNativePath(path), content);
  }

  async appendFile(
    path: string,
    content: FileContent,
    _options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    await fsPromises.appendFile(this.toNativePath(path), content);
  }

  async exists(path: string): Promise<boolean> {
    try {
      await fsPromises.stat(this.toNativePath(path));
      return true;
    } catch {
      return false;
    }
  }

  async stat(path: string): Promise<FsStat> {
    const s = await fsPromises.stat(this.toNativePath(path));
    return {
      isFile: s.isFile(),
      isDirectory: s.isDirectory(),
      isSymbolicLink: s.isSymbolicLink(),
      mode: s.mode,
      size: s.size,
      mtime: s.mtime
    };
  }

  async lstat(path: string): Promise<FsStat> {
    const s = await fsPromises.lstat(this.toNativePath(path));
    return {
      isFile: s.isFile(),
      isDirectory: s.isDirectory(),
      isSymbolicLink: s.isSymbolicLink(),
      mode: s.mode,
      size: s.size,
      mtime: s.mtime
    };
  }

  async mkdir(path: string, options?: MkdirOptions): Promise<void> {
    await fsPromises.mkdir(this.toNativePath(path), {
      recursive: !!options?.recursive
    });
  }

  async readdir(path: string): Promise<string[]> {
    return fsPromises.readdir(this.toNativePath(path));
  }

  async readdirWithFileTypes(path: string): Promise<DirentEntry[]> {
    const dirents = await fsPromises.readdir(this.toNativePath(path), {
      withFileTypes: true
    });
    return dirents.map((d) => ({
      name: d.name,
      isFile: d.isFile(),
      isDirectory: d.isDirectory(),
      isSymbolicLink: d.isSymbolicLink()
    }));
  }

  async rm(path: string, options?: RmOptions): Promise<void> {
    await fsPromises.rm(this.toNativePath(path), {
      recursive: !!options?.recursive,
      force: !!options?.force
    });
  }

  async cp(src: string, dest: string, options?: CpOptions): Promise<void> {
    await fsPromises.cp(this.toNativePath(src), this.toNativePath(dest), {
      recursive: !!options?.recursive
    });
  }

  async mv(src: string, dest: string): Promise<void> {
    await fsPromises.rename(this.toNativePath(src), this.toNativePath(dest));
  }

  async chmod(path: string, mode: number): Promise<void> {
    await fsPromises.chmod(this.toNativePath(path), mode);
  }

  async symlink(target: string, linkPath: string): Promise<void> {
    await fsPromises.symlink(target, this.toNativePath(linkPath));
  }

  async link(existingPath: string, newPath: string): Promise<void> {
    await fsPromises.link(
      this.toNativePath(existingPath),
      this.toNativePath(newPath)
    );
  }

  async readlink(path: string): Promise<string> {
    return fsPromises.readlink(this.toNativePath(path));
  }

  async realpath(path: string): Promise<string> {
    // Return the unix path unchanged — the virtual fs layer handles resolution
    return path;
  }

  async utimes(path: string, atime: Date, mtime: Date): Promise<void> {
    await fsPromises.utimes(this.toNativePath(path), atime, mtime);
  }

  resolvePath(base: string, relPath: string): string {
    if (relPath.startsWith("/")) return relPath;
    const combined = base === "/" ? `/${relPath}` : `${base}/${relPath}`;
    const parts = combined.split("/").filter((p) => p && p !== ".");
    const resolved: string[] = [];
    for (const part of parts) {
      if (part === "..") {
        resolved.pop();
      } else {
        resolved.push(part);
      }
    }
    return `/${resolved.join("/")}`;
  }

  getAllPaths(): string[] {
    return [];
  }
}
