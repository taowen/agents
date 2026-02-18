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
 * IPC function signature matching the preload bridge.
 */
type IpcFn = (params: Record<string, unknown>) => Promise<{
  ok: boolean;
  result?: unknown;
  error?: string;
  code?: string;
}>;

/**
 * Creates an ENOENT-style error with a code property, matching Node.js fs errors.
 */
function fsError(message: string, code?: string): Error {
  const err = new Error(message);
  if (code) (err as Error & { code: string }).code = code;
  return err;
}

/**
 * Filesystem adapter that routes IFileSystem calls through Electron IPC
 * to access real Windows/WSL filesystems from the renderer process.
 *
 * Each instance is bound to a Windows root path (e.g. "C:\\" or "\\\\wsl.localhost\\Ubuntu\\").
 * Unix-style relative paths from MountableFs are translated to Windows absolute paths
 * before being sent over IPC.
 */
export class WindowsFsAdapter implements IFileSystem {
  private rootPath: string;
  private ipcFn: IpcFn;

  constructor(rootPath: string, ipcFn: IpcFn) {
    this.rootPath = rootPath;
    this.ipcFn = ipcFn;
  }

  /**
   * Convert a Unix-style relative path (from MountableFs) to a Windows absolute path.
   * "/" -> "C:\" , "/Users/foo" -> "C:\Users\foo"
   */
  private toWindowsPath(relativePath: string): string {
    // relativePath comes from MountableFs.routePath and always starts with "/"
    if (relativePath === "/") {
      return this.rootPath;
    }
    // Remove leading slash, convert / to \, append to root
    const suffix = relativePath.slice(1).replace(/\//g, "\\");
    // rootPath already ends with \ (e.g. "C:\\" or "\\wsl.localhost\Ubuntu\")
    return this.rootPath + suffix;
  }

  private async call(
    op: string,
    params: Record<string, unknown> = {}
  ): Promise<unknown> {
    const res = await this.ipcFn({ op, ...params });
    if (!res.ok) {
      throw fsError(res.error || `fs:${op} failed`, res.code);
    }
    return res.result;
  }

  async readFile(
    path: string,
    _options?: ReadFileOptions | BufferEncoding
  ): Promise<string> {
    return (await this.call("readFile", {
      path: this.toWindowsPath(path)
    })) as string;
  }

  async readFileBuffer(path: string): Promise<Uint8Array> {
    return (await this.call("readFileBuffer", {
      path: this.toWindowsPath(path)
    })) as Uint8Array;
  }

  async writeFile(
    path: string,
    content: FileContent,
    _options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    await this.call("writeFile", {
      path: this.toWindowsPath(path),
      content: typeof content === "string" ? content : Array.from(content)
    });
  }

  async appendFile(
    path: string,
    content: FileContent,
    _options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    await this.call("appendFile", {
      path: this.toWindowsPath(path),
      content: typeof content === "string" ? content : Array.from(content)
    });
  }

  async exists(path: string): Promise<boolean> {
    return (await this.call("exists", {
      path: this.toWindowsPath(path)
    })) as boolean;
  }

  async stat(path: string): Promise<FsStat> {
    const raw = (await this.call("stat", {
      path: this.toWindowsPath(path)
    })) as {
      isFile: boolean;
      isDirectory: boolean;
      isSymbolicLink: boolean;
      mode: number;
      size: number;
      mtime: string;
    };
    return { ...raw, mtime: new Date(raw.mtime) };
  }

  async lstat(path: string): Promise<FsStat> {
    const raw = (await this.call("lstat", {
      path: this.toWindowsPath(path)
    })) as {
      isFile: boolean;
      isDirectory: boolean;
      isSymbolicLink: boolean;
      mode: number;
      size: number;
      mtime: string;
    };
    return { ...raw, mtime: new Date(raw.mtime) };
  }

  async mkdir(path: string, options?: MkdirOptions): Promise<void> {
    await this.call("mkdir", {
      path: this.toWindowsPath(path),
      recursive: options?.recursive
    });
  }

  async readdir(path: string): Promise<string[]> {
    return (await this.call("readdir", {
      path: this.toWindowsPath(path)
    })) as string[];
  }

  async readdirWithFileTypes(path: string): Promise<DirentEntry[]> {
    return (await this.call("readdirWithFileTypes", {
      path: this.toWindowsPath(path)
    })) as DirentEntry[];
  }

  async rm(path: string, options?: RmOptions): Promise<void> {
    await this.call("rm", {
      path: this.toWindowsPath(path),
      recursive: options?.recursive,
      force: options?.force
    });
  }

  async cp(src: string, dest: string, options?: CpOptions): Promise<void> {
    await this.call("cp", {
      src: this.toWindowsPath(src),
      dest: this.toWindowsPath(dest),
      recursive: options?.recursive
    });
  }

  async mv(src: string, dest: string): Promise<void> {
    await this.call("mv", {
      src: this.toWindowsPath(src),
      dest: this.toWindowsPath(dest)
    });
  }

  async chmod(path: string, mode: number): Promise<void> {
    await this.call("chmod", {
      path: this.toWindowsPath(path),
      mode
    });
  }

  async symlink(target: string, linkPath: string): Promise<void> {
    await this.call("symlink", {
      target,
      linkPath: this.toWindowsPath(linkPath)
    });
  }

  async link(existingPath: string, newPath: string): Promise<void> {
    await this.call("link", {
      existingPath: this.toWindowsPath(existingPath),
      newPath: this.toWindowsPath(newPath)
    });
  }

  async readlink(path: string): Promise<string> {
    return (await this.call("readlink", {
      path: this.toWindowsPath(path)
    })) as string;
  }

  async realpath(path: string): Promise<string> {
    // realpath on the Windows side returns a Windows absolute path.
    // For our virtual fs purposes, just return the unix relative path unchanged.
    return path;
  }

  async utimes(path: string, atime: Date, mtime: Date): Promise<void> {
    await this.call("utimes", {
      path: this.toWindowsPath(path),
      atime: atime.toISOString(),
      mtime: mtime.toISOString()
    });
  }

  resolvePath(base: string, relPath: string): string {
    if (relPath.startsWith("/")) return relPath;
    const combined = base === "/" ? `/${relPath}` : `${base}/${relPath}`;
    // Normalize . and ..
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
    // Not feasible for a real filesystem — return empty as allowed by the interface
    return [];
  }
}
