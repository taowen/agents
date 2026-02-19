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
 * Creates an error with a code property, matching Node.js fs errors.
 */
function fsError(message: string, code?: string): Error {
  const err = new Error(message);
  if (code) (err as Error & { code: string }).code = code;
  return err;
}

/**
 * Filesystem adapter that routes IFileSystem calls through HTTP
 * to access the main agent's cloud filesystem (D1/R2/git mounts)
 * via the /api/files/* endpoints.
 *
 * Paths received from MountableFs are relative to the mount point.
 * They are passed directly to the HTTP API since both use the same
 * absolute path space.
 */
export class HttpFsAdapter implements IFileSystem {
  private baseUrl: string;

  constructor(baseUrl: string = "") {
    this.baseUrl = baseUrl;
  }

  private async fetchJson(url: string, init?: RequestInit): Promise<unknown> {
    const resp = await fetch(this.baseUrl + url, init);
    if (!resp.ok) {
      let errorMsg: string;
      try {
        const body = (await resp.json()) as { error?: string };
        errorMsg = body.error || `HTTP ${resp.status}`;
      } catch {
        errorMsg = `HTTP ${resp.status}`;
      }
      if (resp.status === 404) {
        throw fsError(errorMsg, "ENOENT");
      }
      throw fsError(errorMsg);
    }
    return resp.json();
  }

  async readFile(
    path: string,
    _options?: ReadFileOptions | BufferEncoding
  ): Promise<string> {
    const resp = await fetch(
      this.baseUrl + `/api/files/content?path=${encodeURIComponent(path)}`
    );
    if (!resp.ok) {
      let errorMsg: string;
      try {
        const body = (await resp.json()) as { error?: string };
        errorMsg = body.error || `HTTP ${resp.status}`;
      } catch {
        errorMsg = `HTTP ${resp.status}`;
      }
      if (resp.status === 404) throw fsError(errorMsg, "ENOENT");
      throw fsError(errorMsg);
    }
    return resp.text();
  }

  async readFileBuffer(path: string): Promise<Uint8Array> {
    const resp = await fetch(
      this.baseUrl + `/api/files/content?path=${encodeURIComponent(path)}`
    );
    if (!resp.ok) {
      let errorMsg: string;
      try {
        const body = (await resp.json()) as { error?: string };
        errorMsg = body.error || `HTTP ${resp.status}`;
      } catch {
        errorMsg = `HTTP ${resp.status}`;
      }
      if (resp.status === 404) throw fsError(errorMsg, "ENOENT");
      throw fsError(errorMsg);
    }
    return new Uint8Array(await resp.arrayBuffer());
  }

  async writeFile(
    path: string,
    content: FileContent,
    _options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    const body =
      typeof content === "string"
        ? new TextEncoder().encode(content)
        : (content as BodyInit);
    const resp = await fetch(
      this.baseUrl + `/api/files/content?path=${encodeURIComponent(path)}`,
      { method: "PUT", body }
    );
    if (!resp.ok) {
      let errorMsg: string;
      try {
        const b = (await resp.json()) as { error?: string };
        errorMsg = b.error || `HTTP ${resp.status}`;
      } catch {
        errorMsg = `HTTP ${resp.status}`;
      }
      if (resp.status === 404) throw fsError(errorMsg, "ENOENT");
      throw fsError(errorMsg);
    }
  }

  async appendFile(
    path: string,
    content: FileContent,
    _options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    // API doesn't support append — read + concat + write
    let existing = "";
    try {
      existing = await this.readFile(path);
    } catch (e) {
      if ((e as Error & { code?: string }).code !== "ENOENT") throw e;
    }
    const appendStr =
      typeof content === "string" ? content : new TextDecoder().decode(content);
    await this.writeFile(path, existing + appendStr);
  }

  async exists(path: string): Promise<boolean> {
    try {
      await this.stat(path);
      return true;
    } catch {
      return false;
    }
  }

  async stat(path: string): Promise<FsStat> {
    const raw = (await this.fetchJson(
      `/api/files/stat?path=${encodeURIComponent(path)}`
    )) as {
      isFile: boolean;
      isDirectory: boolean;
      isSymbolicLink: boolean;
      mode: number;
      size: number;
      mtime: string | null;
    };
    return {
      ...raw,
      mtime: raw.mtime ? new Date(raw.mtime) : new Date(0)
    };
  }

  async lstat(path: string): Promise<FsStat> {
    // Cloud backends don't support symlinks; lstat == stat
    return this.stat(path);
  }

  async mkdir(path: string, _options?: MkdirOptions): Promise<void> {
    const resp = await fetch(this.baseUrl + "/api/files/mkdir", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path })
    });
    if (!resp.ok) {
      let errorMsg: string;
      try {
        const body = (await resp.json()) as { error?: string };
        errorMsg = body.error || `HTTP ${resp.status}`;
      } catch {
        errorMsg = `HTTP ${resp.status}`;
      }
      throw fsError(errorMsg);
    }
  }

  async readdir(path: string): Promise<string[]> {
    const data = (await this.fetchJson(
      `/api/files/list?path=${encodeURIComponent(path)}`
    )) as { entries: { name: string }[] };
    return data.entries.map((e) => e.name);
  }

  async readdirWithFileTypes(path: string): Promise<DirentEntry[]> {
    const data = (await this.fetchJson(
      `/api/files/list?path=${encodeURIComponent(path)}`
    )) as {
      entries: { name: string; isDirectory: boolean }[];
    };
    return data.entries.map((e) => ({
      name: e.name,
      isFile: !e.isDirectory,
      isDirectory: e.isDirectory,
      isSymbolicLink: false
    }));
  }

  async rm(path: string, options?: RmOptions): Promise<void> {
    const recursive = options?.recursive ? "1" : "0";
    const resp = await fetch(
      this.baseUrl +
        `/api/files?path=${encodeURIComponent(path)}&recursive=${recursive}`,
      { method: "DELETE" }
    );
    if (!resp.ok) {
      let errorMsg: string;
      try {
        const body = (await resp.json()) as { error?: string };
        errorMsg = body.error || `HTTP ${resp.status}`;
      } catch {
        errorMsg = `HTTP ${resp.status}`;
      }
      if (resp.status === 404) throw fsError(errorMsg, "ENOENT");
      throw fsError(errorMsg);
    }
  }

  async cp(src: string, dest: string, _options?: CpOptions): Promise<void> {
    const content = await this.readFileBuffer(src);
    await this.writeFile(dest, content);
  }

  async mv(src: string, dest: string): Promise<void> {
    await this.cp(src, dest);
    await this.rm(src);
  }

  async chmod(_path: string, _mode: number): Promise<void> {
    throw fsError("chmod not supported on cloud filesystem", "ENOSYS");
  }

  async symlink(_target: string, _linkPath: string): Promise<void> {
    throw fsError("symlink not supported on cloud filesystem", "ENOSYS");
  }

  async link(_existingPath: string, _newPath: string): Promise<void> {
    throw fsError("link not supported on cloud filesystem", "ENOSYS");
  }

  async readlink(_path: string): Promise<string> {
    throw fsError("readlink not supported on cloud filesystem", "ENOSYS");
  }

  async realpath(path: string): Promise<string> {
    return path;
  }

  async utimes(_path: string, _atime: Date, _mtime: Date): Promise<void> {
    throw fsError("utimes not supported on cloud filesystem", "ENOSYS");
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
