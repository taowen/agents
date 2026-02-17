/**
 * GoogleDriveFsAdapter — IFileSystem backed by Google Drive API.
 *
 * Stores OAuth credentials in /etc/gdrive-credentials.json (D1).
 * Uses a path-to-ID cache with 5-minute TTL to avoid repeated tree walks.
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
import { fromBuffer, getEncoding } from "just-bash";

const DRIVE_API = "https://www.googleapis.com/drive/v3";
const UPLOAD_API = "https://www.googleapis.com/upload/drive/v3";
const FOLDER_MIME = "application/vnd.google-apps.folder";
const CACHE_TTL_MS = 5 * 60 * 1000;

interface GDriveCredentials {
  access_token: string;
  refresh_token: string;
  expires_at: number;
}

interface CacheEntry {
  id: string;
  mimeType: string;
  cachedAt: number;
}

export class GoogleDriveFsAdapter implements IFileSystem {
  private db: D1Database;
  private userId: string;
  private rootPrefix: string;
  private clientId: string;
  private clientSecret: string;
  private rootFolderId: string;

  private credentials: GDriveCredentials | null = null;
  private credentialsLoaded = false;
  private pathCache = new Map<string, CacheEntry>();

  constructor(
    db: D1Database,
    userId: string,
    rootPrefix: string,
    clientId: string,
    clientSecret: string,
    rootFolderId?: string
  ) {
    this.db = db;
    this.userId = userId;
    this.rootPrefix =
      rootPrefix && rootPrefix !== "/"
        ? (rootPrefix.startsWith("/") ? rootPrefix : `/${rootPrefix}`).replace(
            /\/$/,
            ""
          )
        : "";
    this.clientId = clientId;
    this.clientSecret = clientSecret;
    this.rootFolderId = rootFolderId || "root";
  }

  // ---- Credential management ----

  private async ensureCredentials(): Promise<GDriveCredentials> {
    if (!this.credentialsLoaded) {
      await this.loadCredentials();
      this.credentialsLoaded = true;
    }
    if (!this.credentials) {
      throw new Error(
        "EACCES: Google Drive not connected. Visit /oauth/gdrive to authorize."
      );
    }
    // Refresh if expired (with 60s buffer)
    if (Date.now() >= this.credentials.expires_at - 60_000) {
      await this.refreshToken();
    }
    return this.credentials;
  }

  private async loadCredentials(): Promise<void> {
    try {
      const row = await this.db
        .prepare(
          "SELECT CAST(content AS TEXT) as content FROM files WHERE user_id = ? AND path = ?"
        )
        .bind(this.userId, "/etc/gdrive-credentials.json")
        .first<{ content: string | null }>();
      if (row?.content) {
        this.credentials = JSON.parse(row.content);
      }
    } catch {
      // No credentials yet
    }
  }

  private async refreshToken(): Promise<void> {
    if (!this.credentials?.refresh_token) {
      throw new Error("EACCES: No refresh token available for Google Drive");
    }
    const res = await fetch("https://oauth2.googleapis.com/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        client_id: this.clientId,
        client_secret: this.clientSecret,
        refresh_token: this.credentials.refresh_token,
        grant_type: "refresh_token"
      })
    });
    if (!res.ok) {
      throw new Error(
        `EACCES: Failed to refresh Google Drive token: ${res.status}`
      );
    }
    const data = (await res.json()) as {
      access_token: string;
      expires_in: number;
    };
    this.credentials = {
      access_token: data.access_token,
      refresh_token: this.credentials.refresh_token,
      expires_at: Date.now() + data.expires_in * 1000
    };
    await this.saveCredentials();
  }

  private async saveCredentials(): Promise<void> {
    if (!this.credentials) return;
    const content = JSON.stringify(this.credentials);
    const encoded = new TextEncoder().encode(content);
    await this.db
      .prepare(
        `INSERT INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
         VALUES (?, ?, ?, ?, ?, 0, 33188, ?, unixepoch('now'))
         ON CONFLICT(user_id, path) DO UPDATE SET
           content = excluded.content, size = excluded.size, mtime = unixepoch('now')`
      )
      .bind(
        this.userId,
        "/etc/gdrive-credentials.json",
        "/etc",
        "gdrive-credentials.json",
        encoded,
        encoded.length
      )
      .run();
  }

  // ---- Drive API helpers ----

  private async driveApiFetch(
    url: string,
    init?: RequestInit
  ): Promise<Response> {
    const creds = await this.ensureCredentials();
    const headers = new Headers(init?.headers);
    headers.set("Authorization", `Bearer ${creds.access_token}`);
    let res = await fetch(url, { ...init, headers });

    // Retry once on 401 after refreshing token
    if (res.status === 401) {
      await this.refreshToken();
      const newCreds = this.credentials!;
      headers.set("Authorization", `Bearer ${newCreds.access_token}`);
      res = await fetch(url, { ...init, headers });
    }
    return res;
  }

  /**
   * Resolve a virtual path to a Google Drive file ID by walking path segments.
   */
  private async resolveFileId(
    path: string
  ): Promise<{ id: string; mimeType: string } | null> {
    const normalized = this.normalizePath(path);
    if (normalized === "/") {
      return { id: this.rootFolderId, mimeType: FOLDER_MIME };
    }

    // Check cache
    const cached = this.pathCache.get(normalized);
    if (cached && Date.now() - cached.cachedAt < CACHE_TTL_MS) {
      return { id: cached.id, mimeType: cached.mimeType };
    }

    // Walk from root (or nearest cached ancestor)
    const segments = normalized.split("/").filter(Boolean);
    let parentId = this.rootFolderId;
    let currentPath = "";

    for (let i = 0; i < segments.length; i++) {
      const segment = segments[i];
      currentPath += `/${segment}`;

      // Check cache for this intermediate path
      const cachedStep = this.pathCache.get(currentPath);
      if (cachedStep && Date.now() - cachedStep.cachedAt < CACHE_TTL_MS) {
        parentId = cachedStep.id;
        continue;
      }

      // Query Drive for this segment
      const q = `'${parentId}' in parents and name='${segment.replace(/'/g, "\\'")}' and trashed=false`;
      const res = await this.driveApiFetch(
        `${DRIVE_API}/files?q=${encodeURIComponent(q)}&fields=files(id,name,mimeType)&pageSize=1`
      );
      if (!res.ok) {
        throw new Error(`EACCES: Drive API error: ${res.status}`);
      }
      const data = (await res.json()) as {
        files: { id: string; name: string; mimeType: string }[];
      };
      if (data.files.length === 0) {
        return null; // Not found
      }

      const file = data.files[0];
      this.pathCache.set(currentPath, {
        id: file.id,
        mimeType: file.mimeType,
        cachedAt: Date.now()
      });
      parentId = file.id;
    }

    const result = this.pathCache.get(normalized);
    return result ? { id: result.id, mimeType: result.mimeType } : null;
  }

  /**
   * Resolve file ID or throw ENOENT.
   */
  private async requireFileId(
    path: string,
    op: string
  ): Promise<{ id: string; mimeType: string }> {
    const result = await this.resolveFileId(path);
    if (!result) {
      throw new Error(`ENOENT: no such file or directory, ${op} '${path}'`);
    }
    return result;
  }

  /**
   * Resolve parent folder ID, returning {parentId, childName}.
   */
  private async resolveParent(
    path: string
  ): Promise<{ parentId: string; childName: string }> {
    const normalized = this.normalizePath(path);
    const lastSlash = normalized.lastIndexOf("/");
    const parentPath = lastSlash === 0 ? "/" : normalized.slice(0, lastSlash);
    const childName = normalized.slice(lastSlash + 1);

    const parent = await this.resolveFileId(parentPath);
    if (!parent) {
      throw new Error(`ENOENT: no such file or directory, '${parentPath}'`);
    }
    return { parentId: parent.id, childName };
  }

  private normalizePath(path: string): string {
    if (!path || path === "/") return "/";
    const parts = path.split("/").filter((p) => p && p !== ".");
    const resolved: string[] = [];
    for (const part of parts) {
      if (part === "..") resolved.pop();
      else resolved.push(part);
    }
    return `/${resolved.join("/")}` || "/";
  }

  /** Invalidate cache entry for a path and its parent listing. */
  private invalidateCache(path: string): void {
    const normalized = this.normalizePath(path);
    this.pathCache.delete(normalized);
  }

  // ---- IFileSystem methods ----

  async readFile(
    path: string,
    options?: ReadFileOptions | BufferEncoding
  ): Promise<string> {
    const buffer = await this.readFileBuffer(path);
    const encoding = getEncoding(options);
    return fromBuffer(buffer, encoding);
  }

  async readFileBuffer(path: string): Promise<Uint8Array> {
    const { id, mimeType } = await this.requireFileId(path, "open");
    if (mimeType === FOLDER_MIME) {
      throw new Error(
        `EISDIR: illegal operation on a directory, read '${path}'`
      );
    }

    // Check for Google Docs/Sheets/etc. that need export
    if (mimeType.startsWith("application/vnd.google-apps.")) {
      const exportMime = this.getExportMime(mimeType);
      const res = await this.driveApiFetch(
        `${DRIVE_API}/files/${id}/export?mimeType=${encodeURIComponent(exportMime)}`
      );
      if (!res.ok) {
        throw new Error(`EACCES: Drive export failed: ${res.status}`);
      }
      return new Uint8Array(await res.arrayBuffer());
    }

    const res = await this.driveApiFetch(`${DRIVE_API}/files/${id}?alt=media`);
    if (!res.ok) {
      throw new Error(`EACCES: Drive download failed: ${res.status}`);
    }
    return new Uint8Array(await res.arrayBuffer());
  }

  async writeFile(
    path: string,
    content: string | Uint8Array,
    _options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    const data =
      typeof content === "string" ? new TextEncoder().encode(content) : content;

    const existing = await this.resolveFileId(path);
    if (existing) {
      if (existing.mimeType === FOLDER_MIME) {
        throw new Error(
          `EISDIR: illegal operation on a directory, write '${path}'`
        );
      }
      // Update existing file
      const res = await this.driveApiFetch(
        `${UPLOAD_API}/files/${existing.id}?uploadType=media`,
        {
          method: "PATCH",
          body: data as unknown as BodyInit
        }
      );
      if (!res.ok) {
        throw new Error(`EACCES: Drive upload failed: ${res.status}`);
      }
    } else {
      // Create new file
      const { parentId, childName } = await this.resolveParent(path);
      const metadata = JSON.stringify({
        name: childName,
        parents: [parentId]
      });
      const boundary = "---gdrive-boundary-" + Date.now();
      const bodyParts = [
        `--${boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n${metadata}\r\n`,
        `--${boundary}\r\nContent-Type: application/octet-stream\r\n\r\n`
      ];
      const endPart = `\r\n--${boundary}--`;

      const encoder = new TextEncoder();
      const part1 = encoder.encode(bodyParts[0]);
      const part2 = encoder.encode(bodyParts[1]);
      const part3 = encoder.encode(endPart);
      const body = new Uint8Array(
        part1.length + part2.length + data.length + part3.length
      );
      body.set(part1, 0);
      body.set(part2, part1.length);
      body.set(data, part1.length + part2.length);
      body.set(part3, part1.length + part2.length + data.length);

      const res = await this.driveApiFetch(
        `${UPLOAD_API}/files?uploadType=multipart`,
        {
          method: "POST",
          headers: {
            "Content-Type": `multipart/related; boundary=${boundary}`
          },
          body: body as unknown as BodyInit
        }
      );
      if (!res.ok) {
        throw new Error(`EACCES: Drive create failed: ${res.status}`);
      }
      const created = (await res.json()) as {
        id: string;
        mimeType: string;
      };
      // Cache the new file
      const normalized = this.normalizePath(path);
      this.pathCache.set(normalized, {
        id: created.id,
        mimeType: created.mimeType,
        cachedAt: Date.now()
      });
    }
  }

  async appendFile(
    path: string,
    content: string | Uint8Array,
    options?: WriteFileOptions | BufferEncoding
  ): Promise<void> {
    let existing: Uint8Array;
    try {
      existing = await this.readFileBuffer(path);
    } catch {
      existing = new Uint8Array(0);
    }
    const append =
      typeof content === "string" ? new TextEncoder().encode(content) : content;
    const combined = new Uint8Array(existing.length + append.length);
    combined.set(existing);
    combined.set(append, existing.length);
    await this.writeFile(path, combined, options);
  }

  async exists(path: string): Promise<boolean> {
    const result = await this.resolveFileId(path);
    return result !== null;
  }

  async stat(path: string): Promise<FsStat> {
    const { id, mimeType } = await this.requireFileId(path, "stat");

    if (mimeType === FOLDER_MIME) {
      return {
        isFile: false,
        isDirectory: true,
        isSymbolicLink: false,
        mode: 16877,
        size: 0,
        mtime: new Date()
      };
    }

    // Fetch detailed metadata
    const res = await this.driveApiFetch(
      `${DRIVE_API}/files/${id}?fields=id,name,mimeType,size,modifiedTime`
    );
    if (!res.ok) {
      throw new Error(`EACCES: Drive metadata failed: ${res.status}`);
    }
    const meta = (await res.json()) as {
      size?: string;
      modifiedTime?: string;
    };

    return {
      isFile: true,
      isDirectory: false,
      isSymbolicLink: false,
      mode: 33188,
      size: meta.size ? parseInt(meta.size, 10) : 0,
      mtime: meta.modifiedTime ? new Date(meta.modifiedTime) : new Date()
    };
  }

  async lstat(path: string): Promise<FsStat> {
    return this.stat(path);
  }

  async mkdir(path: string, options?: MkdirOptions): Promise<void> {
    // Check if already exists
    const existing = await this.resolveFileId(path);
    if (existing) {
      if (options?.recursive) return;
      throw new Error(`EEXIST: file already exists, mkdir '${path}'`);
    }

    if (options?.recursive) {
      // Ensure all ancestor directories exist
      const segments = this.normalizePath(path).split("/").filter(Boolean);
      let currentPath = "";
      for (const segment of segments) {
        currentPath += `/${segment}`;
        const exists = await this.resolveFileId(currentPath);
        if (!exists) {
          await this.createFolder(currentPath);
        }
      }
    } else {
      await this.createFolder(path);
    }
  }

  private async createFolder(path: string): Promise<void> {
    const { parentId, childName } = await this.resolveParent(path);
    const res = await this.driveApiFetch(`${DRIVE_API}/files`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: childName,
        mimeType: FOLDER_MIME,
        parents: [parentId]
      })
    });
    if (!res.ok) {
      throw new Error(`EACCES: Drive mkdir failed: ${res.status}`);
    }
    const created = (await res.json()) as { id: string; mimeType: string };
    const normalized = this.normalizePath(path);
    this.pathCache.set(normalized, {
      id: created.id,
      mimeType: created.mimeType,
      cachedAt: Date.now()
    });
  }

  async readdir(path: string): Promise<string[]> {
    const { id, mimeType } = await this.requireFileId(path, "scandir");
    if (mimeType !== FOLDER_MIME) {
      throw new Error(`ENOTDIR: not a directory, scandir '${path}'`);
    }

    const names: string[] = [];
    let pageToken: string | undefined;
    const q = `'${id}' in parents and trashed=false`;

    do {
      let url = `${DRIVE_API}/files?q=${encodeURIComponent(q)}&fields=nextPageToken,files(id,name,mimeType)&pageSize=1000`;
      if (pageToken) url += `&pageToken=${encodeURIComponent(pageToken)}`;

      const res = await this.driveApiFetch(url);
      if (!res.ok) {
        throw new Error(`EACCES: Drive list failed: ${res.status}`);
      }
      const data = (await res.json()) as {
        files: { id: string; name: string; mimeType: string }[];
        nextPageToken?: string;
      };

      const normalized = this.normalizePath(path);
      for (const file of data.files) {
        names.push(file.name);
        // Cache child entries
        const childPath =
          normalized === "/" ? `/${file.name}` : `${normalized}/${file.name}`;
        this.pathCache.set(childPath, {
          id: file.id,
          mimeType: file.mimeType,
          cachedAt: Date.now()
        });
      }

      pageToken = data.nextPageToken;
    } while (pageToken);

    return names.sort();
  }

  async rm(path: string, options?: RmOptions): Promise<void> {
    const result = await this.resolveFileId(path);
    if (!result) {
      if (options?.force) return;
      throw new Error(`ENOENT: no such file or directory, rm '${path}'`);
    }

    if (result.mimeType === FOLDER_MIME && !options?.recursive) {
      // Check if directory is empty
      const q = `'${result.id}' in parents and trashed=false`;
      const res = await this.driveApiFetch(
        `${DRIVE_API}/files?q=${encodeURIComponent(q)}&fields=files(id)&pageSize=1`
      );
      if (res.ok) {
        const data = (await res.json()) as { files: { id: string }[] };
        if (data.files.length > 0) {
          throw new Error(`ENOTEMPTY: directory not empty, rm '${path}'`);
        }
      }
    }

    // Drive DELETE on folders is recursive by nature
    const res = await this.driveApiFetch(`${DRIVE_API}/files/${result.id}`, {
      method: "DELETE"
    });
    if (!res.ok && res.status !== 204) {
      throw new Error(`EACCES: Drive delete failed: ${res.status}`);
    }
    this.invalidateCache(path);
  }

  async cp(src: string, dest: string, options?: CpOptions): Promise<void> {
    const srcResult = await this.requireFileId(src, "cp");

    if (srcResult.mimeType === FOLDER_MIME) {
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
      return;
    }

    // Use Drive copy API for files
    const { parentId, childName } = await this.resolveParent(dest);
    const res = await this.driveApiFetch(
      `${DRIVE_API}/files/${srcResult.id}/copy`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: childName,
          parents: [parentId]
        })
      }
    );
    if (!res.ok) {
      throw new Error(`EACCES: Drive copy failed: ${res.status}`);
    }
    const copied = (await res.json()) as { id: string; mimeType: string };
    const normalized = this.normalizePath(dest);
    this.pathCache.set(normalized, {
      id: copied.id,
      mimeType: copied.mimeType,
      cachedAt: Date.now()
    });
  }

  async mv(src: string, dest: string): Promise<void> {
    const srcResult = await this.requireFileId(src, "mv");
    const { parentId: newParentId, childName } = await this.resolveParent(dest);

    // Get current parents
    const metaRes = await this.driveApiFetch(
      `${DRIVE_API}/files/${srcResult.id}?fields=parents`
    );
    if (!metaRes.ok) {
      throw new Error(`EACCES: Drive metadata failed: ${metaRes.status}`);
    }
    const meta = (await metaRes.json()) as { parents?: string[] };
    const oldParents = (meta.parents || []).join(",");

    const res = await this.driveApiFetch(
      `${DRIVE_API}/files/${srcResult.id}?addParents=${newParentId}&removeParents=${oldParents}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: childName })
      }
    );
    if (!res.ok) {
      throw new Error(`EACCES: Drive move failed: ${res.status}`);
    }

    this.invalidateCache(src);
    const normalized = this.normalizePath(dest);
    this.pathCache.set(normalized, {
      id: srcResult.id,
      mimeType: srcResult.mimeType,
      cachedAt: Date.now()
    });
  }

  resolvePath(base: string, path: string): string {
    if (path.startsWith("/")) return this.normalizePath(path);
    const combined = base === "/" ? `/${path}` : `${base}/${path}`;
    return this.normalizePath(combined);
  }

  getAllPaths(): string[] {
    return [];
  }

  async chmod(_path: string, _mode: number): Promise<void> {
    throw new Error("ENOSYS: function not implemented, chmod");
  }

  async symlink(_target: string, _linkPath: string): Promise<void> {
    throw new Error("ENOSYS: function not implemented, symlink");
  }

  async link(_existingPath: string, _newPath: string): Promise<void> {
    throw new Error("ENOSYS: function not implemented, link");
  }

  async readlink(_path: string): Promise<string> {
    throw new Error("ENOSYS: function not implemented, readlink");
  }

  async realpath(path: string): Promise<string> {
    await this.stat(path);
    return this.resolvePath("/", path);
  }

  async utimes(_path: string, _atime: Date, _mtime: Date): Promise<void> {
    // Not supported by Drive — no-op
  }

  // ---- Helpers ----

  private getExportMime(googleMime: string): string {
    switch (googleMime) {
      case "application/vnd.google-apps.document":
        return "text/plain";
      case "application/vnd.google-apps.spreadsheet":
        return "text/csv";
      case "application/vnd.google-apps.presentation":
        return "text/plain";
      case "application/vnd.google-apps.drawing":
        return "image/png";
      default:
        return "text/plain";
    }
  }
}
