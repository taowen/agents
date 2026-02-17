/**
 * File Manager API routes.
 * Extracted from api.ts to keep route handlers focused.
 */

import type { IFileSystem } from "just-bash";
import { D1FsAdapter, R2FsAdapter, normalizePath } from "vfs";
import { MIME_TYPES, getExtension } from "../shared/file-utils";

/** Match normalized path prefix to an adapter. Returns null for disallowed paths. */
function resolveFilesystem(
  env: Env,
  userId: string,
  rawPath: string
): { fs: IFileSystem; localPath: string } | null {
  const path = normalizePath(rawPath);
  if (path.startsWith("/home/user")) {
    return {
      fs: new D1FsAdapter(env.DB, userId, "/home/user"),
      localPath: path.slice("/home/user".length) || "/"
    };
  }
  if (path.startsWith("/etc")) {
    return {
      fs: new D1FsAdapter(env.DB, userId, "/etc"),
      localPath: path.slice("/etc".length) || "/"
    };
  }
  if (path.startsWith("/data") && env.R2) {
    return {
      fs: new R2FsAdapter(env.R2, userId, "/data"),
      localPath: path.slice("/data".length) || "/"
    };
  }
  return null;
}

export async function handleFileRoutes(
  request: Request,
  env: Env,
  userId: string
): Promise<Response | null> {
  const url = new URL(request.url);

  // GET /api/files/list?path=<dir>
  if (url.pathname === "/api/files/list" && request.method === "GET") {
    const rawPath = url.searchParams.get("path") || "/";
    const path = normalizePath(rawPath);

    // Virtual root: list mount points
    if (path === "/") {
      const entries = [
        { name: "home", isDirectory: true, size: 0, mtime: null },
        { name: "etc", isDirectory: true, size: 0, mtime: null },
        ...(env.R2
          ? [{ name: "data", isDirectory: true, size: 0, mtime: null }]
          : [])
      ];
      return Response.json({ entries });
    }
    // Virtual /home → show "user"
    if (path === "/home") {
      return Response.json({
        entries: [{ name: "user", isDirectory: true, size: 0, mtime: null }]
      });
    }

    const resolved = resolveFilesystem(env, userId, path);
    if (!resolved) {
      return Response.json({ error: "Invalid path" }, { status: 400 });
    }
    try {
      const names = await resolved.fs.readdir(resolved.localPath);
      const entries = await Promise.all(
        names.map(async (name: string) => {
          try {
            const childPath =
              resolved.localPath === "/"
                ? `/${name}`
                : `${resolved.localPath}/${name}`;
            const st = await resolved.fs.stat(childPath);
            return {
              name,
              isDirectory: st.isDirectory,
              size: st.size,
              mtime: st.mtime?.toISOString() ?? null
            };
          } catch {
            return { name, isDirectory: false, size: 0, mtime: null };
          }
        })
      );
      return Response.json({ entries });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("ENOENT")) {
        return Response.json({ error: msg }, { status: 404 });
      }
      return Response.json({ error: msg }, { status: 500 });
    }
  }

  // GET /api/files/content?path=<file>  — download file
  if (url.pathname === "/api/files/content" && request.method === "GET") {
    const rawPath = url.searchParams.get("path") || "";
    const resolved = resolveFilesystem(env, userId, rawPath);
    if (!resolved) {
      return Response.json({ error: "Invalid path" }, { status: 400 });
    }
    try {
      const buffer = await resolved.fs.readFileBuffer(resolved.localPath);
      const ext = getExtension(rawPath);
      const contentType = MIME_TYPES[ext] || "application/octet-stream";
      return new Response(buffer, {
        headers: { "Content-Type": contentType }
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("ENOENT")) {
        return Response.json({ error: msg }, { status: 404 });
      }
      if (msg.includes("EISDIR")) {
        return Response.json({ error: msg }, { status: 400 });
      }
      return Response.json({ error: msg }, { status: 500 });
    }
  }

  // PUT /api/files/content?path=<file>  — upload file
  if (url.pathname === "/api/files/content" && request.method === "PUT") {
    const rawPath = url.searchParams.get("path") || "";
    const resolved = resolveFilesystem(env, userId, rawPath);
    if (!resolved) {
      return Response.json({ error: "Invalid path" }, { status: 400 });
    }
    try {
      const buffer = new Uint8Array(await request.arrayBuffer());
      await resolved.fs.writeFile(resolved.localPath, buffer);
      return Response.json({ ok: true });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      return Response.json({ error: msg }, { status: 500 });
    }
  }

  // POST /api/files/mkdir  — create directory
  if (url.pathname === "/api/files/mkdir" && request.method === "POST") {
    const body = (await request.json()) as { path?: string };
    const rawPath = body.path || "";
    const resolved = resolveFilesystem(env, userId, rawPath);
    if (!resolved) {
      return Response.json({ error: "Invalid path" }, { status: 400 });
    }
    try {
      await resolved.fs.mkdir(resolved.localPath, { recursive: true });
      return Response.json({ ok: true });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      return Response.json({ error: msg }, { status: 500 });
    }
  }

  // DELETE /api/files?path=<path>&recursive=0|1  — delete file or directory
  if (url.pathname === "/api/files" && request.method === "DELETE") {
    const rawPath = url.searchParams.get("path") || "";
    const recursive = url.searchParams.get("recursive") === "1";
    const resolved = resolveFilesystem(env, userId, rawPath);
    if (!resolved) {
      return Response.json({ error: "Invalid path" }, { status: 400 });
    }
    try {
      await resolved.fs.rm(resolved.localPath, { recursive });
      return Response.json({ ok: true });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("ENOENT")) {
        return Response.json({ error: msg }, { status: 404 });
      }
      return Response.json({ error: msg }, { status: 500 });
    }
  }

  return null;
}
