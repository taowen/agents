/**
 * File Manager API routes.
 * Routes by filesystem type:
 * - git mount paths → forward to ChatAgent DO
 * - D1/R2 paths → handle directly in Worker
 * - virtual paths (/, /home) → synthetic directory listing
 */

import {
  parseFstab,
  DEFAULT_FSTAB,
  D1FsAdapter,
  R2FsAdapter,
  normalizePath
} from "vfs";
import { MIME_TYPES, getExtension } from "../shared/file-utils";

/** Read /etc/fstab and ensure base D1 directories exist (single batch). */
async function readFstabAndEnsureDirs(
  db: D1Database,
  userId: string
): Promise<string> {
  const mkdirSql = `INSERT OR IGNORE INTO files (user_id, path, parent_path, name, content, is_directory, mode, size, mtime)
     VALUES (?, ?, ?, ?, NULL, 1, 16877, 0, unixepoch('now'))`;
  const results = await db.batch([
    db
      .prepare(
        "SELECT CAST(content AS TEXT) as content FROM files WHERE user_id=? AND path=?"
      )
      .bind(userId, "/etc/fstab"),
    db.prepare(mkdirSql).bind(userId, "/etc", "/", "etc"),
    db.prepare(mkdirSql).bind(userId, "/home", "/", "home"),
    db.prepare(mkdirSql).bind(userId, "/home/user", "/home", "user")
  ]);
  const row = results[0].results[0] as { content: string | null } | undefined;
  return row?.content || DEFAULT_FSTAB;
}

/** Check if path equals or is nested under any mount point. */
function isUnderMount(path: string, mounts: string[]): boolean {
  for (const mp of mounts) {
    if (path === mp || path.startsWith(mp + "/")) return true;
  }
  return false;
}

/** Collect child directory names that mount points imply at the given parent. */
function getVirtualChildren(parentPath: string, mounts: string[]): string[] {
  const prefix = parentPath === "/" ? "/" : parentPath + "/";
  const names = new Set<string>();
  for (const mp of mounts) {
    if (mp.startsWith(prefix)) {
      const rest = mp.slice(prefix.length);
      const first = rest.split("/")[0];
      if (first) names.add(first);
    }
  }
  return [...names];
}

/** Forward request to the ChatAgent DO. */
function forwardToDO(
  request: Request,
  env: Env,
  userId: string
): Promise<Response> {
  const doId = env.ChatAgent.idFromName(`_files_${userId}`);
  const stub = env.ChatAgent.get(doId);
  return stub.fetch(
    new Request(request.url, {
      method: request.method,
      headers: new Headers({
        "x-user-id": userId,
        "x-partykit-room": `_files_${userId}`,
        "content-type": request.headers.get("content-type") || ""
      }),
      body: request.body
    })
  );
}

const D1_MOUNTS = ["/etc", "/home/user"];

/** Resolve a D1/R2 adapter and the path relative to the mount root. */
function resolveAdapter(
  path: string,
  env: Env,
  userId: string
): { adapter: D1FsAdapter | R2FsAdapter; relPath: string } | null {
  for (const mp of D1_MOUNTS) {
    if (path === mp || path.startsWith(mp + "/")) {
      const relPath = path === mp ? "/" : path.slice(mp.length);
      return { adapter: new D1FsAdapter(env.DB, userId, mp), relPath };
    }
  }
  if (env.R2 && (path === "/data" || path.startsWith("/data/"))) {
    const relPath = path === "/data" ? "/" : path.slice("/data".length);
    return { adapter: new R2FsAdapter(env.R2, userId, "/data"), relPath };
  }
  return null;
}

export async function handleFileRoutes(
  request: Request,
  env: Env,
  userId: string
): Promise<Response | null> {
  const url = new URL(request.url);
  if (!url.pathname.startsWith("/api/files")) return null;

  // Read fstab to find git mount points (also ensures /etc, /home, /home/user exist)
  const fstabContent = await readFstabAndEnsureDirs(env.DB, userId);
  const gitMounts = parseFstab(fstabContent)
    .filter((e) => e.type === "git")
    .map((e) => e.mountPoint);

  // Extract target path (mkdir has path in body, others in query)
  let path: string;
  let parsedBody: { path?: string } | undefined;

  if (url.pathname === "/api/files/mkdir" && request.method === "POST") {
    parsedBody = (await request.json()) as { path?: string };
    path = normalizePath(parsedBody.path || "");
  } else {
    path = normalizePath(url.searchParams.get("path") || "/");
  }

  // Git mount → forward to DO
  if (isUnderMount(path, gitMounts)) {
    if (parsedBody !== undefined) {
      // Body already consumed for mkdir — reconstruct request
      const doId = env.ChatAgent.idFromName(`_files_${userId}`);
      const stub = env.ChatAgent.get(doId);
      return stub.fetch(
        new Request(request.url, {
          method: request.method,
          headers: new Headers({
            "x-user-id": userId,
            "x-partykit-room": `_files_${userId}`,
            "content-type": "application/json"
          }),
          body: JSON.stringify(parsedBody)
        })
      );
    }
    return forwardToDO(request, env, userId);
  }

  // D1/R2 or virtual → handle in Worker
  const allMounts = [...D1_MOUNTS, ...(env.R2 ? ["/data"] : []), ...gitMounts];
  const resolved = resolveAdapter(path, env, userId);

  try {
    // GET /api/files/list
    if (url.pathname === "/api/files/list" && request.method === "GET") {
      if (!resolved) {
        // Virtual directory (e.g. "/" or "/home")
        const children = getVirtualChildren(path, allMounts);
        if (children.length === 0) {
          return Response.json(
            { error: `ENOENT: no such directory: ${path}` },
            { status: 404 }
          );
        }
        return Response.json({
          entries: children.map((name) => ({
            name,
            isDirectory: true,
            size: 0,
            mtime: null
          }))
        });
      }

      const { adapter, relPath } = resolved;
      // Real listing from adapter (use relative path)
      const names = await adapter.readdir(relPath);
      const entries = await Promise.all(
        names.map(async (name: string) => {
          try {
            const child = relPath === "/" ? `/${name}` : `${relPath}/${name}`;
            const st = await adapter.stat(child);
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

      // Append git-mount children not already in the listing
      const existing = new Set(names);
      for (const vName of getVirtualChildren(path, gitMounts)) {
        if (!existing.has(vName)) {
          entries.push({
            name: vName,
            isDirectory: true,
            size: 0,
            mtime: null
          });
        }
      }

      return Response.json({ entries });
    }

    // GET /api/files/content
    if (url.pathname === "/api/files/content" && request.method === "GET") {
      if (!resolved) {
        return Response.json({ error: `ENOENT: ${path}` }, { status: 404 });
      }
      const buffer = await resolved.adapter.readFileBuffer(resolved.relPath);
      const ext = getExtension(path);
      const contentType = MIME_TYPES[ext] || "application/octet-stream";
      return new Response(buffer, {
        headers: { "Content-Type": contentType }
      });
    }

    // PUT /api/files/content
    if (url.pathname === "/api/files/content" && request.method === "PUT") {
      if (!resolved) {
        return Response.json(
          { error: `ENOENT: no filesystem at ${path}` },
          { status: 404 }
        );
      }
      const buffer = new Uint8Array(await request.arrayBuffer());
      await resolved.adapter.writeFile(resolved.relPath, buffer);
      return Response.json({ ok: true });
    }

    // POST /api/files/mkdir
    if (url.pathname === "/api/files/mkdir" && request.method === "POST") {
      if (!resolved) {
        return Response.json(
          { error: `ENOENT: no filesystem at ${path}` },
          { status: 404 }
        );
      }
      await resolved.adapter.mkdir(resolved.relPath, { recursive: true });
      return Response.json({ ok: true });
    }

    // DELETE /api/files
    if (url.pathname === "/api/files" && request.method === "DELETE") {
      if (!resolved) {
        return Response.json(
          { error: `ENOENT: no filesystem at ${path}` },
          { status: 404 }
        );
      }
      const recursive = url.searchParams.get("recursive") === "1";
      await resolved.adapter.rm(resolved.relPath, { recursive });
      return Response.json({ ok: true });
    }

    return Response.json({ error: "Not found" }, { status: 404 });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg.includes("ENOENT"))
      return Response.json({ error: msg }, { status: 404 });
    if (msg.includes("EISDIR"))
      return Response.json({ error: msg }, { status: 400 });
    if (msg.includes("EBUSY"))
      return Response.json({ error: msg }, { status: 400 });
    return Response.json({ error: msg }, { status: 500 });
  }
}
