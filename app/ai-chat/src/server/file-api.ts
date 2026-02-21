import type { MountableFs } from "just-bash";
import { normalizePath } from "vfs";
import { MIME_TYPES, getExtension } from "../shared/file-utils";

/**
 * Handle file manager API requests using a MountableFs.
 * Assumes the filesystem is already initialized and mounted.
 */
export async function handleFileRequest(
  request: Request,
  mountableFs: MountableFs
): Promise<Response> {
  const url = new URL(request.url);
  const fs = mountableFs;

  try {
    // GET /api/files/stat?path=<path>
    if (url.pathname === "/api/files/stat" && request.method === "GET") {
      const rawPath = url.searchParams.get("path") || "/";
      const path = normalizePath(rawPath);
      const st = await fs.stat(path);
      return Response.json({
        isFile: st.isFile,
        isDirectory: st.isDirectory,
        isSymbolicLink: st.isSymbolicLink,
        mode: st.mode,
        size: st.size,
        mtime: st.mtime?.toISOString() ?? null
      });
    }

    // GET /api/files/list?path=<dir>
    if (url.pathname === "/api/files/list" && request.method === "GET") {
      const rawPath = url.searchParams.get("path") || "/";
      const path = normalizePath(rawPath);
      const isRecursive = url.searchParams.get("recursive") === "1";

      if (isRecursive) {
        const result: Array<{
          name: string;
          path: string;
          isDirectory: boolean;
          size: number;
          mtime: string | null;
        }> = [];
        const queue = [{ dirPath: path, prefix: "" }];
        while (queue.length > 0) {
          const { dirPath, prefix } = queue.shift()!;
          let names: string[];
          try {
            names = await fs.readdir(dirPath);
          } catch {
            continue;
          }
          for (const name of names) {
            const childPath =
              dirPath === "/" ? `/${name}` : `${dirPath}/${name}`;
            const childPrefix = prefix ? `${prefix}/${name}` : name;
            try {
              const st = await fs.stat(childPath);
              result.push({
                name,
                path: childPrefix,
                isDirectory: st.isDirectory,
                size: st.size,
                mtime: st.mtime?.toISOString() ?? null
              });
              if (st.isDirectory) {
                queue.push({ dirPath: childPath, prefix: childPrefix });
              }
            } catch {
              result.push({
                name,
                path: childPrefix,
                isDirectory: false,
                size: 0,
                mtime: null
              });
            }
          }
        }
        return Response.json({ entries: result });
      }

      const names = await fs.readdir(path);
      const entries = await Promise.all(
        names.map(async (name: string) => {
          try {
            const childPath = path === "/" ? `/${name}` : `${path}/${name}`;
            const st = await fs.stat(childPath);
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
    }

    // GET /api/files/content?path=<file>
    if (url.pathname === "/api/files/content" && request.method === "GET") {
      const rawPath = url.searchParams.get("path") || "";
      const path = normalizePath(rawPath);
      const buffer = await fs.readFileBuffer(path);
      const ext = getExtension(rawPath);
      const contentType = MIME_TYPES[ext] || "application/octet-stream";
      return new Response(buffer as BodyInit, {
        headers: { "Content-Type": contentType }
      });
    }

    // PUT /api/files/content?path=<file>
    if (url.pathname === "/api/files/content" && request.method === "PUT") {
      const rawPath = url.searchParams.get("path") || "";
      const path = normalizePath(rawPath);
      const buffer = new Uint8Array(await request.arrayBuffer());
      await fs.writeFile(path, buffer);
      return Response.json({ ok: true });
    }

    // POST /api/files/mkdir
    if (url.pathname === "/api/files/mkdir" && request.method === "POST") {
      const body = (await request.json()) as { path?: string };
      const rawPath = body.path || "";
      const path = normalizePath(rawPath);
      await fs.mkdir(path, { recursive: true });
      return Response.json({ ok: true });
    }

    // DELETE /api/files?path=<path>&recursive=0|1
    if (url.pathname === "/api/files" && request.method === "DELETE") {
      const rawPath = url.searchParams.get("path") || "";
      const path = normalizePath(rawPath);
      const recursive = url.searchParams.get("recursive") === "1";
      await fs.rm(path, { recursive });
      return Response.json({ ok: true });
    }

    return Response.json({ error: "Not found" }, { status: 404 });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg.includes("ENOENT")) {
      return Response.json({ error: msg }, { status: 404 });
    }
    if (msg.includes("EISDIR")) {
      return Response.json({ error: msg }, { status: 400 });
    }
    if (msg.includes("EBUSY")) {
      return Response.json({ error: msg }, { status: 400 });
    }
    return Response.json({ error: msg }, { status: 500 });
  }
}
