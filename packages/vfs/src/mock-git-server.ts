/**
 * mock-git-server — in-memory git repo + mock HTTP transport for testing.
 *
 * Creates a git repository in memory using isomorphic-git,
 * then provides a mock HTTP transport that implements the git smart HTTP protocol
 * (info/refs + git-upload-pack) so GitFs can clone without network access.
 */

import git from "isomorphic-git";

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

// ---- Minimal in-memory fs for isomorphic-git (same pattern as git-fs.ts) ----

function createMemFs() {
  const files = new Map<string, Uint8Array>();
  const dirs = new Set<string>(["/", "."]);

  function ensureParent(filepath: string) {
    const parts = filepath.split("/");
    for (let i = 1; i < parts.length; i++) {
      const dir = parts.slice(0, i).join("/") || "/";
      dirs.add(dir);
    }
  }

  function makeStat(type: "file" | "dir", size: number) {
    const now = Date.now();
    return {
      type,
      mode: type === "dir" ? 0o40755 : 0o100644,
      size,
      mtimeMs: now,
      ctimeMs: now,
      dev: 0,
      ino: 0,
      uid: 0,
      gid: 0,
      isFile: () => type === "file",
      isDirectory: () => type === "dir",
      isSymbolicLink: () => false
    };
  }

  return {
    async readFile(filepath: string, opts?: any): Promise<Uint8Array | string> {
      const data = files.get(filepath);
      if (data === undefined) {
        const err = new Error(
          `ENOENT: no such file or directory, open '${filepath}'`
        );
        (err as any).code = "ENOENT";
        throw err;
      }
      const encoding = typeof opts === "string" ? opts : opts?.encoding;
      if (encoding) return textDecoder.decode(data);
      return data;
    },
    async writeFile(
      filepath: string,
      data: Uint8Array | string,
      _opts?: any
    ): Promise<void> {
      ensureParent(filepath);
      const buf = typeof data === "string" ? textEncoder.encode(data) : data;
      files.set(filepath, new Uint8Array(buf));
    },
    async mkdir(filepath: string, _opts?: any): Promise<void> {
      dirs.add(filepath);
    },
    async rmdir(filepath: string): Promise<void> {
      dirs.delete(filepath);
    },
    async unlink(filepath: string): Promise<void> {
      files.delete(filepath);
    },
    async stat(filepath: string) {
      if (dirs.has(filepath)) return makeStat("dir", 0);
      if (files.has(filepath))
        return makeStat("file", files.get(filepath)!.length);
      const err = new Error(
        `ENOENT: no such file or directory, stat '${filepath}'`
      );
      (err as any).code = "ENOENT";
      throw err;
    },
    async lstat(filepath: string) {
      if (dirs.has(filepath)) return makeStat("dir", 0);
      if (files.has(filepath))
        return makeStat("file", files.get(filepath)!.length);
      const err = new Error(
        `ENOENT: no such file or directory, stat '${filepath}'`
      );
      (err as any).code = "ENOENT";
      throw err;
    },
    async readdir(filepath: string): Promise<string[]> {
      const prefix = filepath === "/" || filepath === "." ? "" : filepath + "/";
      const entries = new Set<string>();
      for (const f of files.keys()) {
        if (prefix && !f.startsWith(prefix)) continue;
        if (!prefix && f.includes("/")) {
          entries.add(f.split("/")[0]);
          continue;
        }
        const rest = prefix ? f.slice(prefix.length) : f;
        if (rest && !rest.includes("/")) entries.add(rest);
        else if (rest) entries.add(rest.split("/")[0]);
      }
      for (const d of dirs) {
        if (d === filepath || d === "/" || d === ".") continue;
        if (prefix && !d.startsWith(prefix)) continue;
        const rest = prefix ? d.slice(prefix.length) : d;
        if (rest && !rest.includes("/")) entries.add(rest);
        else if (rest) entries.add(rest.split("/")[0]);
      }
      return [...entries];
    },
    async readlink(filepath: string): Promise<string> {
      const data = files.get(filepath);
      if (!data) throw new Error(`ENOENT: readlink '${filepath}'`);
      return textDecoder.decode(data);
    },
    async symlink(target: string, filepath: string): Promise<void> {
      ensureParent(filepath);
      files.set(filepath, textEncoder.encode(target));
    },
    // Expose internals for pack generation
    _files: files,
    _dirs: dirs
  };
}

// ---- pkt-line helpers ----

function pktLine(data: string): string {
  const len = data.length + 4;
  return len.toString(16).padStart(4, "0") + data;
}

function pktLineBytes(data: Uint8Array): Uint8Array {
  const len = data.length + 4;
  const header = textEncoder.encode(len.toString(16).padStart(4, "0"));
  const result = new Uint8Array(header.length + data.length);
  result.set(header);
  result.set(data, header.length);
  return result;
}

const FLUSH = "0000";

// ---- Create mock git server ----

export async function createMockGitServer(
  files: Record<string, string>
): Promise<{
  url: string;
  http: { request: (opts: any) => Promise<any> };
}> {
  const memFs = createMemFs();
  const dir = "/repo";
  const gitdir = "/repo/.git";

  // Init repo and commit files
  await git.init({ fs: memFs, dir, defaultBranch: "main" });

  for (const [path, content] of Object.entries(files)) {
    const fullPath = `${dir}/${path}`;
    // Ensure parent directories exist
    const parts = path.split("/");
    for (let i = 1; i < parts.length; i++) {
      const parentDir = `${dir}/${parts.slice(0, i).join("/")}`;
      try {
        await memFs.mkdir(parentDir);
      } catch {
        /* exists */
      }
    }
    await memFs.writeFile(fullPath, textEncoder.encode(content));
    await git.add({ fs: memFs, dir, filepath: path });
  }

  await git.commit({
    fs: memFs,
    dir,
    message: "Initial commit",
    author: { name: "Test", email: "test@test.com" }
  });

  // Get HEAD oid and ref
  const headOid = await git.resolveRef({ fs: memFs, dir, ref: "HEAD" });

  // Build the mock HTTP handler
  const mockUrl = "http://localhost/__mock_git_repo__";

  const mockHttp = {
    request: async (opts: {
      url: string;
      method?: string;
      headers?: Record<string, string>;
      body?: AsyncIterable<Uint8Array>;
    }) => {
      const requestUrl = opts.url;
      const method = opts.method || "GET";

      // GET /info/refs?service=git-upload-pack
      if (method === "GET" && requestUrl.includes("/info/refs")) {
        const capabilities =
          "multi_ack_detailed no-done side-band-64k thin-pack ofs-delta shallow agent=mock-git-server";
        const refLine = `${headOid} refs/heads/main\n`;
        const headRefLine = `${headOid} HEAD\n`;

        // Smart HTTP: service announcement + ref advertisement
        const serviceLine = pktLine("# service=git-upload-pack\n");
        const firstRef = pktLine(`${headOid} HEAD\0${capabilities}\n`);
        const mainRef = pktLine(`${headOid} refs/heads/main\n`);

        const body = serviceLine + FLUSH + firstRef + mainRef + FLUSH;

        return {
          url: requestUrl,
          method,
          statusCode: 200,
          statusMessage: "OK",
          headers: {
            "content-type": "application/x-git-upload-pack-advertisement"
          },
          body: [textEncoder.encode(body)]
        };
      }

      // POST /git-upload-pack
      if (method === "POST" && requestUrl.includes("git-upload-pack")) {
        // Read the request body to get want lines
        const bodyChunks: Uint8Array[] = [];
        if (opts.body) {
          for await (const chunk of opts.body) {
            bodyChunks.push(chunk);
          }
        }

        // Collect all reachable objects (commit + trees + blobs)
        const oids: string[] = [headOid];
        const commitResult = await git.readCommit({
          fs: memFs,
          gitdir,
          oid: headOid
        });
        const treeOid = commitResult.commit.tree;

        async function walkTree(oid: string) {
          oids.push(oid);
          const { tree } = await git.readTree({ fs: memFs, gitdir, oid });
          for (const entry of tree) {
            if (entry.type === "tree") {
              await walkTree(entry.oid);
            } else {
              oids.push(entry.oid);
            }
          }
        }
        await walkTree(treeOid);

        // Generate packfile using git.packObjects
        const packResult = await git.packObjects({
          fs: memFs,
          gitdir,
          oids,
          write: false
        });
        const packfile = packResult.packfile!;

        // Build response: NAK pkt-line + sideband-64k encoded packfile + flush
        // NAK is a regular pkt-line (not sideband — first byte 'N' is not 1/2/3)
        const nakPkt = pktLineBytes(textEncoder.encode("NAK\n"));

        // Packfile data wrapped in sideband channel 1 packets
        const MAX_CHUNK = 65516 - 1; // 65515 bytes data + 1 byte channel = 65516 payload
        const sidebandPackets: Uint8Array[] = [];
        for (let offset = 0; offset < packfile.length; offset += MAX_CHUNK) {
          const chunk = packfile.slice(offset, offset + MAX_CHUNK);
          const payload = new Uint8Array(1 + chunk.length);
          payload[0] = 1; // sideband channel 1 = pack data
          payload.set(chunk, 1);
          sidebandPackets.push(pktLineBytes(payload));
        }

        // Flush packet signals end of response
        const flushPkt = textEncoder.encode("0000");

        // Concatenate: NAK + sideband packets + flush
        const totalLen =
          nakPkt.length +
          sidebandPackets.reduce((s, p) => s + p.length, 0) +
          flushPkt.length;
        const responseBody = new Uint8Array(totalLen);
        let pos = 0;
        responseBody.set(nakPkt, pos);
        pos += nakPkt.length;
        for (const pkt of sidebandPackets) {
          responseBody.set(pkt, pos);
          pos += pkt.length;
        }
        responseBody.set(flushPkt, pos);

        return {
          url: requestUrl,
          method,
          statusCode: 200,
          statusMessage: "OK",
          headers: {
            "content-type": "application/x-git-upload-pack-result"
          },
          body: [responseBody]
        };
      }

      return {
        url: requestUrl,
        method,
        statusCode: 404,
        statusMessage: "Not Found",
        headers: {},
        body: []
      };
    }
  };

  return { url: mockUrl, http: mockHttp };
}
