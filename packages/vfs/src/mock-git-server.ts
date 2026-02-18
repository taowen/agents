/**
 * mock-git-server — in-memory git repo + mock HTTP transport for testing.
 *
 * Creates a git repository in memory using isomorphic-git,
 * then provides a mock HTTP transport that implements the git smart HTTP protocol
 * (info/refs + git-upload-pack) so GitFs can clone without network access.
 */

import git from "isomorphic-git";
import { createIsomorphicGitMemFs } from "./isomorphic-git-memfs";

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

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

function concatUint8Arrays(arrays: Uint8Array[]): Uint8Array {
  const totalLen = arrays.reduce((s, a) => s + a.length, 0);
  const result = new Uint8Array(totalLen);
  let pos = 0;
  for (const a of arrays) {
    result.set(a, pos);
    pos += a.length;
  }
  return result;
}

// ---- Create mock git server ----

export async function createMockGitServer(
  files: Record<string, string>
): Promise<{
  url: string;
  http: { request: (opts: any) => Promise<any> };
}> {
  const memFs = createIsomorphicGitMemFs();
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

  // Get HEAD oid and ref (mutable — updated by receive-pack)
  let headOid = await git.resolveRef({ fs: memFs, dir, ref: "HEAD" });

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

      // GET /info/refs?service=git-receive-pack
      if (
        method === "GET" &&
        requestUrl.includes("/info/refs") &&
        requestUrl.includes("service=git-receive-pack")
      ) {
        const capabilities =
          "report-status delete-refs ofs-delta agent=mock-git-server";
        const serviceLine = pktLine("# service=git-receive-pack\n");
        const firstRef = pktLine(
          `${headOid} refs/heads/main\0${capabilities}\n`
        );
        const body = serviceLine + FLUSH + firstRef + FLUSH;

        return {
          url: requestUrl,
          method,
          statusCode: 200,
          statusMessage: "OK",
          headers: {
            "content-type": "application/x-git-receive-pack-advertisement"
          },
          body: [textEncoder.encode(body)]
        };
      }

      // GET /info/refs?service=git-upload-pack
      if (method === "GET" && requestUrl.includes("/info/refs")) {
        const capabilities =
          "multi_ack_detailed no-done side-band-64k thin-pack ofs-delta shallow symref=HEAD:refs/heads/main agent=mock-git-server";

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

      // POST /git-receive-pack
      if (
        method === "POST" &&
        requestUrl.includes("git-receive-pack") &&
        !requestUrl.includes("git-upload-pack")
      ) {
        const bodyChunks: Uint8Array[] = [];
        if (opts.body) {
          for await (const chunk of opts.body) {
            bodyChunks.push(chunk);
          }
        }
        const body = concatUint8Arrays(bodyChunks);

        // Parse pkt-lines to find ref update commands and packfile
        let offset = 0;
        const commands: Array<{
          oldOid: string;
          newOid: string;
          refName: string;
        }> = [];

        while (offset < body.length) {
          const lenHex = textDecoder.decode(body.slice(offset, offset + 4));
          if (lenHex === "0000") {
            offset += 4;
            break;
          }
          const len = parseInt(lenHex, 16);
          if (len === 0) {
            offset += 4;
            break;
          }
          const lineBytes = body.slice(offset + 4, offset + len);
          const line = textDecoder.decode(lineBytes);
          offset += len;

          // ref update line: "<old-oid> <new-oid> <refname>\n"
          // first line may have \0capabilities
          const cleanLine = line.split("\0")[0].trim();
          const parts = cleanLine.split(" ");
          if (parts.length >= 3 && parts[0].length === 40) {
            commands.push({
              oldOid: parts[0],
              newOid: parts[1],
              refName: parts[2]
            });
          }
        }

        // Remaining bytes are the packfile — write it to the memFs and index it
        if (offset < body.length) {
          const packData = body.slice(offset);
          // Write the pack to the git objects
          const packPath = `${gitdir}/objects/pack/incoming.pack`;
          await memFs.writeFile(packPath, packData);
          try {
            await git.indexPack({
              fs: memFs,
              dir,
              gitdir,
              filepath: `objects/pack/incoming.pack`
            });
          } catch {
            // indexPack may fail in some isomorphic-git versions;
            // the objects are already unpacked by the push side
          }
        }

        // Update refs
        for (const cmd of commands) {
          if (cmd.refName === "refs/heads/main") {
            headOid = cmd.newOid;
            // Update the actual ref in the memFs git repo
            await memFs.writeFile(
              `${gitdir}/refs/heads/main`,
              cmd.newOid + "\n"
            );
            await memFs.writeFile(`${gitdir}/HEAD`, "ref: refs/heads/main\n");
          }
        }

        // Build report-status response wrapped in sideband channel 1
        // (isomorphic-git always demuxes push responses via GitSideBand.demux)
        const statusParts: Uint8Array[] = [];
        statusParts.push(pktLineBytes(textEncoder.encode("unpack ok\n")));
        for (const cmd of commands) {
          statusParts.push(
            pktLineBytes(textEncoder.encode(`ok ${cmd.refName}\n`))
          );
        }
        statusParts.push(textEncoder.encode(FLUSH));
        const statusData = concatUint8Arrays(statusParts);

        // Wrap in sideband channel 1
        const sbPayload = new Uint8Array(1 + statusData.length);
        sbPayload[0] = 1; // channel 1 = pack data / report-status
        sbPayload.set(statusData, 1);

        const responseParts: Uint8Array[] = [];
        responseParts.push(pktLineBytes(sbPayload));
        responseParts.push(textEncoder.encode(FLUSH));
        const responseBody = concatUint8Arrays(responseParts);

        return {
          url: requestUrl,
          method,
          statusCode: 200,
          statusMessage: "OK",
          headers: {
            "content-type": "application/x-git-receive-pack-result"
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
