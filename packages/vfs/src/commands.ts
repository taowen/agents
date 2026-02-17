/**
 * mount / umount custom commands for just-bash.
 *
 * Usage:
 *   mount                              — list mounts
 *   mount -t git [-o ...] <url> <mp>   — mount a git repo
 *   mount -t agentfs [-o ...] <dev> <mp> — mount an agentfs path
 *   umount <mountpoint>
 */

import { defineCommand } from "just-bash";
import type { CustomCommand, MountableFs } from "just-bash";
import type { AgentFS } from "agentfs-sdk/cloudflare";
import { AgentFsAdapter } from "./agentfs-adapter";
import { D1FsAdapter } from "./d1-fs-adapter";
import { R2FsAdapter } from "./r2-fs-adapter";
import { GitFs } from "./git-fs";
import { parseGitCredentials, findCredential } from "./git-credentials";
import { parseFstab } from "./fstab";
import type { MountOptions } from "./mount";

/**
 * Parse a `-o key=val,key2=val2` options string into a record.
 */
function parseOpts(raw: string): Record<string, string> {
  const result: Record<string, string> = {};
  for (const pair of raw.split(",")) {
    const eq = pair.indexOf("=");
    if (eq === -1) {
      result[pair] = "";
    } else {
      result[pair.slice(0, eq)] = pair.slice(eq + 1);
    }
  }
  return result;
}

export function createMountCommands(
  mountableFs: MountableFs,
  agentFs?: AgentFS,
  options?: MountOptions
): CustomCommand[] {
  const mountCmd = defineCommand("mount", async (args, ctx) => {
    // No args: list mounted filesystems
    if (args.length === 0) {
      const mounts = mountableFs.getMounts();
      const lines = mounts.map((m) => {
        let fsType = "unknown";
        if (m.filesystem instanceof D1FsAdapter) fsType = "d1fs";
        else if (m.filesystem instanceof R2FsAdapter) fsType = "r2fs";
        else if (m.filesystem instanceof AgentFsAdapter) fsType = "agentfs";
        else if (m.filesystem instanceof GitFs) fsType = "git";
        return `${fsType} on ${m.mountPoint}`;
      });
      return {
        stdout: lines.length ? lines.join("\n") + "\n" : "",
        stderr: "",
        exitCode: 0
      };
    }

    // Parse flags: -t <type> and -o <options>
    let type: string | undefined;
    let optsRaw: string | undefined;
    const positional: string[] = [];

    for (let i = 0; i < args.length; i++) {
      if (args[i] === "-t" && i + 1 < args.length) {
        type = args[++i];
      } else if (args[i] === "-o" && i + 1 < args.length) {
        optsRaw = args[++i];
      } else {
        positional.push(args[i]);
      }
    }

    if (!type) {
      return {
        stdout: "",
        stderr:
          "mount: missing -t flag (usage: mount -t <type> <device> <mountpoint>)\n",
        exitCode: 1
      };
    }

    if (type === "git") {
      return mountGit(mountableFs, positional, optsRaw, ctx, options);
    } else if (type === "agentfs") {
      return mountAgentFs(mountableFs, agentFs, positional);
    } else {
      return {
        stdout: "",
        stderr: `mount: unsupported filesystem type '${type}'\n`,
        exitCode: 1
      };
    }
  });

  const umountCmd = defineCommand("umount", async (args) => {
    if (args.length === 0) {
      return {
        stdout: "",
        stderr: "umount: usage: umount <mountpoint>\n",
        exitCode: 1
      };
    }

    const mountpoint = args[0];

    try {
      mountableFs.unmount(mountpoint);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return { stdout: "", stderr: `umount: ${msg}\n`, exitCode: 1 };
    }

    // Auto-remove from /etc/fstab
    try {
      const fstab = (await mountableFs.readFile("/etc/fstab", {
        encoding: "utf8"
      })) as string;
      const lines = fstab.split("\n");
      const filtered = lines.filter((line) => {
        const t = line.trim();
        if (!t || t.startsWith("#")) return true;
        return t.split(/\s+/)[1] !== mountpoint;
      });
      await mountableFs.writeFile("/etc/fstab", filtered.join("\n"));
    } catch {
      /* non-fatal: fstab cleanup failure should not break umount */
    }

    return { stdout: "", stderr: "", exitCode: 0 };
  });

  return [mountCmd, umountCmd];
}

// ---- mount -t git ----

async function mountGit(
  mountableFs: MountableFs,
  positional: string[],
  optsRaw: string | undefined,
  ctx: { env: Map<string, string> },
  mountOptions?: MountOptions
) {
  if (positional.length < 2) {
    return {
      stdout: "",
      stderr: "mount: usage: mount -t git [-o options] <url> <mountpoint>\n",
      exitCode: 1
    };
  }

  const url = positional[0];
  const mountpoint = positional[1];
  const opts = optsRaw ? parseOpts(optsRaw) : {};
  const ref = opts.ref || "main";
  const depth = opts.depth ? parseInt(opts.depth, 10) : 1;

  if (isNaN(depth) || depth < 1) {
    return {
      stdout: "",
      stderr: "mount: invalid depth value\n",
      exitCode: 1
    };
  }

  // Resolve auth: -o > env vars > URL userinfo
  let username: string | undefined = opts.username;
  let password: string | undefined = opts.password;

  if (!username) {
    username = ctx.env.get("GIT_USERNAME") ?? undefined;
  }
  if (!password) {
    password = ctx.env.get("GIT_PASSWORD") ?? undefined;
  }

  // Fallback: extract userinfo from URL
  if (!username) {
    try {
      const parsed = new URL(url);
      if (parsed.username) {
        username = decodeURIComponent(parsed.username);
        if (!password && parsed.password) {
          password = decodeURIComponent(parsed.password);
        }
      }
    } catch {
      // URL parsing failed — will be caught by GitFs later
    }
  }

  // Fallback: read /etc/git-credentials
  if (!username) {
    try {
      const credContent = await mountableFs.readFile("/etc/git-credentials", {
        encoding: "utf8"
      });
      const creds = parseGitCredentials(credContent as string);
      const match = findCredential(creds, url);
      if (match) {
        username = match.username;
        password = match.password;
      }
    } catch {
      // file doesn't exist — skip
    }
  }

  const onAuth = username
    ? () => ({ username: username!, password })
    : undefined;

  const gitFs = new GitFs({
    url,
    ref,
    depth,
    onAuth,
    http: mountOptions?.gitHttp
  });

  try {
    mountableFs.mount(mountpoint, gitFs);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { stdout: "", stderr: `mount: ${msg}\n`, exitCode: 1 };
  }

  // Eagerly clone so errors surface at mount time
  return gitFs.init().then(
    async () => {
      // Auto-persist to /etc/fstab
      try {
        let fstab: string;
        try {
          fstab = (await mountableFs.readFile("/etc/fstab", {
            encoding: "utf8"
          })) as string;
        } catch {
          fstab = "";
        }
        const exists = parseFstab(fstab).some(
          (e) => e.mountPoint === mountpoint
        );
        if (!exists) {
          // Build options field: ref, depth (omit defaults and credentials)
          const optParts: string[] = [];
          if (ref !== "main") optParts.push(`ref=${ref}`);
          if (depth !== 1) optParts.push(`depth=${depth}`);
          const optsField =
            optParts.length > 0 ? optParts.join(",") : "defaults";
          const fstabLine = `${url}  ${mountpoint}  git  ${optsField}  0  0`;
          if (!fstab.endsWith("\n")) fstab += "\n";
          fstab += fstabLine + "\n";
          await mountableFs.writeFile("/etc/fstab", fstab);
        }
      } catch {
        /* non-fatal: fstab persist failure should not break mount */
      }
      return { stdout: "", stderr: "", exitCode: 0 };
    },
    (err) => {
      // Unmount on failure so the mount point isn't left in a broken state
      try {
        mountableFs.unmount(mountpoint);
      } catch {
        /* ignore */
      }
      const msg = err instanceof Error ? err.message : String(err);
      return {
        stdout: "",
        stderr: `mount: clone failed: ${msg}\n`,
        exitCode: 1
      };
    }
  );
}

// ---- mount -t agentfs ----

function mountAgentFs(
  mountableFs: MountableFs,
  agentFs: AgentFS | undefined,
  positional: string[]
) {
  if (!agentFs) {
    return {
      stdout: "",
      stderr: "mount: agentfs not available (no AgentFS provided)\n",
      exitCode: 1
    };
  }

  if (positional.length < 2) {
    return {
      stdout: "",
      stderr:
        "mount: usage: mount -t agentfs [-o options] <device> <mountpoint>\n",
      exitCode: 1
    };
  }

  const mountpoint = positional[1];
  const adapter = new AgentFsAdapter(agentFs, mountpoint);

  try {
    mountableFs.mount(mountpoint, adapter);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { stdout: "", stderr: `mount: ${msg}\n`, exitCode: 1 };
  }

  return { stdout: "", stderr: "", exitCode: 0 };
}
