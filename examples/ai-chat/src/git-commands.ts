/**
 * mount / umount custom commands for just-bash.
 *
 * Usage:
 *   mount -t git [-o ref=dev,depth=1,username=u,password=p] <url> <mountpoint>
 *   umount <mountpoint>
 */

import { defineCommand } from "just-bash";
import type { CustomCommand } from "just-bash";
import { GitFs } from "./git-fs";
import type { MountableFs } from "just-bash";

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

export function createGitMountCommands(fs: MountableFs): CustomCommand[] {
  const mountCmd = defineCommand("mount", async (args, ctx) => {
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

    // Validate -t git
    if (!type) {
      return {
        stdout: "",
        stderr:
          "mount: missing -t flag (usage: mount -t git <url> <mountpoint>)\n",
        exitCode: 1
      };
    }
    if (type !== "git") {
      return {
        stdout: "",
        stderr: `mount: unsupported filesystem type '${type}'\n`,
        exitCode: 1
      };
    }

    // Need exactly <url> <mountpoint>
    if (positional.length < 2) {
      return {
        stdout: "",
        stderr: "mount: usage: mount -t git [-o options] <url> <mountpoint>\n",
        exitCode: 1
      };
    }

    const url = positional[0];
    const mountpoint = positional[1];

    // Parse -o options
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
    let username = opts.username;
    let password = opts.password;

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

    const onAuth = username
      ? () => ({ username: username!, password })
      : undefined;

    const gitFs = new GitFs({ url, ref, depth, onAuth });

    try {
      fs.mount(mountpoint, gitFs);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return { stdout: "", stderr: `mount: ${msg}\n`, exitCode: 1 };
    }

    // Eagerly clone so errors surface at mount time
    try {
      await gitFs.init();
    } catch (err) {
      // Unmount on failure so the mount point isn't left in a broken state
      try {
        fs.unmount(mountpoint);
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

    return { stdout: "", stderr: "", exitCode: 0 };
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
      fs.unmount(mountpoint);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return { stdout: "", stderr: `umount: ${msg}\n`, exitCode: 1 };
    }

    return { stdout: "", stderr: "", exitCode: 0 };
  });

  return [mountCmd, umountCmd];
}
