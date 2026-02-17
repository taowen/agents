import {
  GitFs,
  parseGitCredentials,
  findCredential,
  D1FsAdapter,
  R2FsAdapter
} from "vfs";
import { MountableFs, defineCommand } from "just-bash";
import type { CustomCommand } from "just-bash";

/**
 * Create mount/umount commands for the D1-backed filesystem.
 * Mirrors vfs createMountCommands but without AgentFS dependency.
 */
export function createD1MountCommands(
  mountableFs: MountableFs
): CustomCommand[] {
  const mountCmd = defineCommand(
    "mount",
    async (args: string[], ctx: { env: Map<string, string> }) => {
      if (args.length === 0) {
        const mounts = mountableFs.getMounts();
        const lines = mounts.map((m) => {
          let fsType = "unknown";
          if (m.filesystem instanceof D1FsAdapter) fsType = "d1fs";
          else if (m.filesystem instanceof R2FsAdapter) fsType = "r2fs";
          else if (m.filesystem instanceof GitFs) fsType = "git";
          return `${fsType} on ${m.mountPoint}`;
        });
        return {
          stdout: lines.length ? lines.join("\n") + "\n" : "",
          stderr: "",
          exitCode: 0
        };
      }

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
        if (positional.length < 2) {
          return {
            stdout: "",
            stderr:
              "mount: usage: mount -t git [-o options] <url> <mountpoint>\n",
            exitCode: 1
          };
        }
        const url = positional[0];
        const mountpoint = positional[1];
        const opts = optsRaw ? parseOpts(optsRaw) : {};
        const ref = opts.ref || "main";
        const depth = opts.depth ? parseInt(opts.depth, 10) : 1;

        let username: string | undefined = opts.username;
        let password: string | undefined = opts.password;

        if (!username) username = ctx.env.get("GIT_USERNAME") ?? undefined;
        if (!password) password = ctx.env.get("GIT_PASSWORD") ?? undefined;

        if (!username) {
          try {
            const parsed = new URL(url);
            if (parsed.username) {
              username = decodeURIComponent(parsed.username);
              if (!password && parsed.password)
                password = decodeURIComponent(parsed.password);
            }
          } catch {
            /* ignore — URL may not be parseable */
          }
        }

        if (!username) {
          try {
            const credContent = await mountableFs.readFile(
              "/etc/git-credentials",
              { encoding: "utf8" }
            );
            const creds = parseGitCredentials(credContent as string);
            const match = findCredential(creds, url);
            if (match) {
              username = match.username;
              password = match.password;
            }
          } catch {
            /* ignore — credentials file may not exist */
          }
        }

        const onAuth = username
          ? () => ({ username: username!, password })
          : undefined;
        const gitFs = new GitFs({
          url,
          ref,
          depth: isNaN(depth) || depth < 1 ? 1 : depth,
          onAuth
        });

        try {
          mountableFs.mount(mountpoint, gitFs);
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          return { stdout: "", stderr: `mount: ${msg}\n`, exitCode: 1 };
        }

        return gitFs.init().then(
          () => ({ stdout: "", stderr: "", exitCode: 0 }),
          (err) => {
            try {
              mountableFs.unmount(mountpoint);
            } catch {
              /* ignore — unmount during error recovery */
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

      return {
        stdout: "",
        stderr: `mount: unsupported filesystem type '${type}'\n`,
        exitCode: 1
      };
    }
  );

  const umountCmd = defineCommand("umount", async (args: string[]) => {
    if (args.length === 0) {
      return {
        stdout: "",
        stderr: "umount: usage: umount <mountpoint>\n",
        exitCode: 1
      };
    }
    try {
      mountableFs.unmount(args[0]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return { stdout: "", stderr: `umount: ${msg}\n`, exitCode: 1 };
    }
    return { stdout: "", stderr: "", exitCode: 0 };
  });

  return [mountCmd, umountCmd];
}

function parseOpts(raw: string): Record<string, string> {
  const result: Record<string, string> = {};
  for (const pair of raw.split(",")) {
    const eq = pair.indexOf("=");
    if (eq === -1) result[pair] = "";
    else result[pair.slice(0, eq)] = pair.slice(eq + 1);
  }
  return result;
}
