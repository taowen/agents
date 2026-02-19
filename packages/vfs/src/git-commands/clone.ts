/**
 * git clone — clone a remote git repository.
 *
 * Usage: git clone [--depth <n>] [--branch <ref>] [-b <ref>] <url> [<directory>]
 */

import type { MountableFs } from "just-bash";
import { GitFs } from "../git-fs";
import { parseGitCredentials, findCredential } from "../git-credentials";
import { parseFstab } from "../fstab";
import type { MountOptions } from "../mount";
import type { GitResult } from "./helpers";

/**
 * Extract a repository name from a git URL (strip trailing .git).
 */
function repoNameFromUrl(url: string): string {
  // Remove trailing slash
  let cleaned = url.replace(/\/+$/, "");
  // Remove .git suffix
  cleaned = cleaned.replace(/\.git$/, "");
  // Take the last path segment
  const lastSlash = cleaned.lastIndexOf("/");
  return lastSlash >= 0 ? cleaned.slice(lastSlash + 1) : cleaned;
}

export async function gitClone(
  mountableFs: MountableFs,
  mountOptions: MountOptions | undefined,
  args: string[],
  ctx: { cwd: string; env: Map<string, string> }
): Promise<GitResult> {
  // Parse args — git clone defaults to full history (no depth limit)
  let depth: number | undefined;
  let ref: string | undefined;
  let url: string | undefined;
  let directory: string | undefined;

  const positional: string[] = [];

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--depth" && i + 1 < args.length) {
      depth = parseInt(args[++i], 10);
      if (isNaN(depth!) || depth! < 1) {
        return {
          stdout: "",
          stderr: "fatal: depth must be a positive integer\n",
          exitCode: 128
        };
      }
    } else if ((arg === "--branch" || arg === "-b") && i + 1 < args.length) {
      ref = args[++i];
    } else if (!arg.startsWith("-")) {
      positional.push(arg);
    } else {
      return {
        stdout: "",
        stderr: `fatal: unknown option '${arg}'\n`,
        exitCode: 128
      };
    }
  }

  url = positional[0];
  directory = positional[1];

  if (!url) {
    return {
      stdout: "",
      stderr:
        "fatal: You must specify a repository to clone.\n" +
        "usage: git clone [--depth <n>] [--branch <ref>] [-b <ref>] <url> [<directory>]\n",
      exitCode: 128
    };
  }

  // Default directory: /mnt/<repo-name>
  if (!directory) {
    const repoName = repoNameFromUrl(url);
    directory = `/mnt/${repoName}`;
  }

  // Resolve to absolute path
  if (!directory.startsWith("/")) {
    directory = ctx.cwd === "/" ? `/${directory}` : `${ctx.cwd}/${directory}`;
  }

  // Check mountOptions
  if (!mountOptions?.r2Bucket || !mountOptions?.userId) {
    return {
      stdout: "",
      stderr:
        "fatal: git clone requires r2Bucket and userId in mount options\n",
      exitCode: 1
    };
  }

  // Resolve credentials: env vars → URL userinfo → /etc/git-credentials
  let username: string | undefined;
  let password: string | undefined;

  username = ctx.env.get("GIT_USERNAME") ?? undefined;
  password = ctx.env.get("GIT_PASSWORD") ?? undefined;

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
    http: mountOptions?.gitHttp,
    r2Bucket: mountOptions.r2Bucket,
    userId: mountOptions.userId,
    mountPoint: directory
  });

  try {
    mountableFs.mount(directory, gitFs, "git");
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { stdout: "", stderr: `fatal: ${msg}\n`, exitCode: 128 };
  }

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
          (e) => e.mountPoint === directory
        );
        if (!exists) {
          const optParts: string[] = [];
          const resolvedRef = gitFs.getRef();
          if (resolvedRef && resolvedRef !== "main")
            optParts.push(`ref=${resolvedRef}`);
          if (depth !== undefined) optParts.push(`depth=${depth}`);
          const optsField =
            optParts.length > 0 ? optParts.join(",") : "defaults";
          const fstabLine = `${url}  ${directory}  git  ${optsField}  0  0`;
          if (!fstab.endsWith("\n")) fstab += "\n";
          fstab += fstabLine + "\n";
          await mountableFs.writeFile("/etc/fstab", fstab);
        }
      } catch {
        /* non-fatal */
      }
      return {
        stdout: "",
        stderr: `Cloning into '${directory}'...\n`,
        exitCode: 0
      };
    },
    (err) => {
      try {
        mountableFs.unmount(directory!);
      } catch {
        /* ignore */
      }
      const msg = err instanceof Error ? err.message : String(err);
      return {
        stdout: "",
        stderr: `fatal: clone failed: ${msg}\n`,
        exitCode: 128
      };
    }
  );
}
