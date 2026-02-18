/**
 * git — virtual git command for just-bash.
 *
 * Supports: git status, git commit -m "msg", git push
 * Operates on GitFs mounts found via MountableFs.
 */

import { defineCommand } from "just-bash";
import type { CustomCommand, MountableFs } from "just-bash";
import { GitFs } from "./git-fs";
import { parseGitCredentials, findCredential } from "./git-credentials";

/**
 * Find the GitFs mount whose mountPoint is a prefix of `cwd` (longest prefix wins).
 */
function resolveGitMount(
  mountableFs: MountableFs,
  cwd: string
): { gitFs: GitFs; mountPoint: string } | null {
  let best: { gitFs: GitFs; mountPoint: string } | null = null;

  for (const { mountPoint, filesystem } of mountableFs.getMounts()) {
    if (!(filesystem instanceof GitFs)) continue;
    if (cwd === mountPoint || cwd.startsWith(mountPoint + "/")) {
      if (!best || mountPoint.length > best.mountPoint.length) {
        best = { gitFs: filesystem, mountPoint };
      }
    }
  }

  return best;
}

export function createGitCommands(mountableFs: MountableFs): CustomCommand[] {
  const gitCmd = defineCommand("git", async (args, ctx) => {
    const subcommand = args[0];

    if (!subcommand) {
      return {
        stdout: "",
        stderr:
          "usage: git <command>\n\nAvailable commands: status, commit, push\n",
        exitCode: 1
      };
    }

    const match = resolveGitMount(mountableFs, ctx.cwd);
    if (!match) {
      return {
        stdout: "",
        stderr: "fatal: not a git repository\n",
        exitCode: 128
      };
    }

    const { gitFs, mountPoint } = match;

    switch (subcommand) {
      case "status":
        return gitStatus(gitFs, mountPoint);
      case "commit":
        return gitCommit(gitFs, mountPoint, args.slice(1), ctx);
      case "push":
        return gitPush(gitFs, mountableFs);
      default:
        return {
          stdout: "",
          stderr: `git: '${subcommand}' is not a git command.\n\nAvailable commands: status, commit, push\n`,
          exitCode: 1
        };
    }
  });

  return [gitCmd];
}

async function gitStatus(
  gitFs: GitFs,
  mountPoint: string
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const status = await gitFs.getStatus();
  const hasChanges =
    status.added.length > 0 ||
    status.modified.length > 0 ||
    status.deleted.length > 0;

  const lines: string[] = [];
  const branch = shortBranchName(gitFs.getRef() || "main");
  lines.push(`On branch ${branch}`);

  if (gitFs.hasUnpushedCommits()) {
    lines.push(`Your branch is ahead of 'origin/${branch}'.`);
  }

  if (!hasChanges) {
    lines.push("nothing to commit, working tree clean");
  } else {
    lines.push("Changes to be committed:");
    lines.push("");
    for (const path of status.added) {
      lines.push(`\tnew file:   ${stripLeadingSlash(path)}`);
    }
    for (const path of status.modified) {
      lines.push(`\tmodified:   ${stripLeadingSlash(path)}`);
    }
    for (const path of status.deleted) {
      lines.push(`\tdeleted:    ${stripLeadingSlash(path)}`);
    }
  }

  return { stdout: lines.join("\n") + "\n", stderr: "", exitCode: 0 };
}

async function gitCommit(
  gitFs: GitFs,
  mountPoint: string,
  args: string[],
  ctx: { env: Map<string, string> }
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  // Parse -m flag
  let message: string | undefined;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "-m" && i + 1 < args.length) {
      message = args[i + 1];
      break;
    }
  }

  if (!message) {
    return {
      stdout: "",
      stderr: "error: switch `m' requires a value\n",
      exitCode: 1
    };
  }

  const authorName = ctx.env.get("GIT_AUTHOR_NAME") || "AI Assistant";
  const authorEmail = ctx.env.get("GIT_AUTHOR_EMAIL") || "ai@assistant.local";

  try {
    const oid = await gitFs.commit(message, {
      name: authorName,
      email: authorEmail
    });
    const branch = shortBranchName(gitFs.getRef() || "main");
    const shortOid = oid.slice(0, 7);
    return {
      stdout: `[${branch} ${shortOid}] ${message}\n`,
      stderr: "",
      exitCode: 0
    };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { stdout: "", stderr: `${msg}\n`, exitCode: 1 };
  }
}

async function gitPush(
  gitFs: GitFs,
  mountableFs: MountableFs
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  if (!gitFs.hasUnpushedCommits()) {
    return {
      stdout: "Everything up-to-date\n",
      stderr: "",
      exitCode: 0
    };
  }

  // Try to read credentials from /etc/git-credentials
  let onAuth: (() => { username: string; password?: string }) | undefined;
  try {
    const cred = await mountableFs.readFile("/etc/git-credentials", {
      encoding: "utf8"
    });
    const match = findCredential(
      parseGitCredentials(cred as string),
      gitFs.getUrl()
    );
    if (match) {
      onAuth = () => ({ username: match.username, password: match.password });
    }
  } catch {
    // no credentials file
  }

  try {
    await gitFs.push(onAuth);
    return { stdout: "", stderr: "", exitCode: 0 };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { stdout: "", stderr: `error: push failed: ${msg}\n`, exitCode: 1 };
  }
}

function stripLeadingSlash(path: string): string {
  return path.startsWith("/") ? path.slice(1) : path;
}

function shortBranchName(ref: string): string {
  if (ref.startsWith("refs/heads/")) return ref.slice("refs/heads/".length);
  return ref;
}
