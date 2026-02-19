/**
 * git — virtual git command for just-bash.
 *
 * Supports: git status, git commit, git push, git log, git diff,
 *           git branch, git remote, git show, git rev-parse
 * Operates on GitFs mounts found via MountableFs.
 */

import { defineCommand } from "just-bash";
import type { CustomCommand, MountableFs } from "just-bash";
import { GitFs } from "../git-fs";
import type { MountOptions } from "../mount";
import type { GitMount } from "./helpers";
import { VERSION, USAGE_TEXT } from "./helpers";
import { gitClone } from "./clone";
import { gitStatus } from "./status";
import { gitCommit } from "./commit";
import { gitPush } from "./push";
import { gitLog } from "./log";
import { gitDiff } from "./diff";
import { gitBranch } from "./branch";
import { gitRemote } from "./remote";
import { gitShow } from "./show";
import { gitRevParse } from "./rev-parse";
import { gitPull } from "./pull";

/**
 * Find the GitFs mount whose mountPoint is a prefix of `cwd` (longest prefix wins).
 */
function resolveGitMount(
  mountableFs: MountableFs,
  cwd: string
): GitMount | null {
  let best: GitMount | null = null;

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

export function createGitCommands(
  mountableFs: MountableFs,
  mountOptions?: MountOptions
): CustomCommand[] {
  const gitCmd = defineCommand("git", async (args, ctx) => {
    // Consume -C <path> global options (may appear multiple times, stacks like real git)
    let effectiveCwd = ctx.cwd;
    let remaining = [...args];
    while (remaining.length > 0 && remaining[0] === "-C") {
      if (remaining.length < 2) {
        return {
          stdout: "",
          stderr: "fatal: option '-C' requires a value\n",
          exitCode: 129
        };
      }
      const target = remaining[1];
      effectiveCwd = target.startsWith("/")
        ? target
        : effectiveCwd === "/"
          ? `/${target}`
          : `${effectiveCwd}/${target}`;
      remaining = remaining.slice(2);
    }

    const sub = remaining[0];

    // Global options (don't require a repo)
    if (sub === "--version" || sub === "version") {
      return { stdout: VERSION, stderr: "", exitCode: 0 };
    }
    if (sub === "--help" || sub === "help") {
      return { stdout: USAGE_TEXT, stderr: "", exitCode: 0 };
    }
    if (!sub) {
      return { stdout: "", stderr: USAGE_TEXT, exitCode: 1 };
    }

    // Commands that do NOT require an existing repo
    if (sub === "clone") {
      const subArgs = remaining.slice(1);
      return gitClone(mountableFs, mountOptions, subArgs, ctx);
    }

    // Commands that require a repo
    const match = resolveGitMount(mountableFs, effectiveCwd);
    if (!match) {
      return {
        stdout: "",
        stderr: "fatal: not a git repository\n",
        exitCode: 128
      };
    }

    const subArgs = remaining.slice(1);
    switch (sub) {
      case "status":
        return gitStatus(match, subArgs);
      case "commit":
        return gitCommit(match, subArgs, ctx);
      case "push":
        return gitPush(match, mountableFs);
      case "pull":
        return gitPull(match, mountableFs);
      case "log":
        return gitLog(match, subArgs);
      case "diff":
        return gitDiff(match, subArgs, mountableFs);
      case "branch":
        return gitBranch(match);
      case "remote":
        return gitRemote(match, subArgs);
      case "show":
        return gitShow(match, subArgs, mountableFs);
      case "rev-parse":
        return gitRevParse(match, subArgs);
      default:
        return {
          stdout: "",
          stderr: `git: '${sub}' is not a git command.\n${USAGE_TEXT}`,
          exitCode: 1
        };
    }
  });

  return [gitCmd];
}
