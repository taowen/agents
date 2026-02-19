import type { GitResult, GitMount } from "./helpers";
import { shortBranchName } from "./helpers";

export async function gitCommit(
  match: GitMount,
  args: string[],
  ctx: { env: Map<string, string> }
): Promise<GitResult> {
  const { gitFs } = match;

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

  // Parse --author="Name <email>"
  let authorName = ctx.env.get("GIT_AUTHOR_NAME") || "AI Assistant";
  let authorEmail = ctx.env.get("GIT_AUTHOR_EMAIL") || "ai@assistant.local";

  for (const arg of args) {
    const authorMatch = arg.match(/^--author=["']?(.+?)\s*<(.+?)>["']?$/);
    if (authorMatch) {
      authorName = authorMatch[1].trim();
      authorEmail = authorMatch[2].trim();
    }
  }

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
