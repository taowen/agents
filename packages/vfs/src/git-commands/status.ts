import type { GitResult, GitMount } from "./helpers";
import { stripLeadingSlash, shortBranchName } from "./helpers";

export async function gitStatus(
  match: GitMount,
  args: string[]
): Promise<GitResult> {
  const { gitFs } = match;
  const status = await gitFs.getStatus();
  const hasChanges =
    status.added.length > 0 ||
    status.modified.length > 0 ||
    status.deleted.length > 0;

  const branch = shortBranchName(gitFs.getRef() || "main");
  const short = args.includes("-s") || args.includes("--short");

  if (short) {
    const lines: string[] = [];
    for (const path of status.added) {
      lines.push(`A  ${stripLeadingSlash(path)}`);
    }
    for (const path of status.modified) {
      lines.push(`M  ${stripLeadingSlash(path)}`);
    }
    for (const path of status.deleted) {
      lines.push(`D  ${stripLeadingSlash(path)}`);
    }
    return {
      stdout: lines.length > 0 ? lines.join("\n") + "\n" : "",
      stderr: "",
      exitCode: 0
    };
  }

  const lines: string[] = [];
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
