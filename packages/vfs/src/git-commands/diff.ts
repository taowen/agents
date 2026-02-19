import type { GitResult, GitMount } from "./helpers";
import { stripLeadingSlash } from "./helpers";

export async function gitDiff(
  match: GitMount,
  args: string[]
): Promise<GitResult> {
  const { gitFs } = match;
  const status = await gitFs.getStatus();
  const stat = args.includes("--stat");
  const nameOnly = args.includes("--name-only");

  const allPaths = [...status.added, ...status.modified, ...status.deleted];

  if (allPaths.length === 0) {
    return { stdout: "", stderr: "", exitCode: 0 };
  }

  const lines: string[] = [];

  if (nameOnly) {
    for (const path of allPaths) {
      lines.push(stripLeadingSlash(path));
    }
    return { stdout: lines.join("\n") + "\n", stderr: "", exitCode: 0 };
  }

  if (stat) {
    for (const path of status.added) {
      lines.push(` ${stripLeadingSlash(path)} | new file`);
    }
    for (const path of status.modified) {
      lines.push(` ${stripLeadingSlash(path)} | modified`);
    }
    for (const path of status.deleted) {
      lines.push(` ${stripLeadingSlash(path)} | deleted`);
    }
    lines.push(
      ` ${allPaths.length} file${allPaths.length !== 1 ? "s" : ""} changed`
    );
    return { stdout: lines.join("\n") + "\n", stderr: "", exitCode: 0 };
  }

  // Default: show diff headers for each changed file
  for (const path of status.added) {
    const p = stripLeadingSlash(path);
    lines.push(`diff --git a/${p} b/${p}`);
    lines.push("new file");
  }
  for (const path of status.modified) {
    const p = stripLeadingSlash(path);
    lines.push(`diff --git a/${p} b/${p}`);
  }
  for (const path of status.deleted) {
    const p = stripLeadingSlash(path);
    lines.push(`diff --git a/${p} b/${p}`);
    lines.push("deleted file");
  }

  return { stdout: lines.join("\n") + "\n", stderr: "", exitCode: 0 };
}
