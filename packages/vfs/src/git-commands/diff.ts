import * as Diff from "diff";
import type { MountableFs } from "just-bash";
import type { GitResult, GitMount } from "./helpers";
import { stripLeadingSlash } from "./helpers";
import type { GitFs } from "../git-fs";
import type { GitStatus } from "../git-repo";

/**
 * Generate unified diff lines for the current overlay changes.
 * Shared between `git diff` (default mode) and `git show`.
 */
export async function generateUnifiedDiff(
  gitFs: GitFs,
  mountableFs: MountableFs,
  mountPoint: string,
  status: GitStatus
): Promise<string> {
  const parts: string[] = [];

  for (const path of status.added) {
    const p = stripLeadingSlash(path);
    const fullPath = `${mountPoint}${path}`;
    let newContent: string;
    try {
      newContent = (await mountableFs.readFile(fullPath, {
        encoding: "utf8"
      })) as string;
    } catch {
      newContent = "";
    }
    parts.push(`diff --git a/${p} b/${p}`);
    parts.push("new file");
    const patch = Diff.createTwoFilesPatch(`a/${p}`, `b/${p}`, "", newContent);
    // createTwoFilesPatch includes a header line "Index: ..." — strip it
    const patchLines = patch.split("\n");
    // Find the first "---" line and include from there
    const startIdx = patchLines.findIndex((l) => l.startsWith("---"));
    if (startIdx >= 0) {
      // Replace the --- header to use /dev/null for new files
      patchLines[startIdx] = "--- /dev/null";
      patchLines[startIdx + 1] = `+++ b/${p}`;
      parts.push(patchLines.slice(startIdx).join("\n"));
    }
  }

  for (const path of status.modified) {
    const p = stripLeadingSlash(path);
    let oldContent: string;
    try {
      oldContent = await gitFs.readBlobUtf8(path);
    } catch {
      oldContent = "";
    }
    const fullPath = `${mountPoint}${path}`;
    let newContent: string;
    try {
      newContent = (await mountableFs.readFile(fullPath, {
        encoding: "utf8"
      })) as string;
    } catch {
      newContent = "";
    }
    parts.push(`diff --git a/${p} b/${p}`);
    const patch = Diff.createTwoFilesPatch(
      `a/${p}`,
      `b/${p}`,
      oldContent,
      newContent
    );
    const patchLines = patch.split("\n");
    const startIdx = patchLines.findIndex((l) => l.startsWith("---"));
    if (startIdx >= 0) {
      parts.push(patchLines.slice(startIdx).join("\n"));
    }
  }

  for (const path of status.deleted) {
    const p = stripLeadingSlash(path);
    let oldContent: string;
    try {
      oldContent = await gitFs.readBlobUtf8(path);
    } catch {
      oldContent = "";
    }
    parts.push(`diff --git a/${p} b/${p}`);
    parts.push("deleted file");
    const patch = Diff.createTwoFilesPatch(`a/${p}`, `b/${p}`, oldContent, "");
    const patchLines = patch.split("\n");
    const startIdx = patchLines.findIndex((l) => l.startsWith("---"));
    if (startIdx >= 0) {
      // Replace the +++ header to use /dev/null for deleted files
      patchLines[startIdx] = `--- a/${p}`;
      patchLines[startIdx + 1] = "+++ /dev/null";
      parts.push(patchLines.slice(startIdx).join("\n"));
    }
  }

  return parts.join("\n");
}

export async function gitDiff(
  match: GitMount,
  args: string[],
  mountableFs?: MountableFs
): Promise<GitResult> {
  const { gitFs, mountPoint } = match;
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

  // Default: show unified diff with line-level content
  if (!mountableFs) {
    // Fallback: header-only (shouldn't happen in normal flow)
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

  const output = await generateUnifiedDiff(
    gitFs,
    mountableFs,
    mountPoint,
    status
  );
  return { stdout: output, stderr: "", exitCode: 0 };
}
