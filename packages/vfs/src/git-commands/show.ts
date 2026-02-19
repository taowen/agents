import type { GitResult, GitMount } from "./helpers";

export async function gitShow(
  match: GitMount,
  args: string[]
): Promise<GitResult> {
  const { gitFs } = match;
  const stat = args.includes("--stat");

  const entries = await gitFs.getLog(1);
  if (entries.length === 0) {
    return {
      stdout: "",
      stderr: "fatal: bad default revision 'HEAD'\n",
      exitCode: 128
    };
  }

  const entry = entries[0];
  const lines: string[] = [];
  lines.push(`commit ${entry.oid}`);
  lines.push(`Author: ${entry.author.name} <${entry.author.email}>`);
  const date = new Date(entry.author.timestamp * 1000);
  lines.push(`Date:   ${date.toUTCString()}`);
  lines.push("");
  for (const msgLine of entry.message.trimEnd().split("\n")) {
    lines.push(`    ${msgLine}`);
  }

  if (stat) {
    // Show file change stats from status (approximation — real git compares parent commit)
    const status = await gitFs.getStatus();
    const allPaths = [...status.added, ...status.modified, ...status.deleted];
    if (allPaths.length > 0) {
      lines.push("");
      for (const path of allPaths) {
        const p = path.startsWith("/") ? path.slice(1) : path;
        lines.push(` ${p}`);
      }
      lines.push(
        ` ${allPaths.length} file${allPaths.length !== 1 ? "s" : ""} changed`
      );
    }
  }

  lines.push("");
  return { stdout: lines.join("\n") + "\n", stderr: "", exitCode: 0 };
}
