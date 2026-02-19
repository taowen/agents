import type { GitResult, GitMount } from "./helpers";

export async function gitLog(
  match: GitMount,
  args: string[]
): Promise<GitResult> {
  const { gitFs } = match;

  // Parse options
  const oneline = args.includes("--oneline");
  let maxCount: number | undefined;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "-n" && i + 1 < args.length) {
      maxCount = parseInt(args[i + 1], 10);
      i++;
    } else if (/^-\d+$/.test(arg)) {
      maxCount = parseInt(arg.slice(1), 10);
    }
  }

  const entries = await gitFs.getLog(maxCount);

  if (entries.length === 0) {
    return { stdout: "", stderr: "", exitCode: 0 };
  }

  const lines: string[] = [];

  for (const entry of entries) {
    if (oneline) {
      const shortOid = entry.oid.slice(0, 7);
      const firstLine = entry.message.split("\n")[0];
      lines.push(`${shortOid} ${firstLine}`);
    } else {
      lines.push(`commit ${entry.oid}`);
      lines.push(`Author: ${entry.author.name} <${entry.author.email}>`);
      const date = new Date(entry.author.timestamp * 1000);
      lines.push(`Date:   ${date.toUTCString()}`);
      lines.push("");
      // Indent each line of the message with 4 spaces
      for (const msgLine of entry.message.trimEnd().split("\n")) {
        lines.push(`    ${msgLine}`);
      }
      lines.push("");
    }
  }

  return { stdout: lines.join("\n") + "\n", stderr: "", exitCode: 0 };
}
