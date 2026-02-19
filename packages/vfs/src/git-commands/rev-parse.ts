import type { GitResult, GitMount } from "./helpers";

export async function gitRevParse(
  match: GitMount,
  args: string[]
): Promise<GitResult> {
  const { gitFs } = match;
  const short = args.includes("--short");

  // Filter out flags to find the revision argument
  const revArgs = args.filter((a) => !a.startsWith("-"));

  if (revArgs.length === 0 && !args.includes("HEAD")) {
    // Check if HEAD was passed with --short
    const hasHead = args.some((a) => a === "HEAD");
    if (!hasHead) {
      return {
        stdout: "",
        stderr: "fatal: bad revision\n",
        exitCode: 128
      };
    }
  }

  const oid = gitFs.getCommitOid();
  if (!oid) {
    return {
      stdout: "",
      stderr: "fatal: bad revision 'HEAD'\n",
      exitCode: 128
    };
  }

  const output = short ? oid.slice(0, 7) : oid;
  return { stdout: output + "\n", stderr: "", exitCode: 0 };
}
