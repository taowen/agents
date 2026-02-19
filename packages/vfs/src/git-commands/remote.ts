import type { GitResult, GitMount } from "./helpers";

export async function gitRemote(
  match: GitMount,
  args: string[]
): Promise<GitResult> {
  const { gitFs } = match;
  const verbose = args.includes("-v") || args.includes("--verbose");
  const url = gitFs.getUrl();

  if (verbose) {
    return {
      stdout: `origin\t${url} (fetch)\norigin\t${url} (push)\n`,
      stderr: "",
      exitCode: 0
    };
  }

  return { stdout: "origin\n", stderr: "", exitCode: 0 };
}
