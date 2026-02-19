import type { GitResult, GitMount } from "./helpers";
import { shortBranchName } from "./helpers";

export async function gitBranch(match: GitMount): Promise<GitResult> {
  const { gitFs } = match;
  const branch = shortBranchName(gitFs.getRef() || "main");
  return { stdout: `* ${branch}\n`, stderr: "", exitCode: 0 };
}
