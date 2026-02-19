import type { MountableFs } from "just-bash";
import type { GitMount, GitResult } from "./helpers";
import { parseGitCredentials, findCredential } from "../git-credentials";

export async function gitPush(
  match: GitMount,
  mountableFs: MountableFs
): Promise<GitResult> {
  const { gitFs } = match;

  if (!gitFs.hasUnpushedCommits()) {
    return {
      stdout: "Everything up-to-date\n",
      stderr: "",
      exitCode: 0
    };
  }

  // Try to read credentials from /etc/git-credentials
  let onAuth: (() => { username: string; password?: string }) | undefined;
  try {
    const cred = await mountableFs.readFile("/etc/git-credentials", {
      encoding: "utf8"
    });
    const credential = findCredential(
      parseGitCredentials(cred as string),
      gitFs.getUrl()
    );
    if (credential) {
      onAuth = () => ({
        username: credential.username,
        password: credential.password
      });
    }
  } catch {
    // no credentials file
  }

  try {
    await gitFs.push(onAuth);
    return { stdout: "", stderr: "", exitCode: 0 };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return { stdout: "", stderr: `error: push failed: ${msg}\n`, exitCode: 1 };
  }
}
