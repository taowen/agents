import type { MountableFs } from "just-bash";
import type { GitMount, GitResult } from "./helpers";
import { parseGitCredentials, findCredential } from "../git-credentials";

export async function gitPull(
  match: GitMount,
  mountableFs: MountableFs
): Promise<GitResult> {
  const { gitFs } = match;

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
    const result = await gitFs.pull(onAuth);

    if (!result.updated) {
      return {
        stdout: "Already up to date.\n",
        stderr: "",
        exitCode: 0
      };
    }

    const fromShort = result.fromOid.slice(0, 7);
    const toShort = result.toOid.slice(0, 7);
    return {
      stdout: `Updating ${fromShort}..${toShort}\nFast-forward\n`,
      stderr: "",
      exitCode: 0
    };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return {
      stdout: "",
      stderr: `error: ${msg}\n`,
      exitCode: 1
    };
  }
}
