/**
 * sync — auto-commit and push dirty GitFs mounts.
 *
 * After each bash command, call syncDirtyGitMounts() to find GitFs mounts
 * with overlay changes, then commit and push them to the remote.
 */

import { GitFs } from "./git-fs";
import { parseGitCredentials, findCredential } from "./git-credentials";
import type { MountableFs } from "just-bash";

/**
 * Find all dirty GitFs mounts and commit+push their overlay changes.
 *
 * @param mountableFs - The MountableFs to scan for dirty GitFs mounts
 * @param commitMessage - Commit message (defaults to "Update <mountPoint>")
 */
export async function syncDirtyGitMounts(
  mountableFs: MountableFs,
  commitMessage?: string
): Promise<void> {
  for (const { mountPoint, filesystem } of mountableFs.getMounts()) {
    if (!(filesystem instanceof GitFs) || !filesystem.isDirty()) continue;

    // Try to read credentials from /etc/git-credentials
    let onAuth: (() => { username: string; password?: string }) | undefined;
    try {
      const cred = await mountableFs.readFile("/etc/git-credentials", {
        encoding: "utf8"
      });
      const match = findCredential(
        parseGitCredentials(cred as string),
        filesystem.getUrl()
      );
      if (match) {
        onAuth = () => ({ username: match.username, password: match.password });
      }
    } catch {
      // no credentials file — will use onAuth from mount time
    }

    await filesystem.commitAndPush(
      commitMessage || `Update ${mountPoint}`,
      { name: "AI Assistant", email: "ai@assistant.local" },
      onAuth
    );
  }
}
