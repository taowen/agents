import { Bash, InMemoryFs, MountableFs } from "just-bash";
import { createMockGitServer, type MockCommit } from "./mock-git-server";
import { createMountCommands } from "./commands";
import { createGitCommands } from "./git-commands/index";
import { MockR2Bucket } from "./mock-r2-bucket";
import { GitFs } from "./git-fs";
import type { MountOptions } from "./mount";

export { MockR2Bucket } from "./mock-r2-bucket";
export { GitFs } from "./git-fs";
export { createMockGitServer, type MockCommit } from "./mock-git-server";
export type { MountOptions } from "./mount";

/** Helper: create test env with mock git server + MountableFs + Bash, repo auto-mounted at /mnt/repo */
export async function createGitTestEnv(
  files: Record<string, string> | MockCommit[]
) {
  const { url, http } = await createMockGitServer(files);
  const r2Bucket = new MockR2Bucket() as unknown as R2Bucket;

  const inMemoryFs = new InMemoryFs();
  inMemoryFs.mkdirSync("/mnt");
  const mountableFs = new MountableFs({ base: inMemoryFs });

  const mountOptions: MountOptions = {
    gitHttp: http,
    r2Bucket,
    userId: "test-user"
  };
  const mountCmds = createMountCommands(mountableFs, undefined, mountOptions);
  const gitCmds = createGitCommands(mountableFs, mountOptions);

  const bash = new Bash({
    fs: mountableFs,
    customCommands: [...mountCmds, ...gitCmds],
    cwd: "/mnt/repo"
  });

  // Mount the repo
  const mount = await bash.exec(`mount -t git ${url} /mnt/repo`);
  if (mount.exitCode !== 0) {
    throw new Error(`mount failed: ${mount.stderr}`);
  }

  return { bash, mountableFs, r2Bucket, url, http };
}

/** Helper: create test env without auto-mounting (for clone/pull tests) */
export async function createCloneTestEnv(
  files: Record<string, string> | MockCommit[]
) {
  const { url, http } = await createMockGitServer(files);
  const r2Bucket = new MockR2Bucket() as unknown as R2Bucket;

  const inMemoryFs = new InMemoryFs();
  inMemoryFs.mkdirSync("/mnt");
  const mountableFs = new MountableFs({ base: inMemoryFs });

  const mountOptions: MountOptions = {
    gitHttp: http,
    r2Bucket,
    userId: "test-user"
  };
  const mountCmds = createMountCommands(mountableFs, undefined, mountOptions);
  const gitCmds = createGitCommands(mountableFs, mountOptions);

  const bash = new Bash({
    fs: mountableFs,
    customCommands: [...mountCmds, ...gitCmds],
    cwd: "/home/user"
  });

  return { bash, mountableFs, r2Bucket, url, http };
}
