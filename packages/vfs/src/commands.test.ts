import { describe, it, expect, beforeEach } from "vitest";
import { Bash, InMemoryFs, MountableFs } from "just-bash";
import { createMountCommands } from "./commands";
import { createMockGitServer } from "./mock-git-server";
import { MockR2Bucket } from "./mock-r2-bucket";

/** Helper: create a fresh InMemoryFs + MountableFs + Bash for each test. */
function createTestEnv() {
  const inMemoryFs = new InMemoryFs();
  inMemoryFs.mkdirSync("/mnt");
  const mountableFs = new MountableFs({ base: inMemoryFs });

  const etcFs = new InMemoryFs();
  mountableFs.mount("/etc", etcFs, "d1");

  const fsTypeRegistry = { d1: () => new InMemoryFs() };
  const cmds = createMountCommands(mountableFs, { fsTypeRegistry });

  const bash = new Bash({
    fs: mountableFs,
    customCommands: cmds,
    cwd: "/home/user"
  });

  return { mountableFs, bash };
}

describe("mount command — list mounts", () => {
  let bash: Bash;

  beforeEach(() => {
    ({ bash } = createTestEnv());
  });

  it("mount (no args) lists existing mounts with type", async () => {
    const result = await bash.exec("mount");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("d1 on /etc");
  });
});

describe("mount -t git", () => {
  let bash: Bash;
  let mountableFs: MountableFs;

  beforeEach(() => {
    ({ bash, mountableFs } = createTestEnv());
  });

  it("mounts git repo with mock server, files readable", async () => {
    const { url, http } = await createMockGitServer({
      "README.md": "Hello from git",
      "src/index.ts": "export default 42;"
    });

    const r2Bucket = new MockR2Bucket() as unknown as R2Bucket;
    const cmds = createMountCommands(mountableFs, {
      gitHttp: http,
      r2Bucket,
      userId: "test-user"
    });
    const gitBash = new Bash({
      fs: mountableFs,
      customCommands: cmds,
      cwd: "/home/user"
    });

    const mount = await gitBash.exec(`mount -t git ${url} /mnt/repo`);
    expect(mount.exitCode).toBe(0);

    const cat = await gitBash.exec("cat /mnt/repo/README.md");
    expect(cat.exitCode).toBe(0);
    expect(cat.stdout.trim()).toBe("Hello from git");
  });
});

describe("umount", () => {
  let bash: Bash;

  beforeEach(() => {
    ({ bash } = createTestEnv());
  });

  it("unmounts a previously mounted path", async () => {
    // Mount first via registry
    const mount = await bash.exec("mount -t d1 none /mnt/data");
    expect(mount.exitCode).toBe(0);

    // Verify it shows in mount list
    const list1 = await bash.exec("mount");
    expect(list1.stdout).toContain("d1 on /mnt/data");

    // Unmount
    const umount = await bash.exec("umount /mnt/data");
    expect(umount.exitCode).toBe(0);

    // Verify removed from mount list
    const list2 = await bash.exec("mount");
    expect(list2.stdout).not.toContain("/mnt/data");
  });

  it("umount (no args) shows usage error", async () => {
    const result = await bash.exec("umount");
    expect(result.exitCode).toBe(1);
    expect(result.stderr).toContain("usage");
  });
});
