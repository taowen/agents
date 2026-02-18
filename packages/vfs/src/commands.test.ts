import { describe, it, expect, beforeEach } from "vitest";
import { AgentFS } from "agentfs-sdk";
import { Bash, InMemoryFs, MountableFs } from "just-bash";
import { AgentFsAdapter } from "./agentfs-adapter";
import { createMountCommands } from "./commands";
import { createMockGitServer } from "./mock-git-server";
import { MockR2Bucket } from "./mock-r2-bucket";

/** Helper: create a fresh AgentFS + MountableFs + Bash for each test. */
async function createTestEnv(opts?: { withAgentFs?: boolean }) {
  const agent = await AgentFS.open({ path: ":memory:" });
  const agentFs = agent.fs as any;

  const inMemoryFs = new InMemoryFs();
  inMemoryFs.mkdirSync("/mnt");
  const mountableFs = new MountableFs({ base: inMemoryFs });

  // /etc is always hardcoded-mounted (same as server.ts)
  const etcFs = new AgentFsAdapter(agentFs, "/etc");
  mountableFs.mount("/etc", etcFs, "agentfs");

  const cmds = createMountCommands(
    mountableFs,
    opts?.withAgentFs !== false ? agentFs : undefined
  );

  const bash = new Bash({
    fs: mountableFs,
    customCommands: cmds,
    cwd: "/home/user"
  });

  return { agentFs, mountableFs, bash };
}

describe("mount command — list mounts", () => {
  let bash: Bash;

  beforeEach(async () => {
    ({ bash } = await createTestEnv());
  });

  it("mount (no args) lists existing mounts with type", async () => {
    const result = await bash.exec("mount");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("agentfs on /etc");
  });
});

describe("mount -t agentfs", () => {
  let agentFs: any;
  let bash: Bash;

  beforeEach(async () => {
    ({ agentFs, bash } = await createTestEnv());
  });

  it("mounts agentfs and files are readable/writable", async () => {
    const mount = await bash.exec("mount -t agentfs none /mnt/data");
    expect(mount.exitCode).toBe(0);

    const write = await bash.exec(
      "echo hello > /mnt/data/test.txt && cat /mnt/data/test.txt"
    );
    expect(write.exitCode).toBe(0);
    expect(write.stdout.trim()).toBe("hello");
  });

  it("errors when agentFs not provided", async () => {
    const { bash: noAgentBash } = await createTestEnv({ withAgentFs: false });
    const result = await noAgentBash.exec("mount -t agentfs none /mnt/data");
    expect(result.exitCode).toBe(1);
    expect(result.stderr).toContain("agentfs not available");
  });
});

describe("mount -t git", () => {
  let bash: Bash;
  let mountableFs: MountableFs;

  beforeEach(async () => {
    ({ bash, mountableFs } = await createTestEnv());
  });

  it("mounts git repo with mock server, files readable", async () => {
    const { url, http } = await createMockGitServer({
      "README.md": "Hello from git",
      "src/index.ts": "export default 42;"
    });

    const r2Bucket = new MockR2Bucket() as unknown as R2Bucket;
    const cmds = createMountCommands(mountableFs, undefined, {
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

  beforeEach(async () => {
    ({ bash } = await createTestEnv());
  });

  it("unmounts a previously mounted path", async () => {
    // Mount first
    const mount = await bash.exec("mount -t agentfs none /mnt/data");
    expect(mount.exitCode).toBe(0);

    // Verify it shows in mount list
    const list1 = await bash.exec("mount");
    expect(list1.stdout).toContain("agentfs on /mnt/data");

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
