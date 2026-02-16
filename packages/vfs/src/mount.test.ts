import { describe, it, expect, beforeEach } from "vitest";
import { AgentFS } from "agentfs-sdk";
import { Bash, InMemoryFs, MountableFs } from "just-bash";
import { AgentFsAdapter } from "./agentfs-adapter";
import { mountFstabEntries } from "./mount";
import { createMockGitServer } from "./mock-git-server";

/** Helper: create a fresh AgentFS + MountableFs + Bash for each test. */
async function createTestEnv() {
  const agent = await AgentFS.open({ path: ":memory:" });
  const agentFs = agent.fs as any; // Node AgentFS.fs is duck-type compatible

  const inMemoryFs = new InMemoryFs();
  inMemoryFs.mkdirSync("/mnt");
  const mountableFs = new MountableFs({ base: inMemoryFs });

  // /etc is always hardcoded-mounted (same as server.ts)
  const etcFs = new AgentFsAdapter(agentFs, "/etc");
  mountableFs.mount("/etc", etcFs);

  const bash = new Bash({ fs: mountableFs, cwd: "/home/user" });

  return { agentFs, mountableFs, bash };
}

describe("mount — fstab integration", () => {
  let agentFs: any;
  let mountableFs: MountableFs;
  let bash: Bash;

  beforeEach(async () => {
    ({ agentFs, mountableFs, bash } = await createTestEnv());
  });

  it("default fstab: cat /etc/fstab contains /home/user agentfs", async () => {
    await mountFstabEntries(agentFs, mountableFs);

    const result = await bash.exec("cat /etc/fstab");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("/home/user");
    expect(result.stdout).toContain("agentfs");
  });

  it("default fstab: /home/user is accessible", async () => {
    await mountFstabEntries(agentFs, mountableFs);

    const result = await bash.exec("ls /home/user");
    expect(result.exitCode).toBe(0);
  });

  it("default fstab: files in /home/user are read/writable", async () => {
    await mountFstabEntries(agentFs, mountableFs);

    const write = await bash.exec(
      "echo hello > /home/user/test.txt && cat /home/user/test.txt"
    );
    expect(write.exitCode).toBe(0);
    expect(write.stdout.trim()).toBe("hello");
  });

  it("persistence: written files are readable via AgentFS directly", async () => {
    await mountFstabEntries(agentFs, mountableFs);

    await bash.exec("echo myhost > /etc/hostname");

    // Read back via raw AgentFS
    const data = await agentFs.readFile("/etc/hostname");
    const content =
      typeof data === "string" ? data : new TextDecoder().decode(data);
    expect(content).toContain("myhost");
  });

  it("persistence: /home/user files stored in AgentFS", async () => {
    await mountFstabEntries(agentFs, mountableFs);

    await bash.exec("echo hello > /home/user/test.txt");

    const data = await agentFs.readFile("/home/user/test.txt");
    const content =
      typeof data === "string" ? data : new TextDecoder().decode(data);
    expect(content).toContain("hello");
  });

  it("custom fstab: agentfs entry mounts /data", async () => {
    // Pre-write a custom fstab
    await agentFs.mkdir("/etc");
    await agentFs.writeFile(
      "/etc/fstab",
      Buffer.from("none  /data  agentfs  defaults  0  0\n")
    );

    await mountFstabEntries(agentFs, mountableFs);

    const result = await bash.exec("echo 42 > /data/value && cat /data/value");
    expect(result.exitCode).toBe(0);
    expect(result.stdout.trim()).toBe("42");
  });

  it("git fstab entry: mounts repo with mock server (no network)", async () => {
    const { url, http } = await createMockGitServer({
      "README.md": "Hello",
      "src/index.ts": "export default 42;"
    });

    // Pre-write fstab with git entry
    await agentFs.mkdir("/etc");
    await agentFs.writeFile(
      "/etc/fstab",
      Buffer.from(`${url}  /mnt/repo  git  ref=main,depth=1  0  0\n`)
    );

    await mountFstabEntries(agentFs, mountableFs, { gitHttp: http });

    const ls = await bash.exec("ls /mnt/repo");
    expect(ls.exitCode).toBe(0);
    expect(ls.stdout).toContain("README.md");

    const cat = await bash.exec("cat /mnt/repo/README.md");
    expect(cat.exitCode).toBe(0);
    expect(cat.stdout.trim()).toBe("Hello");
  });

  it("git clone failure does not break other mounts", async () => {
    // fstab: invalid git URL + valid agentfs
    await agentFs.mkdir("/etc");
    await agentFs.writeFile(
      "/etc/fstab",
      Buffer.from(
        "https://invalid.example.com/nonexistent  /mnt/bad  git  ref=main  0  0\n" +
          "none  /data  agentfs  defaults  0  0\n"
      )
    );

    // Should not throw
    await mountFstabEntries(agentFs, mountableFs);

    // agentfs mount should still work
    const result = await bash.exec("echo ok > /data/check && cat /data/check");
    expect(result.exitCode).toBe(0);
    expect(result.stdout.trim()).toBe("ok");
  });
});
