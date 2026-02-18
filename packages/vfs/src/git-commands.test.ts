import { describe, it, expect, beforeEach } from "vitest";
import { Bash, InMemoryFs, MountableFs } from "just-bash";
import { createMockGitServer } from "./mock-git-server";
import { createMountCommands } from "./commands";
import { createGitCommands } from "./git-commands";
import { MockR2Bucket } from "./mock-r2-bucket";
import { GitFs } from "./git-fs";

/** Helper: create test env with mock git server + MountableFs + Bash */
async function createGitTestEnv(files: Record<string, string>) {
  const { url, http } = await createMockGitServer(files);
  const r2Bucket = new MockR2Bucket() as unknown as R2Bucket;

  const inMemoryFs = new InMemoryFs();
  inMemoryFs.mkdirSync("/mnt");
  const mountableFs = new MountableFs({ base: inMemoryFs });

  const mountCmds = createMountCommands(mountableFs, undefined, {
    gitHttp: http,
    r2Bucket,
    userId: "test-user"
  });
  const gitCmds = createGitCommands(mountableFs);

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

describe("git status", () => {
  let bash: Bash;

  beforeEach(async () => {
    ({ bash } = await createGitTestEnv({
      "README.md": "Hello",
      "src/index.ts": "export default 42;"
    }));
  });

  it("shows nothing to commit on clean tree", async () => {
    const result = await bash.exec("git status");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("nothing to commit");
  });

  it("shows new file after writing", async () => {
    await bash.exec("echo hello > /mnt/repo/newfile.txt");
    const result = await bash.exec("git status");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("new file:");
    expect(result.stdout).toContain("newfile.txt");
  });

  it("shows modified after editing existing file", async () => {
    await bash.exec("echo updated > /mnt/repo/README.md");
    const result = await bash.exec("git status");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("modified:");
    expect(result.stdout).toContain("README.md");
  });

  it("shows deleted after removing file", async () => {
    await bash.exec("rm /mnt/repo/README.md");
    const result = await bash.exec("git status");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("deleted:");
    expect(result.stdout).toContain("README.md");
  });
});

describe("git commit", () => {
  let bash: Bash;

  beforeEach(async () => {
    ({ bash } = await createGitTestEnv({
      "README.md": "Hello"
    }));
  });

  it("commits with message and shows branch + oid", async () => {
    await bash.exec("echo new > /mnt/repo/file.txt");
    const result = await bash.exec('git commit -m "add file"');
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("[main ");
    expect(result.stdout).toContain("add file");
  });

  it("shows nothing to commit after clean commit", async () => {
    await bash.exec("echo new > /mnt/repo/file.txt");
    await bash.exec('git commit -m "add file"');
    const status = await bash.exec("git status");
    expect(status.stdout).toContain("nothing to commit");
  });

  it("errors without -m flag", async () => {
    await bash.exec("echo new > /mnt/repo/file.txt");
    const result = await bash.exec("git commit");
    expect(result.exitCode).toBe(1);
    expect(result.stderr).toContain("requires a value");
  });

  it("errors with no changes", async () => {
    const result = await bash.exec('git commit -m "empty"');
    expect(result.exitCode).toBe(1);
    expect(result.stderr).toContain("nothing to commit");
  });

  it("shows ahead of remote after commit before push", async () => {
    await bash.exec("echo new > /mnt/repo/file.txt");
    await bash.exec('git commit -m "add file"');
    const status = await bash.exec("git status");
    expect(status.stdout).toContain("ahead");
  });
});

describe("git push", () => {
  let bash: Bash;

  beforeEach(async () => {
    ({ bash } = await createGitTestEnv({
      "README.md": "Hello"
    }));
  });

  it("pushes committed changes", async () => {
    await bash.exec("echo new > /mnt/repo/file.txt");
    await bash.exec('git commit -m "add file"');
    const result = await bash.exec("git push");
    expect(result.exitCode).toBe(0);
    // After push, status should not show ahead
    const status = await bash.exec("git status");
    expect(status.stdout).not.toContain("ahead");
  });

  it("shows up-to-date when nothing to push", async () => {
    const result = await bash.exec("git push");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("Everything up-to-date");
  });
});

describe("git - error cases", () => {
  it("shows fatal error outside git mount", async () => {
    const inMemoryFs = new InMemoryFs();
    const mountableFs = new MountableFs({ base: inMemoryFs });
    const gitCmds = createGitCommands(mountableFs);
    const bash = new Bash({
      fs: mountableFs,
      customCommands: gitCmds,
      cwd: "/home/user"
    });

    const result = await bash.exec("git status");
    expect(result.exitCode).toBe(128);
    expect(result.stderr).toContain("not a git repository");
  });

  it("works from subdirectory of mount", async () => {
    const { bash } = await createGitTestEnv({
      "README.md": "Hello",
      "src/index.ts": "export default 42;"
    });

    // cd into subdirectory and run git status
    const result = await bash.exec("cd /mnt/repo/src && git status");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("nothing to commit");
  });

  it("shows usage for unknown subcommand", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git foo");
    expect(result.exitCode).toBe(1);
    expect(result.stderr).toContain("not a git command");
  });

  it("shows usage with no arguments", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git");
    expect(result.exitCode).toBe(1);
    expect(result.stderr).toContain("usage");
  });
});

describe("git R2 persistence (hibernation survival)", () => {
  it("overlay files survive GitFs recreation (simulates hibernation)", async () => {
    const { url, http } = await createMockGitServer({
      "README.md": "Hello"
    });
    const r2Bucket = new MockR2Bucket() as unknown as R2Bucket;

    // First GitFs instance — clone and write files
    const gitFs1 = new GitFs({
      url,
      http,
      r2Bucket,
      userId: "test-user",
      mountPoint: "/mnt/repo"
    });
    await gitFs1.init();
    await gitFs1.writeFile("/newfile.txt", "hello from first instance");

    // Verify dirty state
    expect(await gitFs1.isDirty()).toBe(true);

    // Second GitFs instance — same R2 bucket (simulates DO restart)
    const gitFs2 = new GitFs({
      url,
      http,
      r2Bucket,
      userId: "test-user",
      mountPoint: "/mnt/repo"
    });
    await gitFs2.init();

    // State should be restored from R2
    const content = await gitFs2.readFile("/newfile.txt", { encoding: "utf8" });
    expect(content).toBe("hello from first instance");
    expect(await gitFs2.isDirty()).toBe(true);

    const status = await gitFs2.getStatus();
    expect(status.added).toContain("/newfile.txt");
  });

  it("committed state survives GitFs recreation", async () => {
    const { url, http } = await createMockGitServer({
      "README.md": "Hello"
    });
    const r2Bucket = new MockR2Bucket() as unknown as R2Bucket;

    // First instance — clone, write, commit
    const gitFs1 = new GitFs({
      url,
      http,
      r2Bucket,
      userId: "test-user",
      mountPoint: "/mnt/repo"
    });
    await gitFs1.init();
    await gitFs1.writeFile("/file.txt", "committed content");
    await gitFs1.commit("test commit", { name: "Test", email: "t@t.com" });
    expect(await gitFs1.isDirty()).toBe(false);

    // Second instance — should restore committed state from R2
    const gitFs2 = new GitFs({
      url,
      http,
      r2Bucket,
      userId: "test-user",
      mountPoint: "/mnt/repo"
    });
    await gitFs2.init();

    // Should be clean (overlay was cleared after commit)
    expect(await gitFs2.isDirty()).toBe(false);

    // But the committed file should be readable from the git tree
    const content = await gitFs2.readFile("/file.txt", { encoding: "utf8" });
    expect(content).toBe("committed content");

    // Should show unpushed commits
    expect(gitFs2.hasUnpushedCommits()).toBe(true);
  });

  it("deleted files survive GitFs recreation", async () => {
    const { url, http } = await createMockGitServer({
      "README.md": "Hello",
      "toDelete.txt": "bye"
    });
    const r2Bucket = new MockR2Bucket() as unknown as R2Bucket;

    // First instance — delete a file
    const gitFs1 = new GitFs({
      url,
      http,
      r2Bucket,
      userId: "test-user",
      mountPoint: "/mnt/repo"
    });
    await gitFs1.init();
    await gitFs1.rm("/toDelete.txt");

    // Second instance — deleted state should persist
    const gitFs2 = new GitFs({
      url,
      http,
      r2Bucket,
      userId: "test-user",
      mountPoint: "/mnt/repo"
    });
    await gitFs2.init();

    expect(await gitFs2.exists("/toDelete.txt")).toBe(false);
    expect(await gitFs2.isDirty()).toBe(true);

    const status = await gitFs2.getStatus();
    expect(status.deleted).toContain("/toDelete.txt");
  });
});
