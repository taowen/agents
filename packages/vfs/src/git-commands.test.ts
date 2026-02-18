import { describe, it, expect, beforeEach } from "vitest";
import { Bash, InMemoryFs, MountableFs } from "just-bash";
import { createMockGitServer } from "./mock-git-server";
import { createMountCommands } from "./commands";
import { createGitCommands } from "./git-commands";

/** Helper: create test env with mock git server + MountableFs + Bash */
async function createGitTestEnv(files: Record<string, string>) {
  const { url, http } = await createMockGitServer(files);

  const inMemoryFs = new InMemoryFs();
  inMemoryFs.mkdirSync("/mnt");
  const mountableFs = new MountableFs({ base: inMemoryFs });

  const mountCmds = createMountCommands(mountableFs, undefined, {
    gitHttp: http
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

  return { bash, mountableFs };
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
