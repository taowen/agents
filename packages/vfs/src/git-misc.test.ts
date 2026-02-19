import { describe, it, expect } from "vitest";
import { Bash, InMemoryFs, MountableFs } from "just-bash";
import { createGitCommands } from "./git-commands/index";
import { createMountCommands } from "./commands";
import {
  createGitTestEnv,
  createMockGitServer,
  MockR2Bucket
} from "./git-test-helpers";
import type { MountOptions } from "./mount";

describe("git version and help", () => {
  it("shows version with --version", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git --version");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("git version");
    expect(result.stdout).toContain("vfs");
  });

  it("shows version with version subcommand", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git version");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("git version");
  });

  it("shows help with --help", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git --help");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("usage:");
    expect(result.stdout).toContain("status");
    expect(result.stdout).toContain("commit");
  });

  it("shows help with help subcommand", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git help");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("usage:");
  });

  it("--version works outside git repo", async () => {
    const inMemoryFs = new InMemoryFs();
    const mountableFs = new MountableFs({ base: inMemoryFs });
    const gitCmds = createGitCommands(mountableFs);
    const bash = new Bash({
      fs: mountableFs,
      customCommands: gitCmds,
      cwd: "/home/user"
    });
    const result = await bash.exec("git --version");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("git version");
  });

  it("--help works outside git repo", async () => {
    const inMemoryFs = new InMemoryFs();
    const mountableFs = new MountableFs({ base: inMemoryFs });
    const gitCmds = createGitCommands(mountableFs);
    const bash = new Bash({
      fs: mountableFs,
      customCommands: gitCmds,
      cwd: "/home/user"
    });
    const result = await bash.exec("git --help");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("usage:");
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

describe("git -C", () => {
  it("runs status via -C from outside the repo", async () => {
    const inMemoryFs = new InMemoryFs();
    inMemoryFs.mkdirSync("/home");
    inMemoryFs.mkdirSync("/home/user");
    const mountableFs = new MountableFs({ base: inMemoryFs });

    const { url, http } = await createMockGitServer({ "README.md": "Hello" });
    const r2Bucket = new MockR2Bucket() as unknown as R2Bucket;
    const mountCmds = createMountCommands(mountableFs, {
      gitHttp: http,
      r2Bucket,
      userId: "test-user"
    });
    const gitCmds = createGitCommands(mountableFs);
    const bash2 = new Bash({
      fs: mountableFs,
      customCommands: [...mountCmds, ...gitCmds],
      cwd: "/home/user"
    });
    await bash2.exec(`mount -t git ${url} /mnt/repo`);

    const result = await bash2.exec("git -C /mnt/repo status");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("nothing to commit");
  });

  it("runs log via -C", async () => {
    const { url, http } = await createMockGitServer([
      { files: { "README.md": "v1" }, message: "first commit" },
      { files: { "README.md": "v2" }, message: "second commit" }
    ]);
    const r2Bucket = new MockR2Bucket() as unknown as R2Bucket;
    const inMemoryFs = new InMemoryFs();
    inMemoryFs.mkdirSync("/home");
    inMemoryFs.mkdirSync("/home/user");
    const mountableFs = new MountableFs({ base: inMemoryFs });
    const mountCmds = createMountCommands(mountableFs, {
      gitHttp: http,
      r2Bucket,
      userId: "test-user"
    });
    const gitCmds = createGitCommands(mountableFs);
    const bash2 = new Bash({
      fs: mountableFs,
      customCommands: [...mountCmds, ...gitCmds],
      cwd: "/home/user"
    });
    await bash2.exec(`mount -t git ${url} /mnt/repo`);

    const result = await bash2.exec("git -C /mnt/repo log --oneline");
    expect(result.exitCode).toBe(0);
    const lines = result.stdout.trim().split("\n");
    expect(lines.length).toBe(2);
    expect(lines[0]).toContain("second commit");
    expect(lines[1]).toContain("first commit");
  });

  it("errors when -C is missing its path argument", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git -C");
    expect(result.exitCode).toBe(129);
    expect(result.stderr).toContain("requires a value");
  });
});
