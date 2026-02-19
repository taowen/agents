import { describe, it, expect, beforeEach } from "vitest";
import { Bash, InMemoryFs, MountableFs } from "just-bash";
import { createMockGitServer, type MockCommit } from "./mock-git-server";
import { createMountCommands } from "./commands";
import { createGitCommands } from "./git-commands/index";
import { MockR2Bucket } from "./mock-r2-bucket";
import { GitFs } from "./git-fs";
import type { MountOptions } from "./mount";

/** Helper: create test env with mock git server + MountableFs + Bash */
async function createGitTestEnv(files: Record<string, string> | MockCommit[]) {
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

  it("shows short status with -s flag", async () => {
    await bash.exec("echo hello > /mnt/repo/newfile.txt");
    await bash.exec("echo updated > /mnt/repo/README.md");
    const result = await bash.exec("git status -s");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("A  newfile.txt");
    expect(result.stdout).toContain("M  README.md");
  });

  it("shows short status with --short flag", async () => {
    await bash.exec("rm /mnt/repo/README.md");
    const result = await bash.exec("git status --short");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("D  README.md");
  });

  it("shows empty output for short status on clean tree", async () => {
    const result = await bash.exec("git status -s");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toBe("");
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

  it("uses --author flag for commit", async () => {
    await bash.exec("echo new > /mnt/repo/file.txt");
    const result = await bash.exec(
      'git commit -m "authored commit" --author="John Doe <john@example.com>"'
    );
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("authored commit");
    // Verify author via git log
    const log = await bash.exec("git log -1");
    expect(log.stdout).toContain("John Doe");
    expect(log.stdout).toContain("john@example.com");
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

  it("tolerates extra arguments (git push origin main)", async () => {
    const result = await bash.exec("git push origin main");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("Everything up-to-date");
  });
});

describe("git log", () => {
  let bash: Bash;

  beforeEach(async () => {
    ({ bash } = await createGitTestEnv({
      "README.md": "Hello"
    }));
  });

  it("shows initial commit log", async () => {
    const result = await bash.exec("git log");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("commit ");
    expect(result.stdout).toContain("Author:");
    expect(result.stdout).toContain("Date:");
    expect(result.stdout).toContain("Initial commit");
  });

  it("shows oneline format", async () => {
    const result = await bash.exec("git log --oneline");
    expect(result.exitCode).toBe(0);
    // Should be short oid + message
    const lines = result.stdout.trim().split("\n");
    expect(lines.length).toBeGreaterThanOrEqual(1);
    expect(lines[0]).toMatch(/^[a-f0-9]{7} /);
    expect(lines[0]).toContain("Initial commit");
  });

  it("limits entries with -n flag", async () => {
    await bash.exec("echo a > /mnt/repo/a.txt");
    await bash.exec('git commit -m "first"');
    await bash.exec("echo b > /mnt/repo/b.txt");
    await bash.exec('git commit -m "second"');
    const result = await bash.exec("git log --oneline -n 1");
    expect(result.exitCode).toBe(0);
    const lines = result.stdout.trim().split("\n");
    expect(lines.length).toBe(1);
    expect(lines[0]).toContain("second");
  });

  it("limits entries with -N shorthand", async () => {
    await bash.exec("echo a > /mnt/repo/a.txt");
    await bash.exec('git commit -m "first"');
    await bash.exec("echo b > /mnt/repo/b.txt");
    await bash.exec('git commit -m "second"');
    const result = await bash.exec("git log --oneline -1");
    expect(result.exitCode).toBe(0);
    const lines = result.stdout.trim().split("\n");
    expect(lines.length).toBe(1);
  });
});

describe("git diff", () => {
  let bash: Bash;

  beforeEach(async () => {
    ({ bash } = await createGitTestEnv({
      "README.md": "Hello",
      "src/index.ts": "export default 42;"
    }));
  });

  it("shows empty diff on clean tree", async () => {
    const result = await bash.exec("git diff");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toBe("");
  });

  it("shows diff headers for changed files", async () => {
    await bash.exec("echo updated > /mnt/repo/README.md");
    const result = await bash.exec("git diff");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("diff --git a/README.md b/README.md");
  });

  it("shows --name-only output", async () => {
    await bash.exec("echo updated > /mnt/repo/README.md");
    await bash.exec("echo new > /mnt/repo/new.txt");
    const result = await bash.exec("git diff --name-only");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("new.txt");
    expect(result.stdout).toContain("README.md");
  });

  it("shows --stat output", async () => {
    await bash.exec("echo updated > /mnt/repo/README.md");
    const result = await bash.exec("git diff --stat");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("README.md");
    expect(result.stdout).toContain("1 file changed");
  });

  it("shows new file and deleted file in diff", async () => {
    await bash.exec("echo new > /mnt/repo/new.txt");
    await bash.exec("rm /mnt/repo/README.md");
    const result = await bash.exec("git diff");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("new file");
    expect(result.stdout).toContain("deleted file");
  });
});

describe("git branch", () => {
  it("shows current branch", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git branch");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("* main");
  });
});

describe("git remote", () => {
  it("shows origin", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git remote");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toBe("origin\n");
  });

  it("shows verbose remote info", async () => {
    const { bash, url } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git remote -v");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("origin");
    expect(result.stdout).toContain(url);
    expect(result.stdout).toContain("(fetch)");
    expect(result.stdout).toContain("(push)");
  });
});

describe("git show", () => {
  it("shows HEAD commit info", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git show");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("commit ");
    expect(result.stdout).toContain("Author:");
    expect(result.stdout).toContain("Initial commit");
  });

  it("shows HEAD commit with --stat", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    // Make a change and commit so --stat has something to show
    await bash.exec("echo new > /mnt/repo/file.txt");
    await bash.exec('git commit -m "add file"');
    const result = await bash.exec("git show --stat");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("commit ");
    expect(result.stdout).toContain("add file");
  });
});

describe("git rev-parse", () => {
  it("shows full HEAD oid", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git rev-parse HEAD");
    expect(result.exitCode).toBe(0);
    expect(result.stdout.trim()).toMatch(/^[a-f0-9]{40}$/);
  });

  it("shows short HEAD oid with --short", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git rev-parse --short HEAD");
    expect(result.exitCode).toBe(0);
    expect(result.stdout.trim()).toMatch(/^[a-f0-9]{7}$/);
  });

  it("errors with no arguments", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    const result = await bash.exec("git rev-parse");
    expect(result.exitCode).toBe(128);
    expect(result.stderr).toContain("fatal: bad revision");
  });
});

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

describe("git log - multi commit", () => {
  it("shows all commits from a multi-commit repo", async () => {
    const { bash } = await createGitTestEnv([
      { files: { "README.md": "v1" }, message: "first commit" },
      { files: { "README.md": "v2" }, message: "second commit" },
      { files: { "README.md": "v3" }, message: "third commit" }
    ]);
    const result = await bash.exec("git log --oneline");
    expect(result.exitCode).toBe(0);
    const lines = result.stdout.trim().split("\n");
    expect(lines.length).toBe(3);
    expect(lines[0]).toContain("third commit");
    expect(lines[1]).toContain("second commit");
    expect(lines[2]).toContain("first commit");
  });
});

describe("git -C", () => {
  it("runs status via -C from outside the repo", async () => {
    const { bash } = await createGitTestEnv({ "README.md": "Hello" });
    // cwd is /mnt/repo by default; create a bash instance at /home/user
    const inMemoryFs = new InMemoryFs();
    inMemoryFs.mkdirSync("/home");
    inMemoryFs.mkdirSync("/home/user");
    const mountableFs = new MountableFs({ base: inMemoryFs });

    // Re-use the same mock server by mounting manually
    const { url, http } = await createMockGitServer({ "README.md": "Hello" });
    const r2Bucket = new MockR2Bucket() as unknown as R2Bucket;
    const mountCmds = createMountCommands(mountableFs, undefined, {
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
    const mountCmds = createMountCommands(mountableFs, undefined, {
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

describe("git clone", () => {
  it("clones with auto-inferred directory from URL", async () => {
    const { url, http } = await createMockGitServer({
      "README.md": "Hello",
      "src/index.ts": "export default 42;"
    });
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

    // The mock URL is http://localhost/__mock_git_repo__
    const result = await bash.exec(`git clone ${url}`);
    expect(result.exitCode).toBe(0);
    expect(result.stderr).toContain("Cloning into");
    expect(result.stderr).toContain("__mock_git_repo__");

    // Should be able to list files in the auto-inferred directory
    const ls = await bash.exec("ls /mnt/__mock_git_repo__");
    expect(ls.exitCode).toBe(0);
    expect(ls.stdout).toContain("README.md");
  });

  it("clones to a custom directory", async () => {
    const { url, http } = await createMockGitServer({
      "README.md": "Custom dir clone"
    });
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

    const result = await bash.exec(`git clone ${url} /mnt/custom`);
    expect(result.exitCode).toBe(0);
    expect(result.stderr).toContain("Cloning into '/mnt/custom'");

    const ls = await bash.exec("ls /mnt/custom");
    expect(ls.exitCode).toBe(0);
    expect(ls.stdout).toContain("README.md");
  });

  it("clones with --depth 1", async () => {
    const { url, http } = await createMockGitServer([
      { files: { "README.md": "v1" }, message: "first commit" },
      { files: { "README.md": "v2" }, message: "second commit" }
    ]);
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

    const result = await bash.exec(`git clone --depth 1 ${url} /mnt/shallow`);
    expect(result.exitCode).toBe(0);
    expect(result.stderr).toContain("Cloning into");

    const ls = await bash.exec("ls /mnt/shallow");
    expect(ls.exitCode).toBe(0);
    expect(ls.stdout).toContain("README.md");
  });

  it("errors with no arguments", async () => {
    const inMemoryFs = new InMemoryFs();
    const mountableFs = new MountableFs({ base: inMemoryFs });
    const gitCmds = createGitCommands(mountableFs);
    const bash = new Bash({
      fs: mountableFs,
      customCommands: gitCmds,
      cwd: "/home/user"
    });

    const result = await bash.exec("git clone");
    expect(result.exitCode).toBe(128);
    expect(result.stderr).toContain("must specify a repository");
  });

  it("can run git log after clone", async () => {
    const { url, http } = await createMockGitServer([
      { files: { "README.md": "v1" }, message: "first commit" },
      { files: { "README.md": "v2" }, message: "second commit" }
    ]);
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

    await bash.exec(`git clone ${url} /mnt/repo`);
    const log = await bash.exec("git -C /mnt/repo log --oneline");
    expect(log.exitCode).toBe(0);
    expect(log.stdout).toContain("second commit");
  });
});
