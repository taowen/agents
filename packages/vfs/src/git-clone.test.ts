import { describe, it, expect } from "vitest";
import { Bash, InMemoryFs, MountableFs } from "just-bash";
import { createGitCommands } from "./git-commands/index";
import { createMountCommands } from "./commands";
import { createMockGitServer, MockR2Bucket } from "./git-test-helpers";
import type { MountOptions } from "./mount";

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
