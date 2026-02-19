import { describe, it, expect } from "vitest";
import { Bash, InMemoryFs, MountableFs } from "just-bash";
import { createMockGitServer, MockR2Bucket, GitFs } from "./git-test-helpers";
import { createMountCommands } from "./commands";
import { createGitCommands } from "./git-commands/index";
import type { MountOptions } from "./mount";

describe("git pull", () => {
  it("shows 'Already up to date' when no remote changes", async () => {
    const { url, http } = await createMockGitServer({
      "README.md": "Hello"
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
      cwd: "/mnt/repo"
    });

    await bash.exec(`mount -t git ${url} /mnt/repo`);

    const result = await bash.exec("git pull");
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toContain("Already up to date");
  });

  it("pulls remote changes and sees new files", async () => {
    const { url, http, addCommit } = await createMockGitServer({
      "README.md": "Hello"
    });
    const r2Bucket = new MockR2Bucket() as unknown as R2Bucket;

    // Clone
    const gitFs = new GitFs({
      url,
      http,
      r2Bucket,
      userId: "test-user",
      mountPoint: "/mnt/repo"
    });
    await gitFs.init();

    // Add a commit on the server side (simulates another user pushing)
    await addCommit({ "newfile.txt": "from remote" }, "add newfile");

    // Pull
    const result = await gitFs.pull();
    expect(result.updated).toBe(true);
    expect(result.fromOid).not.toBe(result.toOid);

    // Verify the new file is visible
    const content = await gitFs.readFile("/newfile.txt", { encoding: "utf8" });
    expect(content).toBe("from remote");
  });

  it("errors when there are unpushed local commits", async () => {
    const { url, http } = await createMockGitServer({
      "README.md": "Hello"
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
      cwd: "/mnt/repo"
    });

    await bash.exec(`mount -t git ${url} /mnt/repo`);

    // Make a local commit (not pushed)
    await bash.exec("echo new > /mnt/repo/file.txt");
    await bash.exec('git commit -m "local change"');

    const result = await bash.exec("git pull");
    expect(result.exitCode).toBe(1);
    expect(result.stderr).toContain("push first");
  });
});
