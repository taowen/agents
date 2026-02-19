import { describe, it, expect } from "vitest";
import { createMockGitServer, MockR2Bucket, GitFs } from "./git-test-helpers";

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
