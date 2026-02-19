import { describe, it, expect } from "vitest";
import { createGitTestEnv } from "./git-test-helpers";

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
