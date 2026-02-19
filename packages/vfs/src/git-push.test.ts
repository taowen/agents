import { describe, it, expect, beforeEach } from "vitest";
import type { Bash } from "just-bash";
import { createGitTestEnv } from "./git-test-helpers";

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
